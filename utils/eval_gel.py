"""
eval_gel.py

Evaluate the Gated Ensemble Logic (GEL) pipeline on val and/or test splits.

GEL decision path: BVG → RC → OAM → PDWF (classifier-only).
YOLO bbox authentication is UI-only; it does not affect p_final and is
therefore not included in classification metrics here.

Usage:
    python utils/eval_gel.py                  # val sweep → apply to test (default)
    python utils/eval_gel.py --split val
    python utils/eval_gel.py --split test
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import ImageFile
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Config ────────────────────────────────────────────────────────────── #

BATCH_SIZE = 32
IMG_SIZE   = 224
FRAC_CLASS = "Fractured"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RESNET_WEIGHTS   = Path("weights/E4a_m050_best.pth")
DENSENET_WEIGHTS = Path("weights/D1_best.pth")

# GEL hyperparameters — must match inference/config.py GEL_CONFIG exactly
GEL_F1_RESNET    = 0.658   # E4a_m050 val F1 anchor
GEL_F1_DENSENET  = 0.724   # D1 val F1 anchor
GEL_TAU          = 0.35    # BVG gate threshold
GEL_DISAGREE_LIM = 0.40    # OAM outlier disagreement limit
GEL_PENALTY_K    = 0.20    # OAM penalty factor

RESNET_THRESHOLD   = 0.375  # E4a_m050 val-optimal (for individual comparison)
DENSENET_THRESHOLD = 0.175  # D1 val-optimal (for individual comparison)

# ── Model loading ─────────────────────────────────────────────────────── #

def _load_resnet(device):
    if not RESNET_WEIGHTS.exists():
        raise FileNotFoundError(f"ResNet-18 weights not found: {RESNET_WEIGHTS}")
    ckpt     = _safe_load(RESNET_WEIGHTS, device)
    state    = ckpt.get("model_state_dict", ckpt)
    frac_idx = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    has_dropout = "fc.1.weight" in state
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2)) if has_dropout else nn.Linear(512, 2)
    m.load_state_dict(state)
    m.eval()
    return m.to(device), frac_idx


def _load_densenet(device):
    if not DENSENET_WEIGHTS.exists():
        raise FileNotFoundError(f"DenseNet-169 weights not found: {DENSENET_WEIGHTS}")
    ckpt     = _safe_load(DENSENET_WEIGHTS, device)
    state    = ckpt.get("model_state_dict", ckpt)
    frac_idx = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    has_dropout = "classifier.1.weight" in state
    m = tv_models.densenet169(weights=None)
    in_feat = m.classifier.in_features  # 1664
    m.classifier = (
        nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 2))
        if has_dropout else nn.Linear(in_feat, 2)
    )
    m.load_state_dict(state)
    m.eval()
    return m.to(device), frac_idx


def _safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


# ── Preprocessing ─────────────────────────────────────────────────────── #

def _get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Inference ─────────────────────────────────────────────────────────── #

def _collect_probs(resnet, densenet, r_frac_idx, d_frac_idx, loader, device):
    """Return (labels, p_r, p_d) arrays over the full split."""
    all_labels, all_p_r, all_p_d = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs     = imgs.to(device)
            p_r = torch.softmax(resnet(imgs),   dim=1)[:, r_frac_idx].cpu().numpy()
            p_d = torch.softmax(densenet(imgs), dim=1)[:, d_frac_idx].cpu().numpy()
            all_labels.extend(labels.numpy())
            all_p_r.extend(p_r)
            all_p_d.extend(p_d)
    return np.array(all_labels), np.array(all_p_r), np.array(all_p_d)


# ── GEL — vectorized ─────────────────────────────────────────────────── #

def _apply_gel(p_r, p_d):
    """Vectorized GEL: BVG → RC → OAM → PDWF.

    Returns (p_final, gate_passed, g_consensus) — all shape (N,).
    """
    total = GEL_F1_RESNET + GEL_F1_DENSENET

    # Step 2 — BVG (gate controls bbox auth; does not block PDWF)
    g           = (p_r * GEL_F1_RESNET + p_d * GEL_F1_DENSENET) / total
    gate_passed = g >= GEL_TAU

    # Step 3 — RC initialisation
    rc_r = np.full_like(p_r, GEL_F1_RESNET  / total)
    rc_d = np.full_like(p_d, GEL_F1_DENSENET / total)

    # Step 4 — OAM: penalise outliers
    mu   = (p_r + p_d) / 2.0
    rc_r = np.where(np.abs(p_r - mu) > GEL_DISAGREE_LIM, rc_r * GEL_PENALTY_K, rc_r)
    rc_d = np.where(np.abs(p_d - mu) > GEL_DISAGREE_LIM, rc_d * GEL_PENALTY_K, rc_d)

    # Step 5 — PDWF (always computed regardless of gate)
    p_final = (p_r * rc_r + p_d * rc_d) / (rc_r + rc_d)

    return p_final, gate_passed, g


# ── Threshold helpers ─────────────────────────────────────────────────── #

def _sweep_threshold(labels, scores, frac_idx):
    binary = (labels == frac_idx).astype(int)  # 1=Fractured, consistent with _evaluate
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.025):
        preds = (scores >= t).astype(int)
        f1    = f1_score(binary, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _evaluate(labels, scores, threshold, frac_idx):
    preds  = (scores >= threshold).astype(int)
    binary = (labels == frac_idx).astype(int)
    return {
        "f1":        f1_score(binary,        preds, pos_label=1, zero_division=0),
        "recall":    recall_score(binary,    preds, pos_label=1, zero_division=0),
        "precision": precision_score(binary, preds, pos_label=1, zero_division=0),
        "acc":       accuracy_score(binary,  preds),
        "auc":       roc_auc_score(binary,   scores),
    }


# ── Reporting helpers ─────────────────────────────────────────────────── #

def _print_metrics_row(label, m):
    print(
        f"  {label:<28}  "
        f"F1={m['f1']:.4f}  Recall={m['recall']:.4f}  "
        f"Prec={m['precision']:.4f}  Acc={m['acc']:.4f}  AUC={m['auc']:.4f}"
    )


def _gel_diagnostics(p_r, p_d, gate_passed):
    mu      = (p_r + p_d) / 2.0
    oam_r   = (np.abs(p_r - mu) > GEL_DISAGREE_LIM).mean() * 100
    oam_d   = (np.abs(p_d - mu) > GEL_DISAGREE_LIM).mean() * 100
    gate_pct = gate_passed.mean() * 100
    print(f"  BVG gate pass rate    : {gate_pct:.1f}%  (τ = {GEL_TAU})")
    print(f"  OAM ResNet-18 trigger : {oam_r:.1f}%  (δ = {GEL_DISAGREE_LIM})")
    print(f"  OAM DenseNet-169 trigger: {oam_d:.1f}%  (δ = {GEL_DISAGREE_LIM})")


# ── Per-split evaluation ───────────────────────────────────────────────── #

def eval_split(resnet, densenet, r_fi, d_fi, data_dir, device, label, val_thresh=None):
    """
    Run GEL eval on one split.
    val_thresh — if provided, apply as fixed threshold (test mode);
                 if None, sweep and report both 0.5 and optimal (val mode).
    Returns opt_thresh found on this split.
    """
    tf      = _get_transform()
    dataset = ImageFolder(root=str(data_dir), transform=tf)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    frac_idx = dataset.class_to_idx[FRAC_CLASS]

    print(f"\n{'=' * 80}")
    print(f"  Split: {label}  ({data_dir})  —  {len(dataset)} images  |  frac_idx={frac_idx}")
    print(f"{'=' * 80}")

    labels, p_r, p_d = _collect_probs(resnet, densenet, r_fi, d_fi, loader, device)
    p_final, gate_passed, g = _apply_gel(p_r, p_d)

    print("\nGEL Diagnostics:")
    _gel_diagnostics(p_r, p_d, gate_passed)
    print(
        f"  p_final  range        : [{p_final.min():.3f}, {p_final.max():.3f}]  "
        f"mean={p_final.mean():.3f}  std={p_final.std():.3f}"
    )

    # Individual model metrics — sweep to match eval_resnet / eval_densenet behaviour
    r_opt_t, _ = _sweep_threshold(labels, p_r, frac_idx)
    d_opt_t, _ = _sweep_threshold(labels, p_d, frac_idx)
    m_resnet   = _evaluate(labels, p_r, r_opt_t, frac_idx)
    m_densenet = _evaluate(labels, p_d, d_opt_t, frac_idx)

    # GEL metrics
    opt_thresh, _ = _sweep_threshold(labels, p_final, frac_idx)
    m_gel_05  = _evaluate(labels, p_final, 0.5,        frac_idx)
    m_gel_opt = _evaluate(labels, p_final, opt_thresh, frac_idx)

    print("\nMetrics:")
    _print_metrics_row(f"ResNet-18   (thr={r_opt_t:.3f} sweep)", m_resnet)
    _print_metrics_row(f"DenseNet-169 (thr={d_opt_t:.3f} sweep)", m_densenet)
    print(f"  {'-' * 74}")
    _print_metrics_row("GEL  (thr=0.500 fixed)",       m_gel_05)
    _print_metrics_row(f"GEL  (thr={opt_thresh:.3f} sweep-opt)", m_gel_opt)

    if val_thresh is not None:
        m_gel_val = _evaluate(labels, p_final, val_thresh, frac_idx)
        _print_metrics_row(f"GEL  (thr={val_thresh:.3f} val-optimal)", m_gel_val)

    print(f"\n  Val-sweep optimal threshold: {opt_thresh:.3f}")
    return opt_thresh


# ── Main ─────────────────────────────────────────────────────────────── #

def main(split):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ResNet-18   weights: {RESNET_WEIGHTS}")
    print(f"DenseNet-169 weights: {DENSENET_WEIGHTS}")

    resnet,  r_fi = _load_resnet(device)
    densenet, d_fi = _load_densenet(device)
    print(f"Models loaded  (resnet frac_idx={r_fi}, densenet frac_idx={d_fi})")

    if split == "both":
        val_thresh = eval_split(
            resnet, densenet, r_fi, d_fi,
            Path("data/dataset_cls/val"), device, label="VAL",
        )
        eval_split(
            resnet, densenet, r_fi, d_fi,
            Path("data/dataset_cls/test"), device, label="TEST",
            val_thresh=val_thresh,
        )
    else:
        eval_split(
            resnet, densenet, r_fi, d_fi,
            Path(f"data/dataset_cls/{split}"), device, label=split.upper(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="both",
        help="Split to evaluate (default: both — sweeps val then applies to test)",
    )
    args = parser.parse_args()
    main(args.split)
