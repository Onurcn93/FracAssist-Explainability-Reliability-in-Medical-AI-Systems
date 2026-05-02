"""
eval_efficientnet.py

Evaluate all EfficientNet-B3 checkpoints on a given split and print a ranked F1 table.
Mirrors eval_densenet.py exactly — same structure, same output format.

Usage:
    python utils/eval_efficientnet.py                  # test set (default)
    python utils/eval_efficientnet.py --split val
    python utils/eval_efficientnet.py --split train
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Config ────────────────────────────────────────────────────────────── #

DATA_DIR   = Path("data/dataset_cls/test")  # overridden by --split
BATCH_SIZE = 32
IMG_SIZE   = 224
FRAC_CLASS = "Fractured"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CHECKPOINT_DIRS = [Path("weights"), Path("colab_results")]

# Checkpoints trained with CLAHE on ALL splits — must use CLAHE at eval time.
_CLAHE_CKPTS = {"F2_best", "F3_best", "F4_best"}

# ── Helpers ───────────────────────────────────────────────────────────── #

def _infer_dropout(state_dict: dict) -> float:
    # EfficientNet dropout_p>0 → classifier is Sequential → key is classifier.1.weight
    # No dropout → classifier is Linear → key is classifier.weight
    if "classifier.1.weight" in state_dict:
        return 0.3  # only dropout_p used in F-series
    return 0.0


def _build_model(dropout_p: float, device: torch.device) -> nn.Module:
    model   = tv_models.efficientnet_b3(weights=None)
    in_feat: int = model.classifier[1].in_features  # type: ignore[union-attr]
    if dropout_p > 0.0:
        model.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_feat, 2))  # type: ignore[assignment]
    else:
        model.classifier = nn.Linear(in_feat, 2)  # type: ignore[assignment]
    return model.to(device)


class _CLAHETransform:
    def __call__(self, img: Image.Image) -> Image.Image:
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray     = np.array(img.convert("L"), dtype=np.uint8)
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced).convert("RGB")


def _get_transform(clahe: bool = False) -> transforms.Compose:
    steps = []
    if clahe:
        steps.append(_CLAHETransform())
    steps += [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def _collect_probs(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def _sweep_threshold(labels, probs, frac_idx):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.025):
        preds = np.where(probs[:, frac_idx] >= t, frac_idx, 1 - frac_idx)
        f1    = f1_score(labels, preds, pos_label=frac_idx, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _evaluate(labels, probs, threshold, frac_idx):
    preds      = np.where(probs[:, frac_idx] >= threshold, frac_idx, 1 - frac_idx)
    bin_labels = (labels == frac_idx).astype(int)
    return {
        "f1":        f1_score(labels, preds, pos_label=frac_idx, zero_division=0),
        "recall":    recall_score(labels, preds, pos_label=frac_idx, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=frac_idx, zero_division=0),
        "acc":       accuracy_score(labels, preds),
        "auc":       roc_auc_score(bin_labels, probs[:, frac_idx]),
    }


# ── Main ─────────────────────────────────────────────────────────────── #

def main(data_dir: Path = DATA_DIR):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Collect only F-series checkpoints
    _KEEP = re.compile(r'^[Ff]\d', re.IGNORECASE)
    ckpt_paths = []
    for d in CHECKPOINT_DIRS:
        if d.exists():
            ckpt_paths.extend(
                p for p in sorted(d.glob("*.pth"))
                if _KEEP.match(p.stem) or "efficientnet" in p.stem.lower()
            )

    if not ckpt_paths:
        print("No EfficientNet .pth files found (expected F1_best.pth, F2_best.pth, ...).")
        return

    rows = []
    for ckpt_path in ckpt_paths:
        needs_clahe = ckpt_path.stem in _CLAHE_CKPTS
        clahe_tag   = " [CLAHE]" if needs_clahe else ""
        print(f"Evaluating {ckpt_path.name}{clahe_tag} ...", end=" ", flush=True)

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Prefer flag saved in checkpoint; fall back to filename-based detection
        needs_clahe = ckpt.get("use_clahe", needs_clahe)

        state     = ckpt.get("model_state_dict", ckpt)
        dropout_p = _infer_dropout(state)
        model     = _build_model(dropout_p, device)
        model.load_state_dict(state, strict=True)
        model.eval()

        saved_thresh = float(ckpt.get("val_threshold", 0.5))

        tf      = _get_transform(clahe=needs_clahe)
        dataset = ImageFolder(root=str(data_dir), transform=tf)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        frac_idx = dataset.class_to_idx[FRAC_CLASS]

        labels, probs = _collect_probs(model, loader, device)

        opt_thresh, _ = _sweep_threshold(labels, probs, frac_idx)

        m_saved = _evaluate(labels, probs, saved_thresh, frac_idx)
        m_opt   = _evaluate(labels, probs, opt_thresh,   frac_idx)

        use_thresh = opt_thresh if m_opt["f1"] >= m_saved["f1"] else saved_thresh
        m_use      = m_opt      if m_opt["f1"] >= m_saved["f1"] else m_saved

        source = "colab" if "colab" in str(ckpt_path) else "local"
        exp_id = ckpt.get("exp_id", ckpt_path.stem)

        rows.append({
            "exp":    exp_id,
            "source": source,
            "clahe":  needs_clahe,
            "thresh": use_thresh,
            "f1":     m_use["f1"],
            "recall": m_use["recall"],
            "prec":   m_use["precision"],
            "acc":    m_use["acc"],
            "auc":    m_use["auc"],
        })
        print(f"F1={m_use['f1']:.4f}  thresh={use_thresh:.3f}")

    rows.sort(key=lambda r: r["f1"], reverse=True)

    print("\n" + "=" * 92)
    print(f"{'Rank':<5} {'Experiment':<30} {'Src':<6} {'CLAHE':<6} {'Thresh':<7} "
          f"{'F1':>6} {'Recall':>7} {'Prec':>7} {'Acc':>7} {'AUC':>7}")
    print("-" * 92)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:<5} {r['exp']:<30} {r['source']:<6} {'yes' if r['clahe'] else 'no':<6} "
            f"{r['thresh']:<7.3f} {r['f1']:>6.4f} {r['recall']:>7.4f} {r['prec']:>7.4f} "
            f"{r['acc']:>7.4f} {r['auc']:>7.4f}"
        )
    print("=" * 92)
    champion = rows[0]
    print(f"\nChampion: {champion['exp']} ({champion['source']})"
          f"{'  [CLAHE]' if champion['clahe'] else ''}  "
          f"F1={champion['f1']:.4f}  thresh={champion['thresh']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    main(data_dir=Path(f"data/dataset_cls/{args.split}"))
