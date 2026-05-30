"""
C5.0 — compute_gradcam_centroids.py

Loads the 3 GEL champion weights (ResNet-18/E6, DenseNet-169/D1,
EfficientNet-B3/F1) and runs GradCAM on every image in
data/dataset_cls/{train,val,test}.

For each (model, image):
  1. Forward pass  -> forward hook captures feature maps
  2. Backward on frac_idx logit -> backward hook captures gradients
  3. CAM = ReLU( sum_c( mean_spatial(grad_c) * feat_c ) )
  4. centroid_x/y = activation-weighted mean x/y, normalised to [0,1]

Then for each image the Euclidean distance from each model's centroid
to the 3-model mean centroid is computed as the T6 spatial-disagreement
feature.

No training — weights are loaded from saved checkpoints only.

Output: Ensemble_Ablations/data/gradcam_centroids.csv
Columns:
  image_path, split, true_label,
  resnet_cx, resnet_cy, densenet_cx, densenet_cy, eff_cx, eff_cy,
  dist_resnet_to_mean, dist_dense_to_mean, dist_eff_to_mean

GradCAM target layers:
  ResNet-18      : model.layer4           (512-ch, 7×7 @ 224px)
  DenseNet-169   : model.features.denseblock4  (before norm5 to avoid
                   inplace-relu hook corruption)
  EfficientNet-B3: model.features[-1]     (1536-ch, 7×7 @ 224px)
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

THIS_DIR     = Path(__file__).resolve().parent
EA_ROOT      = THIS_DIR.parent
PROJECT_ROOT = EA_ROOT.parent

WEIGHTS_DIR = PROJECT_ROOT / "weights"
IMAGE_ROOT  = PROJECT_ROOT / "data" / "dataset_cls"
OUT_CSV     = EA_ROOT / "data" / "gradcam_centroids.csv"

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
LOG_EVERY     = 200


# ── Transforms ──────────────────────────────────────────────────────────────

class _CLAHETransform:
    def __call__(self, img: Image.Image) -> Image.Image:
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray     = np.array(img.convert("L"), dtype=np.uint8)
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced).convert("RGB")


_TRANSFORM_CLAHE = transforms.Compose([
    _CLAHETransform(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_TRANSFORM_STD = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Model loading ────────────────────────────────────────────────────────────

def _safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def _load_resnet(device):
    ckpt  = _safe_load(WEIGHTS_DIR / "E6_best.pth", device)
    state = ckpt.get("model_state_dict", ckpt)
    fi    = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    m     = tv_models.resnet18(weights=None)
    # Shape-aware head detection: original fc outputs 1000 classes
    if "fc.1.weight" in state:
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
    elif state.get("fc.weight", torch.empty(1000, 1)).shape[0] == 2:
        m.fc = nn.Linear(512, 2)
    m.load_state_dict(state)
    m.eval()
    return m.to(device), fi


def _load_densenet(device):
    ckpt  = _safe_load(WEIGHTS_DIR / "D1_best.pth", device)
    state = ckpt.get("model_state_dict", ckpt)
    fi    = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    m     = tv_models.densenet169(weights=None)
    in_f  = m.classifier.in_features
    # Shape-aware: classifier.1.weight exists in modified Sequential head
    if "classifier.1.weight" in state and state["classifier.1.weight"].shape[0] == 2:
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 2))
    elif state.get("classifier.weight", torch.empty(1000, 1)).shape[0] == 2:
        m.classifier = nn.Linear(in_f, 2)
    m.load_state_dict(state)
    m.eval()
    return m.to(device), fi


def _load_efficientnet(device):
    ckpt  = _safe_load(WEIGHTS_DIR / "F1_best.pth", device)
    state = ckpt.get("model_state_dict", ckpt)
    fi    = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    m     = tv_models.efficientnet_b3(weights=None)
    in_f  = m.classifier[1].in_features
    # Detect head type by which keys are present and their output dimension:
    #   Sequential(Dropout, Linear): classifier.1.weight shape [2, in_f]
    #   Plain Linear:                classifier.weight   shape [2, in_f]
    cls1_w = state.get("classifier.1.weight")
    cls_w  = state.get("classifier.weight")
    if cls1_w is not None and cls1_w.shape[0] == 2:
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 2))
    elif cls_w is not None and cls_w.shape[0] == 2:
        m.classifier = nn.Linear(in_f, 2)
    m.load_state_dict(state)
    m.eval()
    return m.to(device), fi


# ── GradCAM ──────────────────────────────────────────────────────────────────

class _GradCAM:
    """Hook-based GradCAM for a single target layer.

    forward hook: saves feature maps (cloned to guard against inplace ops).
    backward hook: saves gradients of the class score w.r.t. feature maps.
    CAM = ReLU( sum_c( mean_spatial(grad_c) * feat_c ) )
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self._acts  = None
        self._grads = None
        self._h_fwd = target_layer.register_forward_hook(self._fwd_hook)
        self._h_bwd = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        if isinstance(out, torch.Tensor):
            self._acts = out.detach().clone()   # clone: guard inplace relu

    def _bwd_hook(self, module, grad_inp, grad_out):
        if isinstance(grad_out[0], torch.Tensor):
            self._grads = grad_out[0].detach().clone()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return (H, W) ReLU'd CAM as a numpy array."""
        self.model.zero_grad()
        logits = self.model(x)                              # forward
        logits[0, class_idx].backward()                    # backward

        weights = self._grads.mean(dim=[2, 3], keepdim=True)   # (1,C,1,1)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        return cam.squeeze().cpu().numpy()                  # (H, W)

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


# ── Centroid ─────────────────────────────────────────────────────────────────

def _centroid(cam: np.ndarray) -> tuple[float, float]:
    """Activation-weighted centroid of a 2-D heatmap, normalised to [0, 1].

    Returns (cx, cy) where cx is the column axis (x) and cy is the row axis (y).
    Defaults to (0.5, 0.5) for degenerate all-zero CAMs.
    """
    H, W  = cam.shape
    total = float(cam.sum())
    if total < 1e-8:
        return 0.5, 0.5
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    cx = float((cam * xs).sum() / total) / max(W - 1, 1)
    cy = float((cam * ys).sum() / total) / max(H - 1, 1)
    return cx, cy


# ── Image collection ─────────────────────────────────────────────────────────

def _collect_images() -> list[dict]:
    """Walk IMAGE_ROOT/{train,val,test}/<class>/ and collect image records."""
    records = []
    for split in ("train", "val", "test"):
        split_dir = IMAGE_ROOT / split
        if not split_dir.exists():
            print(f"  [WARN] split directory not found: {split_dir}")
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    records.append({
                        "path":  img_path,
                        "split": split,
                        "label": cls_dir.name,
                    })
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model weights (no training)...")
    resnet,       r_fi = _load_resnet(device)
    densenet,     d_fi = _load_densenet(device)
    efficientnet, e_fi = _load_efficientnet(device)
    print(f"  ResNet frac_idx={r_fi} | DenseNet frac_idx={d_fi} | EfficientNet frac_idx={e_fi}")

    # GradCAM targets
    cam_r = _GradCAM(resnet,       resnet.layer4)
    cam_d = _GradCAM(densenet,     densenet.features.denseblock4)
    cam_e = _GradCAM(efficientnet, efficientnet.features[-1])

    images = _collect_images()
    total  = len(images)
    print(f"Found {total} images across train / val / test.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path", "split", "true_label",
        "resnet_cx",   "resnet_cy",
        "densenet_cx", "densenet_cy",
        "eff_cx",      "eff_cy",
        "dist_resnet_to_mean", "dist_dense_to_mean", "dist_eff_to_mean",
    ]

    t0          = time.time()
    n_degen     = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, rec in enumerate(images):
            img_pil = Image.open(rec["path"]).convert("RGB")
            x_clahe = _TRANSFORM_CLAHE(img_pil).unsqueeze(0).to(device)
            x_std   = _TRANSFORM_STD(img_pil).unsqueeze(0).to(device)

            hm_r = cam_r(x_clahe, r_fi)
            hm_d = cam_d(x_std,   d_fi)
            hm_e = cam_e(x_std,   e_fi)

            if hm_r.sum() < 1e-8 or hm_d.sum() < 1e-8 or hm_e.sum() < 1e-8:
                n_degen += 1

            r_cx, r_cy = _centroid(hm_r)
            d_cx, d_cy = _centroid(hm_d)
            e_cx, e_cy = _centroid(hm_e)

            mx = (r_cx + d_cx + e_cx) / 3.0
            my = (r_cy + d_cy + e_cy) / 3.0

            dist_r = float(np.hypot(r_cx - mx, r_cy - my))
            dist_d = float(np.hypot(d_cx - mx, d_cy - my))
            dist_e = float(np.hypot(e_cx - mx, e_cy - my))

            writer.writerow({
                "image_path":          str(rec["path"].relative_to(PROJECT_ROOT)),
                "split":               rec["split"],
                "true_label":          rec["label"],
                "resnet_cx":           round(r_cx, 6),
                "resnet_cy":           round(r_cy, 6),
                "densenet_cx":         round(d_cx, 6),
                "densenet_cy":         round(d_cy, 6),
                "eff_cx":              round(e_cx, 6),
                "eff_cy":              round(e_cy, 6),
                "dist_resnet_to_mean": round(dist_r, 6),
                "dist_dense_to_mean":  round(dist_d, 6),
                "dist_eff_to_mean":    round(dist_e, 6),
            })

            if (i + 1) % LOG_EVERY == 0 or (i + 1) == total:
                elapsed   = time.time() - t0
                rate      = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i+1:4d}/{total}]  "
                    f"elapsed={elapsed:.0f}s  "
                    f"rate={rate:.1f} img/s  "
                    f"ETA={remaining:.0f}s"
                )
                f.flush()

    cam_r.remove()
    cam_d.remove()
    cam_e.remove()

    elapsed = time.time() - t0
    print(f"\nDone. {total} images in {elapsed:.1f}s  ({total/elapsed:.1f} img/s)")
    print(f"Degenerate CAMs defaulted to center: {n_degen}")
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
