"""
utils/gradcam.py

GradCAM (Gradient-weighted Class Activation Mapping) — model-agnostic.

Public API
----------
    overlay = compute_overlay(model, tensor, image_bgr, frac_idx, device,
                              layer_name="features.denseblock4", alpha=0.5)
        Returns a BGR numpy uint8 array — original image blended with heatmap.

    b64 = to_base64(model, tensor, image_path, frac_idx, device,
                    layer_name="features.denseblock4", alpha=0.5)
        Returns a "data:image/png;base64,..." string ready for the frontend.

    save(model, tensor, image_path, frac_idx, device, out_path,
         layer_name="features.denseblock4", alpha=0.5)
        Writes the overlay PNG to out_path (offline plotting / research).

layer_name is a dot-separated path (e.g. "features.denseblock4" for DenseNet-169,
"layer4" for ResNet-18). If the resolved module is a Sequential or ModuleList,
the last child is hooked automatically.

All three functions target the fracture class score directly so the heatmap
always shows *why the model considers the image a fracture*, regardless of
the final prediction label.

Reference
---------
    Selvaraju et al. 2017 — "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization"
    https://arxiv.org/abs/1610.02391
"""

import base64
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn


def _imread(path: Union[str, Path]) -> "np.ndarray | None":
    """cv2.imread that works on Windows paths containing spaces or Unicode."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ── Layer resolution ─────────────────────────────────────────────────────── #

def _resolve_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Return the target layer from a dot-separated name.

    For ResNet-18 the canonical target is "layer4" which resolves to
    model.layer4 (a Sequential) → we hook its last element (BasicBlock).
    Any deeper dotted path (e.g. "layer4.1") is also supported.
    """
    obj = model
    for part in layer_name.split("."):
        obj = getattr(obj, part)
    # If the resolved object is a container, hook its last child
    if isinstance(obj, (nn.Sequential, nn.ModuleList)):
        return obj[-1]
    return obj


# ── Core GradCAM computation ─────────────────────────────────────────────── #

def _compute_cam(
    model: nn.Module,
    tensor: torch.Tensor,
    frac_idx: int,
    device: torch.device,
    layer_name: str,
) -> np.ndarray:
    """Internal: run one GradCAM forward/backward pass.

    Returns a (H, W) float32 numpy array in [0, 1] — the raw CAM map at
    the spatial resolution of the target layer.
    Hooks are always removed in the finally block to prevent memory leaks.
    """
    activations: dict = {}
    gradients:   dict = {}

    target_layer = _resolve_layer(model, layer_name)

    def _fwd(module, inp, out):
        activations["feat"] = out.detach().clone()

    def _bwd(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach().clone()

    h_fwd = target_layer.register_forward_hook(_fwd)
    h_bwd = target_layer.register_full_backward_hook(_bwd)

    was_training = model.training
    try:
        model.eval()
        model.zero_grad()

        t      = tensor.to(device)
        output = model(t)                   # (1, 2) — two class logits
        score  = output[0, frac_idx]        # scalar: fracture class score
        score.backward()                    # gradients flow back to target_layer

        act     = activations["feat"].squeeze(0)   # (C, H, W)
        grad    = gradients["feat"].squeeze(0)     # (C, H, W)
        weights = grad.mean(dim=[1, 2])             # (C,)  global-avg-pool

        cam = torch.relu((weights[:, None, None] * act).sum(0))  # (H, W)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy().astype(np.float32)

    finally:
        h_fwd.remove()
        h_bwd.remove()
        model.zero_grad()
        if was_training:
            model.train()


# ── Overlay builder ───────────────────────────────────────────────────────── #

def compute_overlay(
    model:      nn.Module,
    tensor:     torch.Tensor,
    image_bgr:  np.ndarray,
    frac_idx:   int,
    device:     torch.device,
    layer_name: str   = "features.denseblock4",
    alpha:      float = 0.5,
) -> np.ndarray:
    """Blend a GradCAM heatmap onto an existing BGR image array.

    Args:
        model:      2-class classifier (eval or train mode — handled internally).
        tensor:     Preprocessed (1, 3, H, W) float tensor (val/test transforms).
        image_bgr:  Original image as BGR numpy uint8 array.
        frac_idx:   Integer index of the Fractured class (0 for FracAtlas ImageFolder).
        device:     torch.device for model inference.
        layer_name: Target layer name (default "layer4" → layer4[-1]).
        alpha:      Blend weight for the original image (0 = heatmap only, 1 = original only).

    Returns:
        BGR numpy uint8 array, same spatial size as image_bgr.
    """
    cam_np = _compute_cam(model, tensor, frac_idx, device, layer_name)

    h, w = image_bgr.shape[:2]
    cam_u8  = np.uint8(255 * cv2.resize(cam_np, (w, h)))
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)  # BGR

    return cv2.addWeighted(image_bgr, alpha, heatmap, 1.0 - alpha, 0)


# ── Frontend helper ───────────────────────────────────────────────────────── #

def to_base64(
    model:      nn.Module,
    tensor:     torch.Tensor,
    image_path: Union[str, Path],
    frac_idx:   int,
    device:     torch.device,
    layer_name: str   = "features.denseblock4",
    alpha:      float = 0.5,
    overlay_size: int = 224,
) -> str:
    """Compute GradCAM and return a data:image/png;base64,... string.

    Reads image_path from disk, resizes to overlay_size × overlay_size,
    blends heatmap, then base64-encodes the result.

    Returns the data URI string directly consumable by <img src="...">.
    If image_path cannot be read, falls back to a pure heatmap at overlay_size.
    """
    orig = _imread(image_path)
    if orig is None:
        orig = np.zeros((overlay_size, overlay_size, 3), dtype=np.uint8)
    orig_resized = cv2.resize(orig, (overlay_size, overlay_size))

    blended = compute_overlay(model, tensor, orig_resized, frac_idx, device, layer_name, alpha)

    _, buf = cv2.imencode(".png", blended)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


# ── Offline save helper ───────────────────────────────────────────────────── #

def save(
    model:      nn.Module,
    tensor:     torch.Tensor,
    image_path: Union[str, Path],
    frac_idx:   int,
    device:     torch.device,
    out_path:   Union[str, Path],
    layer_name: str   = "features.denseblock4",
    alpha:      float = 0.5,
) -> None:
    """Compute GradCAM and save the overlay as a PNG file.

    Reads image_path at its original resolution, overlays the heatmap,
    and writes to out_path. Creates parent directories as needed.

    Intended for offline evaluation and research plots — call after
    training or during test-set evaluation to produce explainability figures.
    """
    orig = _imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"[gradcam] Cannot read image: {image_path}")

    blended = compute_overlay(model, tensor, orig, frac_idx, device, layer_name, alpha)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)
    print(f"[gradcam] Saved → {out_path}")
