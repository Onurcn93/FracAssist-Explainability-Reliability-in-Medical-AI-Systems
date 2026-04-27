import os
import torch

# Repo root is one level above this file (inference/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Weight paths — place files in repo_root/weights/
    "yolo_weights":     os.path.join(_ROOT, "weights", "Y1B_detect_best.pt"),
    "resnet_weights":   os.path.join(_ROOT, "weights", "E4a_m050_best.pth"),
    "densenet_weights":     os.path.join(_ROOT, "weights", "D1_best.pth"),
    "efficientnet_weights": os.path.join(_ROOT, "weights", "F1_best.pth"),

    # YOLO inference — fixed from Y1B training
    "yolo_conf_threshold": 0.25,
    "yolo_iou_threshold":  0.5,
    "yolo_imgsz":          600,

    # ResNet-18 inference — fixed from E4a_m050 (val threshold 0.375, optimal sweep)
    "resnet_threshold":  0.375,
    "resnet_input_size": 224,
    "resnet_resize":     256,

    # DenseNet-169 inference — D1 val-sweep optimal threshold (0.175)
    "densenet_threshold":  0.175,
    "densenet_input_size": 224,

    # EfficientNet-B3 inference — placeholder until F1 training completes
    # Update "efficientnet_threshold" to val_threshold from F1_best.pth after training.
    "efficientnet_threshold":  0.5,
    "efficientnet_input_size": 224,

    # GradCAM — DenseNet-169 D1 (final dense block before global avg pool)
    "gradcam_layer": "features.denseblock4",

    # Device: auto-upgrade to CUDA if available
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ImageNet normalisation (val_test_transforms)
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std":  [0.229, 0.224, 0.225],

    # Served static file
    "index_html": os.path.join(_ROOT, "index.html"),
}

# ---------------------------------------------------------------------------
# GEL — Gated Ensemble Logic hyperparameters
#
# Performance anchors (F1): empirical val-sweep results, 2026-04-25.
#   YOLO is intentionally excluded — detection mAP is not comparable to
#   classification F1 and must not enter the PDWF weighted sum.
#
# BVG (Binary Verification Gate):
#   G = sum(P_i * F1_i) / sum(F1_i)  [all loaded classifiers]
#   If G < gel_tau -> gate fails -> bbox suppressed (P_final still shown).
#
# OAM (Outlier-Aware Modification):
#   mu = mean of loaded classifier probabilities
#   If |P_i - mu| > gel_disagree_lim -> RC_i_adj = RC_i * gel_penalty_k
#
# PDWF (Performance-Driven Weighted Fusion):
#   P_final = sum(P_i * RC_i_adj) / sum(RC_i_adj)   [loaded classifiers only]
#
# GEL adapts to 2 or 3 loaded classifiers automatically — see predict.py _run_gel().
# ---------------------------------------------------------------------------

GEL_CONFIG = {
    # Performance anchors — update when champion weights change
    "gel_f1_resnet":       0.658,   # E4a_m050 val F1
    "gel_f1_densenet":     0.724,   # D1 val F1
    # Placeholder — update to actual val F1 after F1 training completes
    "gel_f1_efficientnet": 0.700,   # F1 val F1 (placeholder)

    # BVG gate threshold — below this, YOLO bbox is suppressed
    "gel_tau":          0.35,

    # OAM — disagreement limit and penalty factor
    "gel_disagree_lim": 0.40,
    "gel_penalty_k":    0.20,
}

CONFIG.update(GEL_CONFIG)
