import os
import torch

# Repo root is one level above this file (inference/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Weight paths — place files in repo_root/weights/
    "yolo_weights":     os.path.join(_ROOT, "weights", "Y1B_detect_best.pt"),
    "resnet_weights":   os.path.join(_ROOT, "weights", "E4a_m050_best.pth"),
    "densenet_weights": os.path.join(_ROOT, "weights", "D1_best.pth"),

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
#   G = (P_r·F1_r + P_d·F1_d) / (F1_r + F1_d)
#   If G < gel_tau → gate fails → bbox suppressed (P_final still shown).
#
# OAM (Outlier-Aware Modification):
#   If |P_i − μ| > gel_disagree_lim → RC_i_adj = RC_i × gel_penalty_k
#
# PDWF (Performance-Driven Weighted Fusion):
#   P_final = Σ(P_i · RC_i_adj) / Σ RC_i_adj   [classifiers only]
# ---------------------------------------------------------------------------

GEL_CONFIG = {
    # Performance anchors — update when champion weights change
    "gel_f1_resnet":    0.658,   # E4a_m050 val F1
    "gel_f1_densenet":  0.724,   # D1 val F1

    # BVG gate threshold — below this, YOLO bbox is suppressed
    "gel_tau":          0.35,

    # OAM — disagreement limit and penalty factor
    "gel_disagree_lim": 0.40,
    "gel_penalty_k":    0.20,
}

CONFIG.update(GEL_CONFIG)
