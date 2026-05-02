import os
import sys
import torch

# FracAssist_Inference/ is this file's directory; repo root is one level above.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gel.gel_config import GEL_CONFIG  # noqa: E402

CONFIG = {
    # Weight paths — place files in repo_root/weights/
    "yolo_weights":     os.path.join(_ROOT, "weights", "Y1B_detect_best.pt"),
    "resnet_weights":   os.path.join(_ROOT, "weights", "E6_best.pth"),
    "densenet_weights":     os.path.join(_ROOT, "weights", "D1_best.pth"),
    "efficientnet_weights": os.path.join(_ROOT, "weights", "F1_best.pth"),

    # YOLO inference — fixed from Y1B training
    "yolo_conf_threshold": 0.25,
    "yolo_iou_threshold":  0.5,
    "yolo_imgsz":          600,

    # ResNet-18 inference — E6 CAALMIX champion (val threshold 0.525, CLAHE preprocessing required)
    "resnet_threshold":  0.525,
    "resnet_input_size": 224,
    "resnet_resize":     256,

    # DenseNet-169 inference — D1 val-sweep optimal threshold (0.175)
    "densenet_threshold":  0.175,
    "densenet_input_size": 224,

    # EfficientNet-B3 inference — F1 training complete (2026-04-28)
    # Val F1=0.6707, threshold=0.525, AUC=0.8825 (early stop ep36, best ep21)
    "efficientnet_threshold":  0.525,
    "efficientnet_input_size": 224,

    # GradCAM — DenseNet-169 D1 (final dense block before global avg pool)
    "gradcam_layer": "features.denseblock4",

    # Device: auto-upgrade to CUDA if available
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ImageNet normalisation (val_test_transforms)
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std":  [0.229, 0.224, 0.225],

    # Served static file — lives inside FracAssist_Inference/ alongside scripts.js and style.css
    "index_html": os.path.join(_HERE, "index.html"),
}

# GEL_CONFIG is imported from gel/gel_config.py above.
CONFIG.update(GEL_CONFIG)
