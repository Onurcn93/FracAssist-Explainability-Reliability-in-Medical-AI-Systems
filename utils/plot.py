"""
utils/plot.py

Generic plotting utilities. Saves figures to results/plots/.

Designed to work across all model phases without modification.
The caller defines what series exist and how they are grouped into subplots.

Typical usage — YOLO localization:

    plot_training_curves(
        series={
            "Box Loss": box_losses,
            "mAP@0.5":  map_values,
            "Precision": precisions,
            "Recall":    recalls,
        },
        groups=[["Box Loss"], ["mAP@0.5", "Precision", "Recall"]],
        titles=["Training Loss", "Validation Metrics"],
        ylabels=["Loss", "Value"],
        experiment="Y0 | YOLOv8s | detect | epochs=30 | imgsz=600",
    )

Typical usage — classification (Phase 1):

    plot_training_curves(
        series={
            "Train Loss": tr_losses, "Val Loss": va_losses,
            "Train Acc":  tr_accs,   "Val Acc":  va_accs,
        },
        groups=[["Train Loss", "Val Loss"], ["Train Acc", "Val Acc"]],
        titles=["Loss per Epoch", "Accuracy per Epoch"],
        ylabels=["Loss", "Accuracy"],
        experiment="E3 | ResNet-18 | ...",
    )
"""

import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt

PLOTS_DIR = "results/plots"


def _safe_filename(title: str) -> str:
    s = title.replace(" | ", "_").replace("=", "").replace(" ", "_")
    return re.sub(r"[^\w\-]", "", s)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_training_curves(
    series:     Dict[str, List[float]],
    groups:     List[List[str]],
    titles:     List[str],
    ylabels:    List[str],
    experiment: str,
    out_dir:    str = PLOTS_DIR,
) -> None:
    """Save a multi-subplot training curve figure.

    Args:
        series:     Dict mapping series name to list of per-epoch values.
        groups:     List of subplot groups — each group is a list of series
                    names to plot together in one subplot.
        titles:     Subplot title for each group (same length as groups).
        ylabels:    Y-axis label for each group (same length as groups).
        experiment: Experiment ID used as figure title and filename.
        out_dir:    Directory to save the figure.
    """
    _ensure_dir(out_dir)

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(experiment, fontsize=10, fontweight="bold")

    for ax, group, title, ylabel in zip(axes, groups, titles, ylabels):
        for name in group:
            values = series.get(name, [])
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=name, marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"curves_{_safe_filename(experiment)}.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved training curves → {path}")