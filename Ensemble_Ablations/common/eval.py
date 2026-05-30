"""
common/eval.py

Shared evaluation harness for all EA-series methods.
Every method script calls evaluate() and append_to_table() / append_to_results().

Protocol: fit on train → score on val → check test.
Primary ranking axis: val F1 (threshold-swept).

Two output files:
  EA_comparative_table.csv  — concise view: val/test F1+AUC, vs-GEL deltas, references
  EA_results.csv            — full view: all 3 splits × F1/AUC/acc/recall/spec/threshold
"""

import csv
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

THRESHOLD_RANGE = np.arange(0.01, 1.00, 0.005)

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

TABLE_PATH   = _RESULTS_DIR / "EA_comparative_table.csv"
RESULTS_PATH = _RESULTS_DIR / "EA_results.csv"

TABLE_HEADER = [
    "ea_id", "method", "family", "type",
    "val_f1", "val_auc", "val_recall", "val_precision", "val_threshold",
    "test_f1", "test_auc", "test_recall", "test_precision", "test_threshold",
    "vs_gel_val_f1", "vs_gel_val_auc",
    "reference", "notes",
]

RESULTS_HEADER = [
    "ea_id", "method", "family", "type",
    # train split (populated by learned methods; blank for parameter-free)
    "train_f1", "train_auc", "train_acc", "train_recall", "train_specificity", "train_threshold",
    # val split (primary ranking axis)
    "val_f1",   "val_auc",   "val_acc",   "val_recall",   "val_specificity",   "val_threshold",
    # test split (held out — reported for completeness)
    "test_f1",  "test_auc",  "test_acc",  "test_recall",  "test_specificity",  "test_threshold",
]

GEL_V3_VAL_F1  = 0.7394
GEL_V3_VAL_AUC = 0.9060


def get_logger(ea_id: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{ea_id}.log"
    logger = logging.getLogger(ea_id)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Find threshold that maximises F1; return full metrics at that threshold."""
    best = {
        "f1": 0.0, "threshold": 0.5,
        "recall": 0.0, "precision": 0.0,
        "accuracy": 0.0, "specificity": 0.0,
    }
    for thr in THRESHOLD_RANGE:
        preds = (scores >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best["f1"]:
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            best["f1"]          = float(f1)
            best["threshold"]   = float(thr)
            best["recall"]      = float(recall_score(y_true, preds, zero_division=0))
            best["precision"]   = float(precision_score(y_true, preds, zero_division=0))
            best["accuracy"]    = float((tp + tn) / len(y_true))
            best["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return best


def evaluate(y_true: np.ndarray, scores: np.ndarray, split: str, logger: logging.Logger) -> dict:
    """Full evaluation: threshold-swept F1 + AUC + accuracy + specificity."""
    sweep = threshold_sweep(y_true, scores)
    auc   = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0

    logger.info(
        f"[{split}] F1={sweep['f1']:.4f}  AUC={auc:.4f}  "
        f"Acc={sweep['accuracy']:.4f}  Recall={sweep['recall']:.4f}  "
        f"Spec={sweep['specificity']:.4f}  Prec={sweep['precision']:.4f}  "
        f"thr={sweep['threshold']:.3f}"
    )
    return {
        "f1":          sweep["f1"],
        "auc":         auc,
        "accuracy":    sweep["accuracy"],
        "recall":      sweep["recall"],
        "specificity": sweep["specificity"],
        "precision":   sweep["precision"],
        "threshold":   sweep["threshold"],
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    ea_id: str,
    method: str,
    plot_dir: Path,
    logger: logging.Logger,
) -> None:
    """Seaborn confusion matrix heatmap at the val-optimal threshold."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    labels = ["Non-fractured", "Fractured"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{ea_id}: {method}\nConfusion Matrix  (thr={threshold:.3f})", fontsize=11)
    fig.tight_layout()
    out = plot_dir / f"{ea_id}_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved -> plots/{ea_id}/{ea_id}_confusion_matrix.png")


def plot_confidence_accuracy(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    ea_id: str,
    method: str,
    plot_dir: Path,
    logger: logging.Logger,
    n_bins: int = 10,
) -> None:
    """Reliability bar chart: for each 0.1-wide score bin, show % of correct predictions.

    A prediction is correct when the thresholded label matches the true label.
    Bars are coloured by bin midpoint (low=blue, high=red) to visually indicate
    which end the model is confident about. Count of instances shown on each bar.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    y_pred = (scores >= threshold).astype(int)
    correct = (y_pred == y_true).astype(int)

    accuracies, counts, midpoints = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            accuracies.append(np.nan)
            counts.append(0)
        else:
            accuracies.append(correct[mask].mean() * 100)
            counts.append(mask.sum())
        midpoints.append((lo + hi) / 2)

    cmap = plt.cm.RdYlGn
    colours = [cmap(acc / 100) if not np.isnan(acc) else (0.85, 0.85, 0.85, 1) for acc in accuracies]
    bin_labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(n_bins), accuracies, color=colours, edgecolor="white", linewidth=0.8)

    for bar, acc, cnt in zip(bars, accuracies, counts):
        if cnt == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 2, 95),
            f"{acc:.0f}%\n(n={cnt})",
            ha="center", va="bottom", fontsize=8.5,
        )

    ax.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.4, label="Perfect")
    ax.axvline(threshold * n_bins - 0.5, color="black", linestyle=":", linewidth=1.2,
               label=f"Threshold={threshold:.3f}")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Ensemble score bin", fontsize=11)
    ax.set_ylabel("Prediction accuracy (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(f"{ea_id}: {method}\nConfidence–Accuracy (val)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = plot_dir / f"{ea_id}_confidence_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved -> plots/{ea_id}/{ea_id}_confidence_accuracy.png")


def plot_training_curve(
    ea_id: str,
    method: str,
    plot_dir: Path,
    logger: logging.Logger,
    train_series: list,
    val_series: list,
    x_label: str = "Iteration",
    train_label: str = "Train loss",
    val_label: str = "Val metric",
    best_idx: int = None,
    dual_axis: bool = False,
    x_values: list = None,
    x_log: bool = False,
) -> None:
    """Generic training / regularisation-path curve.

    dual_axis=True: train_series on left y-axis, val_series on right y-axis
                    (use when metrics are on different scales, e.g. loss vs accuracy).
    dual_axis=False: both series share the same y-axis (e.g. both are log-loss).
    x_values: optional list of x-axis tick values (e.g. C grid). If None, uses range(len).
    x_log: log-scale x-axis (useful for regularisation paths over C or alpha).
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    xs = x_values if x_values is not None else list(range(len(train_series)))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(xs, train_series, "o-" if x_values else "-",
             color="steelblue", linewidth=1.8, markersize=5, label=train_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(train_label, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    if x_log:
        ax1.set_xscale("log")

    if dual_axis:
        ax2 = ax1.twinx()
        ax2.plot(xs, val_series, "o--" if x_values else "--",
                 color="tomato", linewidth=1.8, markersize=5, label=val_label)
        ax2.set_ylabel(val_label, color="tomato")
        ax2.tick_params(axis="y", labelcolor="tomato")
        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2, fontsize=9)
    else:
        ax1.plot(xs, val_series, "o--" if x_values else "--",
                 color="tomato", linewidth=1.8, markersize=5, label=val_label)
        ax1.legend(fontsize=9)

    if best_idx is not None:
        bx = xs[best_idx] if x_values else best_idx
        ax1.axvline(bx, color="gray", linestyle=":", linewidth=1.2,
                    label=f"Best @ {bx}")

    ax1.set_title(f"{ea_id}: {method}\nTraining curve")
    fig.tight_layout()
    out = plot_dir / f"{ea_id}_training_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved -> plots/{ea_id}/{ea_id}_training_curve.png")


def _upsert_csv(path: Path, fieldnames: list, row: dict, seed_rows: list = None) -> None:
    """Write row to CSV, replacing any existing row with the same ea_id (idempotent).

    If the file does not exist, it is created with the header and optional seed_rows first.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ea_id = row["ea_id"]

    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        rows = [r for r in existing if r.get("ea_id") != ea_id]
        rows.append(row)
    else:
        rows = (seed_rows or []) + [row]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


_GEL_BASELINE_ROW = {
    "ea_id": "GEL-v3", "method": "Gated Ensemble Logic",
    "family": "Custom 4-stage", "type": "BASELINE",
    "val_f1": GEL_V3_VAL_F1, "val_auc": GEL_V3_VAL_AUC,
    "val_recall": 0.7439, "val_precision": 0.7349, "val_threshold": 0.400,
    "test_f1": 0.6718, "test_auc": 0.8916,
    "test_recall": 0.7213, "test_precision": 0.6286, "test_threshold": 0.325,
    "vs_gel_val_f1": 0.0, "vs_gel_val_auc": 0.0,
    "reference": "This thesis", "notes": "GEL v3 grid-optimal gamma=11.4 delta=0.21",
}


def append_to_table(row: dict) -> None:
    """Upsert one result row into EA_comparative_table.csv (idempotent on ea_id)."""
    row["vs_gel_val_f1"]  = round(row.get("val_f1",  0) - GEL_V3_VAL_F1,  4)
    row["vs_gel_val_auc"] = round(row.get("val_auc", 0) - GEL_V3_VAL_AUC, 4)
    _upsert_csv(TABLE_PATH, TABLE_HEADER, row, seed_rows=[_GEL_BASELINE_ROW])


def append_to_results(row: dict) -> None:
    """Upsert comprehensive per-split metrics into EA_results.csv (idempotent on ea_id)."""
    _upsert_csv(RESULTS_PATH, RESULTS_HEADER, row)
