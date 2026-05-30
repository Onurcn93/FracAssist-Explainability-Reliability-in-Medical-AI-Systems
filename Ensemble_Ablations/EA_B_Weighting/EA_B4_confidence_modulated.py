"""
EA-B4: Confidence-modulated weighting.

Per-sample dynamic weights derived from range_reliability — the historical
train accuracy of each model when its output probability falls in the same
decile band as the current prediction.

For each sample x:
    w_i(x) = range_reliability_i(x) / sum_j(range_reliability_j(x))
    score(x) = sum_i w_i(x) * p_i(x)

This is a per-sample (dynamic) combiner: every image gets its own weight
triple, unlike B1-B3 which use a single global weight vector. The weight
matrix has shape (N, 3) rather than (3,).

range_reliability is computed by build_features.py Tier 3 from train split
only (fit on train, applied to all splits). No data leakage.

Requires: data/all_EA.csv (built with --tiers 1 2 3).

Reference: Extends Woloszynski & Kurzynski (2011) — A measure of competence
           based on random classification for dynamic ensemble selection.
           Pattern Recognition, 44(10-11):2386-2396.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)

EA_ID       = "EA-B4"
METHOD      = "Confidence-modulated weighting"
FAMILY      = "B"
TYPE        = "ORIG"
REF         = "Woloszynski & Kurzynski 2011"

DATA_CSV    = "data/all_EA.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
RR_COLS     = ["range_reliability_res", "range_reliability_dense", "range_reliability_eff"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]


def _compute_per_sample_weights(df: pd.DataFrame, logger) -> np.ndarray:
    """Return weight matrix (N, 3) from range_reliability columns."""
    rr = df[RR_COLS].values.astype(float)          # shape (N, 3)
    total = rr.sum(axis=1, keepdims=True).clip(min=1e-9)
    weights = rr / total                            # each row sums to 1

    # Log per-model mean weight (on val) for diagnostics
    val_mask = df["split"].values == "val"
    mean_w = weights[val_mask].mean(axis=0)
    for name, mw in zip(MODEL_NAMES, mean_w):
        logger.info(f"  Mean val weight {name}: {mw:.4f}")

    # Log per-model mean range_reliability (on val)
    mean_rr = rr[val_mask].mean(axis=0)
    for name, mr in zip(MODEL_NAMES, mean_rr):
        logger.info(f"  Mean val range_reliability {name}: {mr:.4f}")

    return weights


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:  # noqa: ARG001
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV} ({len(df.columns)} columns)")

    # Verify required columns are present
    missing = [c for c in RR_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}. "
            "Run: python data/build_features.py --tiers 1 2 3"
        )

    # --- Per-sample dynamic weights from range_reliability ---
    logger.info("Computing per-sample weights from range_reliability:")
    weights = _compute_per_sample_weights(df, logger)   # (N, 3)

    probs  = df[PROB_COLS].values                       # (N, 3)
    scores = (probs * weights).sum(axis=1)              # (N,)

    labels    = (df["true_label"] == "Fractured").astype(int).values
    val_mask  = df["split"].values == "val"
    test_mask = df["split"].values == "test"

    logger.info("--- Val ---")
    val_m  = evaluate(labels[val_mask],  scores[val_mask],  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(labels[test_mask], scores[test_mask], "test", logger)

    # Mean weights summary string for notes
    mean_w_val = weights[val_mask].mean(axis=0)
    weight_str = "  ".join(f"{n}={w:.4f}" for n, w in zip(MODEL_NAMES, mean_w_val))
    logger.info(f"Mean val weights: {weight_str}")

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Score distribution on val
        fig, ax = plt.subplots(figsize=(7, 4))
        val_scores = scores[val_mask]
        val_labels = labels[val_mask]
        ax.hist(val_scores[val_labels == 0], bins=30, alpha=0.6,
                label="Non-fractured", color="steelblue")
        ax.hist(val_scores[val_labels == 1], bins=30, alpha=0.6,
                label="Fractured", color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("Confidence-modulated ensemble score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}\n"
            f"Mean val weights: {weight_str}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        # Weight distribution: histogram of per-sample weights on val
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for i, (name, ax) in enumerate(zip(MODEL_NAMES, axes)):
            ax.hist(weights[val_mask, i], bins=20, color="steelblue", alpha=0.8)
            ax.axvline(mean_w_val[i], color="tomato", linestyle="--",
                       label=f"Mean={mean_w_val[i]:.3f}")
            ax.set_xlabel("Per-sample weight")
            ax.set_title(name)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Count")
        fig.suptitle(f"{EA_ID}: Per-sample weight distributions (val)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_weight_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_weight_dist.png")

        plot_confusion_matrix(labels[val_mask], scores[val_mask], val_m["threshold"],
                              EA_ID, METHOD, plot_dir, logger)
        plot_confidence_accuracy(labels[val_mask], scores[val_mask], val_m["threshold"],
                                 EA_ID, METHOD, plot_dir, logger)

    # --- Append to comparative table ---
    row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "val_f1":        val_m["f1"],        "val_auc":       val_m["auc"],
        "val_recall":    val_m["recall"],     "val_precision": val_m["precision"],
        "val_threshold": val_m["threshold"],
        "test_f1":       test_m["f1"],        "test_auc":      test_m["auc"],
        "test_recall":   test_m["recall"],    "test_precision":test_m["precision"],
        "test_threshold":test_m["threshold"],
        "reference": REF,
        "notes": (
            f"Per-sample weights from range_reliability (train decile accuracy). "
            f"Mean val weights: {weight_str}. "
            "Dynamic (N x 3) weight matrix — each sample has its own combiner."
        ),
    }
    append_to_table(row)
    logger.info("Row appended to EA_comparative_table.csv")

    # --- Append to full results tracker ---
    results_row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "val_f1":          val_m["f1"],          "val_auc":          val_m["auc"],
        "val_acc":         val_m["accuracy"],     "val_recall":       val_m["recall"],
        "val_specificity": val_m["specificity"],  "val_threshold":    val_m["threshold"],
        "test_f1":         test_m["f1"],          "test_auc":         test_m["auc"],
        "test_acc":        test_m["accuracy"],     "test_recall":      test_m["recall"],
        "test_specificity":test_m["specificity"], "test_threshold":   test_m["threshold"],
    }
    append_to_results(results_row)
    logger.info("Row appended to EA_results.csv")

    return row
