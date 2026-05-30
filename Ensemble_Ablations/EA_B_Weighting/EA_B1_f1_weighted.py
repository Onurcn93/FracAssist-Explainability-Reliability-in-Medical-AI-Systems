"""
EA-B1: F1-weighted average.

Each model's contribution is weighted by its individual val F1 score, computed
via threshold sweep on the val split. Weights are normalised to sum to 1:

    w_i = F1_i / sum(F1_j)
    score = w1*p1 + w2*p2 + w3*p3

This is the simplest performance-weighted combiner — models that performed
better individually get a larger say in the ensemble. No training required;
weights are static scalars derived from known val performance, consistent
with the GEL v3 evaluation protocol (Kuncheva 2014, Ch. 3).

Expected behaviour: if the three models differ meaningfully in val F1,
weighting should improve over the equal-weight mean (EA-A1). If they are
similar in performance, B1 ≈ A1.

Reference: Kuncheva (2014) — Combining Pattern Classifiers, Ch. 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)

EA_ID   = "EA-B1"
METHOD  = "F1-weighted average"
FAMILY  = "B"
TYPE    = "LIT"
REF     = "Kuncheva 2014"

DATA_CSV  = "data/all_base.csv"
PROB_COLS = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]


def _compute_model_weights(df: pd.DataFrame, logger) -> np.ndarray:
    """Compute per-model val F1 via threshold sweep; return normalised weights."""
    val_mask = df["split"].values == "val"
    labels   = (df["true_label"] == "Fractured").astype(int).values

    f1_scores = []
    for col, name in zip(PROB_COLS, MODEL_NAMES):
        sweep = threshold_sweep(labels[val_mask], df[col].values[val_mask])
        f1_scores.append(sweep["f1"])
        logger.info(f"  {name}: val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}")

    weights = np.array(f1_scores)
    weights = weights / weights.sum()
    for name, w, f1 in zip(MODEL_NAMES, weights, f1_scores):
        logger.info(f"  Weight {name}: {w:.4f}  (F1={f1:.4f})")
    return weights


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:  # noqa: ARG001
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    # --- Compute static F1-based weights from val split ---
    logger.info("Computing per-model val F1 weights:")
    weights = _compute_model_weights(df, logger)

    probs  = df[PROB_COLS].values       # shape (N, 3)
    scores = probs @ weights            # weighted sum

    labels    = (df["true_label"] == "Fractured").astype(int).values
    val_mask  = df["split"].values == "val"
    test_mask = df["split"].values == "test"

    logger.info("--- Val ---")
    val_m  = evaluate(labels[val_mask],  scores[val_mask],  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(labels[test_mask], scores[test_mask], "test", logger)

    weight_str = "  ".join(f"{n}={w:.4f}" for n, w in zip(MODEL_NAMES, weights))
    logger.info(f"Weights: {weight_str}")

    # --- Plot: score distribution on val ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        val_scores = scores[val_mask]
        val_labels = labels[val_mask]
        ax.hist(val_scores[val_labels == 0], bins=30, alpha=0.6, label="Non-fractured", color="steelblue")
        ax.hist(val_scores[val_labels == 1], bins=30, alpha=0.6, label="Fractured",     color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("F1-weighted ensemble score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}\n"
            f"{weight_str}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")
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
        "notes": f"F1-weighted average. {weight_str}. Static weights from val threshold-swept F1.",
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
