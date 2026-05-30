"""
EA-B3: AUC-weighted average.

Each model's contribution is weighted by its individual val AUC score,
normalised to sum to 1:

    w_i = AUC_i^val / sum(AUC_j^val)
    score = w1*p1 + w2*p2 + w3*p3

AUC is a threshold-independent metric — it measures discriminative ability
across all operating points, not just the F1-optimal threshold. Weighting
by AUC therefore captures a different aspect of model quality than F1-
weighted (EA-B1) or power-law RC (EA-B2).

Key comparison:
- EA-B1 weights by F1 (threshold-dependent) → DenseNet leads
- EA-B3 weights by AUC (threshold-independent) → model ranking may differ
- If AUC ranking ≠ F1 ranking: the two metrics identify different "best"
  models, and the two weighting schemes will produce different ensembles.

No training required. Weights are static scalars derived from val AUC.

Reference: Large & Bagnall (2019) — The heterogeneous ensembles of standard
           classification algorithms (HESCA). ECML-PKDD 2019.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)

EA_ID       = "EA-B3"
METHOD      = "AUC-weighted average"
FAMILY      = "B"
TYPE        = "LIT"
REF         = "Large & Bagnall 2019"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]


def _compute_auc_weights(df: pd.DataFrame, logger) -> np.ndarray:
    """Compute per-model val AUC; return normalised weights."""
    val_mask = df["split"].values == "val"
    labels   = (df["true_label"] == "Fractured").astype(int).values[val_mask]

    auc_scores = []
    for col, name in zip(PROB_COLS, MODEL_NAMES):
        auc = float(roc_auc_score(labels, df[col].values[val_mask]))
        auc_scores.append(auc)
        logger.info(f"  {name}: val AUC={auc:.4f}")

    weights = np.array(auc_scores)
    weights = weights / weights.sum()
    for name, w, auc in zip(MODEL_NAMES, weights, auc_scores):
        logger.info(f"  Weight {name}: {w:.4f}  (AUC={auc:.4f})")
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

    # --- Compute static AUC-based weights from val split ---
    logger.info("Computing per-model val AUC weights:")
    weights = _compute_auc_weights(df, logger)

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
        ax.set_xlabel("AUC-weighted ensemble score")
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
        "notes": f"AUC-weighted average. {weight_str}. Static weights from val AUC (threshold-independent).",
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
