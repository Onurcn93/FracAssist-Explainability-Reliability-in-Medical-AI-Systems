"""
EA-A2: Product rule ensemble.

Multiplies the three CNN posterior probabilities and normalises the result
to a proper probability in [0, 1]:

    score = (p1 * p2 * p3) / ((p1*p2*p3) + (1-p1)*(1-p2)*(1-p3))

This is the Bayesian product rule for independent classifiers with equal
priors (Kittler et al. 1998). Compared to the mean rule it is more
aggressive: a single uncertain model pulls the fused score toward 0.5,
so disagreement is penalised harder. Expected behaviour: lower recall
(conservative in calling fractures), higher precision, lower AUC due to
score compression.

Reference: Kittler et al. 1998 — On combining classifiers, IEEE TPAMI.
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

EA_ID   = "EA-A2"
METHOD  = "Product rule"
FAMILY  = "A"
TYPE    = "LIT"
REF     = "Kittler et al. 1998"

DATA_CSV  = "data/all_base.csv"
PROB_COLS = ["resnet_probability", "densenet_probability", "efficientnet_probability"]


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:  # noqa: ARG001
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    probs = df[PROB_COLS].values  # shape (N, 3)

    # Normalised product rule (Kittler 1998 posterior form)
    pos = probs.prod(axis=1)                    # P(fracture) product
    neg = (1 - probs).prod(axis=1)              # P(non-fracture) product
    scores = pos / (pos + neg + 1e-12)          # normalise to [0, 1]

    labels = (df["true_label"] == "Fractured").astype(int).values

    val_mask  = df["split"].values == "val"
    test_mask = df["split"].values == "test"

    logger.info("--- Val ---")
    val_m  = evaluate(labels[val_mask],  scores[val_mask],  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(labels[test_mask], scores[test_mask], "test", logger)

    # --- Plot: score distribution on val ---
    if plot:
        plot_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        val_scores = scores[val_mask]
        val_labels = labels[val_mask]
        ax.hist(val_scores[val_labels == 0], bins=30, alpha=0.6, label="Non-fractured", color="steelblue")
        ax.hist(val_scores[val_labels == 1], bins=30, alpha=0.6, label="Fractured",     color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("Product rule score")
        ax.set_ylabel("Count")
        ax.set_title(f"{EA_ID}: {METHOD}\nVal F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")
        plot_confusion_matrix(labels[val_mask], scores[val_mask], val_m["threshold"],
                              EA_ID, METHOD, plot_dir, logger)
        plot_confidence_accuracy(labels[val_mask], scores[val_mask], val_m["threshold"],
                                 EA_ID, METHOD, plot_dir, logger)

    # --- Append to comparative table (concise) ---
    row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "val_f1":        val_m["f1"],        "val_auc":       val_m["auc"],
        "val_recall":    val_m["recall"],     "val_precision": val_m["precision"],
        "val_threshold": val_m["threshold"],
        "test_f1":       test_m["f1"],        "test_auc":      test_m["auc"],
        "test_recall":   test_m["recall"],    "test_precision":test_m["precision"],
        "test_threshold":test_m["threshold"],
        "reference": REF,
        "notes": "Normalised product rule. No training. Penalises disagreement harder than mean.",
    }
    append_to_table(row)
    logger.info(f"Row appended to EA_comparative_table.csv")

    # --- Append to full results tracker (all metrics, all splits) ---
    results_row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        # train: parameter-free — no train metrics
        "val_f1":          val_m["f1"],          "val_auc":          val_m["auc"],
        "val_acc":         val_m["accuracy"],     "val_recall":       val_m["recall"],
        "val_specificity": val_m["specificity"],  "val_threshold":    val_m["threshold"],
        "test_f1":         test_m["f1"],          "test_auc":         test_m["auc"],
        "test_acc":        test_m["accuracy"],    "test_recall":      test_m["recall"],
        "test_specificity":test_m["specificity"], "test_threshold":   test_m["threshold"],
    }
    append_to_results(results_row)
    logger.info(f"Row appended to EA_results.csv")

    return row
