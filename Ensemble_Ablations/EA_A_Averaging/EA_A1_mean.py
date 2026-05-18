"""
EA-A1: Mean (sum rule) ensemble.

Average of the three CNN softmax probabilities — equal weights, no training required.
The canonical parameter-free floor for any ensemble evaluation.

Reference: Kuncheva 2014 — Combining Pattern Classifiers, Ch. 3;
           Müller et al. 2022 — Medical image ensembles review.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import get_logger, evaluate, append_to_table

EA_ID   = "EA-A1"
METHOD  = "Mean (sum rule)"
FAMILY  = "A"
TYPE    = "LIT"
REF     = "Kuncheva 2014; Müller et al. 2022"

DATA_CSV  = "data/all_base.csv"
PROB_COLS = ["resnet_probability", "densenet_probability", "efficientnet_probability"]


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots"
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    # --- Ensemble: simple mean of the three probabilities ---
    scores = df[PROB_COLS].mean(axis=1).values
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
        val_scores  = scores[val_mask]
        val_labels  = labels[val_mask]
        ax.hist(val_scores[val_labels == 0], bins=30, alpha=0.6, label="Non-fractured", color="steelblue")
        ax.hist(val_scores[val_labels == 1], bins=30, alpha=0.6, label="Fractured",     color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("Mean ensemble score")
        ax.set_ylabel("Count")
        ax.set_title(f"{EA_ID}: {METHOD}\nVal F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}_score_dist.png")

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
        "notes": "No training. Equal weights. Parameter-free floor.",
    }
    append_to_table(row)
    logger.info(f"Row appended to EA_comparative_table.csv")

    return row
