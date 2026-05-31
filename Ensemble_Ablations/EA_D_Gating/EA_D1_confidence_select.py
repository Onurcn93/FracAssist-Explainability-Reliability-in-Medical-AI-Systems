"""
EA-D1: Confidence-based selection (highest margin wins).

For each sample, the model with the highest absolute margin
|p_i - 0.5| is selected. That model's probability is used directly
as the ensemble score.

score(x) = p_{argmax_i |p_i - 0.5|}(x)

Rationale: the model with the most extreme output is assumed to be
the most informative for that input. This is the simplest dynamic
classifier selection (DCS) rule — no region-of-competence estimation,
no training required. Serves as the DCS floor.

Parameter-free: threshold swept on val only.

Reference: Cruz, R.M.O., Sabourin, R., & Cavalcanti, G.D.C. (2018).
           Dynamic classifier selection: Recent advances and perspectives.
           Information Fusion, 41:195–216.
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

EA_ID       = "EA-D1"
METHOD      = "Confidence-based selection (highest margin)"
FAMILY      = "D"
TYPE        = "LIT"
REF         = "Cruz et al. 2018"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:  # noqa: ARG001
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    df     = pd.read_csv(root_dir / DATA_CSV)
    labels = (df["true_label"] == "Fractured").astype(int).values
    probs  = df[PROB_COLS].values                              # (N, 3)

    val_mask  = df["split"].values == "val"
    test_mask = df["split"].values == "test"

    logger.info(f"Loaded {len(df)} rows | Val={val_mask.sum()} | Test={test_mask.sum()}")

    # --- Dynamic model selection: argmax |p_i - 0.5| per sample ---
    margins  = np.abs(probs - 0.5)                            # (N, 3)
    selector = margins.argmax(axis=1)                         # (N,) — 0/1/2
    scores   = probs[np.arange(len(probs)), selector]         # (N,)

    # Selection statistics (val)
    val_selector = selector[val_mask]
    logger.info("Model selection frequency on val:")
    for i, name in enumerate(MODEL_NAMES):
        n   = (val_selector == i).sum()
        pct = n / len(val_selector) * 100
        logger.info(f"  {name}: {n} samples ({pct:.1f}%)")

    # Average margin per selected model on val
    val_margins = margins[val_mask]
    for i, name in enumerate(MODEL_NAMES):
        mask_i = val_selector == i
        if mask_i.any():
            avg_m = val_margins[mask_i, i].mean()
            logger.info(f"  {name} avg margin when selected: {avg_m:.4f}")

    # --- Evaluate ---
    logger.info("--- Val ---")
    val_m  = evaluate(labels[val_mask],  scores[val_mask],  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(labels[test_mask], scores[test_mask], "test", logger)

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: score distribution on val
        val_scores = scores[val_mask]
        val_labels = labels[val_mask]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(val_scores[val_labels == 0], bins=30, alpha=0.6,
                label="Non-fractured", color="steelblue")
        ax.hist(val_scores[val_labels == 1], bins=30, alpha=0.6,
                label="Fractured", color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("Selected model probability")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        # Plot 2: selector distribution — model selection frequency on val
        counts  = [(val_selector == i).sum() for i in range(3)]
        colours = ["steelblue", "tomato", "forestgreen"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(MODEL_NAMES, counts, color=colours, edgecolor="white", linewidth=0.8)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{cnt}\n({cnt/len(val_selector)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_ylabel("Count (val samples selected)")
        ax.set_title(
            f"{EA_ID}: Model selection frequency (val)\n"
            "argmax |p_i − 0.5| — most confident model wins"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_selector_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_selector_dist.png")

        # Plot 3: margin distribution per model on val
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for i, (name, ax) in enumerate(zip(MODEL_NAMES, axes)):
            ax.hist(val_margins[:, i], bins=20, color=colours[i], alpha=0.7, edgecolor="white")
            ax.axvline(val_margins[:, i].mean(), color="black", linestyle="--",
                       label=f"Mean={val_margins[:,i].mean():.3f}")
            ax.set_xlabel("|p − 0.5|")
            ax.set_title(name)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Count (val)")
        fig.suptitle(f"{EA_ID}: Confidence margin distributions (val)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_margin_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_margin_dist.png")

        plot_confusion_matrix(labels[val_mask], scores[val_mask], val_m["threshold"],
                              EA_ID, METHOD, plot_dir, logger)
        plot_confidence_accuracy(labels[val_mask], scores[val_mask], val_m["threshold"],
                                 EA_ID, METHOD, plot_dir, logger)

    # --- Append to comparative table ---
    sel_str = "  ".join(
        f"{n}={100*(val_selector==i).mean():.1f}%" for i, n in enumerate(MODEL_NAMES)
    )
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
            f"Select model with highest |p_i - 0.5| per sample. "
            f"Val selection: {sel_str}. Parameter-free."
        ),
    }
    append_to_table(row)
    logger.info("Row appended to EA_comparative_table.csv")

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


if __name__ == "__main__":
    THIS_DIR = Path(__file__).resolve().parent
    ROOT_DIR = THIS_DIR.parent
    run(ROOT_DIR)
