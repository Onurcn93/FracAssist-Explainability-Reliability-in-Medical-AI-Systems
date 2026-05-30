"""
EA-C4: Decision-template combiner (stacking).

Prototype-based, parameter-free combiner. During training, computes one
"decision template" per class: the mean probability vector
[mu_resnet, mu_densenet, mu_efficientnet] across all training samples of
that class.

At inference, a sample's continuous score is its relative similarity to
the fractured template:

  sim(x, T) = 1 - MSE(x, T)
  score(x) = sim(x, T_frac) / (sim(x, T_frac) + sim(x, T_nonfrac))

No gradient descent, no hyperparameters, no iterative training.
The method is fully determined by the two class-mean templates — making
it the simplest possible learned combiner in Family C.

Reference: Kuncheva, L.I., Bezdek, J.C., & Duin, R.P.W. (2001).
           Decision templates for multiple classifier fusion: an
           experimental comparison. Pattern Recognition, 34(2):299-314.
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

EA_ID       = "EA-C4"
METHOD      = "Decision-template combiner"
FAMILY      = "C"
TYPE        = "LIT"
REF         = "Kuncheva et al. 2001 (Pattern Recognition)"

DATA_CSV    = "data/all_base.csv"
FEAT_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]


def _fit_templates(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute class-mean decision templates from training data."""
    T0 = X_train[y_train == 0].mean(axis=0)   # non-fractured template
    T1 = X_train[y_train == 1].mean(axis=0)   # fractured template
    return T0, T1


def _score(X: np.ndarray, T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
    """Relative similarity score toward fractured template.

    sim(x, T) = 1 - mean((x - T)^2)
    score = sim(x, T1) / (sim(x, T0) + sim(x, T1))
    """
    sim0 = 1.0 - np.mean((X - T0) ** 2, axis=1)
    sim1 = 1.0 - np.mean((X - T1) ** 2, axis=1)
    denom = sim0 + sim1
    denom = np.where(denom == 0, 1e-9, denom)   # numerical safety
    return sim1 / denom


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    train_mask = df["split"].values == "train"
    val_mask   = df["split"].values == "val"
    test_mask  = df["split"].values == "test"
    labels     = (df["true_label"] == "Fractured").astype(int).values

    X       = df[FEAT_COLS].values
    X_train = X[train_mask]
    y_train = labels[train_mask]
    X_val   = X[val_mask]
    y_val   = labels[val_mask]

    logger.info(f"Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    # --- Fit: compute class-mean templates ---
    T0, T1 = _fit_templates(X_train, y_train)
    logger.info("Decision templates (class-mean prob vectors):")
    for name, t0, t1 in zip(MODEL_NAMES, T0, T1):
        logger.info(f"  {name:12s}  T_nonfrac={t0:.4f}  T_frac={t1:.4f}")
    logger.info(f"  Template gap (T1-T0): {T1 - T0}")

    # --- Score all splits ---
    scores = _score(X, T0, T1)

    # --- Evaluate ---
    logger.info("--- Train (in-sample, overfitting check) ---")
    train_m = evaluate(labels[train_mask], scores[train_mask], "train", logger)
    logger.info("--- Val ---")
    val_m   = evaluate(labels[val_mask],   scores[val_mask],   "val",   logger)
    logger.info("--- Test ---")
    test_m  = evaluate(labels[test_mask],  scores[test_mask],  "test",  logger)

    logger.info(f"Train-Val F1 gap: {train_m['f1'] - val_m['f1']:.4f}")

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
        ax.set_xlabel("Decision-template score (relative similarity to T_frac)")
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

        # Plot 2: decision templates — grouped bar (T0 vs T1 per model)
        x_pos = np.arange(len(MODEL_NAMES))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x_pos - width/2, T0, width, label="T_nonfrac (non-fractured)",
               color="steelblue", alpha=0.85)
        ax.bar(x_pos + width/2, T1, width, label="T_frac (fractured)",
               color="tomato", alpha=0.85)
        for i, (t0, t1) in enumerate(zip(T0, T1)):
            ax.annotate(f"{t0:.3f}", (x_pos[i] - width/2, t0 + 0.01),
                        ha="center", fontsize=9)
            ax.annotate(f"{t1:.3f}", (x_pos[i] + width/2, t1 + 0.01),
                        ha="center", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(MODEL_NAMES)
        ax.set_ylabel("Mean probability (training set)")
        ax.set_ylim(0, 1.0)
        ax.set_title(
            f"{EA_ID}: Decision templates (class-mean probability vectors)\n"
            f"T_frac - T_nonfrac gap: "
            + "  ".join(f"{n}={t1-t0:+.3f}" for n, t0, t1 in zip(MODEL_NAMES, T0, T1))
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_templates.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_templates.png")

        # Plot 3: similarity scatter (val) — sim_to_T0 vs sim_to_T1, colored by true label
        sim0_val = 1.0 - np.mean((X_val - T0) ** 2, axis=1)
        sim1_val = 1.0 - np.mean((X_val - T1) ** 2, axis=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sim0_val[val_labels == 0], sim1_val[val_labels == 0],
                   c="steelblue", alpha=0.5, s=18, label="Non-fractured")
        ax.scatter(sim0_val[val_labels == 1], sim1_val[val_labels == 1],
                   c="tomato", alpha=0.5, s=18, label="Fractured")
        # decision boundary: sim1 = sim0  =>  score = 0.5
        lims = [
            min(sim0_val.min(), sim1_val.min()) - 0.002,
            max(sim0_val.max(), sim1_val.max()) + 0.002,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="score=0.5 boundary")
        ax.set_xlabel("Similarity to T_nonfrac  (1 - MSE)")
        ax.set_ylabel("Similarity to T_frac  (1 - MSE)")
        ax.set_title(
            f"{EA_ID}: Similarity space (val set)\n"
            f"Points above diagonal → assigned fractured"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_similarity_scatter.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_similarity_scatter.png")

        plot_confusion_matrix(labels[val_mask], scores[val_mask], val_m["threshold"],
                              EA_ID, METHOD, plot_dir, logger)
        plot_confidence_accuracy(labels[val_mask], scores[val_mask], val_m["threshold"],
                                 EA_ID, METHOD, plot_dir, logger)

    # --- Append to comparative table ---
    gap_str = "  ".join(
        f"{n}={t1-t0:+.4f}" for n, t0, t1 in zip(MODEL_NAMES, T0, T1)
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
            f"Parameter-free: 2 class-mean templates on train. "
            f"Score=sim(T_frac)/(sim(T_frac)+sim(T_nonfrac)), sim=1-MSE. "
            f"Template gap (T1-T0): {gap_str}. "
            f"Train in-sample F1={train_m['f1']:.4f}."
        ),
    }
    append_to_table(row)
    logger.info("Row appended to EA_comparative_table.csv")

    results_row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "train_f1":          train_m["f1"],          "train_auc":          train_m["auc"],
        "train_acc":         train_m["accuracy"],     "train_recall":       train_m["recall"],
        "train_specificity": train_m["specificity"],  "train_threshold":    train_m["threshold"],
        "val_f1":            val_m["f1"],              "val_auc":            val_m["auc"],
        "val_acc":           val_m["accuracy"],        "val_recall":         val_m["recall"],
        "val_specificity":   val_m["specificity"],     "val_threshold":      val_m["threshold"],
        "test_f1":           test_m["f1"],             "test_auc":           test_m["auc"],
        "test_acc":          test_m["accuracy"],       "test_recall":        test_m["recall"],
        "test_specificity":  test_m["specificity"],    "test_threshold":     test_m["threshold"],
    }
    append_to_results(results_row)
    logger.info("Row appended to EA_results.csv")

    return row


if __name__ == "__main__":
    THIS_DIR = Path(__file__).resolve().parent
    ROOT_DIR = THIS_DIR.parent
    run(ROOT_DIR)
