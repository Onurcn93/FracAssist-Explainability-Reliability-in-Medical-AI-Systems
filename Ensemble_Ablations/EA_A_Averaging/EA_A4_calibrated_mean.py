"""
EA-A4: Calibrated mean ensemble.

Applies per-model temperature scaling before averaging:

    logit_i  = log(p_i / (1 - p_i))
    p_cal_i  = sigmoid(logit_i / T_i)
    score    = mean(p_cal_1, p_cal_2, p_cal_3)

Each temperature T_i is fit independently on the train split by minimising
negative log-likelihood (NLL). T > 1 pushes probabilities toward 0.5
(less confident); T < 1 sharpens them. After calibration the mean is
taken — same as EA-A1 but on well-calibrated inputs.

Motivation: raw CNN softmax outputs are typically overconfident (Guo 2017),
which biases the mean toward extremes. Calibration corrects this before
combining, potentially improving both AUC and threshold stability.

This is an original contribution: the literature applies temperature scaling
to a single model; here it is applied per-model before ensemble fusion.

Reference: Guo et al. 2017 — On Calibration of Modern Neural Networks, ICML.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit  # numerically stable sigmoid

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)

EA_ID   = "EA-A4"
METHOD  = "Calibrated mean"
FAMILY  = "A"
TYPE    = "ORIG"
REF     = "Guo et al. 2017"

DATA_CSV  = "data/all_base.csv"
PROB_COLS = ["resnet_probability", "densenet_probability", "efficientnet_probability"]


def _fit_temperature(probs_train: np.ndarray, labels_train: np.ndarray) -> float:
    """Minimise NLL over scalar T in [0.1, 10.0] using bounded search."""
    logits = np.log(np.clip(probs_train, 1e-7, 1 - 1e-7) /
                    (1 - np.clip(probs_train, 1e-7, 1 - 1e-7)))

    def nll(T: float) -> float:
        p_cal = expit(logits / T)
        return -np.mean(
            labels_train * np.log(p_cal + 1e-12) +
            (1 - labels_train) * np.log(1 - p_cal + 1e-12)
        )

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) /
                    (1 - np.clip(probs, 1e-7, 1 - 1e-7)))
    return expit(logits / T)


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:  # noqa: ARG001
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

    labels = (df["true_label"] == "Fractured").astype(int).values
    probs  = df[PROB_COLS].values  # shape (N, 3)

    # --- Fit one temperature per model on train split ---
    temperatures = []
    for i, col in enumerate(PROB_COLS):
        T = _fit_temperature(probs[train_mask, i], labels[train_mask])
        temperatures.append(T)
        logger.info(f"Temperature [{col}]: T={T:.4f}")

    # --- Apply calibration to all splits ---
    cal_probs = np.stack(
        [_apply_temperature(probs[:, i], temperatures[i]) for i in range(3)],
        axis=1
    )  # shape (N, 3)

    scores = cal_probs.mean(axis=1)  # calibrated mean

    # --- Evaluate on each split ---
    logger.info("--- Train ---")
    train_m = evaluate(labels[train_mask], scores[train_mask], "train", logger)
    logger.info("--- Val ---")
    val_m   = evaluate(labels[val_mask],   scores[val_mask],   "val",   logger)
    logger.info("--- Test ---")
    test_m  = evaluate(labels[test_mask],  scores[test_mask],  "test",  logger)

    temp_str = "  ".join(f"T{i+1}={t:.3f}" for i, t in enumerate(temperatures))
    logger.info(f"Temperatures: {temp_str}")

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
        ax.set_xlabel("Calibrated mean score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}\n"
            f"{temp_str}"
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
        "notes": (
            f"Per-model temperature scaling (train NLL) before mean. "
            f"{temp_str}. "
            "Corrects overconfidence before fusion."
        ),
    }
    append_to_table(row)
    logger.info(f"Row appended to EA_comparative_table.csv")

    # --- Append to full results tracker (all metrics, all splits) ---
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
    logger.info(f"Row appended to EA_results.csv")

    return row
