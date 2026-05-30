"""
EA-B2: RC-style power weighting.

Replicates GEL v3's RC initialisation formula in isolation — without the
OAM disagreement-triggered penalty or BVG gate. Quantifies the standalone
contribution of expert-dominant power-law weighting to GEL's val AUC.

RC formula (from GEL v3, gel_config.py):
    RC_i = (F1_i + b)^γ / Σ(F1_j + b)^γ

b = BASE_SHIFT = 0.5  — competency floor (matches gel_base_shift).
    Ensures (F1_i + b) > 1.0 for any F1_i > 0.5, so the exponent always
    amplifies rather than compresses the spread.

γ is swept on val; GEL's grid-optimal is γ=11.4 with joint δ search.
If best val performance occurs near γ=11.4 without OAM: RC weighting alone
explains most of GEL's gain. If best γ differs: the OAM-γ interaction was
load-bearing in GEL.

Ablation boundary: RC only.
OAM (Direction-Aware Asymmetric penalty) — EXCLUDED.
PDWF normalisation — INCLUDED (same formula: Σ RC_i × p_i / Σ RC_i, but
since RC already sums to 1, this reduces to the weighted dot product).
BVG gate — EXCLUDED (not a classification step in the ablation context).

Reference: This thesis (GEL v3 RC formula); Kuncheva 2014 Ch. 3
           (power-law generalisation of performance weighting).
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

EA_ID       = "EA-B2"
METHOD      = "RC power weighting"
FAMILY      = "B"
TYPE        = "LIT"
REF         = "This thesis (GEL v3 RC)"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

BASE_SHIFT  = 0.5    # matches gel_base_shift; keeps shifted F1 > 1.0
GEL_GAMMA   = 11.4   # GEL grid-optimal (joint γ×δ search, 10716 combos)
GAMMA_GRID  = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.4, 12.0, 14.0, 16.0, 20.0, 32.0]


def _rc_weights(f1_scores: np.ndarray, gamma: float) -> np.ndarray:
    """RC power-law weights: w_i = (F1_i + BASE_SHIFT)^γ / Σ(F1_j + BASE_SHIFT)^γ"""
    powered = (f1_scores + BASE_SHIFT) ** gamma
    return powered / powered.sum()


def _get_per_model_f1(df: pd.DataFrame, logger) -> np.ndarray:
    """Per-model val F1 via threshold sweep — same protocol as EA-B1."""
    val_mask = df["split"].values == "val"
    labels   = (df["true_label"] == "Fractured").astype(int).values

    f1_scores = []
    for col, name in zip(PROB_COLS, MODEL_NAMES):
        sweep = threshold_sweep(labels[val_mask], df[col].values[val_mask])
        f1_scores.append(sweep["f1"])
        logger.info(f"  {name}: val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}")
    return np.array(f1_scores)


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:  # noqa: ARG001
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    labels    = (df["true_label"] == "Fractured").astype(int).values
    val_mask  = df["split"].values == "val"
    test_mask = df["split"].values == "test"
    probs     = df[PROB_COLS].values   # shape (N, 3)

    # --- Per-model val F1 (feeds RC formula) ---
    logger.info("Computing per-model val F1 for RC formula:")
    f1_scores = _get_per_model_f1(df, logger)

    # --- γ sweep on val to find optimal power ---
    logger.info(f"Sweeping γ over {GAMMA_GRID}:")
    best_gamma, best_val_f1 = None, -1.0
    gamma_log = []
    for gamma in GAMMA_GRID:
        weights = _rc_weights(f1_scores, gamma)
        scores  = probs @ weights
        sweep   = threshold_sweep(labels[val_mask], scores[val_mask])
        gamma_log.append((gamma, weights.copy(), sweep["f1"], sweep["threshold"]))
        gel_marker = "  <-- GEL γ" if abs(gamma - GEL_GAMMA) < 0.05 else ""
        logger.info(
            f"  γ={gamma:5.1f}  w=[{weights[0]:.3f},{weights[1]:.3f},{weights[2]:.3f}]"
            f"  val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}{gel_marker}"
        )
        if sweep["f1"] > best_val_f1:
            best_val_f1, best_gamma = sweep["f1"], gamma

    logger.info(f"Best γ={best_gamma}  val F1={best_val_f1:.4f}")

    # --- Evaluate with best γ ---
    best_weights = _rc_weights(f1_scores, best_gamma)
    scores = probs @ best_weights

    logger.info("--- Val ---")
    val_m  = evaluate(labels[val_mask],  scores[val_mask],  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(labels[test_mask], scores[test_mask], "test", logger)

    weight_str = "  ".join(f"{n}={w:.4f}" for n, w in zip(MODEL_NAMES, best_weights))
    logger.info(f"Best γ={best_gamma} weights: {weight_str}")

    # --- GEL γ reference row ---
    gel_row = next(r for r in gamma_log if abs(r[0] - GEL_GAMMA) < 0.05)
    logger.info(
        f"GEL γ={GEL_GAMMA} reference: val F1={gel_row[2]:.4f}  thr={gel_row[3]:.3f}"
        f"  w=[{gel_row[1][0]:.3f},{gel_row[1][1]:.3f},{gel_row[1][2]:.3f}]"
    )

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: score distribution on val
        fig, ax = plt.subplots(figsize=(7, 4))
        val_scores = scores[val_mask]
        val_labels = labels[val_mask]
        ax.hist(val_scores[val_labels == 0], bins=30, alpha=0.6,
                label="Non-fractured", color="steelblue")
        ax.hist(val_scores[val_labels == 1], bins=30, alpha=0.6,
                label="Fractured", color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("RC power-weighted ensemble score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}  "
            f"γ={best_gamma}  b={BASE_SHIFT}\n"
            f"{weight_str}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        # Plot 2: γ sweep curve
        gammas  = [r[0] for r in gamma_log]
        f1_vals = [r[2] for r in gamma_log]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(gammas, f1_vals, "o-", color="steelblue", linewidth=2, markersize=6)
        ax.axvline(best_gamma, color="tomato", linestyle="--",
                   label=f"Best γ={best_gamma}  (F1={best_val_f1:.4f})")
        ax.axvline(GEL_GAMMA, color="green", linestyle=":",
                   label=f"GEL γ={GEL_GAMMA}")
        ax.set_xlabel("γ (power exponent)")
        ax.set_ylabel("Val F1 (threshold-swept)")
        ax.set_title(
            f"{EA_ID}: γ sweep — RC power weighting\n"
            f"b={BASE_SHIFT}, val split only"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_gamma_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_gamma_sweep.png")

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
            f"RC power weighting (F1+b)^γ. Best γ={best_gamma}  b={BASE_SHIFT}. "
            f"{weight_str}. "
            f"GEL γ={GEL_GAMMA} reference: val F1={gel_row[2]:.4f}. "
            "RC formula only — no OAM, no BVG."
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
