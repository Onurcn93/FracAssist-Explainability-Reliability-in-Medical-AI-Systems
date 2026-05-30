"""
EA-B5: Diversity-penalised weighting.

Extends the accuracy × diversity principle from Brown et al. (2005):
accurate models that are highly correlated with peers are penalised,
since their predictions are redundant in the ensemble.

Step 1 — Per-model val F1 (same protocol as EA-B1).
Step 2 — Pairwise Pearson correlation between model probability outputs
          on the train split. Frozen — no leakage to val/test.
Step 3 — For each model i:
          avg_corr_i = mean(|ρ_ij|) for j ≠ i
Step 4 — Combined raw score:
          raw_i = F1_i × (1 - λ × avg_corr_i)
          w_i   = raw_i / Σ raw_j
Step 5 — λ swept on val; λ=0 recovers B1 (F1-weighted, no diversity penalty).

score = Σ w_i × p_i  (static weighted sum)

The correlation matrix is a key diagnostic: if all pairwise correlations are
high and similar, avg_corr_i will be uniform across models and the penalty
redistributes nothing — the null result for B5 is structurally meaningful.

Reference: Brown, Wyatt, Harris & Yao (2005). Diversity creation methods:
           a survey and categorisation. Information Fusion, 6(1):5-20.
           Kuncheva & Whitaker (2003). Measures of diversity in classifier
           ensembles and their relationship with the ensemble accuracy.
           Machine Learning, 51(2):181-207.
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

EA_ID       = "EA-B5"
METHOD      = "Diversity-penalised weighting"
FAMILY      = "B"
TYPE        = "ORIG"
REF         = "Brown et al. 2005; Kuncheva & Whitaker 2003"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def _get_per_model_f1(df: pd.DataFrame, logger) -> np.ndarray:
    """Per-model val F1 via threshold sweep — same protocol as B1."""
    val_mask = df["split"].values == "val"
    labels   = (df["true_label"] == "Fractured").astype(int).values

    f1_scores = []
    for col, name in zip(PROB_COLS, MODEL_NAMES):
        sweep = threshold_sweep(labels[val_mask], df[col].values[val_mask])
        f1_scores.append(sweep["f1"])
        logger.info(f"  {name}: val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}")
    return np.array(f1_scores)


def _compute_train_correlation(df: pd.DataFrame, logger) -> tuple[np.ndarray, np.ndarray]:
    """
    Pairwise Pearson correlation between model outputs on train split.
    Returns corr_matrix (3, 3) and avg_corr (3,) — mean abs off-diagonal per model.
    """
    train_mask  = df["split"].values == "train"
    probs_train = df[PROB_COLS].values[train_mask]   # (N_train, 3)

    corr_matrix = np.corrcoef(probs_train.T)         # (3, 3)

    logger.info("Train pairwise Pearson correlation:")
    for i, ni in enumerate(MODEL_NAMES):
        for j, nj in enumerate(MODEL_NAMES):
            if j > i:
                logger.info(f"  ρ({ni}, {nj}) = {corr_matrix[i, j]:.4f}")

    n = len(MODEL_NAMES)
    avg_corr = np.array([
        np.mean([np.abs(corr_matrix[i, j]) for j in range(n) if j != i])
        for i in range(n)
    ])
    for name, ac in zip(MODEL_NAMES, avg_corr):
        logger.info(f"  avg |corr| {name}: {ac:.4f}  diversity={1-ac:.4f}")

    return corr_matrix, avg_corr


def _diversity_weights(f1_scores: np.ndarray, avg_corr: np.ndarray,
                       lam: float) -> np.ndarray:
    """w_i ∝ F1_i × (1 - λ × avg_corr_i), clipped and normalised."""
    raw = f1_scores * (1.0 - lam * avg_corr)
    raw = raw.clip(min=1e-9)
    return raw / raw.sum()


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
    probs     = df[PROB_COLS].values   # (N, 3)

    # --- Per-model val F1 ---
    logger.info("Computing per-model val F1:")
    f1_scores = _get_per_model_f1(df, logger)

    # --- Pairwise correlation on train ---
    logger.info("Computing pairwise train correlation:")
    corr_matrix, avg_corr = _compute_train_correlation(df, logger)

    # --- λ sweep on val ---
    logger.info(f"Sweeping λ over {LAMBDA_GRID}:")
    best_lam, best_val_f1 = None, -1.0
    lam_log = []
    for lam in LAMBDA_GRID:
        weights = _diversity_weights(f1_scores, avg_corr, lam)
        scores  = probs @ weights
        sweep   = threshold_sweep(labels[val_mask], scores[val_mask])
        lam_log.append((lam, weights.copy(), sweep["f1"], sweep["threshold"]))
        b1_marker = "  <-- B1 (λ=0)" if lam == 0.0 else ""
        logger.info(
            f"  λ={lam:.2f}  w=[{weights[0]:.3f},{weights[1]:.3f},{weights[2]:.3f}]"
            f"  val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}{b1_marker}"
        )
        if sweep["f1"] > best_val_f1:
            best_val_f1, best_lam = sweep["f1"], lam

    logger.info(f"Best λ={best_lam}  val F1={best_val_f1:.4f}")

    # --- Evaluate with best λ ---
    best_weights = _diversity_weights(f1_scores, avg_corr, best_lam)
    scores = probs @ best_weights

    logger.info("--- Val ---")
    val_m  = evaluate(labels[val_mask],  scores[val_mask],  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(labels[test_mask], scores[test_mask], "test", logger)

    weight_str = "  ".join(f"{n}={w:.4f}" for n, w in zip(MODEL_NAMES, best_weights))
    corr_str   = (f"ρ(Res,Dense)={corr_matrix[0,1]:.3f}  "
                  f"ρ(Res,Eff)={corr_matrix[0,2]:.3f}  "
                  f"ρ(Dense,Eff)={corr_matrix[1,2]:.3f}")
    logger.info(f"Best λ={best_lam} weights: {weight_str}")
    logger.info(f"Train correlations: {corr_str}")

    # B1 reference row (λ=0)
    b1_row = lam_log[0]
    logger.info(
        f"B1 reference (λ=0): val F1={b1_row[2]:.4f}  "
        f"w=[{b1_row[1][0]:.3f},{b1_row[1][1]:.3f},{b1_row[1][2]:.3f}]"
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
        ax.set_xlabel("Diversity-penalised ensemble score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}  λ={best_lam}\n"
            f"{weight_str}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        # Plot 2: λ sweep curve
        lams   = [r[0] for r in lam_log]
        f1vals = [r[2] for r in lam_log]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lams, f1vals, "o-", color="steelblue", linewidth=2, markersize=6)
        ax.axvline(best_lam, color="tomato", linestyle="--",
                   label=f"Best λ={best_lam}  (F1={best_val_f1:.4f})")
        ax.axvline(0.0, color="grey", linestyle=":",
                   label=f"B1 reference (λ=0, F1={lam_log[0][2]:.4f})")
        ax.set_xlabel("λ (diversity penalty strength)")
        ax.set_ylabel("Val F1 (threshold-swept)")
        ax.set_title(
            f"{EA_ID}: λ sweep — diversity penalty\n"
            f"{corr_str}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_lambda_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_lambda_sweep.png")

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
            f"Diversity-penalised: F1_i x (1 - λ x avg_corr_i). "
            f"Best λ={best_lam}. {weight_str}. "
            f"Train correlations: {corr_str}."
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
