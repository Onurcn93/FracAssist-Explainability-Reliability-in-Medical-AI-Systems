"""
EA-D3: Disagreement-triggered gating (ORIG).

For each sample, compute the inter-model disagreement σ = std(p_1, p_2, p_3).
Gate the ensemble combination based on σ:

  score(x) =  mean(p_1, p_2, p_3)          if σ(x) < δ  [agree branch]
              LogReg(p_1, p_2, p_3)          if σ(x) ≥ δ  [disagree branch]

When all models largely agree (low σ), the simple mean is used — no
information is lost by weighting. When models disagree (high σ),
a learned LogReg combiner (identical to EA-C1, C=0.001, fit on train)
is applied — it can leverage DenseNet's learned dominance to resolve
the conflict.

The gating threshold δ is swept on val F1 (no test leakage). The
LogReg combiner is fit once on all train regardless of δ — its quality
is held constant while only the gating boundary is optimised.

Boundary cases:
  δ = 0:   all σ ≥ 0 → all samples use LogReg → recovers EA-C1 (val F1=0.7368)
  δ → ∞:  all σ < δ → all samples use mean   → recovers EA-A1 (val F1=0.7383)

Reference: Extends GEL OAM stage (this thesis).
           Cruz et al. (2018) Dynamic classifier selection;
           Information Fusion, 41:195–216.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)

EA_ID       = "EA-D3"
METHOD      = "Disagreement-triggered gating"
FAMILY      = "D"
TYPE        = "ORIG"
REF         = "Extends GEL OAM (this thesis); Cruz et al. 2018"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

# δ grid: 0.0 recovers C1 (all-disagree); large δ recovers A1 (all-agree)
DELTA_GRID = np.concatenate([[0.0], np.arange(0.005, 0.201, 0.005)])

# LogReg combiner hyperparameter (same as EA-C1 best C)
LOGREG_C = 0.001


def _compute_scores(probs_val: np.ndarray, sigma_val: np.ndarray,
                    delta: float, clf, mean_scores_val: np.ndarray) -> np.ndarray:
    """Compute gated ensemble scores for a given delta threshold."""
    agree_mask    = sigma_val < delta
    disagree_mask = ~agree_mask
    scores = np.empty(len(probs_val))
    scores[agree_mask]    = mean_scores_val[agree_mask]
    if disagree_mask.any():
        scores[disagree_mask] = clf.predict_proba(probs_val[disagree_mask])[:, 1]
    return scores


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    df     = pd.read_csv(root_dir / DATA_CSV)
    labels = (df["true_label"] == "Fractured").astype(int).values
    probs  = df[PROB_COLS].values                              # (N, 3)

    train_mask = df["split"].values == "train"
    val_mask   = df["split"].values == "val"
    test_mask  = df["split"].values == "test"

    X_train = probs[train_mask]
    y_train = labels[train_mask]
    X_val   = probs[val_mask]
    y_val   = labels[val_mask]
    X_test  = probs[test_mask]
    y_test  = labels[test_mask]

    logger.info(f"Loaded {len(df)} rows | Train={train_mask.sum()} | "
                f"Val={val_mask.sum()} | Test={test_mask.sum()}")

    # --- Fit LogReg combiner once on train ---
    logger.info(f"Fitting LogReg combiner (C={LOGREG_C}) on train...")
    clf = LogisticRegression(C=LOGREG_C, max_iter=1000, solver="lbfgs", random_state=seed)
    clf.fit(X_train, y_train)
    coefs = clf.coef_[0]
    logger.info("LogReg coefficients: " +
                "  ".join(f"{n}={w:.4f}" for n, w in zip(MODEL_NAMES, coefs)) +
                f"  bias={clf.intercept_[0]:.4f}")

    # --- Precompute disagreement (std of 3 probs) ---
    sigma = probs.std(axis=1)                                  # (N,)
    sigma_val  = sigma[val_mask]
    sigma_test = sigma[test_mask]

    logger.info(f"Val σ stats: mean={sigma_val.mean():.4f}  "
                f"q33={np.percentile(sigma_val,33):.4f}  "
                f"q67={np.percentile(sigma_val,67):.4f}  "
                f"max={sigma_val.max():.4f}")

    # Mean scores (agree branch, precomputed)
    mean_val  = X_val.mean(axis=1)
    mean_test = X_test.mean(axis=1)

    # --- δ sweep on val ---
    logger.info("Sweeping δ on val F1...")
    best_delta, best_val_f1 = 0.0, -1.0
    delta_log = []

    for delta in DELTA_GRID:
        scores_val = _compute_scores(X_val, sigma_val, delta, clf, mean_val)
        sweep      = threshold_sweep(y_val, scores_val)
        n_agree    = (sigma_val < delta).sum()
        n_disagree = (sigma_val >= delta).sum()
        delta_log.append({
            "delta":     delta,
            "val_f1":    sweep["f1"],
            "val_thr":   sweep["threshold"],
            "n_agree":   n_agree,
            "n_disagree":n_disagree,
        })
        if sweep["f1"] > best_val_f1:
            best_val_f1, best_delta = sweep["f1"], delta

    logger.info(f"Best δ={best_delta:.3f}  (val F1={best_val_f1:.4f})")

    best_entry = next(d for d in delta_log if d["delta"] == best_delta)
    pct_agree  = best_entry["n_agree"]    / val_mask.sum() * 100
    pct_disag  = best_entry["n_disagree"] / val_mask.sum() * 100
    logger.info(f"  At best δ: agree={best_entry['n_agree']} ({pct_agree:.1f}%)  "
                f"disagree={best_entry['n_disagree']} ({pct_disag:.1f}%)")

    # --- Final evaluation at best δ ---
    scores_val_final  = _compute_scores(X_val,  sigma_val,  best_delta, clf, mean_val)
    scores_test_final = _compute_scores(X_test, sigma_test, best_delta, clf, mean_test)

    logger.info("--- Val (best δ) ---")
    val_m  = evaluate(y_val,  scores_val_final,  "val",  logger)
    logger.info("--- Test (best δ, val threshold transferred) ---")
    test_m = evaluate(y_test, scores_test_final, "test", logger)

    # Reference: delta=0 should match C1; delta=max should match A1
    ref0    = threshold_sweep(y_val, _compute_scores(X_val, sigma_val, 0.0, clf, mean_val))
    ref_inf = threshold_sweep(y_val, mean_val)
    logger.info(f"Reference — δ=0 (all LogReg): val F1={ref0['f1']:.4f}  "
                f"[expected ≈ EA-C1: 0.7368]")
    logger.info(f"Reference — δ=∞ (all Mean): val F1={ref_inf['f1']:.4f}  "
                f"[expected ≈ EA-A1: 0.7383]")

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: δ sweep — val F1 vs δ
        deltas = [d["delta"] for d in delta_log]
        f1s    = [d["val_f1"] for d in delta_log]
        agrees = [d["n_agree"] / val_mask.sum() * 100 for d in delta_log]

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(deltas, f1s, "-o", color="steelblue", linewidth=2, markersize=4,
                 label="Val F1 (threshold-swept)")
        ax1.axvline(best_delta, color="tomato", linestyle="--",
                    label=f"Best δ={best_delta:.3f}  (F1={best_val_f1:.4f})")
        ax1.axhline(ref0["f1"], color="gray", linestyle=":", linewidth=1.2,
                    alpha=0.8, label=f"δ=0 (all LogReg, A1-analog F1={ref0['f1']:.4f})")
        ax1.axhline(ref_inf["f1"], color="dimgray", linestyle="-.", linewidth=1.2,
                    alpha=0.8, label=f"δ=∞ (all Mean, EA-A1 F1={ref_inf['f1']:.4f})")
        ax1.set_xlabel("δ (disagreement threshold)")
        ax1.set_ylabel("Val F1 (threshold-swept)", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.legend(fontsize=8, loc="lower right")

        ax2 = ax1.twinx()
        ax2.fill_between(deltas, agrees, alpha=0.12, color="forestgreen")
        ax2.plot(deltas, agrees, "--", color="forestgreen", linewidth=1.2, alpha=0.7,
                 label="% samples → agree branch")
        ax2.set_ylabel("% samples in agree branch (mean)", color="forestgreen")
        ax2.tick_params(axis="y", labelcolor="forestgreen")
        ax2.set_ylim(0, 110)
        ax2.legend(fontsize=8, loc="upper left")

        ax1.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"δ sweep | best δ={best_delta:.3f} ({pct_agree:.1f}% agree / "
            f"{pct_disag:.1f}% disagree)"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_delta_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_delta_sweep.png")

        # Plot 2: σ distribution on val with best δ marked
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sigma_val[y_val == 0], bins=30, alpha=0.6, color="steelblue",
                label="Non-fractured", density=True)
        ax.hist(sigma_val[y_val == 1], bins=30, alpha=0.6, color="tomato",
                label="Fractured", density=True)
        ax.axvline(best_delta, color="black", linestyle="--",
                   label=f"Best δ={best_delta:.3f}")
        ax.set_xlabel("σ_p = std(p_resnet, p_densenet, p_efficientnet)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{EA_ID}: Inter-model disagreement distribution (val)\n"
            f"Left of δ → mean branch  |  Right of δ → LogReg branch"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_sigma_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_sigma_dist.png")

        # Plot 3: score distribution at best δ
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scores_val_final[y_val == 0], bins=30, alpha=0.6, color="steelblue",
                label="Non-fractured")
        ax.hist(scores_val_final[y_val == 1], bins=30, alpha=0.6, color="tomato",
                label="Fractured")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("Gated ensemble score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}  δ={best_delta:.3f}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        plot_confusion_matrix(y_val, scores_val_final, val_m["threshold"],
                              EA_ID, METHOD, plot_dir, logger)
        plot_confidence_accuracy(y_val, scores_val_final, val_m["threshold"],
                                 EA_ID, METHOD, plot_dir, logger)

    # --- Append to comparative table ---
    coef_str = "  ".join(f"{n}={w:.4f}" for n, w in zip(MODEL_NAMES, coefs))
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
            f"LogReg combiner C={LOGREG_C} fit on train ({coef_str}). "
            f"Best δ={best_delta:.3f}: "
            f"agree={pct_agree:.1f}% (mean) / disagree={pct_disag:.1f}% (LogReg). "
            f"δ=0 recovers EA-C1; δ→∞ recovers EA-A1."
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
