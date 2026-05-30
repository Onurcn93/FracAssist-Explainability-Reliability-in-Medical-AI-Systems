"""
EA-C1: Logistic regression meta-learner (stacking).

Fits a 3-parameter logistic regression on the training split
probabilities (3 model outputs) with L2 regularisation.

score(x) = sigmoid(w_res*p_res + w_dense*p_dense + w_eff*p_eff + bias)

Two-step protocol:
  1. C selection — sweep on val F1 using lbfgs (fast, full convergence
     per C value). CV within train is biased when base models are frozen
     (in-sample predictions are overconfident), so val is the unbiased probe.
  2. Final fit — saga solver with warm_start=True, one epoch at a time.
     Tracks train + val log-loss per epoch on the actual val set.
     Weights restored to the best-val-loss epoch (same protocol as C2).

Reference: Wolpert, D.H. (1992). Stacked generalisation.
           Neural Networks, 5(2):241-259.
           Ting, K.M. & Witten, I.H. (1999). Issues in stacked
           generalisation. JAIR, 10:271-289.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy,
                         plot_training_curve)

EA_ID       = "EA-C1"
METHOD      = "Logistic regression meta-learner"
FAMILY      = "C"
TYPE        = "LIT"
REF         = "Wolpert 1992; Ting & Witten 1999"

DATA_CSV    = "data/all_base.csv"
FEAT_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def _val_select_c(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray, logger) -> tuple[float, list]:
    """Sweep C on val F1 using lbfgs (fully converged per C, fast for selection)."""
    best_c, best_val_f1 = C_GRID[0], -1.0
    c_log = []
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        val_ll     = log_loss(y_val, clf.predict_proba(X_val))
        train_ll   = log_loss(y_train, clf.predict_proba(X_train))
        sweep      = threshold_sweep(y_val, val_scores)
        c_log.append((c, sweep["f1"], sweep["threshold"], train_ll, val_ll))
        logger.info(
            f"  C={c:.3f}  val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}"
            f"  train_ll={train_ll:.4f}  val_ll={val_ll:.4f}"
        )
        if sweep["f1"] > best_val_f1:
            best_val_f1, best_c = sweep["f1"], c
    logger.info(f"Best C={best_c}  (val F1={best_val_f1:.4f})")
    return best_c, c_log



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
    logger.info(f"Train fracture rate: {y_train.mean():.3f}")

    # --- Step 1: C selection via val F1 sweep (lbfgs) ---
    logger.info(f"Sweeping C on val F1 from {C_GRID}:")
    best_c, c_log = _val_select_c(X_train, y_train, X_val, y_val, logger)

    # --- Step 2: final fit on full train with best C (lbfgs, fully converged) ---
    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(X_train, y_train)

    coefs = clf.coef_[0]
    bias  = clf.intercept_[0]
    logger.info("Learned coefficients (lbfgs, fully converged):")
    for name, w in zip(MODEL_NAMES, coefs):
        logger.info(f"  {name}: {w:.4f}")
    logger.info(f"  bias: {bias:.4f}")

    scores = clf.predict_proba(X)[:, 1]

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
        coef_str   = "  ".join(f"{n}={w:.3f}" for n, w in zip(MODEL_NAMES, coefs))
        score_min  = float(val_scores.min())
        score_max  = float(val_scores.max())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(val_scores[val_labels == 0], bins=30, alpha=0.6,
                label="Non-fractured", color="steelblue")
        ax.hist(val_scores[val_labels == 1], bins=30, alpha=0.6,
                label="Fractured", color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("LogReg meta-learner score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}  C={best_c}\n"
            f"{coef_str}  bias={bias:.3f}  "
            f"score range [{score_min:.2f}, {score_max:.2f}]"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        # Plot 2: C sweep — val F1 vs C (hyperparameter selection)
        cs  = [r[0] for r in c_log]
        f1s = [r[1] for r in c_log]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(cs, f1s, "o-", color="steelblue", linewidth=2, markersize=6,
                label="Val F1 (threshold-swept)")
        ax.axvline(best_c, color="tomato", linestyle="--",
                   label=f"Best C={best_c}  (F1={max(f1s):.4f})")
        ax.set_xscale("log")
        ax.set_xlabel("C (inverse L2 regularisation strength)")
        ax.set_ylabel("Val F1 (threshold-swept)")
        ax.set_title(
            f"{EA_ID}: C selection sweep\n"
            f"Fit on train, scored on val — frozen-model bias makes CV-within-train unreliable"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_c_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_c_sweep.png")

        # Plot 3: regularisation path — train log-loss + val log-loss vs C
        # LogReg is a convex optimizer with no epoch dynamics; the regularisation
        # path is its equivalent training diagnostic (complexity vs loss).
        train_lls = [r[3] for r in c_log]
        val_lls   = [r[4] for r in c_log]
        cs_plot   = [r[0] for r in c_log]
        plot_training_curve(
            ea_id=EA_ID, method=METHOD, plot_dir=plot_dir, logger=logger,
            train_series=train_lls, val_series=val_lls,
            x_label="C (inverse L2 strength) — higher C = less regularisation",
            train_label="Train log-loss",
            val_label="Val log-loss (actual val set)",
            best_idx=cs_plot.index(best_c),
            dual_axis=False,
            x_values=cs_plot,
            x_log=True,
        )

        plot_confusion_matrix(labels[val_mask], scores[val_mask], val_m["threshold"],
                              EA_ID, METHOD, plot_dir, logger)
        plot_confidence_accuracy(labels[val_mask], scores[val_mask], val_m["threshold"],
                                 EA_ID, METHOD, plot_dir, logger)

    # --- Append to comparative table ---
    coef_str_long    = "  ".join(f"{n}={w:.4f}" for n, w in zip(MODEL_NAMES, coefs))
    best_val_f1_at_c = c_log[[r[0] for r in c_log].index(best_c)][1]
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
            f"C swept on val F1 (lbfgs); best C={best_c} (val F1={best_val_f1_at_c:.4f}). "
            f"Final fit: lbfgs fully converged. "
            f"Coefs: {coef_str_long}. bias={bias:.4f}. "
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
