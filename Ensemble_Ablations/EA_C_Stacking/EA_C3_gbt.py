"""
EA-C3: Gradient-boosted trees meta-learner (stacking).

Shallow GBT on the 3 base model probabilities (max_depth <= 3,
subsample=0.8 for stability on small training sets).

Two-step protocol:
  1. Grid search — sweep max_depth x learning_rate on val F1.
     subsample=0.8 fixed throughout; n_estimators=N_EST_CURVE for selection.
  2. Staged optimisation — refit best config with N_EST_CURVE trees;
     staged_predict_proba gives train+val log-loss at each boosting round.
     Optimal n_trees chosen at min val log-loss (analogous to C2 epoch stopping).
     Final model refit with best_n_trees only.

Frozen-model bias makes CV-within-train unreliable for meta-learner
hyperparameter selection — val set used throughout (same as C1/C2).

Reference: Friedman, J.H. (2001). Greedy function approximation: a
           gradient boosting machine. Annals of Statistics, 29(5):1189-1232.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy,
                         plot_training_curve)

EA_ID       = "EA-C3"
METHOD      = "Gradient-boosted trees meta-learner"
FAMILY      = "C"
TYPE        = "LIT"
REF         = "Friedman 2001 (Annals of Statistics)"

DATA_CSV    = "data/all_base.csv"
FEAT_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]

DEPTH_GRID  = [1, 2, 3]
LR_GRID     = [0.05, 0.1, 0.2]
SUBSAMPLE   = 0.8
N_EST_CURVE = 300        # trees for staged curve and grid search selection


def _grid_search(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    logger, seed: int,
) -> tuple[int, float, dict]:
    """Sweep max_depth x learning_rate on val F1. Returns (best_depth, best_lr, grid)."""
    best_depth, best_lr, best_f1 = DEPTH_GRID[0], LR_GRID[0], -1.0
    grid = {}
    for depth in DEPTH_GRID:
        for lr in LR_GRID:
            clf = GradientBoostingClassifier(
                n_estimators=N_EST_CURVE,
                max_depth=depth,
                learning_rate=lr,
                subsample=SUBSAMPLE,
                random_state=seed,
            )
            clf.fit(X_train, y_train)
            val_scores = clf.predict_proba(X_val)[:, 1]
            sweep = threshold_sweep(y_val, val_scores)
            grid[(depth, lr)] = sweep["f1"]
            logger.info(
                f"  depth={depth}  lr={lr:.2f}  "
                f"val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}"
            )
            if sweep["f1"] > best_f1:
                best_f1, best_depth, best_lr = sweep["f1"], depth, lr
    logger.info(f"Best: depth={best_depth}  lr={best_lr}  val F1={best_f1:.4f}")
    return best_depth, best_lr, grid


def _staged_curve(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    depth: int, lr: float, seed: int,
) -> tuple:
    """Fit N_EST_CURVE trees; compute staged train+val log-loss per boosting round.

    Returns (clf, train_losses, val_losses, best_n_trees).
    best_n_trees is 1-indexed (tree count at min val log-loss).
    """
    clf = GradientBoostingClassifier(
        n_estimators=N_EST_CURVE,
        max_depth=depth,
        learning_rate=lr,
        subsample=SUBSAMPLE,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    train_losses, val_losses = [], []
    for pred in clf.staged_predict_proba(X_train):
        train_losses.append(log_loss(y_train, pred))
    for pred in clf.staged_predict_proba(X_val):
        val_losses.append(log_loss(y_val, pred))

    best_n_trees = int(np.argmin(val_losses)) + 1   # convert 0-idx to tree count
    return clf, train_losses, val_losses, best_n_trees


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

    # --- Step 1: grid search on val F1 ---
    logger.info(f"Grid search: depth={DEPTH_GRID}  lr={LR_GRID}  subsample={SUBSAMPLE}:")
    best_depth, best_lr, grid = _grid_search(X_train, y_train, X_val, y_val, logger, seed)

    # --- Step 2: staged curve to find optimal n_trees ---
    logger.info(f"Staged curve: depth={best_depth}  lr={best_lr}  n_est={N_EST_CURVE}:")
    _, train_losses, val_losses, best_n_trees = _staged_curve(
        X_train, y_train, X_val, y_val, best_depth, best_lr, seed
    )
    logger.info(
        f"  Best n_trees={best_n_trees}/{N_EST_CURVE}  "
        f"train_ll={train_losses[best_n_trees-1]:.4f}  "
        f"val_ll={val_losses[best_n_trees-1]:.4f}"
    )

    # --- Step 3: final refit with best_n_trees ---
    logger.info(f"Final refit: n_trees={best_n_trees}")
    clf = GradientBoostingClassifier(
        n_estimators=best_n_trees,
        max_depth=best_depth,
        learning_rate=best_lr,
        subsample=SUBSAMPLE,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
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
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(val_scores[val_labels == 0], bins=30, alpha=0.6,
                label="Non-fractured", color="steelblue")
        ax.hist(val_scores[val_labels == 1], bins=30, alpha=0.6,
                label="Fractured", color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("GBT meta-learner score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}\n"
            f"depth={best_depth}  lr={best_lr}  n_trees={best_n_trees}  subsample={SUBSAMPLE}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        # Plot 2: grid search heatmap (depth x lr, val F1)
        f1_matrix = np.array(
            [[grid[(d, lr)] for lr in LR_GRID] for d in DEPTH_GRID]
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            f1_matrix,
            annot=True, fmt=".4f", cmap="YlGnBu",
            xticklabels=[str(lr) for lr in LR_GRID],
            yticklabels=[str(d) for d in DEPTH_GRID],
            linewidths=0.5, ax=ax,
            annot_kws={"size": 10},
        )
        best_row = DEPTH_GRID.index(best_depth)
        best_col = LR_GRID.index(best_lr)
        ax.add_patch(plt.Rectangle(
            (best_col, best_row), 1, 1,
            fill=False, edgecolor="tomato", lw=2.5,
        ))
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("max_depth")
        ax.set_title(
            f"{EA_ID}: Val F1 grid (depth × learning_rate)\n"
            f"subsample={SUBSAMPLE}  n_est={N_EST_CURVE}  "
            f"Best: depth={best_depth}  lr={best_lr}  F1={grid[(best_depth, best_lr)]:.4f}"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_grid_search.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_grid_search.png")

        # Plot 3: staged training curve (train+val log-loss vs boosting rounds)
        plot_training_curve(
            ea_id=EA_ID, method=METHOD, plot_dir=plot_dir, logger=logger,
            train_series=train_losses,
            val_series=val_losses,
            x_label="Number of trees (boosting rounds)",
            train_label="Train log-loss",
            val_label="Val log-loss (actual val set)",
            best_idx=best_n_trees - 1,
            dual_axis=False,
        )

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
            f"Grid on val F1: depth={best_depth}  lr={best_lr}  subsample={SUBSAMPLE}. "
            f"Staged curve: best_n_trees={best_n_trees}/{N_EST_CURVE}. "
            f"Final refit with best_n_trees. "
            f"Train in-sample F1={train_m['f1']:.4f}. "
            f"Train-val gap={train_m['f1']-val_m['f1']:.4f}."
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
