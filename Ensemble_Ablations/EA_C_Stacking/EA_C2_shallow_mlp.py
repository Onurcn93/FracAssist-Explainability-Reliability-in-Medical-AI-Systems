"""
EA-C2: Shallow MLP meta-learner (stacking).

Single hidden layer MLP on the 3 base model probabilities.
Architecture: 3 -> n_hidden -> 1 (sigmoid output).

Capacity is controlled by:
  - n_hidden: hidden units, swept in [4, 8, 16]
  - alpha: L2 weight decay, swept in [0.001, 0.01, 0.1, 1.0]
  - early_stopping: sklearn reserves 10% of train internally for loss-based stopping

sklearn's MLPClassifier does not support dropout; L2 + early stopping
together serve the same capacity-limiting role. Compared to C1 this
isolates the effect of the hidden-layer non-linearity on raw prob inputs.

Both hyperparameters swept on val F1 (fit on train, score val).
Frozen-model bias makes CV-within-train unreliable — see EA-C1 notes.

Reference: Muller, A., Gutjahr, G., & Casalicchio, G. (2022).
           Superblend: A flexible approach to stacking with
           arbitrary classifiers. arXiv:2202.09015.
"""

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy,
                         plot_training_curve)

EA_ID       = "EA-C2"
METHOD      = "Shallow MLP meta-learner"
FAMILY      = "C"
TYPE        = "LIT"
REF         = "Muller et al. 2022 (arXiv:2202.09015)"

DATA_CSV    = "data/all_base.csv"
FEAT_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

HIDDEN_GRID = [4, 8, 16]
ALPHA_GRID  = [0.001, 0.01, 0.1, 1.0]


def _grid_search(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 logger, seed: int) -> tuple[int, float, dict]:
    """Sweep n_hidden x alpha on val F1; return (best_n_hidden, best_alpha, grid_results)."""
    best_nh, best_alpha, best_f1 = HIDDEN_GRID[0], ALPHA_GRID[0], -1.0
    grid = {}

    for n_h in HIDDEN_GRID:
        for alpha in ALPHA_GRID:
            clf = MLPClassifier(
                hidden_layer_sizes=(n_h,),
                activation="relu",
                alpha=alpha,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=15,
                validation_fraction=0.1,
                random_state=seed,
                solver="adam",
            )
            clf.fit(X_train, y_train)
            val_scores = clf.predict_proba(X_val)[:, 1]
            sweep = threshold_sweep(y_val, val_scores)
            grid[(n_h, alpha)] = sweep["f1"]
            logger.info(
                f"  n_hidden={n_h:2d}  alpha={alpha:.3f}  "
                f"val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}  "
                f"iter={clf.n_iter_}"
            )
            if sweep["f1"] > best_f1:
                best_f1, best_nh, best_alpha = sweep["f1"], n_h, alpha

    logger.info(f"Best: n_hidden={best_nh}  alpha={best_alpha}  val F1={best_f1:.4f}")
    return best_nh, best_alpha, grid


def _fit_with_curve(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_hidden: int, alpha: float, seed: int,
    max_epochs: int = 300, patience: int = 15,
) -> tuple:
    """Refit MLP one epoch at a time, tracking train + val log-loss on actual val set.

    Uses warm_start=True so weights persist across calls. Saves and restores
    weights at the best val-loss epoch. Returns (fitted_clf, train_losses,
    val_losses, best_epoch).
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(n_hidden,),
        activation="relu",
        alpha=alpha,
        max_iter=1,
        warm_start=True,
        random_state=seed,
        solver="adam",
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0
    best_weights  = None

    for epoch in range(max_epochs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress ConvergenceWarning from max_iter=1
            clf.fit(X_train, y_train)

        train_ll = log_loss(y_train, clf.predict_proba(X_train))
        val_ll   = log_loss(y_val,   clf.predict_proba(X_val))
        train_losses.append(train_ll)
        val_losses.append(val_ll)

        if val_ll < best_val_loss - 1e-5:
            best_val_loss = val_ll
            best_epoch    = epoch
            no_improve    = 0
            best_weights  = (
                copy.deepcopy(clf.coefs_),
                copy.deepcopy(clf.intercepts_),
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Restore weights from best epoch
    if best_weights is not None:
        clf.coefs_      = best_weights[0]
        clf.intercepts_ = best_weights[1]

    return clf, train_losses, val_losses, best_epoch


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

    # --- Grid search on val F1 ---
    logger.info(f"Grid search: n_hidden={HIDDEN_GRID}  alpha={ALPHA_GRID}:")
    best_nh, best_alpha, grid = _grid_search(X_train, y_train, X_val, y_val, logger, seed)

    # --- Refit with best config — epoch loop for train/val log-loss curve ---
    logger.info(f"Final fit: n_hidden={best_nh}  alpha={best_alpha} (epoch-by-epoch):")
    clf, train_losses, val_losses, best_epoch = _fit_with_curve(
        X_train, y_train, X_val, y_val, best_nh, best_alpha, seed
    )
    logger.info(
        f"  Stopped at epoch {len(train_losses)}  "
        f"best val loss at epoch {best_epoch}  "
        f"(train_ll={train_losses[best_epoch]:.4f}  val_ll={val_losses[best_epoch]:.4f})"
    )

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
        ax.set_xlabel("MLP meta-learner score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}\n"
            f"n_hidden={best_nh}  alpha={best_alpha}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        # Plot 2: heatmap of val F1 across n_hidden x alpha grid
        f1_matrix = np.array(
            [[grid[(n_h, alpha)] for alpha in ALPHA_GRID] for n_h in HIDDEN_GRID]
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            f1_matrix,
            annot=True, fmt=".4f", cmap="YlGnBu",
            xticklabels=[str(a) for a in ALPHA_GRID],
            yticklabels=[str(n) for n in HIDDEN_GRID],
            linewidths=0.5, ax=ax,
            annot_kws={"size": 10},
        )
        # Mark best cell
        best_row = HIDDEN_GRID.index(best_nh)
        best_col = ALPHA_GRID.index(best_alpha)
        ax.add_patch(plt.Rectangle(
            (best_col, best_row), 1, 1,
            fill=False, edgecolor="tomato", lw=2.5
        ))
        ax.set_xlabel("alpha (L2 weight decay)")
        ax.set_ylabel("n_hidden")
        ax.set_title(
            f"{EA_ID}: Val F1 grid (n_hidden x alpha)\n"
            f"Best: n_hidden={best_nh}  alpha={best_alpha}  F1={grid[(best_nh, best_alpha)]:.4f}"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_grid_search.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_grid_search.png")

        # Plot 5: training curve — train log-loss + val log-loss on actual val set
        plot_training_curve(
            ea_id=EA_ID, method=METHOD, plot_dir=plot_dir, logger=logger,
            train_series=train_losses,
            val_series=val_losses,
            x_label="Epoch",
            train_label="Train log-loss",
            val_label="Val log-loss (actual val set)",
            best_idx=best_epoch,
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
            f"Grid search on val F1. Best n_hidden={best_nh}  alpha={best_alpha}. "
            f"L2+early_stopping (no dropout in sklearn). "
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
