"""
EA-D4: Region-of-competence weighting (ORIG).

For each test/val sample, the k nearest neighbours in the training set
(measured by Euclidean distance in the 3-dimensional probability space)
define its local region of competence. Each model's weight is its local
accuracy within that region:

  local_acc_i(x) = mean_{j ∈ kNN(x)} 1[pred_i(x_j) == y_j]
  w_i(x)         = (local_acc_i(x) + ε) / Σ_m (local_acc_m(x) + ε)
  score(x)        = Σ_i w_i(x) · p_i(x)

Per-model binary predictions on training neighbours use per-model
train-optimal thresholds (derived from train only — no leakage).
This correctly accounts for DenseNet's low optimal threshold (≈0.175).

k is swept on val F1. All kNN lookups are performed against the
training set only; training labels are used only to evaluate prediction
correctness in the local neighbourhood.

Reference: Extends Woloszynski, M. & Kurzynski, M. (2011).
           A measure of competence based on random classification
           for dynamic ensemble selection.
           Pattern Recognition, 44(10-11):2386–2396.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)

EA_ID       = "EA-D4"
METHOD      = "Region-of-competence weighting"
FAMILY      = "D"
TYPE        = "ORIG"
REF         = "Extends Woloszynski & Kurzynski 2011"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

K_GRID      = [5, 10, 15, 20, 30, 50]
EPSILON     = 1e-6                          # weight floor to avoid zero-division


def _weighted_ensemble(X_query: np.ndarray, X_train: np.ndarray,
                       correct_train: np.ndarray, nn_model, k: int) -> np.ndarray:
    """Compute region-of-competence ensemble scores for a query set.

    Args:
        X_query:       (N_q, 3) probabilities
        X_train:       (N_tr, 3) training probabilities
        correct_train: (N_tr, 3) binary — was model m correct on training sample j?
        nn_model:      fitted NearestNeighbors (n_neighbors >= k)
        k:             number of neighbours to use

    Returns:
        scores: (N_q,) ensemble score per sample
    """
    _, indices = nn_model.kneighbors(X_query)                  # (N_q, max_k)
    knn_correct = correct_train[indices[:, :k]]                # (N_q, k, 3)
    local_acc   = knn_correct.mean(axis=1)                     # (N_q, 3)

    weights = local_acc + EPSILON
    weights = weights / weights.sum(axis=1, keepdims=True)     # (N_q, 3)

    scores = (X_query * weights).sum(axis=1)                   # (N_q,)
    return scores, weights


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

    # --- Per-model train-optimal thresholds (no leakage) ---
    logger.info("Deriving per-model train-optimal thresholds (train only):")
    model_thresholds = np.zeros(3)
    for m in range(3):
        sweep_m = threshold_sweep(y_train, X_train[:, m])
        model_thresholds[m] = sweep_m["threshold"]
        logger.info(f"  {MODEL_NAMES[m]}: thr={sweep_m['threshold']:.3f}  "
                    f"train F1={sweep_m['f1']:.4f}")

    # --- Binary predictions on train using per-model thresholds ---
    model_preds_train = (X_train >= model_thresholds[None, :]).astype(int)  # (N_tr, 3)
    correct_train     = (model_preds_train == y_train[:, None]).astype(int) # (N_tr, 3)

    train_acc_per_model = correct_train.mean(axis=0)
    logger.info("Overall train accuracy at model-specific threshold:")
    for name, acc in zip(MODEL_NAMES, train_acc_per_model):
        logger.info(f"  {name}: {acc:.4f}")

    # --- Fit NearestNeighbors on training probs (fit once for max K) ---
    max_k = max(K_GRID)
    logger.info(f"Fitting NearestNeighbors (n_neighbors={max_k}) on train probs...")
    nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean", n_jobs=-1)
    nn.fit(X_train)

    # --- k sweep on val ---
    logger.info("Sweeping k on val F1:")
    best_k, best_val_f1 = K_GRID[0], -1.0
    k_log = []

    for k in K_GRID:
        scores_val, _ = _weighted_ensemble(X_val, X_train, correct_train, nn, k)
        sweep         = threshold_sweep(y_val, scores_val)
        k_log.append({"k": k, "val_f1": sweep["f1"], "val_auc": 0.0})
        logger.info(f"  k={k:3d}  val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}")
        if sweep["f1"] > best_val_f1:
            best_val_f1, best_k = sweep["f1"], k

    logger.info(f"Best k={best_k}  (val F1={best_val_f1:.4f})")

    # --- Final evaluation at best k ---
    scores_val_final, weights_val = _weighted_ensemble(
        X_val, X_train, correct_train, nn, best_k)
    scores_test_final, _          = _weighted_ensemble(
        X_test, X_train, correct_train, nn, best_k)

    logger.info("--- Val (best k) ---")
    val_m  = evaluate(y_val,  scores_val_final,  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(y_test, scores_test_final, "test", logger)

    # Mean weights and local accuracy stats (val, best k)
    mean_w = weights_val.mean(axis=0)
    logger.info("Mean val weights at best k:")
    for name, mw in zip(MODEL_NAMES, mean_w):
        logger.info(f"  {name}: {mw:.4f}")

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: k sweep — val F1 vs k
        ks  = [d["k"] for d in k_log]
        f1s = [d["val_f1"] for d in k_log]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, f1s, "o-", color="steelblue", linewidth=2, markersize=7)
        ax.axvline(best_k, color="tomato", linestyle="--",
                   label=f"Best k={best_k}  (F1={best_val_f1:.4f})")
        ax.set_xlabel("k (neighbourhood size)")
        ax.set_ylabel("Val F1 (threshold-swept)")
        ax.set_title(
            f"{EA_ID}: k-sweep on val\n"
            f"Local accuracy weights from k-NN in 3-prob space"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_k_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_k_sweep.png")

        # Plot 2: local accuracy distributions per model on val (best k)
        _, nn_all = nn.kneighbors(X_val)
        knn_correct_val = correct_train[nn_all[:, :best_k]]    # (N_val, k, 3)
        local_acc_val   = knn_correct_val.mean(axis=1)         # (N_val, 3)

        colours = ["steelblue", "tomato", "forestgreen"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for m, (name, ax) in enumerate(zip(MODEL_NAMES, axes)):
            ax.hist(local_acc_val[:, m], bins=15, color=colours[m],
                    alpha=0.75, edgecolor="white")
            ax.axvline(local_acc_val[:, m].mean(), color="black", linestyle="--",
                       label=f"Mean={local_acc_val[:,m].mean():.3f}")
            ax.set_xlabel("Local accuracy")
            ax.set_title(name)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Count (val samples)")
        fig.suptitle(
            f"{EA_ID}: Local accuracy distributions (val, k={best_k})\n"
            "Accuracy of each model on kNN neighbours in train"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_local_acc_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_local_acc_dist.png")

        # Plot 3: weight distributions per model (val, best k)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for m, (name, ax) in enumerate(zip(MODEL_NAMES, axes)):
            ax.hist(weights_val[:, m], bins=15, color=colours[m],
                    alpha=0.75, edgecolor="white")
            ax.axvline(weights_val[:, m].mean(), color="black", linestyle="--",
                       label=f"Mean={weights_val[:,m].mean():.3f}")
            ax.set_xlabel("Competence weight")
            ax.set_title(name)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Count (val samples)")
        fig.suptitle(
            f"{EA_ID}: Competence weight distributions (val, k={best_k})"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_weight_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_weight_dist.png")

        # Plot 4: score distribution on val
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scores_val_final[y_val == 0], bins=30, alpha=0.6,
                label="Non-fractured", color="steelblue")
        ax.hist(scores_val_final[y_val == 1], bins=30, alpha=0.6,
                label="Fractured", color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("Region-of-competence ensemble score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}  k={best_k}"
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
    weight_str = "  ".join(f"{n}={mw:.4f}" for n, mw in zip(MODEL_NAMES, mean_w))
    thr_str    = "  ".join(f"{n}={t:.3f}" for n, t in zip(MODEL_NAMES, model_thresholds))
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
            f"k={best_k} (swept on val F1). "
            f"Per-model train thresholds: {thr_str}. "
            f"Mean val weights (best k): {weight_str}."
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
