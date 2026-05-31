"""
EA-D2: META-DES-style meta-classifier (LIT).

For each base classifier i, a binary meta-classifier is trained to predict
whether classifier i will be correct on a new sample. At inference, the
meta-classifier outputs a competence score c_i(x) ∈ [0,1] for each model.
The ensemble score is a competence-weighted average of model probabilities:

  score(x) = Σ_i c_i(x) · p_i(x) / Σ_i c_i(x)

Meta-features for model i on sample x (5 features):
  1. p_i(x)            — model output probability
  2. |p_i(x) - 0.5|   — confidence margin
  3. local_acc_i(x)    — accuracy of model i on k_meta nearest train neighbours
  4. σ_p(x)            — std(p_1, p_2, p_3) — inter-model disagreement
  5. mean_p(x)         — mean(p_1, p_2, p_3) — consensus signal

Meta-labels for training: was_correct_i(x_j) = 1{pred_i(x_j) == y_j} for each
training sample j, using per-model train-optimal thresholds (train only, no leakage).

For meta-training, local accuracy is computed with k_meta nearest neighbours
excluding self (leave-one-out style) to avoid in-sample optimism bias.

Reference: Cruz, R.M.O., Sabourin, R., & Cavalcanti, G.D.C. (2015).
           META-DES: A dynamic ensemble selection framework using
           meta-learning. Pattern Recognition, 48(5):1925–1935.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)

EA_ID       = "EA-D2"
METHOD      = "META-DES-style meta-classifier"
FAMILY      = "D"
TYPE        = "LIT"
REF         = "Cruz et al. 2015"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

K_META          = 10    # k for meta-feature computation (fixed — not swept)
META_LOGREG_C   = 1.0   # meta-classifier regularisation


def _build_meta_features(X: np.ndarray, local_acc: np.ndarray) -> np.ndarray:
    """Assemble 5 meta-features × 3 models for every sample.

    Returns array of shape (N, 3, 5) which is then split per-model for fitting.
    """
    sigma  = X.std(axis=1)                   # (N,)
    mean_p = X.mean(axis=1)                  # (N,)
    margin = np.abs(X - 0.5)                 # (N, 3)

    # Stack for each model m: [p_m, |p_m-0.5|, local_acc_m, sigma, mean_p]
    # local_acc shape: (N, 3)
    meta = np.stack([
        X,                                   # (N, 3) — p_m per model
        margin,                              # (N, 3) — margin per model
        local_acc,                           # (N, 3) — local accuracy per model
        np.tile(sigma[:, None], (1, 3)),     # (N, 3) — same sigma for all models
        np.tile(mean_p[:, None], (1, 3)),    # (N, 3) — same mean for all models
    ], axis=2)                               # (N, 3, 5)
    return meta


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
    logger.info(f"Meta-classifier config: k_meta={K_META}  C={META_LOGREG_C}")

    # --- Per-model train-optimal thresholds (no leakage) ---
    logger.info("Deriving per-model train-optimal thresholds:")
    model_thresholds = np.zeros(3)
    for m in range(3):
        sweep_m = threshold_sweep(y_train, X_train[:, m])
        model_thresholds[m] = sweep_m["threshold"]
        logger.info(f"  {MODEL_NAMES[m]}: thr={sweep_m['threshold']:.3f}  "
                    f"F1={sweep_m['f1']:.4f}")

    # Binary predictions on train (N_tr, 3): was model m correct on train sample j?
    model_preds_train = (X_train >= model_thresholds[None, :]).astype(int)
    correct_train     = (model_preds_train == y_train[:, None]).astype(int)

    # --- Meta-training: k-NN with self-exclusion (LOO) ---
    logger.info(f"Computing LOO k-NN meta-features on train (k={K_META})...")
    nn_loo = NearestNeighbors(n_neighbors=K_META + 1, metric="euclidean", n_jobs=-1)
    nn_loo.fit(X_train)
    _, train_neighbor_idx = nn_loo.kneighbors(X_train)        # (N_tr, K+1)
    knn_idx_loo = train_neighbor_idx[:, 1:]                    # exclude self: (N_tr, K)

    # local accuracy in LOO neighbourhood
    knn_correct_loo   = correct_train[knn_idx_loo]             # (N_tr, K, 3)
    local_acc_train   = knn_correct_loo.mean(axis=1)           # (N_tr, 3)

    # Meta-features for train: (N_tr, 3, 5)
    meta_train = _build_meta_features(X_train, local_acc_train)

    # Meta-labels: was model m correct on THIS training sample?
    meta_y_train = correct_train                               # (N_tr, 3)

    # --- Train 3 meta-classifiers (one per model) ---
    meta_clfs = []
    for m in range(3):
        mX = meta_train[:, m, :]                               # (N_tr, 5)
        my = meta_y_train[:, m]                                # (N_tr,)
        pos_rate = my.mean()
        logger.info(f"Meta-clf {MODEL_NAMES[m]}: pos_rate={pos_rate:.3f}")
        meta_clf_m = LogisticRegression(
            C=META_LOGREG_C, max_iter=1000, solver="lbfgs", random_state=seed)
        meta_clf_m.fit(mX, my)
        coef_str = "  ".join(
            f"f{j}={c:.3f}" for j, c in enumerate(meta_clf_m.coef_[0]))
        logger.info(f"  Coefficients: {coef_str}")
        meta_clfs.append(meta_clf_m)

    # --- Inference: compute meta-features for val/test ---
    logger.info("Computing meta-features for val and test...")
    nn_inf = NearestNeighbors(n_neighbors=K_META, metric="euclidean", n_jobs=-1)
    nn_inf.fit(X_train)

    def _score_split(X_query: np.ndarray) -> np.ndarray:
        """Apply META-DES to a query set and return ensemble scores."""
        _, inf_idx    = nn_inf.kneighbors(X_query)             # (N_q, K_META)
        knn_corr      = correct_train[inf_idx]                  # (N_q, K, 3)
        local_acc_q   = knn_corr.mean(axis=1)                   # (N_q, 3)
        meta_q        = _build_meta_features(X_query, local_acc_q)  # (N_q, 3, 5)
        competences   = np.zeros((len(X_query), 3))
        for m, clf_m in enumerate(meta_clfs):
            competences[:, m] = clf_m.predict_proba(meta_q[:, m, :])[:, 1]
        # Weighted average using competence scores
        weights   = competences + 1e-9
        weights   = weights / weights.sum(axis=1, keepdims=True)
        scores    = (X_query * weights).sum(axis=1)
        return scores, weights

    logger.info("--- Val ---")
    scores_val,  weights_val  = _score_split(X_val)
    val_m  = evaluate(y_val,  scores_val,  "val",  logger)

    logger.info("--- Test ---")
    scores_test, _            = _score_split(X_test)
    test_m = evaluate(y_test, scores_test, "test", logger)

    mean_w = weights_val.mean(axis=0)
    logger.info("Mean val competence weights:")
    for name, mw in zip(MODEL_NAMES, mean_w):
        logger.info(f"  {name}: {mw:.4f}")

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: competence distributions per model on val
        colours = ["steelblue", "tomato", "forestgreen"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for m, (name, ax) in enumerate(zip(MODEL_NAMES, axes)):
            ax.hist(weights_val[:, m], bins=20, color=colours[m],
                    alpha=0.75, edgecolor="white")
            ax.axvline(mean_w[m], color="black", linestyle="--",
                       label=f"Mean={mean_w[m]:.3f}")
            ax.set_xlabel("Meta-competence weight")
            ax.set_title(name)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Count (val)")
        fig.suptitle(
            f"{EA_ID}: Competence weight distributions (val)\n"
            f"META-DES k={K_META} | weights = meta-clf prediction ∝ P(correct)"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_competence_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_competence_dist.png")

        # Plot 2: meta-classifier feature importance (coefficients)
        feature_names = ["p_i", "|p_i-0.5|", "local_acc_i", "σ_p", "mean_p"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for m, (name, ax) in enumerate(zip(MODEL_NAMES, axes)):
            coefs = meta_clfs[m].coef_[0]
            colors_bar = ["tomato" if c < 0 else "steelblue" for c in coefs]
            bars = ax.barh(feature_names, coefs, color=colors_bar, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Coefficient")
            ax.set_title(name)
        fig.suptitle(
            f"{EA_ID}: Meta-classifier coefficients per model\n"
            f"Positive = increases P(correct); C={META_LOGREG_C}"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_meta_coefs.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_meta_coefs.png")

        # Plot 3: score distribution on val
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scores_val[y_val == 0], bins=30, alpha=0.6,
                label="Non-fractured", color="steelblue")
        ax.hist(scores_val[y_val == 1], bins=30, alpha=0.6,
                label="Fractured", color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("META-DES ensemble score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}  "
            f"k={K_META}  C={META_LOGREG_C}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_dist.png")

        plot_confusion_matrix(y_val, scores_val, val_m["threshold"],
                              EA_ID, METHOD, plot_dir, logger)
        plot_confidence_accuracy(y_val, scores_val, val_m["threshold"],
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
            f"3 LogReg meta-clfs (C={META_LOGREG_C}, k={K_META}). "
            f"Meta-features: [p_i, |p_i-0.5|, local_acc_i, sigma_p, mean_p]. "
            f"LOO k-NN for meta-train. Per-model train thresholds: {thr_str}. "
            f"Mean val competence weights: {weight_str}."
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
