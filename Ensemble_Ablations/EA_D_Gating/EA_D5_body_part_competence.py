"""
EA-D5: Body-part-constrained region-of-competence weighting (ORIG).

Extension of EA-D4: the k nearest neighbours are now restricted to
training samples of the SAME anatomical body part (Hand / Leg / Hip /
Shoulder / Mixed). This prevents cross-anatomy contamination — a hand
fracture's competence estimate is no longer polluted by leg cases that
happen to share the same probability triple.

Algorithm per sample x:
  1. Identify body_part(x)
  2. Find k nearest neighbours among train samples with the SAME body_part,
     using Euclidean distance in 3D probability space.
  3. local_acc_i(x) = accuracy of model i on those k neighbours
     (using per-model train-optimal thresholds — no leakage)
  4. w_i(x) = (local_acc_i(x) + ε) / Σ_m (local_acc_m(x) + ε)
  5. score(x) = Σ_i w_i(x) · p_i(x)

Motivation: model error profiles are anatomy-dependent. DenseNet may be
more reliable on Hand cases while ResNet may be more reliable on Hip cases.
Body-part-constrained neighbourhoods expose this local structure; global
probability-space k-NN (EA-D4) averages it away.

Requires: data/all_d5.csv (built by data/build_d5_features.py).

Reference: Extends Woloszynski & Kurzynski (2011) Pattern Recognition,
           44(10-11):2386-2396. Body-part stratification: original
           contribution (this thesis).
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

EA_ID       = "EA-D5"
METHOD      = "Body-part-constrained region-of-competence weighting"
FAMILY      = "D"
TYPE        = "ORIG"
REF         = "Extends Woloszynski & Kurzynski 2011; body-part stratification (this thesis)"

DATA_CSV    = "data/all_d5.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]
BODY_PARTS  = ["Hand", "Hip", "Leg", "Mixed", "Shoulder"]

K_GRID      = [5, 10, 15, 20, 30]
EPSILON     = 1e-6


def _build_bp_nn_models(X_train: np.ndarray, bp_train: np.ndarray,
                        max_k: int) -> dict:
    """Fit one NearestNeighbors per body part on train samples of that part."""
    nn_models = {}
    for bp in BODY_PARTS:
        mask = bp_train == bp
        n    = mask.sum()
        if n == 0:
            continue
        k_fit = min(max_k, n)
        nn = NearestNeighbors(n_neighbors=k_fit, metric="euclidean", n_jobs=-1)
        nn.fit(X_train[mask])
        nn_models[bp] = {"nn": nn, "indices": np.where(mask)[0], "n": n}
    return nn_models


def _compute_scores(X_query: np.ndarray, bp_query: np.ndarray,
                    X_train: np.ndarray, correct_train: np.ndarray,
                    nn_models: dict, k: int) -> tuple:
    """Compute body-part-constrained ensemble scores and per-sample weights."""
    N = len(X_query)
    scores  = np.zeros(N)
    weights = np.zeros((N, 3))

    for i in range(N):
        bp   = bp_query[i]
        info = nn_models.get(bp)
        if info is None:
            # Fallback: equal weights (shouldn't happen with complete data)
            weights[i] = 1 / 3
            scores[i]  = X_query[i].mean()
            continue

        k_use    = min(k, info["n"])
        _, idx_  = info["nn"].kneighbors(X_query[[i]])        # (1, k_use)
        train_idx = info["indices"][idx_[0, :k_use]]           # global train indices

        knn_correct = correct_train[train_idx]                 # (k_use, 3)
        local_acc   = knn_correct.mean(axis=0)                 # (3,)

        w = local_acc + EPSILON
        w = w / w.sum()
        weights[i] = w
        scores[i]  = (X_query[i] * w).sum()

    return scores, weights


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:  # noqa: ARG001
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info(f"{'='*60}")

    # --- Load data ---
    csv_path = root_dir / DATA_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found. Run: python data/build_d5_features.py"
        )
    df     = pd.read_csv(csv_path)
    labels = (df["true_label"] == "Fractured").astype(int).values
    probs  = df[PROB_COLS].values                              # (N, 3)
    bparts = df["body_part"].values                            # (N,)

    train_mask = df["split"].values == "train"
    val_mask   = df["split"].values == "val"
    test_mask  = df["split"].values == "test"

    X_train = probs[train_mask];  y_train = labels[train_mask]
    X_val   = probs[val_mask];    y_val   = labels[val_mask]
    X_test  = probs[test_mask];   y_test  = labels[test_mask]
    bp_train = bparts[train_mask]
    bp_val   = bparts[val_mask]
    bp_test  = bparts[test_mask]

    logger.info(f"Loaded {len(df)} rows | Train={train_mask.sum()} | "
                f"Val={val_mask.sum()} | Test={test_mask.sum()}")

    # Body part pool sizes (train)
    logger.info("Train samples per body part:")
    for bp in BODY_PARTS:
        n = (bp_train == bp).sum()
        logger.info(f"  {bp}: {n}")

    # --- Per-model train-optimal thresholds (no leakage) ---
    logger.info("Deriving per-model train-optimal thresholds:")
    model_thresholds = np.zeros(3)
    for m in range(3):
        sweep_m = threshold_sweep(y_train, X_train[:, m])
        model_thresholds[m] = sweep_m["threshold"]
        logger.info(f"  {MODEL_NAMES[m]}: thr={sweep_m['threshold']:.3f}  "
                    f"F1={sweep_m['f1']:.4f}")

    model_preds_train = (X_train >= model_thresholds[None, :]).astype(int)
    correct_train     = (model_preds_train == y_train[:, None]).astype(int)

    # --- Fit body-part-specific NN models ---
    max_k = max(K_GRID)
    logger.info(f"Fitting {len(BODY_PARTS)} body-part-specific NearestNeighbors "
                f"(max k={max_k})...")
    nn_models = _build_bp_nn_models(X_train, bp_train, max_k)

    # --- k sweep on val ---
    logger.info("Sweeping k on val F1:")
    best_k, best_val_f1 = K_GRID[0], -1.0
    k_log = []

    for k in K_GRID:
        scores_val, _ = _compute_scores(
            X_val, bp_val, X_train, correct_train, nn_models, k)
        sweep = threshold_sweep(y_val, scores_val)
        k_log.append({"k": k, "val_f1": sweep["f1"]})
        logger.info(f"  k={k:3d}  val F1={sweep['f1']:.4f}  thr={sweep['threshold']:.3f}")
        if sweep["f1"] > best_val_f1:
            best_val_f1, best_k = sweep["f1"], k

    logger.info(f"Best k={best_k}  (val F1={best_val_f1:.4f})")

    # --- Final evaluation at best k ---
    scores_val_final, weights_val   = _compute_scores(
        X_val, bp_val, X_train, correct_train, nn_models, best_k)
    scores_test_final, _            = _compute_scores(
        X_test, bp_test, X_train, correct_train, nn_models, best_k)

    logger.info("--- Val (best k) ---")
    val_m  = evaluate(y_val,  scores_val_final,  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(y_test, scores_test_final, "test", logger)

    mean_w = weights_val.mean(axis=0)
    logger.info("Mean val weights (best k):")
    for name, mw in zip(MODEL_NAMES, mean_w):
        logger.info(f"  {name}: {mw:.4f}")

    # --- Per-body-part val breakdown ---
    logger.info("Per-body-part val breakdown (best k, fixed global threshold):")
    bp_results = {}
    for bp in BODY_PARTS:
        mask_bp = bp_val == bp
        n_bp    = mask_bp.sum()
        if n_bp < 5:
            continue
        frac_bp = y_val[mask_bp].mean()
        from sklearn.metrics import f1_score, roc_auc_score
        preds_bp = (scores_val_final[mask_bp] >= val_m["threshold"]).astype(int)
        f1_bp    = f1_score(y_val[mask_bp], preds_bp, zero_division=0)
        auc_bp   = roc_auc_score(y_val[mask_bp], scores_val_final[mask_bp]) \
                   if len(np.unique(y_val[mask_bp])) > 1 else float("nan")
        bp_results[bp] = {"n": n_bp, "frac_rate": frac_bp, "f1": f1_bp, "auc": auc_bp}
        logger.info(f"  {bp}: n={n_bp}  frac={frac_bp:.2f}  "
                    f"F1={f1_bp:.4f}  AUC={auc_bp:.4f}")

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: k sweep
        ks  = [d["k"] for d in k_log]
        f1s = [d["val_f1"] for d in k_log]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, f1s, "o-", color="steelblue", linewidth=2, markersize=7)
        ax.axvline(best_k, color="tomato", linestyle="--",
                   label=f"Best k={best_k}  (F1={best_val_f1:.4f})")
        ax.set_xlabel("k (neighbourhood size, same body part only)")
        ax.set_ylabel("Val F1 (threshold-swept)")
        ax.set_title(
            f"{EA_ID}: k-sweep on val\n"
            "Body-part-constrained k-NN in 3-prob space"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_k_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_k_sweep.png")

        # Plot 2: per-body-part val F1 (D5 vs D4 reference)
        D4_REF_F1 = 0.7394  # EA-D4 val F1 for comparison
        if bp_results:
            bps   = list(bp_results.keys())
            f1s_bp  = [bp_results[bp]["f1"] for bp in bps]
            ns_bp   = [bp_results[bp]["n"] for bp in bps]
            colours = ["steelblue", "tomato", "forestgreen", "darkorange", "purple"][:len(bps)]
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(bps, f1s_bp, color=colours, edgecolor="white")
            for bar, f1v, n in zip(bars, f1s_bp, ns_bp):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{f1v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)
            ax.axhline(D4_REF_F1, color="black", linestyle="--", linewidth=1.2,
                       label=f"EA-D4 overall val F1 = {D4_REF_F1:.4f}")
            ax.set_ylabel("Val F1 (global threshold)")
            ax.set_title(
                f"{EA_ID}: Val F1 per body part (k={best_k})\n"
                "Dashed = EA-D4 overall reference"
            )
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{EA_ID}_body_part_f1.png", dpi=150)
            plt.close(fig)
            logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_body_part_f1.png")

        # Plot 3: weight distributions per model (val, best k)
        colours3 = ["steelblue", "tomato", "forestgreen"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for m, (name, ax) in enumerate(zip(MODEL_NAMES, axes)):
            ax.hist(weights_val[:, m], bins=15, color=colours3[m],
                    alpha=0.75, edgecolor="white")
            ax.axvline(weights_val[:, m].mean(), color="black", linestyle="--",
                       label=f"Mean={weights_val[:,m].mean():.3f}")
            ax.set_xlabel("Competence weight")
            ax.set_title(name)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Count (val)")
        fig.suptitle(f"{EA_ID}: Competence weight distributions (val, k={best_k})")
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
        ax.set_xlabel("Body-part-constrained ensemble score")
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
    bp_str     = "  ".join(
        f"{bp}:F1={v['f1']:.3f}" for bp, v in bp_results.items()
    ) if bp_results else ""

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
            f"k={best_k} (body-part-constrained k-NN). "
            f"Per-model train thresholds: {thr_str}. "
            f"Mean val weights: {weight_str}. "
            f"Per-body-part val F1: {bp_str}."
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
