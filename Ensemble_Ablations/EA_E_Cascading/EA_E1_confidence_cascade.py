"""
EA-E1: Confidence cascade (LIT).

Models are ordered by individual train-optimal F1 score (highest first).
For each sample, the first model in the cascade decides if its confidence
margin exceeds δ (|p_i - 0.5| > δ). If not, the next model tries. The
last model always decides as a fallback.

  cascade_order = argsort(per-model train F1, descending)
  for stage i in cascade_order[:-1]:
    if |p_i(x) - 0.5| > δ  →  score(x) = p_i(x), stop
  fallback: score(x) = p_last(x)

δ is swept on val F1.
  δ = 0    → all samples decided by stage-1 (best model alone)
  δ ≥ 0.5  → all samples reach fallback (last model alone)

Reference: Alpaydin, E. & Kaynak, C. (1998).
           Cascading classifiers.
           Kybernetika, 34(4):369–374.
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

EA_ID       = "EA-E1"
METHOD      = "Confidence cascade"
FAMILY      = "E"
TYPE        = "LIT"
REF         = "Alpaydin & Kaynak 1998"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

DELTA_GRID  = np.arange(0.00, 0.51, 0.01)   # confidence margin threshold (51 values)


def _cascade_scores(probs: np.ndarray, cascade_order: list, delta: float) -> tuple:
    """Apply confidence cascade at threshold delta.

    Args:
        probs:         (N, 3) model probabilities
        cascade_order: list of 3 model indices, highest-F1 first
        delta:         confidence threshold (|p - 0.5| > delta → decide)

    Returns:
        scores: (N,) final score per sample
        stages: (N,) which stage (0/1/2) decided each sample
    """
    N = len(probs)
    scores = np.zeros(N)
    stages = np.full(N, fill_value=len(cascade_order) - 1, dtype=int)

    undecided = np.ones(N, dtype=bool)

    for stage_idx, model_idx in enumerate(cascade_order[:-1]):
        margins = np.abs(probs[undecided, model_idx] - 0.5)
        confident = margins > delta
        undecided_indices = np.where(undecided)[0]
        decided = undecided_indices[confident]

        scores[decided] = probs[decided, model_idx]
        stages[decided] = stage_idx
        undecided[decided] = False

    last_model = cascade_order[-1]
    scores[undecided] = probs[undecided, last_model]

    return scores, stages


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

    # --- Determine cascade order from per-model train F1 (descending) ---
    logger.info("Deriving cascade order from per-model train F1:")
    model_train_f1 = np.zeros(3)
    for m in range(3):
        sweep_m = threshold_sweep(y_train, X_train[:, m])
        model_train_f1[m] = sweep_m["f1"]
        logger.info(f"  {MODEL_NAMES[m]}: train F1={sweep_m['f1']:.4f}  "
                    f"thr={sweep_m['threshold']:.3f}")

    cascade_order  = list(np.argsort(-model_train_f1))           # descending
    ordered_names  = [MODEL_NAMES[i] for i in cascade_order]
    logger.info(f"Cascade order (stage 0→1→2): {' → '.join(ordered_names)}")

    # Reference check: δ=0 → stage-1 model only; δ≥0.5 → last model only
    for ref_delta, label in [(0.0, "stage-1 only"), (0.50, "last model only")]:
        sc, _ = _cascade_scores(X_val, cascade_order, ref_delta)
        sw    = threshold_sweep(y_val, sc)
        logger.info(f"Reference check: δ={ref_delta:.2f} ({label}) → "
                    f"val F1={sw['f1']:.4f}")

    # --- Delta sweep on val F1 ---
    logger.info(f"Sweeping delta on val F1 ({len(DELTA_GRID)} values):")
    best_delta, best_val_f1 = DELTA_GRID[0], -1.0
    delta_log = []

    for delta in DELTA_GRID:
        scores_val, stages_val = _cascade_scores(X_val, cascade_order, delta)
        sweep = threshold_sweep(y_val, scores_val)
        stage_counts = np.bincount(stages_val, minlength=3)
        delta_log.append({
            "delta": delta, "val_f1": sweep["f1"],
            "stage_counts": stage_counts.copy(),
        })
        if sweep["f1"] > best_val_f1:
            best_val_f1, best_delta = sweep["f1"], delta

    logger.info(f"Best delta={best_delta:.2f}  (val F1={best_val_f1:.4f})")

    # --- Final evaluation at best delta ---
    scores_val_final, stages_val_final   = _cascade_scores(X_val,  cascade_order, best_delta)
    scores_test_final, _                 = _cascade_scores(X_test, cascade_order, best_delta)

    logger.info("--- Val (best delta) ---")
    val_m  = evaluate(y_val,  scores_val_final,  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(y_test, scores_test_final, "test", logger)

    stage_counts_val = np.bincount(stages_val_final, minlength=3)
    stage_frac_val   = stage_counts_val / stage_counts_val.sum()
    logger.info(f"Val stage distribution (delta={best_delta:.2f}):")
    for stage_idx, model_idx in enumerate(cascade_order):
        logger.info(f"  Stage {stage_idx} ({MODEL_NAMES[model_idx]}): "
                    f"n={stage_counts_val[stage_idx]}  "
                    f"({stage_frac_val[stage_idx]*100:.1f}%)")

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)
        colours = ["steelblue", "tomato", "forestgreen"]

        # Plot 1: delta sweep — val F1 vs delta
        deltas = [d["delta"] for d in delta_log]
        f1s    = [d["val_f1"] for d in delta_log]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(deltas, f1s, "-", color="steelblue", linewidth=2)
        ax.axvline(best_delta, color="tomato", linestyle="--",
                   label=f"Best δ={best_delta:.2f}  (F1={best_val_f1:.4f})")
        ax.set_xlabel("δ (confidence threshold: |p − 0.5| > δ → decide)")
        ax.set_ylabel("Val F1 (threshold-swept)")
        ax.set_title(
            f"{EA_ID}: δ-sweep on val\n"
            f"Cascade order: {' → '.join(ordered_names)}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_delta_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_delta_sweep.png")

        # Plot 2: cascade stage distribution vs delta
        stage_matrix = np.array([d["stage_counts"] for d in delta_log])  # (n_deltas, 3)
        stage_fracs  = stage_matrix / stage_matrix.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        bottom = np.zeros(len(deltas))
        for s in range(3):
            model_idx = cascade_order[s]
            ax.fill_between(deltas, bottom, bottom + stage_fracs[:, s],
                            alpha=0.75, label=f"Stage {s}: {MODEL_NAMES[model_idx]}",
                            color=colours[s])
            bottom += stage_fracs[:, s]
        ax.axvline(best_delta, color="black", linestyle="--",
                   label=f"Best δ={best_delta:.2f}")
        ax.set_xlabel("δ (confidence threshold)")
        ax.set_ylabel("Fraction of val samples")
        ax.set_title(
            f"{EA_ID}: Cascade stage distribution vs δ\n"
            "Fraction of samples decided at each stage"
        )
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_stage_dist.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_stage_dist.png")

        # Plot 3: score distribution on val at best delta
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scores_val_final[y_val == 0], bins=30, alpha=0.6,
                label="Non-fractured", color="steelblue")
        ax.hist(scores_val_final[y_val == 1], bins=30, alpha=0.6,
                label="Fractured", color="tomato")
        ax.axvline(val_m["threshold"], color="black", linestyle="--",
                   label=f"Threshold={val_m['threshold']:.3f}")
        ax.set_xlabel("Cascade score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{EA_ID}: {METHOD}\n"
            f"Val F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}  δ={best_delta:.2f}"
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
    order_str = " → ".join(f"{MODEL_NAMES[i]} (F1={model_train_f1[i]:.4f})"
                            for i in cascade_order)
    stage_str = "  ".join(
        f"Stage{s}({MODEL_NAMES[cascade_order[s]]})={stage_counts_val[s]}"
        for s in range(3)
    )
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
            f"Cascade order (by train F1): {order_str}. "
            f"Best delta={best_delta:.2f}. "
            f"Val stage distribution: {stage_str}."
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
