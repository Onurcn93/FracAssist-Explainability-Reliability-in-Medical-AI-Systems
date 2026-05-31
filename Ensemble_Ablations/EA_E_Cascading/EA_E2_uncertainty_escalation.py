"""
EA-E2: Uncertainty-escalation cascade (ORIG).

Two-stage cascade: stage-1 is the mean ensemble; stage-2 is a logistic
regression combiner trained on train probabilities. Samples whose mean
score falls in the uncertainty band (|mean − 0.5| < δ) are escalated
to stage-2.

  mean_score = mean(p_1, p_2, p_3)
  if |mean_score − 0.5| < δ  →  escalate  →  score = LogReg(p_1, p_2, p_3)
  else                         →  commit   →  score = mean_score

δ is swept on val F1.
  δ = 0    → nothing escalates → recovers mean ensemble (EA-A1, F1≈0.7383)
  δ → 0.5  → everything escalates → recovers LogReg combiner (EA-C1, F1≈0.7368)

Motivation: mean ensemble is optimal when models are confident; learned
combiner may add value for genuinely uncertain cases where the mean alone
provides little discrimination.

Contrast with EA-D3 (σ-triggered gating): D3 triggers on inter-model
disagreement; E2 triggers on output uncertainty. They are related but
not equivalent — models can agree near 0.5 (uncertain, low σ) or
disagree symmetrically (averaging to ~0.5, high σ).

Reference: Extends Alpaydin, E. & Kaynak, C. (1998). Cascading classifiers.
           Kybernetika, 34(4):369–374.
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

EA_ID       = "EA-E2"
METHOD      = "Uncertainty-escalation cascade"
FAMILY      = "E"
TYPE        = "ORIG"
REF         = "Extends Alpaydin & Kaynak 1998"

DATA_CSV    = "data/all_base.csv"
PROB_COLS   = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
MODEL_NAMES = ["ResNet", "DenseNet", "EfficientNet"]

# |mean - 0.5| ∈ [0, 0.5], so delta sweeps the same range.
DELTA_GRID  = np.concatenate([[0.0], np.arange(0.005, 0.505, 0.005)])  # 101 values
LOGREG_C    = 0.001          # matches EA-C1 / EA-D3


def _compute_scores(X: np.ndarray, logreg, delta: float) -> tuple:
    """Apply uncertainty-escalation cascade.

    Returns:
        scores:    (N,) final score per sample
        escalated: (N,) boolean — True if sent to stage-2 (LogReg)
    """
    mean_scores  = X.mean(axis=1)
    uncertainty  = np.abs(mean_scores - 0.5)
    escalate     = uncertainty < delta

    lr_scores = logreg.predict_proba(X)[:, 1]
    scores    = np.where(escalate, lr_scores, mean_scores)
    return scores, escalate


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
    logger.info(f"Stage-2 LogReg: C={LOGREG_C}  (trained on train split)")

    # --- Train stage-2 LogReg on train split ---
    logreg = LogisticRegression(C=LOGREG_C, max_iter=1000, solver="lbfgs",
                                random_state=seed)
    logreg.fit(X_train, y_train)
    coef_str = "  ".join(f"{n}={c:.3f}" for n, c in zip(MODEL_NAMES, logreg.coef_[0]))
    logger.info(f"Stage-2 LogReg: {coef_str}  bias={logreg.intercept_[0]:.3f}")

    # Reference checks
    sw_a1 = threshold_sweep(y_val, X_val.mean(axis=1))
    sw_c1 = threshold_sweep(y_val, logreg.predict_proba(X_val)[:, 1])
    logger.info(f"Reference check: δ=0 → mean F1={sw_a1['f1']:.4f}  "
                f"(expect EA-A1 ~0.7383)")
    logger.info(f"Reference check: δ→0.5 → LogReg F1={sw_c1['f1']:.4f}  "
                f"(expect EA-C1 ~0.7368)")

    # --- Delta sweep on val F1 ---
    logger.info(f"Sweeping delta on val F1 ({len(DELTA_GRID)} values):")
    best_delta, best_val_f1 = 0.0, -1.0
    delta_log = []

    for delta in DELTA_GRID:
        scores_val, esc_mask = _compute_scores(X_val, logreg, delta)
        sweep = threshold_sweep(y_val, scores_val)
        n_esc = int(esc_mask.sum())
        delta_log.append({
            "delta":     delta,
            "val_f1":    sweep["f1"],
            "n_esc":     n_esc,
            "esc_rate":  n_esc / len(y_val),
        })
        if sweep["f1"] > best_val_f1:
            best_val_f1, best_delta = sweep["f1"], delta

    logger.info(f"Best delta={best_delta:.3f}  (val F1={best_val_f1:.4f})")

    # --- Final evaluation at best delta ---
    scores_val_final,  escalate_val  = _compute_scores(X_val,  logreg, best_delta)
    scores_test_final, escalate_test = _compute_scores(X_test, logreg, best_delta)

    logger.info("--- Val (best delta) ---")
    val_m  = evaluate(y_val,  scores_val_final,  "val",  logger)
    logger.info("--- Test ---")
    test_m = evaluate(y_test, scores_test_final, "test", logger)

    n_esc_val  = int(escalate_val.sum())
    n_esc_test = int(escalate_test.sum())
    logger.info(f"Val  escalation: {n_esc_val}/{len(y_val)}  "
                f"({n_esc_val/len(y_val)*100:.1f}%) → stage-2")
    logger.info(f"Test escalation: {n_esc_test}/{len(y_test)}  "
                f"({n_esc_test/len(y_test)*100:.1f}%) → stage-2")

    # --- Plots ---
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: delta sweep — val F1 + escalation rate vs delta
        deltas    = [d["delta"]    for d in delta_log]
        f1s       = [d["val_f1"]   for d in delta_log]
        esc_rates = [d["esc_rate"] for d in delta_log]
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax2 = ax1.twinx()
        ax1.plot(deltas, f1s, "-", color="steelblue", linewidth=2, label="Val F1")
        ax1.axvline(best_delta, color="tomato", linestyle="--",
                    label=f"Best δ={best_delta:.3f}  (F1={best_val_f1:.4f})")
        ax2.plot(deltas, esc_rates, ":", color="grey", linewidth=1.5,
                 label="Escalation rate")
        ax1.set_xlabel("δ (uncertainty threshold: |mean − 0.5| < δ → escalate)")
        ax1.set_ylabel("Val F1 (threshold-swept)", color="steelblue")
        ax2.set_ylabel("Fraction escalated (val)", color="grey")
        ax1.set_title(
            f"{EA_ID}: δ-sweep on val\n"
            f"Stage-1: mean  |  Stage-2: LogReg (C={LOGREG_C})"
        )
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_delta_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_delta_sweep.png")

        # Plot 2: score distributions by stage on val (best delta)
        committed = ~escalate_val
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, mask, label_text in [
            (axes[0], committed,    f"Stage-1 committed (mean)  n={committed.sum()}"),
            (axes[1], escalate_val, f"Stage-2 escalated (LogReg)  n={escalate_val.sum()}"),
        ]:
            ax.hist(scores_val_final[mask & (y_val == 0)], bins=20, alpha=0.6,
                    color="steelblue", label="Non-fractured")
            ax.hist(scores_val_final[mask & (y_val == 1)], bins=20, alpha=0.6,
                    color="tomato", label="Fractured")
            ax.axvline(val_m["threshold"], color="black", linestyle="--",
                       label=f"Thr={val_m['threshold']:.3f}")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.set_title(label_text)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"{EA_ID}: Score distribution by stage (val, δ={best_delta:.3f})\n"
            f"F1={val_m['f1']:.4f}  AUC={val_m['auc']:.4f}"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_score_by_stage.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved -> plots/{EA_ID}/{EA_ID}_score_by_stage.png")

        # Plot 3: overall score distribution on val
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
    best_esc_rate = next(d["esc_rate"] for d in delta_log
                         if abs(d["delta"] - best_delta) < 1e-9)
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
            f"Stage-1: mean ensemble. Stage-2: LogReg (C={LOGREG_C}) fit on train. "
            f"Escalation: |mean - 0.5| < delta. Best delta={best_delta:.3f}. "
            f"Val escalation rate={best_esc_rate*100:.1f}%. "
            f"Ref: delta=0 → mean F1={sw_a1['f1']:.4f} (EA-A1); "
            f"delta→0.5 → LogReg F1={sw_c1['f1']:.4f} (EA-C1)."
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
