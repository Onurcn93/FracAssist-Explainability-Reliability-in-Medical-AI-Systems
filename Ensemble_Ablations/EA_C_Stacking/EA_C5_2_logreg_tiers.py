"""
EA-C5.2: Feature-engineered stacking — LogReg tier ablation.

Ablates feature tiers T1–T6 cumulatively using Logistic Regression as
the meta-learner. T1 baseline (raw probs only) is the EA-C1 result — no
rerun needed (identical feature set, identical protocol).

Cumulative tier ablation configs (5 new experiments):
  T1+T2       (7 feat)  — + ensemble summary statistics
  T1+T2+T3    (10 feat) — + pairwise disagreement
  T1+T2+T3+T4 (14 feat) — + F1-reliability-weighted probs
  T1+..+T5    (20 feat) — + per-model calibration residuals
  T1+..+T6    (23 feat) — + GradCAM spatial disagreement (ORIG)

Protocol: C swept on val F1 (same grid as EA-C1; fit on train, eval on val).
Best-tier result reported in EA_comparative_table.csv.
Full tier ablation saved to results/per_method/EA_C5_2_tier_ablation.csv.

References:
  Wolpert 1992; Ting & Witten 1999 (LogReg stacking base)
  Feature tiers T4–T6: this thesis (EA-series ORIG)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

THIS_DIR = Path(__file__).resolve().parent
EA_ROOT  = THIS_DIR.parent
sys.path.insert(0, str(EA_ROOT))

from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)
from EA_C_Stacking.C5_1_build_features import TIER_COLS   # noqa: E402

EA_ID  = "EA-C5.2"
METHOD = "Feature-engineered stacking — LogReg tier ablation"
FAMILY = "C"
TYPE   = "ORIG"
REF    = "Wolpert 1992; Ting & Witten 1999 (base); Tier features: this thesis"

DATA_CSV = "data/all_c5.csv"
C_GRID   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# T1 baseline from EA-C1 — no rerun; same 3 features, same protocol
C1_BASELINE = {"val_f1": 0.7368, "val_auc": 0.9042, "label": "T1\n(C1 ref)", "n_feat": 3}

TIER_SEQUENCE = ["T1", "T2", "T3", "T4", "T5", "T6"]


def _build_configs():
    """Build 5 cumulative feature-column configs starting at T1+T2."""
    configs = []
    running = []
    for tier in TIER_SEQUENCE:
        running = running + TIER_COLS[tier]
        if tier == "T1":
            continue   # T1-only = C1 baseline, skip
        label = "+".join(TIER_SEQUENCE[: TIER_SEQUENCE.index(tier) + 1])
        configs.append({"label": label, "cols": list(running), "n_feat": len(running)})
    return configs


def _select_c(X_tr, y_tr, X_val, y_val, logger, tag):
    best_c, best_f1 = C_GRID[0], -1.0
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        sw = threshold_sweep(y_val, clf.predict_proba(X_val)[:, 1])
        logger.info(
            f"  [{tag}] C={c:.3f}  val F1={sw['f1']:.4f}  thr={sw['threshold']:.3f}"
        )
        if sw["f1"] > best_f1:
            best_f1, best_c = sw["f1"], c
    logger.info(f"  [{tag}] → best C={best_c}  (val F1={best_f1:.4f})")
    return best_c


def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info("=" * 60)
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info("=" * 60)

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    train_m = df["split"].values == "train"
    val_m   = df["split"].values == "val"
    test_m  = df["split"].values == "test"
    y       = (df["true_label"] == "Fractured").astype(int).values

    logger.info(f"Train: {train_m.sum()} | Val: {val_m.sum()} | Test: {test_m.sum()}")

    configs = _build_configs()
    logger.info("Tier ablation configs (5 new; T1 baseline from C1):")
    for cfg in configs:
        logger.info(f"  {cfg['label']}  ({cfg['n_feat']} features)")

    tier_rows = []

    for cfg in configs:
        tag    = cfg["label"]
        cols   = cfg["cols"]
        n_feat = cfg["n_feat"]

        X_tr  = df.loc[train_m, cols].values
        X_val = df.loc[val_m,   cols].values
        X_te  = df.loc[test_m,  cols].values

        logger.info(f"\n--- {tag} ({n_feat} features) ---")
        best_c = _select_c(X_tr, y[train_m], X_val, y[val_m], logger, tag)

        clf = LogisticRegression(C=best_c, max_iter=1000, random_state=seed, solver="lbfgs")
        clf.fit(X_tr, y[train_m])

        sc_val  = clf.predict_proba(X_val)[:, 1]
        sc_test = clf.predict_proba(X_te)[:, 1]
        sc_tr   = clf.predict_proba(X_tr)[:, 1]

        vm  = evaluate(y[val_m],   sc_val,  "val",   logger)
        tm  = evaluate(y[test_m],  sc_test, "test",  logger)
        trm = evaluate(y[train_m], sc_tr,   "train", logger)

        tier_rows.append({
            "tier_config":  tag,
            "n_features":   n_feat,
            "best_c":       best_c,
            "val_f1":       vm["f1"],        "val_auc":       vm["auc"],
            "val_recall":   vm["recall"],     "val_precision": vm["precision"],
            "val_threshold":vm["threshold"],  "val_acc":       vm["accuracy"],
            "val_spec":     vm["specificity"],
            "test_f1":      tm["f1"],         "test_auc":      tm["auc"],
            "test_recall":  tm["recall"],     "test_precision":tm["precision"],
            "test_threshold":tm["threshold"], "test_acc":      tm["accuracy"],
            "test_spec":    tm["specificity"],
            "train_f1":     trm["f1"],        "train_auc":     trm["auc"],
            "train_acc":    trm["accuracy"],  "train_recall":  trm["recall"],
            "train_spec":   trm["specificity"],"train_threshold":trm["threshold"],
            # keep scores for plots (best tier)
            "_sc_val":   sc_val,
            "_sc_test":  sc_test,
            "_vm":       vm,
            "_tm":       tm,
            "_trm":      trm,
        })

    best = max(tier_rows, key=lambda r: r["val_f1"])
    logger.info(
        f"\nBest tier: {best['tier_config']}  "
        f"(val F1={best['val_f1']:.4f}, AUC={best['val_auc']:.4f}, C={best['best_c']})"
    )

    # ── Save tier ablation CSV ───────────────────────────────────────────────
    pm_dir = root_dir / "results" / "per_method"
    pm_dir.mkdir(parents=True, exist_ok=True)
    ablation_csv = pm_dir / "EA_C5_2_tier_ablation.csv"
    save_cols = [k for k in tier_rows[0] if not k.startswith("_")]
    pd.DataFrame([{k: r[k] for k in save_cols} for r in tier_rows]).to_csv(
        ablation_csv, index=False
    )
    logger.info(f"Tier ablation table saved → {ablation_csv}")

    # ── Plots ────────────────────────────────────────────────────────────────
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: tier ablation bar chart — Val F1
        x_labels = [C1_BASELINE["label"]] + [r["tier_config"].replace("+", "\n+") for r in tier_rows]
        f1s      = [C1_BASELINE["val_f1"]] + [r["val_f1"] for r in tier_rows]
        colors   = ["#aaaaaa"] + [
            "#4CAF50" if r["val_f1"] > C1_BASELINE["val_f1"] else "#2196F3"
            for r in tier_rows
        ]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(x_labels, f1s, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(C1_BASELINE["val_f1"], color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"C1 baseline (T1, 3 feat) = {C1_BASELINE['val_f1']:.4f}")
        for bar, f1 in zip(bars, f1s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{f1:.4f}", ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylabel("Val F1 (threshold-swept)")
        y_lo = max(0.0, min(f1s) - 0.04)
        y_hi = min(1.0, max(f1s) + 0.06)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"{EA_ID}: Tier ablation — LogReg meta-learner\n"
            f"Green = beats C1 baseline | Best: {best['tier_config']}"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_tier_f1.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_tier_f1.png")

        # Plot 2: AUC line — shows monotonicity of tier contributions
        x_pos = list(range(len(tier_rows) + 1))
        aucs  = [C1_BASELINE["val_auc"]] + [r["val_auc"] for r in tier_rows]
        x_tck = [C1_BASELINE["label"].replace("\n", " ")] + [
            r["tier_config"] for r in tier_rows
        ]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x_pos, aucs, "o-", color="steelblue", linewidth=2, markersize=7)
        ax.axhline(C1_BASELINE["val_auc"], color="black", linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"C1 baseline = {C1_BASELINE['val_auc']:.4f}")
        for xp, auc in zip(x_pos, aucs):
            ax.text(xp, auc + 0.002, f"{auc:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_tck, rotation=15, ha="right")
        ax.set_ylabel("Val AUC")
        ax.set_ylim(max(0, min(aucs) - 0.02), min(1.0, max(aucs) + 0.03))
        ax.set_title(f"{EA_ID}: AUC tier ablation — LogReg meta-learner")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_tier_auc.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_tier_auc.png")

        # Plot 3 & 4: confusion matrix + confidence-accuracy for best tier
        plot_confusion_matrix(
            y[val_m], best["_sc_val"], best["_vm"]["threshold"],
            EA_ID, f"{METHOD} (best: {best['tier_config']})", plot_dir, logger,
        )
        plot_confidence_accuracy(
            y[val_m], best["_sc_val"], best["_vm"]["threshold"],
            EA_ID, f"{METHOD} (best: {best['tier_config']})", plot_dir, logger,
        )

    # ── Append best-tier to EA tables ────────────────────────────────────────
    row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "val_f1":        best["val_f1"],        "val_auc":       best["val_auc"],
        "val_recall":    best["val_recall"],     "val_precision": best["val_precision"],
        "val_threshold": best["val_threshold"],
        "test_f1":       best["test_f1"],        "test_auc":      best["test_auc"],
        "test_recall":   best["test_recall"],    "test_precision":best["test_precision"],
        "test_threshold":best["test_threshold"],
        "reference": REF,
        "notes": (
            f"Tier ablation: 5 cumulative configs (T1+T2 → T1+..+T6). "
            f"T1 baseline from EA-C1 (val F1={C1_BASELINE['val_f1']:.4f}). "
            f"Best: {best['tier_config']} ({best['n_features']} feat, "
            f"C={best['best_c']}, val F1={best['val_f1']:.4f}). "
            f"Train in-sample F1={best['train_f1']:.4f}."
        ),
    }
    append_to_table(row)
    logger.info("Best-tier row appended to EA_comparative_table.csv")

    results_row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "train_f1":          best["train_f1"],   "train_auc":       best["train_auc"],
        "train_acc":         best["train_acc"],   "train_recall":    best["train_recall"],
        "train_specificity": best["train_spec"],  "train_threshold": best["train_threshold"],
        "val_f1":            best["val_f1"],       "val_auc":         best["val_auc"],
        "val_acc":           best["val_acc"],       "val_recall":      best["val_recall"],
        "val_specificity":   best["val_spec"],      "val_threshold":   best["val_threshold"],
        "test_f1":           best["test_f1"],       "test_auc":        best["test_auc"],
        "test_acc":          best["test_acc"],       "test_recall":     best["test_recall"],
        "test_specificity":  best["test_spec"],      "test_threshold":  best["test_threshold"],
    }
    append_to_results(results_row)
    logger.info("Best-tier row appended to EA_results.csv")

    return row


if __name__ == "__main__":
    THIS_DIR = Path(__file__).resolve().parent
    ROOT_DIR = THIS_DIR.parent
    run(ROOT_DIR)
