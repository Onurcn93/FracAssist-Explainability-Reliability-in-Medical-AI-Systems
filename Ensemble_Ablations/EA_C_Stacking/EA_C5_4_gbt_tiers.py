"""
EA-C5.4: Feature-engineered stacking — GBT tier ablation.

Ablates feature tiers T1–T6 cumulatively using Gradient-Boosted Trees
as the meta-learner. T1 baseline (raw probs only) is the EA-C3 result.

Cumulative tier ablation configs (5 new experiments):
  T1+T2       (7 feat)  — + ensemble summary statistics
  T1+T2+T3    (10 feat) — + pairwise disagreement
  T1+T2+T3+T4 (14 feat) — + F1-reliability-weighted probs
  T1+..+T5    (20 feat) — + per-model calibration residuals
  T1+..+T6    (23 feat) — + GradCAM spatial disagreement (ORIG)

Grid per tier (40 configs):
  DEPTH_GRID    : [1, 2, 3, 4, 5]
  LR_GRID       : [0.05, 0.1, 0.2, 0.3]
  SUBSAMPLE_GRID: [0.6, 0.8]
  → 5 × 4 × 2 = 40 configs per tier × 5 tiers = 200 GBT fits

Protocol: grid search on val F1 → staged curve on best config to find
optimal n_trees at min val log-loss → final refit with best_n_trees.
Staged curve and grid heatmap saved for best (tier, config).

References:
  Friedman 2001 (Annals of Statistics) (GBT base)
  Feature tiers T4–T6: this thesis (EA-series ORIG)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

THIS_DIR = Path(__file__).resolve().parent
EA_ROOT  = THIS_DIR.parent
sys.path.insert(0, str(EA_ROOT))

from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy,
                         plot_training_curve)
from EA_C_Stacking.C5_1_build_features import TIER_COLS   # noqa: E402

EA_ID  = "EA-C5.4"
METHOD = "Feature-engineered stacking — GBT tier ablation"
FAMILY = "C"
TYPE   = "ORIG"
REF    = "Friedman 2001 (Annals of Statistics) (base); Tier features: this thesis"

DATA_CSV = "data/all_c5.csv"

DEPTH_GRID     = [1, 2, 3, 4, 5]
LR_GRID        = [0.05, 0.1, 0.2, 0.3]
SUBSAMPLE_GRID = [0.6, 0.8]
N_EST_CURVE    = 400

# T1 baseline from EA-C3 (no rerun — identical feature set, identical protocol)
C3_BASELINE = {"val_f1": 0.7013, "val_auc": 0.8079, "label": "T1\n(C3 ref)", "n_feat": 3}

TIER_SEQUENCE = ["T1", "T2", "T3", "T4", "T5", "T6"]


def _build_configs():
    configs = []
    running = []
    for tier in TIER_SEQUENCE:
        running = running + TIER_COLS[tier]
        if tier == "T1":
            continue
        label = "+".join(TIER_SEQUENCE[: TIER_SEQUENCE.index(tier) + 1])
        configs.append({"label": label, "cols": list(running), "n_feat": len(running)})
    return configs


# ── Grid search ──────────────────────────────────────────────────────────────

def _grid_search(X_tr, y_tr, X_val, y_val, logger, tag, seed):
    """Sweep depth × lr × subsample on val F1. Returns (best_depth, best_lr, best_sub, grid)."""
    best_depth, best_lr, best_sub, best_f1 = (
        DEPTH_GRID[0], LR_GRID[0], SUBSAMPLE_GRID[0], -1.0
    )
    grid = {}

    for sub in SUBSAMPLE_GRID:
        for depth in DEPTH_GRID:
            for lr in LR_GRID:
                clf = GradientBoostingClassifier(
                    n_estimators=N_EST_CURVE,
                    max_depth=depth,
                    learning_rate=lr,
                    subsample=sub,
                    random_state=seed,
                )
                clf.fit(X_tr, y_tr)
                sw = threshold_sweep(y_val, clf.predict_proba(X_val)[:, 1])
                grid[(sub, depth, lr)] = sw["f1"]
                logger.info(
                    f"  [{tag}] sub={sub:.1f}  depth={depth}  lr={lr:.2f}"
                    f"  val F1={sw['f1']:.4f}  thr={sw['threshold']:.3f}"
                )
                if sw["f1"] > best_f1:
                    best_f1, best_depth, best_lr, best_sub = sw["f1"], depth, lr, sub

    logger.info(
        f"  [{tag}] → best sub={best_sub}  depth={best_depth}  lr={best_lr}"
        f"  val F1={best_f1:.4f}"
    )
    return best_depth, best_lr, best_sub, grid


# ── Staged curve ─────────────────────────────────────────────────────────────

def _staged_curve(X_tr, y_tr, X_val, y_val, depth, lr, sub, seed):
    """Fit N_EST_CURVE trees; return staged train+val log-loss and best_n_trees."""
    clf = GradientBoostingClassifier(
        n_estimators=N_EST_CURVE,
        max_depth=depth,
        learning_rate=lr,
        subsample=sub,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)

    tr_losses, val_losses = [], []
    for pred in clf.staged_predict_proba(X_tr):
        tr_losses.append(log_loss(y_tr, pred))
    for pred in clf.staged_predict_proba(X_val):
        val_losses.append(log_loss(y_val, pred))

    best_n = int(np.argmin(val_losses)) + 1   # 1-indexed tree count
    return clf, tr_losses, val_losses, best_n


# ── Grid heatmap ─────────────────────────────────────────────────────────────

def _plot_grid_heatmap(grid, best_sub, best_depth, best_lr, tier_label,
                       plot_dir, ea_id, logger):
    """One heatmap panel per subsample value; best cell highlighted in red."""
    n_sub = len(SUBSAMPLE_GRID)
    fig, axes = plt.subplots(1, n_sub, figsize=(7 * n_sub, 5), sharey=True)
    if n_sub == 1:
        axes = [axes]

    for ax, sub in zip(axes, SUBSAMPLE_GRID):
        mat = np.array(
            [[grid[(sub, d, lr)] for lr in LR_GRID] for d in DEPTH_GRID]
        )
        sns.heatmap(
            mat, annot=True, fmt=".4f", cmap="YlGnBu",
            xticklabels=[str(lr) for lr in LR_GRID],
            yticklabels=[str(d) for d in DEPTH_GRID],
            linewidths=0.4, ax=ax, annot_kws={"size": 9},
            cbar=(sub == SUBSAMPLE_GRID[-1]),
        )
        if sub == best_sub:
            br = DEPTH_GRID.index(best_depth)
            bc = LR_GRID.index(best_lr)
            ax.add_patch(plt.Rectangle(
                (bc, br), 1, 1, fill=False, edgecolor="tomato", lw=2.5
            ))
        ax.set_title(f"subsample={sub}")
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("max_depth")

    fig.suptitle(
        f"{ea_id}: Val F1 grid — {tier_label}\n"
        f"Best: sub={best_sub}  depth={best_depth}  lr={best_lr}"
        f"  F1={grid[(best_sub, best_depth, best_lr)]:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fname = plot_dir / f"{ea_id}_grid_heatmap.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved → plots/{ea_id}/{ea_id}_grid_heatmap.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info("=" * 60)
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info("=" * 60)
    n_grid = len(DEPTH_GRID) * len(LR_GRID) * len(SUBSAMPLE_GRID)
    logger.info(
        f"Grid: depth={DEPTH_GRID}  lr={LR_GRID}  sub={SUBSAMPLE_GRID}"
        f"  → {n_grid} configs/tier × 5 tiers = {n_grid*5} GBT fits"
    )

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    tr_m  = df["split"].values == "train"
    val_m = df["split"].values == "val"
    te_m  = df["split"].values == "test"
    y     = (df["true_label"] == "Fractured").astype(int).values

    logger.info(f"Train: {tr_m.sum()} | Val: {val_m.sum()} | Test: {te_m.sum()}")

    configs = _build_configs()
    logger.info("Tier ablation configs (5 new; T1 baseline from C3):")
    for cfg in configs:
        logger.info(f"  {cfg['label']}  ({cfg['n_feat']} features)")

    tier_rows = []

    for cfg in configs:
        tag    = cfg["label"]
        cols   = cfg["cols"]
        n_feat = cfg["n_feat"]

        X_tr  = df.loc[tr_m,  cols].values.astype(float)
        X_val = df.loc[val_m, cols].values.astype(float)
        X_te  = df.loc[te_m,  cols].values.astype(float)

        logger.info(f"\n{'─'*50}")
        logger.info(f"Tier: {tag}  ({n_feat} features)")
        logger.info(f"{'─'*50}")

        best_depth, best_lr, best_sub, grid = _grid_search(
            X_tr, y[tr_m], X_val, y[val_m], logger, tag, seed
        )

        # Staged curve for best config
        logger.info(
            f"  [{tag}] Staged curve: sub={best_sub}  depth={best_depth}"
            f"  lr={best_lr}  n_est={N_EST_CURVE}"
        )
        _, tr_losses, val_losses, best_n = _staged_curve(
            X_tr, y[tr_m], X_val, y[val_m], best_depth, best_lr, best_sub, seed
        )
        logger.info(
            f"  [{tag}] best_n_trees={best_n}/{N_EST_CURVE}"
            f"  train_ll={tr_losses[best_n-1]:.4f}"
            f"  val_ll={val_losses[best_n-1]:.4f}"
        )

        # Final refit with best_n_trees
        clf = GradientBoostingClassifier(
            n_estimators=best_n,
            max_depth=best_depth,
            learning_rate=best_lr,
            subsample=best_sub,
            random_state=seed,
        )
        clf.fit(X_tr, y[tr_m])

        sc_val  = clf.predict_proba(X_val)[:, 1]
        sc_test = clf.predict_proba(X_te)[:, 1]
        sc_tr   = clf.predict_proba(X_tr)[:, 1]

        vm  = evaluate(y[val_m],  sc_val,  "val",   logger)
        tm  = evaluate(y[te_m],   sc_test, "test",  logger)
        trm = evaluate(y[tr_m],   sc_tr,   "train", logger)

        # Feature importances (GBT provides these natively)
        fi = clf.feature_importances_
        top3 = sorted(zip(cols, fi), key=lambda x: x[1], reverse=True)[:3]
        logger.info(
            f"  [{tag}] Top-3 features: "
            + "  ".join(f"{c}={v:.4f}" for c, v in top3)
        )

        tier_rows.append({
            "tier_config":    tag,
            "n_features":     n_feat,
            "best_depth":     best_depth,
            "best_lr":        best_lr,
            "best_subsample": best_sub,
            "best_n_trees":   best_n,
            "val_f1":         vm["f1"],        "val_auc":        vm["auc"],
            "val_recall":     vm["recall"],    "val_precision":  vm["precision"],
            "val_threshold":  vm["threshold"], "val_acc":        vm["accuracy"],
            "val_spec":       vm["specificity"],
            "test_f1":        tm["f1"],         "test_auc":       tm["auc"],
            "test_recall":    tm["recall"],     "test_precision": tm["precision"],
            "test_threshold": tm["threshold"],  "test_acc":       tm["accuracy"],
            "test_spec":      tm["specificity"],
            "train_f1":       trm["f1"],        "train_auc":      trm["auc"],
            "train_acc":      trm["accuracy"],  "train_recall":   trm["recall"],
            "train_spec":     trm["specificity"],"train_threshold":trm["threshold"],
            # stash for plots
            "_sc_val":        sc_val,
            "_vm":            vm,
            "_tr_losses":     tr_losses,
            "_val_losses":    val_losses,
            "_best_n":        best_n,
            "_grid":          grid,
            "_best_sub":      best_sub,
            "_best_depth":    best_depth,
            "_best_lr":       best_lr,
            "_top3":          top3,
        })

    best = max(tier_rows, key=lambda r: r["val_f1"])
    logger.info(
        f"\nBest tier: {best['tier_config']}"
        f"  sub={best['best_subsample']}  depth={best['best_depth']}"
        f"  lr={best['best_lr']}  n_trees={best['best_n_trees']}"
        f"  val F1={best['val_f1']:.4f}  AUC={best['val_auc']:.4f}"
    )

    # ── Save tier ablation CSV ───────────────────────────────────────────────
    pm_dir = root_dir / "results" / "per_method"
    pm_dir.mkdir(parents=True, exist_ok=True)
    save_cols = [k for k in tier_rows[0] if not k.startswith("_")]
    pd.DataFrame([{k: r[k] for k in save_cols} for r in tier_rows]).to_csv(
        pm_dir / "EA_C5_4_tier_ablation.csv", index=False
    )
    logger.info("Tier ablation table saved → results/per_method/EA_C5_4_tier_ablation.csv")

    # ── Plots ────────────────────────────────────────────────────────────────
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: tier ablation bar chart — Val F1
        x_labels = [C3_BASELINE["label"]] + [r["tier_config"].replace("+", "\n+")
                                               for r in tier_rows]
        f1s   = [C3_BASELINE["val_f1"]] + [r["val_f1"] for r in tier_rows]
        descs = ["—"] + [
            f"d={r['best_depth']} lr={r['best_lr']}\nn={r['best_n_trees']}"
            for r in tier_rows
        ]
        colors = ["#aaaaaa"] + [
            "#4CAF50" if r["val_f1"] > C3_BASELINE["val_f1"] else "#2196F3"
            for r in tier_rows
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(x_labels, f1s, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(C3_BASELINE["val_f1"], color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"C3 baseline (T1, 3 feat) = {C3_BASELINE['val_f1']:.4f}")
        for bar, f1, desc in zip(bars, f1s, descs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{f1:.4f}\n({desc})", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("Val F1 (threshold-swept)")
        y_lo = max(0.0, min(f1s) - 0.04)
        y_hi = min(1.0, max(f1s) + 0.08)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"{EA_ID}: Tier ablation — GBT meta-learner\n"
            f"Green = beats C3 baseline | Best: {best['tier_config']}"
            f"  depth={best['best_depth']}  lr={best['best_lr']}"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_tier_f1.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_tier_f1.png")

        # Plot 2: AUC line across tier configs
        x_pos = list(range(len(tier_rows) + 1))
        aucs  = [C3_BASELINE["val_auc"]] + [r["val_auc"] for r in tier_rows]
        x_tck = [C3_BASELINE["label"].replace("\n", " ")] + [
            r["tier_config"] for r in tier_rows
        ]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x_pos, aucs, "o-", color="steelblue", linewidth=2, markersize=7)
        ax.axhline(C3_BASELINE["val_auc"], color="black", linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"C3 baseline = {C3_BASELINE['val_auc']:.4f}")
        for xp, auc in zip(x_pos, aucs):
            ax.text(xp, auc + 0.002, f"{auc:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_tck, rotation=15, ha="right")
        ax.set_ylabel("Val AUC")
        ax.set_ylim(max(0, min(aucs) - 0.02), min(1.0, max(aucs) + 0.03))
        ax.set_title(f"{EA_ID}: AUC tier ablation — GBT meta-learner")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_tier_auc.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_tier_auc.png")

        # Plot 3: grid heatmap for best tier
        _plot_grid_heatmap(
            best["_grid"], best["_best_sub"], best["_best_depth"], best["_best_lr"],
            best["tier_config"], plot_dir, EA_ID, logger,
        )

        # Plot 4: staged training curve for best tier
        plot_training_curve(
            ea_id=EA_ID, method=METHOD, plot_dir=plot_dir, logger=logger,
            train_series=best["_tr_losses"],
            val_series=best["_val_losses"],
            x_label="Number of trees (boosting rounds)",
            train_label="Train log-loss",
            val_label="Val log-loss (actual val set)",
            best_idx=best["_best_n"] - 1,
            dual_axis=False,
        )

        # Plot 5: feature importance for best tier
        top3_cols = [c for c, _ in best["_top3"]]
        top3_vals = [v for _, v in best["_top3"]]
        best_cfg  = next(cfg for cfg in configs if cfg["label"] == best["tier_config"])
        all_cols  = best_cfg["cols"]
        all_fi    = []
        # refit to get importances (already fitted, but stored in best["_top3"] only partially)
        clf_fi = GradientBoostingClassifier(
            n_estimators=best["best_n_trees"],
            max_depth=best["best_depth"],
            learning_rate=best["best_lr"],
            subsample=best["best_subsample"],
            random_state=seed,
        )
        X_tr_best = df.loc[tr_m, all_cols].values.astype(float)
        clf_fi.fit(X_tr_best, y[tr_m])
        importances = clf_fi.feature_importances_
        fi_pairs = sorted(zip(all_cols, importances), key=lambda x: x[1], reverse=True)

        fi_cols = [c for c, _ in fi_pairs]
        fi_vals = [v for _, v in fi_pairs]

        fig, ax = plt.subplots(figsize=(9, max(4, len(fi_cols) * 0.35)))
        y_pos = np.arange(len(fi_cols))
        bars  = ax.barh(y_pos, fi_vals[::-1], color="steelblue", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fi_cols[::-1], fontsize=8)
        ax.set_xlabel("Feature importance (mean decrease in impurity)")
        ax.set_title(
            f"{EA_ID}: Feature importances — {best['tier_config']}\n"
            f"depth={best['best_depth']}  lr={best['best_lr']}"
            f"  sub={best['best_subsample']}  n_trees={best['best_n_trees']}"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_feature_importance.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_feature_importance.png")

        # Plot 6 & 7: confusion matrix + confidence-accuracy for best tier
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
            f"T1 baseline from EA-C3 (val F1={C3_BASELINE['val_f1']:.4f}). "
            f"Grid: {len(DEPTH_GRID)}×{len(LR_GRID)}×{len(SUBSAMPLE_GRID)}"
            f"={n_grid} configs/tier. "
            f"Best: {best['tier_config']} sub={best['best_subsample']}"
            f" depth={best['best_depth']} lr={best['best_lr']}"
            f" n_trees={best['best_n_trees']} (val F1={best['val_f1']:.4f}). "
            f"Train in-sample F1={best['train_f1']:.4f}."
        ),
    }
    append_to_table(row)
    logger.info("Best-tier row appended to EA_comparative_table.csv")

    results_row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "train_f1":          best["train_f1"],    "train_auc":       best["train_auc"],
        "train_acc":         best["train_acc"],    "train_recall":    best["train_recall"],
        "train_specificity": best["train_spec"],   "train_threshold": best["train_threshold"],
        "val_f1":            best["val_f1"],        "val_auc":         best["val_auc"],
        "val_acc":           best["val_acc"],        "val_recall":      best["val_recall"],
        "val_specificity":   best["val_spec"],       "val_threshold":   best["val_threshold"],
        "test_f1":           best["test_f1"],        "test_auc":        best["test_auc"],
        "test_acc":          best["test_acc"],        "test_recall":     best["test_recall"],
        "test_specificity":  best["test_spec"],       "test_threshold":  best["test_threshold"],
    }
    append_to_results(results_row)
    logger.info("Best-tier row appended to EA_results.csv")

    return row


if __name__ == "__main__":
    THIS_DIR = Path(__file__).resolve().parent
    ROOT_DIR = THIS_DIR.parent
    run(ROOT_DIR)
