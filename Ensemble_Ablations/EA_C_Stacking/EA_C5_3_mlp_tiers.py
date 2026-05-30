"""
EA-C5.3: Feature-engineered stacking — MLP tier ablation.

Ablates feature tiers T1–T6 cumulatively using a Multi-Layer Perceptron
as the meta-learner. T1 baseline (raw probs only) is the EA-C2 result.

Cumulative tier ablation configs (5 new experiments):
  T1+T2       (7 feat)  — + ensemble summary statistics
  T1+T2+T3    (10 feat) — + pairwise disagreement
  T1+T2+T3+T4 (14 feat) — + F1-reliability-weighted probs
  T1+..+T5    (20 feat) — + per-model calibration residuals
  T1+..+T6    (23 feat) — + GradCAM spatial disagreement (ORIG)

Architecture grid covers both wide (many units, 1 layer) and tall
(fewer units, 2 layers) designs. Each cell swept on val F1.

Grid:
  ARCH_GRID  — 8 architectures (4 single-layer + 4 double-layer)
  ALPHA_GRID — 4 L2 regularisation strengths
  → 32 configs per tier × 5 tiers = 160 training runs

Protocol: fit on train, select on val F1. Final epoch curve saved
for the best (tier, arch) combination.

References:
  Muller et al. 2022 (arXiv:2202.09015) (MLP stacking base)
  Feature tiers T4–T6: this thesis (EA-series ORIG)
"""

import copy
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

THIS_DIR = Path(__file__).resolve().parent
EA_ROOT  = THIS_DIR.parent
sys.path.insert(0, str(EA_ROOT))

from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy,
                         plot_training_curve)
from EA_C_Stacking.C5_1_build_features import TIER_COLS   # noqa: E402

EA_ID  = "EA-C5.3"
METHOD = "Feature-engineered stacking — MLP tier ablation"
FAMILY = "C"
TYPE   = "ORIG"
REF    = "Muller et al. 2022 (arXiv:2202.09015) (base); Tier features: this thesis"

DATA_CSV = "data/all_c5.csv"

# ── Architecture grid (wide × tall) ─────────────────────────────────────────
# Single hidden layer — wide approach: vary unit count
_SINGLE = [(16,), (32,), (64,), (128,)]
# Double hidden layer — tall approach: pyramid-style + square configs
_DOUBLE = [(32, 16), (64, 32), (128, 64), (64, 64)]
ARCH_GRID  = _SINGLE + _DOUBLE
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0]

MAX_EPOCHS = 500
PATIENCE   = 20

# T1 baseline from EA-C2 (no rerun — identical feature set, identical protocol)
C2_BASELINE = {"val_f1": 0.7320, "val_auc": 0.8857, "label": "T1\n(C2 ref)", "n_feat": 3}

TIER_SEQUENCE = ["T1", "T2", "T3", "T4", "T5", "T6"]


def _arch_label(arch: tuple) -> str:
    return "+".join(str(u) for u in arch)


def _build_configs():
    """Build 5 cumulative feature-column configs starting at T1+T2."""
    configs = []
    running = []
    for tier in TIER_SEQUENCE:
        running = running + TIER_COLS[tier]
        if tier == "T1":
            continue   # T1-only = C2 baseline, skip
        label = "+".join(TIER_SEQUENCE[: TIER_SEQUENCE.index(tier) + 1])
        configs.append({"label": label, "cols": list(running), "n_feat": len(running)})
    return configs


# ── Grid search ─────────────────────────────────────────────────────────────

def _grid_search(X_tr, y_tr, X_val, y_val, logger, tag, seed):
    """Sweep ARCH_GRID × ALPHA_GRID on val F1.

    Returns (best_arch, best_alpha, grid_dict).
    grid_dict keys: (arch_tuple, alpha) → val_f1
    """
    best_arch, best_alpha, best_f1 = ARCH_GRID[0], ALPHA_GRID[0], -1.0
    grid = {}

    for arch in ARCH_GRID:
        for alpha in ALPHA_GRID:
            clf = MLPClassifier(
                hidden_layer_sizes=arch,
                activation="relu",
                alpha=alpha,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=PATIENCE,
                validation_fraction=0.1,
                random_state=seed,
                solver="adam",
            )
            clf.fit(X_tr, y_tr)
            sw = threshold_sweep(y_val, clf.predict_proba(X_val)[:, 1])
            grid[(arch, alpha)] = sw["f1"]
            logger.info(
                f"  [{tag}] arch={_arch_label(arch):9s}  alpha={alpha:.3f}"
                f"  val F1={sw['f1']:.4f}  thr={sw['threshold']:.3f}"
                f"  iter={clf.n_iter_:3d}"
            )
            if sw["f1"] > best_f1:
                best_f1, best_arch, best_alpha = sw["f1"], arch, alpha

    logger.info(
        f"  [{tag}] → best arch={_arch_label(best_arch)}  "
        f"alpha={best_alpha}  val F1={best_f1:.4f}"
    )
    return best_arch, best_alpha, grid


# ── Epoch-by-epoch fit for training curve ────────────────────────────────────

def _fit_with_curve(X_tr, y_tr, X_val, y_val, arch, alpha, seed,
                    max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Refit MLP one epoch at a time on actual val set; restore best-val-loss weights."""
    clf = MLPClassifier(
        hidden_layer_sizes=arch,
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
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        tr_ll  = log_loss(y_tr,  clf.predict_proba(X_tr))
        val_ll = log_loss(y_val, clf.predict_proba(X_val))
        train_losses.append(tr_ll)
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

    if best_weights is not None:
        clf.coefs_      = best_weights[0]
        clf.intercepts_ = best_weights[1]

    return clf, train_losses, val_losses, best_epoch


# ── Grid heatmap ─────────────────────────────────────────────────────────────

def _plot_grid_heatmap(grid, best_arch, best_alpha, tier_label, plot_dir, ea_id, logger):
    arch_labels = [_arch_label(a) for a in ARCH_GRID]
    f1_matrix   = np.array(
        [[grid[(arch, alpha)] for alpha in ALPHA_GRID] for arch in ARCH_GRID]
    )

    # Separate row groups: single vs double layer
    n_single = len(_SINGLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        f1_matrix,
        annot=True, fmt=".4f", cmap="YlGnBu",
        xticklabels=[str(a) for a in ALPHA_GRID],
        yticklabels=arch_labels,
        linewidths=0.4, ax=ax,
        annot_kws={"size": 9},
    )
    # Highlight best cell
    br = ARCH_GRID.index(best_arch)
    bc = ALPHA_GRID.index(best_alpha)
    ax.add_patch(plt.Rectangle((bc, br), 1, 1, fill=False, edgecolor="tomato", lw=2.5))
    # Separator between single and double layers
    ax.axhline(n_single, color="navy", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.text(len(ALPHA_GRID) + 0.05, n_single / 2, "single\nlayer",
            va="center", ha="left", fontsize=8, color="navy")
    ax.text(len(ALPHA_GRID) + 0.05, n_single + len(_DOUBLE) / 2, "double\nlayer",
            va="center", ha="left", fontsize=8, color="navy")
    ax.set_xlabel("alpha (L2 weight decay)")
    ax.set_ylabel("Hidden architecture  (units per layer)")
    ax.set_title(
        f"{ea_id}: Val F1 grid — {tier_label}\n"
        f"Best: arch={_arch_label(best_arch)}  alpha={best_alpha}"
        f"  F1={grid[(best_arch, best_alpha)]:.4f}"
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
    logger.info(
        f"Grid: {len(ARCH_GRID)} archs × {len(ALPHA_GRID)} alphas = "
        f"{len(ARCH_GRID)*len(ALPHA_GRID)} configs per tier"
    )

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    tr_m  = df["split"].values == "train"
    val_m = df["split"].values == "val"
    te_m  = df["split"].values == "test"
    y     = (df["true_label"] == "Fractured").astype(int).values

    logger.info(f"Train: {tr_m.sum()} | Val: {val_m.sum()} | Test: {te_m.sum()}")

    configs = _build_configs()
    logger.info("Tier ablation configs (5 new; T1 baseline from C2):")
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

        best_arch, best_alpha, grid = _grid_search(
            X_tr, y[tr_m], X_val, y[val_m], logger, tag, seed
        )

        # Epoch-by-epoch refit with actual val set for clean training curve
        logger.info(
            f"  [{tag}] Refitting with epoch loop: "
            f"arch={_arch_label(best_arch)}  alpha={best_alpha}"
        )
        clf, tr_losses, val_losses, best_ep = _fit_with_curve(
            X_tr, y[tr_m], X_val, y[val_m], best_arch, best_alpha, seed
        )
        logger.info(
            f"  [{tag}] Stopped at epoch {len(tr_losses)} | "
            f"best val loss @ epoch {best_ep} "
            f"(train={tr_losses[best_ep]:.4f}  val={val_losses[best_ep]:.4f})"
        )

        sc_val  = clf.predict_proba(X_val)[:, 1]
        sc_test = clf.predict_proba(X_te)[:, 1]
        sc_tr   = clf.predict_proba(X_tr)[:, 1]

        vm  = evaluate(y[val_m],  sc_val,  "val",   logger)
        tm  = evaluate(y[te_m],   sc_test, "test",  logger)
        trm = evaluate(y[tr_m],   sc_tr,   "train", logger)

        tier_rows.append({
            "tier_config":    tag,
            "n_features":     n_feat,
            "best_arch":      _arch_label(best_arch),
            "best_alpha":     best_alpha,
            "best_epoch":     best_ep,
            "total_epochs":   len(tr_losses),
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
            "_sc_test":       sc_test,
            "_vm":            vm,
            "_tm":            tm,
            "_trm":           trm,
            "_tr_losses":     tr_losses,
            "_val_losses":    val_losses,
            "_best_ep":       best_ep,
            "_best_arch":     best_arch,
            "_best_alpha":    best_alpha,
            "_grid":          grid,
        })

    best = max(tier_rows, key=lambda r: r["val_f1"])
    logger.info(
        f"\nBest tier: {best['tier_config']}  "
        f"arch={best['best_arch']}  alpha={best['best_alpha']}  "
        f"val F1={best['val_f1']:.4f}  AUC={best['val_auc']:.4f}"
    )

    # ── Save tier ablation CSV ───────────────────────────────────────────────
    pm_dir = root_dir / "results" / "per_method"
    pm_dir.mkdir(parents=True, exist_ok=True)
    save_cols = [k for k in tier_rows[0] if not k.startswith("_")]
    pd.DataFrame([{k: r[k] for k in save_cols} for r in tier_rows]).to_csv(
        pm_dir / "EA_C5_3_tier_ablation.csv", index=False
    )
    logger.info(f"Tier ablation table saved → results/per_method/EA_C5_3_tier_ablation.csv")

    # ── Plots ────────────────────────────────────────────────────────────────
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: tier ablation bar chart — Val F1
        x_labels = [C2_BASELINE["label"]] + [r["tier_config"].replace("+", "\n+")
                                               for r in tier_rows]
        f1s   = [C2_BASELINE["val_f1"]] + [r["val_f1"] for r in tier_rows]
        archs = ["—"] + [r["best_arch"] for r in tier_rows]
        colors = ["#aaaaaa"] + [
            "#4CAF50" if r["val_f1"] > C2_BASELINE["val_f1"] else "#2196F3"
            for r in tier_rows
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(x_labels, f1s, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(C2_BASELINE["val_f1"], color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"C2 baseline (T1, 3 feat) = {C2_BASELINE['val_f1']:.4f}")
        for bar, f1, arch in zip(bars, f1s, archs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{f1:.4f}\n({arch})", ha="center", va="bottom", fontsize=7.5)
        ax.set_ylabel("Val F1 (threshold-swept)")
        y_lo = max(0.0, min(f1s) - 0.04)
        y_hi = min(1.0, max(f1s) + 0.07)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"{EA_ID}: Tier ablation — MLP meta-learner\n"
            f"Green = beats C2 baseline | Best: {best['tier_config']}"
            f" arch={best['best_arch']}"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_tier_f1.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_tier_f1.png")

        # Plot 2: AUC line across tier configs
        x_pos = list(range(len(tier_rows) + 1))
        aucs  = [C2_BASELINE["val_auc"]] + [r["val_auc"] for r in tier_rows]
        x_tck = [C2_BASELINE["label"].replace("\n", " ")] + [
            r["tier_config"] for r in tier_rows
        ]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x_pos, aucs, "o-", color="steelblue", linewidth=2, markersize=7)
        ax.axhline(C2_BASELINE["val_auc"], color="black", linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"C2 baseline = {C2_BASELINE['val_auc']:.4f}")
        for xp, auc in zip(x_pos, aucs):
            ax.text(xp, auc + 0.002, f"{auc:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_tck, rotation=15, ha="right")
        ax.set_ylabel("Val AUC")
        ax.set_ylim(max(0, min(aucs) - 0.02), min(1.0, max(aucs) + 0.03))
        ax.set_title(f"{EA_ID}: AUC tier ablation — MLP meta-learner")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_tier_auc.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_tier_auc.png")

        # Plot 3: grid heatmap for best tier
        _plot_grid_heatmap(
            best["_grid"], best["_best_arch"], best["_best_alpha"],
            best["tier_config"], plot_dir, EA_ID, logger,
        )

        # Plot 4: training curve for best (tier, arch)
        plot_training_curve(
            ea_id=EA_ID, method=METHOD, plot_dir=plot_dir, logger=logger,
            train_series=best["_tr_losses"],
            val_series=best["_val_losses"],
            x_label="Epoch",
            train_label="Train log-loss",
            val_label="Val log-loss (actual val set)",
            best_idx=best["_best_ep"],
            dual_axis=False,
        )

        # Plot 5 & 6: confusion matrix + confidence-accuracy for best tier
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
            f"T1 baseline from EA-C2 (val F1={C2_BASELINE['val_f1']:.4f}). "
            f"Grid: {len(ARCH_GRID)} archs (4 single + 4 double layer) "
            f"× {len(ALPHA_GRID)} alphas = {len(ARCH_GRID)*len(ALPHA_GRID)} configs/tier. "
            f"Best: {best['tier_config']} arch={best['best_arch']} "
            f"alpha={best['best_alpha']} (val F1={best['val_f1']:.4f}). "
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
