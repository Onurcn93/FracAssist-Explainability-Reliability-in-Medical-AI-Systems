"""
EA-C5.6: Feature-engineered stacking — disagreement stratification analysis.

Stratifies val and test sets by inter-model disagreement
(σ_p = std of the 3 CNN probabilities) and measures how each of the three
best C5 meta-learners performs per stratum:
  - LogReg  T1+T2       (C5.2 best, val F1=0.7368, C=0.001)
  - MLP     T1+..+T6    (C5.3 best, val F1≈0.7421, arch=(64,) α=1.0)
  - GBT     T1+T2+T3    (C5.4 best, val F1=0.7229, d=3 lr=0.30)

Disagreement measure:
  σ_p = std(p_ResNet, p_DenseNet, p_EfficientNet) ∈ [0, 0.5]
  Stratum thresholds: 33rd / 67th percentile of σ_p on val set only.
  Low  = σ_p < q33   (models largely agree)
  Med  = q33 ≤ σ_p < q67
  High = σ_p ≥ q67   (high inter-model disagreement)

Protocol:
  1. Refit each meta-learner on train (same hyperparameters as source C5.x).
  2. Score val and test with each meta-learner.
  3. Find global optimal threshold per method from full val set.
  4. Apply that fixed threshold to each stratum — no per-stratum sweep
     (stratum n ≈ 30; per-stratum sweep would overfit to noise).
  5. Report n, F1, AUC, ECE per (stratum × method) for val and test.

Research question:
  Does MLP+T6 gain over LogReg/GBT specifically in High-disagreement cases?
  T6 (GradCAM spatial distances) provides information unavailable to LogReg
  and GBT, and that spatial information is most relevant when probability
  signals conflict strongly.

Global EA table entry: best overall method (MLP) val/test metrics.
Novel contribution: per-stratum breakdown in per_method CSV and plots.

References:
  Kuncheva 2014 (diversity in ensembles)
  Margineantu & Dietterich 1997 (disagreement measures)
  Source meta-learners: this thesis (EA-C5.2 / C5.3 / C5.4)
"""

import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix as sk_confusion_matrix,
)

THIS_DIR = Path(__file__).resolve().parent
EA_ROOT  = THIS_DIR.parent
sys.path.insert(0, str(EA_ROOT))

from common.eval import (get_logger, evaluate, append_to_table,
                         append_to_results, plot_confusion_matrix,
                         plot_confidence_accuracy)
from EA_C_Stacking.C5_1_build_features import TIER_COLS   # noqa: E402

EA_ID  = "EA-C5.6"
METHOD = "Feature-engineered stacking — disagreement stratification analysis"
FAMILY = "C"
TYPE   = "ORIG"
REF    = ("Kuncheva 2014 (diversity in ensembles); "
          "Margineantu & Dietterich 1997 (disagreement measures); "
          "Source meta-learners: this thesis (EA-C5.2/C5.3/C5.4)")

DATA_CSV     = "data/all_c5.csv"
DISAGREE_COL = "std_p"      # T2: std of 3 CNN probabilities, range ≈ [0, 0.5]
STRATA       = ["Low", "Medium", "High"]
N_STRATA     = 3
N_BINS_ECE   = 10

_T = TIER_COLS

META_CONFIGS = [
    {
        "ea_id": "EA-C5.2", "learner": "LogReg", "tier_label": "T1+T2",
        "cols": _T["T1"] + _T["T2"],
        "logreg_c": 0.001,
    },
    {
        "ea_id": "EA-C5.3", "learner": "MLP", "tier_label": "T1+..+T6",
        "cols": [c for t in ["T1", "T2", "T3", "T4", "T5", "T6"] for c in _T[t]],
        "arch": (64,), "alpha": 1.0,
    },
    {
        "ea_id": "EA-C5.4", "learner": "GBT", "tier_label": "T1+T2+T3",
        "cols": _T["T1"] + _T["T2"] + _T["T3"],
        "depth": 3, "lr": 0.30, "subsample": 0.8, "n_trees": 3,
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _compute_ece(y_true: np.ndarray, scores: np.ndarray,
                 n_bins: int = N_BINS_ECE) -> float:
    """Expected Calibration Error: Σ_b (|B_b|/n) × |conf − acc|."""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(float(scores[mask].mean()) - float(y_true[mask].mean()))
    return ece


def _eval_at_threshold(y_true: np.ndarray, scores: np.ndarray,
                       thr: float) -> dict:
    """Fixed-threshold evaluation — no sweep. Safe for small-n strata."""
    if len(y_true) == 0:
        return dict(f1=np.nan, auc=np.nan, recall=np.nan, precision=np.nan,
                    specificity=np.nan, accuracy=np.nan, ece=np.nan,
                    n=0, n_pos=0)
    preds = (scores >= thr).astype(int)
    f1    = float(f1_score(y_true, preds, zero_division=0.0))
    rec   = float(recall_score(y_true, preds, zero_division=0.0))
    prec  = float(precision_score(y_true, preds, zero_division=0.0))
    try:
        auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan
    cm = sk_confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    spec   = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc    = float((preds == y_true).mean())
    ece    = _compute_ece(y_true, scores)
    return dict(f1=f1, auc=auc, recall=rec, precision=prec,
                specificity=spec, accuracy=acc, ece=ece,
                n=len(y_true), n_pos=int(y_true.sum()))


def _fit_meta(cfg: dict, X_tr: np.ndarray, y_tr: np.ndarray,
              seed: int):
    """Refit meta-learner with identical hyperparameters as source C5.x."""
    learner = cfg["learner"]
    if learner == "LogReg":
        clf = LogisticRegression(C=cfg["logreg_c"], max_iter=1000,
                                 random_state=seed, solver="lbfgs")
        clf.fit(X_tr, y_tr)
    elif learner == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=cfg["arch"], activation="relu",
            alpha=cfg["alpha"], max_iter=500, early_stopping=True,
            n_iter_no_change=20, validation_fraction=0.1,
            random_state=seed, solver="adam",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
    elif learner == "GBT":
        clf = GradientBoostingClassifier(
            n_estimators=cfg["n_trees"], max_depth=cfg["depth"],
            learning_rate=cfg["lr"], subsample=cfg["subsample"],
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)
    return clf


# ── Main ────────────────────────────────────────────────────────────────────

def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info("=" * 60)
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info("=" * 60)
    logger.info(
        "Disagreement: σ_p = std(p_ResNet, p_DenseNet, p_EfficientNet)  [T2: std_p]\n"
        "Strata: tertile thresholds from val only → Low / Medium / High\n"
        "Threshold: global optimal (full val) applied per-stratum — no per-stratum sweep."
    )

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    tr_m  = df["split"].values == "train"
    val_m = df["split"].values == "val"
    te_m  = df["split"].values == "test"
    y     = (df["true_label"] == "Fractured").astype(int).values

    disagree = df[DISAGREE_COL].values   # std_p already computed in C5.1

    # Tertile thresholds from val
    q33 = float(np.percentile(disagree[val_m], 33.33))
    q67 = float(np.percentile(disagree[val_m], 66.67))
    logger.info(f"Tertile thresholds (val): q33={q33:.4f}  q67={q67:.4f}")

    def _assign_stratum(arr: np.ndarray) -> np.ndarray:
        labels = np.full(len(arr), "Medium", dtype=object)
        labels[arr < q33]  = "Low"
        labels[arr >= q67] = "High"
        return labels

    val_strata = _assign_stratum(disagree[val_m])
    te_strata  = _assign_stratum(disagree[te_m])

    for split_tag, strata in [("Val", val_strata), ("Test", te_strata)]:
        counts = {s: (strata == s).sum() for s in STRATA}
        logger.info(
            f"{split_tag} stratum sizes: "
            + "  ".join(f"{s}={counts[s]}" for s in STRATA)
        )

    # ── Fit meta-learners, collect global + per-stratum metrics ──────────────
    method_records = []   # per-stratum rows → per_method CSV
    global_results = []   # overall val/test metrics per meta-learner

    for cfg in META_CONFIGS:
        tag  = f"{cfg['learner']} ({cfg['tier_label']})"
        cols = cfg["cols"]

        X_tr  = df.loc[tr_m,  cols].values.astype(float)
        X_val = df.loc[val_m, cols].values.astype(float)
        X_te  = df.loc[te_m,  cols].values.astype(float)

        logger.info(f"\n{'─'*50}")
        logger.info(f"{tag}  ({len(cols)} features)")

        clf = _fit_meta(cfg, X_tr, y[tr_m], seed)
        val_scores = clf.predict_proba(X_val)[:, 1]
        te_scores  = clf.predict_proba(X_te)[:, 1]

        # Global full-set evaluation → optimal threshold
        vm  = evaluate(y[val_m], val_scores, "val (full)", logger)
        tm  = evaluate(y[te_m],  te_scores,  "test (full)", logger)
        thr = vm["threshold"]

        global_results.append({
            "learner":       cfg["learner"],
            "tier_label":    cfg["tier_label"],
            "val_f1":        vm["f1"],
            "val_auc":       vm["auc"],
            "val_recall":    vm["recall"],
            "val_precision": vm["precision"],
            "val_threshold": thr,
            "val_accuracy":  vm["accuracy"],
            "val_specificity": vm["specificity"],
            "test_f1":       tm["f1"],
            "test_auc":      tm["auc"],
            "test_recall":   tm["recall"],
            "test_precision":tm["precision"],
            "test_threshold":tm["threshold"],
            "test_accuracy": tm["accuracy"],
            "test_specificity": tm["specificity"],
            "_val_scores":   val_scores,
            "_te_scores":    te_scores,
            "_vm":           vm,
            "_tm":           tm,
        })

        # Per-stratum metrics at the global threshold
        for stratum in STRATA:
            val_mask = val_strata == stratum
            te_mask  = te_strata  == stratum

            vm_s = _eval_at_threshold(y[val_m][val_mask], val_scores[val_mask], thr)
            tm_s = _eval_at_threshold(y[te_m][te_mask],   te_scores[te_mask],   thr)

            logger.info(
                f"  [{stratum:6s}] "
                f"val  n={vm_s['n']:2d}  F1={vm_s['f1']:.4f}  AUC={vm_s['auc']:.4f}  "
                f"ECE={vm_s['ece']:.4f}  |  "
                f"test n={tm_s['n']:2d}  F1={tm_s['f1']:.4f}  AUC={tm_s['auc']:.4f}  "
                f"ECE={tm_s['ece']:.4f}"
            )

            rec = {
                "ea_id":       EA_ID,
                "source":      cfg["ea_id"],
                "learner":     cfg["learner"],
                "tier_label":  cfg["tier_label"],
                "stratum":     stratum,
                "q33":         q33,
                "q67":         q67,
                "global_threshold": thr,
            }
            for split_tag, ms in [("val", vm_s), ("test", tm_s)]:
                for k, v in ms.items():
                    rec[f"{split_tag}_{k}"] = v
            method_records.append(rec)

    # Best method by overall val F1
    best = max(global_results, key=lambda r: r["val_f1"])
    logger.info(
        f"\n{'='*60}\n"
        f"Best overall: {best['learner']} {best['tier_label']}\n"
        f"  val  F1={best['val_f1']:.4f}  AUC={best['val_auc']:.4f}\n"
        f"  test F1={best['test_f1']:.4f}  AUC={best['test_auc']:.4f}"
    )

    # ── Save per-method CSV ───────────────────────────────────────────────────
    pm_dir = root_dir / "results" / "per_method"
    pm_dir.mkdir(parents=True, exist_ok=True)
    csv_cols = [k for k in method_records[0] if not k.startswith("_")]
    pd.DataFrame([{k: r[k] for k in csv_cols} for r in method_records]).to_csv(
        pm_dir / "EA_C5_6_disagreement_stratification.csv", index=False
    )
    logger.info(
        "Per-method table saved → results/per_method/"
        "EA_C5_6_disagreement_stratification.csv"
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Colour assignments
        meta_colors = {"LogReg": "#2196F3", "MLP": "#4CAF50", "GBT": "#FF9800"}
        stratum_span_colors = {
            "Low":    "#BBDEFB",
            "Medium": "#FFF9C4",
            "High":   "#FFCDD2",
        }
        x_strata = np.arange(N_STRATA)
        w = 0.22

        def _bar_annotate(ax, bars, vals, fmt="{:.3f}"):
            for bar, v in zip(bars, vals):
                if not np.isnan(v) and v > 0.01:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        fmt.format(v), ha="center", va="bottom", fontsize=8,
                    )

        # ── Plot 1: Disagreement distribution (val + test) ───────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (split_tag, mask) in zip(
            axes, [("Val", val_m), ("Test", te_m)]
        ):
            dis = disagree[mask]
            counts, _, _ = ax.hist(dis, bins=20, color="#78909C",
                                   alpha=0.75, edgecolor="white")
            ymax = float(counts.max()) * 1.25
            ax.axvline(q33, color="steelblue", linestyle="--",
                       linewidth=1.5, label=f"q33 = {q33:.3f}")
            ax.axvline(q67, color="tomato", linestyle="--",
                       linewidth=1.5, label=f"q67 = {q67:.3f}")
            spans = [
                (float(dis.min()), q33, "Low",    stratum_span_colors["Low"]),
                (q33,              q67, "Medium",  stratum_span_colors["Medium"]),
                (q67, float(dis.max()), "High",   stratum_span_colors["High"]),
            ]
            for lo, hi, s, col in spans:
                ax.axvspan(lo, hi, alpha=0.25, color=col, zorder=0)
                mid = (lo + hi) / 2
                ax.text(mid, ymax * 0.88, s, ha="center", fontsize=9,
                        fontweight="bold")
            ax.set_ylim(0, ymax)
            ax.set_xlabel("σ_p = std(p_ResNet, p_DenseNet, p_Eff)", fontsize=10)
            ax.set_ylabel("Sample count", fontsize=10)
            ax.set_title(f"{split_tag} disagreement distribution", fontsize=11)
            ax.legend(fontsize=9)
        fig.suptitle(
            f"{EA_ID}: Inter-model disagreement (σ_p) distribution "
            "— val tertile boundaries applied to both splits",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_disagreement_distribution.png",
                    dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_disagreement_distribution.png")

        # ── Plot 2: Val F1 by stratum × method ───────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cfg in enumerate(META_CONFIGS):
            lname = cfg["learner"]
            vals  = [
                next(r for r in method_records
                     if r["learner"] == lname and r["stratum"] == s)["val_f1"]
                for s in STRATA
            ]
            bars = ax.bar(x_strata + (i - 1) * w, vals, w,
                          label=lname, color=meta_colors[lname], alpha=0.85)
            _bar_annotate(ax, bars, vals)
        ax.set_xticks(x_strata)
        ax.set_xticklabels(STRATA)
        ax.set_xlabel("Disagreement stratum (σ_p tertile)", fontsize=10)
        ax.set_ylabel("F1 at global-optimal threshold", fontsize=10)
        ax.set_title(f"{EA_ID}: Val F1 by disagreement stratum", fontsize=11)
        ax.legend(fontsize=9)
        all_f1v = [r["val_f1"] for r in method_records
                   if not np.isnan(r["val_f1"])]
        if all_f1v:
            ax.set_ylim(max(0, min(all_f1v) - 0.08),
                        min(1.0, max(all_f1v) + 0.10))
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_val_f1_by_stratum.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_val_f1_by_stratum.png")

        # ── Plot 3: Test F1 by stratum × method ──────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cfg in enumerate(META_CONFIGS):
            lname = cfg["learner"]
            vals  = [
                next(r for r in method_records
                     if r["learner"] == lname and r["stratum"] == s)["test_f1"]
                for s in STRATA
            ]
            bars = ax.bar(x_strata + (i - 1) * w, vals, w,
                          label=lname, color=meta_colors[lname], alpha=0.85)
            _bar_annotate(ax, bars, vals)
        ax.set_xticks(x_strata)
        ax.set_xticklabels(STRATA)
        ax.set_xlabel("Disagreement stratum (σ_p tertile)", fontsize=10)
        ax.set_ylabel("F1 at global-optimal threshold", fontsize=10)
        ax.set_title(
            f"{EA_ID}: Test F1 by disagreement stratum (honest)", fontsize=11
        )
        ax.legend(fontsize=9)
        all_f1t = [r["test_f1"] for r in method_records
                   if not np.isnan(r["test_f1"])]
        if all_f1t:
            ax.set_ylim(max(0, min(all_f1t) - 0.08),
                        min(1.0, max(all_f1t) + 0.10))
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_test_f1_by_stratum.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_test_f1_by_stratum.png")

        # ── Plot 4: Val AUC by stratum × method ──────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cfg in enumerate(META_CONFIGS):
            lname = cfg["learner"]
            vals  = [
                next(r for r in method_records
                     if r["learner"] == lname and r["stratum"] == s)["val_auc"]
                for s in STRATA
            ]
            bars = ax.bar(x_strata + (i - 1) * w, vals, w,
                          label=lname, color=meta_colors[lname], alpha=0.85)
            _bar_annotate(ax, bars, vals)
        ax.set_xticks(x_strata)
        ax.set_xticklabels(STRATA)
        ax.set_xlabel("Disagreement stratum (σ_p tertile)", fontsize=10)
        ax.set_ylabel("AUC-ROC", fontsize=10)
        ax.set_title(f"{EA_ID}: Val AUC by disagreement stratum", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.08)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_val_auc_by_stratum.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_val_auc_by_stratum.png")

        # ── Plot 5: ECE by stratum × method (val) ────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cfg in enumerate(META_CONFIGS):
            lname = cfg["learner"]
            vals  = [
                next(r for r in method_records
                     if r["learner"] == lname and r["stratum"] == s)["val_ece"]
                for s in STRATA
            ]
            bars = ax.bar(x_strata + (i - 1) * w, vals, w,
                          label=lname, color=meta_colors[lname], alpha=0.85)
            _bar_annotate(ax, bars, vals)
        ax.set_xticks(x_strata)
        ax.set_xticklabels(STRATA)
        ax.set_xlabel("Disagreement stratum (σ_p tertile)", fontsize=10)
        ax.set_ylabel("ECE (↓ = better calibrated)", fontsize=10)
        ax.set_title(f"{EA_ID}: Val ECE by disagreement stratum", fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_val_ece_by_stratum.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_val_ece_by_stratum.png")

        # ── Plot 6: Heatmap — val F1 (stratum × meta-learner) ────────────────
        learner_labels = [c["learner"] for c in META_CONFIGS]
        f1_matrix = np.array(
            [
                [
                    next(r for r in method_records
                         if r["learner"] == cfg["learner"]
                         and r["stratum"] == s)["val_f1"]
                    for cfg in META_CONFIGS
                ]
                for s in STRATA
            ],
            dtype=float,
        )
        annot = np.where(
            np.isnan(f1_matrix),
            "n/a",
            np.array([[f"{v:.3f}" for v in row] for row in f1_matrix]),
        )
        valid_vals = f1_matrix[~np.isnan(f1_matrix)]
        vmin = float(valid_vals.min()) - 0.05 if len(valid_vals) else 0.5
        vmax = float(valid_vals.max()) + 0.02 if len(valid_vals) else 1.0

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            f1_matrix, annot=annot, fmt="", cmap="YlGn",
            xticklabels=learner_labels, yticklabels=STRATA,
            ax=ax, linewidths=0.5, vmin=vmin, vmax=vmax,
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax.set_xlabel("Meta-learner", fontsize=10)
        ax.set_ylabel("Disagreement stratum", fontsize=10)
        ax.set_title(
            f"{EA_ID}: Val F1 heatmap (stratum × meta-learner)\n"
            f"Global threshold applied per-stratum  |  q33={q33:.3f}  q67={q67:.3f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_f1_heatmap.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_f1_heatmap.png")

        # ── Plot 7: Stratum sample counts (val + test) ────────────────────────
        val_counts  = np.array([(val_strata == s).sum() for s in STRATA])
        test_counts = np.array([(te_strata  == s).sum() for s in STRATA])
        x2 = np.arange(N_STRATA)
        fig, ax = plt.subplots(figsize=(7, 4))
        b1 = ax.bar(x2 - 0.2, val_counts,  0.38,
                    label="Val",  color="#42A5F5", alpha=0.85)
        b2 = ax.bar(x2 + 0.2, test_counts, 0.38,
                    label="Test", color="#EF5350", alpha=0.85)
        for bar, v in list(zip(b1, val_counts)) + list(zip(b2, test_counts)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(v), ha="center", va="bottom", fontsize=11)
        ax.set_xticks(x2)
        ax.set_xticklabels(STRATA)
        ax.set_xlabel("Disagreement stratum", fontsize=10)
        ax.set_ylabel("Sample count", fontsize=10)
        ax.set_title(
            f"{EA_ID}: Stratum sizes\n"
            f"q33={q33:.3f}  q67={q67:.3f}  (thresholds from val)",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_stratum_counts.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_stratum_counts.png")

        # ── Plot 8+9: Confusion matrix + confidence-accuracy (best, val) ─────
        plot_confusion_matrix(
            y[val_m], best["_val_scores"], best["_vm"]["threshold"],
            EA_ID,
            f"{METHOD}\n(best: {best['learner']} {best['tier_label']})",
            plot_dir, logger,
        )
        plot_confidence_accuracy(
            y[val_m], best["_val_scores"], best["_vm"]["threshold"],
            EA_ID,
            f"{METHOD}\n(best: {best['learner']} {best['tier_label']})",
            plot_dir, logger,
        )

    # ── Append best to EA tables ─────────────────────────────────────────────
    row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "val_f1":        best["val_f1"],
        "val_auc":       best["val_auc"],
        "val_recall":    best["val_recall"],
        "val_precision": best["val_precision"],
        "val_threshold": best["val_threshold"],
        "test_f1":       best["test_f1"],
        "test_auc":      best["test_auc"],
        "test_recall":   best["test_recall"],
        "test_precision":best["test_precision"],
        "test_threshold":best["test_threshold"],
        "reference": REF,
        "notes": (
            f"Disagreement stratification: σ_p = std(3 CNN probs), "
            f"tertile thresholds q33={q33:.4f} q67={q67:.4f} from val. "
            f"3 strata: Low/Medium/High.  Fixed global threshold per method. "
            f"Best meta-learner: {best['learner']} {best['tier_label']} "
            f"(val F1={best['val_f1']:.4f} / test F1={best['test_f1']:.4f}). "
            f"Primary contribution: per-stratum breakdown in per_method CSV+plots."
        ),
    }
    append_to_table(row)
    logger.info("Best row appended to EA_comparative_table.csv")

    results_row = {
        "ea_id": EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "val_f1":           best["val_f1"],
        "val_auc":          best["val_auc"],
        "val_acc":          best["val_accuracy"],
        "val_recall":       best["val_recall"],
        "val_specificity":  best["val_specificity"],
        "val_threshold":    best["val_threshold"],
        "test_f1":          best["test_f1"],
        "test_auc":         best["test_auc"],
        "test_acc":         best["test_accuracy"],
        "test_recall":      best["test_recall"],
        "test_specificity": best["test_specificity"],
        "test_threshold":   best["test_threshold"],
    }
    append_to_results(results_row)
    logger.info("Best row appended to EA_results.csv")

    return row


if __name__ == "__main__":
    THIS_DIR = Path(__file__).resolve().parent
    ROOT_DIR = THIS_DIR.parent
    run(ROOT_DIR)
