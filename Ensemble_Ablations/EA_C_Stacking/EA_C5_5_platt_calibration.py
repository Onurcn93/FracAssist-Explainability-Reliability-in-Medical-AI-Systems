"""
EA-C5.5: Feature-engineered stacking — Platt scaling calibration.

Applies Platt scaling (sigmoid calibration) post-hoc to the best
(meta-learner, tier) configuration from each of C5.2 / C5.3 / C5.4:
  - LogReg  T1+T2       (best of C5.2, val F1=0.7368, C=0.001)
  - MLP     T1+..+T6    (best of C5.3, val F1=0.7421, arch=(64,) alpha=1.0)
  - GBT     T1+T2+T3    (best of C5.4, val F1=0.7229, d=3 lr=0.30 sub=0.8 n=3)

Calibration protocol:
  1. Refit meta-learner on train (identical hyperparameters to source C5.x).
  2. Fit sigmoid  p_calib = σ(A·score + B)  on val labels via
     CalibratedClassifierCV(cv='prefit', method='sigmoid').
  3. Report ECE (Expected Calibration Error) before/after on val AND test.
  4. Reliability diagrams (probability vs actual fraction positive) before/after
     for val (all methods) and test (all methods — honest).
  5. F1/AUC comparison before/after for val and test.

Data honesty note:
  Platt is fit on the same val set used for meta-learner selection, so val
  Platt metrics are inflated by construction.  Test metrics are fully honest.
  Primary result is test ECE before→after.  Val F1 is reported only for
  consistency with the EA comparative table convention.

ECE = Σ_b (|B_b| / n) × |mean_confidence(B_b) − fraction_positive(B_b)|

Best C5.5 variant: selected by val F1 after Platt (consistent with series).
Notes in EA table flag the val-inflation caveat.

References:
  Platt 1999 (sigmoid calibration)
  Niculescu-Mizil & Caruana 2005 (reliability diagrams)
  Guo et al. 2017 (ECE for neural network calibration)
  Base C5 meta-learners: this thesis (EA-C5.2 / C5.3 / C5.4)
"""

import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

THIS_DIR = Path(__file__).resolve().parent
EA_ROOT  = THIS_DIR.parent
sys.path.insert(0, str(EA_ROOT))

from common.eval import (get_logger, evaluate, threshold_sweep,
                         append_to_table, append_to_results,
                         plot_confusion_matrix, plot_confidence_accuracy)
from EA_C_Stacking.C5_1_build_features import TIER_COLS   # noqa: E402

EA_ID  = "EA-C5.5"
METHOD = "Feature-engineered stacking — Platt scaling calibration"
FAMILY = "C"
TYPE   = "ORIG"
REF    = ("Platt 1999 (sigmoid calibration); "
          "Niculescu-Mizil & Caruana 2005 (reliability); "
          "Guo et al. 2017 (ECE); Base meta-learners: this thesis")

DATA_CSV = "data/all_c5.csv"
N_BINS   = 10

# ── Best configs from C5.2 / C5.3 / C5.4 (hyperparameters hardcoded) ────────
_T = TIER_COLS
META_CONFIGS = [
    {
        "source":     "EA-C5.2",
        "learner":    "LogReg",
        "tier_label": "T1+T2",
        "cols":       _T["T1"] + _T["T2"],
        "logreg_c":   0.001,
        "val_f1_ref": 0.7368,
        "val_auc_ref": 0.9037,
    },
    {
        "source":     "EA-C5.3",
        "learner":    "MLP",
        "tier_label": "T1+..+T6",
        "cols":       [c for t in ["T1", "T2", "T3", "T4", "T5", "T6"] for c in _T[t]],
        "arch":       (64,),
        "alpha":      1.0,
        "val_f1_ref": 0.7421,
        "val_auc_ref": 0.8497,
    },
    {
        "source":     "EA-C5.4",
        "learner":    "GBT",
        "tier_label": "T1+T2+T3",
        "cols":       _T["T1"] + _T["T2"] + _T["T3"],
        "depth":      3,
        "lr":         0.30,
        "subsample":  0.8,
        "n_trees":    3,
        "val_f1_ref": 0.7229,
        "val_auc_ref": 0.8525,
    },
]


# ── ECE ───────────────────────────────────────────────────────────────────────

def _compute_ece(y_true: np.ndarray, scores: np.ndarray,
                 n_bins: int = N_BINS) -> float:
    """Expected Calibration Error: Σ_b (|B_b|/n) × |conf(B_b) − acc(B_b)|."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        conf = float(scores[mask].mean())
        acc  = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


# ── Reliability diagram ───────────────────────────────────────────────────────

def _reliability_diagram(ax, y_true: np.ndarray, scores: np.ndarray,
                          title: str, n_bins: int = N_BINS) -> float:
    """Draw reliability diagram on ax; return ECE."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf, mean_acc = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            mean_conf.append((lo + hi) / 2)
            mean_acc.append(np.nan)
        else:
            mean_conf.append(float(scores[mask].mean()))
            mean_acc.append(float(y_true[mask].mean()))

    ece = _compute_ece(y_true, scores, n_bins)

    # Gap shading
    valid = [(c, a) for c, a in zip(mean_conf, mean_acc) if not np.isnan(a)]
    if valid:
        vc, va = zip(*valid)
        ax.fill_between(vc, va, vc, alpha=0.15, color="tomato", label="Gap")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Perfect")
    ax.plot(mean_conf, mean_acc, "o-", color="steelblue",
            linewidth=1.8, markersize=5, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_ylabel("Fraction of positives", fontsize=9)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=9)
    ax.legend(fontsize=8)
    return ece


# ── Base meta-learner fit ─────────────────────────────────────────────────────

def _fit_base(cfg: dict, X_tr, y_tr, seed: int):
    """Refit base meta-learner with identical hyperparameters as C5.x source."""
    learner = cfg["learner"]
    if learner == "LogReg":
        clf = LogisticRegression(
            C=cfg["logreg_c"], max_iter=1000, random_state=seed, solver="lbfgs"
        )
        clf.fit(X_tr, y_tr)
    elif learner == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=cfg["arch"],
            activation="relu",
            alpha=cfg["alpha"],
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=seed,
            solver="adam",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
    elif learner == "GBT":
        clf = GradientBoostingClassifier(
            n_estimators=cfg["n_trees"],
            max_depth=cfg["depth"],
            learning_rate=cfg["lr"],
            subsample=cfg["subsample"],
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)
    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

def run(root_dir: Path, seed: int = 42, plot: bool = True) -> dict:
    log_dir  = root_dir / "logs"
    plot_dir = root_dir / "plots" / EA_ID
    logger   = get_logger(EA_ID, log_dir)

    logger.info("=" * 60)
    logger.info(f"{EA_ID}: {METHOD}")
    logger.info("=" * 60)
    logger.info("Applying Platt scaling to best C5.2/C5.3/C5.4 configs.")
    logger.info("NOTE: val Platt metrics are overfit (Platt fit on val).")
    logger.info("      Test ECE is the primary honest result.")

    df = pd.read_csv(root_dir / DATA_CSV)
    logger.info(f"Loaded {len(df)} rows from {DATA_CSV}")

    tr_m  = df["split"].values == "train"
    val_m = df["split"].values == "val"
    te_m  = df["split"].values == "test"
    y     = (df["true_label"] == "Fractured").astype(int).values

    logger.info(f"Train: {tr_m.sum()} | Val: {val_m.sum()} | Test: {te_m.sum()}")

    results = []   # one dict per meta-learner

    for cfg in META_CONFIGS:
        tag    = f"{cfg['learner']} ({cfg['tier_label']})"
        cols   = cfg["cols"]
        n_feat = len(cols)

        X_tr  = df.loc[tr_m,  cols].values.astype(float)
        X_val = df.loc[val_m, cols].values.astype(float)
        X_te  = df.loc[te_m,  cols].values.astype(float)

        logger.info(f"\n{'─'*50}")
        logger.info(f"{tag}  ({n_feat} features)")
        logger.info(f"Source: {cfg['source']}  "
                    f"ref val F1={cfg['val_f1_ref']:.4f}  "
                    f"ref val AUC={cfg['val_auc_ref']:.4f}")
        logger.info(f"{'─'*50}")

        # 1. Fit base meta-learner (identical to source C5.x)
        clf = _fit_base(cfg, X_tr, y[tr_m], seed)

        raw_val  = clf.predict_proba(X_val)[:, 1]
        raw_test = clf.predict_proba(X_te)[:, 1]

        logger.info("  [raw scores]")
        vm_raw  = evaluate(y[val_m], raw_val,  "val",  logger)
        tm_raw  = evaluate(y[te_m],  raw_test, "test", logger)

        # 2. Platt scaling: sigmoid fit on val labels
        cal = CalibratedClassifierCV(estimator=FrozenEstimator(clf), method="sigmoid")
        cal.fit(X_val, y[val_m])

        platt_val  = cal.predict_proba(X_val)[:, 1]
        platt_test = cal.predict_proba(X_te)[:, 1]

        logger.info("  [Platt-calibrated scores]")
        vm_platt  = evaluate(y[val_m], platt_val,  "val",  logger)
        tm_platt  = evaluate(y[te_m],  platt_test, "test", logger)

        # 3. ECE before / after
        ece_rv  = _compute_ece(y[val_m], raw_val)
        ece_rt  = _compute_ece(y[te_m],  raw_test)
        ece_pv  = _compute_ece(y[val_m], platt_val)
        ece_pt  = _compute_ece(y[te_m],  platt_test)

        logger.info(
            f"  ECE  raw  → val={ece_rv:.4f}  test={ece_rt:.4f}\n"
            f"  ECE  platt→ val={ece_pv:.4f}  test={ece_pt:.4f}  "
            f"(ΔECE test={ece_pt - ece_rt:+.4f})"
        )

        results.append({
            "source":             cfg["source"],
            "learner":            cfg["learner"],
            "tier_label":         cfg["tier_label"],
            "n_features":         n_feat,
            # raw
            "raw_val_f1":         vm_raw["f1"],
            "raw_val_auc":        vm_raw["auc"],
            "raw_val_recall":     vm_raw["recall"],
            "raw_val_precision":  vm_raw["precision"],
            "raw_val_threshold":  vm_raw["threshold"],
            "raw_test_f1":        tm_raw["f1"],
            "raw_test_auc":       tm_raw["auc"],
            "raw_test_recall":    tm_raw["recall"],
            "raw_test_precision": tm_raw["precision"],
            "raw_test_threshold": tm_raw["threshold"],
            "ece_raw_val":        ece_rv,
            "ece_raw_test":       ece_rt,
            # platt
            "platt_val_f1":         vm_platt["f1"],
            "platt_val_auc":        vm_platt["auc"],
            "platt_val_recall":     vm_platt["recall"],
            "platt_val_precision":  vm_platt["precision"],
            "platt_val_threshold":  vm_platt["threshold"],
            "platt_test_f1":        tm_platt["f1"],
            "platt_test_auc":       tm_platt["auc"],
            "platt_test_recall":    tm_platt["recall"],
            "platt_test_precision": tm_platt["precision"],
            "platt_test_threshold": tm_platt["threshold"],
            "ece_platt_val":        ece_pv,
            "ece_platt_test":       ece_pt,
            "delta_ece_test":       ece_pt - ece_rt,
            # stash for plots
            "_raw_val":    raw_val,
            "_raw_test":   raw_test,
            "_platt_val":  platt_val,
            "_platt_test": platt_test,
            "_y_val":      y[val_m],
            "_y_test":     y[te_m],
            "_vm_platt":   vm_platt,
            "_tm_platt":   tm_platt,
        })

    # Best: highest val F1 after Platt (consistent with EA series convention)
    best = max(results, key=lambda r: r["platt_val_f1"])
    logger.info(
        f"\n{'='*60}\n"
        f"Best after Platt: {best['learner']} {best['tier_label']}\n"
        f"  val  F1={best['platt_val_f1']:.4f}  AUC={best['platt_val_auc']:.4f}  "
        f"ECE={best['ece_platt_val']:.4f}\n"
        f"  test F1={best['platt_test_f1']:.4f}  AUC={best['platt_test_auc']:.4f}  "
        f"ECE={best['ece_platt_test']:.4f}  "
        f"(raw test ECE was {best['ece_raw_test']:.4f})"
    )

    # ── Save per-method CSV ──────────────────────────────────────────────────
    pm_dir = root_dir / "results" / "per_method"
    pm_dir.mkdir(parents=True, exist_ok=True)
    save_cols = [k for k in results[0] if not k.startswith("_")]
    pd.DataFrame([{k: r[k] for k in save_cols} for r in results]).to_csv(
        pm_dir / "EA_C5_5_platt_calibration.csv", index=False
    )
    logger.info("Per-method table saved → results/per_method/EA_C5_5_platt_calibration.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)
        n_methods = len(results)
        labels = [f"{r['learner']}\n{r['tier_label']}" for r in results]
        x = np.arange(n_methods)
        w = 0.2

        # ── Plot 1: ECE bar chart ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - 1.5*w, [r["ece_raw_val"]   for r in results], w,
                    label="Raw val",    color="#2196F3", alpha=0.85)
        b2 = ax.bar(x - 0.5*w, [r["ece_platt_val"] for r in results], w,
                    label="Platt val*", color="#4CAF50", alpha=0.85)
        b3 = ax.bar(x + 0.5*w, [r["ece_raw_test"]  for r in results], w,
                    label="Raw test",   color="#FF9800", alpha=0.85)
        b4 = ax.bar(x + 1.5*w, [r["ece_platt_test"]for r in results], w,
                    label="Platt test", color="#E91E63", alpha=0.85)
        for bars in [b1, b2, b3, b4]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("ECE (↓ = better calibrated)")
        ax.set_ylim(0, max(
            r["ece_raw_val"] for r in results
        ) * 1.40)
        ax.set_title(
            f"{EA_ID}: Expected Calibration Error before/after Platt scaling\n"
            "* val Platt is overfit (Platt fit on val); test is honest"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_ece_comparison.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_ece_comparison.png")

        # ── Plot 2: Reliability diagrams — VAL (2 rows × n_methods cols) ─────
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 9))
        for col, r in enumerate(results):
            label = f"{r['learner']} {r['tier_label']}"
            _reliability_diagram(axes[0, col], r["_y_val"], r["_raw_val"],
                                 f"Raw val\n{label}")
            _reliability_diagram(axes[1, col], r["_y_val"], r["_platt_val"],
                                 f"Platt val*\n{label}")
        fig.suptitle(
            f"{EA_ID}: Val reliability diagrams — raw vs Platt\n"
            "Row 1: raw scores  |  Row 2: Platt-calibrated  (* overfit to val)",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_reliability_val.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_reliability_val.png")

        # ── Plot 3: Reliability diagrams — TEST (honest, 2 rows × n_methods) ─
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 9))
        for col, r in enumerate(results):
            label = f"{r['learner']} {r['tier_label']}"
            _reliability_diagram(axes[0, col], r["_y_test"], r["_raw_test"],
                                 f"Raw test\n{label}")
            _reliability_diagram(axes[1, col], r["_y_test"], r["_platt_test"],
                                 f"Platt test\n{label}")
        fig.suptitle(
            f"{EA_ID}: Test reliability diagrams (honest)\n"
            "Row 1: raw scores  |  Row 2: Platt-calibrated",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_reliability_test.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_reliability_test.png")

        # ── Plot 4: F1 performance comparison ────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - 1.5*w, [r["raw_val_f1"]   for r in results], w,
                    label="Raw val",    color="#2196F3", alpha=0.85)
        b2 = ax.bar(x - 0.5*w, [r["platt_val_f1"] for r in results], w,
                    label="Platt val*", color="#4CAF50", alpha=0.85)
        b3 = ax.bar(x + 0.5*w, [r["raw_test_f1"]  for r in results], w,
                    label="Raw test",   color="#FF9800", alpha=0.85)
        b4 = ax.bar(x + 1.5*w, [r["platt_test_f1"]for r in results], w,
                    label="Platt test", color="#E91E63", alpha=0.85)
        for bars in [b1, b2, b3, b4]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                            f"{h:.4f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("F1 (threshold-swept)")
        all_f1 = ([r["raw_val_f1"] for r in results] +
                  [r["platt_val_f1"] for r in results] +
                  [r["raw_test_f1"] for r in results] +
                  [r["platt_test_f1"] for r in results])
        ax.set_ylim(max(0, min(all_f1) - 0.04), min(1.0, max(all_f1) + 0.07))
        ax.set_title(
            f"{EA_ID}: F1 before/after Platt scaling\n"
            f"Best (test): {best['learner']} {best['tier_label']} "
            f"F1={best['platt_test_f1']:.4f}  |  (* val is overfit)"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{EA_ID}_f1_comparison.png", dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → plots/{EA_ID}/{EA_ID}_f1_comparison.png")

        # ── Plot 5 & 6: confusion matrix + confidence-accuracy (best Platt) ──
        plot_confusion_matrix(
            best["_y_val"], best["_platt_val"], best["_vm_platt"]["threshold"],
            EA_ID,
            f"{METHOD}\n(best: {best['learner']} {best['tier_label']} — Platt)",
            plot_dir, logger,
        )
        plot_confidence_accuracy(
            best["_y_val"], best["_platt_val"], best["_vm_platt"]["threshold"],
            EA_ID,
            f"{METHOD}\n(best: {best['learner']} {best['tier_label']} — Platt)",
            plot_dir, logger,
        )

    # ── Append best to EA tables ─────────────────────────────────────────────
    row = {
        "ea_id":  EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "val_f1":        best["platt_val_f1"],
        "val_auc":       best["platt_val_auc"],
        "val_recall":    best["platt_val_recall"],
        "val_precision": best["platt_val_precision"],
        "val_threshold": best["platt_val_threshold"],
        "test_f1":       best["platt_test_f1"],
        "test_auc":      best["platt_test_auc"],
        "test_recall":   best["platt_test_recall"],
        "test_precision":best["platt_test_precision"],
        "test_threshold":best["platt_test_threshold"],
        "reference": REF,
        "notes": (
            f"Platt scaling on best C5 configs: LogReg T1+T2, MLP T1+..+T6, "
            f"GBT T1+T2+T3. CalibratedClassifierCV cv=prefit method=sigmoid fit on val. "
            f"Best by val F1: {best['learner']} {best['tier_label']} "
            f"(val F1={best['platt_val_f1']:.4f} — INFLATED, fit on val). "
            f"Honest test ECE: {best['ece_raw_test']:.4f} → {best['ece_platt_test']:.4f} "
            f"(ΔECE={best['delta_ece_test']:+.4f}). "
            f"Test F1 raw/platt: {best['raw_test_f1']:.4f}/{best['platt_test_f1']:.4f}. "
            f"Primary result = test ECE improvement."
        ),
    }
    append_to_table(row)
    logger.info("Best-Platt row appended to EA_comparative_table.csv")

    results_row = {
        "ea_id":  EA_ID, "method": METHOD, "family": FAMILY, "type": TYPE,
        "val_f1":           best["platt_val_f1"],
        "val_auc":          best["platt_val_auc"],
        "val_acc":          best["_vm_platt"]["accuracy"],
        "val_recall":       best["platt_val_recall"],
        "val_specificity":  best["_vm_platt"]["specificity"],
        "val_threshold":    best["platt_val_threshold"],
        "test_f1":          best["platt_test_f1"],
        "test_auc":         best["platt_test_auc"],
        "test_acc":         best["_tm_platt"]["accuracy"],
        "test_recall":      best["platt_test_recall"],
        "test_specificity": best["_tm_platt"]["specificity"],
        "test_threshold":   best["platt_test_threshold"],
    }
    append_to_results(results_row)
    logger.info("Best-Platt row appended to EA_results.csv")

    return row


if __name__ == "__main__":
    THIS_DIR = Path(__file__).resolve().parent
    ROOT_DIR = THIS_DIR.parent
    run(ROOT_DIR)
