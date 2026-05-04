"""
utils/tune_gel.py — GEL hyperparameter tuning from pre-computed predictions.

Reads review/all.csv (pre-computed per-model probabilities for every image).
No model loading or GPU required — pure numpy/sklearn on cached outputs.

Modes
-----
  gamma   Sweep gel_gamma 1–15; report val AUC, val F1, test AUC, test F1
          for each value. Identifies the gamma that maximises val F1.

  oam     Sweep gel_disagree_lim 0.05–0.80 step 0.05 (gamma fixed at current
          GEL_CONFIG value, penalties unchanged). Reports val AUC, val F1, OAM trigger rate.

  grid    Joint grid search over gamma (step 0.1) × delta (step 0.01).
          Produces 3D surface plots for val AUC and val F1. Reports the
          (gamma, delta) pair that maximises each metric plus test metrics
          for those two optimal configs.

Usage
-----
    python utils/tune_gel.py                        # gamma sweep (default)
    python utils/tune_gel.py --mode gamma
    python utils/tune_gel.py --mode gamma --step 0.5
    python utils/tune_gel.py --mode oam
    python utils/tune_gel.py --mode oam --step 0.05
    python utils/tune_gel.py --mode grid
    python utils/tune_gel.py --mode grid --gamma-step 0.5 --delta-step 0.05
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

LOG_DIR   = Path("results/logs")
PLOTS_DIR = Path("results/plots")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gel.gel_config import GEL_CONFIG    # noqa: E402
from gel.gel_pipeline import apply_gel   # noqa: E402

ALL_CSV = Path("review/all.csv")
FRAC_LABEL = "Fractured"


# ── Data loading ──────────────────────────────────────────────────────────── #

def _load_split(df, split):
    sub = df[df["split"] == split].reset_index(drop=True)
    labels = (sub["true_label"] == FRAC_LABEL).astype(int).values
    p_r = sub["resnet_probability"].astype(float).values
    p_d = sub["densenet_probability"].astype(float).values
    p_e = sub["efficientnet_probability"].astype(float).values
    return labels, p_r, p_d, p_e


# ── Metric helpers ────────────────────────────────────────────────────────── #

def _sweep_f1(labels, scores):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.025):
        f1 = f1_score(labels, (scores >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _f1_at(labels, scores, threshold):
    return f1_score(labels, (scores >= threshold).astype(int), pos_label=1, zero_division=0)


def _auc(labels, scores):
    return roc_auc_score(labels, scores)


def _oam_rates(p_r, p_d, p_e, delta):
    mu   = (p_r + p_d + p_e) / 3.0
    r    = (np.abs(p_r - mu) > delta).mean() * 100
    d    = (np.abs(p_d - mu) > delta).mean() * 100
    e    = (np.abs(p_e - mu) > delta).mean() * 100
    any_ = ((np.abs(p_r - mu) > delta) | (np.abs(p_d - mu) > delta) |
            (np.abs(p_e - mu) > delta)).mean() * 100
    return r, d, e, any_


# ── Plotting ─────────────────────────────────────────────────────────────── #

def _plot_gamma(results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    gammas   = [r["gamma"]    for r in results]
    val_auc  = [r["val_auc"]  for r in results]
    val_f1   = [r["val_f1"]   for r in results]
    test_auc = [r["test_auc"] for r in results]
    test_f1  = [r["test_f1"]  for r in results]

    best_vauc = max(results, key=lambda r: r["val_auc"])
    best_vf1  = max(results, key=lambda r: r["val_f1"])
    xticks    = gammas if len(gammas) <= 30 else np.arange(1, 16)
    subtitle  = (f"RC$_i$ = (F1$_i$ + {GEL_CONFIG['gel_base_shift']})$^γ$"
                 f" / Σ(F1$_j$ + {GEL_CONFIG['gel_base_shift']})$^γ$")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gammas, val_auc,  color="#4C72B0", linewidth=2, label="Val AUC")
    ax.plot(gammas, test_auc, color="#DD8452", linewidth=2, label="Test AUC")
    ax.axvline(best_vauc["gamma"], color="grey", linestyle=":", linewidth=1.2,
               label=f"Best val AUC (γ={best_vauc['gamma']:.1f})")
    ax.set_xlabel("Gamma (γ)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(f"GEL Expert-Dominant RC — Gamma Sweep — AUC\n{subtitle}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xticks)
    out_auc = PLOTS_DIR / "gel_gamma_sweep_auc.png"
    fig.tight_layout()
    fig.savefig(out_auc, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_auc}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gammas, val_f1,  color="#4C72B0", linewidth=2, label="Val F1")
    ax.plot(gammas, test_f1, color="#DD8452", linewidth=2, label="Test F1")
    ax.axvline(best_vf1["gamma"], color="grey", linestyle=":", linewidth=1.2,
               label=f"Best val F1 (γ={best_vf1['gamma']:.1f})")
    ax.set_xlabel("Gamma (γ)", fontsize=12)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title(f"GEL Expert-Dominant RC — Gamma Sweep — F1\n{subtitle}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xticks)
    out_f1 = PLOTS_DIR / "gel_gamma_sweep_f1.png"
    fig.tight_layout()
    fig.savefig(out_f1, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_f1}")


def _plot_oam(results, gamma, k_low, k_high):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    deltas   = [r["delta"]    for r in results]
    val_auc  = [r["val_auc"]  for r in results]
    val_f1   = [r["val_f1"]   for r in results]
    test_auc = [r["test_auc"] for r in results]
    test_f1  = [r["test_f1"]  for r in results]

    best_vauc = max(results, key=lambda r: r["val_auc"])
    best_vf1  = max(results, key=lambda r: r["val_f1"])
    xticks    = deltas if len(deltas) <= 30 else np.arange(0.05, 0.85, 0.10)
    subtitle  = (f"γ={gamma:.0f}, k_low={k_low}, k_high={k_high}"
                 f"  |  Trigger: |P$_i$ − μ| > δ  →  RC$_i$ × k")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(deltas, val_auc,  color="#4C72B0", linewidth=2, label="Val AUC")
    ax.plot(deltas, test_auc, color="#DD8452", linewidth=2, label="Test AUC")
    ax.axvline(best_vauc["delta"], color="grey", linestyle=":", linewidth=1.2,
               label=f"Best val AUC (δ={best_vauc['delta']:.2f})")
    ax.set_xlabel("OAM Disagree Limit (δ)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(f"GEL OAM Sweep — AUC\n{subtitle}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xticks)
    out_auc = PLOTS_DIR / "gel_oam_sweep_auc.png"
    fig.tight_layout()
    fig.savefig(out_auc, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_auc}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(deltas, val_f1,  color="#4C72B0", linewidth=2, label="Val F1")
    ax.plot(deltas, test_f1, color="#DD8452", linewidth=2, label="Test F1")
    ax.axvline(best_vf1["delta"], color="grey", linestyle=":", linewidth=1.2,
               label=f"Best val F1 (δ={best_vf1['delta']:.2f})")
    ax.set_xlabel("OAM Disagree Limit (δ)", fontsize=12)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title(f"GEL OAM Sweep — F1\n{subtitle}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xticks)
    out_f1 = PLOTS_DIR / "gel_oam_sweep_f1.png"
    fig.tight_layout()
    fig.savefig(out_f1, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_f1}")


def _plot_grid(G, D, val_auc_grid, val_f1_grid):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    best_auc_idx = np.unravel_index(np.argmax(val_auc_grid), val_auc_grid.shape)
    best_f1_idx  = np.unravel_index(np.argmax(val_f1_grid),  val_f1_grid.shape)

    for grid, label, cmap, best_idx, out_name in [
        (val_auc_grid, "Val AUC", "viridis", best_auc_idx, "gel_grid_val_auc.png"),
        (val_f1_grid,  "Val F1",  "plasma",  best_f1_idx,  "gel_grid_val_f1.png"),
    ]:
        fig = plt.figure(figsize=(13, 7))
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot_surface(G, D, grid, cmap=cmap, alpha=0.88, linewidth=0)

        bx = G[best_idx]
        by = D[best_idx]
        bz = grid[best_idx]
        ax.scatter([bx], [by], [bz], color="red", s=60, zorder=5,
                   label=f"Best (γ={bx:.1f}, δ={by:.2f}) = {bz:.4f}")

        ax.set_xlabel("Gamma (γ)", fontsize=11, labelpad=8)
        ax.set_ylabel("Disagree Limit (δ)", fontsize=11, labelpad=8)
        ax.set_zlabel(label, fontsize=11, labelpad=8)
        ax.set_title(f"GEL Grid Search — {label}\nγ × δ joint optimisation", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")

        out = PLOTS_DIR / out_name
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Plot saved → {out}")


# ── Modes ─────────────────────────────────────────────────────────────────── #

def mode_gamma(df, step):
    val_labels,  vr, vd, ve = _load_split(df, "val")
    test_labels, tr, td, te = _load_split(df, "test")

    gammas = np.arange(1.0, 15.0 + step / 2, step)

    print(f"\n{'=' * 88}")
    print(f"  GEL Gamma Sweep  |  gamma {gammas[0]:.1f} → {gammas[-1]:.1f}  step={step}"
          f"  |  base_shift={GEL_CONFIG['gel_base_shift']}")
    print(f"{'=' * 88}")
    print(f"  {'gamma':>6}  {'val_AUC':>8}  {'val_F1':>7}  {'val_thr':>7}"
          f"  {'test_AUC':>9}  {'test_F1':>8}  {'test_F1(val-thr)':>16}")
    print(f"  {'-' * 80}")

    results = []
    for g in gammas:
        GEL_CONFIG["gel_gamma"] = float(g)
        v_final, _ = apply_gel(vr, vd, ve)
        t_final, _ = apply_gel(tr, td, te)

        v_auc        = _auc(val_labels, v_final)
        v_thr, v_f1  = _sweep_f1(val_labels, v_final)
        t_auc        = _auc(test_labels, t_final)
        _, t_f1      = _sweep_f1(test_labels, t_final)
        t_f1_valthr  = _f1_at(test_labels, t_final, v_thr)

        results.append(dict(gamma=g, val_auc=v_auc, val_f1=v_f1, val_thr=v_thr,
                            test_auc=t_auc, test_f1=t_f1, test_f1_valthr=t_f1_valthr))
        print(f"  {g:>6.1f}  {v_auc:>8.4f}  {v_f1:>7.4f}  {v_thr:>7.3f}"
              f"  {t_auc:>9.4f}  {t_f1:>8.4f}  {t_f1_valthr:>16.4f}")

    print(f"  {'-' * 80}")

    best_tauc  = max(results, key=lambda r: r["test_auc"])
    best_vauc  = max(results, key=lambda r: r["val_auc"])
    best_tf1   = max(results, key=lambda r: r["test_f1"])
    best_vf1   = max(results, key=lambda r: r["val_f1"])

    print(f"\n  Best test AUC  : gamma={best_tauc['gamma']:.1f}"
          f"  val_AUC={best_tauc['val_auc']:.4f}  val_F1={best_tauc['val_f1']:.4f}  test_AUC={best_tauc['test_auc']:.4f}  test_F1={best_tauc['test_f1']:.4f}")
    print(f"  Best val  AUC  : gamma={best_vauc['gamma']:.1f}"
          f"  val_AUC={best_vauc['val_auc']:.4f}  val_F1={best_vauc['val_f1']:.4f}  test_AUC={best_vauc['test_auc']:.4f}  test_F1={best_vauc['test_f1']:.4f}")
    print(f"  Best test F1   : gamma={best_tf1['gamma']:.1f}"
          f"  val_AUC={best_tf1['val_auc']:.4f}  val_F1={best_tf1['val_f1']:.4f}  test_AUC={best_tf1['test_auc']:.4f}  test_F1={best_tf1['test_f1']:.4f}")
    print(f"  Best val  F1   : gamma={best_vf1['gamma']:.1f}"
          f"  val_AUC={best_vf1['val_auc']:.4f}  val_F1={best_vf1['val_f1']:.4f}  test_AUC={best_vf1['test_auc']:.4f}  test_F1={best_vf1['test_f1']:.4f}")
    print()
    _plot_gamma(results)


def mode_oam(df, step):
    val_labels,  vr, vd, ve = _load_split(df, "val")
    test_labels, tr, td, te = _load_split(df, "test")

    deltas = np.arange(0.05, 0.80 + step / 2, step)
    gamma  = GEL_CONFIG["gel_gamma"]
    k_low  = GEL_CONFIG["gel_penalty_k_low"]
    k_high = GEL_CONFIG["gel_penalty_k_high"]

    print(f"\n{'=' * 104}")
    print(f"  GEL OAM Sweep  |  delta {deltas[0]:.2f} → {deltas[-1]:.2f}  step={step}"
          f"  |  gamma={gamma}  k_low={k_low}  k_high={k_high}")
    print(f"{'=' * 104}")
    print(f"  {'delta':>6}  {'val_AUC':>8}  {'val_F1':>7}  {'val_thr':>7}"
          f"  {'test_AUC':>9}  {'test_F1':>8}  {'test_F1(vt)':>11}"
          f"  {'R%':>5}  {'D%':>5}  {'E%':>5}  {'any%':>5}")
    print(f"  {'-' * 96}")

    results = []
    for delta in deltas:
        GEL_CONFIG["gel_disagree_lim"] = float(delta)
        v_final, _ = apply_gel(vr, vd, ve)
        t_final, _ = apply_gel(tr, td, te)

        v_auc        = _auc(val_labels, v_final)
        v_thr, v_f1  = _sweep_f1(val_labels, v_final)
        t_auc        = _auc(test_labels, t_final)
        _, t_f1      = _sweep_f1(test_labels, t_final)
        t_f1_valthr  = _f1_at(test_labels, t_final, v_thr)
        vr_, vd_, ve_, any_ = _oam_rates(vr, vd, ve, delta)

        results.append(dict(delta=delta, val_auc=v_auc, val_f1=v_f1, val_thr=v_thr,
                            test_auc=t_auc, test_f1=t_f1, test_f1_valthr=t_f1_valthr,
                            trig_r=vr_, trig_d=vd_, trig_e=ve_, trig_any=any_))
        print(f"  {delta:>6.2f}  {v_auc:>8.4f}  {v_f1:>7.4f}  {v_thr:>7.3f}"
              f"  {t_auc:>9.4f}  {t_f1:>8.4f}  {t_f1_valthr:>11.4f}"
              f"  {vr_:>4.1f}  {vd_:>4.1f}  {ve_:>4.1f}  {any_:>4.1f}")

    print(f"  {'-' * 96}")

    best_vf1  = max(results, key=lambda r: r["val_f1"])
    best_vauc = max(results, key=lambda r: r["val_auc"])
    best_tf1  = max(results, key=lambda r: r["test_f1"])
    best_tauc = max(results, key=lambda r: r["test_auc"])

    print(f"\n  Best val  F1   : delta={best_vf1['delta']:.2f}"
          f"  val_AUC={best_vf1['val_auc']:.4f}  val_F1={best_vf1['val_f1']:.4f}"
          f"  test_AUC={best_vf1['test_auc']:.4f}  test_F1={best_vf1['test_f1']:.4f}")
    print(f"  Best val  AUC  : delta={best_vauc['delta']:.2f}"
          f"  val_AUC={best_vauc['val_auc']:.4f}  val_F1={best_vauc['val_f1']:.4f}"
          f"  test_AUC={best_vauc['test_auc']:.4f}  test_F1={best_vauc['test_f1']:.4f}")
    print(f"  Best test F1   : delta={best_tf1['delta']:.2f}"
          f"  val_AUC={best_tf1['val_auc']:.4f}  val_F1={best_tf1['val_f1']:.4f}"
          f"  test_AUC={best_tf1['test_auc']:.4f}  test_F1={best_tf1['test_f1']:.4f}")
    print(f"  Best test AUC  : delta={best_tauc['delta']:.2f}"
          f"  val_AUC={best_tauc['val_auc']:.4f}  val_F1={best_tauc['val_f1']:.4f}"
          f"  test_AUC={best_tauc['test_auc']:.4f}  test_F1={best_tauc['test_f1']:.4f}")
    print()
    _plot_oam(results, gamma, k_low, k_high)


def mode_grid(df, gamma_step, delta_step):
    val_labels,  vr, vd, ve = _load_split(df, "val")
    test_labels, tr, td, te = _load_split(df, "test")

    gammas = np.arange(1.0,  15.0 + gamma_step / 2, gamma_step)
    deltas = np.arange(0.05,  0.80 + delta_step / 2, delta_step)
    total  = len(gammas) * len(deltas)

    print(f"\n{'=' * 88}")
    print(f"  GEL Grid Search  |  γ {gammas[0]:.1f}→{gammas[-1]:.1f} step={gamma_step}"
          f"  |  δ {deltas[0]:.2f}→{deltas[-1]:.2f} step={delta_step}"
          f"  |  {len(gammas)}×{len(deltas)}={total} combos")
    print(f"{'=' * 88}")

    G, D         = np.meshgrid(gammas, deltas)
    val_auc_grid = np.zeros_like(G)
    val_f1_grid  = np.zeros_like(G)

    done = 0
    for i, delta in enumerate(deltas):
        GEL_CONFIG["gel_disagree_lim"] = float(delta)
        for j, gamma in enumerate(gammas):
            GEL_CONFIG["gel_gamma"] = float(gamma)
            v_final, _ = apply_gel(vr, vd, ve)
            val_auc_grid[i, j] = _auc(val_labels, v_final)
            _, val_f1_grid[i, j] = _sweep_f1(val_labels, v_final)
        done += len(gammas)
        if (i + 1) % 10 == 0 or i == len(deltas) - 1:
            print(f"  δ={delta:.2f}  {done}/{total}  "
                  f"best_val_AUC={val_auc_grid[:i+1].max():.4f}  "
                  f"best_val_F1={val_f1_grid[:i+1].max():.4f}")

    print(f"\n  {'-' * 60}")
    for label, idx, grid in [("val AUC", np.unravel_index(np.argmax(val_auc_grid), val_auc_grid.shape), val_auc_grid),
                              ("val F1",  np.unravel_index(np.argmax(val_f1_grid),  val_f1_grid.shape),  val_f1_grid)]:
        bg = float(gammas[idx[1]])
        bd = float(deltas[idx[0]])
        GEL_CONFIG["gel_gamma"]        = bg
        GEL_CONFIG["gel_disagree_lim"] = bd
        v_final, _ = apply_gel(vr, vd, ve)
        t_final, _ = apply_gel(tr, td, te)
        v_auc       = _auc(val_labels, v_final)
        v_thr, v_f1 = _sweep_f1(val_labels, v_final)
        t_auc       = _auc(test_labels, t_final)
        _, t_f1     = _sweep_f1(test_labels, t_final)
        print(f"  Best {label}: γ={bg:.1f}  δ={bd:.2f}"
              f"  val_AUC={v_auc:.4f}  val_F1={v_f1:.4f}"
              f"  test_AUC={t_auc:.4f}  test_F1={t_f1:.4f}")
    print()
    _plot_grid(G, D, val_auc_grid, val_f1_grid)


# ── Main ─────────────────────────────────────────────────────────────────── #

class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def main(mode, step, gamma_step, delta_step):
    if step is None:
        step = 1.0 if mode == "gamma" else 0.05

    if not ALL_CSV.exists():
        raise FileNotFoundError(f"Predictions file not found: {ALL_CSV}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"tune_gel_{mode}.log"
    tee = _Tee(log_path)

    try:
        df = pd.read_csv(ALL_CSV)
        print(f"Loaded {ALL_CSV}  ({len(df)} rows — "
              + ", ".join(f"{s}:{(df['split']==s).sum()}" for s in ["train", "val", "test"]) + ")")

        if mode == "gamma":
            mode_gamma(df, step)
        elif mode == "oam":
            mode_oam(df, step)
        elif mode == "grid":
            mode_grid(df, gamma_step, delta_step)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    finally:
        tee.close()
    print(f"Log saved → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["gamma", "oam", "grid"], default="gamma",
        help="Tuning mode (default: gamma)",
    )
    parser.add_argument(
        "--step", type=float, default=None,
        help="Step size for gamma/oam sweep (gamma default: 1.0, oam default: 0.05)",
    )
    parser.add_argument(
        "--gamma-step", type=float, default=0.1,
        help="Gamma step for grid search (default: 0.1)",
    )
    parser.add_argument(
        "--delta-step", type=float, default=0.01,
        help="Delta step for grid search (default: 0.01)",
    )
    args = parser.parse_args()
    main(args.mode, args.step, args.gamma_step, args.delta_step)
