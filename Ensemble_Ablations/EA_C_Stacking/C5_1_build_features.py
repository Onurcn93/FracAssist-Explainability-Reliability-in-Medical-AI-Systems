"""
C5.1 — build_c5_features.py

Joins all_base.csv + gradcam_centroids.csv and engineers all
feature tiers T1–T6 into a single Ensemble_Ablations/data/all_c5.csv.

Tier definitions (used by C5.2 / C5.3 / C5.4 via TIER_COLS):

  T1 (3)  raw probabilities         : resnet_probability, densenet_probability, eff_probability
  T2 (4)  ensemble summary          : mean_p, std_p, min_p, max_p
  T3 (3)  pairwise disagreement     : diff_rd, diff_re, diff_de
  T4 (4)  reliability-weighted      : p_res_f1wtd, p_dense_f1wtd, p_eff_f1wtd, f1_wtd_avg
  T5 (6)  per-model calibration     : {resnet,densenet,eff}_bin_frac
                                      {resnet,densenet,eff}_calib_res
  T6 (3)  GradCAM spatial disagree  : dist_resnet_to_mean, dist_dense_to_mean, dist_eff_to_mean

T5 calibration bins are fitted on training data only (10 equal-width bins
over [0,1]) and applied as a lookup for val/test. Empty bins default to
the training fracture prevalence.

F1 weights for T4 come from gel/gel_config.py (the GEL performance anchors).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

THIS_DIR     = Path(__file__).resolve().parent
EA_ROOT      = THIS_DIR.parent
PROJECT_ROOT = EA_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gel.gel_config import GEL_CONFIG   # noqa: E402

BASE_CSV     = EA_ROOT / "data" / "all_base.csv"
CENTROID_CSV = EA_ROOT / "data" / "gradcam_centroids.csv"
OUT_CSV      = EA_ROOT / "data" / "all_c5.csv"

# F1 weights from GEL performance anchors
F1_RESNET  = GEL_CONFIG["gel_f1_resnet"]       # 0.689
F1_DENSENET = GEL_CONFIG["gel_f1_densenet"]    # 0.724
F1_EFF     = GEL_CONFIG["gel_f1_efficientnet"] # 0.671
F1_SUM     = F1_RESNET + F1_DENSENET + F1_EFF

N_BINS = 10   # calibration bins, same width as confidence-accuracy plots

# ── Tier column registry (imported by C5.2 / C5.3 / C5.4) ─────────────────
TIER_COLS = {
    "T1": ["resnet_probability", "densenet_probability", "efficientnet_probability"],
    "T2": ["mean_p", "std_p", "min_p", "max_p"],
    "T3": ["diff_rd", "diff_re", "diff_de"],
    "T4": ["p_res_f1wtd", "p_dense_f1wtd", "p_eff_f1wtd", "f1_wtd_avg"],
    "T5": ["resnet_bin_frac", "densenet_bin_frac", "eff_bin_frac",
            "resnet_calib_res", "densenet_calib_res", "eff_calib_res"],
    "T6": ["dist_resnet_to_mean", "dist_dense_to_mean", "dist_eff_to_mean"],
}

ALL_FEAT_COLS = [c for cols in TIER_COLS.values() for c in cols]


# ── Calibration bin lookup ─────────────────────────────────────────────────

def _fit_calib_bins(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit per-bin empirical fracture fraction on training data.

    Returns array of length N_BINS where entry b = fraction of training
    samples in bin b that are actually fractured. Empty bins → class prior.
    """
    bins  = np.linspace(0.0, 1.0, N_BINS + 1)
    fracs = np.full(N_BINS, labels.mean())   # default: class prior
    for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            fracs[b] = labels[mask].mean()
    return fracs


def _lookup_bin_frac(probs: np.ndarray, bin_fracs: np.ndarray) -> np.ndarray:
    """Look up empirical fracture fraction for each probability value."""
    bins     = np.linspace(0.0, 1.0, N_BINS + 1)
    indices  = np.clip(np.digitize(probs, bins) - 1, 0, N_BINS - 1)
    return bin_fracs[indices]


# ── Main ───────────────────────────────────────────────────────────────────

def build():
    print("Loading all_base.csv ...")
    base = pd.read_csv(BASE_CSV)
    print(f"  {len(base)} rows | cols: {list(base.columns)}")

    print("Loading gradcam_centroids.csv ...")
    centroids = pd.read_csv(CENTROID_CSV)
    # Extract filename for join key
    centroids["image_id"] = centroids["image_path"].apply(lambda p: Path(p).name)
    print(f"  {len(centroids)} rows | {centroids['image_id'].nunique()} unique image_ids")

    # Join on image_id
    df = base.merge(
        centroids[["image_id", "dist_resnet_to_mean", "dist_dense_to_mean", "dist_eff_to_mean"]],
        on="image_id", how="left",
    )
    missing = df[["dist_resnet_to_mean", "dist_dense_to_mean", "dist_eff_to_mean"]].isna().any(axis=1).sum()
    print(f"  Joined: {len(df)} rows | unmatched: {missing}")

    # Rename EfficientNet probability column to match TIER_COLS
    df = df.rename(columns={"efficientnet_probability": "efficientnet_probability"})

    pr = df["resnet_probability"].values
    pd_ = df["densenet_probability"].values
    pe = df["efficientnet_probability"].values

    # ── T2: ensemble summary ─────────────────────────────────────────────
    prob_stack = np.stack([pr, pd_, pe], axis=1)
    df["mean_p"] = prob_stack.mean(axis=1)
    df["std_p"]  = prob_stack.std(axis=1)
    df["min_p"]  = prob_stack.min(axis=1)
    df["max_p"]  = prob_stack.max(axis=1)

    # ── T3: pairwise absolute differences ────────────────────────────────
    df["diff_rd"] = np.abs(pr - pd_)
    df["diff_re"] = np.abs(pr - pe)
    df["diff_de"] = np.abs(pd_ - pe)

    # ── T4: F1-reliability-weighted features ─────────────────────────────
    df["p_res_f1wtd"]   = pr  * F1_RESNET
    df["p_dense_f1wtd"] = pd_ * F1_DENSENET
    df["p_eff_f1wtd"]   = pe  * F1_EFF
    df["f1_wtd_avg"]    = (pr * F1_RESNET + pd_ * F1_DENSENET + pe * F1_EFF) / F1_SUM

    # ── T5: per-model empirical calibration (fit on train only) ──────────
    labels   = (df["true_label"] == "Fractured").astype(int).values
    train_m  = df["split"].values == "train"

    bins_r = _fit_calib_bins(pr[train_m],  labels[train_m])
    bins_d = _fit_calib_bins(pd_[train_m], labels[train_m])
    bins_e = _fit_calib_bins(pe[train_m],  labels[train_m])

    df["resnet_bin_frac"]   = _lookup_bin_frac(pr,  bins_r)
    df["densenet_bin_frac"] = _lookup_bin_frac(pd_, bins_d)
    df["eff_bin_frac"]      = _lookup_bin_frac(pe,  bins_e)

    df["resnet_calib_res"]   = pr  - df["resnet_bin_frac"]
    df["densenet_calib_res"] = pd_ - df["densenet_bin_frac"]
    df["eff_calib_res"]      = pe  - df["eff_bin_frac"]

    # T6 columns already merged from centroid CSV

    # Verify all feature columns present
    missing_cols = [c for c in ALL_FEAT_COLS if c not in df.columns]
    if missing_cols:
        print(f"  [WARN] missing feature columns: {missing_cols}")
    else:
        print(f"  All {len(ALL_FEAT_COLS)} feature columns present across T1–T6.")

    # Save
    out_cols = ["split", "image_id", "true_label"] + ALL_FEAT_COLS
    df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  ({len(df)} rows × {len(out_cols)} cols)")

    # Summary stats per tier
    print("\nFeature summary (val split):")
    val = df[df["split"] == "val"]
    for tier, cols in TIER_COLS.items():
        present = [c for c in cols if c in df.columns]
        print(f"  {tier} ({len(present)} features): "
              + "  ".join(f"{c}={val[c].mean():.3f}" for c in present[:3])
              + ("  ..." if len(present) > 3 else ""))


if __name__ == "__main__":
    build()
