"""
data/build_features.py

Builds the enriched feature table (all_EA.csv) from the base input (all_base.csv).
Run once before any Family C/D/E methods that need engineered features.

Usage:
    python build_features.py              # builds Tier 1 + Tier 2
    python build_features.py --tiers 1    # Tier 1 only
    python build_features.py --tiers 1 2  # Tier 1 + 2 (default)
    python build_features.py --tiers 1 2 3 4  # all non-GradCAM tiers

Fit on train, apply to val and test. No data leakage.

Tier 1 — Probability (no-brainer):
    p_res_norm, p_dense_norm, p_eff_norm  (per-sample probability mass distribution)
    p_mean, p_max, p_min, p_std, p_range

Tier 2 — Calibration:
    p_res_calib, p_dense_calib, p_eff_calib  (temperature-scaled; T fit on train)
    margin_res, margin_dense, margin_eff      (|p - 0.5|)
    calib_gap_res, calib_gap_dense, calib_gap_eff  (predicted vs empirical accuracy per bin)

Tier 3 — Confidence (original):
    range_reliability_res, range_reliability_dense, range_reliability_eff
    ens_confidence
    expected_correctness

Tier 4 — Agreement & Diversity (original):
    agree_count
    pairwise_disagree
    dissent_direction

Tier 5 — GradCAM Evidence (DEFERRED — requires batch GradCAM pre-computation):
    cam_iou, cam_union_area, cam_consensus_peak
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT    = Path(__file__).resolve().parents[1]
IN_CSV  = ROOT / "data" / "all_base.csv"
OUT_CSV = ROOT / "data" / "all_EA.csv"
LOG_DIR = ROOT / "logs"

PROB_COLS = ["resnet_probability", "densenet_probability", "efficientnet_probability"]
SHORT     = ["res", "dense", "eff"]


# --------------------------------------------------------------------------- #
# Tier 1 — Probability features
# --------------------------------------------------------------------------- #

def build_tier1(df: pd.DataFrame) -> pd.DataFrame:
    p = df[PROB_COLS].values
    total = p.sum(axis=1, keepdims=True).clip(min=1e-9)

    for i, s in enumerate(SHORT):
        df[f"p_{s}_norm"] = p[:, i] / total[:, 0]

    df["p_mean"]  = p.mean(axis=1)
    df["p_max"]   = p.max(axis=1)
    df["p_min"]   = p.min(axis=1)
    df["p_std"]   = p.std(axis=1)
    df["p_range"] = p.max(axis=1) - p.min(axis=1)
    return df


# --------------------------------------------------------------------------- #
# Tier 2 — Calibration features
# --------------------------------------------------------------------------- #

def _nll(T, probs, labels):
    """Negative log-likelihood for temperature scaling."""
    scaled = np.clip(probs / T, 1e-9, 1 - 1e-9)
    return -np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))


def fit_temperature(train_probs: np.ndarray, train_labels: np.ndarray) -> float:
    result = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded",
                             args=(train_probs, train_labels))
    return float(result.x)


def build_tier2(df: pd.DataFrame, temperatures: dict | None = None) -> tuple[pd.DataFrame, dict]:
    train = df[df["split"] == "train"]
    labels = (train["true_label"] == "Fractured").astype(int).values

    if temperatures is None:
        temperatures = {}
        for col, s in zip(PROB_COLS, SHORT):
            T = fit_temperature(train[col].values, labels)
            temperatures[s] = T
            print(f"  [tier2] Temperature {s}: T={T:.4f}")

    for col, s in zip(PROB_COLS, SHORT):
        T = temperatures[s]
        df[f"p_{s}_calib"] = (df[col] / T).clip(0.0, 1.0)
        df[f"margin_{s}"]  = (df[col] - 0.5).abs()

    # Calibration gap — per model, per decile bin (fit on train, apply all)
    for col, s in zip(PROB_COLS, SHORT):
        bins = np.percentile(train[col].values, np.arange(0, 110, 10))
        bins = np.unique(bins)
        bin_ids = np.digitize(df[col].values, bins) - 1
        bin_ids = np.clip(bin_ids, 0, len(bins) - 2)

        train_bin_ids = np.digitize(train[col].values, bins) - 1
        train_bin_ids = np.clip(train_bin_ids, 0, len(bins) - 2)
        train_correct = ((train["true_label"] == "Fractured").astype(int).values ==
                         (train[col].values >= 0.5).astype(int))

        calib_gap = np.zeros(len(df))
        for b in range(len(bins) - 1):
            mask_train = train_bin_ids == b
            mask_all   = bin_ids == b
            if mask_train.sum() == 0:
                continue
            mean_prob = train[col].values[mask_train].mean()
            emp_acc   = train_correct[mask_train].mean()
            calib_gap[mask_all] = mean_prob - emp_acc

        df[f"calib_gap_{s}"] = calib_gap

    return df, temperatures


# --------------------------------------------------------------------------- #
# Tier 3 — Confidence features (original)
# --------------------------------------------------------------------------- #

def build_tier3(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    train_labels = (train["true_label"] == "Fractured").astype(int).values

    for col, s in zip(PROB_COLS, SHORT):
        # range_reliability: historical accuracy in same decile band on train
        bins = np.percentile(train[col].values, np.arange(0, 110, 10))
        bins = np.unique(bins)
        train_bin = np.clip(np.digitize(train[col].values, bins) - 1, 0, len(bins) - 2)
        train_pred = (train[col].values >= 0.5).astype(int)
        train_correct = (train_pred == train_labels).astype(float)

        all_bin = np.clip(np.digitize(df[col].values, bins) - 1, 0, len(bins) - 2)
        rr = np.zeros(len(df))
        for b in range(len(bins) - 1):
            mask_t = train_bin == b
            mask_a = all_bin == b
            rr[mask_a] = train_correct[mask_t].mean() if mask_t.sum() > 0 else 0.5
        df[f"range_reliability_{s}"] = rr

    # ens_confidence: agreement (1 - p_std normalised) × mean margin
    std_norm  = 1.0 - (df["p_std"] / 0.5).clip(0, 1)
    mean_marg = df[[f"margin_{s}" for s in SHORT]].mean(axis=1)
    df["ens_confidence"] = (std_norm * mean_marg).clip(0, 1)

    # expected_correctness: linear regression on agree + calibration features from train
    feature_cols = [f"range_reliability_{s}" for s in SHORT] + ["ens_confidence"]
    train_sub = df[df["split"] == "train"]
    X_tr = train_sub[feature_cols].values
    y_tr = (train_sub["true_label"] == "Fractured").astype(int).values
    # simple least-squares weights
    X_aug = np.column_stack([X_tr, np.ones(len(X_tr))])
    coef, *_ = np.linalg.lstsq(X_aug, y_tr.astype(float), rcond=None)
    X_all = np.column_stack([df[feature_cols].values, np.ones(len(df))])
    df["expected_correctness"] = np.clip(X_all @ coef, 0.0, 1.0)

    return df


# --------------------------------------------------------------------------- #
# Tier 4 — Agreement & Diversity features (original)
# --------------------------------------------------------------------------- #

def build_tier4(df: pd.DataFrame, opt_thresholds: dict) -> pd.DataFrame:
    """opt_thresholds: {"res": 0.525, "dense": 0.175, "eff": 0.525}"""
    votes = np.column_stack([
        (df[col].values >= opt_thresholds[s]).astype(int)
        for col, s in zip(PROB_COLS, SHORT)
    ])
    df["agree_count"] = votes.sum(axis=1)

    p = df[PROB_COLS].values
    df["pairwise_disagree"] = (
        np.abs(p[:, 0] - p[:, 1]) +
        np.abs(p[:, 0] - p[:, 2]) +
        np.abs(p[:, 1] - p[:, 2])
    ) / 3.0

    # dissent_direction: which model dissents and in which direction
    # encode as integer: 0=consensus, 1=res_high, 2=res_low, 3=dense_high, ...
    majority = votes.sum(axis=1)
    dissent  = np.zeros(len(df), dtype=int)
    for i, (s, v_col) in enumerate(zip(SHORT, votes.T)):
        lone_high = (majority == 1) & (v_col == 1)
        lone_low  = (majority == 2) & (v_col == 0)
        dissent[lone_high] = i * 2 + 1
        dissent[lone_low]  = i * 2 + 2
    df["dissent_direction"] = dissent

    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2],
                        choices=[1, 2, 3, 4],
                        help="Feature tiers to build (default: 1 2)")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    print(f"[build_features] Loading {IN_CSV}")
    df = pd.read_csv(IN_CSV)
    transforms = []

    if 1 in args.tiers:
        print("[build_features] Building Tier 1 — probability features")
        df = build_tier1(df)
        transforms.append("tier1_probability")

    temperatures = None
    if 2 in args.tiers:
        print("[build_features] Building Tier 2 — calibration features (fit on train)")
        df, temperatures = build_tier2(df)
        transforms.append("tier2_calibration")

    if 3 in args.tiers:
        if 2 not in args.tiers:
            raise ValueError("Tier 3 requires Tier 2 (margins). Include --tiers 1 2 3")
        print("[build_features] Building Tier 3 — confidence features (fit on train)")
        df = build_tier3(df)
        transforms.append("tier3_confidence")

    if 4 in args.tiers:
        opt_thr = {"res": 0.525, "dense": 0.175, "eff": 0.525}
        print(f"[build_features] Building Tier 4 — agreement features (opt thresholds: {opt_thr})")
        df = build_tier4(df, opt_thr)
        transforms.append("tier4_agreement")

    df.to_csv(OUT_CSV, index=False)
    print(f"[build_features] Saved {OUT_CSV}  ({len(df)} rows, {len(df.columns)} columns)")

    log = {"transforms": transforms, "columns_added": [c for c in df.columns
                                                        if c not in pd.read_csv(IN_CSV).columns]}
    if temperatures:
        log["temperatures"] = temperatures
    with open(LOG_DIR / "build_features.log", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[build_features] Transform log → logs/build_features.log")


if __name__ == "__main__":
    main()
