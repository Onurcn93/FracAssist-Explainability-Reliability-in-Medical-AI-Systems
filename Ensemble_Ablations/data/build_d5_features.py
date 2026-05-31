"""
build_d5_features.py

Extends all_base.csv with a single body_part label per image, sourced
from FracAtlas/dataset.csv (one-hot columns: hand, leg, hip, shoulder, mixed).

Output: data/all_d5.csv  (all_base columns + body_part column)

Body-part assignment rule:
  mixed=1            → "Mixed"   (image contains multiple body parts)
  hand=1             → "Hand"
  leg=1              → "Leg"
  hip=1              → "Hip"
  shoulder=1         → "Shoulder"

All 4,083 image_ids in all_base.csv have a confirmed match in dataset.csv.

Usage:
    python data/build_d5_features.py
"""

from pathlib import Path
import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]   # Ensemble_Ablations/
FRACATLAS   = ROOT.parent / "FracAtlas" / "dataset.csv"
ALL_BASE    = ROOT / "data" / "all_base.csv"
ALL_D5      = ROOT / "data" / "all_d5.csv"


def assign_body_part(row) -> str:
    if row["mixed"] == 1:
        return "Mixed"
    if row["hand"] == 1:
        return "Hand"
    if row["leg"] == 1:
        return "Leg"
    if row["hip"] == 1:
        return "Hip"
    if row["shoulder"] == 1:
        return "Shoulder"
    return "Unknown"


def build(verbose: bool = True) -> pd.DataFrame:
    base  = pd.read_csv(ALL_BASE)
    frac  = pd.read_csv(FRACATLAS)

    frac["body_part"] = frac.apply(assign_body_part, axis=1)

    merged = base.merge(frac[["image_id", "body_part"]], on="image_id", how="left")

    nulls = merged["body_part"].isna().sum()
    if nulls > 0:
        raise RuntimeError(f"{nulls} image_ids in all_base.csv have no match in dataset.csv")

    merged.to_csv(ALL_D5, index=False)

    if verbose:
        print(f"all_d5.csv written: {len(merged)} rows, {len(merged.columns)} columns")
        print("Body part distribution (all splits):")
        for bp, cnt in merged["body_part"].value_counts().items():
            print(f"  {bp}: {cnt}")
        print("Body part × split counts:")
        print(merged.groupby(["split", "body_part"]).size().unstack(fill_value=0).to_string())

    return merged


if __name__ == "__main__":
    build()
