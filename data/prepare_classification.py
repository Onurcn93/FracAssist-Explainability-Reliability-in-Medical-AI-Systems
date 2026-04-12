"""
data/prepare_classification.py

Builds the ImageFolder classification dataset from local FracAtlas files.

Output structure:
    data/dataset_cls/
    ├── train/
    │   ├── Fractured/
    │   └── Non_fractured/
    ├── val/
    │   ├── Fractured/
    │   └── Non_fractured/
    └── test/
        ├── Fractured/
        └── Non_fractured/

Fractured images    : official train/valid/test.csv splits from FracAtlas
Non-fractured images: proportional 80/12/8 split, seed=42

Usage:
    python data/prepare_classification.py
    python data/prepare_classification.py --out_dir data/dataset_cls_custom
    python data/prepare_classification.py --clean          # wipe and rebuild
"""

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd

FRACATLAS_ROOT = Path("FracAtlas")
FRAC_DIR       = FRACATLAS_ROOT / "images" / "Fractured"
NONFRAC_DIR    = FRACATLAS_ROOT / "images" / "Non_fractured"
SPLIT_DIR      = FRACATLAS_ROOT / "Utilities" / "Fracture Split"

SEED = 42
# Non-fractured proportional ratios (remainder goes to test)
NONFRAC_TR_RATIO  = 0.80
NONFRAC_VAL_RATIO = 0.12


def _read_ids(csv_path: Path) -> set:
    """Return a set of image ID stems from a FracAtlas split CSV."""
    df  = pd.read_csv(csv_path)
    col = df.columns[0]
    raw = df[col].astype(str).tolist()
    # Accept both full filename and stem (without extension)
    return set(raw) | {Path(r).stem for r in raw}


def _copy_images(images: list, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for src in images:
        shutil.copy2(src, dst / src.name)
    return len(images)


def build(out_dir: Path, clean: bool = False) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"[clean] Removed {out_dir}")

    if (out_dir / "train").exists():
        print(f"[skip]  {out_dir} already exists — pass --clean to rebuild")
        _print_counts(out_dir)
        return

    # ── Verify source paths ──────────────────────────────────────── #
    for path in (FRAC_DIR, NONFRAC_DIR, SPLIT_DIR):
        if not path.exists():
            raise FileNotFoundError(f"[error] Source not found: {path}")

    # ── Fractured: use official CSV splits ───────────────────────── #
    train_ids = _read_ids(SPLIT_DIR / "train.csv")
    val_ids   = _read_ids(SPLIT_DIR / "valid.csv")
    test_ids  = _read_ids(SPLIT_DIR / "test.csv")

    all_frac   = sorted(FRAC_DIR.glob("*.jpg"))
    frac_train = [f for f in all_frac if f.stem in train_ids or f.name in train_ids]
    frac_val   = [f for f in all_frac if f.stem in val_ids   or f.name in val_ids]
    frac_test  = [f for f in all_frac if f.stem in test_ids  or f.name in test_ids]

    unassigned = len(all_frac) - len(frac_train) - len(frac_val) - len(frac_test)
    if unassigned:
        print(f"[warn]  {unassigned} fractured images not matched by any CSV — excluded")

    # ── Non-fractured: proportional split (no official CSV) ──────── #
    all_nf = sorted(NONFRAC_DIR.glob("*.jpg"))
    rng    = random.Random(SEED)
    rng.shuffle(all_nf)
    n      = len(all_nf)
    n_tr   = int(n * NONFRAC_TR_RATIO)
    n_va   = int(n * NONFRAC_VAL_RATIO)
    nf_train = all_nf[:n_tr]
    nf_val   = all_nf[n_tr : n_tr + n_va]
    nf_test  = all_nf[n_tr + n_va :]

    # ── Copy into split dirs ─────────────────────────────────────── #
    counts = {}
    for split, frac_imgs, nf_imgs in [
        ("train", frac_train, nf_train),
        ("val",   frac_val,   nf_val),
        ("test",  frac_test,  nf_test),
    ]:
        n_f  = _copy_images(frac_imgs, out_dir / split / "Fractured")
        n_nf = _copy_images(nf_imgs,   out_dir / split / "Non_fractured")
        counts[split] = (n_f, n_nf)

    print(f"\n[done]  Dataset written to {out_dir}")
    _print_counts_from(counts)


def _print_counts(out_dir: Path) -> None:
    counts = {}
    for split in ("train", "val", "test"):
        n_f  = len(list((out_dir / split / "Fractured").glob("*.jpg")))
        n_nf = len(list((out_dir / split / "Non_fractured").glob("*.jpg")))
        counts[split] = (n_f, n_nf)
    _print_counts_from(counts)


def _print_counts_from(counts: dict) -> None:
    print(f"\n  {'Split':6s}  {'Frac':>6}  {'NonFrac':>8}  {'Total':>6}  {'Ratio':>7}")
    print("  " + "─" * 44)
    for split, (n_f, n_nf) in counts.items():
        ratio = n_nf / n_f if n_f else float("inf")
        print(f"  {split:6s}  {n_f:>6}  {n_nf:>8}  {n_f+n_nf:>6}  {ratio:>5.1f}:1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare FracAtlas classification dataset (ImageFolder layout)"
    )
    parser.add_argument(
        "--out_dir", default="data/dataset_cls",
        help="Output directory (default: data/dataset_cls)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Wipe output directory and rebuild from scratch",
    )
    args = parser.parse_args()
    build(Path(args.out_dir), clean=args.clean)


if __name__ == "__main__":
    main()
