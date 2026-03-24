"""
data/prepare_yolo.py

Builds the YOLO dataset folder structure from FracAtlas source files.

Reads the official Fracture Split CSVs to determine which images belong in
each split, then copies the images and YOLO labels into the layout expected
by Ultralytics:

    data/dataset_yolo/
    ├── data.yaml
    ├── train/
    │   ├── images/   (574 fractured images)
    │   └── labels/   (corresponding YOLO .txt files)
    └── valid/
        ├── images/   (82 fractured images)
        └── labels/

Only fractured images are used — this matches the author's training setup.
Non-fractured images are excluded from YOLO training entirely.

Usage:
    python data/prepare_yolo.py
    python data/prepare_yolo.py --fracatlas_root /custom/path/FracAtlas
    python data/prepare_yolo.py --clean     # wipe and rebuild from scratch
"""

import argparse
import csv
import shutil
from pathlib import Path


SPLITS = ["train", "valid"]
SPLIT_CSV = {"train": "train.csv", "valid": "valid.csv"}


def read_split_csv(csv_path: Path) -> list:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["image_id"] for row in reader]


def prepare_split(
    split: str,
    image_ids: list,
    images_src: Path,
    labels_src: Path,
    out_dir: Path,
) -> None:
    img_dst = out_dir / split / "images"
    lbl_dst = out_dir / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    missing_images = []
    missing_labels = []

    for image_id in image_ids:
        src_img = images_src / image_id
        if src_img.exists():
            shutil.copy2(src_img, img_dst / image_id)
        else:
            missing_images.append(image_id)

        label_id = Path(image_id).stem + ".txt"
        src_lbl = labels_src / label_id
        if src_lbl.exists():
            shutil.copy2(src_lbl, lbl_dst / label_id)
        else:
            missing_labels.append(label_id)

    print(f"[data] {split:>5}: {len(image_ids) - len(missing_images)}/{len(image_ids)} images copied")

    if missing_images:
        print(f"[warn] {split}: {len(missing_images)} images not found: {missing_images[:5]}")
    if missing_labels:
        print(f"[warn] {split}: {len(missing_labels)} labels not found: {missing_labels[:5]}")


def write_data_yaml(out_dir: Path) -> None:
    content = (
        "train: train/images\n"
        "val:   valid/images\n"
        "\n"
        "nc: 1\n"
        'names: ["fractured"]\n'
    )
    path = out_dir / "data.yaml"
    path.write_text(content)
    print(f"[data] Written {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from FracAtlas")
    parser.add_argument(
        "--fracatlas_root",
        type=str,
        default=None,
        help="Path to FracAtlas folder. Defaults to <repo_root>/FracAtlas",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <repo_root>/data/dataset_yolo",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove and rebuild the output directory from scratch",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    fracatlas_root = Path(args.fracatlas_root) if args.fracatlas_root else repo_root / "FracAtlas"
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "data" / "dataset_yolo"

    if not fracatlas_root.exists():
        raise FileNotFoundError(
            f"FracAtlas not found at: {fracatlas_root}\n"
            "Place the dataset at the repo root — see README for setup instructions."
        )

    images_src = fracatlas_root / "images" / "Fractured"
    labels_src = fracatlas_root / "Annotations" / "YOLO"
    splits_src = fracatlas_root / "Utilities" / "Fracture Split"

    for p in [images_src, labels_src, splits_src]:
        if not p.exists():
            raise FileNotFoundError(f"Expected FracAtlas path not found: {p}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"[data] Cleaned {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] FracAtlas : {fracatlas_root}")
    print(f"[data] Output    : {out_dir}")
    print()

    for split in SPLITS:
        csv_path = splits_src / SPLIT_CSV[split]
        image_ids = read_split_csv(csv_path)
        prepare_split(split, image_ids, images_src, labels_src, out_dir)

    write_data_yaml(out_dir)

    print()
    print("[data] Done. Dataset ready at:", out_dir)


if __name__ == "__main__":
    main()