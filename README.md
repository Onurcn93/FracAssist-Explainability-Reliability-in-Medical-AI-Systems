# Explainability & Reliability in Medical AI Systems

MSc thesis repository. Bone fracture detection on musculoskeletal X-rays using deep
learning, with a focus on model explainability and clinical reliability.

The core thesis argument: clinical AI should **support prediction with evidence**, not
explain after the fact. Every model component is evaluated against the question —
*"How does this help a clinician make a better decision?"*

**Dataset:** FracAtlas (Abedeen et al., 2023) — 4,083 X-ray images, 717 fractured,
annotated for classification, localization, and segmentation.

---

## Phases

| Phase | Model | Task | Primary Metric | Status |
|-------|-------|------|----------------|--------|
| 1 | ResNet-18 | Binary fracture classification | F1 (fractured class) | Complete |
| 2 | YOLOv8s / YOLOv8s-seg | Localization & segmentation | mAP@0.5 | Active |
| 3 | CBM + Prototypes + Counterfactuals | XAI — three-pillar architecture | Task-specific | Pending |

Phase 3 is the core thesis contribution: integrating clinically-grounded attribute
explanations (Pillar 1), precedent-based example retrieval (Pillar 2), and contrastive
counterfactual explanations (Pillar 3) in a single system for fracture detection.

---

## Repository Structure

```
/
├── configs/                  # Experiment config files (YAML)
├── data/                     # Data preparation scripts
├── models/
│   ├── classification/       # ResNet-18 experiments (E-series)
│   └── yolo/                 # YOLO localization & segmentation (Y / YS-series)
├── xai/                      # XAI pillar implementations (Phase 3)
├── utils/
│   ├── logger.py             # Experiment logging
│   └── plot.py               # Training curves, metric plots
├── results/                  # Saved metrics and plots (gitignored)
└── weights/                  # Saved model weights (gitignored)
```

All experiments are implemented as plain Python scripts.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Supported Models

| Model | Task | Phase |
|-------|------|-------|
| YOLOv8s | Fracture localization (detect) | 2 |
| YOLOv8s-seg | Fracture segmentation | 2 |
| ResNet-18 | Binary classification | 1 |
| CBM / Prototypes / Counterfactuals | XAI explainability | 3 |

---

## Key Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--config` | path to YAML | Experiment config file |
| `--task` | key in config / `all` | Which experiment(s) to run |
| `--no-plot` | flag | Disable plot generation |
| `--weights` | path to `.pt` | Weights for standalone evaluation |
| `--split` | `val` / `test` | Eval split — use `test` only for final reporting |

---

## Usage

### 1. Prepare dataset (one-time)

```bash
python data/prepare_yolo.py
```

Reads the official FracAtlas Fracture Split CSVs and builds the YOLO folder
structure under `data/dataset_yolo/`. Run once — output persists locally.

```bash
python data/prepare_yolo.py --clean   # wipe and rebuild from scratch
```

### 2. Train

```bash
python main.py --config configs/yolo_baseline.yaml --task localization
python main.py --config configs/yolo_baseline.yaml --task segmentation
python main.py --config configs/yolo_baseline.yaml --task all
```

`--task all` runs every entry in the config file sequentially.
`--no-plot` disables figure generation (overrides config).

### 3. Evaluate

```bash
python models/yolo/evaluate.py \
    --weights weights/Y0_best.pt \
    --data    data/dataset_yolo/data.yaml \
    --task    detect \
    --imgsz   600
```

Use `--split val` during development (default). Use `--split test` only for
final per-phase reporting.

### Config format

```yaml
# configs/yolo_baseline.yaml
localization:
  experiment_id : "Y0"
  task          : "detect"
  model_weights : "yolov8s.pt"
  data_yaml     : "data/dataset_yolo/data.yaml"
  epochs        : 30
  imgsz         : 600
  device        : "0"        # GPU index, or "cpu"
  plot          : true
```

Each top-level key is a runnable task. Add new experiments by adding new keys.

---

## Dataset Setup

FracAtlas is not committed to this repository. Download from Figshare and place at the
repo root:

**Download:** https://doi.org/10.6084/m9.figshare.22363012

```
FracAtlas/
├── images/
│   ├── Fractured/
│   └── Non_fractured/
├── Annotations/
│   ├── YOLO/
│   ├── COCO JSON/
│   └── PASCAL VOC/
├── Utilities/
│   └── Fracture Split/
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
└── dataset.csv
```

> All YOLO experiments use the official Fracture Split CSVs — not a random split — to
> enable benchmark comparison with the original paper.

---

## Phase 2 — YOLO Baseline (Abedeen et al., 2023)

| Task | Model | Box P | Box R | mAP@0.5 | Mask P | Mask R | Mask mAP@0.5 |
|------|-------|-------|-------|---------|--------|--------|--------------|
| Localization | YOLOv8s | 0.807 | 0.473 | 0.562 | — | — | — |
| Segmentation | YOLOv8s-seg | 0.718 | 0.607 | 0.627 | 0.830 | 0.499 | 0.589 |

Config: `epochs=30`, `imgsz=600`, COCO pre-trained weights, Ultralytics defaults.
Split: 574 train / 82 val / 61 test (fractured images only).

---

## Reproducibility

- Validation set is used for all tuning decisions; test set is used once per phase.
- Experiment IDs are stable: `E-series` (classification), `Y-series` (localization),
  `YS-series` (segmentation).
- Random seeds are set in all training scripts.

---

## Citation

```bibtex
@article{abedeen2023fracatlas,
  title={FracAtlas: A Dataset for Fracture Classification, Localization and
         Segmentation of Musculoskeletal Radiographs},
  author={Abedeen, Ifra and others},
  journal={Scientific Data},
  publisher={Nature Portfolio},
  year={2023},
  doi={10.1038/s41597-023-02432-4}
}
```
