"""
common/eval.py

Shared evaluation harness for all EA-series methods.
Every method script calls evaluate() and append_to_table() / append_to_results().

Protocol: fit on train → score on val → check test.
Primary ranking axis: val F1 (threshold-swept).

Two output files:
  EA_comparative_table.csv  — concise view: val/test F1+AUC, vs-GEL deltas, references
  EA_results.csv            — full view: all 3 splits × F1/AUC/acc/recall/spec/threshold
"""

import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

THRESHOLD_RANGE = np.arange(0.01, 1.00, 0.005)

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

TABLE_PATH   = _RESULTS_DIR / "EA_comparative_table.csv"
RESULTS_PATH = _RESULTS_DIR / "EA_results.csv"

TABLE_HEADER = [
    "ea_id", "method", "family", "type",
    "val_f1", "val_auc", "val_recall", "val_precision", "val_threshold",
    "test_f1", "test_auc", "test_recall", "test_precision", "test_threshold",
    "vs_gel_val_f1", "vs_gel_val_auc",
    "reference", "notes",
]

RESULTS_HEADER = [
    "ea_id", "method", "family", "type",
    # train split (populated by learned methods; blank for parameter-free)
    "train_f1", "train_auc", "train_acc", "train_recall", "train_specificity", "train_threshold",
    # val split (primary ranking axis)
    "val_f1",   "val_auc",   "val_acc",   "val_recall",   "val_specificity",   "val_threshold",
    # test split (held out — reported for completeness)
    "test_f1",  "test_auc",  "test_acc",  "test_recall",  "test_specificity",  "test_threshold",
]

GEL_V3_VAL_F1  = 0.7394
GEL_V3_VAL_AUC = 0.9060


def get_logger(ea_id: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{ea_id}.log"
    logger = logging.getLogger(ea_id)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Find threshold that maximises F1; return full metrics at that threshold."""
    best = {
        "f1": 0.0, "threshold": 0.5,
        "recall": 0.0, "precision": 0.0,
        "accuracy": 0.0, "specificity": 0.0,
    }
    for thr in THRESHOLD_RANGE:
        preds = (scores >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best["f1"]:
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            best["f1"]          = float(f1)
            best["threshold"]   = float(thr)
            best["recall"]      = float(recall_score(y_true, preds, zero_division=0))
            best["precision"]   = float(precision_score(y_true, preds, zero_division=0))
            best["accuracy"]    = float((tp + tn) / len(y_true))
            best["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return best


def evaluate(y_true: np.ndarray, scores: np.ndarray, split: str, logger: logging.Logger) -> dict:
    """Full evaluation: threshold-swept F1 + AUC + accuracy + specificity."""
    sweep = threshold_sweep(y_true, scores)
    auc   = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0

    logger.info(
        f"[{split}] F1={sweep['f1']:.4f}  AUC={auc:.4f}  "
        f"Acc={sweep['accuracy']:.4f}  Recall={sweep['recall']:.4f}  "
        f"Spec={sweep['specificity']:.4f}  Prec={sweep['precision']:.4f}  "
        f"thr={sweep['threshold']:.3f}"
    )
    return {
        "f1":          sweep["f1"],
        "auc":         auc,
        "accuracy":    sweep["accuracy"],
        "recall":      sweep["recall"],
        "specificity": sweep["specificity"],
        "precision":   sweep["precision"],
        "threshold":   sweep["threshold"],
    }


def append_to_table(row: dict) -> None:
    """Append one result row to EA_comparative_table.csv (creates with header + GEL v3 baseline if missing)."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not TABLE_PATH.exists()

    row["vs_gel_val_f1"]  = round(row.get("val_f1",  0) - GEL_V3_VAL_F1,  4)
    row["vs_gel_val_auc"] = round(row.get("val_auc", 0) - GEL_V3_VAL_AUC, 4)

    with open(TABLE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            writer.writerow({
                "ea_id": "GEL-v3", "method": "Gated Ensemble Logic",
                "family": "Custom 4-stage", "type": "BASELINE",
                "val_f1": GEL_V3_VAL_F1, "val_auc": GEL_V3_VAL_AUC,
                "val_recall": 0.7439, "val_precision": 0.7349, "val_threshold": 0.400,
                "test_f1": 0.6718, "test_auc": 0.8916,
                "test_recall": 0.7213, "test_precision": 0.6286, "test_threshold": 0.325,
                "vs_gel_val_f1": 0.0, "vs_gel_val_auc": 0.0,
                "reference": "This thesis", "notes": "GEL v3 grid-optimal gamma=11.4 delta=0.21",
            })
        writer.writerow(row)


def append_to_results(row: dict) -> None:
    """Append comprehensive per-split metrics to EA_results.csv (creates with header if missing).

    row keys follow the pattern: {split}_{metric} where split in {train, val, test}
    and metric in {f1, auc, acc, recall, specificity, threshold}.
    Train columns are optional — omit them for parameter-free methods.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists()

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, extrasaction="ignore",
                                restval="")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
