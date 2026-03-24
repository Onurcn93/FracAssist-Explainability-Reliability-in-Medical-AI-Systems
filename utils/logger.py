"""
utils/logger.py

Generic experiment logger. Prints formatted output to stdout and saves to
results/logs/.

Designed to work across all model phases without modification:

    Phase 1 — ResNet-18 classification
        metrics=["TrLoss", "TrAcc", "VaLoss", "VaAcc"], primary="VaAcc"

    Phase 2 — YOLO localization
        metrics=["BoxLoss", "mAP@0.5", "P", "R"], primary="mAP@0.5"

    Phase 2 — YOLO segmentation
        metrics=["BoxLoss", "MaskLoss", "mAP@0.5", "MaskMAP", "P", "R"], primary="mAP@0.5"

    Phase 3 — XAI experiments
        metrics defined per experiment

The caller defines metric columns and primary metric at init time.
"""

import os
import time
from typing import Dict, List, Optional

SEP  = "=" * 72
LINE = "─" * 72
COL_W = 10


class Logger:
    def __init__(
        self,
        experiment: str,
        metrics: List[str],
        primary_metric: str,
        log_dir: str = "results/logs",
    ):
        """
        Args:
            experiment:     Human-readable experiment ID.
                            e.g. "Y0 | YOLOv8s | detect | epochs=30 | imgsz=600"
            metrics:        Ordered list of metric names shown per epoch.
                            e.g. ["BoxLoss", "mAP@0.5", "P", "R"]
            primary_metric: Metric used to track best model and report at completion.
                            e.g. "mAP@0.5"
            log_dir:        Directory for log files. Created if absent.
        """
        self.experiment     = experiment
        self.metrics        = metrics
        self.primary_metric = primary_metric
        self._start: Optional[float] = None
        self._file          = None

        os.makedirs(log_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in experiment)
        path = os.path.join(log_dir, f"{safe}.log")
        self._file = open(path, "w", encoding="utf-8")

    # ------------------------------------------------------------------ #

    def _w(self, line: str = "") -> None:
        print(line)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    # ------------------------------------------------------------------ #

    def log_start(self, config: Dict[str, str]) -> None:
        """Print experiment header before training begins.

        Args:
            config: Ordered dict of label -> value pairs describing the run.
                    e.g. {"Data": "data/dataset_yolo/data.yaml",
                           "Epochs": "30  |  imgsz: 600  |  Device: cuda",
                           "Weights": "yolov8s.pt (COCO pretrained)"}
        """
        self._start = time.time()

        self._w()
        self._w(f"▶▶▶  Starting  {self.experiment}")
        self._w(SEP)
        self._w(f"  Experiment : {self.experiment}")
        for label, value in config.items():
            self._w(f"  {label:<11}: {value}")
        self._w(f"  Save by    : {self.primary_metric}")
        self._w(SEP)
        self._w()

        header = f" {'Epoch':>5}"
        for m in self.metrics:
            header += f" | {m:>{COL_W}}"
        self._w(header)
        self._w(LINE)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log one epoch row.

        Args:
            epoch:   Epoch number (1-indexed).
            metrics: Dict mapping metric name to float value.
                     Keys must match self.metrics; missing keys shown as nan.
        """
        row = f" {epoch:>5}"
        for m in self.metrics:
            val = metrics.get(m, float("nan"))
            row += f" | {val:>{COL_W}.4f}"
        self._w(row)

    def log_best(self, value: float, save_path: str) -> None:
        """Annotate a new best metric value beneath the current epoch row."""
        self._w(f"      ↑ New best {self.primary_metric}: {value:.4f} — checkpoint saved → {save_path}")

    def log_complete(self, best_value: float, save_path: str) -> None:
        """Print training completion summary."""
        elapsed = time.time() - self._start if self._start else 0.0
        mins, secs = divmod(int(elapsed), 60)
        self._w()
        self._w(f"✓  Training complete  |  Best {self.primary_metric}: {best_value:.4f}")
        self._w(f"   Time: {elapsed:.0f}s ({mins}m {secs:02d}s)  |  Checkpoint: {save_path}")
        self._w()

    def log_message(self, msg: str) -> None:
        """Log a free-form message — warnings, notes, mid-run info."""
        self._w(msg)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None