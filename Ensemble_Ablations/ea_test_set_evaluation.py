"""
ea_test_set_evaluation.py

Standalone test-set evaluation for all 25 EA-series methods.
Re-runs each method from scratch (re-trains learned models on train split,
then evaluates on the held-out test split) without touching the existing
EA_results.csv or EA_comparative_table.csv.

Output: results/EA_test_evaluation.csv
Columns: ea_id, method, family, type,
         test_f1, test_auc, test_recall, test_precision, test_threshold,
         val_f1, val_auc   (included for cross-reference)

GEL v3 is appended as a reference row (values from eval_gel.py run).

Usage (from Ensemble_Ablations/ directory):
    python ea_test_set_evaluation.py
"""

import csv
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Suppress writes to existing CSVs during this run
# --------------------------------------------------------------------------- #
import common.eval as _eval_module

_eval_module.append_to_table   = lambda row: None
_eval_module.append_to_results = lambda row: None

# --------------------------------------------------------------------------- #
# Full 25-method registry (corrected — includes all C5 sub-variants)
# --------------------------------------------------------------------------- #
REGISTRY = {
    # Family A — Averaging
    "EA-A1":   ("EA_A_Averaging.EA_A1_mean",             "Mean (sum rule)",             "A", "LIT"),
    "EA-A2":   ("EA_A_Averaging.EA_A2_product",          "Product rule",                "A", "LIT"),
    "EA-A3":   ("EA_A_Averaging.EA_A3_max",              "Max rule",                    "A", "LIT"),
    "EA-A4":   ("EA_A_Averaging.EA_A4_calibrated_mean",  "Calibrated mean",             "A", "ORIG"),
    # Family B — Weighting
    "EA-B1":   ("EA_B_Weighting.EA_B1_f1_weighted",           "F1-weighted average",             "B", "LIT"),
    "EA-B2":   ("EA_B_Weighting.EA_B2_rc_power",              "RC power weighting",              "B", "LIT"),
    "EA-B3":   ("EA_B_Weighting.EA_B3_auc_weighted",          "AUC-weighted average",            "B", "LIT"),
    "EA-B4":   ("EA_B_Weighting.EA_B4_confidence_modulated",  "Confidence-modulated weighting",  "B", "ORIG"),
    "EA-B5":   ("EA_B_Weighting.EA_B5_diversity_penalised",   "Diversity-penalised weighting",   "B", "ORIG"),
    # Family C — Stacking (base meta-learners)
    "EA-C1":   ("EA_C_Stacking.EA_C1_logreg",            "Logistic regression meta-learner",  "C", "LIT"),
    "EA-C2":   ("EA_C_Stacking.EA_C2_shallow_mlp",       "Shallow MLP meta-learner",          "C", "LIT"),
    "EA-C3":   ("EA_C_Stacking.EA_C3_gbt",               "Gradient-boosted trees",            "C", "LIT"),
    "EA-C4":   ("EA_C_Stacking.EA_C4_decision_template", "Decision-template combiner",        "C", "LIT"),
    # Family C — Feature-engineered stacking (C5 tier ablations)
    "EA-C5.2": ("EA_C_Stacking.EA_C5_2_logreg_tiers",                     "Feature-eng stacking — LogReg tiers",   "C", "ORIG"),
    "EA-C5.3": ("EA_C_Stacking.EA_C5_3_mlp_tiers",                        "Feature-eng stacking — MLP tiers",      "C", "ORIG"),
    "EA-C5.4": ("EA_C_Stacking.EA_C5_4_gbt_tiers",                        "Feature-eng stacking — GBT tiers",      "C", "ORIG"),
    "EA-C5.5": ("EA_C_Stacking.EA_C5_5_platt_calibration",                "Feature-eng stacking — Platt scaling",  "C", "ORIG"),
    "EA-C5.6": ("EA_C_Stacking.EA_C5_6_disagreement_stratification",      "Feature-eng stacking — disagreement",   "C", "ORIG"),
    # Family D — Gating
    "EA-D1":   ("EA_D_Gating.EA_D1_confidence_select",    "Confidence-based selection",              "D", "LIT"),
    "EA-D2":   ("EA_D_Gating.EA_D2_meta_des",             "META-DES-style meta-classifier",          "D", "LIT"),
    "EA-D3":   ("EA_D_Gating.EA_D3_disagreement_gating",  "Disagreement-triggered gating",           "D", "ORIG"),
    "EA-D4":   ("EA_D_Gating.EA_D4_region_competence",    "Region-of-competence weighting",          "D", "ORIG"),
    "EA-D5":   ("EA_D_Gating.EA_D5_body_part_competence", "Body-part-constrained competence",        "D", "ORIG"),
    # Family E — Cascading
    "EA-E1":   ("EA_E_Cascading.EA_E1_confidence_cascade",      "Confidence cascade",              "E", "LIT"),
    "EA-E2":   ("EA_E_Cascading.EA_E2_uncertainty_escalation",  "Uncertainty-escalation cascade",  "E", "ORIG"),
}

OUTPUT_COLS = [
    "ea_id", "method", "family", "type",
    "test_f1", "test_auc", "test_recall", "test_precision", "test_threshold",
    "val_f1", "val_auc",
]

GEL_V3_ROW = {
    "ea_id": "GEL-v3", "method": "Gated Ensemble Logic (BVG+RC+OAM+PDWF)",
    "family": "—", "type": "NOVEL",
    # Test at the deployed/val-optimal threshold 0.40 (carried), not test-swept 0.325.
    "test_f1": 0.6614, "test_auc": 0.8916,
    "test_recall": 0.6885, "test_precision": 0.6364, "test_threshold": 0.400,
    "val_f1": 0.7394, "val_auc": 0.9060,
}


def _run_method(ea_id: str) -> dict | None:
    module_path, display, family, mtype = REGISTRY[ea_id]
    print(f"  Running {ea_id}: {display} ...", end=" ", flush=True)
    try:
        mod = importlib.import_module(module_path)
        result = mod.run(root_dir=ROOT, seed=42, plot=False)
        print(f"val_F1={result.get('val_f1', 0):.4f}  "
              f"test_F1={result.get('test_f1', 0):.4f}  "
              f"test_AUC={result.get('test_auc', 0):.4f}")
        return result
    except Exception as exc:
        print(f"FAILED — {exc}")
        return None


def _save_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved -> {out_path}")


def _print_summary(rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r.get("test_auc", 0), reverse=True)
    header = f"{'ea_id':<12} {'method':<48} {'test_F1':>8} {'test_AUC':>9} {'test_Rec':>9} {'test_Pre':>9}"
    print("\n" + "=" * len(header))
    print("TEST SET RESULTS — sorted by test AUC (descending)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows_sorted:
        marker = " <-- GEL" if r["ea_id"] == "GEL-v3" else ""
        print(
            f"{r['ea_id']:<12} {r['method']:<48} "
            f"{r.get('test_f1', 0):>8.4f} {r.get('test_auc', 0):>9.4f} "
            f"{r.get('test_recall', 0):>9.4f} {r.get('test_precision', 0):>9.4f}"
            f"{marker}"
        )
    print("=" * len(header))


def main():
    print(f"\n{'='*72}")
    print("  EA-Series Test Set Evaluation (all 25 methods + GEL v3 reference)")
    print(f"{'='*72}\n")

    rows = []
    failed = []

    for ea_id in REGISTRY:
        result = _run_method(ea_id)
        if result is None:
            failed.append(ea_id)
            continue
        rows.append({
            "ea_id":           result.get("ea_id", ea_id),
            "method":          result.get("method", REGISTRY[ea_id][1]),
            "family":          result.get("family", REGISTRY[ea_id][2]),
            "type":            result.get("type",   REGISTRY[ea_id][3]),
            "test_f1":         round(result.get("test_f1",        0), 6),
            "test_auc":        round(result.get("test_auc",       0), 6),
            "test_recall":     round(result.get("test_recall",    0), 6),
            "test_precision":  round(result.get("test_precision", 0), 6),
            "test_threshold":  round(result.get("test_threshold", 0), 4),
            "val_f1":          round(result.get("val_f1",         0), 6),
            "val_auc":         round(result.get("val_auc",        0), 6),
        })

    # GEL v3 reference row (from eval_gel.py confirmed run)
    rows.insert(0, GEL_V3_ROW)

    out_path = ROOT / "results" / "EA_test_evaluation.csv"
    _save_csv(rows, out_path)
    _print_summary(rows)

    if failed:
        print(f"\nFailed methods ({len(failed)}): {', '.join(failed)}")

    print(f"\nDone. {len(rows) - 1} EA methods evaluated + GEL v3 reference.\n")


if __name__ == "__main__":
    main()
