"""
ensemble_main.py

Entry point for all Ensemble Ablations (EA-series) experiments.
Run from inside the Ensemble_Ablations/ directory.

Usage:
    python ensemble_main.py --method EA-A1
    python ensemble_main.py --method EA-A1 --no-plot
    python ensemble_main.py --family A
    python ensemble_main.py --family B
    python ensemble_main.py --all
    python ensemble_main.py --all --seed 99

Each method script exposes run(root_dir, seed, plot) and returns a metrics dict.
Results are appended to results/EA_comparative_table.csv automatically.
"""

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Method registry — EA-ID → (module path, display name)
# --------------------------------------------------------------------------- #
REGISTRY = {
    # Family A — Averaging
    "EA-A1": ("EA_A_Averaging.EA_A1_mean",             "Mean (sum rule)"),
    "EA-A2": ("EA_A_Averaging.EA_A2_product",          "Product rule"),
    "EA-A3": ("EA_A_Averaging.EA_A3_max",              "Max rule"),
    "EA-A4": ("EA_A_Averaging.EA_A4_calibrated_mean",  "Calibrated mean"),
    # Family B — Weighting
    "EA-B1": ("EA_B_Weighting.EA_B1_f1_weighted",          "F1-weighted average"),
    "EA-B2": ("EA_B_Weighting.EA_B2_rc_power",             "RC power weighting"),
    "EA-B3": ("EA_B_Weighting.EA_B3_auc_weighted",         "AUC-weighted average"),
    "EA-B4": ("EA_B_Weighting.EA_B4_confidence_modulated", "Confidence-modulated weighting"),
    "EA-B5": ("EA_B_Weighting.EA_B5_diversity_penalised",  "Diversity-penalised weighting"),
    # Family C — Stacking
    "EA-C1": ("EA_C_Stacking.EA_C1_logreg",               "Logistic regression meta-learner"),
    "EA-C2": ("EA_C_Stacking.EA_C2_shallow_mlp",          "Shallow MLP meta-learner"),
    "EA-C3": ("EA_C_Stacking.EA_C3_gbt",                  "Gradient-boosted trees"),
    "EA-C4": ("EA_C_Stacking.EA_C4_decision_template",    "Decision-template combiner"),
    "EA-C5": ("EA_C_Stacking.EA_C5_feature_engineered",   "Feature-engineered stacking"),
    # Family D — Gating
    "EA-D1": ("EA_D_Gating.EA_D1_confidence_select",    "Confidence-based selection"),
    "EA-D2": ("EA_D_Gating.EA_D2_meta_des",             "META-DES-style meta-classifier"),
    "EA-D3": ("EA_D_Gating.EA_D3_disagreement_gating",  "Disagreement-triggered gating"),
    "EA-D4": ("EA_D_Gating.EA_D4_region_competence",    "Region-of-competence weighting"),
    "EA-D5": ("EA_D_Gating.EA_D5_body_part_competence", "Body-part-constrained competence"),
    # Family E — Cascading
    "EA-E1": ("EA_E_Cascading.EA_E1_confidence_cascade",       "Confidence cascade"),
    "EA-E2": ("EA_E_Cascading.EA_E2_uncertainty_escalation",   "Uncertainty-escalation cascade"),
}

FAMILIES = {
    "A": [k for k in REGISTRY if k.startswith("EA-A")],
    "B": [k for k in REGISTRY if k.startswith("EA-B")],
    "C": [k for k in REGISTRY if k.startswith("EA-C")],
    "D": [k for k in REGISTRY if k.startswith("EA-D")],
    "E": [k for k in REGISTRY if k.startswith("EA-E")],
}


def _ensure_dirs():
    (ROOT / "plots").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "results" / "per_method").mkdir(parents=True, exist_ok=True)


def _run_method(ea_id: str, seed: int, plot: bool) -> dict:
    if ea_id not in REGISTRY:
        print(f"[error] Unknown method: {ea_id}. Available: {list(REGISTRY)}")
        sys.exit(1)

    module_path, display = REGISTRY[ea_id]
    print(f"\n{'='*72}")
    print(f"  {ea_id}: {display}")
    print(f"{'='*72}\n")

    sys.path.insert(0, str(ROOT))
    module = importlib.import_module(module_path)
    return module.run(root_dir=ROOT, seed=seed, plot=plot)


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Ablations (EA-series) experiment runner"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--method", type=str, help="Run a single method, e.g. EA-A1")
    group.add_argument("--family", type=str, choices=list(FAMILIES), help="Run all methods in a family (A/B/C/D/E)")
    group.add_argument("--all", action="store_true", help="Run all methods in the registry (note: use ea_test_set_evaluation.py for the full 25-method test-set run)")
    parser.add_argument("--seed",    type=int,  default=42,    help="Random seed (default: 42)")
    parser.add_argument("--no-plot", action="store_true",      help="Skip plot generation")
    args = parser.parse_args()

    _ensure_dirs()
    plot = not args.no_plot

    if args.method:
        methods = [args.method.upper()]
    elif args.family:
        methods = FAMILIES[args.family.upper()]
    else:
        methods = list(REGISTRY)

    results = {}
    for ea_id in methods:
        try:
            results[ea_id] = _run_method(ea_id, seed=args.seed, plot=plot)
        except NotImplementedError:
            print(f"  [{ea_id}] not yet implemented — skipped")

    print(f"\n{'='*72}")
    print(f"  Done. {len(results)} method(s) completed.")
    print(f"  Results -> results/EA_comparative_table.csv")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
