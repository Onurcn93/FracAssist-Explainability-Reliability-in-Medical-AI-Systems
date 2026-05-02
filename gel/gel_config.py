"""
gel/gel_config.py — Single source of truth for all GEL hyperparameters.

All downstream consumers (FracAssist_Inference/config.py, utils/eval_gel.py,
review/generate_predictions.py, FracAssist_Inference/app.py via /api/gel-config)
import from this module.  Change a value here once; it propagates everywhere.

Pipeline order: RC init -> Asymmetric OAM -> PDWF -> P_final -> BVG gate.

OAM (Outlier-Aware Modification) — Asymmetric:
  mu = mean of loaded classifier probabilities
  HIGH outlier (p_i > mu): lone fracture signal       -> RC_i_adj = RC_i * gel_penalty_k_high (lenient)
  LOW  outlier (p_i < mu): lone no-frac dissenter     -> RC_i_adj = RC_i * gel_penalty_k_low  (aggressive)
  gel_penalty_k_standard (0.20) preserved as symmetric / balanced reference.

PDWF (Performance-Driven Weighted Fusion):
  P_final = sum(P_i * RC_i_adj) / sum(RC_i_adj)   [loaded classifiers only]

BVG (Binary Verification Gate) — uses P_final, not a separate pre-OAM average:
  If P_final < gel_tau -> gate fails -> bbox suppressed (P_final still shown).

GEL adapts to 2 or 3 loaded classifiers automatically.

Performance anchors reflect val-sweep results (2026-04-25).
YOLO is intentionally excluded — detection mAP is not comparable to
classification F1 and must not enter the PDWF weighted sum.
"""

GEL_CONFIG = {
    # Performance anchors — update when champion weights change
    "gel_f1_resnet":          0.689,   # E6 val F1 (CAALMIX champion)
    "gel_f1_densenet":        0.724,   # D1 val F1
    "gel_f1_efficientnet":    0.671,   # F1 val F1 (confirmed 2026-04-28 — val F1=0.6707 @ thr=0.525)

    # BVG gate threshold — below this, YOLO bbox is suppressed
    "gel_tau":                0.35,

    # OAM — disagreement limit and asymmetric penalty factors
    "gel_disagree_lim":       0.40,
    "gel_penalty_k_low":      0.10,    # LOW outlier  — aggressive: lone no-frac dissenter against fracture consensus
    "gel_penalty_k_high":     0.30,    # HIGH outlier — lenient:    lone fracture signal against no-frac consensus
    "gel_penalty_k_standard": 0.20,    # Balanced symmetric reference (not used in OAM — preserved for comparison)
}
