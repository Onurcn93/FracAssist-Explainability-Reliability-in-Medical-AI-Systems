"""
gel/gel_pipeline.py — Canonical GEL computation.

All consumers import apply_gel from here.  Change the math once; it propagates
to inference (predict.py), evaluation (eval_gel.py), and offline review
(generate_predictions.py) automatically.

Hyperparameters are read from gel/gel_config.py — single source of truth.
"""

import numpy as np

from gel.gel_config import GEL_CONFIG


def apply_gel(p_r, p_d, p_e=None):
    """RC init -> Asymmetric OAM -> PDWF -> P_final -> BVG gate.

    Works with scalar floats (single-image inference) and numpy arrays
    (vectorized evaluation / batch review).  Returns (p_final, gate_passed)
    in the same type as the inputs — scalars in, scalars out.

    Asymmetric OAM:
      HIGH outlier (p_i > mu, lone fracture signal)   -> k_high (lenient)
        — preserve the fracture signal against a no-frac consensus
      LOW  outlier (p_i < mu, lone no-frac dissenter) -> k_low  (aggressive)
        — protect against a single defector suppressing a fracture consensus

    BVG gate uses P_final (post-OAM), so the gate decision and the
    fracture probability shown to the clinician derive from the same
    calibrated estimate.

    p_r, p_d are required (ResNet-18, DenseNet-169).
    p_e is optional (EfficientNet-B3); pass None for 2-model mode.

    Any model position may be None (e.g. if a weight file failed to load).
    Positions that are None are excluded from all calculations; GEL runs in
    degraded mode on the remaining classifiers.

    YOLO is intentionally excluded from PDWF — detection confidence is not
    a classification probability and cannot be meaningfully mixed with
    softmax outputs in a weighted sum.
    """
    cfg = GEL_CONFIG

    # Build active (prob_array, f1_weight) pairs, skipping unloaded models
    _candidates = [
        (p_r, cfg["gel_f1_resnet"]),
        (p_d, cfg["gel_f1_densenet"]),
        (p_e, cfg["gel_f1_efficientnet"]),
    ]
    pairs = [(np.asarray(p, dtype=float), f1) for p, f1 in _candidates if p is not None]

    probs = [p for p, _ in pairs]
    f1s   = [f1 for _, f1 in pairs]

    # Step 2 — RC initialisation (Expert-Dominant shifted exponential)
    # RC_i = (F1_i + base_shift)^gamma / sum((F1_j + base_shift)^gamma)
    # Amplifies the gap between top and bottom performers; base_shift keeps all
    # values above 1.0 after shifting so the exponent works in the right direction.
    gamma      = cfg["gel_gamma"]
    base_shift = cfg["gel_base_shift"]
    shifted    = [(max(f1, 0.5) + base_shift) ** gamma for f1 in f1s]
    total_shifted = sum(shifted)
    rcs = [np.full_like(p, s / total_shifted) for p, s in zip(probs, shifted)]

    # Step 3 — Asymmetric OAM
    mu = sum(probs) / len(probs)
    new_rcs = []
    for p, rc in zip(probs, rcs):
        outlier = np.abs(p - mu) > cfg["gel_disagree_lim"]
        k = np.where(p < mu, cfg["gel_penalty_k_low"], cfg["gel_penalty_k_high"])
        new_rcs.append(np.where(outlier, rc * k, rc))

    # Step 4 — Performance-Driven Weighted Fusion
    total_rc = sum(new_rcs)
    p_final  = sum(p * rc for p, rc in zip(probs, new_rcs)) / total_rc

    # Step 5 — BVG gate: authenticate YOLO bbox using P_final (post-OAM)
    gate_passed = p_final >= cfg["gel_tau"]

    # Return scalars when scalar inputs were given (0-d numpy array → Python scalar)
    if p_final.ndim == 0:
        return float(p_final), bool(gate_passed)
    return p_final, gate_passed
