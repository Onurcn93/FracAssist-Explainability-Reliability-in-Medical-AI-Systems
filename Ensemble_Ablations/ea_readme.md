# Phase 4 — Ensemble Ablations (EA-Series)

Benchmarking GEL v3 against 20 literature-mapped ensemble methods over the 3 frozen CNN classifiers.  
Supervisor meeting: 2026-05-12 · Next meeting: ~2026-05-26 · Jury: 15 Jul 2026

**Scope:** 3 frozen CNNs (ResNet-18 E6, DenseNet-169 D1, EfficientNet-B3 F1). No model training. No backprop.  
**Input:** `data/all_base.csv` — model probabilities + labels, gel columns stripped.  
**Deliverable:** `results/EA_comparative_table.csv` — all methods vs GEL v3, val metrics, literature references.

---

## Running Experiments

```bash
# Single method
python ensemble_main.py --method EA-A1

# Full family
python ensemble_main.py --family A

# All 20 methods
python ensemble_main.py --all

# Skip plots
python ensemble_main.py --method EA-B2 --no-plot
```

---

## Method Catalogue

### Family A — Averaging (parameter-free floor)

| EA-ID | Method | Type | Anchor |
|---|---|---|---|
| EA-A1 | Mean (sum rule) | LIT | Kuncheva 2014; Müller 2022 |
| EA-A2 | Product rule | LIT | Kittler 1998 |
| EA-A3 | Max rule | LIT | Kittler 1998 |
| EA-A4 | Calibrated mean | ORIG | Guo et al. 2017 |

### Family B — Weighting (isolates GEL's RC stage)

| EA-ID | Method | Type | Anchor |
|---|---|---|---|
| EA-B1 | F1-weighted average | LIT | Kuncheva 2014 |
| EA-B2 | RC power weighting ★ most important ablation | LIT | This thesis (GEL) |
| EA-B3 | AUC-weighted average | LIT | Large & Bagnall 2019 |
| EA-B4 | Confidence-modulated weighting | ORIG | Extends Woloszynski 2011 |
| EA-B5 | Diversity-penalised weighting | ORIG | Extends Brown et al. 2005 |

### Family C — Stacking (learned meta-combiners)

| EA-ID | Method | Type | Anchor |
|---|---|---|---|
| EA-C1 | Logistic regression meta-learner | LIT | Wolpert 1992; Ting & Witten 1999 |
| EA-C2 | Shallow MLP meta-learner | LIT | Müller et al. 2022 |
| EA-C3 | Gradient-boosted trees | LIT | Frontiers SEDL 2023 |
| EA-C4 | Decision-template combiner | LIT | Kuncheva et al. 2001 |
| EA-C5 | Feature-engineered stacking | ORIG | Extends Wolpert 1992 |

### Family D — Gating (isolates GEL's OAM stage)

| EA-ID | Method | Type | Anchor |
|---|---|---|---|
| EA-D1 | Confidence-based selection | LIT | Cruz et al. 2018 |
| EA-D2 | META-DES-style meta-classifier | LIT | Cruz et al. 2015 |
| EA-D3 | Disagreement-triggered gating | ORIG | Extends GEL OAM |
| EA-D4 | Region-of-competence weighting | ORIG | Extends Woloszynski & Kurzynski 2011 |

### Family E — Cascading

| EA-ID | Method | Type | Anchor |
|---|---|---|---|
| EA-E1 | Confidence cascade | LIT | Alpaydin & Kaynak 1998 |
| EA-E2 | Uncertainty-escalation cascade | ORIG | Extends Alpaydin 1998 |

---

## Evaluation Protocol

**Fit on train → score on val → check test. No exceptions.**

- Parameter-free methods: thresholds/temperatures derived from train distribution
- Static weight methods (B1, B3): weights = known val F1/AUC scores (consistent with GEL v3)
- Calibration (EA-A4, Tier 2): temperature T fit on train, frozen before val/test
- Feature parameters (range_reliability, calib_gap): estimated from train, applied to val/test
- Learned combiners (Family C, D): fit on train (3,266 rows) → score val (485 rows) → check test (332 rows)

Val F1 is the primary ranking axis. vs-GEL delta reported for every method.

---

## Feature Engineering

Build enriched features before running Family C/D methods:

```bash
python data/build_features.py --tiers 1 2      # Tier 1+2 (probability + calibration)
python data/build_features.py --tiers 1 2 3 4  # All non-GradCAM tiers
```

| Tier | Features |
|---|---|
| 1 | p_res/dense/eff, p_norm×3, p_mean/max/min/std/range |
| 2 | p_*_calib (temp scaled), margin_*, calib_gap_* |
| 3 ★ | range_reliability_*, ens_confidence, expected_correctness |
| 4 ★ | agree_count, pairwise_disagree, dissent_direction |
| 5 ★ | cam_iou, cam_union_area, cam_consensus_peak (DEFERRED) |

---

## GEL v3 Baseline (fixed reference)

| Split | F1 | AUC | Recall | Precision | Threshold |
|---|---|---|---|---|---|
| Val | 0.7394 | 0.9060 | 0.7439 | 0.7349 | 0.400 |
| Test | 0.6718 | 0.8916 | 0.7213 | 0.6286 | 0.325 |
