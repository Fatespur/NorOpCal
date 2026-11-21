"""PB1-DS PB1 shared library — the frozen publication-benchmark definitions.

This module is the single, pre-registered source of truth for the PB1 protocol
freeze. It is written BEFORE any publication benchmark result is produced and must
not be edited to fit results. It is consumed by the PB1 freeze tool
(``pb1ds_pb1_freeze.py``) and asserted by ``tests/test_pb1ds_pb1.py``.

The frozen objects are:
  - the publication research question + deployment scope + claim boundary;
  - the method-role registry (NO_ADAPT, ALWAYS_ON_QM, A_GLOBAL, D_CATEGORY_ONLY,
    C_CATEGORY_CONDITIONED, ORACLE) and the strong external baseline budget;
  - the primary / co-primary / selector / secondary endpoints;
  - the few-normal K grid + per-K feasibility (data-capacity audited);
  - the target-normal contamination grid + injection rule;
  - the detector plan (PatchCore + DINOv2 primary, EfficientAD feasibility-gated);
  - the AeBAD_S secondary-scoped-evidence policy;
  - the statistical hierarchy (block resampling unit, bootstrap/permutation counts,
    empirical p reporting);
  - the confirmatory/exploratory registry, the publication readiness gates, the
    multi-objective policy-selection hierarchy, the execution order, and the
    publication terminal states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

PB1_OUT = Path("reports/pb1ds_publication/pb1_protocol_freeze")
C1_OUT = Path("reports/pb1ds_publication/c1_category_conditioned")
PUB0_OUT = Path("reports/pb1ds_publication/pub0_attribution_freeze")

# --------------------------------------------------------------------------- #
# Inherited (frozen) upstream terminals — NOT re-interpretable here
# --------------------------------------------------------------------------- #
C1_TERMINAL = "PB1DS_C1_COMPLETE_KNOWN_CATEGORY_SELECTIVE_CALIBRATION_STRONGLY_SUPPORTED"
PUB0_TERMINAL = "PB1DS_PUB0_COMPLETE_CATEGORY_PRIOR_AND_DEFORMATION_BOTH_CONTRIBUTE"
BEST_CANDIDATE = "C_CATEGORY_CONDITIONED"

# --------------------------------------------------------------------------- #
# Publication research question / deployment scope / claim boundary
# --------------------------------------------------------------------------- #
PUBLICATION_QUESTION = (
    "In a deployment setting where the product category is known but acquisition "
    "conditions may change, can a frozen anomaly detector use limited presumed-normal "
    "target observations and known category context to decide when score recalibration "
    "should be applied, improving detection utility while avoiding the severe harm of "
    "unconditional recalibration?"
)
MECHANISM_SUBQUESTION = (
    "Does normal-score deformation provide useful condition-level safety/discrimination "
    "information beyond category priors?"
)
MECHANISM_SUBQUESTION_FORBIDDEN_UPGRADE = (
    "Does deformation independently explain recalibration benefit?"  # NOT claimed
)

ALLOWED_SCOPE: List[str] = [
    "known product category",
    "unknown acquisition condition",
    "frozen anomaly detector",
    "few presumed-normal target observations",
    "dataset-scoped evidence",
    "category-conditioned deployment",
]
FORBIDDEN_CLAIMS: List[str] = [
    "unseen-category generalization",
    "universal selector",
    "category-agnostic selector",
    "dataset-agnostic adaptation",
    "generic domain adaptation",
    "fully unsupervised universal calibration",
    "all-category generalization",
    "arbitrary unseen industrial products",
    "deformation-dominant mechanism",
    "deformation independently explains benefit",
]

STRONGEST_LEGAL_CLAIM = (
    "In a known-product-category / unseen-acquisition-condition deployment setting, "
    "recalibration benefit is strongly category dependent. Category-conditioned "
    "selective recalibration can improve detection utility while avoiding much of the "
    "harm of unconditional recalibration. Normal-score deformation provides additional "
    "discrimination and safety refinement, although its incremental contribution "
    "beyond category priors has not yet been statistically established."
)

# --------------------------------------------------------------------------- #
# Method roles (pre-registered; C is NOT pre-declared the winner)
# --------------------------------------------------------------------------- #
METHOD_ROLES: Dict[str, Dict[str, Any]] = {
    "NO_ADAPT": {
        "function": "h1 = source threshold only, no recalibration",
        "scientific_role": "base operating baseline",
        "selective": False,
        "target_normal_required": False,
    },
    "ALWAYS_ON_QM": {
        "function": "apply frozen quantile-mapping to every target condition",
        "scientific_role": "demonstrate recalibration opportunity AND recalibration harm",
        "selective": False,
        "target_normal_required": True,
    },
    "A_GLOBAL": {
        "function": "p = h(Z), deformation only, no category/detector",
        "scientific_role": "category-agnostic negative control (not a recommended method)",
        "selective": True,
        "target_normal_required": True,
    },
    "D_CATEGORY_ONLY": {
        "function": "p = h(c, d), category one-hot + detector one-hot ONLY, zero deformation",
        "scientific_role": "FIRST-CLASS STRONG category-prior selective baseline",
        "selective": True,
        "target_normal_required": False,  # selector is category/detector only
        "promotion": "must appear in main table, main methods, main discussion (not appendix-only)",
    },
    "C_CATEGORY_CONDITIONED": {
        "function": "p = h(Z, c, d), 10 deformation features + category + detector",
        "scientific_role": "deformation-refined category-conditioned selective candidate",
        "selective": True,
        "target_normal_required": True,
        "expected_strengths": ["discrimination", "harm avoidance", "selectivity", "safety refinement"],
        "not_guaranteed": "must NOT be expected to exceed D on mean Delta_BA",
    },
    "ORACLE": {
        "function": "adapt exactly when benefit is positive (upper bound reference)",
        "scientific_role": "upper bound / decision headroom reference ONLY",
        "selective": True,
        "target_normal_required": False,
        "forbidden": "must NOT participate in parameter/threshold selection",
    },
}

# Frozen method identities (must remain byte/definition equivalent to C1/PUB0)
# ---------------------------------------------------------------
SELECTOR_MODEL = "LogisticRegression(penalty=l2, solver=lbfgs, class_weight=balanced, random_state=42)"
SELECTOR_C_GRID = [0.1, 1.0, 10.0]
SELECTOR_C_SELECTION = "inner (leave-one-category-condition-out) CV, dev-only"
DEFORMATION_FEATURES = [
    "mean_shift", "median_shift", "mad_ratio", "std_ratio",
    "d_q90", "d_q95", "d_q99", "upper_tail_spread_change",
    "wasserstein", "ks",
]
BENEFIT_LABEL = "delta_ba >= +0.02 AND delta_fpr <= -0.05 AND delta_fnr <= +0.02"
HOW_QM = "frozen monotone quantile mapping, source threshold quantile P95"
QM_QUANTILE = 0.95
OPERATING_POINT_RULE = (
    "precision(ADAPT->BENEFIT) >= 0.50; among feasible thresholds maximize mean Delta_BA; "
    "tie-break 1 lower Global Harm, tie-break 2 lower coverage; no valid point => ABSTAIN_ALL"
)
MIN_PRECISION = 0.50
THRESH_GRID = list(round(0.005 * i, 6) for i in range(500, 955, 5))  # 0.500..0.950 step 0.005

# --------------------------------------------------------------------------- #
# Primary / co-primary / secondary endpoints
# --------------------------------------------------------------------------- #
PRIMARY_DOWNSTREAM_ENDPOINT = "mean Delta_BA vs NO_ADAPT (percentage points) + block-bootstrap 95% CI"
CO_PRIMARY_SAFETY_ENDPOINT = "Global Harm (fraction of adapted-and-harmful units)"
RELATIVE_HARM_REDUCTION = "1 - Harm_selective / Harm_AlwaysOn (>0 means harm reduced vs AlwaysOn)"
SELECTOR_ENDPOINTS = ["AUROC", "AUPRC", "AUPRC / benefit prevalence", "ABP", "Benefit Recall", "Coverage"]
SECONDARY_DOWNSTREAM_ENDPOINTS = [
    "Delta FPR", "Delta FNR", "F1", "HAR", "NAR",
    "worst-unit Delta_BA", "median-unit Delta_BA",
    "positive-unit fraction", "negative-unit fraction",
]

# --------------------------------------------------------------------------- #
# Strong baseline set + external baseline budget
# --------------------------------------------------------------------------- #
FIXED_BASELINES = ["NO_ADAPT", "ALWAYS_ON_QM", "A_GLOBAL",
                   "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED", "ORACLE"]
EXTERNAL_BASELINE_BUDGET_MAX = 4
EXTERNAL_BASELINE_CANDIDATES = {
    "BC_ISOTONIC_CALIBRATION": {
        "candidate_role": "reimplemented strong calibration comparator",
        "implementation_source": "isotonic (scikit-learn) on frozen detector source->target score mapping",
        "frozen_detector_ok": True,
        "retrains_detector": False,
        "few_normal_compatible": True,
        "anomaly_label_required": False,
        "final_selection": "PLANNED_IN_FREEZE",
        "reason_for_inclusion": "reviewer-expected score-calibration comparator on frozen detector output",
    },
    "BC_RANK_BASED_THRESHOLD": {
        "candidate_role": "reimplemented re-threshold baseline",
        "implementation_source": "target-normal empirical quantile threshold (source-rank/deformation-free) at frozen risk level",
        "frozen_detector_ok": True,
        "retrains_detector": False,
        "few_normal_compatible": True,
        "anomaly_label_required": False,
        "final_selection": "PLANNED_IN_FREEZE",
        "reason_for_inclusion": "direct target-normal threshold recalibration, few-normal compatible, no new model family",
    },
    "BC_QUANTILE_BINNING": {
        "candidate_role": "reimplemented calibration-binning comparator",
        "implementation_source": "histogram/quantile binning calibrator on frozen detector scores",
        "frozen_detector_ok": True,
        "retrains_detector": False,
        "few_normal_compatible": True,
        "anomaly_label_required": False,
        "final_selection": "PLANNED_IN_FREEZE",
        "reason_for_inclusion": "standard non-parametric calibration comparator; scope-limited to few-normal regime",
    },
    "BC_PER_CATEGORY_THRESHOLD": {
        "candidate_role": "category-aware non-learned baseline",
        "implementation_source": "per-category source-quantile threshold, no target recalibration",
        "frozen_detector_ok": True,
        "retrains_detector": False,
        "few_normal_compatible": False,  # selector needs no target normals, but it is re-threshold only
        "anomaly_label_required": False,
        "final_selection": "PLANNED_IN_FREEZE",
        "reason_for_inclusion": "mild category-aware comparator to bound the value of target recalibration",
    },
}
EXTERNAL_BASELINE_RULE = (
    "At most 4 external strong calibration baselines. Names, implementation sources and "
    "hyperparameters are frozen in PB1 and finalized by an implementation inventory in PB2 "
    "BEFORE any result is seen. No baseline is ever added after results to make C/D look "
    "better; any necessary comparator added later requires a BASELINE_EXCEPTION_LEDGER "
    "entry explaining why its omission would invalidate the publication comparison."
)

# --------------------------------------------------------------------------- #
# Few-normal K protocol (data-capacity decided pre-run)
# --------------------------------------------------------------------------- #
K_GRID = [4, 8, "FULL"]                # FULL = per-entity full target-normal support (audited)
K_16_32_STATUS = (
    "INFEASIBLE if it exceeds the maximum audited target-normal support; "
    "the freeze tool emits the audited min/max support from the data."
)
K_CAPACITY_AUDIT = "per-K eligible entity counts computed by pb1ds_pb1_freeze from the frozen score cache; not hardcoded"
K_REPEATS = 30                      # >= 20 required; 30 frozen
K_SEED_DERIVATION = (
    "SHA256 over canonical string PB1|K|<K_value>|<category>|<condition>|<detector>|<repeat>; "
    "sealed before running; seeds fixed and never swapped after seeing results"
)
K_CELL_INFEASIBLE_RULE = (
    "Any (K, entity, detector) cell whose feature/QM statistics are undefined "
    "(e.g. MAD=0, degenerate percentile) is recorded INFEASIBLE and counted in the "
    "per-K failure / abstain rate; it is NOT dropped silently."
)
K_SELECTOR_VS_QMT_DISTINCTION = (
    "D_CATEGORY_ONLY selector is independent of K (category+detector only), but its "
    "downstream utility depends on K through the recalibration transform and threshold; "
    "must report selector-dependence and recalibration-transform-dependence separately."
)

# --------------------------------------------------------------------------- #
# Target-normal contamination robustness protocol
# --------------------------------------------------------------------------- #
CONTAMINATION_GRID = [0.0, 0.05, 0.10, 0.20]   # prep: fractions
CONTAMINATION_INJECTION = "n_cont = ceil(rho * K) anomalies drawn from the entity's own 15-sample regular-defect set, injected into the target-normal pool"
EFFECTIVE_RATE = "effective rate = n_cont / (K + n_cont) reported alongside nominal rho"
CONTAMINATION_REPEATS = 20
CONTAMINATION_SCOPE = "primary at K=FULL, plus each feasible K cell; 0% = clean control"
CATASTROPHIC_FAILURE_ALERT = ["Delta_BA <= -2pp", "Global Harm >= 10%"]
CONTAMINATION_FORBIDDEN = "contaminated anomalies used ONLY as controlled stress corruption, NEVER for selector/benefit/threshold tuning or method redesign"

# --------------------------------------------------------------------------- #
# Detector protocol
# --------------------------------------------------------------------------- #
PRIMARY_DETECTORS = ["PatchCore (wide_resnet50_2)", "DINOv2 (dinov2_vits14)"]
EFFICIENTAD_POLICY = {
    "status": "THIRD_DETECTOR_CANDIDATE",
    "gate": "IMPLEMENTATION_FEASIBILITY_AUDIT (real execution, real scores, same dataset protocol, no leakage)",
    "historical_note": "historical simulated EfficientAD evidence must NOT be used as formal third-detector evidence",
    "if_feasible": "include as THIRD DETECTOR",
    "if_infeasible": "mark NOT_VERIFIED (incompatible pipeline / missing implementation / unrecoverable preprocessing / resource infeasibility); never substituted by simulation",
    "stage": 6,
}

# --------------------------------------------------------------------------- #
# AeBAD_S secondary evidence policy
# --------------------------------------------------------------------------- #
AEBAD_POLICY = {
    "role": "SECONDARY SCOPED EVIDENCE",
    "not": "forced reproduction of the main result",
    "not_required": ["same effect size", "same category dependency", "same optimal policy", "same benefit prevalence"],
    "questions": ["does acquisition-shift calibration risk recur?", "does the frozen policy have finite secondary support?", "describe failure/success boundary"],
    "if_consistent": "secondary supportive evidence",
    "if_not": "dataset-specific limitation (no method rescue for AeBAD_S)",
}

# --------------------------------------------------------------------------- #
# Statistical hierarchy
# --------------------------------------------------------------------------- #
RESAMPLING_UNIT = "category-condition block (row-wise iid bootstrap FORBIDDEN)"
BOOTSTRAP_REPLICATES = 5000
PERMUTATION_REPLICATES = 5000
P_VALUE_REPORTING = "empirical p = (k+1)/(B+1); never report p=0.000; with B=5000 min reportable p ~= 0.0002; report exact empirical value"

# --------------------------------------------------------------------------- #
# Confirmatory vs exploratory registry
# --------------------------------------------------------------------------- #
CONFIRMATORY_ITEMS = [
    "main C/D strong-baseline benchmark", "few-normal K sensitivity",
    "contamination robustness", "per-category / per-condition / per-detector transparency",
    "primary statistical closure", "detector robustness (primary detectors)",
]
EXPLORATORY_OR_SECONDARY_ITEMS = [
    "EfficientAD (if added after feasibility audit)",
    "AeBAD_S secondary scoped evidence",
    "failure-case visualization", "additional diagnostic score geometry",
]
MIX_RULE = "exploratory positive results must NOT be mixed into the confirmatory primary claim"

# --------------------------------------------------------------------------- #
# Publication readiness gates (G1-G7, frozen pre-benchmark)
# --------------------------------------------------------------------------- #
GATES: Dict[str, str] = {
    "G1_MAIN_UTILITY": "at least one frozen deployment policy has mean Delta_BA > 0 and (preferred) 95% CI lower bound >= 0",
    "G2_SAFETY": "at least one policy has Global Harm << AlwaysOn; STRONG = Relative Harm Reduction >= 50%",
    "G3_DETECTOR_CONSISTENCY": "PatchCore Delta_BA >= 0 AND DINOv2 Delta_BA >= 0; if one is negative, restrict to detector-specific claim",
    "G4_FEW_NORMAL_VIABILITY": "a practical K (<=16; here K=8 or K=4) preserves positive utility and substantial harm reduction; only-FULL success => few-normal claim FAILS",
    "G5_CONTAMINATION_SAFETY": "no systematic catastrophic failure at >=5% contamination; STRONG = positive utility or safe abstention at 10%",
    "G6_CATEGORY_TRANSPARENCY": "report ALL categories and ALL held-out conditions without selective omission",
    "G7_PRIOR_VS_DEFORMATION_HONESTY": "category-prior-dominant / mixed mechanism retained unless new confirmatory evidence genuinely establishes a deformation-independent increment; PUB0 must not be re-interpreted",
}

# --------------------------------------------------------------------------- #
# Policy selection hierarchy (multi-objective, frozen; prevents result-driven picking)
# --------------------------------------------------------------------------- #
SELECTION_HIERARCHY = [
    "Level1-SAFETY_ADMISSIBILITY: candidate must satisfy Global Harm < AlwaysOn Global Harm and show substantial harm reduction",
    "Level2-POSITIVE_UTILITY: mean Delta_BA > 0, preferred 95% CI lower bound >= 0; CI crossing 0 => incremental improvement not statistically established",
    "Level3-DEPLOYMENT_USEFULNESS: Coverage, Benefit Recall, ABP, HAR, NAR (never only Delta_BA)",
    "Level4-STABILITY: few-normal K robustness, contamination robustness, detector robustness, condition robustness, category transparency",
]
PRIMARY_POLICY_OUTCOMES = [
    "P-D: D_CATEGORY_ONLY is primary deployment policy, C is safety-refinement secondary method",
    "P-C: C_CATEGORY_CONDITIONED is primary deployment policy, D is strong category-prior baseline",
    "P-DUAL: D and C form a utility-safety Pareto pair",
]
POLICY_OUTCOME_RULE = "no primary-policy outcome is pre-declared; it is determined only by the multi-objective hierarchy after benchmark results"

# --------------------------------------------------------------------------- #
# Execution order + publication terminals
# --------------------------------------------------------------------------- #
EXECUTION_ORDER = [
    "PB2 Main Strong-Baseline Benchmark",
    "PB3 Few-Normal K Sensitivity",
    "PB4 Target-Normal Contamination Robustness",
    "PB5 Per-Category / Per-Condition Transparency Closure",
    "PB6 Detector Robustness + EfficientAD Feasibility/Execution",
    "PB7 AeBAD_S Secondary Scoped Evidence",
    "PB8 Final Statistical Closure",
    "PB9 Publication Viability & Claim Freeze",
]

PUBLICATION_TERMINALS = {
    "PB1DS_PB_COMPLETE_PUBLICATION_READY_STRONG": "positive main utility, strong harm reduction, few-normal viability, detector consistency, acceptable contamination robustness, transparent category-dependent limitations",
    "PB1DS_PB_COMPLETE_PUBLICATION_READY_SCOPED": "strong on selected known categories / detector-specific / moderate-K requirement but still a defensible paper",
    "PB1DS_PB_COMPLETE_SAFETY_VALUE_SUPPORTED_PERFORMANCE_CLAIM_LIMITED": "main contribution is avoiding harmful recalibration, not increasing aggregate detection performance",
    "PB1DS_PB_COMPLETE_CURRENT_EVIDENCE_NOT_PUBLICATION_SUFFICIENT": "do not open C2, do not add a new selector, do not rescue via feature search; re-evaluate the research route",
    "PB1DS_PB_STOP_EXECUTION_INTEGRITY_FAILURE": "execution integrity failure",
}

# --------------------------------------------------------------------------- #
# Compute budget model (frozen estimate; relies on cached frozen detector scores)
# --------------------------------------------------------------------------- #
COMPUTE_BUDGET = {
    "reuse_frozen_detector_score_caches": True,
    "no_detector_feature_recompute": 78,  # frozen entity-level score arrays reused across all stages
    "evaluation_units": 78,               # (category, condition, detector)
    "candidates_compared": 6,             # NO_ADAPT, ALWAYS_ON_QM, A_GLOBAL, D, C, ORACLE
    "external_baselines_cap": 4,
    "pb3_k_sensitivity": {
        "filename": "PB3",
        "k_values": 3, "repeats_per_k": 30, "selectors": 2,
        "cells": "evaluated on K-eligible entity x detector x selector x repeat; ~78*2*3*30 worst-case core-feature recompute on cached scores",
    },
    "pb4_contamination": {
        "filename": "PB4",
        "rho_levels": 4, "repeats_per_level": 20, "injection": "ceil-based",
        "cells": "on feasible K cells only; ~K-eligible entity x detector x selector x rho x 20",
    },
    "pb6_efficientad": {"filename": "PB6", "gate": "IMPLEMENTATION_FEASIBILITY_AUDIT before any execution"},
    "dominant_cost": "repeated core-feature + QM recompute over frozen score arrays; NOT detector re-extraction",
}


def seed_identity(k_value: Any, category: str, condition: str, detector: str,
                  repeat: int) -> str:
    """Deterministic SHA256-based seed identity for K sampling / contamination."""
    return f"PB1|K|{k_value}|{category}|{condition}|{detector}|{repeat}"