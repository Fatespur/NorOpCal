"""PB1-DS PB1F — Final Frozen-Protocol Binding (PB1F) shared library.

PB1F is the FINAL consistency-closure before PB2. It does not redesign PB1; it
closes the last possible post-hoc loopholes and uniquely binds the PB2 input:

  F0   read-only upstream audit of C1 / PUB0 / PB1;
  F1   baseline-registry closure (concrete external baselines + hyperparameters + feasibility);
  F2   canonical PB2 support binding (PB2 = FULL == the exact regime that produced C1/PUB0);
  F3   PB2 input-binding manifest (frozen inputs with paths/hashes/schemas/roles);
  F4   statistical binding (block bootstrap 5000, permutation 5000, empirical p).

Everything here is written BEFORE any PB2 benchmark result is produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

PB1F_OUT = Path("reports/pb1ds_publication/pb1f_final_binding")
PB2_OUT = Path("reports/pb1ds_publication/pb2_main_benchmark")
PB1_OUT = Path("reports/pb1ds_publication/pb1_protocol_freeze")
PUB0_OUT = Path("reports/pb1ds_publication/pub0_attribution_freeze")
C1_OUT = Path("reports/pb1ds_publication/c1_category_conditioned")

# --------------------------------------------------------------------------- #
# PB1F terminal states (mutually exclusive; exactly one is emitted)
# --------------------------------------------------------------------------- #
TERM_OK = "PB1DS_PB1F_COMPLETE_FINAL_PROTOCOL_BINDING_VERIFIED"
TERM_BASELINE = "PB1DS_PB1F_STOP_BASELINE_REGISTRY_NOT_CLOSED"
TERM_SUPPORT = "PB1DS_PB1F_STOP_CANONICAL_SUPPORT_NOT_UNIQUELY_BOUND"
TERM_BOUNDARY = "PB1DS_PB1F_STOP_BENCHMARK_STAGE_BOUNDARY_AMBIGUOUS"

# --------------------------------------------------------------------------- #
# Inherited frozen terminals (never re-interpreted here)
# --------------------------------------------------------------------------- #
PB1_TERMINAL = "PB1DS_PB1_COMPLETE_PUBLICATION_BENCHMARK_PROTOCOL_FROZEN"
PUB0_TERMINAL = "PB1DS_PUB0_COMPLETE_CATEGORY_PRIOR_AND_DEFORMATION_BOTH_CONTRIBUTE"
C1_TERMINAL = "PB1DS_C1_COMPLETE_KNOWN_CATEGORY_SELECTIVE_CALIBRATION_STRONGLY_SUPPORTED"

# Frozen membership facts (verified by F0, not assumed)
N_CATEGORY_CONDITION_UNITS = 39
N_EXPECTED_DETECTOR_UNIT_ROWS = 78
N_UNIQUE_ENTITIES = 78

# --------------------------------------------------------------------------- #
# F2 — canonical PB2 support binding
# --------------------------------------------------------------------------- #
# Historical fact verified from the C1/PUB0 execution: the frozen C1/PUB0
# scientific object was produced by using ALL available presumed-normal target
# scores per (category, condition, detector) — NO cap K, NO sub-sample, NO seed.
CANONICAL_SUPPORT_DEFINITION = (
    "FULL: use ALL legally-available presumed-normal target image scores for the "
    "(category, condition, detector) unit, exactly as the frozen C1/PUB0 OOF was "
    "produced (no cap, no subsample, no seed)."
)
CANONICAL_SUPPORT_RULE = "PB2 uses the exact frozen canonical support regime == FULL(entity)"
CANONICAL_SUPPORT_SIZE_RULE = "per-entity FULL_n == n_available_target_normal (from the frozen C1 unit manifest)"
SAMPLING_RULE = "none (all scores used; no subsampling in the canonical C1/PUB0 regime)"
SEED_RULE = "none required at canonical FULL support (no sampling)"

# --------------------------------------------------------------------------- #
# F4 — statistical binding (paired block bootstrap + permutation, PB1 frozen)
# --------------------------------------------------------------------------- #
BOOTSTRAP_REPLICATES = 5000
PERMUTATION_REPLICATES = 5000
BOOTSTRAP_BLOCK = "category-condition block (both detectors resampled together)"
PAIRED_COMPARISONS = [
    "D_vs_NO_ADAPT", "C_vs_NO_ADAPT", "D_vs_ALWAYS_ON", "C_vs_ALWAYS_ON",
    "C_vs_D", "D_vs_STRONGEST_EXTERNAL", "C_vs_STRONGEST_EXTERNAL",
]
PERMUTATION_SELECTORS = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
P_VALUE_RULE = "empirical p = (k+1)/(B+1); never report p=0.000"

# --------------------------------------------------------------------------- #
# Primary / safety endpoints bound for PB2 (frozen in PB1, asserted here)
# --------------------------------------------------------------------------- #
PRIMARY_DOWNSTREAM_ENDPOINT = "mean Delta_BA vs NO_ADAPT (percentage points) + block-bootstrap 95% CI"
CO_PRIMARY_SAFETY_ENDPOINT = "Global Harm (fraction of adapted-and-harmful units)"
RELATIVE_HARM_REDUCTION = "Relative Harm Reduction vs ALWAYS_ON = 1 - Harm_selective/Harm_AlwaysOn"

# --------------------------------------------------------------------------- #
# F1 — external-baseline closure (concrete, hyperparameter-finalized, feasibility)
# --------------------------------------------------------------------------- #
EXTERNAL_BASELINE_CLOSURE: Dict[str, Dict[str, Any]] = {
    "BC_ISOTONIC_CALIBRATION": {
        "baseline_name": "BC_ISOTONIC_CALIBRATION",
        "scientific_role": "strong learned calibration comparator (unconditional source-alignment)",
        "algorithm_definition": "isotonic regression map target-normal->source; apply to test scores; threshold Q95(source)",
        "implementation": "sklearn.isotonic.IsotonicRegression",
        "hyperparameters": {"increasing": True, "out_of_bounds": "clip",
                            "fit_x": "target_normal_sorted", "fit_y": "source_normal_sorted",
                            "threshold": "Q95(source)", "operating": "always_on"},
        "requires_target_normal": True,
        "requires_anomaly_label": False,
        "requires_detector_retraining": False,
        "feasibility": "READY",
        "primary_or_secondary": "external_strong_candidate",
        "reason_for_inclusion": "reviewer-expected score-calibration comparator on frozen detector output",
    },
    "BC_RANK_BASED_THRESHOLD": {
        "baseline_name": "BC_RANK_BASED_THRESHOLD",
        "scientific_role": "deformation-free target-normal re-threshold baseline (unconditional)",
        "algorithm_definition": "per-entity detection threshold = Q95(target-normal scores); no score remap",
        "implementation": "numpy percentile over target-normal; no learned model",
        "hyperparameters": {"risk_quantile": 0.95, "operating": "always_on"},
        "requires_target_normal": True,
        "requires_anomaly_label": False,
        "requires_detector_retraining": False,
        "feasibility": "READY",
        "primary_or_secondary": "external_strong_candidate",
        "reason_for_inclusion": "direct target-normal threshold recalibration, few-normal compatible, no new model family",
    },
    "BC_QUANTILE_BINNING": {
        "baseline_name": "BC_QUANTILE_BINNING",
        "scientific_role": "standard non-parametric calibration-binning comparator (unconditional)",
        "algorithm_definition": "quantile-binning map target->source (n_bins); apply to test scores; threshold Q95(source)",
        "implementation": "numpy searchsorted over emp. target quantile bin edges; no learned model",
        "hyperparameters": {"n_bins": 20, "threshold": "Q95(source)", "operating": "always_on"},
        "requires_target_normal": True,
        "requires_anomaly_label": False,
        "requires_detector_retraining": False,
        "feasibility": "READY",
        "primary_or_secondary": "external_strong_candidate",
        "reason_for_inclusion": "standard non-parametric calibration comparator; scope-limited to few-normal regime",
    },
    "BC_PER_CATEGORY_THRESHOLD": {
        "baseline_name": "BC_PER_CATEGORY_THRESHOLD",
        "scientific_role": "category-aware zero-target-recalibration comparator (bounds value of target recalibration)",
        "algorithm_definition": "per-(category,detector) source-quantile threshold applied to every unit; no target adaptation",
        "implementation": "numpy percentile over source regular scores; no learned model",
        "hyperparameters": {"risk_quantile": 0.95, "scope": "per_(category,detector)_source",
                            "operating": "always_on"},
        "requires_target_normal": False,
        "requires_anomaly_label": False,
        "requires_detector_retraining": False,
        "feasibility": "READY",
        "primary_or_secondary": "external_strong_candidate",
        "reason_for_inclusion": "mild category-aware comparator to bound the value of target recalibration",
    },
}

# PB2 fixed + external method set (order of the main table; ORACLE last, excluded from ranking)
MAIN_TABLE_ORDER = [
    "NO_ADAPT", "ALWAYS_ON_QM",
    "BC_ISOTONIC_CALIBRATION", "BC_RANK_BASED_THRESHOLD",
    "BC_QUANTILE_BINNING", "BC_PER_CATEGORY_THRESHOLD",
    "A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED", "ORACLE",
]
PRACTICAL_COMPARATORS = ["NO_ADAPT", "ALWAYS_ON_QM"] + list(EXTERNAL_BASELINE_CLOSURE.keys())

# --------------------------------------------------------------------------- #
# PB2 gates (interim; final publication gates G1-G7 remain in PB9)
# --------------------------------------------------------------------------- #
PB2_GATES: Dict[str, str] = {
    "G2-1_MAIN_UTILITY": "at least one of D/C has mean Delta_BA > 0; STRONG = 95% CI lower bound >= 0",
    "G2-2_SAFETY": "at least one of D/C has Global Harm < AlwaysOn; STRONG = Relative Harm Reduction >= 50%",
    "G2-3_STRONG_BASELINE_COMPETITIVENESS": "at least one of D/C shows material utility OR safety OR utility-safety Pareto advantage over the strongest practical non-oracle baseline",
    "G2-4_DETECTOR_NON_COLLAPSE": "PatchCore pooled Delta_BA >= 0 AND DINOv2 pooled Delta_BA >= 0; else DETECTOR_SCOPED_WARNING",
    "G2-5_TRANSPARENCY": "all 39 (category,condition) units, all 78 entities, all categories, all conditions reported",
}

# --------------------------------------------------------------------------- #
# PB2 terminal states
# --------------------------------------------------------------------------- #
PB2_TERMINALS = {
    "PB1DS_PB2_COMPLETE_MAIN_BENCHMARK_STRONGLY_SUPPORTED": "positive utility, strong harm reduction, competitive vs frozen strong baselines, no major detector collapse",
    "PB1DS_PB2_COMPLETE_UTILITY_SAFETY_PARETO_SUPPORTED": "D higher recall/utility, C lower harm/better selectivity -> utility-safety Pareto",
    "PB1DS_PB2_COMPLETE_MAIN_BENCHMARK_SCOPED_SUPPORT": "e.g. one detector weak, external baseline competitive, or effect concentrated in limited categories; warning retained",
    "PB1DS_PB2_COMPLETE_STRONG_BASELINE_CHALLENGES_NOVELTY": "external strong baseline largely explains or exceeds D/C; novelty must be downgraded but PB3/PB4 still allowed",
    "PB1DS_PB2_COMPLETE_MAIN_BENCHMARK_NOT_SUPPORTED": "no new selector / no C-D modification / no feature search / no strong-baseline removal",
    "PB1DS_PB2_STOP_EXECUTION_INTEGRITY_FAILURE": "execution integrity failure",
}

# Frozen interpretation framework (Case U/S/P/E/N)
INTERPRETATION_CASES = {
    "U": "D utility dominant (D strongest deployment utility; C provides safety refinement)",
    "S": "C safety dominant (C comparable utility with materially lower harm / better discrimination)",
    "P": "Pareto pair (D better recall/utility, C better precision/safety)",
    "E": "external baseline competitive (matches/exceeds D/C; re-evaluate novelty)",
    "N": "no meaningful advantage (D/C fail to beat reasonable strong baselines; scoped negative)",
}

# Frozen strongest-external rule (pre-registered, no best-baseline fishing):
# among safety-admissible external baselines, prefer positive Delta_BA -> lowest
# Global Harm -> highest mean Delta_BA.
STRONGEST_EXTERNAL_RULE = (
    "Among safety-admissible external baselines (Global Harm < AlwaysOn), prefer "
    "positive Delta_BA; then lowest Global Harm; then highest mean Delta_BA. "
    "Never pick the baseline post-hoc to make C/D look strongest."
)