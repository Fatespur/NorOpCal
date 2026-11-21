"""PB1-DS PB2 shared library — frozen method-equipment set, external-baseline
actuators, and deterministic adjudication rule for the Main Strong-Baseline
Benchmark.

Inherits the PB1F final binding directly (no re-interpretation). PB2:
  - REUSES the sealed C1/PUB0 OOF for NO_ADAPT, ALWAYS_ON_QM, A_GLOBAL,
    D_CATEGORY_ONLY, C_CATEGORY_CONDITIONED, ORACLE (no refit / no threshold
    retune / no reselection); controls are policy applications of the frozen
    quantile-mapping frame exactly as in PUB0.
  - COMPUTES the four sealed external baselines (BC_ISOTONIC_CALIBRATION,
    BC_RANK_BASED_THRESHOLD, BC_QUANTILE_BINNING, BC_PER_CATEGORY_THRESHOLD)
    as unconditional (always_on) recalibrators from raw frozen detector scores
    under the canonical FULL support, re-deriving downstream metrics with the
    SAME frozen benefit taxonomy and the SAME default source threshold.

Everything here is a frozen, result-independent execution definition.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import md0sr_common as C  # raw score io (source/cond/anomaly rels, detection_metrics)
import pb1ds_c1_common as M  # score loading / detection metrics / QM thes
import pb1ds_pb1_common as P  # frozen shared protocol definitions
import pb1ds_pb1f_common as F  # final binding (baselines, gates, terminals)

PB2_OUT = F.PB2_OUT
EXTERNAL_BASELINE_CLOSURE = F.EXTERNAL_BASELINE_CLOSURE
MAIN_TABLE_ORDER = F.MAIN_TABLE_ORDER
PRACTICAL_COMPARATORS = F.PRACTICAL_COMPARATORS
BOOTSTRAP_REPLICATES = F.BOOTSTRAP_REPLICATES
PERMUTATION_REPLICATES = F.PERMUTATION_REPLICATES

# Evaluation methods (role-classified). ORACLE is upper-bound reference only and
# is excluded from any ranking/selection.
FROZEN_SELECTOR_METHODS: List[str] = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
CONTROL_METHODS: List[str] = ["NO_ADAPT", "ALWAYS_ON_QM", "ORACLE"]
EXTERNAL_METHODS: List[str] = list(EXTERNAL_BASELINE_CLOSURE.keys())
SELECTOR_METHODS: List[str] = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
ALL_METHODS: List[str] = list(MAIN_TABLE_ORDER)

BENEFIT_LABEL = P.BENEFIT_LABEL
HOW_QM = P.HOW_QM

# --------------------------------------------------------------------------- #
# External-baseline actuator definitions (deterministic, no leakage)
# --------------------------------------------------------------------------- #
ActuatorOut = Tuple[np.ndarray, np.ndarray, float]  # (mapped_normal, mapped_anom, tau_after)


def _default_before(cnd: np.ndarray, anm: np.ndarray, tau: float) -> Dict[str, float]:
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    return C.detection_metrics(np.concatenate([cnd, anm]), labels, tau)


def actuator_bc_rank_threshold(src: np.ndarray, cnd: np.ndarray,
                               anm: np.ndarray, **kw: Any) -> ActuatorOut:
    """Per-entity threshold = Q95(target-normal); no score remap (always_on)."""
    risk_q = float(kw.get("risk_quantile", 0.95))
    tau = float(np.percentile(cnd, 100.0 * risk_q))
    return cnd, anm, tau


def actuator_bc_per_category_threshold(src: np.ndarray, cnd: np.ndarray,
                                       anm: np.ndarray, **kw: Any) -> ActuatorOut:
    """Per-(category,detector) source-quantile threshold; no target adaptation.
    Degenerate by construction: source is category-scoped so tau == default."""
    risk_q = float(kw.get("risk_quantile", 0.95))
    tau = float(np.percentile(src, 100.0 * risk_q))
    return cnd, anm, tau


def actuator_bc_isotonic_calibration(src: np.ndarray, cnd: np.ndarray,
                                     anm: np.ndarray, **kw: Any) -> ActuatorOut:
    """Isotonic regression target-ranks -> source-quantiles; apply to anomaly;
    threshold Q95(source)."""
    from sklearn.isotonic import IsotonicRegression
    threshold = float(np.percentile(src, 95.0))
    n_pair = min(len(cnd), len(src))
    qs = (np.arange(1, n_pair + 1) - 0.5) / n_pair
    x = np.quantile(cnd, qs)
    y = np.quantile(src, qs)
    iso = IsotonicRegression(increasing=bool(kw.get("increasing", True)),
                             out_of_bounds=str(kw.get("out_of_bounds", "clip")))
    iso.fit(x, y)
    return iso.predict(cnd), iso.predict(anm), threshold


def map_qbin(value: np.ndarray, edges: np.ndarray, src_vals: np.ndarray) -> np.ndarray:
    """Step-wise monotone quantile-binning map target->source."""
    b = np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(src_vals) - 1)
    return src_vals[b]


def actuator_bc_quantile_binning(src: np.ndarray, cnd: np.ndarray,
                                 anm: np.ndarray, **kw: Any) -> ActuatorOut:
    """Empirical target quantile-binning map (n_bins) -> source; threshold Q95(src)."""
    n_bins = int(kw.get("n_bins", 20))
    threshold = float(np.percentile(src, 95.0))
    edges = np.quantile(cnd, np.linspace(0.0, 1.0, n_bins + 1))
    src_vals = np.quantile(src, (np.arange(n_bins) + 0.5) / n_bins)
    return map_qbin(cnd, edges, src_vals), map_qbin(anm, edges, src_vals), threshold


ACTUATORS: Dict[str, Callable[..., ActuatorOut]] = {
    "BC_RANK_BASED_THRESHOLD": actuator_bc_rank_threshold,
    "BC_PER_CATEGORY_THRESHOLD": actuator_bc_per_category_threshold,
    "BC_ISOTONIC_CALIBRATION": actuator_bc_isotonic_calibration,
    "BC_QUANTILE_BINNING": actuator_bc_quantile_binning,
}


def external_unit_row(method: str, cat: str, cond: str, det: str,
                      guard: Any = None) -> Dict[str, Any]:
    """Per-entity downstream row for an external (always_on) recalibrator.

    'before' uses the canonical default threshold Q95(source), removed/original
    scores — identical to the sealed frozen 'before' row (integrity-checked).
    """
    src = C.load_image_scores(C.source_rel(cat, det))
    cnd = C.load_image_scores(C.cond_normal_rel(cat, cond, det))
    anm = C.load_image_scores(C.anomaly_rel(cat, det))
    cfg = EXTERNAL_BASELINE_CLOSURE[method]
    after_n, after_a, tau_after = ACTUATORS[method](src, cnd, anm,
                                                    **(cfg["hyperparameters"] if "hyperparameters" in cfg else {}))
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    mb = _default_before(cnd, anm, float(np.percentile(src, 95.0)))
    ma = C.detection_metrics(np.concatenate([after_n, after_a]), labels, tau_after)
    benefit, harm, neutral = M.classify_change(ma["ba"] - mb["ba"],
                                               ma["fpr"] - mb["fpr"],
                                               ma["fnr"] - mb["fnr"])
    return {
        "category": cat, "condition": cond, "baseline": det,
        "baseline_name": M.BASELINE_NAMES[det],
        "method": method, "adapt": 1, "selector_prob": float("nan"),
        "ba_before": mb["ba"], "ba_after": ma["ba"],
        "fpr_before": mb["fpr"], "fpr_after": ma["fpr"],
        "fnr_before": mb["fnr"], "fnr_after": ma["fnr"],
        "f1_before": mb["f1"], "f1_after": ma["f1"],
        "benefit": benefit, "harm": harm, "neutral": neutral,
        "tau": float(tau_after),
    }


# --------------------------------------------------------------------------- #
# Deterministic strongest-external-baseline rule (frozen, no post-hoc fishing)
# --------------------------------------------------------------------------- #
def pick_strongest_external(metrics: Dict[str, Dict[str, float]]) -> str:
    """Among external baselines: safety-admissible (Global Harm < AlwaysOn),
    prefer positive Delta_BA; then lowest Global Harm; then highest Delta_BA.
    Deterministic fallbacks documented in PB1F."""

    def harm(method: str) -> float:
        return float(metrics[method].get("global_harm", float("inf")))

    def dba(method: str) -> float:
        return float(metrics[method].get("delta_ba_vs_noadapt", float("-inf")))

    always_on_harm = harm("ALWAYS_ON_QM")
    admiss = [m for m in EXTERNAL_METHODS if harm(m) < always_on_harm]
    pool = admiss if admiss else EXTERNAL_METHODS
    positive = [m for m in pool if dba(m) > 1e-9]
    cand = positive if positive else pool
    return min(cand, key=lambda m: (harm(m) if positive else 0.0, -dba(m) if not positive else -dba(m)))


# --------------------------------------------------------------------------- #
# Deterministic terminal / interpretation adjudication (frozen input -> verdict)
# --------------------------------------------------------------------------- #
def interpret(dba: Dict[str, float], harm: Dict[str, float],
              strongest_ext: str) -> Tuple[str, str, Dict[str, Any]]:
    """Return (interpretation_case, pb2_terminal, rationale)."""
    always_on_harm = harm["ALWAYS_ON_QM"]
    pu_methods = [m for m in ("D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED") if dba[m] > 1e-9]
    positive_util = len(pu_methods) > 0
    harm_lt_ao = [m for m in ("D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED")
                  if harm[m] < always_on_harm]
    any_harm_lt = len(harm_lt_ao) > 0

    d, c = dba["D_CATEGORY_ONLY"], dba["C_CATEGORY_CONDITIONED"]
    hd, hc = harm["D_CATEGORY_ONLY"], harm["C_CATEGORY_CONDITIONED"]
    ed, eh = dba[strongest_ext], harm[strongest_ext]

    if not positive_util or not any_harm_lt:
        return ("N", "PB1DS_PB2_COMPLETE_MAIN_BENCHMARK_NOT_SUPPORTED",
                {"reason": "no positive mean Delta_BA or no harm reduction vs AlwaysOn for D/C",
                 "positive_methods": pu_methods, "harm_reducing_methods": harm_lt_ao})

    if ed > max(d, c) + 1e-9 and eh <= min(hd, hc) + 1e-6:
        return ("E", "PB1DS_PB2_COMPLETE_STRONG_BASELINE_CHALLENGES_NOVELTY",
                {"reason": "strongest external baseline surpasses both D and C in utility without higher harm",
                 "strongest_ext": strongest_ext, "ext_dba": ed, "ext_harm": eh})

    pareto = (d > c and hd > hc) and abs(hc - hd) > 1e-6 and (c > 0 or d > 0)
    if pareto and d > 0 and c > 0:
        return ("P", "PB1DS_PB2_COMPLETE_UTILITY_SAFETY_PARETO_SUPPORTED",
                {"reason": "D higher utility, C lower harm -> utility-safety Pareto pair",
                 "d_dba": d, "c_dba": c, "d_harm": hd, "c_harm": hc})

    if hc < hd - 1e-6 and c <= d + 1e-6 and (d is not None):
        # C comparable-or-worse utility with materially lower harm => Case S
        return ("S", "PB1DS_PB2_COMPLETE_MAIN_BENCHMARK_SCOPED_SUPPORT",
                {"reason": "C safety-dominant; comparable utility, lower harm (scoped support retained)",
                 "d_dba": d, "c_dba": c, "d_harm": hd, "c_harm": hc})

    return ("U", "PB1DS_PB2_COMPLETE_MAIN_BENCHMARK_STRONGLY_SUPPORTED",
            {"reason": "D utility-dominant with C safety refinement; positive utility and harm reduction",
             "d_dba": d, "c_dba": c, "d_harm": hd, "c_harm": hc})