"""NorOpCal core scientific implementation.

Normal-Only Selective Operating-Point Calibration (NorOpCal) adapts the decision
threshold of a frozen anomaly detector under acquisition shift using only normal
target samples, without fine-tuning detector backbones or requiring defect annotations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

# Sealed default specification
FROZEN_SPEC: Dict[str, Any] = {
    "principal_candidate": {
        "family": "A_MEDIAN_RAW",
        "lambda": 0.25,
        "cutoff_key": "Q50",
        "cutoff_value": 0.015890181064605713,
        "gate": "adapt iff median_shift >= cutoff_value else NoAdapt",
        "bounding": "tau_used = tau_source + lambda * (tau_raw - tau_source)",
    },
    "k_settings": {
        "principal": "8",
        "auxiliary": ["4", "FULL"],
    },
    "baselines": {
        "B0": "NoAdapt (tau_source)",
        "B4": "Bounded + Abstaining (A_MEDIAN_RAW lambda=0.25 cutoff=Q50) [principal]",
    },
    "frozen": True,
}


def load_frozen_spec() -> Dict[str, Any]:
    """Return the frozen specification dictionary."""
    return FROZEN_SPEC


def correctability_score(
    src_normal_scores: np.ndarray,
    target_normal_scores: np.ndarray,
    tau_source: float = 0.0,
) -> float:
    """Compute the median-shift correctability signal Delta_m = median_t - median_s."""
    src = np.asarray(src_normal_scores, dtype=float)
    tgt = np.asarray(target_normal_scores, dtype=float)
    if src.size == 0 or tgt.size == 0:
        raise ValueError("Score arrays must not be empty.")
    return float(np.median(tgt) - np.median(src))


def gate_adapt(score: float, cutoff: float) -> bool:
    """Evaluate correctability gate: adapt iff score >= cutoff."""
    return bool(score >= cutoff)


def actuator_raw(
    family: str,
    src_normal_scores: np.ndarray,
    target_normal_scores: np.ndarray,
    tau_source: float,
) -> float:
    """Calculate raw threshold actuator update for specified family."""
    src = np.asarray(src_normal_scores, dtype=float)
    tgt = np.asarray(target_normal_scores, dtype=float)
    if src.size == 0 or tgt.size == 0:
        raise ValueError("Score arrays must not be empty.")

    med_s = float(np.median(src))
    med_t = float(np.median(tgt))

    if family == "A_MEDIAN_RAW":
        return float(tau_source + (med_t - med_s))
    elif family == "A_T2_RAW":
        mad_s = float(np.median(np.abs(src - med_s)))
        mad_t = float(np.median(np.abs(tgt - med_t)))
        ratio = (mad_t / mad_s) if mad_s > 1e-12 else 1.0
        return float(med_t + ratio * (tau_source - med_s))
    elif family == "A_TAIL_RAW":
        return float(np.percentile(tgt, 95.0))
    else:
        raise ValueError(f"Unknown actuator family: {family}")


def bounded_tau(
    family: str,
    src_normal_scores: np.ndarray,
    target_normal_scores: np.ndarray,
    tau_source: float,
    lam: float,
) -> float:
    """Apply conservative bounded contraction: tau = tau_source + lam * (tau_raw - tau_source)."""
    raw = actuator_raw(family, src_normal_scores, target_normal_scores, tau_source)
    return float(tau_source + lam * (raw - tau_source))
