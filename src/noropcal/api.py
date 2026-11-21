"""NorOpCal public API.

Normal-Only Selective Operating-Point Calibration (NorOpCal) selectively and
conservatively recalibrates the decision threshold of a frozen industrial
anomaly detector under acquisition shift, using only a handful of known-normal
target observations.

Example:
    >>> import numpy as np
    >>> from noropcal import NorOpCal
    >>> policy = NorOpCal()
    >>> src = np.array([0.10] * 50)
    >>> tgt = np.array([0.30] * 8)
    >>> tau_source = 0.2
    >>> policy.threshold(src, tgt, tau_source)
    0.25
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from .core import (
    load_frozen_spec,
    correctability_score as _core_correctability,
    gate_adapt as _core_gate,
    actuator_raw as _core_actuator,
    bounded_tau as _core_bounded,
)
from .version import __version__


class NorOpCal:
    """Normal-Only Selective Operating-Point Calibration policy.

    Components:
    1. Median-shift correctability signal: Delta_m = median(target) - median(source)
    2. Q50 correctability gate: adapt iff Delta_m >= Q50 (cutoff), else abstain (tau_source)
    3. Conservative bounded actuator: tau_target = tau_source + lambda * (tau_raw - tau_source)
    """

    def __init__(
        self,
        lam: Optional[float] = None,
        cutoff: Optional[float] = None,
        family: Optional[str] = None,
    ) -> None:
        spec = load_frozen_spec()["principal_candidate"]
        self.family: str = str(family) if family is not None else str(spec["family"])
        self.lam: float = float(spec["lambda"]) if lam is None else float(lam)
        self.cutoff: float = float(spec["cutoff_value"]) if cutoff is None else float(cutoff)

        if (lam is not None or cutoff is not None or family is not None) and (
            self.lam != float(spec["lambda"])
            or self.cutoff != float(spec["cutoff_value"])
            or self.family != str(spec["family"])
        ):
            self.frozen_defaults: bool = False
        else:
            self.frozen_defaults = True

    def correctability_score(
        self,
        src_normal_scores: np.ndarray,
        target_normal_scores: np.ndarray,
    ) -> float:
        """Compute median-shift correctability signal Delta_m = median_t - median_s."""
        return _core_correctability(src_normal_scores, target_normal_scores)

    def gate(self, score: float) -> bool:
        """Evaluate correctability gate: adapt iff score >= cutoff (Q50)."""
        return _core_gate(score, self.cutoff)

    def raw_threshold(
        self,
        src_normal_scores: np.ndarray,
        target_normal_scores: np.ndarray,
        tau_source: float,
    ) -> float:
        """Calculate raw threshold for the chosen actuator family."""
        return _core_actuator(self.family, src_normal_scores, target_normal_scores, tau_source)

    def threshold(
        self,
        src_normal_scores: np.ndarray,
        target_normal_scores: np.ndarray,
        tau_source: float,
    ) -> float:
        """Run full policy: gate check -> bounded update, else abstain (return tau_source)."""
        score = self.correctability_score(src_normal_scores, target_normal_scores)
        if not self.gate(score):
            return float(tau_source)
        return _core_bounded(
            self.family,
            src_normal_scores,
            target_normal_scores,
            tau_source,
            self.lam,
        )


__all__ = ["NorOpCal", "__version__", "load_frozen_spec"]
