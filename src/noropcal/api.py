"""NorOpCal public API — a thin compatibility facade.

Every computation delegates to the frozen scientific implementation in
``tools/pb7_p1_safety_actuator/pb7_p1_common.py`` (the canonical PB7-P1 stage
code that produced the paper numbers). This facade adds no mathematics, no
defaults, and no parameter reinterpretation; frozen values (lambda=0.25,
Q50 cutoff) are loaded from the sealed specification artifact.

Example
-------
>>> import numpy as np
>>> from noropcal import NorOpCal
>>> policy = NorOpCal()                      # frozen lambda / Q50 defaults
>>> src = np.array([0.10] * 50)              # source-normal scores
>>> tgt = np.array([0.30] * 8)               # K target-normal scores
>>> tau_source = 0.2
>>> policy.threshold(src, tgt, tau_source)   # gate pass -> bounded update
0.25
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._bridge import load_frozen_spec, scientific_impl
from .version import __version__


class NorOpCal:
    """Normal-Only Selective Operating-Point Calibration (frozen policy).

    Thin delegating wrapper around the PB7-P1 frozen implementation:
    median-shift correctability signal -> Q50 gate -> bounded A_MEDIAN_RAW
    threshold update with conservative shrinkage, abstaining to ``tau_source``
    on gate failure.
    """

    def __init__(
        self,
        lam: Optional[float] = None,
        cutoff: Optional[float] = None,
    ) -> None:
        spec = load_frozen_spec()["principal_candidate"]
        self.family: str = spec["family"]
        self.lam: float = float(spec["lambda"]) if lam is None else float(lam)
        self.cutoff: float = float(spec["cutoff_value"]) if cutoff is None else float(cutoff)
        if (lam is not None or cutoff is not None) and (
            self.lam != float(spec["lambda"]) or self.cutoff != float(spec["cutoff_value"])
        ):
            # Non-frozen overrides are allowed for exploration only; the paper
            # policy is the default construction below.
            self.frozen_defaults: bool = False
        else:
            self.frozen_defaults = True
        self._impl = scientific_impl()

    # -- components (all delegate to the frozen implementation) -------------
    def correctability_score(self, src_normal_scores: np.ndarray,
                             target_normal_scores: np.ndarray) -> float:
        """Median-shift correctability signal Delta_m = median_t - median_s."""
        return self._impl.correctability_score(
            np.asarray(src_normal_scores, dtype=float),
            np.asarray(target_normal_scores, dtype=float),
            tau_source=0.0,  # unused by the frozen median-shift signal
        )

    def gate(self, score: float) -> bool:
        """Frozen correctability gate: adapt iff score >= Q50."""
        return self._impl.gate_adapt(float(score), self.cutoff)

    def raw_threshold(self, src_normal_scores: np.ndarray,
                      target_normal_scores: np.ndarray, tau_source: float) -> float:
        """Raw actuator threshold tau_raw for the frozen family (A_MEDIAN_RAW)."""
        return self._impl.actuator_raw(
            self.family,
            np.asarray(src_normal_scores, dtype=float),
            np.asarray(target_normal_scores, dtype=float),
            float(tau_source),
        )

    def threshold(self, src_normal_scores: np.ndarray,
                  target_normal_scores: np.ndarray, tau_source: float) -> float:
        """Full frozen policy: gate + bounded update, else abstain (tau_s)."""
        score = self.correctability_score(src_normal_scores, target_normal_scores)
        if not self.gate(score):
            return float(tau_source)
        return self._impl.bounded_tau(
            self.family,
            np.asarray(src_normal_scores, dtype=float),
            np.asarray(target_normal_scores, dtype=float),
            float(tau_source),
            self.lam,
        )


__all__ = ["NorOpCal", "__version__", "load_frozen_spec"]
