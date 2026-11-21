"""Conservative source/target/consistency score fusion.

Fusion follows the protocol safety rules (Section 5.4): a positive lower
bound on the source weight, zero target weight before maturity, a frozen cap
on the target weight, and byte/metric equivalence with A0 when adaptation is
disabled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from shiftad.adaptation.base import FusedScore
from shiftad.adaptation.state import TargetState

W_S_MIN = 0.5  # Frozen source-weight safety floor (candidate, protocol 5.4)
W_T_MAX = 0.35  # Frozen target-weight cap (candidate, protocol 5.4)
W_C_MAX = 0.25  # Frozen consistency-weight cap (candidate, protocol 5.4)


class ScoreFusion(ABC):
    """Combine source, target, and consistency scores into one fused score."""

    @abstractmethod
    def fuse(
        self,
        source_score: float,
        target_score: float | None,
        consistency_score: float | None,
        state: TargetState,
        *,
        shift_magnitude: float | None = None,
    ) -> FusedScore:
        """Return the fused score.

        Args:
            source_score: Source-memory score.
            target_score: Target-prototype score (None before maturity).
            consistency_score: Consistency score (None when unused).
            state: Current target state (maturity read-only).
            shift_magnitude: Optional estimated shift magnitude used by the
                frozen fusion schedule (None when shift is not estimated).

        Returns:
            The fused score with the applied weights.
        """


class ConservativeScoreFusion(ScoreFusion):
    """V1 conservative fusion with frozen safety bounds and shift schedule.

    Implements the frozen fusion schedule ``g(D_shift, n_accepted,
    reliability)`` (protocol Section 5.4, frozen at Gate E) inside the frozen
    caps: ``w_s >= w_s_min`` (safety floor), ``w_t <= w_t_max`` and
    ``w_t = 0`` before target maturity, ``w_c <= W_C_MAX``. The schedule
    monotonically reduces the consistency weight and the accepted target
    weight as the estimated shift magnitude grows, so a strong distribution
    shift never lets a weak consistency/target signal dilute the source
    safety floor. All parameters stay inside the frozen bounds; no value is
    fitted from shifted data.
    """

    def __init__(
        self,
        *,
        w_s_min: float = W_S_MIN,
        w_t_max: float = W_T_MAX,
        shift_scale: float = 2.0,
    ) -> None:
        if not 0.0 < w_s_min <= 1.0:
            raise ValueError("w_s_min must lie in (0, 1].")
        if not 0.0 <= w_t_max < 1.0:
            raise ValueError("w_t_max must lie in [0, 1).")
        if not shift_scale > 0.0:
            raise ValueError("shift_scale must be positive.")
        self._w_s_min = w_s_min
        self._w_t_max = w_t_max
        self._shift_scale = shift_scale

    def fuse(
        self,
        source_score: float,
        target_score: float | None,
        consistency_score: float | None,
        state: TargetState,
        *,
        shift_magnitude: float | None = None,
    ) -> FusedScore:
        mature = state.is_mature()
        # Frozen schedule g(D_shift, n_accepted, reliability): the consistency
        # and target weights shrink monotonically with the estimated shift
        # magnitude (clipped to zero at large shift), keeping the source
        # safety floor dominant under distribution shift.
        shift_factor = 1.0
        if shift_magnitude is not None and shift_magnitude > 0.0:
            shift_factor = max(0.0, 1.0 - shift_magnitude / self._shift_scale)
        # A target score is usable only when the prototype is mature AND the
        # score is actually provided; otherwise the target weight must be
        # zero so the source score is not silently scaled down.
        w_t = 0.0 if (not mature or target_score is None) else min(self._w_t_max, W_T_MAX)
        # Normalize so w_s + w_t + w_c == 1 and w_s >= w_s_min: the combined
        # target+consistency weight may not exceed 1 - w_s_min.
        remaining = 1.0 - self._w_s_min
        w_t = min(w_t, remaining) * shift_factor
        w_c = 0.0 if consistency_score is None else min(W_C_MAX, remaining - w_t)
        w_c = w_c * shift_factor
        w_s = max(self._w_s_min, 1.0 - w_t - w_c)
        fused = w_s * source_score
        if target_score is not None and w_t > 0.0:
            fused += w_t * target_score
        if consistency_score is not None and w_c > 0.0:
            fused += w_c * consistency_score
        return FusedScore(
            source_score=source_score,
            target_score=target_score,
            consistency_score=consistency_score,
            fused_score=fused,
            weights={"w_s": w_s, "w_t": w_t, "w_c": w_c},
        )


class AdaptationOffFusion(ScoreFusion):
    """Identity fusion used when adaptation is disabled (A0 equivalence).

    The fused score is exactly the source score; target and consistency
    components never contribute.
    """

    def fuse(
        self,
        source_score: float,
        target_score: float | None,
        consistency_score: float | None,
        state: TargetState,
        *,
        shift_magnitude: float | None = None,
    ) -> FusedScore:
        return FusedScore(
            source_score=source_score,
            target_score=None,
            consistency_score=None,
            fused_score=source_score,
            weights={"w_s": 1.0, "w_t": 0.0, "w_c": 0.0},
        )
