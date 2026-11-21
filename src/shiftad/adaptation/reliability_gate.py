"""Reliability gate interface and deterministic fixture implementation.

The gate admits a candidate into quarantine only when every enabled condition
passes (protocol Section 5.2). Rejection codes are machine-readable and the
gate never receives labels, masks, defect types, or evaluator outcomes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from shiftad.adaptation.base import (
    FIXTURE_SCOPE,
    REASON_CONSISTENCY,
    REASON_IDENTITY,
    REASON_INPUT_QUALITY,
    REASON_RATE_LIMIT,
    REASON_SCORE_LIMIT,
    REASON_SHIFT_MEANINGFUL,
    REASON_SUSPICIOUS_AREA,
    GateDecision,
    SchemaRejectionError,
    ShiftEstimate,
)

ACCEPTED_CODE = "ACCEPTED"


class ReliabilityGate(ABC):
    """Decide whether one sample may update target state."""

    @abstractmethod
    def evaluate(
        self,
        source_score: float,
        shift_estimate: ShiftEstimate | None,
        *,
        suspicious_area_ratio: float = 0.0,
        consistency_ok: bool = True,
        input_quality_ok: bool = True,
        identity_ok: bool = True,
        rate_ok: bool = True,
    ) -> GateDecision:
        """Evaluate all enabled gate conditions.

        Args:
            source_score: Source anomaly score for the sample.
            shift_estimate: Current shift estimate (None when unused).
            suspicious_area_ratio: Predicted abnormal-area fraction.
            consistency_ok: Weak-view consistency verdict.
            input_quality_ok: Input-quality verdict.
            identity_ok: Category/model identity verdict.
            rate_ok: Update-rate/cooldown/capacity verdict.

        Returns:
            A deterministic gate decision with a machine-readable reason.
        """


class V1ReliabilityGate(ReliabilityGate):
    """Protocol-consistent reliability gate (candidate; frozen at Gate D).

    Applies all seven gate conditions from protocol Section 5.2 with
    machine-readable rejection codes. Rate/cooldown/quarantine admission is
    handled by a deterministic internal state machine; the gate is
    label-blind and never receives labels, masks, defect types, or evaluator
    outcomes.

    Args:
        score_limit: Source anomaly score limit (calibrated on D2/D3).
        consistency_tolerance: Weak-view consistency tolerance.
        suspicious_area_ratio_max: Maximum suspicious-area ratio.
        update_rate_max: Maximum accepted updates per rolling window.
        rate_window_steps: Rolling window length for the update rate.
        cooldown_steps: Minimum steps between accepted updates.
    """

    def __init__(
        self,
        *,
        score_limit: float = 0.5,
        consistency_tolerance: float = 0.1,
        suspicious_area_ratio_max: float = 0.05,
        update_rate_max: int = 4,
        rate_window_steps: int = 16,
        cooldown_steps: int = 4,
    ) -> None:
        if not score_limit > 0:
            raise SchemaRejectionError("score_limit must be positive.")
        if not 0 <= consistency_tolerance <= 1:
            raise SchemaRejectionError("consistency_tolerance must lie in [0, 1].")
        if not 0 <= suspicious_area_ratio_max <= 1:
            raise SchemaRejectionError("suspicious_area_ratio_max must lie in [0, 1].")
        if update_rate_max < 1:
            raise SchemaRejectionError("update_rate_max must be >= 1.")
        if rate_window_steps < 1:
            raise SchemaRejectionError("rate_window_steps must be >= 1.")
        if cooldown_steps < 0:
            raise SchemaRejectionError("cooldown_steps must be >= 0.")
        self._score_limit = float(score_limit)
        self._consistency_tolerance = float(consistency_tolerance)
        self._suspicious_area_ratio_max = float(suspicious_area_ratio_max)
        self._update_rate_max = int(update_rate_max)
        self._rate_window_steps = int(rate_window_steps)
        self._cooldown_steps = int(cooldown_steps)
        # Deterministic internal state (no RNG, no wall clock).
        self._step = 0
        self._accept_steps: list[int] = []
        self._last_accept_step = -1

    # ------------------------------------------------------------------
    # Deterministic state helpers (label-blind)
    # ------------------------------------------------------------------
    def _rate_ok(self) -> bool:
        """True when the accepted-update rate stays within ``update_rate_max``
        over the last ``rate_window_steps`` evaluations."""
        if len(self._accept_steps) < self._update_rate_max:
            return True
        window_start = self._step - self._rate_window_steps
        recent = [s for s in self._accept_steps if s > window_start]
        return len(recent) < self._update_rate_max

    def _cooldown_ok(self) -> bool:
        """True when at least ``cooldown_steps`` evaluations passed since the
        last accepted update."""
        if self._last_accept_step < 0:
            return True
        return self._step - self._last_accept_step >= self._cooldown_steps

    def _record_accept(self) -> None:
        self._accept_steps.append(self._step)
        self._last_accept_step = self._step
        # Prune steps that can no longer affect the rolling window so the
        # bookkeeping stays bounded over long streams.
        prune_before = self._step - self._rate_window_steps
        if prune_before > 0 and self._accept_steps and self._accept_steps[0] <= prune_before:
            self._accept_steps = [s for s in self._accept_steps if s > prune_before]

    # ------------------------------------------------------------------
    # Public evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        source_score: float,
        shift_estimate: ShiftEstimate | None,
        *,
        suspicious_area_ratio: float = 0.0,
        consistency_ok: bool = True,
        input_quality_ok: bool = True,
        identity_ok: bool = True,
        rate_ok: bool | None = None,
    ) -> GateDecision:
        """Evaluate all seven gate conditions deterministically.

        Args:
            source_score: Source anomaly score for the sample.
            shift_estimate: Current shift estimate (None when unused).
            suspicious_area_ratio: Predicted abnormal-area fraction.
            consistency_ok: Weak-view consistency verdict (True when the
                consistency distance is within tolerance).
            input_quality_ok: Input-quality verdict.
            identity_ok: Category/model identity verdict.
            rate_ok: Explicit rate override (None uses the internal state
                machine).

        Returns:
            A deterministic gate decision with a machine-readable reason.
        """
        rate_ok = self._rate_ok() if rate_ok is None else bool(rate_ok)
        cooldown_ok = self._cooldown_ok()
        shift_ok = shift_estimate is None or shift_estimate.in_envelope
        self._step += 1  # every evaluation advances the deterministic clock
        checks: dict[str, bool] = {
            "score_limit": source_score <= self._score_limit,
            "consistency": consistency_ok,
            "suspicious_area": suspicious_area_ratio <= self._suspicious_area_ratio_max,
            "input_quality": input_quality_ok,
            "shift_meaningful": shift_ok,
            "identity": identity_ok,
            "rate_limit": rate_ok,
            "cooldown": cooldown_ok,
        }
        if not checks["score_limit"]:
            return GateDecision(False, REASON_SCORE_LIMIT, checks)
        if not checks["consistency"]:
            return GateDecision(False, REASON_CONSISTENCY, checks)
        if not checks["suspicious_area"]:
            return GateDecision(False, REASON_SUSPICIOUS_AREA, checks)
        if not checks["input_quality"]:
            return GateDecision(False, REASON_INPUT_QUALITY, checks)
        if not checks["shift_meaningful"]:
            return GateDecision(False, REASON_SHIFT_MEANINGFUL, checks)
        if not checks["identity"]:
            return GateDecision(False, REASON_IDENTITY, checks)
        if not checks["rate_limit"]:
            return GateDecision(False, REASON_RATE_LIMIT, checks)
        if not checks["cooldown"]:
            return GateDecision(False, REASON_RATE_LIMIT, checks)
        self._record_accept()
        return GateDecision(True, ACCEPTED_CODE, checks)


class FixtureReliabilityGate(ReliabilityGate):
    """Deterministic fixture gate (FIXTURE_SCOPE only).

    Applies a single frozen source-score limit; all other conditions are
    forwarded as given. Used for software safety tests only.
    """

    def __init__(self, *, score_limit: float = 0.5) -> None:
        self._score_limit = score_limit
        self.scope = FIXTURE_SCOPE

    def evaluate(
        self,
        source_score: float,
        shift_estimate: ShiftEstimate | None,
        *,
        suspicious_area_ratio: float = 0.0,
        consistency_ok: bool = True,
        input_quality_ok: bool = True,
        identity_ok: bool = True,
        rate_ok: bool = True,
    ) -> GateDecision:
        checks: dict[str, bool] = {
            "score_limit": source_score <= self._score_limit,
            "consistency": consistency_ok,
            "suspicious_area": suspicious_area_ratio <= 0.05,
            "input_quality": input_quality_ok,
            "identity": identity_ok,
            "rate_limit": rate_ok,
        }
        if not checks["score_limit"]:
            return GateDecision(False, REASON_SCORE_LIMIT, checks)
        if not checks["consistency"]:
            return GateDecision(False, REASON_CONSISTENCY, checks)
        if not checks["suspicious_area"]:
            return GateDecision(False, REASON_SUSPICIOUS_AREA, checks)
        if not checks["input_quality"]:
            return GateDecision(False, REASON_INPUT_QUALITY, checks)
        if not checks["identity"]:
            return GateDecision(False, REASON_IDENTITY, checks)
        if not checks["rate_limit"]:
            return GateDecision(False, REASON_RATE_LIMIT, checks)
        if shift_estimate is not None and not shift_estimate.in_envelope:
            return GateDecision(False, REASON_SHIFT_MEANINGFUL, checks)
        return GateDecision(True, ACCEPTED_CODE, checks)
