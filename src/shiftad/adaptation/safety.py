"""Safety monitor and fail-closed rollback decisions.

The safety monitor watches target/source distance, acceptance rates,
score-variance collapse, and rollback counts (protocol Section 5.5).
Violations trigger fail-closed disablement or exact state rollback to the
last validated snapshot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from shiftad.adaptation.base import (
    FIXTURE_SCOPE,
    PRODUCTION_SCOPE,
    AdaptationError,
    SchemaRejectionError,
)
from shiftad.adaptation.state import TargetState, TargetStateSnapshot

#: Machine-readable reason for an exhausted rollback budget (shared by the
#: safety monitor and the stream-engine orchestration).
REASON_ROLLBACK_COUNT = "ROLLBACK_COUNT"


@dataclass(frozen=True)
class SafetyVerdict:
    """Machine-readable safety-monitor outcome.

    Attributes:
        ok: True when the system may continue adapting.
        fail_closed: True when adaptation must be disabled immediately.
        rollback_to: Snapshot to restore (None keeps current state).
        reason: Machine-readable reason code.
    """

    ok: bool
    fail_closed: bool = False
    rollback_to: TargetStateSnapshot | None = None
    reason: str = "OK"

    def to_dict(self) -> dict[str, str | bool]:
        """Serialize the verdict deterministically."""
        return {
            "ok": self.ok,
            "fail_closed": self.fail_closed,
            "rollback": self.rollback_to is not None,
            "reason": self.reason,
        }


class SafetyMonitor(ABC):
    """Monitor adaptation health and issue fail-closed verdicts."""

    @abstractmethod
    def check(
        self,
        state: TargetState,
        *,
        target_source_distance: float | None = None,
        acceptance_rate: float | None = None,
        score_variance_ratio: float | None = None,
        rollback_count: int = 0,
    ) -> SafetyVerdict:
        """Evaluate the safety conditions.

        Args:
            state: Current target state.
            target_source_distance: Distance between target and source
                statistics (None when unavailable).
            acceptance_rate: Observed acceptance rate (None when unavailable).
            score_variance_ratio: Score-variance collapse ratio (None when
                unavailable).
            rollback_count: Number of rollbacks so far.

        Returns:
            A deterministic safety verdict.
        """


class FixtureSafetyMonitor(SafetyMonitor):
    """Deterministic fixture safety monitor (FIXTURE_SCOPE only).

    Applies fixed thresholds; used for software safety tests only.
    """

    def __init__(
        self,
        *,
        max_target_source_distance: float = 2.0,
        max_acceptance_rate: float = 0.9,
        min_score_variance_ratio: float = 0.1,
        max_rollback_count: int = 3,
    ) -> None:
        self._max_distance = max_target_source_distance
        self._max_acceptance = max_acceptance_rate
        self._min_variance = min_score_variance_ratio
        self._max_rollbacks = max_rollback_count
        self.scope = FIXTURE_SCOPE

    def check(
        self,
        state: TargetState,
        *,
        target_source_distance: float | None = None,
        acceptance_rate: float | None = None,
        score_variance_ratio: float | None = None,
        rollback_count: int = 0,
    ) -> SafetyVerdict:
        # Rollback-budget exhaustion is checked first: repeated rollbacks
        # indicate persistent contamination, so the budget is the most
        # urgent fail-closed signal (and must not be shadowed by the
        # distance branch while contamination persists).
        if rollback_count > self._max_rollbacks:
            return SafetyVerdict(False, fail_closed=True, reason=REASON_ROLLBACK_COUNT)
        if target_source_distance is not None and target_source_distance > self._max_distance:
            return SafetyVerdict(False, fail_closed=True, reason="TARGET_SOURCE_DISTANCE")
        if acceptance_rate is not None and acceptance_rate > self._max_acceptance:
            return SafetyVerdict(False, fail_closed=True, reason="ACCEPTANCE_RATE")
        if score_variance_ratio is not None and score_variance_ratio < self._min_variance:
            return SafetyVerdict(False, fail_closed=True, reason="SCORE_VARIANCE_COLLAPSE")
        return SafetyVerdict(True, reason="OK")


def require_valid_snapshot(snapshot: TargetStateSnapshot | None) -> None:
    """Reject None snapshots for rollback decisions.

    Raises:
        AdaptationError: If the snapshot is None.
    """
    if snapshot is None:
        raise AdaptationError("Rollback requires a validated snapshot.")


class V1SafetyMonitor(SafetyMonitor):
    """Production safety monitor (PRODUCTION_SCOPE; protocol Section 5.5).

    Applies the frozen Gate F thresholds (``configs/phase6/gate_f_policy.yaml``
    primary monitor block) with full input validation. Violations trigger
    fail-closed disablement; rollback-budget exhaustion is checked first so
    persistent contamination cannot be shadowed by the distance branch.

    Args:
        max_target_source_distance: Frozen target/source distance limit
            (Gate F: 2.0).
        max_acceptance_rate: Frozen acceptance-rate limit (Gate F: 1.0).
        min_score_variance_ratio: Frozen score-variance collapse ratio
            (Gate F: 0.1).
        max_rollback_count: Frozen rollback budget (Gate F: 3).

    Raises:
        SchemaRejectionError: If any threshold is invalid.
    """

    def __init__(
        self,
        *,
        max_target_source_distance: float = 2.0,
        max_acceptance_rate: float = 1.0,
        min_score_variance_ratio: float = 0.1,
        max_rollback_count: int = 3,
    ) -> None:
        if not max_target_source_distance > 0:
            raise SchemaRejectionError("max_target_source_distance must be positive.")
        if not 0 <= max_acceptance_rate <= 2:
            raise SchemaRejectionError("max_acceptance_rate must lie in [0, 2].")
        if not 0 < min_score_variance_ratio <= 1:
            raise SchemaRejectionError("min_score_variance_ratio must lie in (0, 1].")
        if max_rollback_count < 0:
            raise SchemaRejectionError("max_rollback_count must be non-negative.")
        self._max_distance = max_target_source_distance
        self._max_acceptance = max_acceptance_rate
        self._min_variance = min_score_variance_ratio
        self._max_rollbacks = max_rollback_count
        self.scope = PRODUCTION_SCOPE

    def check(
        self,
        state: TargetState,
        *,
        target_source_distance: float | None = None,
        acceptance_rate: float | None = None,
        score_variance_ratio: float | None = None,
        rollback_count: int = 0,
    ) -> SafetyVerdict:
        if rollback_count > self._max_rollbacks:
            return SafetyVerdict(False, fail_closed=True, reason=REASON_ROLLBACK_COUNT)
        if target_source_distance is not None and target_source_distance > self._max_distance:
            return SafetyVerdict(False, fail_closed=True, reason="TARGET_SOURCE_DISTANCE")
        if acceptance_rate is not None and acceptance_rate > self._max_acceptance:
            return SafetyVerdict(False, fail_closed=True, reason="ACCEPTANCE_RATE")
        if score_variance_ratio is not None and score_variance_ratio < self._min_variance:
            return SafetyVerdict(False, fail_closed=True, reason="SCORE_VARIANCE_COLLAPSE")
        return SafetyVerdict(True, reason="OK")
