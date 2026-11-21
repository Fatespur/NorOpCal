"""Target memory interface and deterministic fixture implementation.

The target memory owns the quarantine buffer and the target prototype; state
mutations live in ``TargetState`` (see ``state.py``). This module defines the
interface and a fixture policy used only for software safety tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shiftad.adaptation.base import (
    FIXTURE_SCOPE,
    PRODUCTION_SCOPE,
    GateDecision,
    SchemaRejectionError,
)
from shiftad.adaptation.state import TargetState


class TargetMemory(ABC):
    """Own the target-domain normal prototypes."""

    @abstractmethod
    def maybe_update(self, decision: GateDecision, features: Any, state: TargetState) -> bool:
        """Apply one accepted decision to the target state (if enabled).

        Args:
            decision: Gate decision for the current sample.
            features: Feature row of the accepted candidate.
            state: The mutable target state.

        Returns:
            True when the state changed.
        """

    @abstractmethod
    def score(self, features: Any, state: TargetState) -> float | None:
        """Return the target-prototype distance (None before maturity)."""


class FixtureTargetMemory(TargetMemory):
    """Deterministic fixture target memory (FIXTURE_SCOPE only).

    The fixture memory mirrors the engine's quarantine semantics: accepted
    decisions append the feature row and attempt a promotion. Used only for
    software safety tests; never formal evidence.
    """

    def __init__(self) -> None:
        self.scope = FIXTURE_SCOPE

    def maybe_update(self, decision: GateDecision, features: Any, state: TargetState) -> bool:
        if not decision.accepted:
            return False
        version_before = state.version
        row = tuple(float(v) for v in features)
        state.add_candidate(row)
        state.promote()
        return state.version != version_before

    def score(self, features: Any, state: TargetState) -> float | None:
        if not state.is_mature():
            return None
        row = tuple(float(v) for v in features)
        return float(sum((a - b) ** 2 for a, b in zip(row, state.prototype, strict=False)))


class V1TargetMemory(TargetMemory):
    """Production target memory (PRODUCTION_SCOPE; protocol Section 5.3).

    Scoring-only in the formal streaming workflow: state mutation (quarantine
    append + promotion) is owned exclusively by ``CausalStreamEngine.process``
    so each accepted sample contributes exactly once to the prototype. This
    class validates and scores candidate feature rows against the current
    prototype; its ``maybe_update`` is kept for interface completeness and
    unit tests but has no production caller.

    Feature rows must be finite (NaN/inf rejected fail-closed) and
    dimension-consistent.
    """

    def __init__(self) -> None:
        self.scope = PRODUCTION_SCOPE

    @staticmethod
    def _row(features: Any) -> tuple[float, ...]:
        try:
            row = tuple(float(v) for v in features)
        except (TypeError, ValueError) as exc:
            raise SchemaRejectionError(f"Target candidate must be numeric: {exc}") from exc
        if not row or any(not (v == v) or abs(v) == float("inf") for v in row):
            raise SchemaRejectionError("Target candidates must be finite, non-empty rows.")
        return row

    def maybe_update(self, decision: GateDecision, features: Any, state: TargetState) -> bool:
        if not decision.accepted:
            return False
        version_before = state.version
        state.add_candidate(self._row(features))
        state.promote()
        return state.version != version_before

    def score(self, features: Any, state: TargetState) -> float | None:
        if not state.is_mature():
            return None
        row = self._row(features)
        return float(sum((a - b) ** 2 for a, b in zip(row, state.prototype, strict=False)))
