"""Read-only adapter for already-computed frozen source outputs."""

from __future__ import annotations

from typing import Any

from shiftad.adaptation.v2.contracts import FrozenSourceObservation


class FrozenSourceAdapter:
    """Validate and package existing source outputs without reinterpreting inference."""

    def observe(
        self,
        *,
        category: str,
        sequence_id: int,
        source_score: float,
        source_threshold: float,
        source_map: Any,
        weak_view_scores: tuple[float, ...],
        source_anchor_hashes: tuple[tuple[str, str], ...],
    ) -> FrozenSourceObservation:
        """Return a validated observation preserving the exact map object."""
        return FrozenSourceObservation(
            category=category,
            sequence_id=sequence_id,
            source_score=float(source_score),
            source_threshold=float(source_threshold),
            source_map=source_map,
            weak_view_scores=tuple(float(v) for v in weak_view_scores),
            source_anchor_hashes=tuple(sorted(source_anchor_hashes)),
        )
