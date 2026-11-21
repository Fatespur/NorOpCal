"""SAPCA evidence producers (Cycle #29 U2/U3/U4/U5).

All producers are deterministic, truth-free, source-only, and fail-closed.
They evaluate numeric evidence only; they never issue a proposal, certificate,
or action, and they never access evaluator truth, labels, masks, or outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from shiftad.adaptation.image_stats import IMAGE_STATISTIC_NAMES
from shiftad.adaptation.v2.contracts import (
    SapcaContractError,
    SourceOnlyParameterPackageV2,
)

#: Frozen producer versions (Cycle #29).
WEAK_VIEW_PROPOSED_DELTAS_VERSION = "WEAK_VIEW_PROPOSED_DELTAS_V1"
SOURCE_ANCHOR_ENVELOPE_VERSION = "SOURCE_ANCHOR_ENVELOPE_V1"
BOUNDED_CAUSAL_STATE_INFLUENCE_VERSION = "BOUNDED_CAUSAL_STATE_INFLUENCE_V1"
ORDERED_WEAK_VIEWS_VERSION = "ORDERED_WEAK_VIEWS_V1"


def raw_requested_delta(
    margin: float,
    package: SourceOnlyParameterPackageV2,
    *,
    delta_scale: float = 0.5,
    delta_min: float = 0.0,
) -> float:
    """Frozen raw requested-delta function RAW_DELTA_V1.

    ``clamp(delta_scale * (tau_margin - margin) * source_score_std, 0, cap)``.
    Nonfinite inputs raise; the caller converts any failure into veto/fallback.
    """
    if not math.isfinite(margin) or not math.isfinite(package.tau_margin):
        raise SapcaContractError("raw delta inputs must be finite.")
    delta = delta_scale * (package.tau_margin - margin) * package.source_score_std
    return min(max(delta, delta_min), package.correction_cap)


def standardized_stats(
    image_stats: Mapping[str, float],
    package: SourceOnlyParameterPackageV2,
) -> tuple[float, ...]:
    """Standardize the nine frozen image statistics (source-normalized)."""
    values = tuple(float(image_stats[name]) for name in IMAGE_STATISTIC_NAMES)
    if not all(math.isfinite(value) for value in values):
        raise SapcaContractError("image statistics must be finite.")
    return tuple(
        (value - mean) / std
        for value, mean, std in zip(
            values, package.image_stat_mean, package.image_stat_std, strict=True
        )
    )


def standardized_margin(
    source_score: float,
    package: SourceOnlyParameterPackageV2,
) -> float:
    """Standardized source-score margin."""
    if not math.isfinite(source_score):
        raise SapcaContractError("source score must be finite.")
    return (source_score - package.source_score_mean) / package.source_score_std


def weak_view_proposed_deltas(
    view_scores: Sequence[float],
    view_image_stats: Sequence[Mapping[str, float]],
    package: SourceOnlyParameterPackageV2,
    *,
    history_median_shift: float,
    history_mad_ratio: float,
) -> tuple[float, ...]:
    """U2 producer: two separately evaluated bounded raw deltas (WEAK_VIEW_PROPOSED_DELTAS_V1).

    For each ordered weak view, the nine view statistics and the view's
    standardized source-score margin vary; the shared causal-history fields and
    all bindings stay identical. Returns exactly two finite raw deltas bounded
    to ``[0, correction_cap]``.
    """
    if len(view_scores) != 2 or len(view_image_stats) != 2:
        raise SapcaContractError("U2 requires exactly two ordered weak views.")
    if not math.isfinite(history_median_shift) or not math.isfinite(history_mad_ratio):
        raise SapcaContractError("U2 shared causal fields must be finite.")
    deltas: list[float] = []
    for score, stats in zip(view_scores, view_image_stats, strict=True):
        stats_std = standardized_stats(stats, package)
        margin = standardized_margin(score, package)
        # The frozen 12-field proposer context is rebuilt per view; only the
        # nine view statistics and the view margin vary (Cycle #29 U2).
        _context = (*stats_std, margin, history_median_shift, history_mad_ratio)
        if len(_context) != 12:
            raise SapcaContractError("U2 view context must be exactly 12 fields.")
        delta = raw_requested_delta(margin, package)
        if not math.isfinite(delta) or delta < 0 or delta > package.correction_cap:
            raise SapcaContractError("U2 raw delta out of bounds.")
        deltas.append(delta)
    return tuple(deltas)


def weak_view_proposed_disagreement(
    view_scores: Sequence[float],
    view_image_stats: Sequence[Mapping[str, float]],
    package: SourceOnlyParameterPackageV2,
    *,
    history_median_shift: float,
    history_mad_ratio: float,
) -> float:
    """U2 output: finite max - min of the two raw deltas."""
    deltas = weak_view_proposed_deltas(
        view_scores,
        view_image_stats,
        package,
        history_median_shift=history_median_shift,
        history_mad_ratio=history_mad_ratio,
    )
    return max(deltas) - min(deltas)


def anchor_distance(
    image_stats: Mapping[str, float],
    source_score: float,
    package: SourceOnlyParameterPackageV2,
) -> float:
    """U3 producer: ten-component standardized max-abs anchor distance."""
    stats_std = standardized_stats(image_stats, package)
    margin = standardized_margin(source_score, package)
    components = (*stats_std, margin)
    if len(components) != 10:
        raise SapcaContractError("U3 anchor vector must have exactly 10 components.")
    if not all(math.isfinite(value) for value in components):
        raise SapcaContractError("U3 anchor components must be finite.")
    return max(abs(value) for value in components)


def anchor_envelope_violation(
    image_stats: Mapping[str, float],
    source_score: float,
    package: SourceOnlyParameterPackageV2,
) -> float:
    """U3 producer output: ``max(0.0, anchor_distance - anchor_envelope_bound)``."""
    distance = anchor_distance(image_stats, source_score, package)
    if not math.isfinite(package.anchor_envelope_bound) or package.anchor_envelope_bound < 0:
        raise SapcaContractError("U3 anchor envelope bound must be finite and non-negative.")
    return max(0.0, distance - package.anchor_envelope_bound)


def causal_state_influence(
    raw_delta_observed: float,
    raw_delta_intervened: float,
    package: SourceOnlyParameterPackageV2,
) -> float:
    """U4 producer output: ``clip(abs(d_obs - d_int) / cap, 0.0, 1.0)``.

    The caller computes the two raw deltas from the observed 12-field context
    and the source-reference intervention ``(history_median_shift=0.0,
    history_mad_ratio=1.0)`` (U4). Nonfinite deltas or a nonpositive cap veto.
    """
    if not math.isfinite(raw_delta_observed) or not math.isfinite(raw_delta_intervened):
        raise SapcaContractError("U4 raw deltas must be finite.")
    if not math.isfinite(package.correction_cap) or package.correction_cap <= 0:
        raise SapcaContractError("U4 correction cap must be finite and strictly positive.")
    return min(
        max(abs(raw_delta_observed - raw_delta_intervened) / package.correction_cap, 0.0), 1.0
    )


def weak_view_source_disagreement(view_scores: Sequence[float]) -> float:
    """U5 output: finite max - min over the two ordered weak-view source scores.

    Missing or nonfinite view input/score vetoes (caller converts the raised
    error into exact-source fallback).
    """
    if len(view_scores) != 2:
        raise SapcaContractError("U5 requires exactly two ordered weak views.")
    if not all(math.isfinite(float(value)) for value in view_scores):
        raise SapcaContractError("U5 weak-view scores must be finite.")
    return max(float(view_scores[0]), float(view_scores[1])) - min(
        float(view_scores[0]), float(view_scores[1])
    )
