"""Immutable contracts for the SAPCA engineering foundation.

The contracts intentionally contain no target truth and no production acting
parameters. Canonical hashes bind proposals, certificates, provenance, and audit
events without relying on timestamps or process-local object identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

PCE_SCHEMA_VERSION = "pce_v1"
SAPCA_BLUEPRINT_VERSION = "phase6_v2_blueprint_v1.0"
MAX_SOURCE_ONLY_CAP = 0.239674
FORBIDDEN_TRUTH_KEYS = ("label", "mask", "defect", "truth", "outcome", "future")


class SapcaContractError(ValueError):
    """Report an invalid SAPCA contract or provenance binding."""


def reject_truth_like_keys(payload: Any) -> None:
    """Recursively reject mappings whose keys can carry target truth."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_TRUTH_KEYS):
                raise SapcaContractError(f"Forbidden truth-like key: {key!r}.")
            reject_truth_like_keys(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            reject_truth_like_keys(value)


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_value(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return {str(k): _canonical_value(v) for k, v in sorted(value.items())}
    if isinstance(value, tuple):
        return [_canonical_value(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    """Return deterministic compact JSON for a contract value."""
    return json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 binding of a contract value."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_vector(values: tuple[float, ...], size: int, name: str) -> None:
    if len(values) != size:
        raise SapcaContractError(f"{name} must contain exactly {size} values.")
    if not all(math.isfinite(float(value)) for value in values):
        raise SapcaContractError(f"{name} must contain only finite float64 values.")


def _validate_identity(category: str, sequence_id: int) -> None:
    if not category.strip():
        raise SapcaContractError("category must be non-empty.")
    if sequence_id < 1:
        raise SapcaContractError("sequence_id must be a positive monotonic identifier.")


@dataclass(frozen=True)
class ProposalContextView:
    """Proposal-only view exposing exactly the twelve frozen context values."""

    category: str
    sequence_id: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_identity(self.category, self.sequence_id)
        _validate_vector(self.values, 12, "proposal_context")


@dataclass(frozen=True)
class CertificateInputView:
    """Certificate-only view exposing exactly six action-response values."""

    category: str
    sequence_id: int
    values: tuple[float, ...]
    source_anchor_hashes: tuple[tuple[str, str], ...]
    state_hash: str

    def __post_init__(self) -> None:
        _validate_identity(self.category, self.sequence_id)
        _validate_vector(self.values, 6, "action_response_certificate")
        if not self.source_anchor_hashes or not self.state_hash:
            raise SapcaContractError("Certificate view requires anchor and state hashes.")


@dataclass(frozen=True)
class ProposalCertificateEvidence:
    """Fixed pce_v1 evidence with dependency-separated typed views."""

    proposal_view: ProposalContextView
    certificate_view: CertificateInputView
    history_count: int
    validity_flags: tuple[str, ...]
    schema_version: str = PCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PCE_SCHEMA_VERSION:
            raise SapcaContractError("Unsupported PCE schema version.")
        if self.proposal_view.category != self.certificate_view.category:
            raise SapcaContractError("PCE views must bind the same category.")
        if self.proposal_view.sequence_id != self.certificate_view.sequence_id:
            raise SapcaContractError("PCE views must bind the same sequence.")
        if self.history_count < 0:
            raise SapcaContractError("history_count must be non-negative.")


@dataclass(frozen=True)
class SourceOnlyParameterProvenance:
    """Source-only derivation and authorization metadata."""

    category: str
    allowed_source_split: str
    split_fingerprint: str
    source_identities_hash: str
    parameter_names: tuple[str, ...]
    creation_tool: str
    creation_version: str
    authorized: bool
    fixture_only: bool
    formal_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.category or not self.allowed_source_split:
            raise SapcaContractError("Parameter provenance requires category and source split.")
        if not self.split_fingerprint or not self.source_identities_hash:
            raise SapcaContractError("Parameter provenance requires source identity bindings.")
        if self.fixture_only and self.formal_evidence:
            raise SapcaContractError("Fixture provenance cannot be formal evidence.")


@dataclass(frozen=True)
class SourceOnlyParameterPackage:
    """Hash-bound source-only parameters; absent from production in this cycle."""

    provenance: SourceOnlyParameterProvenance
    image_stat_mean: tuple[float, ...]
    image_stat_std: tuple[float, ...]
    source_score_mean: float
    source_score_std: float
    history_median_reference: float
    history_mad_reference: float
    source_threshold: float
    epsilon_s: float
    correction_cap: float
    package_hash: str = ""

    def __post_init__(self) -> None:
        _validate_vector(self.image_stat_mean, 9, "image_stat_mean")
        _validate_vector(self.image_stat_std, 9, "image_stat_std")
        scalars = (
            self.source_score_mean,
            self.source_score_std,
            self.history_median_reference,
            self.history_mad_reference,
            self.source_threshold,
            self.epsilon_s,
            self.correction_cap,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise SapcaContractError("Source-only parameters must be finite.")
        if any(value <= 0 for value in self.image_stat_std):
            raise SapcaContractError("Image-stat standard deviations must be positive.")
        if self.source_score_std <= 0 or self.history_mad_reference <= 0:
            raise SapcaContractError("Score and history scale references must be positive.")
        if self.epsilon_s < 0 or not 0 <= self.correction_cap <= MAX_SOURCE_ONLY_CAP:
            raise SapcaContractError("Invalid epsilon or correction cap.")
        expected = self.content_hash()
        if self.package_hash and self.package_hash != expected:
            raise SapcaContractError("Parameter package hash mismatch.")
        object.__setattr__(self, "package_hash", expected)

    def content_hash(self) -> str:
        """Hash all package fields except the self-referential package hash."""
        payload = asdict(self)
        payload.pop("package_hash", None)
        return canonical_hash(payload)

    @property
    def production_usable(self) -> bool:
        """Return whether a future production-authorized package may act."""
        return self.provenance.authorized and not self.provenance.fixture_only


@dataclass(frozen=True)
class FrozenSourceObservation:
    """Read-only source inference outputs consumed by SAPCA."""

    category: str
    sequence_id: int
    source_score: float
    source_threshold: float
    source_map: Any
    weak_view_scores: tuple[float, ...]
    source_anchor_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _validate_identity(self.category, self.sequence_id)
        if not math.isfinite(self.source_score) or not math.isfinite(self.source_threshold):
            raise SapcaContractError("Source score and threshold must be finite.")
        if not self.weak_view_scores or not all(math.isfinite(v) for v in self.weak_view_scores):
            raise SapcaContractError("Weak-view scores must be non-empty and finite.")
        if not self.source_anchor_hashes:
            raise SapcaContractError("Source-anchor hashes are required.")


@dataclass(frozen=True)
class CorrectionProposal:
    """Proposal-only request for a bounded downward image-score delta."""

    proposal_id: str
    category: str
    sequence_id: int
    requested_delta: float
    expires_after_sequence: int
    proposer_version: str
    policy_provenance_hash: str

    def __post_init__(self) -> None:
        _validate_identity(self.category, self.sequence_id)
        if not self.proposal_id or not self.proposer_version or not self.policy_provenance_hash:
            raise SapcaContractError("Proposal binding fields must be non-empty.")
        if not math.isfinite(self.requested_delta) or self.requested_delta < 0:
            raise SapcaContractError("requested_delta must be non-negative and finite.")
        if self.expires_after_sequence < self.sequence_id:
            raise SapcaContractError("Proposal expiry precedes issue sequence.")

    def binding_hash(self) -> str:
        """Return the immutable full-payload binding hash."""
        return canonical_hash(self)


@dataclass(frozen=True)
class SafetyCertificate:
    """Independent authorization token bound to one exact proposal."""

    certificate_id: str
    schema_version: str
    proposal_hash: str
    proposal_id: str
    category: str
    sequence_id: int
    approved_delta: float
    source_anchor_hashes: tuple[tuple[str, str], ...]
    parameter_package_hash: str
    policy_provenance_hash: str
    certifier_version: str
    expires_after_sequence: int

    def __post_init__(self) -> None:
        _validate_identity(self.category, self.sequence_id)
        if self.schema_version != PCE_SCHEMA_VERSION:
            raise SapcaContractError("Certificate schema mismatch.")
        if not all(
            (
                self.certificate_id,
                self.proposal_hash,
                self.proposal_id,
                self.parameter_package_hash,
                self.policy_provenance_hash,
                self.certifier_version,
            )
        ):
            raise SapcaContractError("Certificate binding fields must be non-empty.")
        if not math.isfinite(self.approved_delta) or self.approved_delta < 0:
            raise SapcaContractError("approved_delta must be non-negative and finite.")
        if self.expires_after_sequence < self.sequence_id:
            raise SapcaContractError("Certificate expiry precedes issue sequence.")


@dataclass(frozen=True)
class ActuationResult:
    """Exact source fallback or certified bounded image-score result."""

    source_score: float
    method_score: float
    source_map: Any
    acted: bool
    reason: str
    proposal_id: str | None = None
    certificate_id: str | None = None


@dataclass(frozen=True)
class ProtectedStateEvent:
    """Immutable state transition event."""

    category: str
    sequence_id: int
    pre_state_hash: str
    post_state_hash: str
    decision: str
    rollback_count: int


@dataclass(frozen=True)
class V2AuditRecord:
    """Evaluator-independent V2 audit record committed before truth join."""

    category: str
    sequence_id: int
    source_observation_hash: str
    proposal_hash: str | None
    certificate_hash: str | None
    state_event: ProtectedStateEvent
    acted: bool
    reason: str
    previous_hash: str
    fixture_only: bool
    formal_evidence: bool = False

    def record_hash(self) -> str:
        """Return the canonical record hash."""
        return canonical_hash(self)


# ---------------------------------------------------------------------------
# Cycle #29/#30 append-only v2 contracts (additive; legacy classes untouched).
# Schema: phase6_v2_sapca_source_only_parameter_package_v2 (Cycle #29 U8).
# ---------------------------------------------------------------------------

V2_PACKAGE_SCHEMA_VERSION = "phase6_v2_sapca_source_only_parameter_package_v2"
V2_CANONICALIZATION_VERSION = "phase6_v2_canonical_json_v1"
V2_WEAK_VIEW_COUNT = 2
V2_WEAK_VIEW_IDENTITIES = ("view_0_original", "view_1_gaussian_blur")
V2_MINIMUM_HISTORY_COUNT = 2
V2_TRAINING_SEEDS = (0, 1, 2)

#: Frozen eleven-field certificate identity (Cycle #29 U7 / instruction 7.8).
CERTIFICATE_ID_FIELDS_11 = (
    "certificate_schema_version",
    "proposal_id",
    "proposal_hash",
    "category",
    "sequence_id",
    "approved_delta",
    "source_anchor_hashes",
    "parameter_package_hash",
    "proposer_policy_provenance_hash",
    "certifier_policy_provenance_hash",
    "expires_after_sequence",
)

#: Frozen ten proposer/certifier thresholds plus anchor envelope bound (U8).
PROPOSER_THRESHOLD_NAMES = ("tau_shift", "tau_mad", "tau_mad_low", "tau_stats_env", "tau_margin")
CERTIFIER_THRESHOLD_NAMES = (
    "tau_weak_source",
    "tau_weak_delta",
    "tau_causal",
    "rho_low",
    "rho_high",
)
ANCHOR_ENVELOPE_BOUND_NAME = "anchor_envelope_bound"

#: Frozen producer versions (Cycle #29 U1-U5).
PRODUCER_VERSIONS = {
    "U1": "SOURCE_ONLY_CALIBRATION_STREAMS_V1",
    "U2": "WEAK_VIEW_PROPOSED_DELTAS_V1",
    "U3": "SOURCE_ANCHOR_ENVELOPE_V1",
    "U4": "BOUNDED_CAUSAL_STATE_INFLUENCE_V1",
    "U5": "ORDERED_WEAK_VIEWS_V1",
}


@dataclass(frozen=True)
class SourceOnlyParameterProvenanceV2:
    """V2 source-only derivation and authorization provenance (U8)."""

    category: str
    allowed_source_split: str
    split_fingerprint: str
    category_identity_count: int
    category_identity_subset_hash: str
    source_identities_hash: str
    parameter_names: tuple[str, ...]
    creation_tool: str
    creation_version: str
    instruction_sha256: str
    supplement_sha256: str
    git_head: str
    environment: str
    authorized: bool
    execution_enabled: bool
    fixture_only: bool
    preregistration_hash: str | None
    formal_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.category or not self.allowed_source_split:
            raise SapcaContractError("V2 provenance requires category and source split.")
        if not self.split_fingerprint or not self.source_identities_hash:
            raise SapcaContractError("V2 provenance requires source identity bindings.")
        if self.category_identity_count < 1:
            raise SapcaContractError("V2 provenance requires a positive category identity count.")
        if self.fixture_only and self.formal_evidence:
            raise SapcaContractError("Fixture V2 provenance cannot be formal evidence.")
        if self.authorized and (not self.execution_enabled or self.preregistration_hash is None):
            raise SapcaContractError(
                "Authorized V2 packages require execution enablement and a preregistration hash."
            )
        if self.preregistration_hash is not None and not self.authorized:
            raise SapcaContractError("A V2 package with a preregistration hash must be authorized.")


@dataclass(frozen=True)
class SourceOnlyParameterPackageV2:
    """Append-only v2 source-only parameter package (Cycle #29 U8 / instruction 7.9)."""

    provenance: SourceOnlyParameterProvenanceV2
    image_stat_mean: tuple[float, ...]
    image_stat_std: tuple[float, ...]
    source_score_mean: float
    source_score_std: float
    history_median_reference: float
    history_mad_reference: float
    minimum_history_count: int = V2_MINIMUM_HISTORY_COUNT
    source_threshold_by_training_seed: tuple[tuple[int, float], ...] = ()
    epsilon_s: float = 0.0
    correction_cap: float = 0.0
    anchor_envelope_bound: float = 0.0
    tau_shift: float = 0.0
    tau_mad: float = 0.0
    tau_mad_low: float = 0.0
    tau_stats_env: float = 0.0
    tau_margin: float = 0.0
    tau_weak_source: float = 0.0
    tau_weak_delta: float = 0.0
    tau_causal: float = 0.0
    rho_low: float = 0.0
    rho_high: float = 0.0
    weak_view_count: int = V2_WEAK_VIEW_COUNT
    weak_view_identities: tuple[str, ...] = V2_WEAK_VIEW_IDENTITIES
    producer_versions: tuple[tuple[str, str], ...] = ()
    formula_ids: tuple[str, ...] = ()
    canonicalization_version: str = V2_CANONICALIZATION_VERSION
    model_binding: tuple[tuple[str, str], ...] = ()
    config_binding: tuple[tuple[str, str], ...] = ()
    code_binding: tuple[tuple[str, str], ...] = ()
    cycle27_public_identity_manifest_hash: str = ""
    cycle27_canonical_stream_hashes_hash: str = ""
    cycle27_canonical_stream_hash_set: tuple[str, ...] = ()
    calibration_stream_hashes: tuple[str, ...] = ()
    schema_version: str = V2_PACKAGE_SCHEMA_VERSION
    content_hash: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != V2_PACKAGE_SCHEMA_VERSION:
            raise SapcaContractError(f"Unsupported V2 package schema: {self.schema_version!r}.")
        if self.canonicalization_version != V2_CANONICALIZATION_VERSION:
            raise SapcaContractError("Unsupported V2 canonicalization version.")
        _validate_vector(self.image_stat_mean, 9, "v2_image_stat_mean")
        _validate_vector(self.image_stat_std, 9, "v2_image_stat_std")
        if any(value <= 0 for value in self.image_stat_std):
            raise SapcaContractError("V2 image-stat standard deviations must be positive.")
        scalars = (
            self.source_score_mean,
            self.source_score_std,
            self.history_median_reference,
            self.history_mad_reference,
            self.epsilon_s,
            self.correction_cap,
            self.anchor_envelope_bound,
            self.tau_shift,
            self.tau_mad,
            self.tau_mad_low,
            self.tau_stats_env,
            self.tau_margin,
            self.tau_weak_source,
            self.tau_weak_delta,
            self.tau_causal,
            self.rho_low,
            self.rho_high,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise SapcaContractError("V2 package scalars must be finite.")
        if self.source_score_std <= 0 or self.history_mad_reference <= 0:
            raise SapcaContractError("V2 score and history scale references must be positive.")
        if self.minimum_history_count != V2_MINIMUM_HISTORY_COUNT:
            raise SapcaContractError("V2 minimum history count must be the frozen literal 2.")
        if self.weak_view_count != V2_WEAK_VIEW_COUNT:
            raise SapcaContractError("V2 weak view count must be the frozen literal 2.")
        if tuple(self.weak_view_identities) != V2_WEAK_VIEW_IDENTITIES:
            raise SapcaContractError("V2 weak view identities must be the frozen ordered pair.")
        seeds = tuple(seed for seed, _ in self.source_threshold_by_training_seed)
        if tuple(sorted(seeds)) != V2_TRAINING_SEEDS or len(seeds) != 3:
            raise SapcaContractError("V2 source-threshold seed keys must be exactly (0, 1, 2).")
        if not all(
            math.isfinite(value) and value > 0
            for _, value in self.source_threshold_by_training_seed
        ):
            raise SapcaContractError("V2 source thresholds must be finite and positive.")
        if self.epsilon_s < 0:
            raise SapcaContractError("V2 epsilon_s must be non-negative.")
        if not 0 < self.correction_cap <= MAX_SOURCE_ONLY_CAP:
            raise SapcaContractError("V2 correction cap must lie in (0, MAX_SOURCE_ONLY_CAP].")
        if self.anchor_envelope_bound < 0:
            raise SapcaContractError("V2 anchor envelope bound must be non-negative.")
        if len(self.calibration_stream_hashes) != 33:
            raise SapcaContractError(
                "V2 calibration stream hashes must contain exactly 33 category entries."
            )
        authorization_flags = (
            self.provenance.preregistration_hash is not None
            or self.provenance.authorized
            or self.provenance.execution_enabled
        )
        if authorization_flags and not (
            self.provenance.fixture_only
            and not self.provenance.authorized
            and not self.provenance.execution_enabled
        ):
            raise SapcaContractError(
                "V2 formal packages must stay unauthorized/disabled/null-preregistration; "
                "only fixture packages may differ."
            )
        expected_provenance = canonical_hash(self.provenance)
        if self.provenance_hash and self.provenance_hash != expected_provenance:
            raise SapcaContractError("V2 provenance hash mismatch.")
        expected_content = self.compute_content_hash()
        if self.content_hash and self.content_hash != expected_content:
            raise SapcaContractError("V2 package content hash mismatch.")
        object.__setattr__(self, "provenance_hash", expected_provenance)
        object.__setattr__(self, "content_hash", expected_content)

    def compute_content_hash(self) -> str:
        """Hash all package fields except the two self-referential hashes."""
        payload = asdict(self)
        payload.pop("content_hash", None)
        payload.pop("provenance_hash", None)
        return canonical_hash(payload)

    def source_threshold_for(self, training_seed: int) -> float | None:
        """Return the sole threshold for a training seed or None when missing."""
        matches = [
            value for seed, value in self.source_threshold_by_training_seed if seed == training_seed
        ]
        return matches[0] if len(matches) == 1 else None

    @property
    def production_usable(self) -> bool:
        """Return whether a future production-authorized v2 package may act."""
        return (
            self.provenance.authorized
            and self.provenance.execution_enabled
            and not self.provenance.fixture_only
            and self.provenance.preregistration_hash is not None
        )
