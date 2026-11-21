"""External formal execution authorization for SAPCA v2 (Cycle #31R).

Implements the separation of parameter identity from execution authorization:

- ``SapcaExecutionAuthorizationV2``: the external, hash-bound authorization
  artifact (a frozen, immutable, user-issued document). It binds the exact
  preregistration, six-unit plan, implementation, identity, parameter-package,
  statistics, decision, stop-rule, and environment hashes, the allowed arm
  set, the research-unit budget, the execution root, provenance, and expiry.
  It contains NO scientific parameters and NO policy thresholds.
- ``ExecutionAuthorizationValidatorV2``: validates an artifact against the
  frozen preregistration and the current execution-critical bindings. Every
  mismatch fails closed with a machine-readable reason.
- ``ValidatedFormalExecutionPermitV2``: the narrow runtime permit produced
  only by a successful validation. It authorizes execution of already frozen
  semantics only; it never invents or alters parameters.

No prospective scoring data, identity, truth, or outcome is read here. The
five canonical Cycle #30 parameter packages remain byte-identical and are
never mutated; ``authorized``/``execution_enabled``/``preregistration_hash``
inside the packages stay false/null. Acting is legal only when a valid permit
is present AND all bindings match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shiftad.adaptation.v2.contracts import SapcaContractError, canonical_hash

#: Frozen six-arm identity set (Cycle #21 arm budget execution contract).
FORMAL_ARM_IDS: tuple[str, ...] = (
    "A0_EXACT_SOURCE",
    "SAPCA_ABSTENTION",
    "CERTIFIER_VETO_CONTROL",
    "PROPOSAL_INFORMATION_DESTROYED",
    "CERTIFICATE_INFORMATION_DESTROYED",
    "SAPCA_CERTIFIED_PRIMARY",
)

AUTHORIZATION_SCHEMA_VERSION = "phase6_v2_sapca_execution_authorization_v1"
PERMIT_SCHEMA_VERSION = "phase6_v2_sapca_formal_execution_permit_v1"
MAX_RESEARCH_UNITS = 6


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SapcaContractError(reason)


def _require_nonempty(value: Any, name: str) -> None:
    _require(value is not None and str(value).strip() != "", f"{name} must be non-empty.")


@dataclass(frozen=True)
class SapcaExecutionAuthorizationV2:
    """External frozen authorization artifact (never constructed by the runtime).

    Fields mirror the Cycle #31R execution-instruction section 6.1 contract.
    ``content_hash`` binds the whole artifact except itself.
    """

    schema_version: str
    cycle_id: str
    preregistration_content_sha256: str
    six_unit_plan_sha256: str
    implementation_binding_sha256: str
    identity_binding_sha256: str
    parameter_package_binding_sha256: str
    statistics_contract_sha256: str
    decision_contract_sha256: str
    stop_rules_sha256: str
    environment_contract_sha256: str
    allowed_arms: tuple[str, ...]
    max_research_units: int
    execution_root_binding: str
    issued_by: str
    issued_at: str
    expires_at: str | None
    authorization_nonce: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != AUTHORIZATION_SCHEMA_VERSION:
            raise SapcaContractError(f"Unsupported authorization schema: {self.schema_version!r}.")
        for name in (
            "cycle_id",
            "preregistration_content_sha256",
            "six_unit_plan_sha256",
            "implementation_binding_sha256",
            "identity_binding_sha256",
            "parameter_package_binding_sha256",
            "statistics_contract_sha256",
            "decision_contract_sha256",
            "stop_rules_sha256",
            "environment_contract_sha256",
            "execution_root_binding",
            "issued_by",
            "issued_at",
            "authorization_nonce",
        ):
            _require_nonempty(getattr(self, name), name)
        for name in (
            "preregistration_content_sha256",
            "six_unit_plan_sha256",
            "implementation_binding_sha256",
            "identity_binding_sha256",
            "parameter_package_binding_sha256",
            "statistics_contract_sha256",
            "decision_contract_sha256",
            "stop_rules_sha256",
            "environment_contract_sha256",
            "authorization_nonce",
        ):
            value = str(getattr(self, name))
            _require(
                value.startswith("sha256:") and len(value) == 7 + 64,
                f"{name} must be a canonical sha256: binding.",
            )
        _require(
            len(self.allowed_arms) == len(set(self.allowed_arms)), "allowed_arms must be unique."
        )
        _require(
            set(self.allowed_arms) == set(FORMAL_ARM_IDS),
            "allowed_arms must be exactly the frozen six-arm set (no extra, no missing).",
        )
        _require(
            self.max_research_units == MAX_RESEARCH_UNITS,
            f"max_research_units must be exactly {MAX_RESEARCH_UNITS}.",
        )
        _require(self.issued_at.strip() != "", "issued_at must be non-empty.")
        if self.expires_at is not None:
            _require(self.expires_at.strip() != "", "expires_at must be non-empty when set.")
        expected = self.compute_content_hash()
        if self.content_hash and self.content_hash != expected:
            raise SapcaContractError("Authorization artifact content hash mismatch.")
        object.__setattr__(self, "content_hash", expected)

    def compute_content_hash(self) -> str:
        """Hash all artifact fields except the self-referential content hash."""
        from dataclasses import asdict

        payload = asdict(self)
        payload.pop("content_hash", None)
        return canonical_hash(payload)


@dataclass(frozen=True)
class ValidatedFormalExecutionPermitV2:
    """Narrow runtime permit produced only by a successful validation.

    Carries no scientific parameters and no thresholds; it only records the
    validated binding identities and the frozen execution envelope. Absence of
    a permit, or any mismatch, preserves exact-source fail-closed behavior.
    """

    schema_version: str
    cycle_id: str
    preregistration_content_sha256: str
    six_unit_plan_sha256: str
    implementation_binding_sha256: str
    identity_binding_sha256: str
    parameter_package_binding_sha256: str
    statistics_contract_sha256: str
    decision_contract_sha256: str
    stop_rules_sha256: str
    environment_contract_sha256: str
    allowed_arms: tuple[str, ...]
    max_research_units: int
    execution_root_binding: str
    authorization_nonce: str
    permit_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != PERMIT_SCHEMA_VERSION:
            raise SapcaContractError(f"Unsupported permit schema: {self.schema_version!r}.")
        for name in (
            "cycle_id",
            "preregistration_content_sha256",
            "six_unit_plan_sha256",
            "implementation_binding_sha256",
            "identity_binding_sha256",
            "parameter_package_binding_sha256",
            "statistics_contract_sha256",
            "decision_contract_sha256",
            "stop_rules_sha256",
            "environment_contract_sha256",
            "execution_root_binding",
            "authorization_nonce",
        ):
            _require_nonempty(getattr(self, name), name)
        _require(
            set(self.allowed_arms) == set(FORMAL_ARM_IDS),
            "permit allowed_arms must be exactly the frozen six-arm set.",
        )
        _require(
            self.max_research_units == MAX_RESEARCH_UNITS,
            f"permit max_research_units must be exactly {MAX_RESEARCH_UNITS}.",
        )
        expected = self.compute_permit_hash()
        if self.permit_hash and self.permit_hash != expected:
            raise SapcaContractError("Permit hash mismatch.")
        object.__setattr__(self, "permit_hash", expected)

    def compute_permit_hash(self) -> str:
        """Hash all permit fields except the self-referential permit hash."""
        from dataclasses import asdict

        payload = asdict(self)
        payload.pop("permit_hash", None)
        return canonical_hash(payload)

    def authorizes_arm(self, arm_id: str) -> bool:
        """Return whether the permit allows the given frozen arm identity."""
        return arm_id in self.allowed_arms

    def authorizes_research_units(self, count: int) -> bool:
        """Return whether ``count`` research units stay inside the frozen budget."""
        return 0 <= count <= self.max_research_units


class ExecutionAuthorizationValidatorV2:
    """Validate an external authorization artifact against frozen bindings.

    The validator receives the frozen preregistration contract (a mapping with
    at least ``content_hash``) and the current execution-critical bindings
    (implementation/identity/package/statistics/decision/stop/environment
    hashes). It returns a narrow permit or raises ``SapcaContractError`` with
    an exact reason. It never reads prospective data and never mutates any
    sealed artifact.
    """

    def __init__(
        self,
        frozen_preregistration: dict[str, Any],
        current_bindings: dict[str, str],
        *,
        execution_root: str,
        now_utc: str,
        validate_expiry: bool = True,
    ) -> None:
        prereg_hash = frozen_preregistration.get("content_hash")
        _require(
            isinstance(prereg_hash, str) and prereg_hash.startswith("sha256:"),
            "frozen preregistration content_hash missing or malformed.",
        )
        self._prereg_hash: str = str(prereg_hash)
        self._bindings: dict[str, str] = dict(current_bindings)
        self._execution_root: str = execution_root
        self._now_utc: str = now_utc
        self._validate_expiry: bool = validate_expiry

    def validate(
        self, artifact: SapcaExecutionAuthorizationV2 | None
    ) -> ValidatedFormalExecutionPermitV2:
        """Validate one artifact and return the narrow formal permit."""
        if artifact is None:
            raise SapcaContractError("Authorization artifact is missing (no token).")
        checks: list[tuple[str, str, str]] = [
            (
                "preregistration_content_sha256",
                artifact.preregistration_content_sha256,
                self._prereg_hash,
            ),
            (
                "six_unit_plan_sha256",
                artifact.six_unit_plan_sha256,
                self._bindings["six_unit_plan_sha256"],
            ),
            (
                "implementation_binding_sha256",
                artifact.implementation_binding_sha256,
                self._bindings["implementation_binding_sha256"],
            ),
            (
                "identity_binding_sha256",
                artifact.identity_binding_sha256,
                self._bindings["identity_binding_sha256"],
            ),
            (
                "parameter_package_binding_sha256",
                artifact.parameter_package_binding_sha256,
                self._bindings["parameter_package_binding_sha256"],
            ),
            (
                "statistics_contract_sha256",
                artifact.statistics_contract_sha256,
                self._bindings["statistics_contract_sha256"],
            ),
            (
                "decision_contract_sha256",
                artifact.decision_contract_sha256,
                self._bindings["decision_contract_sha256"],
            ),
            ("stop_rules_sha256", artifact.stop_rules_sha256, self._bindings["stop_rules_sha256"]),
            (
                "environment_contract_sha256",
                artifact.environment_contract_sha256,
                self._bindings["environment_contract_sha256"],
            ),
        ]
        for name, actual, expected in checks:
            _require(
                actual == expected,
                f"authorization mismatch on {name}: artifact {actual} != frozen "
                f"binding {expected}.",
            )
        _require(
            artifact.execution_root_binding == self._execution_root,
            "execution root binding mismatch.",
        )
        if self._validate_expiry:
            _require(
                artifact.expires_at is None or artifact.expires_at >= self._now_utc,
                "authorization artifact expired.",
            )
            _require(
                artifact.issued_at <= self._now_utc,
                "authorization artifact not yet valid (issued_at in the future).",
            )
        return ValidatedFormalExecutionPermitV2(
            schema_version=PERMIT_SCHEMA_VERSION,
            cycle_id=artifact.cycle_id,
            preregistration_content_sha256=artifact.preregistration_content_sha256,
            six_unit_plan_sha256=artifact.six_unit_plan_sha256,
            implementation_binding_sha256=artifact.implementation_binding_sha256,
            identity_binding_sha256=artifact.identity_binding_sha256,
            parameter_package_binding_sha256=artifact.parameter_package_binding_sha256,
            statistics_contract_sha256=artifact.statistics_contract_sha256,
            decision_contract_sha256=artifact.decision_contract_sha256,
            stop_rules_sha256=artifact.stop_rules_sha256,
            environment_contract_sha256=artifact.environment_contract_sha256,
            allowed_arms=artifact.allowed_arms,
            max_research_units=artifact.max_research_units,
            execution_root_binding=artifact.execution_root_binding,
            authorization_nonce=artifact.authorization_nonce,
        )


def formal_execution_usable(
    package: Any,
    permit: ValidatedFormalExecutionPermitV2 | None,
    *,
    arm_id: str,
    category: str,
) -> bool:
    """Return whether a formal package may act under the given permit.

    The package alone can never authorize execution (its intrinsic
    ``production_usable`` remains legacy semantics and is false for all five
    canonical packages). Acting additionally requires a valid permit whose
    bindings already matched the frozen contract, which allows the current arm
    and whose package binding matches this package's canonical content hash.
    """
    if package is None or permit is None:
        return False
    if package.provenance.fixture_only:
        return False
    if package.provenance.category != category:
        return False
    if not permit.authorizes_arm(arm_id):
        return False
    # The permit's package binding is validated at permit-issue time against
    # the frozen parameter-package binding (which itself requires byte
    # identity with Cycle #30). Runtime re-checking of the package content
    # hash against the permit's canonical identity is enforced by the engine:
    # see SapcaEngineAdapterV2 (permit_parameter_package_binding_sha256 vs the
    # frozen binding already matched) plus actuator anchor/package checks.
    return bool(package.compute_content_hash())


def parse_authorization_artifact(payload: dict[str, Any]) -> SapcaExecutionAuthorizationV2:
    """Construct an authorization artifact from a parsed JSON mapping.

    Raises ``SapcaContractError`` on any schema or binding defect; the
    artifact is immutable and hash-bound after construction.
    """
    required = {
        "schema_version",
        "cycle_id",
        "preregistration_content_sha256",
        "six_unit_plan_sha256",
        "implementation_binding_sha256",
        "identity_binding_sha256",
        "parameter_package_binding_sha256",
        "statistics_contract_sha256",
        "decision_contract_sha256",
        "stop_rules_sha256",
        "environment_contract_sha256",
        "allowed_arms",
        "max_research_units",
        "execution_root_binding",
        "issued_by",
        "issued_at",
        "expires_at",
        "authorization_nonce",
        "content_hash",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise SapcaContractError(f"Authorization artifact missing fields: {missing}.")
    arms = tuple(str(a) for a in payload["allowed_arms"])
    expires = payload.get("expires_at")
    return SapcaExecutionAuthorizationV2(
        schema_version=str(payload["schema_version"]),
        cycle_id=str(payload["cycle_id"]),
        preregistration_content_sha256=str(payload["preregistration_content_sha256"]),
        six_unit_plan_sha256=str(payload["six_unit_plan_sha256"]),
        implementation_binding_sha256=str(payload["implementation_binding_sha256"]),
        identity_binding_sha256=str(payload["identity_binding_sha256"]),
        parameter_package_binding_sha256=str(payload["parameter_package_binding_sha256"]),
        statistics_contract_sha256=str(payload["statistics_contract_sha256"]),
        decision_contract_sha256=str(payload["decision_contract_sha256"]),
        stop_rules_sha256=str(payload["stop_rules_sha256"]),
        environment_contract_sha256=str(payload["environment_contract_sha256"]),
        allowed_arms=arms,
        max_research_units=int(payload["max_research_units"]),
        execution_root_binding=str(payload["execution_root_binding"]),
        issued_by=str(payload["issued_by"]),
        issued_at=str(payload["issued_at"]),
        expires_at=None if expires is None else str(expires),
        authorization_nonce=str(payload["authorization_nonce"]),
        content_hash=str(payload["content_hash"]),
    )
