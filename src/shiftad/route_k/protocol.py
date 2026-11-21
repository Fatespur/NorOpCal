"""Route K K0 protocol loading and validation.

Loads the frozen K0 protocol and geometry YAML files and exposes typed
accessors. Every K0 tool must load the protocol through this module so that
freeze checks (instruction hash, schema version) run once per process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_FILENAME = "k0_protocol.yaml"
GEOMETRY_FILENAME = "k0_geometry.yaml"
INSTRUCTION_FILENAME = "ROUTE_K_K0_ORACLE_GEOMETRY_FEASIBILITY_INSTRUCTIONS.md"


class K0ProtocolError(RuntimeError):
    """Report an invalid or unfrozen K0 protocol."""


@dataclass(frozen=True)
class K0Protocol:
    """Validated K0 protocol snapshot."""

    protocol: dict[str, Any]
    geometry: dict[str, Any]
    protocol_sha256: str
    geometry_sha256: str

    @property
    def categories(self) -> list[str]:
        return list(self.protocol["dataset"]["categories"])

    @property
    def shift_families(self) -> list[str]:
        return list(self.protocol["shift_families"])

    @property
    def fit_severities(self) -> list[int]:
        return list(self.protocol["severity_split"]["fit"])

    @property
    def eval_severities(self) -> list[int]:
        return list(self.protocol["severity_split"]["eval"])

    @property
    def local_neighbors(self) -> int:
        return int(self.geometry["local_neighbors"])

    @property
    def rank_max(self) -> int:
        return int(self.geometry["rank_rule"]["max_rank"])

    @property
    def rank_energy(self) -> float:
        return float(self.geometry["rank_rule"]["min_energy_fraction"])


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(container: dict[str, Any], key: str) -> Any:
    if key not in container:
        raise K0ProtocolError(f"Missing required protocol key {key!r}.")
    return container[key]


def load_protocol(config_dir: Path, repo_root: Path) -> K0Protocol:
    """Load, validate, and freeze-check the K0 protocol files.

    Args:
        config_dir: Directory containing ``k0_protocol.yaml`` and
            ``k0_geometry.yaml``.
        repo_root: Repository root used to resolve the instruction file.

    Returns:
        A validated protocol snapshot with content hashes.

    Raises:
        K0ProtocolError: If files are missing, unfrozen, or inconsistent.
    """
    protocol_path = config_dir / PROTOCOL_FILENAME
    geometry_path = config_dir / GEOMETRY_FILENAME
    for path in (protocol_path, geometry_path):
        if not path.is_file():
            raise K0ProtocolError(f"Missing protocol file {path}.")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    geometry = yaml.safe_load(geometry_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not isinstance(geometry, dict):
        raise K0ProtocolError("Protocol files must contain YAML mappings.")
    if protocol.get("status") != "FROZEN_K0_A":
        raise K0ProtocolError(
            f"Protocol status must be FROZEN_K0_A, got {protocol.get('status')!r}."
        )
    if protocol.get("schema_version") != "1.0":
        raise K0ProtocolError(
            f"Protocol schema_version must be '1.0', got {protocol.get('schema_version')!r}."
        )
    instruction_path = repo_root / "docs" / INSTRUCTION_FILENAME
    if not instruction_path.is_file():
        raise K0ProtocolError(f"Missing instruction file {instruction_path}.")
    instruction_sha = _sha256_file(instruction_path)
    declared = protocol.get("instruction_sha256")
    if declared and declared != instruction_sha:
        raise K0ProtocolError(
            "Instruction SHA-256 mismatch: protocol declares "
            f"{declared}, file is {instruction_sha}."
        )
    # Structural validation of the frozen dimensions.
    dataset = _require(protocol, "dataset")
    categories = _require(dataset, "categories")
    families = _require(protocol, "shift_families")
    if len(categories) != 5 or len(families) != 13:
        raise K0ProtocolError("Protocol must freeze exactly 5 categories and 13 shift families.")
    severity_split = _require(protocol, "severity_split")
    if set(severity_split["fit"]) != {1, 2} or set(severity_split["eval"]) != {3, 4}:
        raise K0ProtocolError(
            "Severity split must be fit=[1, 2] and eval=[3, 4] for the frozen K0 protocol."
        )
    methods = _require(protocol, "methods")
    required_methods = {
        "m0_a0",
        "m1_normal_pca",
        "m2_global_shift_pca",
        "m3_random_matched_rank",
        "m4_destroyed_correspondence",
        "m5_oracle_local_paired_shift",
    }
    if set(methods) != required_methods:
        raise K0ProtocolError(f"Method matrix mismatch: {methods}.")
    return K0Protocol(
        protocol=protocol,
        geometry=geometry,
        protocol_sha256=_sha256_file(protocol_path),
        geometry_sha256=_sha256_file(geometry_path),
    )
