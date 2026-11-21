"""Shift benchmark workflow with atomic resume/failure records.

Fixture scope generates synthetic-fixture shift cells (category x shift x
severity) at source resolution, records per-sample metadata and fingerprints,
and writes atomic cell manifests with completion markers.

Formal scope (Stage D) evaluates the frozen Phase 5 matrix over the accepted
Phase 4 models: source-resolution shifting of MVTec AD test samples, frozen
Phase 4 thresholds, per-cell result manifests per protocol Section 9, atomic
completion markers, interruption-safe resume, and a commit-bound Stage C
authorization artifact. Formal execution refuses to start without the
artifact, refuses output roots that overlap Phase 4 or dataset roots, and
never tunes thresholds from shifted data.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor

from shiftad.data.base import DatasetError, DatasetIntegrityError
from shiftad.data.fixture import FIXTURE_SCOPE, SyntheticFixtureDataset
from shiftad.data.mvtec_ad import MVTecADDataset
from shiftad.evaluation.image_metrics import MetricError, compute_image_metrics
from shiftad.evaluation.pixel_metrics import compute_pixel_metrics
from shiftad.methods.factory import load_detector
from shiftad.shifts.base import BASE_SEED, PROTOCOL_VERSION, derive_seed
from shiftad.shifts.fingerprint import compute_fingerprint, compute_suite_fingerprint
from shiftad.shifts.metadata import (
    compute_shift_metadata,
    file_sha256,
    image_sha256,
    mask_sha256,
    per_sample_fingerprint,
    sha256_bytes,
)
from shiftad.shifts.registry import (
    CLEAN_SHIFT_NAME,
    CLEAN_VARIANT,
    SHIFT_NAMES,
    apply_shift,
    derive_clean_seed,
    shift_variant,
    validate_cell_identity,
)
from shiftad.utils.device import resolve_device

FIXTURE_SCOPE_MARKER = "fixture_scope.txt"
FORMAL_SCOPE_MARKER = "formal_scope.txt"
COMPLETED_MARKER = "COMPLETED"
FAILED_MARKER = "FAILED"
FORMAL_AUTHORIZATION_TOKEN = "STAGE_C_FORMAL_AUTHORIZED"
# Frozen formal matrix constants (protocol Section 10.1; identical to the
# Stage C pre-registration builder's constants).
FORMAL_MODELS = ("dino_memory", "patchcore", "efficientad")
FORMAL_CATEGORIES = ("bottle", "cable", "hazelnut", "metal_nut", "transistor")
FORMAL_TRAINING_SEEDS = (0, 1, 2)
FORMAL_SHIFTED_SEVERITIES = (1, 2, 3, 4)
# Phase 4 evaluation batches use batch_size 4 (configs/experiments/phase45).
FORMAL_BATCH_SIZE = 4
# Category names become directory components under output_dir/cells, so only
# safe characters are accepted (path traversal defense, security review MEDIUM).
_SAFE_CELL_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+\Z")
# The Stage C authorization artifact must contain exactly one line:
# ``STAGE_C_FORMAL_AUTHORIZED <40-hex-commit>``. The commit must be an
# ancestor of the current HEAD (same binding semantics as the Stage C
# readiness auditor). This replaces the former bare substring token check.
_AUTHORIZATION_PATTERN = re.compile(r"^STAGE_C_FORMAL_AUTHORIZED ([0-9a-f]{40})\s*$")
# Fixed timestamp for fixture metadata: keeps fixture manifests byte-deterministic
# (created_at is excluded from fingerprints). Formal artifacts use real UTC.
FIXTURE_CREATED_AT = "2026-08-03T00:00:00Z"


class ShiftBenchmarkError(RuntimeError):
    """Report a benchmark configuration, scope, or workflow failure."""

    def __init__(self, message: str, *, code: str = "SHIFT_BENCHMARK_ERROR") -> None:
        super().__init__(message)
        self.code = code

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error with path redaction for the CLI record."""

        return {"code": self.code, "message": _redact_message(str(self))}


@dataclass(frozen=True)
class FixtureSample:
    """One deterministic synthetic sample before shifting."""

    category: str
    split: str
    defect_type: str
    sample_id: str
    image: np.ndarray
    mask: np.ndarray | None
    label: int


@dataclass(frozen=True)
class CellSpec:
    """One evaluation cell: category x shift family x severity."""

    category: str
    shift_name: str
    severity: int

    @property
    def cell_id(self) -> str:
        """Return the stable directory identifier for this cell."""

        return f"{self.category}__{self.shift_name}__s{self.severity}"


@dataclass(frozen=True)
class FormalCellSpec:
    """One formal evaluation cell (protocol Section 9.1).

    Attributes:
        model: One of ``FORMAL_MODELS``.
        category: MVTec AD category.
        training_seed: One of ``FORMAL_TRAINING_SEEDS``.
        shift_name: Canonical shift family, or ``"clean"`` for Level 0.
        shift_variant: Frozen variant label, or ``"none"`` for clean.
        severity: 0 for the deduplicated clean cell, else 1-4.
    """

    model: str
    category: str
    training_seed: int
    shift_name: str
    shift_variant: str
    severity: int

    @property
    def cell_id(self) -> str:
        """Return the stable directory identifier for this cell."""

        return (
            f"{self.model}__{self.category}__seed{self.training_seed}"
            f"__{self.shift_name}__s{self.severity}"
        )


def _require_mapping(value: object, section: str) -> dict[str, Any]:
    """Require a mapping for one configuration section."""

    if not isinstance(value, dict):
        raise ShiftBenchmarkError(
            f"Configuration section {section!r} must be a mapping.",
            code="CONFIG_SCHEMA",
        )
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    """Reject unknown configuration keys for one section."""

    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ShiftBenchmarkError(
            f"Unknown keys in configuration section {section!r}: {unknown}.",
            code="CONFIG_SCHEMA",
        )


def load_benchmark_config(path: Path) -> dict[str, Any]:
    """Load and structurally validate a Phase 5 benchmark YAML configuration.

    Args:
        path: YAML configuration path.

    Returns:
        The validated raw configuration mapping.

    Raises:
        ShiftBenchmarkError: If the file is missing, invalid YAML, or has an
            unknown top-level schema.
    """

    if not path.is_file():
        raise ShiftBenchmarkError(
            f"Cannot load benchmark config from {path}: expected an existing YAML file.",
            code="CONFIG_LOAD",
        )
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ShiftBenchmarkError(
            f"Cannot parse benchmark config {path}: {exc}",
            code="CONFIG_LOAD",
        ) from exc
    mapping = _require_mapping(parsed, "benchmark")
    _reject_unknown(
        mapping,
        {
            "schema_version",
            "protocol_version",
            "scope",
            "fixture",
            "formal",
            "status",
            "formal_authorized",
        },
        "benchmark",
    )
    if mapping.get("protocol_version") != PROTOCOL_VERSION:
        raise ShiftBenchmarkError(
            f"Config protocol_version {mapping.get('protocol_version')!r} does not "
            f"match frozen protocol {PROTOCOL_VERSION!r}.",
            code="PROTOCOL_MISMATCH",
        )
    scope = mapping.get("scope")
    if scope not in {"fixture", "formal"}:
        raise ShiftBenchmarkError(
            f"Config scope must be 'fixture' or 'formal', got {scope!r}.",
            code="CONFIG_SCHEMA",
        )
    if scope == "formal":
        return _load_formal_section(mapping)
    fixture = _require_mapping(mapping.get("fixture"), "fixture")
    _reject_unknown(
        fixture,
        {
            "dataset",
            "categories",
            "image_sizes",
            "base_seed",
            "model_input_shapes",
            "shifts",
            "severities",
        },
        "fixture",
    )
    if fixture.get("dataset") != "synthetic":
        raise ShiftBenchmarkError(
            f"Fixture scope requires dataset 'synthetic', got {fixture.get('dataset')!r}.",
            code="CONFIG_SCHEMA",
        )
    model_input_shapes = fixture.get("model_input_shapes")
    if not isinstance(model_input_shapes, dict) or not model_input_shapes:
        raise ShiftBenchmarkError(
            "Fixture config requires a non-empty 'model_input_shapes' mapping.",
            code="CONFIG_SCHEMA",
        )
    categories = [str(item) for item in fixture["categories"]]
    for category in categories:
        shape = model_input_shapes.get(category)
        if not (
            isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise ShiftBenchmarkError(
                f"model_input_shapes[{category!r}] must be a positive [H, W] pair, got {shape!r}.",
                code="CONFIG_SCHEMA",
            )
    return mapping


def _load_formal_section(mapping: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen formal benchmark configuration section.

    The frozen ``benchmark_formal.yaml`` keeps ``status:
    FROZEN_STAGE_C_PENDING_STAGE_D``, ``formal_authorized: false`` and
    ``execution_disabled: true`` at all times (Stage C guards); formal
    execution is gated exclusively by the commit-bound Stage C authorization
    artifact. This loader performs structural and matrix validation and
    refuses any deviation from the frozen matrix.

    Args:
        mapping: Parsed benchmark YAML mapping with ``scope == "formal"``.

    Returns:
        The validated raw configuration mapping.

    Raises:
        ShiftBenchmarkError: If any frozen field is missing, altered, or
            inconsistent with the pre-registration matrix.
    """

    if mapping.get("formal_authorized") is not False:
        raise ShiftBenchmarkError(
            "Formal config must keep formal_authorized: false; authorization is "
            "granted only by the commit-bound Stage C artifact.",
            code="CONFIG_SCHEMA",
        )
    if "fixture" in mapping:
        raise ShiftBenchmarkError(
            "Formal config must not contain a 'fixture' section.",
            code="CONFIG_SCHEMA",
        )
    if mapping.get("status") != "FROZEN_STAGE_C_PENDING_STAGE_D":
        raise ShiftBenchmarkError(
            "Formal config status must remain FROZEN_STAGE_C_PENDING_STAGE_D.",
            code="CONFIG_SCHEMA",
        )
    formal = _require_mapping(mapping.get("formal"), "formal")
    _reject_unknown(
        formal,
        {
            "dataset",
            "dataset_root",
            "models",
            "categories",
            "training_seeds",
            "shift_families",
            "shifted_severities",
            "clean_severity",
            "clean_cells_deduplicated",
            "shift_generation",
            "image_cache",
            "aggregation_epsilon",
            "threshold_source",
            "threshold_quantile",
            "model_working_sizes",
            "phase4_manifest",
            "output_root",
            "authorization_required",
            "execution_disabled",
        },
        "formal",
    )
    if formal.get("dataset") != "mvtec_ad":
        raise ShiftBenchmarkError(
            f"Formal dataset must be 'mvtec_ad', got {formal.get('dataset')!r}.",
            code="CONFIG_SCHEMA",
        )
    if [str(item) for item in formal.get("models", [])] != list(FORMAL_MODELS):
        raise ShiftBenchmarkError(
            f"Formal models must be exactly {list(FORMAL_MODELS)}.",
            code="CONFIG_SCHEMA",
        )
    if [str(item) for item in formal.get("categories", [])] != list(FORMAL_CATEGORIES):
        raise ShiftBenchmarkError(
            f"Formal categories must be exactly {list(FORMAL_CATEGORIES)}.",
            code="CONFIG_SCHEMA",
        )
    if [int(item) for item in formal.get("training_seeds", [])] != list(FORMAL_TRAINING_SEEDS):
        raise ShiftBenchmarkError(
            f"Formal training_seeds must be exactly {list(FORMAL_TRAINING_SEEDS)}.",
            code="CONFIG_SCHEMA",
        )
    if [str(item) for item in formal.get("shift_families", [])] != list(SHIFT_NAMES):
        raise ShiftBenchmarkError(
            "Formal shift_families must match the frozen 13-family taxonomy.",
            code="CONFIG_SCHEMA",
        )
    if [int(item) for item in formal.get("shifted_severities", [])] != list(
        FORMAL_SHIFTED_SEVERITIES
    ):
        raise ShiftBenchmarkError(
            f"Formal shifted_severities must be exactly {list(FORMAL_SHIFTED_SEVERITIES)}.",
            code="CONFIG_SCHEMA",
        )
    if formal.get("clean_severity") != 0:
        raise ShiftBenchmarkError("Formal clean_severity must be 0.", code="CONFIG_SCHEMA")
    if formal.get("clean_cells_deduplicated") is not True:
        raise ShiftBenchmarkError(
            "Formal clean_cells_deduplicated must be true.",
            code="CONFIG_SCHEMA",
        )
    if formal.get("shift_generation") != "on_the_fly":
        raise ShiftBenchmarkError(
            "Formal shift_generation must be on_the_fly.",
            code="CONFIG_SCHEMA",
        )
    if formal.get("image_cache") != "none":
        raise ShiftBenchmarkError(
            "Formal image_cache must be none.",
            code="CONFIG_SCHEMA",
        )
    if formal.get("aggregation_epsilon") != 1e-10:
        raise ShiftBenchmarkError(
            "Formal aggregation_epsilon must be 1e-10.",
            code="CONFIG_SCHEMA",
        )
    if formal.get("threshold_source") != "normal_train_holdout_quantile":
        raise ShiftBenchmarkError(
            "Formal threshold_source must be normal_train_holdout_quantile.",
            code="CONFIG_SCHEMA",
        )
    if formal.get("threshold_quantile") != 0.95:
        raise ShiftBenchmarkError(
            "Formal threshold_quantile must be 0.95.",
            code="CONFIG_SCHEMA",
        )
    if formal.get("authorization_required") is not True:
        raise ShiftBenchmarkError(
            "Formal authorization_required must be true.",
            code="CONFIG_SCHEMA",
        )
    if formal.get("execution_disabled") is not True:
        raise ShiftBenchmarkError(
            "Formal execution_disabled must stay true; the execution gate is the "
            "commit-bound authorization artifact.",
            code="CONFIG_SCHEMA",
        )
    for reference_name in ("dataset_root", "output_root"):
        reference = formal.get(reference_name)
        if not isinstance(reference, str) or not reference.startswith("env:"):
            raise ShiftBenchmarkError(
                f"Formal {reference_name} must use an env: reference.",
                code="CONFIG_SCHEMA",
            )
        variable = reference.split("env:", 1)[-1]
        if not variable or os.environ.get(variable) in (None, ""):
            raise ShiftBenchmarkError(
                f"Formal {reference_name} references unset environment variable "
                f"{variable!r}; set it before running the formal benchmark.",
                code="CONFIG_LOAD",
            )
    working_sizes = formal.get("model_working_sizes")
    if not isinstance(working_sizes, dict) or sorted(working_sizes) != sorted(list(FORMAL_MODELS)):
        raise ShiftBenchmarkError(
            "Formal model_working_sizes must cover every formal model.",
            code="CONFIG_SCHEMA",
        )
    for model in FORMAL_MODELS:
        shape = working_sizes[model]
        if not (
            isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise ShiftBenchmarkError(
                f"model_working_sizes[{model!r}] must be a positive [H, W] pair, got {shape!r}.",
                code="CONFIG_SCHEMA",
            )
    phase4_manifest = formal.get("phase4_manifest")
    if not isinstance(phase4_manifest, str) or not phase4_manifest:
        raise ShiftBenchmarkError(
            "Formal phase4_manifest must be a non-empty path.",
            code="CONFIG_SCHEMA",
        )
    return mapping


def _resolve_env_path(reference: str, *, section: str) -> Path:
    """Resolve an ``env:VAR`` configuration reference to a filesystem path.

    Args:
        reference: Configuration value starting with ``env:``.
        section: Configuration section label for error messages.

    Returns:
        The resolved path (not normalized or existence-checked).

    Raises:
        ShiftBenchmarkError: If the variable is missing or empty.
    """

    if not isinstance(reference, str) or not reference.startswith("env:"):
        raise ShiftBenchmarkError(
            f"{section} must use an env: reference, got {reference!r}.",
            code="CONFIG_SCHEMA",
        )
    variable = reference.split("env:", 1)[-1]
    resolved = os.environ.get(variable)
    if not resolved:
        raise ShiftBenchmarkError(
            f"{section} references unset environment variable {variable!r}.",
            code="CONFIG_LOAD",
        )
    return Path(resolved)


def _require_ancestor_commit(commit: str, repo_root: Path) -> None:
    """Require that ``commit`` is an ancestor of (or equal to) the HEAD.

    Args:
        commit: Hex commit hash (already validated by the authorization regex).
        repo_root: Repository root.

    Raises:
        ShiftBenchmarkError: If Git fails or the commit is not an ancestor.
    """

    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ShiftBenchmarkError(
            "Cannot verify the authorization commit against the repository HEAD.",
            code="GIT_PROVENANCE_MISSING",
        ) from exc
    if result.returncode != 0:
        raise ShiftBenchmarkError(
            f"Authorization commit {commit} is not an ancestor of the current "
            "HEAD; the frozen binding is broken.",
            code="FORMAL_NOT_AUTHORIZED",
        )


def _authorize_formal(
    config: dict[str, Any],
    authorization: Path | None,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify the commit-bound Stage C authorization artifact for formal scope.

    The artifact must contain exactly ``STAGE_C_FORMAL_AUTHORIZED
    <40-hex-commit>``; the commit must be an ancestor of the current HEAD
    (same binding semantics as the Stage C readiness auditor), and the
    artifact bytes are SHA-256 hashed so the auditor can re-verify the
    binding (Stage D instruction Sections 3.2 and 5.3).

    Args:
        config: Loaded benchmark configuration.
        authorization: Authorization artifact path.
        repo_root: Repository root for the Git binding check.

    Returns:
        An authorization record for the output manifest, or an empty mapping
        for non-formal scope.

    Raises:
        ShiftBenchmarkError: If the artifact is missing, malformed, or not
            bound to the frozen commit chain.
    """

    if config["scope"] != "formal":
        return {}
    if authorization is None or not authorization.is_file():
        raise ShiftBenchmarkError(
            "Formal scope requires a commit-bound Stage C authorization artifact "
            "(--stage-c-authorization). Formal Phase 5 runs are NOT authorized.",
            code="FORMAL_NOT_AUTHORIZED",
        )
    content = authorization.read_text(encoding="utf-8")
    match = _AUTHORIZATION_PATTERN.fullmatch(content)
    if match is None:
        raise ShiftBenchmarkError(
            f"Authorization artifact {authorization} must contain exactly one line "
            f"'{FORMAL_AUTHORIZATION_TOKEN} <40-hex-commit>'.",
            code="FORMAL_NOT_AUTHORIZED",
        )
    commit = match.group(1)
    _require_ancestor_commit(commit, repo_root)
    # Hash the raw artifact bytes (not the text re-encoded after universal
    # newline handling) so the auditor's read_bytes() check matches exactly.
    artifact_sha256 = sha256_bytes(authorization.read_bytes())
    return {
        "token": FORMAL_AUTHORIZATION_TOKEN,
        "authorized_commit": commit,
        "artifact_sha256": artifact_sha256,
        "verified_head": _git_head(repo_root, strict=True),
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _formal_output_dir(
    config: dict[str, Any],
    cli_output_dir: Path,
    *,
    repo_root: Path,
) -> Path:
    """Resolve and isolate the formal output directory.

    The CLI output directory must equal the resolved ``formal.output_root``
    environment reference, and must not overlap the Phase 4 output root or
    the dataset root (Stage D instruction Section 5.4).

    Args:
        config: Loaded benchmark configuration.
        cli_output_dir: Output directory requested on the command line.
        repo_root: Repository root used to resolve relative Phase 4 paths.

    Returns:
        The resolved output directory.

    Raises:
        ShiftBenchmarkError: On any overlap or mismatch.
    """

    configured = _resolve_env_path(
        str(config["formal"]["output_root"]), section="formal.output_root"
    ).resolve()
    requested = cli_output_dir.resolve()
    if requested != configured:
        raise ShiftBenchmarkError(
            "Formal output-dir must equal the resolved SHIFTAD_PHASE5_FORMAL_OUTPUT "
            f"({configured}); refusing to write elsewhere.",
            code="OUTPUT_ISOLATION",
        )
    phase4_manifest = (repo_root / str(config["formal"]["phase4_manifest"])).resolve()
    phase4_root = phase4_manifest.parent
    if requested.is_relative_to(phase4_root) or phase4_root.is_relative_to(requested):
        raise ShiftBenchmarkError(
            f"Formal output root {requested} overlaps the Phase 4 output root "
            f"{phase4_root}; refusing to write.",
            code="OUTPUT_ISOLATION",
        )
    dataset_root = _resolve_env_path(
        str(config["formal"]["dataset_root"]), section="formal.dataset_root"
    ).resolve()
    if requested.is_relative_to(dataset_root) or dataset_root.is_relative_to(requested):
        raise ShiftBenchmarkError(
            f"Formal output root {requested} overlaps the dataset root "
            f"{dataset_root}; refusing to write.",
            code="OUTPUT_ISOLATION",
        )
    return requested


def _git_head(repo_root: Path, *, strict: bool = False) -> str:
    """Return the current Git HEAD hash.

    Args:
        repo_root: Repository root directory.
        strict: When True, a Git failure raises instead of returning
            ``"unknown"`` (formal artifacts require Git provenance).

    Returns:
        The HEAD hash, or ``"unknown"`` when ``strict`` is False.

    Raises:
        ShiftBenchmarkError: If ``strict`` is True and Git is unavailable.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        if strict:
            raise ShiftBenchmarkError(
                "Formal artifacts require Git provenance but 'git rev-parse HEAD' "
                "failed; refusing to continue.",
                code="GIT_PROVENANCE_MISSING",
            ) from exc
        return "unknown"


def fixture_samples(
    category: str,
    image_size: tuple[int, int],
    base_seed: int,
) -> list[FixtureSample]:
    """Build deterministic synthetic samples for one category/size.

    Uses the repository's synthetic fixture dataset (test split: two normal
    and two anomalous samples) converted to the numpy source-resolution
    contract.

    Args:
        category: Synthetic category label.
        image_size: Source ``(height, width)``.
        base_seed: Frozen base seed (used only for deterministic ordering).

    Returns:
        Four fixture samples with images in ``[H, W, 3]`` float32.
    """

    dataset = SyntheticFixtureDataset(
        split="test",
        image_size=image_size,
        category=category,
    )
    samples: list[FixtureSample] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image_tensor: Tensor = sample["image"]
        mask_tensor: Tensor | None = sample["mask"]
        image = image_tensor.permute(1, 2, 0).numpy().astype(np.float32)
        mask = mask_tensor[0].numpy().astype(np.float32) if mask_tensor is not None else None
        metadata = sample["metadata"]
        split = str(metadata["split"]) if isinstance(metadata.get("split"), str) else "test"
        samples.append(
            FixtureSample(
                category=category,
                split=split,
                defect_type=sample["defect_type"],
                sample_id=sample["sample_id"],
                image=image,
                mask=mask,
                label=int(sample["label"]),
            )
        )
    # Deterministic ordering independent of dataset internals.
    rng = np.random.default_rng(base_seed)
    order = rng.permutation(len(samples))
    return [samples[int(index)] for index in order]


def plan_cells(config: dict[str, Any]) -> list[CellSpec]:
    """Enumerate the fixture cells from the frozen protocol matrix.

    Clean (Level 0) is evaluated once per category; severities 1-4 cover the
    13 families (protocol Section 10.2 deduplication).
    """

    fixture = config["fixture"]
    categories = [str(item) for item in fixture["categories"]]
    for category in categories:
        if not _SAFE_CELL_IDENTIFIER.match(category):
            raise ShiftBenchmarkError(
                f"Unsafe category {category!r} in fixture config: categories are "
                "used as directory components and must match [A-Za-z0-9._-]+.",
                code="CONFIG_SCHEMA",
            )
    configured_shifts = fixture.get("shifts")
    shifts = [str(item) for item in configured_shifts] if configured_shifts else list(SHIFT_NAMES)
    severities = [int(item) for item in fixture.get("severities", [0, 1, 2, 3, 4])]
    for shift in shifts:
        if shift not in SHIFT_NAMES:
            raise ShiftBenchmarkError(
                f"Unknown shift family {shift!r} in fixture config.",
                code="CONFIG_SCHEMA",
            )
    cells: list[CellSpec] = []
    for category in categories:
        if 0 in severities:
            cells.append(CellSpec(category, CLEAN_SHIFT_NAME, 0))
        for shift in shifts:
            for severity in severities:
                if severity > 0:
                    cells.append(CellSpec(category, shift, severity))
    return cells


def plan_formal_cells(config: dict[str, Any]) -> list[FormalCellSpec]:
    """Enumerate the frozen 2,385-cell formal matrix (protocol Section 10.1).

    Returns 2,340 shifted cells plus 45 deduplicated clean cells, ordered by
    ``(model, category, training_seed, shift_name, severity)`` exactly like the
    Stage C pre-registration plan (ordering affects scheduling only).

    Raises:
        ShiftBenchmarkError: If the matrix cardinality deviates from the
            frozen 2,340/45/2,385 counts.
    """

    formal = config["formal"]
    cells: list[FormalCellSpec] = []
    for model in (str(item) for item in formal["models"]):
        for category in (str(item) for item in formal["categories"]):
            if not _SAFE_CELL_IDENTIFIER.match(category):
                raise ShiftBenchmarkError(
                    f"Unsafe category {category!r} in formal config.",
                    code="CONFIG_SCHEMA",
                )
            for seed in (int(item) for item in formal["training_seeds"]):
                cells.append(
                    FormalCellSpec(
                        model=model,
                        category=category,
                        training_seed=seed,
                        shift_name=CLEAN_SHIFT_NAME,
                        shift_variant=CLEAN_VARIANT,
                        severity=0,
                    )
                )
                for shift_name in (str(item) for item in formal["shift_families"]):
                    for severity in (int(item) for item in formal["shifted_severities"]):
                        cells.append(
                            FormalCellSpec(
                                model=model,
                                category=category,
                                training_seed=seed,
                                shift_name=shift_name,
                                shift_variant=shift_variant(shift_name),
                                severity=severity,
                            )
                        )
    cells.sort(
        key=lambda cell: (
            cell.model,
            cell.category,
            cell.training_seed,
            cell.shift_name,
            cell.severity,
        )
    )
    shifted = sum(1 for cell in cells if cell.severity > 0)
    clean = len(cells) - shifted
    if shifted != 2340 or clean != 45 or len(cells) != 2385:
        raise ShiftBenchmarkError(
            f"Formal matrix cardinality mismatch: shifted={shifted} clean={clean} "
            f"total={len(cells)} (expected 2340/45/2385).",
            code="MATRIX_MISMATCH",
        )
    return cells


def _atomic_write(path: Path, content: str) -> None:
    """Write text atomically via an exclusive temp file and rename.

    Uses ``tempfile.mkstemp`` in the destination directory (O_EXCL semantics,
    no symlink following) and ``os.replace`` so partial or raced writes never
    appear as completed outputs.

    Args:
        path: Destination file path.
        content: UTF-8 text to write.

    Raises:
        OSError: If the destination cannot be created or written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise


def _redact_message(message: str) -> str:
    """Replace absolute path-like fragments with a placeholder.

    Failure records must not leak operator-local absolute paths into
    committed artifacts (security review LOW item). Covers drive-letter,
    UNC, and Unix-style absolute paths with at least two separators so
    ratios like ``0/5`` and URLs are not over-matched.
    """

    import re as _re

    return _re.sub(
        r"(?:~[^\s\"']*|[A-Za-z]:[\\/][^\s\"']+|\\\\[^\s\"']+\\[^\s\"']+|\$\{[^}\s]+\}|/(?:[^\s\"'/]+/)+[^\s\"']*)",
        "<path>",
        message,
    )


def _record_failure(output_dir: Path, cell: CellSpec | FormalCellSpec, error: Exception) -> None:
    """Append one machine-readable failure record atomically."""

    failures_path = output_dir / "failures.json"
    records: list[dict[str, Any]] = []
    if failures_path.is_file():
        try:
            parsed = json.loads(failures_path.read_text(encoding="utf-8"))
            if isinstance(parsed, list):
                records = parsed
        except (OSError, json.JSONDecodeError):
            # Preserve a corrupt record file instead of silently discarding
            # the historical failure evidence.
            corrupt_path = failures_path.with_name(f"failures.json.corrupt.{time.time():.0f}")
            with contextlib.suppress(OSError):
                os.replace(failures_path, corrupt_path)
    records.append(
        {
            "run_id": cell.cell_id,
            "status": "failed",
            "failure_code": (
                error.code if isinstance(error, ShiftBenchmarkError) else "CELL_ERROR"
            ),
            "failure_message": _redact_message(str(error)),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    _atomic_write(failures_path, json.dumps(records, indent=2, sort_keys=True) + "\n")


def _clear_failure(output_dir: Path, cell: CellSpec | FormalCellSpec) -> None:
    """Remove one cell's stale failure record after it completes.

    Args:
        output_dir: Benchmark output directory.
        cell: The cell that just completed (or was skipped as completed).
    """

    failures_path = output_dir / "failures.json"
    if not failures_path.is_file():
        return
    try:
        parsed = json.loads(failures_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(parsed, list):
        return
    remaining = [record for record in parsed if record.get("run_id") != cell.cell_id]
    if not remaining:
        failures_path.unlink(missing_ok=True)
    else:
        _atomic_write(
            failures_path,
            json.dumps(remaining, indent=2, sort_keys=True) + "\n",
        )


def _shift_one_sample(
    sample: FixtureSample,
    cell: CellSpec,
    config: dict[str, Any],
    git_head: str,
    base_seed: int,
) -> dict[str, Any]:
    """Shift one sample and assemble its full metadata record."""

    height, width = sample.image.shape[0], sample.image.shape[1]
    category = cell.category
    shift_name = cell.shift_name
    severity = cell.severity
    fixture = config["fixture"]
    model_input_shape = [int(item) for item in fixture["model_input_shapes"][category]]
    clean = shift_name == CLEAN_SHIFT_NAME
    variant = CLEAN_VARIANT if clean else shift_variant(shift_name)
    validate_cell_identity(
        category=category,
        sample_id=sample.sample_id,
        shift_name=shift_name,
        severity=severity,
        base_seed=base_seed,
    )
    derived = (
        derive_clean_seed(category=category, sample_id=sample.sample_id, base_seed=base_seed)
        if clean
        else derive_seed(
            category=category,
            sample_id=sample.sample_id,
            shift_name=shift_name,
            severity=severity,
            base_seed=base_seed,
        )
    )
    if clean:
        result_image = sample.image
        result_mask = sample.mask
        exact_parameters: dict[str, Any] = {"identity": True}
        area_before = None
        area_after = None
        visible = None
    else:
        result = apply_shift(
            shift_name,
            severity,
            sample.image,
            sample.mask,
            derived,
        )
        result_image = result.image
        result_mask = result.mask
        exact_parameters = result.exact_parameters
        area_before = result.mask_area_before
        area_after = result.mask_area_after
        visible = result.defect_visible_after_shift
    return compute_shift_metadata(
        category=category,
        source_split=sample.split,
        source_sample_id=sample.sample_id,
        source_image_path_hash=sha256_bytes(sample.sample_id.encode("utf-8")),
        source_mask_path_hash=(
            sha256_bytes(f"mask:{sample.sample_id}".encode()) if sample.mask is not None else None
        ),
        source_image_sha256=image_sha256(sample.image),
        source_mask_sha256=mask_sha256(sample.mask),
        shift_name=shift_name,
        shift_variant=variant,
        severity=severity,
        exact_parameters=exact_parameters,
        base_seed=base_seed,
        derived_seed=derived,
        transform_code_git_head=git_head,
        input_shape=[height, width],
        output_shape=[height, width],
        model_input_shape=model_input_shape,
        mask_area_before=area_before,
        mask_area_after=area_after,
        defect_visible_after_shift=visible,
        output_image_sha256=image_sha256(result_image),
        output_mask_sha256=mask_sha256(result_mask),
        fixture=True,
        created_at=FIXTURE_CREATED_AT,
    )


def _run_cell(
    output_dir: Path,
    cell: CellSpec,
    samples: list[FixtureSample],
    config: dict[str, Any],
    git_head: str,
    base_seed: int,
) -> dict[str, Any]:
    """Shift all samples of one cell and atomically write its manifest."""

    cell_dir = output_dir / "cells" / cell.cell_id
    if (cell_dir / COMPLETED_MARKER).is_file():
        return {"cell": cell.cell_id, "status": "completed_exists"}
    records = [_shift_one_sample(sample, cell, config, git_head, base_seed) for sample in samples]
    fingerprints = [per_sample_fingerprint(record) for record in records]
    suite = compute_suite_fingerprint(fingerprints)
    manifest = {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "scope": FIXTURE_SCOPE,
        "cell_key": {
            "category": cell.category,
            "shift_name": cell.shift_name,
            "severity": cell.severity,
        },
        "status": "completed",
        "completed_samples": len(records),
        "failed_samples": 0,
        "shift_suite_fingerprint": suite,
        "samples": records,
        "created_at": FIXTURE_CREATED_AT,
    }
    _atomic_write(
        cell_dir / "cell_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        cell_dir / COMPLETED_MARKER,
        suite + "\n",
    )
    return {"cell": cell.cell_id, "status": "completed", "suite_fingerprint": suite}


# ---------------------------------------------------------------------------
# Formal (Stage D) scope helpers
# ---------------------------------------------------------------------------

# Frozen model assets per model family (identical to the Stage C
# pre-registration builder's FORMAL_MODEL_ASSETS).
FORMAL_MODEL_ASSETS = {
    "dino_memory": ("model/memory_bank.pt", "model/coreset_indices.pt"),
    "patchcore": ("model/memory_bank.pt", "model/coreset_indices.pt"),
    "efficientad": ("checkpoints/checkpoint_step_00070000.pt", "model/teacher.pt"),
}
# Principal asset reported as ``model_checkpoint_sha256`` in cell manifests.
FORMAL_MODEL_PRINCIPAL_ASSET = {
    "dino_memory": "model/memory_bank.pt",
    "patchcore": "model/memory_bank.pt",
    "efficientad": "checkpoints/checkpoint_step_00070000.pt",
}
# Repository-rooted path to the Stage C pre-registration package. Formal
# execution anchors model asset hashes to this frozen package before any
# inference (security review MEDIUM: pickle checkpoints must be bound to the
# pre-registered values).
PREREGISTRATION_REPO_PATH = Path("reports/phase5_stage_c/preregistration.json")


def _preregistration_asset_map(
    repo_root: Path,
) -> dict[tuple[str, str, int], dict[str, str]]:
    """Load the Stage C pre-registration asset-hash bindings.

    Returns ``{(model, category, training_seed): {relative: sha256}}`` from
    the frozen package's 45 provenance entries.

    Raises:
        ShiftBenchmarkError: If the package is missing or malformed
            (fail-closed).
    """

    path = repo_root / PREREGISTRATION_REPO_PATH
    if not path.is_file():
        raise ShiftBenchmarkError(
            f"Stage C pre-registration package missing at {PREREGISTRATION_REPO_PATH};",
            code="MODEL_ASSET_MISSING",
        )
    try:
        package = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShiftBenchmarkError(
            "Stage C pre-registration package is corrupt.",
            code="MANIFEST_MISMATCH",
        ) from exc
    entries = package.get("provenance_entries")
    if not isinstance(entries, list) or len(entries) != 45:
        raise ShiftBenchmarkError(
            "Stage C pre-registration provenance entries != 45.",
            code="MANIFEST_MISMATCH",
        )
    mapping: dict[tuple[str, str, int], dict[str, str]] = {}
    for entry in entries:
        model = entry.get("model")
        category = entry.get("category")
        training_seed = entry.get("training_seed")
        assets = entry.get("asset_sha256")
        if not (
            isinstance(model, str)
            and isinstance(category, str)
            and isinstance(training_seed, int)
            and isinstance(assets, dict)
            and assets
        ):
            raise ShiftBenchmarkError(
                "Stage C pre-registration entry is malformed.",
                code="MANIFEST_MISMATCH",
            )
        mapping[(model, category, training_seed)] = {
            str(key): str(value) for key, value in assets.items()
        }
    return mapping


def _formal_model_dir(
    config: dict[str, Any],
    cell: FormalCellSpec,
    repo_root: Path,
) -> Path:
    """Return the accepted Phase 4 model directory for one formal cell.

    The model directory is derived from the frozen Phase 4 canonical
    manifest location: ``<phase4_root>/<category>_<model>_seed<seed>``.
    """

    manifest_path = (repo_root / str(config["formal"]["phase4_manifest"])).resolve()
    return manifest_path.parent / f"{cell.category}_{cell.model}_seed{cell.training_seed}"


def _formal_expected_sample_ids(
    model_dir: Path,
    category: str,
) -> tuple[list[str], str]:
    """Load the expected test sample set recorded by the Phase 4 evaluation.

    Returns ``(sample_ids, dataset_manifest_sha256)``. The manifest lives at
    ``<model_dir>/evaluation/dataset_manifest.json`` and records relative
    sample identities only (no private paths).

    Raises:
        ShiftBenchmarkError: If the manifest is missing or malformed.
    """

    manifest_path = model_dir / "evaluation" / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ShiftBenchmarkError(
            f"Dataset manifest missing for model {model_dir.name}: {manifest_path}.",
            code="MODEL_ASSET_MISSING",
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShiftBenchmarkError(
            f"Dataset manifest {manifest_path} is corrupt.",
            code="MANIFEST_MISMATCH",
        ) from exc
    sample_ids = data.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ShiftBenchmarkError(
            f"Dataset manifest {manifest_path} has no sample_ids.",
            code="MANIFEST_MISMATCH",
        )
    recorded_categories = data.get("categories")
    if recorded_categories != [category]:
        raise ShiftBenchmarkError(
            f"Dataset manifest {manifest_path} categories {recorded_categories!r} "
            f"!= expected [{category!r}].",
            code="MANIFEST_MISMATCH",
        )
    return [str(item) for item in sample_ids], file_sha256(str(manifest_path))


def _source_image_size(dataset_root: Path, category: str) -> tuple[int, int]:
    """Probe the native MVTec AD source size of one category.

    All images of a category share the same size; the first ``test/good``
    image determines ``(height, width)``.

    Raises:
        ShiftBenchmarkError: If the category has no readable test images.
    """

    good_dir = dataset_root / category / "test" / "good"
    if not good_dir.is_dir():
        raise ShiftBenchmarkError(
            f"Missing test/good directory {good_dir} below the MVTec AD root.",
            code="DATASET_MISSING",
        )
    candidates = sorted(path for path in good_dir.iterdir() if path.is_file())
    if not candidates:
        raise ShiftBenchmarkError(
            f"No test images under {good_dir}.",
            code="DATASET_MISSING",
        )
    try:
        with Image.open(candidates[0]) as source:
            width, height = source.size
    except OSError as exc:
        raise ShiftBenchmarkError(
            f"Cannot probe the source size of {candidates[0]}.",
            code="DATASET_MISSING",
        ) from exc
    return height, width


def _load_formal_test_samples(
    dataset_root: Path,
    category: str,
) -> dict[str, Any]:
    """Load the MVTec AD test split at native source resolution.

    Returns ``{"source_size": [H, W], "samples": [...]}`` with each sample
    holding the relative ``sample_id``, ``defect_type``, ``label``, the
    decoded ``image`` (``[H, W, 3]`` float32 in [0, 1]), the binary ``mask``
    (``[H, W]`` float32 or None), and the absolute ``image_path`` (used only
    for file hashing; never written to manifests).

    Raises:
        ShiftBenchmarkError: If the dataset root or category is invalid.
    """

    try:
        source_size = _source_image_size(dataset_root, category)
        dataset = MVTecADDataset(dataset_root, (category,), "test", image_size=source_size)
        samples: list[dict[str, Any]] = []
        for index in range(len(dataset)):
            sample = dataset[index]
            image = sample["image"].permute(1, 2, 0).numpy().astype(np.float32)
            mask = (
                sample["mask"][0].numpy().astype(np.float32) if sample["mask"] is not None else None
            )
            samples.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "defect_type": str(sample["defect_type"]),
                    "label": int(sample["label"]),
                    "image": image,
                    "mask": mask,
                    "image_path": str(sample["image_path"]),
                }
            )
    except (DatasetError, DatasetIntegrityError) as exc:
        raise ShiftBenchmarkError(
            f"Cannot load MVTec AD test samples for {category!r}: {exc}",
            code="DATASET_MISSING",
        ) from exc
    return {"source_size": list(source_size), "samples": samples}


def _resize_for_model_input(
    image: np.ndarray,
    size: tuple[int, int],
    *,
    mask: bool,
) -> np.ndarray:
    """Resize a shifted source-resolution array to the model working size.

    Mirrors the Phase 4 preprocessing pipeline: PIL BILINEAR for images and
    PIL NEAREST with non-zero re-binarization for masks
    (``shiftad.data.preprocessing``).

    Args:
        image: ``[H, W, 3]`` float32 in [0, 1] for images, or ``[H, W]``
            float32 in {0.0, 1.0} for masks.
        size: Model input ``(height, width)``.
        mask: Whether the input is a binary mask.

    Returns:
        The resized array with the same dtype contract.
    """

    if mask:
        array = (image * 255.0).round().astype(np.uint8)
        resized = Image.fromarray(array, mode="L").resize(
            (size[1], size[0]),
            resample=Image.Resampling.NEAREST,
        )
        return (np.asarray(resized, dtype=np.float32) > 0).astype(np.float32)
    array = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    resized = Image.fromarray(array, mode="RGB").resize(
        (size[1], size[0]),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _shift_formal_sample(
    sample: dict[str, Any],
    cell: FormalCellSpec,
    config: dict[str, Any],
    git_head: str,
) -> dict[str, Any]:
    """Shift one formal sample at source resolution and record its metadata.

    The per-sample record binds the source image by its file bytes hash
    (protocol Section 8.2: "binds the source image hash") and the shifted
    output by its array hash. ``created_at`` uses real UTC (Stage C:
    fingerprint-excluded).
    """

    height, width = sample["image"].shape[0], sample["image"].shape[1]
    clean = cell.shift_name == CLEAN_SHIFT_NAME
    validate_cell_identity(
        category=cell.category,
        sample_id=sample["sample_id"],
        shift_name=cell.shift_name,
        severity=cell.severity,
        base_seed=BASE_SEED,
    )
    derived = (
        derive_clean_seed(
            category=cell.category,
            sample_id=sample["sample_id"],
            base_seed=BASE_SEED,
        )
        if clean
        else derive_seed(
            category=cell.category,
            sample_id=sample["sample_id"],
            shift_name=cell.shift_name,
            severity=cell.severity,
            base_seed=BASE_SEED,
        )
    )
    if clean:
        result_image = sample["image"]
        result_mask = sample["mask"]
        exact_parameters: dict[str, Any] = {"identity": True}
        area_before = None
        area_after = None
        visible = None
    else:
        result = apply_shift(
            cell.shift_name,
            cell.severity,
            sample["image"],
            sample["mask"],
            derived,
        )
        result_image = result.image
        result_mask = result.mask
        exact_parameters = result.exact_parameters
        area_before = result.mask_area_before
        area_after = result.mask_area_after
        visible = result.defect_visible_after_shift
    return {
        "record": compute_shift_metadata(
            category=cell.category,
            source_split="test",
            source_sample_id=sample["sample_id"],
            source_image_path_hash=sha256_bytes(sample["sample_id"].encode("utf-8")),
            source_mask_path_hash=(
                sha256_bytes(f"mask:{sample['sample_id']}".encode())
                if sample["mask"] is not None
                else None
            ),
            source_image_sha256=file_sha256(sample["image_path"]),
            source_mask_sha256=mask_sha256(sample["mask"]),
            shift_name=cell.shift_name,
            shift_variant=CLEAN_VARIANT if clean else cell.shift_variant,
            severity=cell.severity,
            exact_parameters=exact_parameters,
            base_seed=BASE_SEED,
            derived_seed=derived,
            transform_code_git_head=git_head,
            input_shape=[height, width],
            output_shape=[height, width],
            model_input_shape=[
                int(item) for item in config["formal"]["model_working_sizes"][cell.model]
            ],
            mask_area_before=area_before,
            mask_area_after=area_after,
            defect_visible_after_shift=visible,
            output_image_sha256=image_sha256(result_image),
            output_mask_sha256=mask_sha256(result_mask),
            fixture=False,
            created_at=None,
        ),
        "image": result_image,
        "mask": result_mask,
        "label": sample["label"],
    }


def _evaluate_formal_samples(
    detector: Any,
    samples: list[dict[str, Any]],
    shifted: list[dict[str, Any]],
    model_input_shape: tuple[int, int],
    *,
    image_threshold: float,
    pixel_threshold: float,
) -> dict[str, Any]:
    """Run the frozen-threshold evaluation over one cell's shifted samples.

    Batches follow the Phase 4 ``batch_size: 4`` convention. Pixel metrics
    are recorded as null with a reason when every anomaly mask vanishes
    after a geometric shift (protocol risk register: handle gracefully);
    any other metric error fails the cell.
    """

    labels: list[int] = []
    scores: list[float] = []
    masks: list[Tensor] = []
    maps: list[Tensor] = []
    for start in range(0, len(samples), FORMAL_BATCH_SIZE):
        batch_samples = shifted[start : start + FORMAL_BATCH_SIZE]
        images = torch.stack(
            [
                torch.from_numpy(
                    _resize_for_model_input(item["image"], model_input_shape, mask=False)
                ).permute(2, 0, 1)
                for item in batch_samples
            ]
        )
        prediction = detector.predict(images)
        for index, item in enumerate(batch_samples):
            labels.append(item["label"])
            scores.append(float(prediction.image_scores[index]))
            resized_mask = (
                _resize_for_model_input(
                    item["mask"],
                    model_input_shape,
                    mask=True,
                )
                if item["mask"] is not None
                else None
            )
            masks.append(
                torch.from_numpy(resized_mask).unsqueeze(0)
                if resized_mask is not None
                else torch.zeros(
                    (1, model_input_shape[0], model_input_shape[1]),
                    dtype=torch.float32,
                )
            )
            maps.append(prediction.anomaly_maps[index])
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float32)
    image_metrics = compute_image_metrics(labels, scores, threshold=image_threshold)
    mask_tensor = torch.stack(masks)
    map_tensor = torch.stack(maps)
    pixel_fields: dict[str, Any]
    try:
        pixel_metrics = compute_pixel_metrics(
            mask_tensor,
            map_tensor,
            threshold=pixel_threshold,
        )
        pixel_fields = pixel_metrics.to_dict()
    except MetricError as exc:
        message = str(exc)
        if ("background and anomalous mask pixels" in message) or (
            "connected anomaly region" in message
        ):
            pixel_fields = {
                "auroc": None,
                "aupr": None,
                "threshold": pixel_threshold,
                "iou": None,
                "dice": None,
                "aupro": None,
                "aupro_max_false_positive_rate": 0.3,
                "reason": "no_anomalous_pixels_after_shift",
            }
        else:
            raise
    normal_scores = scores_array[labels_array == 0]
    abnormal_scores = scores_array[labels_array == 1]
    normal_fpr = float(np.mean(normal_scores > image_threshold)) if normal_scores.size else None
    return {
        "image_auroc": float(image_metrics.auroc),
        "image_aupr": float(image_metrics.aupr),
        "image_f1_at_frozen_threshold": float(image_metrics.f1),
        "pixel_auroc": pixel_fields.get("auroc"),
        "pixel_aupr": pixel_fields.get("aupr"),
        "aupro": pixel_fields.get("aupro"),
        "pixel_metrics_note": pixel_fields.get("reason"),
        "normal_fpr": normal_fpr,
        "normal_score_mean": float(normal_scores.mean()) if normal_scores.size else None,
        "normal_score_std": float(normal_scores.std()) if normal_scores.size else None,
        "abnormal_score_mean": (float(abnormal_scores.mean()) if abnormal_scores.size else None),
        "frozen_image_threshold": float(image_threshold),
        "frozen_pixel_threshold": float(pixel_threshold),
    }


def _load_formal_model(
    model_dir: Path,
    cell: FormalCellSpec,
    config: dict[str, Any],
    repo_root: Path,
    device: torch.device,
    *,
    expected_assets: dict[str, str] | None,
) -> dict[str, Any]:
    """Load one frozen Phase 4 detector and verify its provenance bindings.

    Verifies every asset hash against the Stage C pre-registration package,
    the frozen calibration thresholds, the Phase 4 dataset manifest, and the
    EfficientAD training step count before inference. Any mismatch aborts
    the run (fail-closed).

    Raises:
        ShiftBenchmarkError: With ``MODEL_LOAD_FAILED`` for load failures and
            ``MODEL_ASSET_MISSING`` / ``HASH_MISMATCH`` for provenance breaks.
    """

    assets: dict[str, str] = {}
    for relative in FORMAL_MODEL_ASSETS[cell.model]:
        path = model_dir / relative
        if not path.is_file():
            raise ShiftBenchmarkError(
                f"Model asset missing for {model_dir.name}: {path}.",
                code="MODEL_ASSET_MISSING",
            )
        assets[relative] = file_sha256(str(path))
    if expected_assets is None or assets != expected_assets:
        raise ShiftBenchmarkError(
            f"Model asset hashes for {model_dir.name} do not match the Stage C "
            "pre-registration package; refusing to load.",
            code="HASH_MISMATCH",
        )
    calibration_path = model_dir / "model" / "calibration.json"
    if not calibration_path.is_file():
        raise ShiftBenchmarkError(
            f"calibration.json missing for {model_dir.name}.",
            code="MODEL_ASSET_MISSING",
        )
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShiftBenchmarkError(
            f"calibration.json is corrupt for {model_dir.name}.",
            code="HASH_MISMATCH",
        ) from exc
    image_threshold = calibration.get("image_threshold")
    pixel_threshold = calibration.get("pixel_threshold")
    threshold_source = calibration.get("threshold_source")
    threshold_quantile = calibration.get("quantile")
    if (
        threshold_source != "normal_train_holdout_quantile"
        or threshold_quantile != config["formal"]["threshold_quantile"]
        or not isinstance(image_threshold, (int, float))
        or not isinstance(pixel_threshold, (int, float))
    ):
        raise ShiftBenchmarkError(
            f"Frozen threshold provenance broken for {model_dir.name}.",
            code="HASH_MISMATCH",
        )
    expected_ids, dataset_manifest_sha256 = _formal_expected_sample_ids(model_dir, cell.category)
    split_fingerprint = compute_fingerprint({"sample_ids": sorted(expected_ids)})
    total_optimizer_steps: int | None = None
    if cell.model == "efficientad":
        metadata_path = model_dir / "model" / "metadata.json"
        if not metadata_path.is_file():
            raise ShiftBenchmarkError(
                f"metadata.json missing for {model_dir.name}.",
                code="MODEL_ASSET_MISSING",
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ShiftBenchmarkError(
                f"metadata.json is corrupt for {model_dir.name}.",
                code="MANIFEST_MISMATCH",
            ) from exc
        total_optimizer_steps = metadata.get("build", {}).get("train_steps")
        if not isinstance(total_optimizer_steps, int) or total_optimizer_steps <= 0:
            raise ShiftBenchmarkError(
                f"Unexpected train_steps in {metadata_path}.",
                code="MANIFEST_MISMATCH",
            )
        # Security review MEDIUM: the inference weights that EfficientAD
        # actually loads (student/autoencoder/normalizer) are not covered by
        # the frozen pre-registration package; anchor them explicitly to the
        # hashes declared in metadata.json (fail-closed against replacing a
        # weight file without updating its self-declared digest).
        declared_hashes = metadata.get("hashes")
        if not isinstance(declared_hashes, dict):
            raise ShiftBenchmarkError(
                f"metadata.json hashes missing for {model_dir.name}.",
                code="MANIFEST_MISMATCH",
            )
        for weight_name in ("student", "autoencoder", "normalizer"):
            weight_path = model_dir / "model" / f"{weight_name}.pt"
            if not weight_path.is_file():
                raise ShiftBenchmarkError(
                    f"Inference weight missing for {model_dir.name}: {weight_name}.pt.",
                    code="MODEL_ASSET_MISSING",
                )
            declared = declared_hashes.get(weight_name)
            if not isinstance(declared, str) or file_sha256(str(weight_path)) != declared:
                raise ShiftBenchmarkError(
                    f"Inference weight {weight_name}.pt hash for {model_dir.name} does "
                    "not match metadata.json; refusing to load.",
                    code="HASH_MISMATCH",
                )
    try:
        # The accepted Phase 4 run layout keeps the detector package below
        # ``<run_dir>/model/`` (manifest.json, config.resolved.yaml, assets).
        detector = load_detector(model_dir / "model", device=device)
    except Exception as exc:
        raise ShiftBenchmarkError(
            f"Cannot load detector {model_dir.name}: {_redact_message(str(exc))}",
            code="MODEL_LOAD_FAILED",
        ) from exc
    return {
        "detector": detector,
        "model_assets_sha256": assets,
        "model_checkpoint_sha256": assets[FORMAL_MODEL_PRINCIPAL_ASSET[cell.model]],
        "image_threshold": float(image_threshold),
        "pixel_threshold": float(pixel_threshold),
        "threshold_source": str(threshold_source),
        "expected_sample_ids": expected_ids,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_fingerprint": split_fingerprint,
        "total_optimizer_steps": total_optimizer_steps,
    }


def _execute_formal_cell(
    output_dir: Path,
    cell: FormalCellSpec,
    dataset: dict[str, Any],
    model_info: dict[str, Any],
    config: dict[str, Any],
    git_head: str,
) -> dict[str, Any]:
    """Execute one formal cell and atomically write its result manifest.

    Raises:
        ShiftBenchmarkError: If the cell was already completed elsewhere or
            any per-cell integrity check fails.
    """

    cell_dir = output_dir / "cells" / cell.cell_id
    if (cell_dir / COMPLETED_MARKER).is_file():
        return {"cell": cell.cell_id, "status": "completed_exists"}
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    shifted = [
        _shift_formal_sample(sample, cell, config, git_head) for sample in dataset["samples"]
    ]
    records = [item["record"] for item in shifted]
    fingerprints = [per_sample_fingerprint(record) for record in records]
    suite = compute_suite_fingerprint(fingerprints)
    working_size = config["formal"]["model_working_sizes"][cell.model]
    model_input_shape = (int(working_size[0]), int(working_size[1]))
    metrics = _evaluate_formal_samples(
        model_info["detector"],
        dataset["samples"],
        shifted,
        model_input_shape,
        image_threshold=model_info["image_threshold"],
        pixel_threshold=model_info["pixel_threshold"],
    )
    end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "scope": "formal",
        "cell_key": {
            "model": cell.model,
            "category": cell.category,
            "training_seed": cell.training_seed,
            "shift_name": cell.shift_name,
            "shift_variant": cell.shift_variant,
            "severity": cell.severity,
        },
        "status": "completed",
        "completed_samples": len(records),
        "failed_samples": 0,
        **{
            key: metrics[key]
            for key in (
                "image_auroc",
                "image_aupr",
                "image_f1_at_frozen_threshold",
                "pixel_auroc",
                "pixel_aupr",
                "aupro",
                "normal_fpr",
                "normal_score_mean",
                "normal_score_std",
                "abnormal_score_mean",
                "frozen_image_threshold",
                "frozen_pixel_threshold",
            )
        },
        "pixel_metrics_note": metrics["pixel_metrics_note"],
        "threshold_source": model_info["threshold_source"],
        "total_optimizer_steps": model_info["total_optimizer_steps"],
        "model_checkpoint_sha256": model_info["model_checkpoint_sha256"],
        "model_assets_sha256": model_info["model_assets_sha256"],
        "split_fingerprint": model_info["split_fingerprint"],
        "dataset_manifest_sha256": model_info["dataset_manifest_sha256"],
        "shift_suite_fingerprint": suite,
        "start_timestamp": start_utc,
        "end_timestamp": end_utc,
        "failure_code": None,
        "created_at": end_utc,
        "samples": records,
    }
    _atomic_write(
        cell_dir / "cell_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        cell_dir / COMPLETED_MARKER,
        suite + "\n",
    )
    return {"cell": cell.cell_id, "status": "completed", "suite_fingerprint": suite}


def _validate_completed_formal_cell(
    output_dir: Path,
    cell: FormalCellSpec,
    config: dict[str, Any],
    repo_root: Path,
) -> None:
    """Validate an existing formal completion marker and its cell manifest.

    Mirrors the fixture validation: manifest parses, scope/protocol/cell key
    match, status completed, no failed samples, the sample set equals the
    Phase 4 recorded test set, the suite fingerprint recomputes, and the
    marker content equals the suite fingerprint.

    Raises:
        ShiftBenchmarkError: If any completion integrity check fails.
    """

    cell_dir = output_dir / "cells" / cell.cell_id
    marker_path = cell_dir / COMPLETED_MARKER
    if not marker_path.is_file():
        return
    manifest_path = cell_dir / "cell_manifest.json"
    if not manifest_path.is_file():
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} has a completion marker but no cell_manifest.json; "
            "refusing to treat it as completed.",
            code="MANIFEST_MISSING",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest is corrupt; refusing to treat it as completed.",
            code="MANIFEST_CORRUPT",
        ) from exc
    expected_key = {
        "model": cell.model,
        "category": cell.category,
        "training_seed": cell.training_seed,
        "shift_name": cell.shift_name,
        "shift_variant": cell.shift_variant,
        "severity": cell.severity,
    }
    if manifest.get("cell_key") != expected_key:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest cell_key mismatch; refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("scope") != "formal":
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest scope mismatch; refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest protocol_version mismatch; refusing "
            "to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("status") != "completed":
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest status {manifest.get('status')!r} "
            "is not 'completed'; refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("failed_samples") != 0:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest records failed samples; refusing "
            "to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest has no samples; refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    model_dir = _formal_model_dir(config, cell, repo_root)
    expected_ids, _ = _formal_expected_sample_ids(model_dir, cell.category)
    actual_ids = sorted(str(record["source_sample_id"]) for record in samples)
    if actual_ids != sorted(expected_ids):
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest sample set does not match the Phase 4 "
            f"recorded test set ({len(actual_ids)} vs {len(expected_ids)}); "
            "refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("completed_samples") != len(samples):
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest completed_samples "
            f"{manifest.get('completed_samples')} != sample count {len(samples)}; "
            "refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    fingerprints = [per_sample_fingerprint(record) for record in samples]
    recomputed = compute_suite_fingerprint(fingerprints)
    if recomputed != manifest.get("shift_suite_fingerprint"):
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} suite fingerprint mismatch; refusing to treat it as completed.",
            code="FINGERPRINT_MISMATCH",
        )
    marker_content = marker_path.read_text(encoding="utf-8").strip()
    if marker_content != recomputed:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} completion marker does not match its suite "
            "fingerprint; refusing to treat it as completed.",
            code="MARKER_MISMATCH",
        )


def _validate_completed_cell(
    output_dir: Path,
    cell: CellSpec,
    config: dict[str, Any],
) -> None:
    """Validate an existing completion marker and its cell manifest.

    A cell is treated as completed only when its manifest exists, parses,
    matches the expected scope/protocol/cell key, and its suite fingerprint
    recomputes from the per-sample fingerprints and equals both the manifest
    and the marker content (Stage B instruction: partial or forged outputs
    must never be treated as completed cells).

    Args:
        output_dir: Benchmark output directory.
        cell: The cell being resumed.
        config: Loaded benchmark configuration.

    Raises:
        ShiftBenchmarkError: If any completion integrity check fails.
    """

    cell_dir = output_dir / "cells" / cell.cell_id
    marker_path = cell_dir / COMPLETED_MARKER
    if not marker_path.is_file():
        return
    manifest_path = cell_dir / "cell_manifest.json"
    if not manifest_path.is_file():
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} has a completion marker but no cell_manifest.json; "
            "refusing to treat it as completed.",
            code="MANIFEST_MISSING",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest is corrupt; refusing to treat it as completed.",
            code="MANIFEST_CORRUPT",
        ) from exc
    expected_key = {
        "category": cell.category,
        "shift_name": cell.shift_name,
        "severity": cell.severity,
    }
    if manifest.get("cell_key") != expected_key:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest cell_key mismatch; refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("scope") != FIXTURE_SCOPE:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest scope mismatch; refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest protocol_version mismatch; refusing "
            "to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("status") != "completed":
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest status {manifest.get('status')!r} "
            "is not 'completed'; refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("failed_samples") != 0:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest records failed samples; refusing "
            "to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest has no samples; refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    # Rebuild the expected fixture sample set from the current configuration:
    # every configured image size contributes its deterministic samples, and
    # the manifest must match exactly (partial outputs are never completed).
    base_seed = int(config["fixture"]["base_seed"])
    expected_samples: list[tuple[str, tuple[int, int]]] = []
    for size in config["fixture"]["image_sizes"]:
        size_tuple = (int(size[0]), int(size[1]))
        for sample in fixture_samples(cell.category, size_tuple, base_seed):
            expected_samples.append(
                (sample.sample_id, (sample.image.shape[0], sample.image.shape[1]))
            )
    actual_samples = [
        (
            str(record["source_sample_id"]),
            (int(record["input_shape"][0]), int(record["input_shape"][1])),
        )
        for record in samples
    ]
    if sorted(actual_samples) != sorted(expected_samples):
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest sample set does not match the "
            f"configured fixture samples ({len(actual_samples)} vs expected "
            f"{len(expected_samples)}); refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    if manifest.get("completed_samples") != len(samples):
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} manifest completed_samples "
            f"{manifest.get('completed_samples')} != sample count {len(samples)}; "
            "refusing to treat it as completed.",
            code="MANIFEST_MISMATCH",
        )
    fingerprints = [per_sample_fingerprint(record) for record in samples]
    recomputed = compute_suite_fingerprint(fingerprints)
    if recomputed != manifest.get("shift_suite_fingerprint"):
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} suite fingerprint mismatch; refusing to treat it as completed.",
            code="FINGERPRINT_MISMATCH",
        )
    marker_content = marker_path.read_text(encoding="utf-8").strip()
    if marker_content != recomputed:
        raise ShiftBenchmarkError(
            f"Cell {cell.cell_id!r} completion marker does not match its suite "
            "fingerprint; refusing to treat it as completed.",
            code="MARKER_MISMATCH",
        )


def _run_formal_benchmark(
    config: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path,
    dry_run: bool = False,
    resume: bool = False,
    stage_c_authorization: Path | None = None,
) -> dict[str, Any]:
    """Run the formal Phase 5 benchmark workflow (Stage D).

    Executes the frozen 2,385-cell matrix over the accepted Phase 4 models
    with source-resolution shifting, frozen thresholds, atomic cell
    manifests, and interruption-safe resume. The commit-bound Stage C
    authorization artifact is verified before any write; model/dataset
    provenance failures abort the run (fail-closed), while per-cell
    execution failures are recorded in ``failures.json`` and re-attempted
    on resume.

    Raises:
        ShiftBenchmarkError: For authorization, isolation, provenance, or
            configuration failures.
    """

    authorization = _authorize_formal(
        config,
        stage_c_authorization,
        repo_root=repo_root,
    )
    resolved_output = _formal_output_dir(config, output_dir, repo_root=repo_root)
    cells = plan_formal_cells(config)
    summary: dict[str, Any] = {
        "scope": "formal",
        "planned_cells": len(cells),
        "dry_run": dry_run,
        "cells": [cell.cell_id for cell in cells],
    }
    if dry_run:
        return summary
    _atomic_write(resolved_output / FORMAL_SCOPE_MARKER, "formal\n")
    _atomic_write(
        resolved_output / "authorization.json",
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
    )
    git_head = _git_head(repo_root, strict=True)
    dataset_root = _resolve_env_path(
        str(config["formal"]["dataset_root"]), section="formal.dataset_root"
    )
    if not dataset_root.is_dir():
        raise ShiftBenchmarkError(
            f"MVTec AD dataset root {dataset_root} does not exist.",
            code="DATASET_MISSING",
        )
    device = resolve_device("auto")
    preregistration_assets = _preregistration_asset_map(repo_root)
    dataset_cache: dict[str, dict[str, Any]] = {}
    model_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for cell in cells:
        cell_dir = resolved_output / "cells" / cell.cell_id
        if (cell_dir / COMPLETED_MARKER).is_file():
            _validate_completed_formal_cell(resolved_output, cell, config, repo_root)
            if resume:
                _clear_failure(resolved_output, cell)
                skipped.append(cell.cell_id)
                continue
            raise ShiftBenchmarkError(
                f"Refusing to overwrite completed cell {cell.cell_id!r}; "
                "pass --resume to skip completed cells.",
                code="OVERWRITE_REFUSED",
            )
        try:
            group_key = (cell.model, cell.category, cell.training_seed)
            if group_key not in model_cache:
                model_dir = _formal_model_dir(config, cell, repo_root)
                if not model_dir.is_dir():
                    raise ShiftBenchmarkError(
                        f"Model directory missing: {model_dir}.",
                        code="MODEL_ASSET_MISSING",
                    )
                model_info = _load_formal_model(
                    model_dir,
                    cell,
                    config,
                    repo_root,
                    device,
                    expected_assets=preregistration_assets.get(group_key),
                )
                model_cache[group_key] = model_info
            else:
                model_info = model_cache[group_key]
            if cell.category not in dataset_cache:
                dataset = _load_formal_test_samples(dataset_root, cell.category)
                dataset_cache[cell.category] = dataset
            else:
                dataset = dataset_cache[cell.category]
            # Re-verify the loaded sample set against the current model group's
            # Phase 4 dataset manifest even on cache hits: a discrepancy between
            # model manifests would otherwise go undetected.
            actual_ids = sorted(str(sample["sample_id"]) for sample in dataset["samples"])
            if actual_ids != sorted(model_info["expected_sample_ids"]):
                raise ShiftBenchmarkError(
                    f"Dataset sample set for {cell.category!r} does not match the "
                    "Phase 4 recorded test set; refusing to continue.",
                    code="MANIFEST_MISMATCH",
                )
            result = _execute_formal_cell(
                resolved_output,
                cell,
                dataset,
                model_info,
                config,
                git_head,
            )
            if result["status"] == "completed_exists":
                skipped.append(cell.cell_id)
            else:
                completed.append(cell.cell_id)
            _clear_failure(resolved_output, cell)
        except ShiftBenchmarkError:
            raise
        except Exception as exc:
            _record_failure(resolved_output, cell, exc)
            failed.append(cell.cell_id)
    summary.update(
        {
            "completed_cells": len(completed),
            "skipped_cells": len(skipped),
            "failed_cells": len(failed),
            "completed": sorted(completed),
            "skipped": sorted(skipped),
            "failed": sorted(failed),
            "output_dir": str(resolved_output),
        }
    )
    return summary


def run_shift_benchmark(
    config_path: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    stage_c_authorization: Path | None = None,
) -> dict[str, Any]:
    """Run the shift benchmark workflow for the configured scope.

    Args:
        config_path: Benchmark YAML configuration path.
        output_dir: Destination directory for cell manifests and markers.
        repo_root: Repository root for Git provenance (defaults to cwd).
        dry_run: Validate and print the plan without writing anything.
        resume: Skip cells that already have a completion marker.
        stage_c_authorization: Commit-bound Stage C authorization artifact
            path; required for formal scope.

    Returns:
        A summary mapping with planned/completed/skipped/failed counts.

    Raises:
        ShiftBenchmarkError: For invalid configuration, refused overwrites,
            or unauthorized formal scope.
    """

    config = load_benchmark_config(config_path)
    if config["scope"] == "formal":
        return _run_formal_benchmark(
            config,
            output_dir,
            repo_root=repo_root or Path.cwd(),
            dry_run=dry_run,
            resume=resume,
            stage_c_authorization=stage_c_authorization,
        )
    _authorize_formal(config, stage_c_authorization, repo_root=repo_root or Path.cwd())
    cells = plan_cells(config)
    summary: dict[str, Any] = {
        "scope": config["scope"],
        "planned_cells": len(cells),
        "dry_run": dry_run,
        "cells": [cell.cell_id for cell in cells],
    }
    if dry_run:
        return summary
    if config["scope"] == "fixture":
        _atomic_write(
            output_dir / FIXTURE_SCOPE_MARKER,
            FIXTURE_SCOPE + "\n",
        )
    git_head = _git_head(
        repo_root or Path.cwd(),
        strict=(config["scope"] == "formal"),
    )
    base_seed = int(config["fixture"]["base_seed"])
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for cell in cells:
        cell_dir = output_dir / "cells" / cell.cell_id
        if (cell_dir / COMPLETED_MARKER).is_file():
            _validate_completed_cell(output_dir, cell, config)
            if resume:
                _clear_failure(output_dir, cell)
                skipped.append(cell.cell_id)
                continue
            raise ShiftBenchmarkError(
                f"Refusing to overwrite completed cell {cell.cell_id!r}; "
                "pass --resume to skip completed cells.",
                code="OVERWRITE_REFUSED",
            )
        try:
            samples: list[FixtureSample] = []
            for size in config["fixture"]["image_sizes"]:
                size_tuple = (int(size[0]), int(size[1]))
                samples.extend(fixture_samples(cell.category, size_tuple, base_seed))
            result = _run_cell(
                output_dir,
                cell,
                samples,
                config,
                git_head,
                base_seed,
            )
            if result["status"] == "completed_exists":
                skipped.append(cell.cell_id)
            else:
                completed.append(cell.cell_id)
            _clear_failure(output_dir, cell)
        except Exception as exc:
            _record_failure(output_dir, cell, exc)
            failed.append(cell.cell_id)
    summary.update(
        {
            "completed_cells": len(completed),
            "skipped_cells": len(skipped),
            "failed_cells": len(failed),
            "completed": sorted(completed),
            "skipped": sorted(skipped),
            "failed": sorted(failed),
            "output_dir": str(output_dir),
        }
    )
    return summary
