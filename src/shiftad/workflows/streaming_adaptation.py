"""Phase 6-H formal causal streaming adaptation workflow.

Implements the frozen Phase 6-G protocol for the formal evaluation matrix:
5 methods (A0-A4) x 5 categories x 3 training seeds x 11 core stream types
= 825 method-streams. Each stream is a deterministic, label-blind, causal
sequence of all MVTec AD test samples of one (category, training_seed,
stream_type) triple; labels/masks are joined by the evaluator only after
each decision/update transaction.

The workflow is production scope (PRODUCTION_SCOPE components only:
V1ReliabilityGate, V1ShiftEstimator, V1TargetMemory, V1SafetyMonitor,
ConservativeScoreFusion, CausalStreamEngine). Fixture components are never
used here.

Execution model (mirrors Phase 5 Stage D): one cell per method-stream,
atomic ``cell_manifest.json`` + ``COMPLETED`` marker, interruption-safe
``--resume`` that validates completed cells and re-executes only
incomplete/failed cells, and path-redacted failure records.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment]

from shiftad.adaptation import (
    PROTOCOL_VERSION,
    CausalStreamEngine,
    ConservativeScoreFusion,
    SourceAnchor,
    V1ReliabilityGate,
    V1SafetyMonitor,
    V1ShiftEstimator,
    V1TargetMemory,
)
from shiftad.adaptation.audit import AuditLog
from shiftad.adaptation.base import REASON_SCORE_LIMIT, ShiftEstimate
from shiftad.adaptation.image_stats import compute_image_stats
from shiftad.adaptation.shift_estimator import SourceNormalStatistics
from shiftad.adaptation.state import TargetState
from shiftad.adaptation.v2.contracts import FrozenSourceObservation
from shiftad.adaptation.v2.engine_adapter import (
    SapcaEngineAdapter,
    SapcaTransaction,
)
from shiftad.data.mvtec_ad import MVTecADDataset
from shiftad.methods.dino_memory import DinoMemoryDetector
from shiftad.methods.factory import load_detector
from shiftad.shifts.base import BASE_SEED, Shift, derive_seed
from shiftad.shifts.registry import get_shift
from shiftad.workflows.dino_memory import _narrow_detector

COMPLETED_MARKER = "COMPLETED"
SCOPE_MARKER = "formal"
FORMAL_BATCH_SIZE = 1  # causal streaming: one sample at a time

METHODS = ("A0", "A1", "A2", "A3", "A4")
CATEGORIES = ("bottle", "cable", "hazelnut", "metal_nut", "transistor")
TRAINING_SEEDS = (0, 1, 2)
CORE_STREAM_TYPES: tuple[dict[str, Any], ...] = (
    {"name": "clean", "shift_name": "clean", "severity": 0},
    {"name": "brightness_s2", "shift_name": "brightness", "severity": 2},
    {"name": "brightness_s4", "shift_name": "brightness", "severity": 4},
    {"name": "gaussian_noise_s2", "shift_name": "gaussian_noise", "severity": 2},
    {"name": "gaussian_noise_s4", "shift_name": "gaussian_noise", "severity": 4},
    {"name": "motion_blur_s2", "shift_name": "motion_blur", "severity": 2},
    {"name": "motion_blur_s4", "shift_name": "motion_blur", "severity": 4},
    {"name": "resolution_reduction_s2", "shift_name": "resolution_reduction", "severity": 2},
    {"name": "resolution_reduction_s4", "shift_name": "resolution_reduction", "severity": 4},
    {"name": "rotation_s2", "shift_name": "rotation", "severity": 2},
    {"name": "rotation_s4", "shift_name": "rotation", "severity": 4},
)

WARMUP_SAMPLES = 20  # frozen Gate G

# Frozen Gate F policy (configs/phase6/gate_f_policy.yaml, primary).
GATE_PARAMS = {
    "score_limit": 0.3,
    "consistency_tolerance": 0.1,
    "suspicious_area_ratio_max": 0.05,
    "update_rate_max": 2,
    "rate_window_steps": 16,
    "cooldown_steps": 2,
}
MONITOR_PARAMS = {
    "max_target_source_distance": 2.0,
    "max_acceptance_rate": 1.0,
    "min_score_variance_ratio": 0.1,
    "max_rollback_count": 3,
}
FUSION_PARAMS = {"w_s_min": 0.5, "w_t_max": 0.35}
ESTIMATOR_PARAMS = {
    "lambda_img": 1.0,
    "lambda_mu": 1.0,
    "lambda_sigma": 0.5,
    "window_length": 16,
}

SHIFT_SEED_PROTOCOL = "phase5_v1.1"


class StreamingWorkflowError(RuntimeError):
    """Report a formal streaming-workflow failure."""


def _require_mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StreamingWorkflowError(f"Configuration section {section!r} must be a mapping.")
    return value


def _atomic_write(path: Path, content: str) -> None:
    """Atomically write text via mkstemp + os.replace (no partial files)."""

    import tempfile

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise


def load_formal_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen formal configuration."""

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise StreamingWorkflowError("Formal config must be a mapping.")
    if parsed.get("protocol_version") != PROTOCOL_VERSION:
        raise StreamingWorkflowError(
            f"Formal config protocol_version {parsed.get('protocol_version')!r} "
            f"!= {PROTOCOL_VERSION!r}."
        )
    if parsed.get("status") != "FROZEN_STAGE_G_PENDING_STAGE_H":
        raise StreamingWorkflowError(
            f"Formal config status must be FROZEN_STAGE_G_PENDING_STAGE_H, "
            f"got {parsed.get('status')!r}."
        )
    if parsed.get("formal_authorized") is not False:
        raise StreamingWorkflowError("Formal config must keep formal_authorized: false.")
    formal = _require_mapping(parsed.get("formal"), "formal")
    if formal.get("execution_disabled") is not True:
        raise StreamingWorkflowError("Formal config must set execution_disabled: true.")
    return parsed


#: The Stage G authorization artifact must contain exactly one line:
#: ``STAGE_G_FORMAL_AUTHORIZED <40-hex-commit>``. The commit must be an
#: ancestor of the current HEAD (same binding semantics as the Gate G
#: readiness auditor), and the artifact bytes are SHA-256 hashed so the
#: results auditor can re-verify the binding (PHASE6_FORMAL_INSTRUCTIONS).
_AUTHORIZATION_PATTERN = re.compile(r"^STAGE_G_FORMAL_AUTHORIZED ([0-9a-f]{40})\s*$")


def verify_authorization(
    config: dict[str, Any],
    authorization: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Verify the commit-bound Stage G authorization artifact for formal scope.

    The artifact must contain exactly ``STAGE_G_FORMAL_AUTHORIZED
    <40-hex-commit>`` and the commit must be an ancestor of the current
    HEAD; the artifact bytes are SHA-256 hashed into the returned record.

    Args:
        config: Loaded formal configuration.
        authorization: Authorization artifact path.
        repo_root: Repository root for the Git binding check.

    Returns:
        An authorization record for the output manifest.

    Raises:
        StreamingWorkflowError: If the artifact is missing, malformed, or
            not bound to the frozen commit chain.
    """

    path = Path(authorization)
    if not path.is_file():
        raise StreamingWorkflowError(f"Authorization artifact not found: {authorization}.")
    content = path.read_text(encoding="utf-8")
    match = _AUTHORIZATION_PATTERN.fullmatch(content)
    if match is None:
        raise StreamingWorkflowError(
            f"Authorization artifact {authorization} must contain exactly one line "
            "'STAGE_G_FORMAL_AUTHORIZED <40-hex-commit>'."
        )
    commit = match.group(1)
    repo_root = repo_root or Path.cwd()
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
        raise StreamingWorkflowError(
            "Cannot verify the authorization commit against the repository HEAD."
        ) from exc
    if result.returncode != 0:
        raise StreamingWorkflowError(
            f"Authorization commit {commit} is not an ancestor of the current "
            "HEAD; the frozen binding is broken."
        )
    return {
        "token": "STAGE_G_FORMAL_AUTHORIZED",
        "commit": commit,
        "artifact_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


@dataclass(frozen=True)
class StreamCellSpec:
    """One formal method-stream cell (A0-A4 x 165 streams = 825 cells)."""

    method: str
    category: str
    training_seed: int
    stream_type: str
    shift_name: str
    severity: int

    @property
    def cell_id(self) -> str:
        return f"{self.method}__{self.category}__seed{self.training_seed}__{self.stream_type}"


def plan_cells() -> list[StreamCellSpec]:
    """Construct the frozen 825-cell plan deterministically."""

    cells: list[StreamCellSpec] = []
    for method in METHODS:
        for category in CATEGORIES:
            for seed in TRAINING_SEEDS:
                for stream_type in CORE_STREAM_TYPES:
                    cells.append(
                        StreamCellSpec(
                            method=method,
                            category=category,
                            training_seed=seed,
                            stream_type=stream_type["name"],
                            shift_name=stream_type["shift_name"],
                            severity=stream_type["severity"],
                        )
                    )
    return cells


def stream_order_seed(category: str, training_seed: int, stream_type: str) -> int:
    """Derive the deterministic stream order seed (protocol Section 7.2)."""

    seed_input = (
        "phase6_v1.0"
        + ":"
        + category
        + ":"
        + str(training_seed)
        + ":"
        + stream_type
        + ":"
        + str(BASE_SEED)
    )
    full_digest = hashlib.sha256(seed_input.encode("utf-8")).digest()
    return int.from_bytes(full_digest[0:8], byteorder="big") % 2**32


def load_stream_samples(
    dataset_root: Path, category: str, stream_type: str, severity: int
) -> list[dict[str, Any]]:
    """Load and deterministically order the MVTec test samples of one stream.

    The stream applies the frozen shift transformation on the fly; sample
    order is the deterministic permutation seeded by the protocol Section
    7.2 order seed (identical across methods A0-A4 and across reruns).
    """

    dataset = MVTecADDataset(dataset_root, (category,), "test")
    samples: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        samples.append(
            {
                "sample_id": str(item["sample_id"]),
                "defect_type": str(item["defect_type"]),
                "label": int(item["label"]),
                "image": item["image"].permute(1, 2, 0).numpy().astype(np.float32),
                "mask": (
                    item["mask"][0].numpy().astype(np.float32) if item["mask"] is not None else None
                ),
            }
        )
    rng = np.random.default_rng(stream_order_seed(category, 0, stream_type))
    order = rng.permutation(len(samples))
    ordered = [samples[int(i)] for i in order]

    if severity == 0:
        return ordered
    shift: Shift = get_shift(
        name=stream_type,
        severity=severity,
    )
    shifted: list[dict[str, Any]] = []
    for sample in ordered:
        sample_seed = derive_seed(
            protocol_version=SHIFT_SEED_PROTOCOL,
            category=category,
            sample_id=sample["sample_id"],
            shift_name=stream_type,
            severity=severity,
            base_seed=BASE_SEED,
        )
        result = shift.apply(sample["image"], sample["mask"], derived_seed=sample_seed)
        sample_image = result.image
        sample_mask = result.mask
        shifted.append(
            {
                **sample,
                "image": np.asarray(sample_image, dtype=np.float32),
                "mask": sample_mask,
            }
        )
    return shifted


def build_source_anchor(
    fit_report: dict[str, Any],
    category: str,
    *,
    backbone_hash: str,
    source_memory_hash: str,
) -> SourceAnchor:
    """Construct the immutable source anchor from the frozen Gate C fit."""

    stats = fit_report["source_statistics"][category]
    sigma = stats["sigma_s"]
    if sigma and isinstance(sigma[0], (int, float)):
        sigma_rows: tuple[tuple[float, ...], ...] = (tuple(float(v) for v in sigma),)
    else:
        sigma_rows = tuple(tuple(float(v) for v in row) for row in sigma)
    return SourceAnchor(
        category=category,
        backbone_hash=backbone_hash,
        source_memory_hash=source_memory_hash,
        mu_s=tuple(float(v) for v in stats["mu_s"]),
        sigma_s=sigma_rows,
        protocol_version=PROTOCOL_VERSION,
    )


def _image_tensor(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Resize a float32 [H, W, 3] array to the model input size and tensor.

    Mirrors the Phase 4 preprocessing pipeline (PIL BILINEAR).
    """

    from PIL import Image

    array = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    resized = Image.fromarray(array, mode="RGB").resize(
        (size[1], size[0]),
        resample=Image.Resampling.BILINEAR,
    )
    return (
        torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _downsample_map(anomaly_map: np.ndarray, target: int) -> np.ndarray:
    """Downsample a 2D float32 anomaly map with PIL BILINEAR (deterministic)."""

    from PIL import Image

    array = (np.clip(anomaly_map, 0.0, None) * 255.0).round().astype(np.uint8)
    resized = Image.fromarray(array, mode="L").resize(
        (target, target),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _feature_row(fused: torch.Tensor) -> tuple[float, ...]:
    """Mean-pool a patch feature map into one image-level feature row."""

    flat = fused.reshape(fused.shape[0], -1, fused.shape[-1]).mean(dim=1)
    return tuple(float(v) for v in flat[0])


class MethodPipeline:
    """One A0-A4 method pipeline over one deterministic stream.

    The pipeline is label-blind: it receives only the current sample image
    (and its own prior state). Truth joining happens in the evaluator after
    each committed transaction.
    """

    def __init__(
        self,
        method: str,
        detector: DinoMemoryDetector,
        anchor: SourceAnchor,
        *,
        device: torch.device,
        sapca_engine: SapcaEngineAdapter | None = None,
    ) -> None:
        if method not in METHODS:
            raise StreamingWorkflowError(f"Unknown method {method!r}.")
        self.method = method
        self.detector = detector
        self.anchor = anchor
        self.device = device
        # SAPCA is an additive, disabled-by-default boundary. Production and
        # formal construction supply no engine, so historical A0-A4 paths are
        # byte-for-byte unchanged and cannot produce a V2 action.
        self.sapca_engine = sapca_engine
        self.target_memory = V1TargetMemory()
        self.fusion = ConservativeScoreFusion(**FUSION_PARAMS)
        self.gate = (
            V1ReliabilityGate(
                score_limit=float(GATE_PARAMS["score_limit"]),
                consistency_tolerance=float(GATE_PARAMS["consistency_tolerance"]),
                suspicious_area_ratio_max=float(GATE_PARAMS["suspicious_area_ratio_max"]),
                update_rate_max=int(GATE_PARAMS["update_rate_max"]),
                rate_window_steps=int(GATE_PARAMS["rate_window_steps"]),
                cooldown_steps=int(GATE_PARAMS["cooldown_steps"]),
            )
            if method in ("A2", "A3", "A4")
            else None
        )
        source_stats = SourceNormalStatistics(
            image_mean=tuple(float(v) for v in anchor.mu_s[:9]),
            image_std=(1.0,) * 9,
            mu_s=anchor.mu_s,
            sigma_s=anchor.sigma_s[0] if anchor.sigma_s else (),
            n_samples=1,
        )
        self.estimator = (
            V1ShiftEstimator(
                source_stats,
                lambda_img=float(ESTIMATOR_PARAMS["lambda_img"]),
                lambda_mu=float(ESTIMATOR_PARAMS["lambda_mu"]),
                lambda_sigma=float(ESTIMATOR_PARAMS["lambda_sigma"]),
                window_length=int(ESTIMATOR_PARAMS["window_length"]),
            )
            if method == "A4"
            else None
        )
        self.safety_monitor = (
            V1SafetyMonitor(
                max_target_source_distance=float(MONITOR_PARAMS["max_target_source_distance"]),
                max_acceptance_rate=float(MONITOR_PARAMS["max_acceptance_rate"]),
                min_score_variance_ratio=float(MONITOR_PARAMS["min_score_variance_ratio"]),
                max_rollback_count=int(MONITOR_PARAMS["max_rollback_count"]),
            )
            if method == "A4"
            else None
        )
        self.state = TargetState(category=anchor.category)
        self.engine = CausalStreamEngine(
            anchor=anchor,
            state=self.state,
            audit_log=AuditLog(path=Path(tempfile.mkdtemp()) / "audit.jsonl"),
            enabled=method != "A0",
            warmup_samples=WARMUP_SAMPLES,
            safety_monitor=self.safety_monitor,
        )
        self._window_image_stats: list[dict[str, float]] = []
        self._window_feature_rows: list[tuple[float, ...]] = []
        self._window_length = int(ESTIMATOR_PARAMS["window_length"])

    def consistency_score(self, image: np.ndarray) -> float | None:
        """Weak-view consistency distance (A3/A4 only), label-blind."""

        if self.method not in ("A3", "A4"):
            return None
        weak = cv2.GaussianBlur(image, (5, 5), 1.0).astype(np.float32)
        with torch.inference_mode():
            fused_strong, _, _ = self.detector._fused(_image_tensor(image))
            fused_weak, _, _ = self.detector._fused(_image_tensor(weak))
        row_s = _feature_row(fused_strong)
        row_w = _feature_row(fused_weak)
        return float(sum((a - b) ** 2 for a, b in zip(row_s, row_w, strict=False)) ** 0.5)

    def process_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process one stream sample; returns the evaluator-joinable record.

        The returned record carries no labels/masks (truth is joined by the
        evaluator afterward); it records the gate decision, fused score,
        shift components, state versions, and audit record.
        """

        image = sample["image"]
        with torch.inference_mode():
            prediction = self.detector.predict(_image_tensor(image))
            fused, _, _ = self.detector._fused(_image_tensor(image))
        source_score = float(prediction.image_scores[0])
        # Pixel-level anomaly map (downsampled to 112x112 to bound storage);
        # used only by the evaluator-side pixel aggregation (pixel AUROC /
        # AUPRO), never by the adaptation path.
        anomaly_map = np.asarray(prediction.anomaly_maps[0].cpu().numpy(), dtype=np.float32)
        if anomaly_map.shape[0] > 112:
            anomaly_map = _downsample_map(anomaly_map, 112)
        feature_row = _feature_row(fused)
        target_score = self.target_memory.score(feature_row, self.state)
        consistency = self.consistency_score(image)
        consistency_ok = (
            consistency <= GATE_PARAMS["consistency_tolerance"] if consistency is not None else True
        )
        image_stats = compute_image_stats(image)

        shift_estimate: ShiftEstimate | None = None
        if self.estimator is not None:
            self._window_image_stats.append(dict(image_stats))
            self._window_feature_rows.append(feature_row)
            if len(self._window_image_stats) > self._window_length:
                self._window_image_stats.pop(0)
                self._window_feature_rows.pop(0)
            from shiftad.adaptation.shift_estimator import WindowStatistics

            window = WindowStatistics(
                image_stats={
                    name: float(np.mean([stats[name] for stats in self._window_image_stats]))
                    for name in sorted(self._window_image_stats[0])
                },
                mu_t=tuple(
                    float(np.mean([row[i] for row in self._window_feature_rows]))
                    for i in range(len(feature_row))
                ),
                sigma_t=None,
                occupancy=len(self._window_image_stats) / self._window_length,
            )
            shift_estimate = self.estimator.estimate(image_stats, window)

        if self.gate is None:
            # A0: adaptation off (engine disabled) -> decision ACCEPTED but
            # state frozen; A1: ungated target prototype -> always accepted.
            decision = self.gate_decision(source_score, shift_estimate, consistency_ok)
        else:
            decision = self.gate.evaluate(
                source_score,
                shift_estimate,
                suspicious_area_ratio=0.0,
                consistency_ok=consistency_ok,
                input_quality_ok=True,
                identity_ok=True,
            )
        transaction = self.engine.process(
            sample["sample_id"],
            feature_row,
            decision,
            shift_estimate,
        )
        # NOTE: the engine already mutates the target state for accepted
        # decisions (add_candidate + promote after warmup); V1TargetMemory is
        # used only for scoring (prototype distance), never re-applied here,
        # so each accepted sample contributes exactly once to the prototype.

        # Method output score: A0 evaluates with the pure source score
        # (adaptation off); A1-A4 evaluate with the fused score so the
        # adaptation components (target prototype, consistency) contribute
        # to the reported score.
        fused_score: float = source_score
        weights: dict[str, float] = {"w_s": 1.0, "w_t": 0.0, "w_c": 0.0}
        if self.method != "A0":
            fused_out = self.fusion.fuse(
                source_score,
                target_score,
                consistency,
                self.state,
                shift_magnitude=(None if shift_estimate is None else shift_estimate.magnitude),
            )
            fused_score = float(fused_out.fused_score)
            weights = dict(fused_out.weights)

        record: dict[str, Any] = {
            "sequence_id": transaction.sequence_id,
            "sample_id": sample["sample_id"],
            "source_score": source_score,
            "fused_score": fused_score,
            "fusion_weights": weights,
            "target_score": target_score,
            "consistency_score": consistency,
            "shift_magnitude": None if shift_estimate is None else shift_estimate.magnitude,
            "shift_in_envelope": None if shift_estimate is None else shift_estimate.in_envelope,
            "decision_accepted": decision.accepted,
            "decision_reason": decision.reason_code,
            "state_version_before": transaction.state_version_before,
            "state_version_after": transaction.state_version_after,
            "audit_record_hash": transaction.audit.record_hash(),
            "anomaly_map": anomaly_map.tolist(),
            "method": self.method,
        }
        return record

    def process_sapca_observation(
        self,
        observation: FrozenSourceObservation,
        image_stats: dict[str, float],
    ) -> SapcaTransaction:
        """Process an already-computed source observation through V2.

        This additive method is unreachable from the production/formal CLI in
        this implementation cycle. A missing engine fails closed before any
        evaluator-visible record can be produced.
        """
        if self.sapca_engine is None:
            raise StreamingWorkflowError(
                "SAPCA is disabled; no authorized engine/policy package is configured."
            )
        return self.sapca_engine.process(observation, image_stats)

    def gate_decision(
        self,
        source_score: float,
        shift_estimate: ShiftEstimate | None,
        consistency_ok: bool,
    ) -> Any:
        from shiftad.adaptation import GateDecision

        if self.method == "A1":
            return GateDecision(accepted=True, reason_code="ACCEPTED")
        return GateDecision(
            accepted=source_score <= GATE_PARAMS["score_limit"],
            reason_code=(
                "ACCEPTED" if source_score <= GATE_PARAMS["score_limit"] else REASON_SCORE_LIMIT
            ),
        )


def run_cell(
    cell: StreamCellSpec,
    *,
    dataset_root: Path,
    model_dir: Path,
    fit_report: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    detector: DinoMemoryDetector | None = None,
) -> dict[str, Any]:
    """Execute one formal method-stream cell atomically.

    Returns the cell summary (evaluator-truth metrics included).
    """

    if detector is None:
        narrowed = _narrow_detector(load_detector(model_dir / "model", device=device))
        if not isinstance(narrowed, DinoMemoryDetector):
            raise StreamingWorkflowError(
                f"Model {model_dir} is not a DinoMemoryDetector; Phase 6-H requires "
                "the accepted dino_memory baseline."
            )
        detector = narrowed
    anchor = build_source_anchor(
        fit_report,
        cell.category,
        backbone_hash=str((model_dir / "model" / "manifest.json").read_text(encoding="utf-8")),
        source_memory_hash=str((model_dir / "model" / "memory_bank.pt").read_bytes()),
    )
    pipeline = MethodPipeline(cell.method, detector, anchor, device=device)
    samples = load_stream_samples(dataset_root, cell.category, cell.shift_name, cell.severity)
    records: list[dict[str, Any]] = []
    labels: list[int] = []
    pixel_maps: list[Any] = []
    pixel_masks: list[Any] = []
    for sample in samples:
        record = pipeline.process_one(sample)
        labels.append(sample["label"])
        # Move the pixel anomaly map out of the JSON manifest into the
        # compact per-cell pixel archive (used only by the evaluator-side
        # pixel aggregation).
        pixel_maps.append(record.pop("anomaly_map"))
        pixel_masks.append(sample["mask"])
        records.append(record)
    summary = {
        "cell_id": cell.cell_id,
        "method": cell.method,
        "category": cell.category,
        "training_seed": cell.training_seed,
        "stream_type": cell.stream_type,
        "shift_name": cell.shift_name,
        "severity": cell.severity,
        "n_samples": len(records),
        "n_accepted": sum(1 for r in records if r["decision_accepted"]),
        "n_rejected": sum(1 for r in records if not r["decision_accepted"]),
        "state_versions": [r["state_version_after"] for r in records],
        "rollback_count": pipeline.engine.rollback_count,
        "adaptation_enabled_final": pipeline.engine.enabled,
        "pixel_archive": "pixel_data.npz",
        "records": records,
        "labels": labels,
        "scope": SCOPE_MARKER,
    }
    cell_dir = output_dir / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)

    maps_array = np.asarray(pixel_maps, dtype=np.float32)
    masks_array = np.asarray(
        [
            (_downsample_map(m, 112) if m is not None else np.zeros_like(maps_array[0]))
            for m in pixel_masks
        ],
        dtype=np.float32,
    )
    labels_array = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(
        cell_dir / "pixel_data.npz",
        maps=maps_array,
        masks=masks_array,
        labels=labels_array,
    )
    _atomic_write(
        cell_dir / "cell_manifest.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        cell_dir / COMPLETED_MARKER,
        hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest() + "\n",
    )
    return summary


def _validate_completed_cell(cell_dir: Path) -> bool:
    """Validate a completed cell's manifest and marker (never skip tampered)."""

    marker_path = cell_dir / COMPLETED_MARKER
    manifest_path = cell_dir / "cell_manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = marker_path.read_text(encoding="utf-8").strip()
    except (OSError, json.JSONDecodeError):
        return False
    expected = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if marker != expected:
        return False
    if manifest.get("scope") != SCOPE_MARKER:
        return False
    return not manifest.get("n_samples", 0) < 1


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the formal streaming CLI."""

    parser = argparse.ArgumentParser(
        prog="stream-adaptation",
        description="Phase 6-H formal causal streaming evaluation (A0-A4).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase6/formal.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-base", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--stage-g-authorization", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    config = load_formal_config(args.config)
    authorization_record = verify_authorization(
        config, args.stage_g_authorization, repo_root=Path.cwd()
    )
    fit_report = json.loads(args.fit_report.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reset stale failure records from earlier runs so the final failures.json
    # reflects exactly this run (interruption-safe resume re-attempts failed
    # cells; a new failure rewrites the file).
    _atomic_write(
        output_dir / "failures.json",
        json.dumps({"failures": []}, indent=2, sort_keys=True) + "\n",
    )

    cells = plan_cells()
    completed = 0
    failed: list[str] = []
    detector_cache: dict[str, DinoMemoryDetector] = {}
    for cell in cells:
        cell_dir = output_dir / cell.cell_id
        if args.resume and _validate_completed_cell(cell_dir):
            completed += 1
            continue
        model_dir = args.model_base / f"{cell.category}_dino_memory_seed{cell.training_seed}"
        detector = detector_cache.get(str(model_dir))
        if detector is None:
            narrowed = _narrow_detector(load_detector(model_dir / "model", device=device))
            if not isinstance(narrowed, DinoMemoryDetector):
                raise StreamingWorkflowError(
                    f"Model {model_dir} is not a DinoMemoryDetector; Phase 6-H "
                    "requires the accepted dino_memory baseline."
                )
            detector = narrowed
            detector_cache[str(model_dir)] = detector
        try:
            run_cell(
                cell,
                dataset_root=args.dataset_root,
                model_dir=model_dir,
                fit_report=fit_report,
                output_dir=output_dir,
                device=device,
                detector=detector,
            )
            completed += 1
        except Exception as exc:
            message = str(exc).replace(str(args.dataset_root), "<dataset-root>")
            message = message.replace(str(args.model_base), "<model-base>")
            failed.append(f"{cell.cell_id}: {message}")
            _atomic_write(
                output_dir / "failures.json",
                json.dumps({"failures": failed}, indent=2, sort_keys=True) + "\n",
            )
    _atomic_write(
        output_dir / "aggregation.json",
        json.dumps(
            {
                "completed": completed,
                "failed": failed,
                "total": len(cells),
                "scope": SCOPE_MARKER,
                "authorization": authorization_record,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(f"completed={completed}/{len(cells)} failed={len(failed)}")
    return 0 if not failed else 1
