"""PatchCore anomaly detector with frozen-backbone patch memory.

PatchCore builds a normal patch-feature memory from intermediate layers of a
frozen pretrained backbone (typically a torchvision Wide ResNet), compresses it
with coreset subsampling, and scores test patches by nearest-neighbour distance
to the memory.  This implementation reuses the same memory bank, coreset,
nearest-neighbour, and scoring infrastructure as
:class:`~shiftad.methods.dino_memory.DinoMemoryDetector`, differing only in
the default backbone and configuration dispatch.

Reference:
    Roth, K., Berga, L., Zell, P., Bhattacharjee, B., Oswald, M. R.,
    "Towards Total Recall in Industrial Anomaly Detection," CVPR 2022.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from shiftad import __version__
from shiftad.backbones.base import FeatureBackbone
from shiftad.backbones.factory import create_backbone
from shiftad.backbones.feature_extractor import FeatureFuser
from shiftad.memory.coreset import select_coreset
from shiftad.memory.memory_bank import MemoryBank, MemoryBankError
from shiftad.memory.nearest_neighbors import (
    NearestNeighborIndex,
    create_nearest_neighbor,
)
from shiftad.methods.base import AnomalyDetector, DetectorError, PredictionBatch
from shiftad.methods.scoring import aggregate_image_scores, patch_scores_to_anomaly_maps
from shiftad.utils.config import PatchCoreConfig
from shiftad.utils.device import resolve_device

ARTIFACT_VERSION = 1
REQUIRED_ARTIFACTS = (
    "config.resolved.yaml",
    "memory_bank.pt",
    "coreset_indices.pt",
    "metadata.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


class PatchCoreDetector(AnomalyDetector):
    """Build and query a frozen-backbone normal patch memory (PatchCore).

    Args:
        config: Validated PatchCore model configuration.
        device: CPU or available accelerator request.
        backbone: Optional injected provider for isolated tests.
    """

    def __init__(
        self,
        config: PatchCoreConfig,
        *,
        device: torch.device | str = "cpu",
        backbone: FeatureBackbone | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(str(device))
        self.backbone = backbone or create_backbone(config.backbone, device=self.device)
        self.fuser = FeatureFuser(
            mode=config.feature_fusion.mode,
            normalize=config.feature_fusion.normalize,
            target_grid=config.feature_fusion.target_grid,
        )
        neighbor_config = config.memory.nearest_neighbors
        self.neighbor_index: NearestNeighborIndex = create_nearest_neighbor(
            backend=neighbor_config.backend,
            metric=neighbor_config.metric,
            k=neighbor_config.k,
            chunk_size=neighbor_config.chunk_size,
            device=self.device,
        )
        self.memory_bank: MemoryBank | None = None
        self.coreset_indices: Tensor | None = None
        self.build_metadata: dict[str, Any] = {}

    def _fused(self, images: Tensor) -> tuple[Tensor, tuple[int, int], dict[str, object]]:
        feature_batch = self.backbone.extract_features(images)
        feature_batch.validate()
        fused = self.fuser(
            layer_features=feature_batch.layer_features,
            patch_grid_shapes=feature_batch.patch_grid_shapes,
        )
        return (
            fused.features,
            fused.grid_shape,
            {
                **feature_batch.metadata,
                **fused.metadata,
            },
        )

    @staticmethod
    def _batch_field(batch: object, name: str) -> object:
        if not isinstance(batch, Mapping) or name not in batch:
            raise DetectorError(f"Training batch must be a mapping containing {name!r}.")
        return batch[name]

    @torch.inference_mode()
    def fit(self, train_loader: object) -> None:
        """Construct immutable normal memory from normal-only batches.

        Raises:
            DetectorError: If labels indicate leakage or batches are invalid.
        """

        if not isinstance(train_loader, Iterable):
            raise DetectorError("fit() expected an iterable train loader.")
        bank = MemoryBank(
            metadata={"model": "patchcore"},
            normalized=self.config.feature_fusion.normalize,
        )
        grid_shape: tuple[int, int] | None = None
        backbone_metadata: dict[str, object] = {}
        batches = 0
        for batch in train_loader:
            images = self._batch_field(batch, "image")
            labels = self._batch_field(batch, "label")
            if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
                raise DetectorError("Training image and label fields must be tensors.")
            if labels.numel() != images.shape[0]:
                raise DetectorError("Training labels do not match the image batch size.")
            if torch.any(labels != 0):
                raise DetectorError(
                    "Training leakage risk: PatchCore fit accepts normal labels only."
                )
            fused, current_grid, metadata = self._fused(images)
            if grid_shape is not None and current_grid != grid_shape:
                raise DetectorError(
                    f"Training patch grid changed from {grid_shape} to {current_grid}."
                )
            grid_shape = current_grid
            backbone_metadata = metadata
            bank.add(fused.reshape(-1, fused.shape[-1]))
            batches += 1
        if batches == 0 or grid_shape is None:
            raise DetectorError("Cannot fit PatchCore memory from an empty train loader.")
        coreset_config = self.config.memory.coreset
        try:
            result = select_coreset(
                bank.accumulated_features(),
                method=coreset_config.method,
                ratio=coreset_config.ratio,
                target_size=coreset_config.target_size,
                seed=0,
                chunk_size=self.config.memory.nearest_neighbors.chunk_size,
            )
            bank.finalize(result.features)
        except (ValueError, MemoryBankError) as exc:
            raise DetectorError(f"Memory construction failed: {exc}") from exc
        self.memory_bank = bank
        self.coreset_indices = result.indices
        self.neighbor_index.fit(bank.to(self.device))
        self.build_metadata = {
            "train_batches": batches,
            "original_feature_count": result.original_size,
            "memory_size": bank.size,
            "feature_dimension": bank.feature_dimension,
            "patch_grid": list(grid_shape),
            "memory_bytes": bank.memory_bytes,
            "normalized": bank.normalized,
            "backbone": backbone_metadata,
            "actual_neighbor_backend": self.neighbor_index.backend_name,
            "real_dinov2_weights": self.backbone.is_real_dinov2,
        }

    @torch.inference_mode()
    def predict(self, images: Tensor) -> PredictionBatch:
        """Score images against immutable normal patch memory."""

        if self.memory_bank is None or not self.memory_bank.finalized:
            raise DetectorError("PatchCore memory must be fitted or loaded before predict().")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        fused, grid_shape, metadata = self._fused(images)
        batch_size, patch_count, dimension = fused.shape
        if dimension != self.memory_bank.feature_dimension:
            raise DetectorError(
                f"Prediction feature dimension {dimension} does not match memory "
                f"{self.memory_bank.feature_dimension}."
            )
        search = self.neighbor_index.search(fused.reshape(-1, dimension))
        patch_scores = search.distances.mean(dim=1).reshape(batch_size, patch_count)
        anomaly_maps = patch_scores_to_anomaly_maps(
            patch_scores,
            grid_shape=grid_shape,
            output_size=(images.shape[-2], images.shape[-1]),
            smoothing_sigma=self.config.scoring.smoothing_sigma,
        )
        image_scores = aggregate_image_scores(
            patch_scores,
            mode=self.config.scoring.image_aggregation,
            top_k=self.config.scoring.top_k,
        )
        threshold = self.config.scoring.image_threshold
        predicted_labels = (
            None if threshold is None else (image_scores >= threshold).to(torch.int64)
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return PredictionBatch(
            image_scores=image_scores.cpu(),
            anomaly_maps=anomaly_maps.cpu(),
            predicted_labels=None if predicted_labels is None else predicted_labels.cpu(),
            threshold=threshold,
            inference_times_ms=[elapsed_ms / batch_size] * batch_size,
            metadata=[
                {
                    **metadata,
                    "nearest_neighbor_backend": search.backend,
                    "distance_metric": self.config.memory.nearest_neighbors.metric,
                    "artifact_scope": (
                        "real_patchcore_inference"
                        if not self.backbone.is_real_dinov2
                        and self.config.backbone.name != "deterministic_fixture"
                        else "synthetic_fixture_acceptance_only"
                    ),
                }
                for _ in range(batch_size)
            ],
        )

    def save(self, output_dir: Path) -> None:
        """Persist a complete model with hashes and safe tensor-only payloads."""

        if self.memory_bank is None or self.coreset_indices is None:
            raise DetectorError("Cannot save PatchCore memory before fit().")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise DetectorError(f"Refusing to overwrite non-empty model directory {output_dir}.")
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.resolved.yaml"
        config_path.write_text(
            yaml.safe_dump(self.config.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        self.memory_bank.save(output_dir / "memory_bank.pt")
        torch.save({"indices": self.coreset_indices.cpu()}, output_dir / "coreset_indices.pt")
        (output_dir / "metadata.json").write_text(
            json.dumps(self.build_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "project_version": __version__,
            "model_type": "patchcore",
            "backbone_name": self.config.backbone.name,
            "feature_layers": list(self.config.backbone.layers),
            "feature_dimension": self.memory_bank.feature_dimension,
            "patch_grid": self.build_metadata["patch_grid"],
            "memory_size": self.memory_bank.original_size,
            "coreset_size": self.memory_bank.size,
            "distance_metric": self.config.memory.nearest_neighbors.metric,
            "score_aggregation": self.config.scoring.image_aggregation,
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "git_commit": _git_commit(),
            "real_dinov2_weights": self.backbone.is_real_dinov2,
            "artifacts": {name: _sha256(output_dir / name) for name in REQUIRED_ARTIFACTS},
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        model_dir: Path,
        *,
        device: torch.device | str = "cpu",
        backbone: FeatureBackbone | None = None,
    ) -> PatchCoreDetector:
        """Load and verify a model directory using weights-only tensor reads."""

        manifest_path = model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise DetectorError(f"Model manifest is missing: {manifest_path}.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DetectorError(f"Cannot read model manifest {manifest_path}: {exc}") from exc
        if (
            manifest.get("artifact_version") != ARTIFACT_VERSION
            or manifest.get("model_type") != "patchcore"
        ):
            raise DetectorError(f"Incompatible model manifest version or type in {manifest_path}.")
        hashes = manifest.get("artifacts")
        if not isinstance(hashes, dict):
            raise DetectorError("Model manifest lacks artifact hashes.")
        for name in REQUIRED_ARTIFACTS:
            path = model_dir / name
            if not path.is_file() or hashes.get(name) != _sha256(path):
                raise DetectorError(f"Model artifact is missing or hash-mismatched: {path}.")
        try:
            raw_config = yaml.safe_load(
                (model_dir / "config.resolved.yaml").read_text(encoding="utf-8")
            )
            if not isinstance(raw_config, Mapping):
                raise DetectorError("Resolved model configuration is not a mapping.")
            config = PatchCoreConfig.from_mapping(raw_config)
            metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise DetectorError("Model metadata is not a mapping.")
            bank = MemoryBank.load(model_dir / "memory_bank.pt", metadata=metadata)
            coreset_payload = torch.load(
                model_dir / "coreset_indices.pt",
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, ValueError, RuntimeError, MemoryBankError) as exc:
            raise DetectorError(f"Cannot safely load model artifact {model_dir}: {exc}") from exc
        if not isinstance(coreset_payload, dict) or not isinstance(
            coreset_payload.get("indices"), Tensor
        ):
            raise DetectorError("Coreset artifact lacks tensor indices.")
        indices = coreset_payload["indices"]
        if indices.ndim != 1 or len(indices) != bank.size:
            raise DetectorError("Coreset indices are inconsistent with the memory size.")
        if (
            manifest.get("feature_dimension") != bank.feature_dimension
            or manifest.get("coreset_size") != bank.size
            or manifest.get("backbone_name") != config.backbone.name
        ):
            raise DetectorError("Manifest dimensions or configuration are inconsistent.")
        detector = cls(config, device=device, backbone=backbone)
        detector.memory_bank = bank
        detector.coreset_indices = indices
        detector.build_metadata = metadata
        detector.neighbor_index.fit(bank.to(detector.device))
        return detector
