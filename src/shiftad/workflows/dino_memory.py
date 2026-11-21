"""Atomic DINO memory and PatchCore build, calibration, and evaluation workflows."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Sized
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from shiftad.backbones.base import BackboneError
from shiftad.data.base import AnomalySample, DatasetError
from shiftad.data.calibration_split import (
    CalibrationSplit,
    CalibrationSplitError,
    split_normal_indices,
)
from shiftad.data.fixture import (
    FIXTURE_SCOPE,
    SyntheticFixtureDataset,
    collate_anomaly_samples,
)
from shiftad.data.mvtec_ad import MVTecADDataset
from shiftad.data.validation import validate_mvtec_ad
from shiftad.evaluation.image_metrics import MetricError, compute_image_metrics
from shiftad.evaluation.pixel_metrics import compute_pixel_metrics
from shiftad.evaluation.thresholds import (
    ThresholdCalibration,
    calibrate_normal_quantile,
)
from shiftad.methods.base import AnomalyDetector, DetectorError
from shiftad.methods.dino_memory import DinoMemoryDetector
from shiftad.methods.factory import create_detector, load_detector
from shiftad.methods.patchcore import PatchCoreDetector
from shiftad.utils.config import (
    DinoMemoryConfig,
    EfficientADModelConfig,
    ExperimentConfig,
    PatchCoreConfig,
    save_resolved_config,
)
from shiftad.utils.device import DeviceError, resolve_device
from shiftad.utils.environment import collect_environment, write_environment
from shiftad.utils.seed import seed_everything
from shiftad.visualization.heatmaps import save_anomaly_visualization

#: Threshold source label for the held-out normal calibration protocol.
HELD_OUT_THRESHOLD_SOURCE = "normal_train_holdout_quantile"

#: Threshold source label for the legacy self-scoring protocol (disabled by default).
LEGACY_THRESHOLD_SOURCE = "in_memory_self_score_quantile_legacy"


@dataclass
class WorkflowError(RuntimeError):
    """Describe a machine-readable CLI workflow failure.

    Attributes:
        code: Machine-readable error code.
        message: Human-readable error description.
        remediation: Suggested action to resolve the failure.

    Note:
        This dataclass is intentionally not frozen.  Python's exception
        machinery assigns ``__traceback__`` to raised exceptions, which
        would trigger ``FrozenInstanceError`` on a frozen dataclass.
    """

    code: str
    message: str
    remediation: str

    def to_dict(self) -> dict[str, str]:
        """Return a stable structured error payload."""

        return {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
        }


def _require_model(config: ExperimentConfig) -> None:
    if config.model is None:
        raise WorkflowError(
            "MODEL_CONFIG_MISSING",
            "Experiment configuration does not contain a model section.",
            "Use a Phase 3 DINO memory or PatchCore experiment configuration.",
        )


def _require_memory_model(
    config: ExperimentConfig,
) -> DinoMemoryConfig | PatchCoreConfig:
    """Return the model config narrowed to memory-based methods.

    Args:
        config: Experiment configuration.

    Returns:
        The model config typed as DinoMemoryConfig or PatchCoreConfig.

    Raises:
        WorkflowError: If the model is missing or is not a memory-based method.
    """

    _require_model(config)
    assert config.model is not None
    if isinstance(config.model, EfficientADModelConfig):
        raise WorkflowError(
            "MODEL_TYPE_MISMATCH",
            "This workflow supports memory-based methods only (dino_memory, patchcore).",
            "Use 'shiftad train' for EfficientAD.",
        )
    return config.model


def _narrow_detector(detector: AnomalyDetector) -> DinoMemoryDetector | PatchCoreDetector:
    """Narrow the abstract detector to a concrete type for attribute access.

    Args:
        detector: A detector returned by the factory.

    Returns:
        The same detector typed as a concrete subclass.

    Raises:
        WorkflowError: If the detector type is unsupported (should not happen).
    """

    if isinstance(detector, (DinoMemoryDetector, PatchCoreDetector)):
        return detector
    raise WorkflowError(
        "UNSUPPORTED_DETECTOR",
        f"Unsupported detector type {type(detector).__name__}.",
        "Use a DinoMemoryDetector or PatchCoreDetector configuration.",
    )


def _artifact_scope(
    config: ExperimentConfig,
    detector: DinoMemoryDetector | PatchCoreDetector,
    phase: Literal["build", "evaluation"],
) -> str:
    """Determine the artifact scope label from backbone provenance.

    Args:
        config: Experiment configuration with a validated model section.
        detector: Fitted or loaded concrete detector.
        phase: ``"build"`` or ``"evaluation"`` for the scope suffix.

    Returns:
        A scope string indicating real or synthetic provenance.
    """

    assert config.model is not None
    model = _require_memory_model(config)
    if model.backbone.name == "deterministic_fixture":
        return FIXTURE_SCOPE
    if detector.backbone.is_real_dinov2:
        return f"real_dinov2_{phase}"
    return f"real_patchcore_{phase}"


def _dataset(
    config: ExperimentConfig,
    split: Literal["train", "test"],
) -> Dataset[AnomalySample]:
    if config.dataset.name == "synthetic_fixture":
        return SyntheticFixtureDataset(
            split,
            config.dataset.image_size,
            config.dataset.categories[0],
        )
    if config.dataset.name == "mvtec_ad":
        report = validate_mvtec_ad(config.dataset.root, config.dataset.categories)
        if not report.valid:
            issue_codes = sorted({issue.code for issue in report.issues})
            raise WorkflowError(
                "DATASET_INVALID",
                f"MVTec AD validation failed with issue codes {issue_codes}.",
                "Correct dataset.root and the reported layout issues, then rerun validation.",
            )
        return MVTecADDataset(
            config.dataset.root,
            config.dataset.categories,
            split,
            config.dataset.image_size,
        )
    raise WorkflowError(
        "DATASET_UNSUPPORTED",
        f"DINO memory workflow does not support dataset {config.dataset.name!r}.",
        "Use synthetic_fixture for offline acceptance or mvtec_ad for real data.",
    )


def _loader(
    dataset: Dataset[AnomalySample],
    *,
    batch_size: int,
) -> DataLoader[AnomalySample]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_anomaly_samples,
    )


def _score_normal_subset(
    detector: DinoMemoryDetector | PatchCoreDetector,
    dataset: Dataset[AnomalySample],
    *,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    """Score normal images and return per-image and max-pixel scores.

    Args:
        detector: A fitted or loaded detector.
        dataset: Normal-only dataset subset to score.
        batch_size: Evaluation batch size.

    Returns:
        A tuple of (image_scores, max_pixel_scores) as lists of floats.

    Raises:
        WorkflowError: If scoring fails or produces non-finite values.
    """

    loader = _loader(dataset, batch_size=batch_size)
    image_scores: list[float] = []
    max_pixel_scores: list[float] = []
    for batch in loader:
        images = batch["image"]
        if not isinstance(images, Tensor):
            raise WorkflowError(
                "CALIBRATION_SCORING_FAILED",
                "Calibration batch image field is not a tensor.",
                "Verify the dataset adapter and collate function.",
            )
        prediction = detector.predict(images)
        for index in range(images.shape[0]):
            score = float(prediction.image_scores[index])
            pixel_max = float(prediction.anomaly_maps[index].max())
            if not np.isfinite(score) or not np.isfinite(pixel_max):
                raise WorkflowError(
                    "CALIBRATION_SCORING_FAILED",
                    f"Calibration produced non-finite score {score} or pixel max {pixel_max}.",
                    "Check backbone features, memory bank, and nearest-neighbour search.",
                )
            image_scores.append(score)
            max_pixel_scores.append(pixel_max)
    if not image_scores:
        raise WorkflowError(
            "CALIBRATION_EMPTY",
            "Calibration subset produced no scores.",
            "Increase the training set size or reduce calibration.ratio.",
        )
    return image_scores, max_pixel_scores


def _calibrate_from_held_out(
    image_scores: list[float],
    pixel_scores: list[float],
    *,
    quantile: float,
) -> tuple[ThresholdCalibration, ThresholdCalibration]:
    """Calibrate image and pixel thresholds from held-out normal scores.

    Args:
        image_scores: Per-image anomaly scores from the calibration subset.
        pixel_scores: Max per-image pixel anomaly scores from the calibration subset.
        quantile: Quantile in (0, 1) for threshold selection.

    Returns:
        A tuple of (image_calibration, pixel_calibration).

    Raises:
        WorkflowError: If calibration fails due to invalid scores.
    """

    try:
        image_calibration = calibrate_normal_quantile(image_scores, quantile=quantile)
        pixel_calibration = calibrate_normal_quantile(pixel_scores, quantile=quantile)
    except MetricError as exc:
        raise WorkflowError(
            "CALIBRATION_FAILED",
            f"Threshold calibration failed: {exc}",
            "Verify that the calibration subset contains finite normal scores.",
        ) from exc
    return image_calibration, pixel_calibration


def _apply_calibrated_thresholds(
    config: ExperimentConfig,
    *,
    image_threshold: float,
    pixel_threshold: float,
) -> ExperimentConfig:
    """Return a new ExperimentConfig with calibrated scoring thresholds.

    Args:
        config: Original experiment configuration.
        image_threshold: Calibrated image-level threshold.
        pixel_threshold: Calibrated pixel-level threshold.

    Returns:
        A new ExperimentConfig with updated scoring thresholds and source.
    """

    assert config.model is not None
    model = _require_memory_model(config)
    new_scoring = replace(
        model.scoring,
        image_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
        threshold_source=HELD_OUT_THRESHOLD_SOURCE,
    )
    new_model = replace(model, scoring=new_scoring)
    return replace(config, model=new_model)


def _save_calibration_manifest(
    output_dir: Path,
    *,
    split: CalibrationSplit,
    image_calibration: ThresholdCalibration,
    pixel_calibration: ThresholdCalibration,
    quantile: float,
) -> dict[str, Any]:
    """Write the calibration and split manifests to the model directory.

    Args:
        output_dir: Model directory to write into.
        split: The calibration split used.
        image_calibration: Calibrated image threshold record.
        pixel_calibration: Calibrated pixel threshold record.
        quantile: Quantile used for calibration.

    Returns:
        A summary dictionary for the build summary.
    """

    split.save(output_dir / "calibration_split.json")
    manifest = {
        "threshold_source": HELD_OUT_THRESHOLD_SOURCE,
        "quantile": quantile,
        "image_threshold": image_calibration.threshold,
        "pixel_threshold": pixel_calibration.threshold,
        "image_calibration": image_calibration.to_dict(),
        "pixel_calibration": pixel_calibration.to_dict(),
        "split": split.to_dict(),
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _temporary_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise WorkflowError(
            "OUTPUT_EXISTS",
            f"Refusing to overwrite existing output directory {destination}.",
            "Select a new runtime.output_dir or remove the previous generated output explicitly.",
        )
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-partial-",
            dir=destination.parent,
        )
    )


def run_build_memory(
    config: ExperimentConfig,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically persist a configured memory model with calibration.

    When ``config.calibration.enabled`` is True (default), normal training
    images are split into a memory subset and a calibration subset.  Memory
    is built from the memory subset only, and thresholds are calibrated
    from the held-out calibration subset scores.  When disabled, all normal
    training images are used for memory and predeclared thresholds are kept.

    Args:
        config: Validated experiment configuration with a model section.
        repo_root: Optional repository root for environment capture.

    Returns:
        A summary dictionary with build metadata and model path.

    Raises:
        WorkflowError: If the build, split, or calibration fails.
    """

    _require_model(config)
    assert config.model is not None
    seed_everything(config.runtime.seed, deterministic_algorithms=True)
    model_dir = config.runtime.output_dir / "model"
    temporary = _temporary_directory(model_dir)
    try:
        device = resolve_device(config.runtime.device)
        train_dataset = _dataset(config, "train")
        train_size = len(cast(Sized, train_dataset))
        calibration_config = config.calibration
        calibration_summary: dict[str, Any] | None = None
        if calibration_config.enabled:
            try:
                split = split_normal_indices(
                    train_size,
                    calibration_ratio=calibration_config.ratio,
                    seed=calibration_config.seed,
                )
            except CalibrationSplitError as exc:
                raise WorkflowError(
                    "CALIBRATION_SPLIT_FAILED",
                    str(exc),
                    "Adjust calibration.ratio or provide more normal training data.",
                ) from exc
            memory_dataset: Dataset[AnomalySample] = Subset(
                train_dataset,
                list(split.memory_indices),
            )
            calibration_dataset: Dataset[AnomalySample] | None = Subset(
                train_dataset,
                list(split.calibration_indices),
            )
        else:
            memory_dataset = train_dataset
            split = None
            calibration_dataset = None

        detector = _narrow_detector(create_detector(config.model, device=device))
        detector.fit(_loader(memory_dataset, batch_size=config.runtime.batch_size))

        image_cal: ThresholdCalibration | None = None
        pixel_cal: ThresholdCalibration | None = None
        if calibration_config.enabled and split is not None and calibration_dataset is not None:
            image_scores, pixel_scores = _score_normal_subset(
                detector,
                calibration_dataset,
                batch_size=config.runtime.batch_size,
            )
            image_cal, pixel_cal = _calibrate_from_held_out(
                image_scores,
                pixel_scores,
                quantile=calibration_config.quantile,
            )
            config = _apply_calibrated_thresholds(
                config,
                image_threshold=image_cal.threshold,
                pixel_threshold=pixel_cal.threshold,
            )
            assert config.model is not None
            detector.config = _require_memory_model(config)

        detector.save(temporary)
        save_resolved_config(config, temporary / "experiment.resolved.yaml")
        if (
            calibration_config.enabled
            and split is not None
            and calibration_dataset is not None
            and image_cal is not None
            and pixel_cal is not None
        ):
            calibration_summary = _save_calibration_manifest(
                temporary,
                split=split,
                image_calibration=image_cal,
                pixel_calibration=pixel_cal,
                quantile=calibration_config.quantile,
            )
        write_environment(
            collect_environment(
                repo_root=repo_root,
                command=["shiftad", "build-memory"],
            ),
            temporary / "environment.json",
        )
        scope = _artifact_scope(config, detector, "build")
        assert config.model is not None
        summary: dict[str, Any] = {
            "artifact_scope": scope,
            "model_type": config.model.name,
            "dataset": config.dataset.name,
            "category_count": len(config.dataset.categories),
            "build": detector.build_metadata,
            "calibration_enabled": calibration_config.enabled,
        }
        if calibration_summary is not None:
            summary["calibration"] = calibration_summary
        (temporary / "build_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(model_dir)
        return {**summary, "model_dir": model_dir.as_posix()}
    except WorkflowError:
        raise
    except (
        BackboneError,
        DatasetError,
        DetectorError,
        DeviceError,
        CalibrationSplitError,
        OSError,
    ) as exc:
        raise WorkflowError(
            "BUILD_MEMORY_FAILED",
            str(exc),
            "Verify the dataset, backbone checkpoint/download policy, device, and output path.",
        ) from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _semantic_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_evaluation(
    config: ExperimentConfig,
    *,
    model_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a saved memory model and atomically persist traceable artifacts.

    When the experiment configuration does not include preconfigured
    thresholds, thresholds are read from the saved model's calibrated
    scoring configuration.  This supports the held-out normal calibration
    workflow where thresholds are calibrated during ``build-memory``.

    Args:
        config: Experiment configuration.  May omit thresholds if the model
            was built with calibration enabled.
        model_dir: Directory produced by ``build-memory``.
        repo_root: Optional repository root for environment capture.

    Returns:
        A summary dictionary with evaluation metrics and artifact path.

    Raises:
        WorkflowError: If the model cannot be loaded, thresholds are
            unavailable, or evaluation fails.
    """

    _require_model(config)
    assert config.model is not None
    seed_everything(config.runtime.seed, deterministic_algorithms=True)
    evaluation_dir = config.runtime.output_dir / "evaluation"
    temporary = _temporary_directory(evaluation_dir)
    try:
        detector = _narrow_detector(
            load_detector(
                model_dir,
                device=resolve_device(config.runtime.device),
            )
        )

        def _strip_thresholds(model_cfg: Any) -> Any:
            """Return a model config copy with threshold fields nulled."""
            return replace(
                model_cfg,
                scoring=replace(
                    model_cfg.scoring,
                    image_threshold=None,
                    pixel_threshold=None,
                    threshold_source=None,
                ),
            )

        if (
            _strip_thresholds(config.model).to_dict()
            != _strip_thresholds(detector.config).to_dict()
        ):
            raise WorkflowError(
                "MODEL_CONFIG_MISMATCH",
                "Evaluation model configuration differs from the saved model "
                "(excluding calibrated thresholds).",
                "Use the experiment configuration that built this model, "
                "or omit thresholds to inherit calibrated values.",
            )

        assert config.model is not None
        model = _require_memory_model(config)
        config_has_thresholds = (
            model.scoring.image_threshold is not None
            and model.scoring.pixel_threshold is not None
            and model.scoring.threshold_source is not None
        )
        if not config_has_thresholds:
            config = replace(
                config,
                model=replace(
                    model,
                    scoring=detector.config.scoring,
                ),
            )
        assert config.model is not None
        narrowed = _require_memory_model(config)
        image_threshold = narrowed.scoring.image_threshold
        pixel_threshold = narrowed.scoring.pixel_threshold
        threshold_source = narrowed.scoring.threshold_source
        if image_threshold is None or pixel_threshold is None or threshold_source is None:
            raise WorkflowError(
                "THRESHOLD_PROVENANCE_MISSING",
                "No calibrated thresholds found in the model or configuration.",
                "Run build-memory with calibration.enabled: true, "
                "or provide scoring thresholds in the experiment configuration.",
            )
        dataset = _dataset(config, "test")
        loader = _loader(dataset, batch_size=config.runtime.batch_size)
        labels: list[int] = []
        scores: list[float] = []
        masks: list[Tensor] = []
        anomaly_maps: list[Tensor] = []
        rows: list[dict[str, Any]] = []
        heatmap_dir = temporary / "heatmaps"
        for batch in loader:
            images = batch["image"]
            batch_labels = batch["label"]
            sample_ids = batch["sample_id"]
            batch_masks = batch["mask"]
            if (
                not isinstance(images, Tensor)
                or not isinstance(batch_labels, Tensor)
                or not isinstance(sample_ids, list)
                or not isinstance(batch_masks, list)
            ):
                raise DetectorError("Evaluation loader returned invalid batch fields.")
            prediction = detector.predict(images)
            for index, sample_id in enumerate(sample_ids):
                label = int(batch_labels[index])
                labels.append(label)
                score = float(prediction.image_scores[index])
                scores.append(score)
                sample_mask = batch_masks[index]
                if sample_mask is None:
                    sample_mask = torch.zeros(
                        (1, images.shape[-2], images.shape[-1]),
                        dtype=torch.float32,
                    )
                if not isinstance(sample_mask, Tensor):
                    raise DetectorError("Evaluation mask must be a tensor or None.")
                masks.append(sample_mask)
                anomaly_map = prediction.anomaly_maps[index]
                anomaly_maps.append(anomaly_map)
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "label": label,
                        "image_score": score,
                        "inference_time_ms": prediction.inference_times_ms[index],
                    }
                )
                save_anomaly_visualization(
                    images[index],
                    anomaly_map,
                    heatmap_dir / f"{str(sample_id).replace('/', '__')}.png",
                )
        image_metrics = compute_image_metrics(
            labels,
            scores,
            threshold=image_threshold,
        )
        mask_tensor = torch.stack(masks)
        map_tensor = torch.stack(anomaly_maps)
        pixel_metrics = compute_pixel_metrics(
            mask_tensor,
            map_tensor,
            threshold=pixel_threshold,
        )
        scope = _artifact_scope(config, detector, "evaluation")
        metric_sections = {
            "image": image_metrics.to_dict(),
            "pixel": pixel_metrics.to_dict(),
        }
        metrics: dict[str, Any] = {
            "artifact_scope": scope,
            "threshold_source": threshold_source,
            **metric_sections,
        }
        if not all(
            np.isfinite(value)
            for section in ("image", "pixel")
            for value in metric_sections[section].values()
        ):
            raise DetectorError("Evaluation metrics contain NaN or Infinity.")
        save_resolved_config(config, temporary / "config.resolved.yaml")
        (temporary / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (temporary / "metrics.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(["scope", "metric", "value"])
            for section in ("image", "pixel"):
                for name, value in sorted(metric_sections[section].items()):
                    csv_writer.writerow([section, name, value])
        with (temporary / "per_sample_predictions.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            prediction_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            prediction_writer.writeheader()
            prediction_writer.writerows(rows)
        (temporary / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": config.dataset.name,
                    "categories": list(config.dataset.categories),
                    "sample_ids": [row["sample_id"] for row in rows],
                    "artifact_scope": scope,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "runtime.json").write_text(
            json.dumps(
                {
                    "inference_times_ms": [row["inference_time_ms"] for row in rows],
                    "device": str(detector.device),
                    "batch_size": config.runtime.batch_size,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        write_environment(
            collect_environment(
                repo_root=repo_root,
                command=["shiftad", "evaluate"],
            ),
            temporary / "environment.json",
        )
        semantic_payload = {
            "metrics": metrics,
            "predictions": [
                {
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "image_score": row["image_score"],
                }
                for row in rows
            ],
        }
        summary = {
            "artifact_scope": scope,
            "semantic_sha256": _semantic_hash(semantic_payload),
            "sample_count": len(rows),
            "heatmap_count": len(rows),
        }
        (temporary / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(evaluation_dir)
        return {**summary, "evaluation_dir": evaluation_dir.as_posix()}
    except WorkflowError:
        raise
    except (
        BackboneError,
        DatasetError,
        DetectorError,
        DeviceError,
        OSError,
        ValueError,
    ) as exc:
        raise WorkflowError(
            "EVALUATION_FAILED",
            str(exc),
            "Verify model integrity, dataset layout, thresholds, device, and output path.",
        ) from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
