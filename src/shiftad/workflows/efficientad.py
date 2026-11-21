"""Atomic EfficientAD training, calibration, and evaluation workflows.

This module provides the training and evaluation workflows for the
EfficientAD detector, following the same calibration and artifact
conventions as the DINO memory and PatchCore workflows but adapted for
the training-based EfficientAD architecture.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Sized
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

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
from shiftad.methods.base import DetectorError
from shiftad.methods.efficientad import EfficientADDetector
from shiftad.methods.factory import create_detector, load_detector
from shiftad.utils.config import (
    EfficientADModelConfig,
    ExperimentConfig,
    save_resolved_config,
)
from shiftad.utils.device import DeviceError, resolve_device
from shiftad.utils.environment import collect_environment, write_environment
from shiftad.utils.seed import seed_everything
from shiftad.visualization.heatmaps import save_anomaly_visualization

#: Threshold source label for the held-out normal calibration protocol.
HELD_OUT_THRESHOLD_SOURCE = "normal_train_holdout_quantile"


class WorkflowError(RuntimeError):
    """Describe a machine-readable CLI workflow failure.

    Attributes:
        code: Machine-readable error code.
        message: Human-readable error description.
        remediation: Suggested action to resolve the failure.
    """

    def __init__(self, code: str, message: str, remediation: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation

    def to_dict(self) -> dict[str, str]:
        """Return a stable structured error payload."""

        return {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
        }


def _require_model(config: ExperimentConfig) -> EfficientADModelConfig:
    """Return the model config or raise a clear error."""

    if config.model is None:
        raise WorkflowError(
            "MODEL_CONFIG_MISSING",
            "Experiment configuration does not contain a model section.",
            "Use an EfficientAD experiment configuration.",
        )
    if not isinstance(config.model, EfficientADModelConfig):
        raise WorkflowError(
            "MODEL_TYPE_MISMATCH",
            f"Expected EfficientADModelConfig, got {type(config.model).__name__}.",
            "Use a configuration with model.name: efficientad.",
        )
    return config.model


def _dataset(
    config: ExperimentConfig,
    split: Literal["train", "test"],
) -> Dataset[AnomalySample]:
    """Create the configured dataset for training or evaluation."""

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
        f"EfficientAD workflow does not support dataset {config.dataset.name!r}.",
        "Use synthetic_fixture for offline acceptance or mvtec_ad for real data.",
    )


def _loader(
    dataset: Dataset[AnomalySample],
    *,
    batch_size: int,
) -> DataLoader[AnomalySample]:
    """Create a standard anomaly-sample data loader."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_anomaly_samples,
    )


def _score_normal_subset(
    detector: EfficientADDetector,
    dataset: Dataset[AnomalySample],
    *,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    """Score normal images and return per-image and max-pixel scores.

    Args:
        detector: A fitted detector.
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
                    "Check teacher weights, student training, and autoencoder training.",
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
    """Calibrate image and pixel thresholds from held-out normal scores."""

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
    model_config: EfficientADModelConfig,
    *,
    image_threshold: float,
    pixel_threshold: float,
) -> ExperimentConfig:
    """Return a new ExperimentConfig with calibrated scoring thresholds."""

    new_scoring = replace(
        model_config.scoring,
        image_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
        threshold_source=HELD_OUT_THRESHOLD_SOURCE,
    )
    new_model = replace(model_config, scoring=new_scoring)
    return replace(config, model=new_model)


def _save_calibration_manifest(
    output_dir: Path,
    *,
    split: CalibrationSplit,
    image_calibration: ThresholdCalibration,
    pixel_calibration: ThresholdCalibration,
    quantile: float,
) -> dict[str, Any]:
    """Write the calibration and split manifests to the model directory."""

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
    """Create a temporary directory for atomic output writing."""

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


def _artifact_scope(config: ExperimentConfig, detector: EfficientADDetector) -> str:
    """Determine the artifact scope label."""

    model_config = _require_model(config)
    if model_config.is_fixture:
        return FIXTURE_SCOPE
    return "real_efficientad"


def _update_detector_config(
    detector: EfficientADDetector,
    config: ExperimentConfig,
) -> None:
    """Update the detector's runtime config with calibrated thresholds."""

    model_config = _require_model(config)
    scoring = model_config.scoring
    detector.config.image_threshold = scoring.image_threshold
    detector.config.pixel_threshold = scoring.pixel_threshold
    detector.config.threshold_source = scoring.threshold_source


def run_train(
    config: ExperimentConfig,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Train EfficientAD, calibrate thresholds, and atomically persist.

    This workflow:
    1. Splits normal training images into train and calibration subsets.
    2. Trains the student and autoencoder on the train subset.
    3. Scores the calibration subset to calibrate thresholds.
    4. Saves the model with calibration metadata.

    Args:
        config: Validated experiment configuration with an EfficientAD model.
        repo_root: Optional repository root for environment capture.

    Returns:
        A summary dictionary with training metadata and model path.

    Raises:
        WorkflowError: If training, split, or calibration fails.
    """

    model_config = _require_model(config)
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
            train_dataset_subset: Dataset[AnomalySample] = Subset(
                train_dataset,
                list(split.memory_indices),
            )
            calibration_dataset: Dataset[AnomalySample] | None = Subset(
                train_dataset,
                list(split.calibration_indices),
            )
        else:
            train_dataset_subset = train_dataset
            split = None
            calibration_dataset = None

        detector = cast(
            EfficientADDetector,
            create_detector(model_config, device=device),
        )

        # Prepare checkpoint directory and compatibility info.
        checkpoint_cfg = model_config.checkpoint
        ckpt_dir: Path | None = None
        if checkpoint_cfg.enabled:
            ckpt_dir = (
                checkpoint_cfg.dir
                if checkpoint_cfg.dir is not None
                else config.runtime.output_dir / "checkpoints"
            )
        compatibility_info: dict[str, Any] = {
            "category": ",".join(config.dataset.categories),
            "seed": config.runtime.seed,
            "split_fingerprint": (
                hashlib.sha256(
                    json.dumps(split.to_dict(), sort_keys=True).encode("utf-8")
                ).hexdigest()
                if split is not None
                else None
            ),
        }

        detector.fit(
            _loader(train_dataset_subset, batch_size=config.runtime.batch_size),
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=checkpoint_cfg.interval,
            resume_from=checkpoint_cfg.resume_from,
            auto_resume=checkpoint_cfg.auto_resume,
            max_to_keep=checkpoint_cfg.max_to_keep,
            compatibility_info=compatibility_info,
        )

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
                model_config,
                image_threshold=image_cal.threshold,
                pixel_threshold=pixel_cal.threshold,
            )
            _update_detector_config(detector, config)

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
                command=["shiftad", "train"],
            ),
            temporary / "environment.json",
        )
        scope = _artifact_scope(config, detector)
        summary: dict[str, Any] = {
            "artifact_scope": scope,
            "model_type": "efficientad",
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
        DatasetError,
        DetectorError,
        DeviceError,
        CalibrationSplitError,
        OSError,
        RuntimeError,
    ) as exc:
        raise WorkflowError(
            "TRAIN_FAILED",
            str(exc),
            "Verify the dataset, teacher weights, device, and output path.",
        ) from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _semantic_hash(payload: dict[str, Any]) -> str:
    """Return a SHA-256 hash of the semantic payload."""

    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_evaluation(
    config: ExperimentConfig,
    *,
    model_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a saved EfficientAD model and atomically persist artifacts.

    Args:
        config: Experiment configuration. May omit thresholds if the model
            was built with calibration enabled.
        model_dir: Directory produced by ``run_train``.
        repo_root: Optional repository root for environment capture.

    Returns:
        A summary dictionary with evaluation metrics and artifact path.

    Raises:
        WorkflowError: If the model cannot be loaded, thresholds are
            unavailable, or evaluation fails.
    """

    model_config = _require_model(config)
    seed_everything(config.runtime.seed, deterministic_algorithms=True)
    evaluation_dir = config.runtime.output_dir / "evaluation"
    temporary = _temporary_directory(evaluation_dir)
    try:
        detector = cast(
            EfficientADDetector,
            load_detector(
                model_dir,
                device=resolve_device(config.runtime.device),
            ),
        )

        config_has_thresholds = (
            model_config.scoring.image_threshold is not None
            and model_config.scoring.pixel_threshold is not None
            and model_config.scoring.threshold_source is not None
        )
        if not config_has_thresholds:
            # Inherit calibrated thresholds from the saved model.
            model_config = replace(
                model_config,
                scoring=replace(
                    model_config.scoring,
                    image_threshold=detector.config.image_threshold,
                    pixel_threshold=detector.config.pixel_threshold,
                    threshold_source=detector.config.threshold_source,
                ),
            )
            config = replace(config, model=model_config)

        image_threshold = model_config.scoring.image_threshold
        pixel_threshold = model_config.scoring.pixel_threshold
        threshold_source = model_config.scoring.threshold_source
        if image_threshold is None or pixel_threshold is None or threshold_source is None:
            raise WorkflowError(
                "THRESHOLD_PROVENANCE_MISSING",
                "No calibrated thresholds found in the model or configuration.",
                "Run train with calibration.enabled: true, "
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
        scope = _artifact_scope(config, detector)
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
