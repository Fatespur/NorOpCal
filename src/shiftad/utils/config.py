"""Validated configuration loading for ShiftAD experiments."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Report an invalid experiment configuration with remediation context."""


def _resolve_path_value(value: str, field_name: str) -> str:
    """Resolve a path string that may use ``env:`` prefix for portability.

    If ``value`` starts with ``env:``, the remainder is treated as an
    environment variable name whose value provides the path.  This allows
    formal version-controlled configurations to avoid hard-coding local
    absolute paths.

    Args:
        value: Raw path string from YAML (may be ``env:VAR_NAME``).
        field_name: Configuration field name for error messages.

    Returns:
        The resolved path string.

    Raises:
        ConfigurationError: If an ``env:`` prefix is used but the
            environment variable is unset or empty.
    """

    if value.startswith("env:"):
        var_name = value[4:]
        resolved = os.environ.get(var_name, "")
        if not resolved:
            raise ConfigurationError(
                f"Invalid {field_name}: environment variable {var_name!r} is not set "
                f"or is empty. Set {var_name} to the dataset root path, "
                f"e.g. $env:{var_name}='C:\\path\\to\\data'."
            )
        return resolved
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Invalid {section} configuration: unknown fields {unknown}. "
            f"Expected only {sorted(allowed)}."
        )


def _require_mapping(value: object, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"Invalid {section} configuration: received {type(value).__name__}; "
            "expected a YAML mapping."
        )
    return value


@dataclass(frozen=True)
class DatasetConfig:
    """Describe dataset location and preprocessing inputs.

    Args:
        name: Registered dataset name.
        root: Dataset root supplied by the user.
        categories: Non-empty categories to include.
        split: Dataset split requested by the command.
        image_size: Output height and width in pixels.

    Raises:
        ConfigurationError: If a field violates the schema.

    Assumptions:
        The path is validated for existence by dataset-specific commands.
    """

    name: str
    root: Path
    categories: tuple[str, ...]
    split: str
    image_size: tuple[int, int]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DatasetConfig:
        """Build a dataset configuration from parsed YAML values.

        Args:
            raw: Dataset section mapping.

        Returns:
            An immutable validated dataset configuration.

        Raises:
            ConfigurationError: If required values are missing or malformed.
        """

        allowed = {"name", "root", "categories", "split", "image_size"}
        _reject_unknown(raw, allowed, "dataset")
        missing = sorted({"name", "root", "categories"} - set(raw))
        if missing:
            raise ConfigurationError(
                f"Invalid dataset configuration: missing required fields {missing}."
            )

        name = raw["name"]
        root = _resolve_path_value(raw["root"], "dataset.root")
        categories = raw["categories"]
        split = raw.get("split", "test")
        image_size = raw.get("image_size", [256, 256])

        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("Invalid dataset.name: expected a non-empty string.")
        if not isinstance(root, str) or not root.strip():
            raise ConfigurationError("Invalid dataset.root: expected a non-empty path string.")
        if (
            not isinstance(categories, list)
            or not categories
            or not all(isinstance(item, str) and item.strip() for item in categories)
        ):
            raise ConfigurationError(
                "Invalid dataset.categories: expected a non-empty list of category names."
            )
        if split not in {"train", "validation", "test"}:
            raise ConfigurationError(
                f"Invalid dataset.split {split!r}: expected train, validation, or test."
            )
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or not all(isinstance(value, int) and value > 0 for value in image_size)
        ):
            raise ConfigurationError(
                "Invalid dataset.image_size: expected [height, width] with positive integers."
            )
        return cls(
            name=name.strip(),
            root=Path(root),
            categories=tuple(item.strip() for item in categories),
            split=split,
            image_size=(image_size[0], image_size[1]),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Describe reproducible runtime controls and output behavior."""

    seed: int = 0
    device: str = "auto"
    output_dir: Path = Path("outputs/default")
    log_level: str = "INFO"
    batch_size: int = 4

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RuntimeConfig:
        """Build validated runtime settings from parsed YAML values.

        Args:
            raw: Runtime section mapping.

        Returns:
            An immutable runtime configuration.

        Raises:
            ConfigurationError: If a runtime field is invalid or unknown.
        """

        allowed = {"seed", "device", "output_dir", "log_level", "batch_size"}
        _reject_unknown(raw, allowed, "runtime")
        seed = raw.get("seed", 0)
        device = raw.get("device", "auto")
        output_dir = raw.get("output_dir", "outputs/default")
        log_level = raw.get("log_level", "INFO")
        batch_size = raw.get("batch_size", 4)
        if not isinstance(seed, int) or seed < 0:
            raise ConfigurationError("Invalid runtime.seed: expected a non-negative integer.")
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError("Invalid runtime.device: expected a non-empty string.")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ConfigurationError(
                "Invalid runtime.output_dir: expected a non-empty path string."
            )
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError(
                f"Invalid runtime.log_level {log_level!r}: expected a standard uppercase level."
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError("Invalid runtime.batch_size: expected a positive integer.")
        return cls(
            seed=seed,
            device=device.strip(),
            output_dir=Path(output_dir),
            log_level=log_level,
            batch_size=batch_size,
        )


@dataclass(frozen=True)
class BackboneConfig:
    """Configure a frozen feature provider and its weight source."""

    name: str
    layers: tuple[int, ...]
    input_size: tuple[int, int]
    checkpoint: Path | None = None
    allow_download: bool = False
    use_cache: bool = True
    dtype: str = "float32"
    fixture_patch_size: int = 8

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BackboneConfig:
        """Validate backbone selection, feature layers, and loading policy."""

        allowed = {
            "name",
            "layers",
            "input_size",
            "checkpoint",
            "allow_download",
            "use_cache",
            "dtype",
            "fixture_patch_size",
        }
        _reject_unknown(raw, allowed, "model.backbone")
        name = raw.get("name")
        layers = raw.get("layers")
        input_size = raw.get("input_size")
        checkpoint = raw.get("checkpoint")
        allow_download = raw.get("allow_download", False)
        use_cache = raw.get("use_cache", True)
        dtype = raw.get("dtype", "float32")
        fixture_patch_size = raw.get("fixture_patch_size", 8)
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("Invalid model.backbone.name: expected a non-empty string.")
        if (
            not isinstance(layers, list)
            or not layers
            or not all(isinstance(layer, int) and layer >= 0 for layer in layers)
            or len(set(layers)) != len(layers)
        ):
            raise ConfigurationError(
                "Invalid model.backbone.layers: expected unique non-negative integers."
            )
        if (
            not isinstance(input_size, list)
            or len(input_size) != 2
            or not all(isinstance(value, int) and value > 0 for value in input_size)
        ):
            raise ConfigurationError(
                "Invalid model.backbone.input_size: expected positive [height, width]."
            )
        if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
            raise ConfigurationError(
                "Invalid model.backbone.checkpoint: expected null or a non-empty path."
            )
        if not isinstance(allow_download, bool) or not isinstance(use_cache, bool):
            raise ConfigurationError(
                "Invalid backbone loading flags: allow_download and use_cache must be booleans."
            )
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ConfigurationError(
                "Invalid model.backbone.dtype: expected float32, float16, or bfloat16."
            )
        if not isinstance(fixture_patch_size, int) or fixture_patch_size <= 0:
            raise ConfigurationError(
                "Invalid model.backbone.fixture_patch_size: expected a positive integer."
            )
        return cls(
            name=name.strip(),
            layers=tuple(layers),
            input_size=(input_size[0], input_size[1]),
            checkpoint=Path(checkpoint) if checkpoint is not None else None,
            allow_download=allow_download,
            use_cache=use_cache,
            dtype=dtype,
            fixture_patch_size=fixture_patch_size,
        )


@dataclass(frozen=True)
class FeatureFusionConfig:
    """Configure multi-layer patch alignment and fusion."""

    mode: str = "concat"
    normalize: bool = True
    target_grid: tuple[int, int] | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> FeatureFusionConfig:
        """Validate feature-fusion behavior."""

        _reject_unknown(raw, {"mode", "normalize", "target_grid"}, "model.feature_fusion")
        mode = raw.get("mode", "concat")
        normalize = raw.get("normalize", True)
        target_grid = raw.get("target_grid")
        if mode not in {"concat", "mean"}:
            raise ConfigurationError("Invalid model.feature_fusion.mode: expected concat or mean.")
        if not isinstance(normalize, bool):
            raise ConfigurationError("Invalid model.feature_fusion.normalize: expected a boolean.")
        if target_grid is not None and (
            not isinstance(target_grid, list)
            or len(target_grid) != 2
            or not all(isinstance(value, int) and value > 0 for value in target_grid)
        ):
            raise ConfigurationError(
                "Invalid model.feature_fusion.target_grid: expected null or positive [H, W]."
            )
        grid = None if target_grid is None else (target_grid[0], target_grid[1])
        return cls(mode=mode, normalize=normalize, target_grid=grid)


@dataclass(frozen=True)
class CoresetConfig:
    """Configure deterministic memory compression."""

    method: str = "kcenter"
    ratio: float = 0.1
    target_size: int | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CoresetConfig:
        """Validate coreset selection controls."""

        _reject_unknown(raw, {"method", "ratio", "target_size"}, "model.memory.coreset")
        method = raw.get("method", "kcenter")
        ratio = raw.get("ratio", 0.1)
        target_size = raw.get("target_size")
        if method not in {"none", "random", "kcenter"}:
            raise ConfigurationError("Invalid coreset method: expected none, random, or kcenter.")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 < ratio <= 1:
            raise ConfigurationError("Invalid coreset ratio: expected a value in (0, 1].")
        if target_size is not None and (not isinstance(target_size, int) or target_size <= 0):
            raise ConfigurationError(
                "Invalid coreset target_size: expected null or a positive integer."
            )
        return cls(method=method, ratio=float(ratio), target_size=target_size)


@dataclass(frozen=True)
class NearestNeighborConfig:
    """Configure exact or optional accelerated nearest-neighbour search."""

    backend: str = "torch"
    metric: str = "euclidean"
    k: int = 1
    chunk_size: int = 1024

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> NearestNeighborConfig:
        """Validate nearest-neighbour backend and memory bounds."""

        _reject_unknown(
            raw,
            {"backend", "metric", "k", "chunk_size"},
            "model.memory.nearest_neighbors",
        )
        backend = raw.get("backend", "torch")
        metric = raw.get("metric", "euclidean")
        k = raw.get("k", 1)
        chunk_size = raw.get("chunk_size", 1024)
        if backend not in {"auto", "torch", "faiss"}:
            raise ConfigurationError(
                "Invalid nearest-neighbour backend: expected auto, torch, or faiss."
            )
        if metric not in {"euclidean", "cosine"}:
            raise ConfigurationError(
                "Invalid nearest-neighbour metric: expected euclidean or cosine."
            )
        if not isinstance(k, int) or k <= 0:
            raise ConfigurationError("Invalid nearest-neighbour k: expected a positive integer.")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ConfigurationError(
                "Invalid nearest-neighbour chunk_size: expected a positive integer."
            )
        return cls(backend=backend, metric=metric, k=k, chunk_size=chunk_size)


@dataclass(frozen=True)
class MemoryConfig:
    """Combine coreset and nearest-neighbour configuration."""

    coreset: CoresetConfig
    nearest_neighbors: NearestNeighborConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MemoryConfig:
        """Validate all normal-memory settings."""

        _reject_unknown(raw, {"coreset", "nearest_neighbors"}, "model.memory")
        return cls(
            coreset=CoresetConfig.from_mapping(
                _require_mapping(raw.get("coreset", {}), "model.memory.coreset")
            ),
            nearest_neighbors=NearestNeighborConfig.from_mapping(
                _require_mapping(
                    raw.get("nearest_neighbors", {}),
                    "model.memory.nearest_neighbors",
                )
            ),
        )


@dataclass(frozen=True)
class ScoringConfig:
    """Configure patch-map smoothing and image-score aggregation."""

    image_aggregation: str = "max"
    top_k: int = 1
    smoothing_sigma: float = 0.0
    image_threshold: float | None = None
    pixel_threshold: float | None = None
    threshold_source: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ScoringConfig:
        """Validate anomaly scoring and optional predeclared thresholds."""

        allowed = {
            "image_aggregation",
            "top_k",
            "smoothing_sigma",
            "image_threshold",
            "pixel_threshold",
            "threshold_source",
        }
        _reject_unknown(raw, allowed, "model.scoring")
        aggregation = raw.get("image_aggregation", "max")
        top_k = raw.get("top_k", 1)
        sigma = raw.get("smoothing_sigma", 0.0)
        image_threshold = raw.get("image_threshold")
        pixel_threshold = raw.get("pixel_threshold")
        threshold_source = raw.get("threshold_source")
        if aggregation not in {"max", "top_k_mean"}:
            raise ConfigurationError("Invalid image aggregation: expected max or top_k_mean.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ConfigurationError("Invalid scoring top_k: expected a positive integer.")
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or sigma < 0:
            raise ConfigurationError("Invalid smoothing_sigma: expected a non-negative number.")
        for name, value in (
            ("image_threshold", image_threshold),
            ("pixel_threshold", pixel_threshold),
        ):
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise ConfigurationError(f"Invalid {name}: expected null or a number.")
        if threshold_source is not None and (
            not isinstance(threshold_source, str) or not threshold_source.strip()
        ):
            raise ConfigurationError(
                "Invalid threshold_source: expected null or a non-empty provenance string."
            )
        if (image_threshold is None) != (pixel_threshold is None):
            raise ConfigurationError(
                "Image and pixel thresholds must either both be configured or both be null."
            )
        if image_threshold is not None and threshold_source is None:
            raise ConfigurationError("Configured thresholds require a non-empty threshold_source.")
        return cls(
            image_aggregation=aggregation,
            top_k=top_k,
            smoothing_sigma=float(sigma),
            image_threshold=None if image_threshold is None else float(image_threshold),
            pixel_threshold=None if pixel_threshold is None else float(pixel_threshold),
            threshold_source=None if threshold_source is None else threshold_source.strip(),
        )


@dataclass(frozen=True)
class DinoMemoryConfig:
    """Describe the complete DINO memory model pipeline."""

    name: str
    backbone: BackboneConfig
    feature_fusion: FeatureFusionConfig
    memory: MemoryConfig
    scoring: ScoringConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DinoMemoryConfig:
        """Validate a DINO memory model configuration."""

        _reject_unknown(
            raw,
            {"name", "backbone", "feature_fusion", "memory", "scoring"},
            "model",
        )
        name = raw.get("name")
        if name != "dino_memory":
            raise ConfigurationError(f"Invalid model.name {name!r}: Phase 3 expects 'dino_memory'.")
        if "backbone" not in raw:
            raise ConfigurationError("Invalid model configuration: missing backbone section.")
        return cls(
            name=name,
            backbone=BackboneConfig.from_mapping(
                _require_mapping(raw["backbone"], "model.backbone")
            ),
            feature_fusion=FeatureFusionConfig.from_mapping(
                _require_mapping(raw.get("feature_fusion", {}), "model.feature_fusion")
            ),
            memory=MemoryConfig.from_mapping(
                _require_mapping(raw.get("memory", {}), "model.memory")
            ),
            scoring=ScoringConfig.from_mapping(
                _require_mapping(raw.get("scoring", {}), "model.scoring")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-safe model configuration mapping."""

        result = asdict(self)
        result["backbone"]["layers"] = list(self.backbone.layers)
        result["backbone"]["input_size"] = list(self.backbone.input_size)
        result["backbone"]["checkpoint"] = (
            None if self.backbone.checkpoint is None else self.backbone.checkpoint.as_posix()
        )
        if self.feature_fusion.target_grid is not None:
            result["feature_fusion"]["target_grid"] = list(self.feature_fusion.target_grid)
        return result


@dataclass(frozen=True)
class PatchCoreConfig:
    """Describe the complete PatchCore model pipeline.

    Structurally identical to :class:`DinoMemoryConfig` but validates
    ``name == "patchcore"`` so that the detector factory can dispatch
    correctly.

    Args:
        name: Must be ``"patchcore"``.
        backbone: Frozen feature provider configuration.
        feature_fusion: Multi-layer patch alignment and fusion settings.
        memory: Coreset and nearest-neighbour configuration.
        scoring: Patch-map smoothing and image-score aggregation.

    Raises:
        ConfigurationError: If the name is not ``"patchcore"`` or a section
            is invalid.
    """

    name: str
    backbone: BackboneConfig
    feature_fusion: FeatureFusionConfig
    memory: MemoryConfig
    scoring: ScoringConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PatchCoreConfig:
        """Validate a PatchCore model configuration."""

        _reject_unknown(
            raw,
            {"name", "backbone", "feature_fusion", "memory", "scoring"},
            "model",
        )
        name = raw.get("name")
        if name != "patchcore":
            raise ConfigurationError(f"Invalid model.name {name!r}: expected 'patchcore'.")
        if "backbone" not in raw:
            raise ConfigurationError("Invalid model configuration: missing backbone section.")
        return cls(
            name=name,
            backbone=BackboneConfig.from_mapping(
                _require_mapping(raw["backbone"], "model.backbone")
            ),
            feature_fusion=FeatureFusionConfig.from_mapping(
                _require_mapping(raw.get("feature_fusion", {}), "model.feature_fusion")
            ),
            memory=MemoryConfig.from_mapping(
                _require_mapping(raw.get("memory", {}), "model.memory")
            ),
            scoring=ScoringConfig.from_mapping(
                _require_mapping(raw.get("scoring", {}), "model.scoring")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-safe model configuration mapping."""

        result = asdict(self)
        result["backbone"]["layers"] = list(self.backbone.layers)
        result["backbone"]["input_size"] = list(self.backbone.input_size)
        result["backbone"]["checkpoint"] = (
            None if self.backbone.checkpoint is None else self.backbone.checkpoint.as_posix()
        )
        if self.feature_fusion.target_grid is not None:
            result["feature_fusion"]["target_grid"] = list(self.feature_fusion.target_grid)
        return result


@dataclass(frozen=True)
class EfficientADTeacherConfig:
    """Configure the frozen teacher network for EfficientAD.

    Args:
        pretrained: Whether to load ImageNet pretrained ResNet18 weights.
        checkpoint: Optional local checkpoint path for custom teacher weights.
        allow_download: Permit torchvision network access for weight download.
        use_cache: Permit reuse of cached torchvision weights.
        out_channels: Teacher output channel count (determined by ResNet18 layer2).
    """

    pretrained: bool = True
    checkpoint: Path | None = None
    allow_download: bool = False
    use_cache: bool = True
    out_channels: int = 128

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EfficientADTeacherConfig:
        """Validate teacher configuration from parsed YAML."""

        _reject_unknown(
            raw,
            {"pretrained", "checkpoint", "allow_download", "use_cache", "out_channels"},
            "model.teacher",
        )
        pretrained = raw.get("pretrained", True)
        checkpoint = raw.get("checkpoint")
        allow_download = raw.get("allow_download", False)
        use_cache = raw.get("use_cache", True)
        out_channels = raw.get("out_channels", 128)
        if not isinstance(pretrained, bool):
            raise ConfigurationError("Invalid model.teacher.pretrained: expected a boolean.")
        if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
            raise ConfigurationError(
                "Invalid model.teacher.checkpoint: expected null or a non-empty path."
            )
        if not isinstance(allow_download, bool) or not isinstance(use_cache, bool):
            raise ConfigurationError(
                "Invalid teacher flags: allow_download and use_cache must be booleans."
            )
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ConfigurationError(
                "Invalid model.teacher.out_channels: expected a positive integer."
            )
        resolved_checkpoint = (
            _resolve_path_value(checkpoint, "model.teacher.checkpoint") if checkpoint else None
        )
        return cls(
            pretrained=pretrained,
            checkpoint=Path(resolved_checkpoint) if resolved_checkpoint else None,
            allow_download=allow_download,
            use_cache=use_cache,
            out_channels=out_channels,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable teacher configuration."""

        return {
            "pretrained": self.pretrained,
            "checkpoint": None if self.checkpoint is None else self.checkpoint.as_posix(),
            "allow_download": self.allow_download,
            "use_cache": self.use_cache,
            "out_channels": self.out_channels,
        }


@dataclass(frozen=True)
class EfficientADTrainingConfig:
    """Configure the EfficientAD training loop.

    Args:
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        num_steps: Total training steps.
        batch_size: Training batch size.
        lambda_global: Global (autoencoder) loss weight.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_steps: int = 70000
    batch_size: int = 4
    lambda_global: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EfficientADTrainingConfig:
        """Validate training configuration from parsed YAML."""

        _reject_unknown(
            raw,
            {"learning_rate", "weight_decay", "num_steps", "batch_size", "lambda_global"},
            "model.training",
        )
        learning_rate = raw.get("learning_rate", 1e-3)
        weight_decay = raw.get("weight_decay", 1e-5)
        num_steps = raw.get("num_steps", 70000)
        batch_size = raw.get("batch_size", 4)
        lambda_global = raw.get("lambda_global", 1.0)
        for name, value in (("learning_rate", learning_rate), ("weight_decay", weight_decay)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ConfigurationError(
                    f"Invalid model.training.{name}: expected a non-negative number."
                )
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ConfigurationError(
                "Invalid model.training.num_steps: expected a positive integer."
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError(
                "Invalid model.training.batch_size: expected a positive integer."
            )
        if (
            not isinstance(lambda_global, (int, float))
            or isinstance(lambda_global, bool)
            or lambda_global < 0
        ):
            raise ConfigurationError(
                "Invalid model.training.lambda_global: expected a non-negative number."
            )
        return cls(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            num_steps=num_steps,
            batch_size=batch_size,
            lambda_global=float(lambda_global),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable training configuration."""

        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "lambda_global": self.lambda_global,
        }


@dataclass(frozen=True)
class EfficientADScoringConfig:
    """Configure EfficientAD anomaly map fusion and scoring.

    Args:
        local_weight: Weight for the local (student-teacher) anomaly map.
        global_weight: Weight for the global (autoencoder) anomaly map.
        smoothing_sigma: Gaussian smoothing sigma in output pixels.
        image_threshold: Calibrated image-level threshold (or None).
        pixel_threshold: Calibrated pixel-level threshold (or None).
        threshold_source: Threshold provenance string (or None).
    """

    local_weight: float = 1.0
    global_weight: float = 1.0
    smoothing_sigma: float = 4.0
    image_threshold: float | None = None
    pixel_threshold: float | None = None
    threshold_source: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EfficientADScoringConfig:
        """Validate scoring configuration from parsed YAML."""

        allowed = {
            "local_weight",
            "global_weight",
            "smoothing_sigma",
            "image_threshold",
            "pixel_threshold",
            "threshold_source",
        }
        _reject_unknown(raw, allowed, "model.scoring")
        local_weight = raw.get("local_weight", 1.0)
        global_weight = raw.get("global_weight", 1.0)
        smoothing_sigma = raw.get("smoothing_sigma", 4.0)
        image_threshold = raw.get("image_threshold")
        pixel_threshold = raw.get("pixel_threshold")
        threshold_source = raw.get("threshold_source")
        for name, value in (("local_weight", local_weight), ("global_weight", global_weight)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ConfigurationError(f"Invalid scoring {name}: expected a non-negative number.")
        if (
            not isinstance(smoothing_sigma, (int, float))
            or isinstance(smoothing_sigma, bool)
            or smoothing_sigma < 0
        ):
            raise ConfigurationError("Invalid smoothing_sigma: expected a non-negative number.")
        for name, value in (
            ("image_threshold", image_threshold),
            ("pixel_threshold", pixel_threshold),
        ):
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise ConfigurationError(f"Invalid {name}: expected null or a number.")
        if threshold_source is not None and (
            not isinstance(threshold_source, str) or not threshold_source.strip()
        ):
            raise ConfigurationError(
                "Invalid threshold_source: expected null or a non-empty provenance string."
            )
        if (image_threshold is None) != (pixel_threshold is None):
            raise ConfigurationError(
                "Image and pixel thresholds must either both be configured or both be null."
            )
        if image_threshold is not None and threshold_source is None:
            raise ConfigurationError("Configured thresholds require a non-empty threshold_source.")
        return cls(
            local_weight=float(local_weight),
            global_weight=float(global_weight),
            smoothing_sigma=float(smoothing_sigma),
            image_threshold=None if image_threshold is None else float(image_threshold),
            pixel_threshold=None if pixel_threshold is None else float(pixel_threshold),
            threshold_source=None if threshold_source is None else threshold_source.strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable scoring configuration."""

        return {
            "local_weight": self.local_weight,
            "global_weight": self.global_weight,
            "smoothing_sigma": self.smoothing_sigma,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "threshold_source": self.threshold_source,
        }


@dataclass(frozen=True)
class EfficientADCheckpointConfig:
    """Configure checkpoint saving and resume for EfficientAD training.

    Args:
        enabled: Whether to save periodic checkpoints during training.
        interval: Number of training steps between checkpoint saves.
        dir: Directory to store checkpoint files (created if missing).
        resume_from: Explicit checkpoint file path to resume from. When set,
            this takes priority over auto_resume.
        auto_resume: Whether to auto-resume from the latest valid checkpoint
            found in ``dir`` at the start of training.
        max_to_keep: Maximum number of checkpoint files to retain in ``dir``.
            Older checkpoints beyond this count are deleted after a successful
            save.
    """

    enabled: bool = True
    interval: int = 10000
    dir: Path | None = None
    resume_from: Path | None = None
    auto_resume: bool = True
    max_to_keep: int = 3

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EfficientADCheckpointConfig:
        """Validate checkpoint configuration from parsed YAML.

        Args:
            raw: Checkpoint section mapping.

        Returns:
            An immutable checkpoint configuration.

        Raises:
            ConfigurationError: If a field is invalid or unknown.
        """

        allowed = {"enabled", "interval", "dir", "resume_from", "auto_resume", "max_to_keep"}
        _reject_unknown(raw, allowed, "model.checkpoint")
        enabled = raw.get("enabled", True)
        interval = raw.get("interval", 10000)
        ckpt_dir = raw.get("dir")
        resume_from = raw.get("resume_from")
        auto_resume = raw.get("auto_resume", True)
        max_to_keep = raw.get("max_to_keep", 3)
        if not isinstance(enabled, bool):
            raise ConfigurationError("Invalid model.checkpoint.enabled: expected a boolean.")
        if not isinstance(interval, int) or interval <= 0:
            raise ConfigurationError(
                "Invalid model.checkpoint.interval: expected a positive integer."
            )
        if ckpt_dir is not None and (not isinstance(ckpt_dir, str) or not ckpt_dir.strip()):
            raise ConfigurationError(
                "Invalid model.checkpoint.dir: expected null or a non-empty path."
            )
        if resume_from is not None and (
            not isinstance(resume_from, str) or not resume_from.strip()
        ):
            raise ConfigurationError(
                "Invalid model.checkpoint.resume_from: expected null or a non-empty path."
            )
        if not isinstance(auto_resume, bool):
            raise ConfigurationError("Invalid model.checkpoint.auto_resume: expected a boolean.")
        if not isinstance(max_to_keep, int) or max_to_keep <= 0:
            raise ConfigurationError(
                "Invalid model.checkpoint.max_to_keep: expected a positive integer."
            )
        resolved_dir = (
            Path(_resolve_path_value(ckpt_dir, "model.checkpoint.dir")) if ckpt_dir else None
        )
        resolved_resume = (
            Path(_resolve_path_value(resume_from, "model.checkpoint.resume_from"))
            if resume_from
            else None
        )
        return cls(
            enabled=enabled,
            interval=interval,
            dir=resolved_dir,
            resume_from=resolved_resume,
            auto_resume=auto_resume,
            max_to_keep=max_to_keep,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable checkpoint configuration."""

        return {
            "enabled": self.enabled,
            "interval": self.interval,
            "dir": None if self.dir is None else self.dir.as_posix(),
            "resume_from": None if self.resume_from is None else self.resume_from.as_posix(),
            "auto_resume": self.auto_resume,
            "max_to_keep": self.max_to_keep,
        }


@dataclass(frozen=True)
class EfficientADModelConfig:
    """Describe the complete EfficientAD model pipeline.

    Args:
        name: Must be ``"efficientad"``.
        is_fixture: Whether to use fixture (small deterministic) networks.
        teacher: Frozen teacher network configuration.
        student: Student network configuration (ae_channels).
        training: Training loop configuration.
        scoring: Anomaly map fusion and scoring configuration.
        input_size: Input image height and width.

    Raises:
        ConfigurationError: If the name is not ``"efficientad"`` or a section
            is invalid.
    """

    name: str
    is_fixture: bool = False
    teacher: EfficientADTeacherConfig = EfficientADTeacherConfig()
    training: EfficientADTrainingConfig = EfficientADTrainingConfig()
    scoring: EfficientADScoringConfig = EfficientADScoringConfig()
    checkpoint: EfficientADCheckpointConfig = EfficientADCheckpointConfig()
    input_size: tuple[int, int] = (256, 256)
    ae_channels: int = 128

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EfficientADModelConfig:
        """Validate an EfficientAD model configuration."""

        allowed = {
            "name",
            "is_fixture",
            "teacher",
            "student",
            "training",
            "scoring",
            "checkpoint",
            "input_size",
        }
        _reject_unknown(raw, allowed, "model")
        name = raw.get("name")
        if name != "efficientad":
            raise ConfigurationError(f"Invalid model.name {name!r}: expected 'efficientad'.")
        is_fixture = raw.get("is_fixture", False)
        if not isinstance(is_fixture, bool):
            raise ConfigurationError("Invalid model.is_fixture: expected a boolean.")
        teacher = EfficientADTeacherConfig.from_mapping(
            _require_mapping(raw.get("teacher", {}), "model.teacher")
        )
        training = EfficientADTrainingConfig.from_mapping(
            _require_mapping(raw.get("training", {}), "model.training")
        )
        scoring = EfficientADScoringConfig.from_mapping(
            _require_mapping(raw.get("scoring", {}), "model.scoring")
        )
        checkpoint = EfficientADCheckpointConfig.from_mapping(
            _require_mapping(raw.get("checkpoint", {}), "model.checkpoint")
        )
        input_size = raw.get("input_size", [256, 256])
        if (
            not isinstance(input_size, list)
            or len(input_size) != 2
            or not all(isinstance(v, int) and v > 0 for v in input_size)
        ):
            raise ConfigurationError("Invalid model.input_size: expected positive [height, width].")
        student_raw = _require_mapping(raw.get("student", {}), "model.student")
        _reject_unknown(student_raw, {"ae_channels"}, "model.student")
        ae_channels = student_raw.get("ae_channels", 128)
        if not isinstance(ae_channels, int) or ae_channels <= 0:
            raise ConfigurationError(
                "Invalid model.student.ae_channels: expected a positive integer."
            )
        return cls(
            name=name,
            is_fixture=is_fixture,
            teacher=teacher,
            training=training,
            scoring=scoring,
            checkpoint=checkpoint,
            input_size=(input_size[0], input_size[1]),
            ae_channels=ae_channels,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-safe model configuration mapping."""

        return {
            "name": self.name,
            "is_fixture": self.is_fixture,
            "teacher": self.teacher.to_dict(),
            "student": {"ae_channels": self.ae_channels},
            "training": self.training.to_dict(),
            "scoring": self.scoring.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "input_size": list(self.input_size),
        }


def _model_config_from_mapping(
    raw: Mapping[str, Any],
) -> DinoMemoryConfig | PatchCoreConfig | EfficientADModelConfig:
    """Dispatch model configuration parsing based on the ``name`` field.

    Args:
        raw: Parsed YAML model section mapping.

    Returns:
        A validated model configuration dataclass.

    Raises:
        ConfigurationError: If the name is missing or unsupported.
    """

    name = raw.get("name")
    if name == "dino_memory":
        return DinoMemoryConfig.from_mapping(raw)
    if name == "patchcore":
        return PatchCoreConfig.from_mapping(raw)
    if name == "efficientad":
        return EfficientADModelConfig.from_mapping(raw)
    raise ConfigurationError(
        f"Invalid model.name {name!r}: expected 'dino_memory', 'patchcore', or 'efficientad'."
    )


#: Union of all supported model configuration types.
ModelConfig = DinoMemoryConfig | PatchCoreConfig | EfficientADModelConfig


@dataclass(frozen=True)
class CalibrationConfig:
    """Configure held-out normal-only threshold calibration.

    When enabled, normal training images are split into a memory subset
    (used for memory bank construction) and a calibration subset (used for
    threshold calibration).  This prevents the self-scoring bias that
    occurs when the same images are used for both memory and threshold
    calibration.

    Args:
        enabled: When True, split and calibrate; when False, use all
            normal training images for memory and rely on predeclared
            thresholds.
        ratio: Proportion of normal training samples reserved for
            calibration, in the open interval ``(0, 1)``.
        seed: Deterministic seed for the image-level shuffle.
        quantile: Quantile in ``(0, 1)`` used for threshold selection
            from held-out normal calibration scores.

    Raises:
        ConfigurationError: If any field violates the schema.
    """

    enabled: bool = True
    ratio: float = 0.1
    seed: int = 0
    quantile: float = 0.995

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CalibrationConfig:
        """Validate calibration configuration from parsed YAML.

        Args:
            raw: Calibration section mapping (may be empty for defaults).

        Returns:
            An immutable validated calibration configuration.

        Raises:
            ConfigurationError: If any field is invalid or unknown.
        """

        _reject_unknown(raw, {"enabled", "ratio", "seed", "quantile"}, "calibration")
        enabled = raw.get("enabled", True)
        ratio = raw.get("ratio", 0.1)
        seed = raw.get("seed", 0)
        quantile = raw.get("quantile", 0.995)
        if not isinstance(enabled, bool):
            raise ConfigurationError("Invalid calibration.enabled: expected a boolean.")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0.0 < ratio < 1.0:
            raise ConfigurationError("Invalid calibration.ratio: expected a value in (0, 1).")
        if not isinstance(seed, int) or seed < 0:
            raise ConfigurationError("Invalid calibration.seed: expected a non-negative integer.")
        if (
            not isinstance(quantile, (int, float))
            or isinstance(quantile, bool)
            or not 0.0 < quantile < 1.0
        ):
            raise ConfigurationError("Invalid calibration.quantile: expected a value in (0, 1).")
        return cls(
            enabled=enabled,
            ratio=float(ratio),
            seed=seed,
            quantile=float(quantile),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable calibration configuration."""

        return {
            "enabled": self.enabled,
            "ratio": self.ratio,
            "seed": self.seed,
            "quantile": self.quantile,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Combine dataset and runtime schemas for a reproducible experiment.

    Args:
        dataset: Dataset location and preprocessing configuration.
        runtime: Reproducibility and output controls.
        experiment_name: Human-readable experiment label.
        model: Model architecture configuration (may be absent for non-model
            commands such as dataset validation).
        calibration: Held-out normal-only threshold calibration controls.

    Raises:
        ConfigurationError: If any section is invalid.
    """

    dataset: DatasetConfig
    runtime: RuntimeConfig
    experiment_name: str
    model: DinoMemoryConfig | PatchCoreConfig | EfficientADModelConfig | None = None
    calibration: CalibrationConfig = CalibrationConfig()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        """Validate and construct a complete experiment configuration.

        Args:
            raw: Top-level parsed YAML mapping.

        Returns:
            A complete immutable configuration.

        Raises:
            ConfigurationError: If sections are missing, unknown, or invalid.
        """

        allowed = {"dataset", "runtime", "experiment_name", "model", "calibration"}
        _reject_unknown(raw, allowed, "experiment")
        if "dataset" not in raw:
            raise ConfigurationError(
                "Invalid experiment configuration: missing required 'dataset' section."
            )
        name = raw.get("experiment_name", "shiftad_experiment")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("Invalid experiment_name: expected a non-empty string.")
        dataset = DatasetConfig.from_mapping(_require_mapping(raw["dataset"], "dataset"))
        runtime = RuntimeConfig.from_mapping(_require_mapping(raw.get("runtime", {}), "runtime"))
        model = (
            _model_config_from_mapping(_require_mapping(raw["model"], "model"))
            if "model" in raw
            else None
        )
        calibration = CalibrationConfig.from_mapping(
            _require_mapping(raw.get("calibration", {}), "calibration")
        )
        if (
            model is not None
            and isinstance(model, (DinoMemoryConfig, PatchCoreConfig))
            and dataset.image_size != model.backbone.input_size
        ):
            raise ConfigurationError(
                f"dataset.image_size {dataset.image_size} must match "
                f"model.backbone.input_size {model.backbone.input_size}."
            )
        if (
            model is not None
            and isinstance(model, EfficientADModelConfig)
            and dataset.image_size != model.input_size
        ):
            raise ConfigurationError(
                f"dataset.image_size {dataset.image_size} must match "
                f"model.input_size {model.input_size}."
            )
        return cls(
            dataset=dataset,
            runtime=runtime,
            experiment_name=name.strip(),
            model=model,
            calibration=calibration,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable resolved configuration.

        Returns:
            A dictionary with paths converted to portable strings.
        """

        result = asdict(self)
        result["dataset"]["root"] = self.dataset.root.as_posix()
        result["dataset"]["categories"] = list(self.dataset.categories)
        result["dataset"]["image_size"] = list(self.dataset.image_size)
        result["runtime"]["output_dir"] = self.runtime.output_dir.as_posix()
        result["calibration"] = self.calibration.to_dict()
        if self.model is not None:
            result["model"] = self.model.to_dict()
        else:
            result.pop("model", None)
        return result


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load a UTF-8 YAML experiment configuration.

    Args:
        path: Existing YAML file.

    Returns:
        A validated experiment configuration.

    Raises:
        ConfigurationError: If the file is missing, unreadable, invalid YAML, or
            violates the schema.
    """

    if not path.is_file():
        raise ConfigurationError(
            f"Cannot load configuration from {path}: expected an existing YAML file."
        )
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration from {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Cannot parse YAML configuration {path}: {exc}") from exc
    return ExperimentConfig.from_mapping(_require_mapping(parsed, "experiment"))


def save_resolved_config(config: ExperimentConfig, path: Path) -> None:
    """Write a resolved configuration as deterministic UTF-8 YAML.

    Args:
        config: Validated experiment configuration.
        path: Destination file. Parent directories are created.

    Raises:
        OSError: If the destination cannot be created or written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
