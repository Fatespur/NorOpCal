"""EfficientAD anomaly detector with teacher-student distillation.

Implements the unified AnomalyDetector interface for EfficientAD, combining
a frozen teacher, trainable student, and trainable autoencoder to produce
image-level scores and pixel-level anomaly maps.

The detector trains on normal images only: the student learns to predict
teacher features (local branch), and the autoencoder learns to reconstruct
student AE channels (global branch). At inference, anomalies are detected
as regions where the student fails to match the teacher (local) or the
autoencoder fails to reconstruct (global).

Reference:
    B. J. Holzinger et al., "EfficientAD: Accurate Visual Anomaly Detection
    at Millisecond-Level Latencies," WACV 2024.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch import Tensor, nn

from shiftad import __version__
from shiftad.methods.base import AnomalyDetector, DetectorError, PredictionBatch
from shiftad.models.efficientad.autoencoder import AutoEncoder, FixtureAutoEncoder
from shiftad.models.efficientad.losses import EfficientADLoss
from shiftad.models.efficientad.normalization import FeatureNormalizer
from shiftad.models.efficientad.student import FixtureStudentNetwork, StudentNetwork
from shiftad.models.efficientad.teacher import FixtureTeacherNetwork, TeacherNetwork
from shiftad.utils.device import resolve_device

ARTIFACT_VERSION = 2
REQUIRED_ARTIFACTS = (
    "manifest.json",
    "config.resolved.yaml",
    "teacher.pt",
    "student.pt",
    "autoencoder.pt",
    "normalizer.pt",
    "metadata.json",
)

#: Scope label for synthetic fixture runs.
FIXTURE_SCOPE = "synthetic_fixture_acceptance_only"

#: Scope label for real-data runs.
REAL_SCOPE = "real_efficientad"

#: Checkpoint format version for forward compatibility checks.
#:
#: Version 2 removes the complete teacher feature list from the checkpoint
#: and replaces it with a compact Welford sufficient-statistics accumulator
#: for the normalizer.  It also introduces the ``.complete`` transaction
#: marker file.
CHECKPOINT_VERSION = 2

#: Module-level logger for checkpoint operations.
_logger = logging.getLogger(__name__)


class CheckpointError(DetectorError):
    """Report a checkpoint save, load, or compatibility failure."""


class NormalizerAccumulator:
    """Welford accumulator for channel-wise teacher feature statistics.

    Tracks running ``count``, ``mean``, and ``M2`` (sum of squared
    deviations) across all spatial positions and samples seen, enabling
    memory-efficient reconstruction of the :class:`FeatureNormalizer` without
    storing the complete feature tensor list.

    The accumulator computes per-channel statistics over the ``N x H x W``
    dimension of ``[B, C, H, W]`` feature maps, matching the semantics of
    :meth:`FeatureNormalizer.fit_from_features`.

    Attributes:
        num_channels: Number of feature channels (C).
        count: Total number of per-channel values accumulated.
        mean: Running mean tensor shaped ``[C]``.
        m2: Running sum of squared deviations shaped ``[C]``.
    """

    def __init__(self, num_channels: int, *, device: torch.device | str = "cpu") -> None:
        """Initialise an empty accumulator.

        Args:
            num_channels: Number of feature channels to track.
            device: Device on which to allocate internal tensors.
        """

        self.num_channels = num_channels
        self._device = torch.device(device) if isinstance(device, str) else device
        self.count: int = 0
        self.mean: Tensor = torch.zeros(num_channels, device=self._device)
        self.m2: Tensor = torch.zeros(num_channels, device=self._device)

    def update(self, features: Tensor) -> None:
        """Incorporate a batch of teacher features.

        Args:
            features: Tensor shaped ``[B, C, H, W]``.  All values are
                detached and moved to the accumulator device before
                processing.

        Raises:
            ValueError: If the channel dimension does not match
                ``num_channels``.
        """

        if features.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] features, got shape {tuple(features.shape)}.")
        if features.shape[1] != self.num_channels:
            raise ValueError(
                f"Feature channels {features.shape[1]} != accumulator channels {self.num_channels}."
            )
        features = features.detach().to(device=self._device, dtype=self.mean.dtype)
        b, _c, h, w = features.shape
        batch_count = b * h * w
        # Flatten to [batch_count, C] for per-channel statistics.
        flat = features.permute(0, 2, 3, 1).reshape(-1, self.num_channels)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_m2 = batch_var * batch_count

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean.clone()
            self.m2 = batch_m2.clone()
        else:
            count_new = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * batch_count / count_new
            self.m2 = self.m2 + batch_m2 + (delta * delta) * (self.count * batch_count / count_new)
            self.count = count_new

    def to_normalizer(self) -> FeatureNormalizer:
        """Build a :class:`FeatureNormalizer` from accumulated statistics.

        Returns:
            A fitted normalizer, or an unfitted one if fewer than 2 values
            have been accumulated.
        """

        if self.count < 2:
            return FeatureNormalizer()
        std = torch.sqrt(self.m2 / (self.count - 1))
        std = std.clamp(min=FeatureNormalizer.EPSILON)
        return FeatureNormalizer(mean=self.mean.clone(), std=std)

    def state_dict(self) -> dict[str, Any]:
        """Return a CPU-resident serialisable state.

        Returns:
            A dictionary with ``count``, ``mean``, ``m2``, and
            ``num_channels`` keys.  Tensors are moved to CPU to avoid
            device coupling in saved checkpoints.
        """

        return {
            "count": self.count,
            "mean": self.mean.cpu(),
            "m2": self.m2.cpu(),
            "num_channels": self.num_channels,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> NormalizerAccumulator:
        """Reconstruct an accumulator from a saved state.

        Args:
            state: Dictionary produced by :meth:`state_dict`.
            device: Target device for the restored tensors.

        Returns:
            A restored accumulator ready for further updates.

        Raises:
            CheckpointError: If the state is malformed.
        """

        try:
            num_channels = int(state["num_channels"])
            obj = cls(num_channels, device=device)
            obj.count = int(state["count"])
            obj.mean = state["mean"].to(device)
            obj.m2 = state["m2"].to(device)
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"Malformed normalizer accumulator state: {exc}") from exc
        return obj


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_teacher_state_hash(state_dict: dict[str, torch.Tensor]) -> str:
    """Return the canonical SHA-256 hash of a Teacher state dict.

    This is the single authoritative implementation of the Teacher state hash.
    The hash is computed over the state dict in its natural (insertion) order:
    each parameter name and its raw CPU bytes are fed into the digest. The
    natural order is used because the original production implementation and
    the frozen Phase 4 checkpoints were produced with this exact algorithm;
    sorting the keys would change the digest and break compatibility with the
    recorded checkpoint `teacher_hash` values.

    Args:
        state_dict: Mapping of parameter names to tensors (e.g. the result of
            ``torch.load(teacher_path)`` for a raw state dict, or
            ``model.state_dict()`` for a module).

    Returns:
        Lowercase hex SHA-256 digest of the serialized state dict.

    Note:
        Callers must pass the Teacher's state dict exactly as the production
        code did at checkpoint time: a plain dict / OrderedDict of name ->
        tensor. Do not sort keys and do not reorder tensors.
    """

    digest = hashlib.sha256()
    for name, param in state_dict.items():
        digest.update(name.encode("utf-8"))
        digest.update(param.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    """Return the current Git commit hash or None if unavailable."""

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


class EfficientADConfig:
    """Lightweight runtime configuration for EfficientAD.

    This is a plain data container constructed from the validated YAML
    configuration. It avoids coupling the detector to the frozen dataclass
    hierarchy, allowing the detector to update scoring thresholds after
    calibration without mutating immutable config objects.

    Attributes:
        is_fixture: Whether to use fixture (small deterministic) networks.
        teacher_pretrained: Whether the teacher uses pretrained weights.
        teacher_checkpoint: Optional local teacher checkpoint path.
        teacher_allow_download: Whether to allow downloading teacher weights.
        teacher_use_cache: Whether to use cached teacher weights.
        teacher_channels: Number of teacher output channels.
        ae_channels: Number of autoencoder input channels.
        input_size: Input image height and width.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        num_steps: Total training steps.
        batch_size: Training batch size.
        lambda_global: Global loss weight.
        local_weight: Local anomaly map fusion weight.
        global_weight: Global anomaly map fusion weight.
        smoothing_sigma: Gaussian smoothing sigma for anomaly maps.
        image_threshold: Calibrated image-level threshold (or None).
        pixel_threshold: Calibrated pixel-level threshold (or None).
        threshold_source: Threshold provenance string (or None).
        checkpoint_enabled: Whether periodic checkpointing is active.
        checkpoint_interval: Steps between checkpoint saves.
        checkpoint_dir: Directory for checkpoint files.
        resume_from: Explicit checkpoint path to resume from.
        auto_resume: Whether to auto-resume from the latest checkpoint.
        max_to_keep: Maximum checkpoint files to retain.
    """

    def __init__(
        self,
        *,
        is_fixture: bool = False,
        teacher_pretrained: bool = True,
        teacher_checkpoint: Path | None = None,
        teacher_allow_download: bool = False,
        teacher_use_cache: bool = True,
        teacher_channels: int = 128,
        ae_channels: int = 128,
        input_size: tuple[int, int] = (256, 256),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_steps: int = 70000,
        batch_size: int = 4,
        lambda_global: float = 1.0,
        local_weight: float = 1.0,
        global_weight: float = 1.0,
        smoothing_sigma: float = 4.0,
        image_threshold: float | None = None,
        pixel_threshold: float | None = None,
        threshold_source: str | None = None,
        checkpoint_enabled: bool = True,
        checkpoint_interval: int = 10000,
        checkpoint_dir: Path | None = None,
        resume_from: Path | None = None,
        auto_resume: bool = True,
        max_to_keep: int = 3,
    ) -> None:
        self.is_fixture = is_fixture
        self.teacher_pretrained = teacher_pretrained
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_allow_download = teacher_allow_download
        self.teacher_use_cache = teacher_use_cache
        self.teacher_channels = teacher_channels
        self.ae_channels = ae_channels
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.lambda_global = lambda_global
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.smoothing_sigma = smoothing_sigma
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold
        self.threshold_source = threshold_source
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.resume_from = resume_from
        self.auto_resume = auto_resume
        self.max_to_keep = max_to_keep

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable configuration mapping."""

        return {
            "is_fixture": self.is_fixture,
            "teacher": {
                "pretrained": self.teacher_pretrained,
                "checkpoint": (
                    None if self.teacher_checkpoint is None else str(self.teacher_checkpoint)
                ),
                "allow_download": self.teacher_allow_download,
                "use_cache": self.teacher_use_cache,
                "out_channels": self.teacher_channels,
            },
            "student": {
                "ae_channels": self.ae_channels,
            },
            "autoencoder": {},
            "training": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_steps": self.num_steps,
                "batch_size": self.batch_size,
                "lambda_global": self.lambda_global,
            },
            "scoring": {
                "local_weight": self.local_weight,
                "global_weight": self.global_weight,
                "smoothing_sigma": self.smoothing_sigma,
                "image_threshold": self.image_threshold,
                "pixel_threshold": self.pixel_threshold,
                "threshold_source": self.threshold_source,
            },
            "checkpoint": {
                "enabled": self.checkpoint_enabled,
                "interval": self.checkpoint_interval,
                "dir": (None if self.checkpoint_dir is None else str(self.checkpoint_dir)),
                "resume_from": (None if self.resume_from is None else str(self.resume_from)),
                "auto_resume": self.auto_resume,
                "max_to_keep": self.max_to_keep,
            },
            "input_size": list(self.input_size),
        }


class EfficientADDetector(AnomalyDetector):
    """EfficientAD anomaly detector with teacher-student distillation.

    Args:
        config: EfficientAD runtime configuration.
        device: CPU or available accelerator request.

    Attributes:
        config: The active configuration (may be updated after calibration).
        device: Resolved execution device.
        teacher: Frozen teacher network (None until initialized).
        student: Trainable student network (None until initialized).
        autoencoder: Trainable autoencoder network (None until initialized).
        normalizer: Feature normalizer (None until fit or loaded).
        is_fitted: Whether the detector has been trained or loaded.
    """

    def __init__(
        self,
        config: EfficientADConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = resolve_device(str(device))
        self.teacher: nn.Module | None = None
        self.student: StudentNetwork | FixtureStudentNetwork | None = None
        self.autoencoder: AutoEncoder | FixtureAutoEncoder | None = None
        self.normalizer: FeatureNormalizer = FeatureNormalizer()
        self.is_fitted: bool = False
        self.build_metadata: dict[str, Any] = {}
        self._init_networks()

    @staticmethod
    def _deterministic_init(module: nn.Module) -> None:
        """Initialize fixture Conv2d weights deterministically with Kaiming.

        Seeds the global PyTorch RNG with a fixed value (0) and applies
        Kaiming-uniform initialisation to all Conv2d layers.  The caller is
        responsible for saving and restoring the global RNG state around the
        entire fixture-teacher creation process, because the
        ``FixtureTeacherNetwork`` constructor itself consumes RNG state
        during default Conv2d initialisation.

        Args:
            module: The module whose Conv2d layers to reinitialize.
        """

        torch.manual_seed(0)
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def _init_networks(self) -> None:
        """Initialize teacher, student, and autoencoder networks."""

        if self.config.is_fixture:
            # Save global RNG state *before* creating the fixture teacher.
            # The FixtureTeacherNetwork constructor consumes RNG state for
            # default Conv2d initialisation, and _deterministic_init reseeds
            # with a fixed seed.  We restore the original state afterward so
            # that student and autoencoder initialisation (and any downstream
            # code) sees an unchanged RNG state.
            saved_torch = torch.get_rng_state()
            saved_numpy = np.random.get_state()
            saved_python = random.getstate()

            self.teacher = FixtureTeacherNetwork(out_channels=self.config.teacher_channels)
            self._deterministic_init(self.teacher)

            torch.set_rng_state(saved_torch)
            np.random.set_state(saved_numpy)
            random.setstate(saved_python)

            self.student = FixtureStudentNetwork(
                teacher_channels=self.config.teacher_channels,
                ae_channels=self.config.ae_channels,
            )
            self.autoencoder = FixtureAutoEncoder(in_channels=self.config.ae_channels)
        else:
            self.teacher = TeacherNetwork(
                pretrained=self.config.teacher_pretrained,
                checkpoint=self.config.teacher_checkpoint,
                allow_download=self.config.teacher_allow_download,
                use_cache=self.config.teacher_use_cache,
            )
            # Use the actual teacher output channels.
            actual_channels = self.teacher.out_channels
            if actual_channels != self.config.teacher_channels:
                self.config.teacher_channels = actual_channels
            self.student = StudentNetwork(
                teacher_channels=self.config.teacher_channels,
                ae_channels=self.config.ae_channels,
            )
            self.autoencoder = AutoEncoder(in_channels=self.config.ae_channels)

        self.teacher.to(self.device)
        self.student.to(self.device)
        self.autoencoder.to(self.device)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

    @staticmethod
    def _batch_field(batch: object, name: str) -> object:
        """Extract a named field from a training batch."""

        if not isinstance(batch, Mapping) or name not in batch:
            raise DetectorError(f"Training batch must be a mapping containing {name!r}.")
        return batch[name]

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _compute_config_fingerprint(self) -> str:
        """Return a SHA-256 fingerprint of architecture-relevant config fields.

        The fingerprint covers all fields that affect the model structure or
        training dynamics: ``is_fixture``, ``teacher_channels``,
        ``ae_channels``, ``input_size``, ``num_steps``, ``learning_rate``,
        ``weight_decay``, ``batch_size``, and ``lambda_global``.

        Returns:
            A hex digest string.

        Raises:
            DetectorError: If networks are not initialized.
        """

        payload = json.dumps(
            {
                "is_fixture": self.config.is_fixture,
                "teacher_channels": self.config.teacher_channels,
                "ae_channels": self.config.ae_channels,
                "input_size": list(self.config.input_size),
                "num_steps": self.config.num_steps,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "lambda_global": self.config.lambda_global,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _compute_teacher_state_hash(self) -> str:
        """Return a SHA-256 hash of the frozen teacher's state_dict.

        Returns:
            A hex digest string, or ``"no_teacher"`` if the teacher is absent.
        """

        if self.teacher is None:
            return "no_teacher"
        return compute_teacher_state_hash(self.teacher.state_dict())

    def save_checkpoint(
        self,
        path: Path,
        *,
        step: int,
        optimizer: torch.optim.Optimizer,
        normalizer_accumulator: NormalizerAccumulator,
        compatibility_info: dict[str, Any] | None = None,
    ) -> None:
        """Atomically save a training checkpoint as a complete transaction.

        Writes three files in order: ``.pt``, ``.sha256``, and ``.complete``.
        The ``.complete`` marker is always created last; a checkpoint is only
        considered valid when all three files exist and the hash matches.

        The checkpoint stores student, autoencoder, and normalizer states,
        optimizer state, current step, RNG states, a compact
        :class:`NormalizerAccumulator` sufficient-statistics state (not the
        full feature list), and a compatibility block for resume validation.

        Args:
            path: Destination checkpoint file path (``.pt``).
            step: Number of completed optimizer updates.
            optimizer: The current optimizer whose state to persist.
            normalizer_accumulator: Welford accumulator for normalizer
                statistics.
            compatibility_info: Optional dict with keys ``category``,
                ``seed``, ``split_fingerprint``.

        Raises:
            CheckpointError: If networks are missing or the atomic write
                fails.
        """

        if self.student is None or self.autoencoder is None:
            raise CheckpointError("Cannot save checkpoint: networks not initialized.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rng_state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        }

        compatibility: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "config_fingerprint": self._compute_config_fingerprint(),
            "teacher_hash": self._compute_teacher_state_hash(),
            "is_fixture": self.config.is_fixture,
            "teacher_channels": self.config.teacher_channels,
            "ae_channels": self.config.ae_channels,
            "input_size": list(self.config.input_size),
            "num_steps": self.config.num_steps,
        }
        if compatibility_info is not None:
            compatibility["category"] = compatibility_info.get("category")
            compatibility["seed"] = compatibility_info.get("seed")
            compatibility["split_fingerprint"] = compatibility_info.get("split_fingerprint")

        checkpoint: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "step": step,
            "created_at": datetime.now(UTC).isoformat(),
            "compatibility": compatibility,
            "model_state": {
                "student": self.student.state_dict(),
                "autoencoder": self.autoencoder.state_dict(),
                "normalizer": self.normalizer.state_dict(),
                "normalizer_accumulator": normalizer_accumulator.state_dict(),
            },
            "optimizer_state": optimizer.state_dict(),
            "rng_state": rng_state,
        }

        hash_path = path.with_suffix(".pt.sha256")
        complete_path = path.with_suffix(".pt.complete")

        # --- Phase 1: write .pt to a temp file, then atomically replace. ---
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=".ckpt_tmp_",
            suffix=".pt",
            dir=str(path.parent),
        )
        tmp_pt = Path(tmp_name)
        tmp_hash: Path | None = None
        try:
            os.close(tmp_fd)
            torch.save(checkpoint, tmp_pt)
            file_hash = _sha256(tmp_pt)

            # --- Phase 2: write hash to a temp file, then atomically replace. ---
            tmp_fd2, tmp_name2 = tempfile.mkstemp(
                prefix=".ckpt_tmp_",
                suffix=".sha256",
                dir=str(path.parent),
            )
            tmp_hash = Path(tmp_name2)
            os.close(tmp_fd2)
            tmp_hash.write_text(f"{file_hash}  {path.name}\n", encoding="utf-8")

            # Atomically place .pt and .sha256.
            tmp_pt.replace(path)
            tmp_hash.replace(hash_path)
            tmp_hash = None  # successfully placed

            # --- Phase 3: create .complete marker (must be last). ---
            tmp_fd3, tmp_name3 = tempfile.mkstemp(
                prefix=".ckpt_tmp_",
                suffix=".complete",
                dir=str(path.parent),
            )
            tmp_complete = Path(tmp_name3)
            os.close(tmp_fd3)
            tmp_complete.write_text(
                f"{file_hash}\n{step}\n{datetime.now(UTC).isoformat()}\n",
                encoding="utf-8",
            )
            tmp_complete.replace(complete_path)
        except Exception:
            # Clean up all temp files on failure; existing checkpoints
            # remain untouched because we only replace at the end.
            for tmp in (tmp_pt, tmp_hash, complete_path):
                if tmp is not None and tmp.exists():
                    with contextlib.suppress(OSError):
                        tmp.unlink()
            # Also clean up tmp_complete if it was created.
            for tmp_name_extra in Path(path.parent).glob(".ckpt_tmp_*.complete"):
                with contextlib.suppress(OSError):
                    tmp_name_extra.unlink()
            raise

        _logger.info(
            "Checkpoint saved: %s (step=%d, hash=%s...)",
            path,
            step,
            file_hash[:12],
        )

    def _validate_checkpoint_file(
        self,
        path: Path,
        compatibility_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate a checkpoint file without applying any state.

        Performs hash verification, schema/version checks, and compatibility
        checks.  Does **not** modify model weights, optimizer state, or RNG
        state.  This is used by auto-resume to safely probe multiple
        candidates.

        Args:
            path: Checkpoint file path.
            compatibility_info: Optional compatibility requirements.

        Returns:
            The loaded checkpoint dictionary if all checks pass.

        Raises:
            CheckpointError: If any check fails.
        """

        path = Path(path)
        if not path.is_file():
            raise CheckpointError(f"Checkpoint file not found: {path}.")

        # Verify .complete marker exists.
        complete_path = path.with_suffix(".pt.complete")
        if not complete_path.is_file():
            raise CheckpointError(
                f"Checkpoint complete marker missing: {complete_path}. "
                "Refusing to load incomplete checkpoint."
            )

        # Verify hash sidecar exists.
        hash_path = path.with_suffix(".pt.sha256")
        if not hash_path.is_file():
            raise CheckpointError(
                f"Checkpoint hash sidecar missing: {hash_path}. "
                "Refusing to load unverified checkpoint."
            )
        expected_hash_line = hash_path.read_text(encoding="utf-8").strip()
        expected_hash = expected_hash_line.split()[0] if expected_hash_line else ""
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise CheckpointError(
                f"Checkpoint hash mismatch for {path}: "
                f"expected {expected_hash[:12]}..., got {actual_hash[:12]}..."
            )

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise CheckpointError(f"Cannot load checkpoint {path}: {exc}") from exc

        if not isinstance(checkpoint, dict):
            raise CheckpointError(f"Checkpoint {path} is not a dictionary.")

        ckpt_version = checkpoint.get("checkpoint_version", 0)
        if ckpt_version != CHECKPOINT_VERSION:
            raise CheckpointError(
                f"Checkpoint version mismatch: expected {CHECKPOINT_VERSION}, got {ckpt_version}."
            )

        # Compatibility checks.
        stored = checkpoint.get("compatibility", {})
        expected_fingerprint = self._compute_config_fingerprint()
        if stored.get("config_fingerprint") != expected_fingerprint:
            raise CheckpointError(
                "Config fingerprint mismatch: the checkpoint was created with a "
                "different model architecture or training configuration."
            )
        expected_teacher_hash = self._compute_teacher_state_hash()
        if stored.get("teacher_hash") != expected_teacher_hash:
            raise CheckpointError(
                "Teacher weight hash mismatch: the frozen teacher weights differ "
                "from those used when the checkpoint was created."
            )
        if compatibility_info is not None:
            for field in ("category", "seed", "split_fingerprint"):
                expected_val = compatibility_info.get(field)
                stored_val = stored.get(field)
                if expected_val != stored_val:
                    raise CheckpointError(
                        f"Compatibility check failed for {field!r}: "
                        f"expected {expected_val!r}, got {stored_val!r}."
                    )
        return checkpoint

    def load_checkpoint(
        self,
        path: Path,
        *,
        optimizer: torch.optim.Optimizer,
        compatibility_info: dict[str, Any] | None = None,
    ) -> tuple[int, NormalizerAccumulator]:
        """Load a checkpoint and restore all training states.

        Validates the checkpoint (hash, schema, compatibility) and then
        restores student, autoencoder, and normalizer weights, optimizer
        state, RNG states, and the normalizer accumulator.

        Args:
            path: Checkpoint file path (``.pt``).
            optimizer: The optimizer whose state to restore.
            compatibility_info: Optional dict with ``category``, ``seed``,
                and ``split_fingerprint`` for cross-run validation.

        Returns:
            A tuple of (restored_step, normalizer_accumulator).

        Raises:
            CheckpointError: If any validation or restoration step fails.
        """

        checkpoint = self._validate_checkpoint_file(Path(path), compatibility_info)
        actual_hash = _sha256(Path(path))

        # Restore model states.
        model_state = checkpoint.get("model_state", {})
        if self.student is not None and "student" in model_state:
            self.student.load_state_dict(model_state["student"])
        if self.autoencoder is not None and "autoencoder" in model_state:
            self.autoencoder.load_state_dict(model_state["autoencoder"])
        if "normalizer" in model_state:
            self.normalizer = FeatureNormalizer.from_state_dict(model_state["normalizer"])

        # Restore normalizer accumulator.
        acc_state = model_state.get("normalizer_accumulator")
        if acc_state is None:
            raise CheckpointError(
                "Checkpoint missing normalizer_accumulator state. "
                "The checkpoint may be from an older version."
            )
        accumulator = NormalizerAccumulator.from_state_dict(acc_state, device=self.device)

        # Restore optimizer state.
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Restore RNG states (must be on CPU for torch.set_rng_state).
        rng_state = checkpoint.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
            torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if (
            "torch_cuda" in rng_state
            and rng_state["torch_cuda"] is not None
            and torch.cuda.is_available()
        ):
            cuda_rng = rng_state["torch_cuda"]
            cuda_rng = [t.cpu() for t in cuda_rng] if isinstance(cuda_rng, list) else cuda_rng.cpu()
            torch.cuda.set_rng_state_all(cuda_rng)

        restored_step = int(checkpoint.get("step", 0))
        _logger.info(
            "Checkpoint restored: %s (step=%d, hash=%s...)",
            path,
            restored_step,
            actual_hash[:12],
        )
        return restored_step, accumulator

    @staticmethod
    def _list_checkpoint_candidates(checkpoint_dir: Path) -> list[tuple[int, Path]]:
        """List checkpoint candidates sorted by step descending.

        A candidate must have:
        - filename matching ``checkpoint_step_XXXXXXXX.pt``
        - a corresponding ``.sha256`` sidecar
        - a corresponding ``.complete`` marker

        Temporary files (``.ckpt_tmp_*``) are never included.

        Args:
            checkpoint_dir: Directory to scan.

        Returns:
            A list of (step, path) tuples sorted by step descending.
        """

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            return []
        candidates: list[tuple[int, Path]] = []
        for entry in checkpoint_dir.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.startswith("checkpoint_step_") or not entry.name.endswith(".pt"):
                continue
            # Ensure both sidecars exist.
            if not entry.with_suffix(".pt.sha256").is_file():
                continue
            if not entry.with_suffix(".pt.complete").is_file():
                continue
            stem = entry.stem  # "checkpoint_step_00000006"
            try:
                step = int(stem.replace("checkpoint_step_", ""))
            except ValueError:
                continue
            candidates.append((step, entry))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    @staticmethod
    def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
        """Find the latest checkpoint candidate by structural existence only.

        This method performs a quick file-existence scan (``.pt`` +
        ``.sha256`` + ``.complete``) without hash or compatibility
        verification.  Use :meth:`_auto_resume_from_latest` for full
        validation with fallback.

        Args:
            checkpoint_dir: Directory to scan.

        Returns:
            The path to the latest candidate, or None if none found.
        """

        candidates = EfficientADDetector._list_checkpoint_candidates(checkpoint_dir)
        return candidates[0][1] if candidates else None

    def _auto_resume_from_latest(
        self,
        checkpoint_dir: Path,
        *,
        optimizer: torch.optim.Optimizer,
        compatibility_info: dict[str, Any] | None,
    ) -> tuple[int, NormalizerAccumulator | None, Path | None]:
        """Attempt to resume from the latest valid checkpoint with fallback.

        Scans ``checkpoint_dir`` for checkpoint candidates sorted by step
        descending.  For each candidate, validates hash, schema, and
        compatibility.  The first valid candidate is loaded and its state
        applied.  If a candidate fails validation, the reason is logged and
        the next candidate is tried.

        Args:
            checkpoint_dir: Directory containing checkpoints.
            optimizer: The optimizer to restore state into.
            compatibility_info: Compatibility requirements.

        Returns:
            A tuple of (start_step, accumulator, checkpoint_path).
            If no valid checkpoint is found, returns (0, None, None).
        """

        candidates = self._list_checkpoint_candidates(Path(checkpoint_dir))
        for _step, path in candidates:
            try:
                restored_step, accumulator = self.load_checkpoint(
                    path,
                    optimizer=optimizer,
                    compatibility_info=compatibility_info,
                )
                _logger.info(
                    "Auto-resume selected checkpoint %s (step=%d).",
                    path,
                    restored_step,
                )
                return restored_step, accumulator, path
            except CheckpointError as exc:
                _logger.warning("Auto-resume rejected checkpoint %s: %s", path, exc)
                continue
        _logger.info(
            "Auto-resume found no valid checkpoint in %s; starting from step 0.",
            checkpoint_dir,
        )
        return 0, None, None

    @staticmethod
    def _cleanup_old_checkpoints(checkpoint_dir: Path, max_to_keep: int) -> None:
        """Delete old checkpoints beyond ``max_to_keep``.

        Removes the ``.pt``, ``.sha256``, and ``.complete`` files for each
        evicted checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
            max_to_keep: Maximum number of checkpoint triplets to retain.
        """

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir() or max_to_keep <= 0:
            return
        pairs: list[tuple[int, Path]] = []
        for entry in checkpoint_dir.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.startswith("checkpoint_step_") or not entry.name.endswith(".pt"):
                continue
            stem = entry.stem
            try:
                step = int(stem.replace("checkpoint_step_", ""))
            except ValueError:
                continue
            pairs.append((step, entry))
        pairs.sort(key=lambda item: item[0], reverse=True)
        for _step, ckpt_path in pairs[max_to_keep:]:
            hash_path = ckpt_path.with_suffix(".pt.sha256")
            complete_path = ckpt_path.with_suffix(".pt.complete")
            ckpt_path.unlink(missing_ok=True)
            hash_path.unlink(missing_ok=True)
            complete_path.unlink(missing_ok=True)
            _logger.debug("Removed old checkpoint: %s", ckpt_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: object,
        *,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 10000,
        resume_from: Path | None = None,
        auto_resume: bool = True,
        max_to_keep: int = 3,
        compatibility_info: dict[str, Any] | None = None,
    ) -> None:
        """Train the student and autoencoder on normal images.

        When ``checkpoint_dir`` is provided, checkpoints are saved at every
        ``checkpoint_interval`` steps (and at the final step) using atomic
        writes with hash sidecars.  When ``resume_from`` is set or
        ``auto_resume`` is True and a checkpoint exists in
        ``checkpoint_dir``, training resumes from the stored step without
        repeating completed steps.

        Args:
            train_loader: Iterable of batches with ``image`` and ``label``
                fields. Labels must all be 0 (normal).
            checkpoint_dir: Directory for periodic checkpoint saves.
            checkpoint_interval: Steps between checkpoint saves.
            resume_from: Explicit checkpoint path to resume from.  Takes
                priority over ``auto_resume``.
            auto_resume: Whether to auto-resume from the latest checkpoint
                found in ``checkpoint_dir``.
            max_to_keep: Maximum checkpoint files to retain in
                ``checkpoint_dir``.
            compatibility_info: Optional dict with ``category``, ``seed``,
                and ``split_fingerprint`` for cross-run validation.

        Raises:
            DetectorError: If training fails, labels indicate leakage, or
                the loader is empty.
            CheckpointError: If a resume checkpoint cannot be loaded or
                fails compatibility checks.
        """

        if not isinstance(train_loader, Iterable):
            raise DetectorError("fit() expected an iterable train loader.")
        if self.student is None or self.autoencoder is None or self.teacher is None:
            raise DetectorError("Networks not initialized.")

        loss_fn = EfficientADLoss(lambda_global=self.config.lambda_global)
        optimizer = torch.optim.Adam(
            list(self.student.parameters()) + list(self.autoencoder.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Collect all training images for iteration.
        train_images: list[Tensor] = []
        for batch in train_loader:
            images = self._batch_field(batch, "image")
            labels = self._batch_field(batch, "label")
            if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
                raise DetectorError("Training image and label fields must be tensors.")
            if labels.numel() != images.shape[0]:
                raise DetectorError("Training labels do not match the image batch size.")
            if torch.any(labels != 0):
                raise DetectorError(
                    "Training leakage risk: EfficientAD fit accepts normal labels only."
                )
            train_images.append(images)

        if not train_images:
            raise DetectorError("Cannot train EfficientAD from an empty train loader.")

        all_images = torch.cat(train_images, dim=0).to(self.device)
        dataset_size = all_images.shape[0]
        batch_size = min(self.config.batch_size, dataset_size)

        # Determine start step via resume.
        start_step = 0
        normalizer_accumulator = NormalizerAccumulator(
            self.config.teacher_channels, device=self.device
        )

        if resume_from is not None:
            # Explicit resume: no fallback, must succeed or fail clearly.
            start_step, normalizer_accumulator = self.load_checkpoint(
                Path(resume_from),
                optimizer=optimizer,
                compatibility_info=compatibility_info,
            )
            _logger.info(
                "Resumed EfficientAD training from %s at step %d.",
                resume_from,
                start_step,
            )
        elif auto_resume and checkpoint_dir is not None:
            # Auto-resume with fallback to previous valid checkpoint.
            start_step, restored_acc, _ = self._auto_resume_from_latest(
                Path(checkpoint_dir),
                optimizer=optimizer,
                compatibility_info=compatibility_info,
            )
            if restored_acc is not None:
                normalizer_accumulator = restored_acc

        if start_step < self.config.num_steps:
            # Training loop.
            self.student.train()
            self.autoencoder.train()
            for step in range(start_step, self.config.num_steps):
                # Sample a random batch.
                indices = torch.randint(0, dataset_size, (batch_size,), device=self.device)
                batch_images = all_images[indices]

                with torch.inference_mode():
                    teacher_output = self.teacher(batch_images)

                student_output = self.student(batch_images)
                student_match, ae_input = self.student.split_output(student_output)
                ae_reconstruction = self.autoencoder(ae_input)

                losses = loss_fn(student_match, teacher_output, ae_reconstruction, ae_input)
                loss = losses["total"]

                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

                if step == 0 or (step + 1) % max(1, self.config.num_steps // 10) == 0:
                    _logger.debug(
                        "EfficientAD training step %d/%d, loss=%.6f",
                        step + 1,
                        self.config.num_steps,
                        float(loss),
                    )

                # Update normalizer accumulator (first epoch only).
                if step < dataset_size // batch_size + 1:
                    normalizer_accumulator.update(teacher_output)

                # Save checkpoint at intervals.
                if checkpoint_dir is not None and checkpoint_interval > 0:
                    should_save = (step + 1) % checkpoint_interval == 0 or (
                        step + 1
                    ) == self.config.num_steps
                    if should_save:
                        ckpt_path = Path(checkpoint_dir) / f"checkpoint_step_{step + 1:08d}.pt"
                        self.save_checkpoint(
                            ckpt_path,
                            step=step + 1,
                            optimizer=optimizer,
                            normalizer_accumulator=normalizer_accumulator,
                            compatibility_info=compatibility_info,
                        )
                        self._cleanup_old_checkpoints(Path(checkpoint_dir), max_to_keep)

        # Build feature normalizer from accumulated sufficient statistics.
        self.normalizer = normalizer_accumulator.to_normalizer()

        self.student.eval()
        self.autoencoder.eval()
        self.is_fitted = True
        self.build_metadata = {
            "train_samples": dataset_size,
            "train_steps": self.config.num_steps,
            "teacher_channels": self.config.teacher_channels,
            "ae_channels": self.config.ae_channels,
            "student_parameters": sum(p.numel() for p in self.student.parameters()),
            "autoencoder_parameters": sum(p.numel() for p in self.autoencoder.parameters()),
            "teacher_frozen": True,
            "normalizer_fit": self.normalizer.is_fit,
            "resumed_from_step": start_step,
        }

    @torch.inference_mode()
    def predict(self, images: Tensor) -> PredictionBatch:
        """Predict image scores and pixel-level anomaly maps.

        Args:
            images: RGB tensor shaped ``[B, 3, H, W]``.

        Returns:
            A PredictionBatch with image scores, anomaly maps, and metadata.

        Raises:
            DetectorError: If the detector is not fitted or prediction fails.
        """

        if not self.is_fitted:
            raise DetectorError("EfficientAD must be fitted or loaded before predict().")
        if self.teacher is None or self.student is None or self.autoencoder is None:
            raise DetectorError("Networks not available for prediction.")
        if images.ndim != 4 or images.shape[1] != 3:
            raise DetectorError(
                f"Predict expects [B, 3, H, W] input, got shape {tuple(images.shape)}."
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        images = images.to(self.device)
        batch_size = images.shape[0]
        _, _, height, width = images.shape

        teacher_output = self.teacher(images)
        student_output = self.student(images)
        student_match, ae_input = self.student.split_output(student_output)
        ae_reconstruction = self.autoencoder(ae_input)

        # Normalize features for stable anomaly map computation.
        teacher_norm = self.normalizer.normalize(teacher_output)
        student_match_norm = self.normalizer.normalize(student_match)

        # Local anomaly: per-channel absolute difference, then mean across channels.
        local_diff = (student_match_norm - teacher_norm).abs()
        local_map = local_diff.mean(dim=1, keepdim=True)  # [B, 1, H', W']

        # Global anomaly: per-channel absolute difference, then mean across channels.
        global_diff = (ae_reconstruction - ae_input).abs()
        global_map = global_diff.mean(dim=1, keepdim=True)  # [B, 1, H', W']

        # Fuse maps: weighted combination.
        fused_map = self.config.local_weight * local_map + self.config.global_weight * global_map

        # Upsample to input resolution.
        fused_map = functional.interpolate(
            fused_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # Apply optional Gaussian smoothing.
        if self.config.smoothing_sigma > 0:
            fused_map = self._apply_gaussian_smoothing(fused_map)

        anomaly_maps = fused_map[:, 0]  # [B, H, W]

        # Image score: max pixel value.
        image_scores = anomaly_maps.flatten(1).max(dim=1).values

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Apply threshold for predicted labels.
        threshold = self.config.image_threshold
        predicted_labels = (image_scores > threshold).long() if threshold is not None else None

        scope = FIXTURE_SCOPE if self.config.is_fixture else REAL_SCOPE
        metadata: list[dict[str, Any]] = []
        for _index in range(batch_size):
            metadata.append(
                {
                    "artifact_scope": scope,
                    "teacher_channels": self.config.teacher_channels,
                    "ae_channels": self.config.ae_channels,
                    "inference_time_ms": elapsed_ms / batch_size,
                    "threshold_source": self.config.threshold_source,
                }
            )

        return PredictionBatch(
            image_scores=image_scores.cpu(),
            anomaly_maps=anomaly_maps.cpu(),
            predicted_labels=predicted_labels.cpu() if predicted_labels is not None else None,
            threshold=threshold,
            inference_times_ms=[elapsed_ms / batch_size] * batch_size,
            metadata=metadata,
        )

    def _apply_gaussian_smoothing(self, maps: Tensor) -> Tensor:
        """Apply Gaussian smoothing to anomaly maps.

        Args:
            maps: Tensor shaped ``[B, 1, H, W]``.

        Returns:
            Smoothed maps with the same shape.
        """

        import math

        sigma = self.config.smoothing_sigma
        radius = max(1, math.ceil(3.0 * sigma))
        coordinates = torch.arange(
            -radius,
            radius + 1,
            device=maps.device,
            dtype=maps.dtype,
        )
        kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, *kernel_2d.shape)
        padding = kernel.shape[-1] // 2
        return functional.conv2d(maps, kernel, padding=padding)

    def save(self, output_dir: Path) -> None:
        """Save the complete model with hash validation.

        Args:
            output_dir: Destination directory (must not exist).

        Raises:
            DetectorError: If saving fails or the model is not fitted.
        """

        if not self.is_fitted:
            raise DetectorError("Cannot save an unfitted EfficientAD model.")
        if self.student is None or self.autoencoder is None or self.teacher is None:
            raise DetectorError("Networks not available for saving.")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save teacher weights (frozen but needed for exact reproducibility).
        teacher_path = output_dir / "teacher.pt"
        torch.save(self.teacher.state_dict(), teacher_path)

        # Save student weights.
        student_path = output_dir / "student.pt"
        torch.save(self.student.state_dict(), student_path)

        # Save autoencoder weights.
        ae_path = output_dir / "autoencoder.pt"
        torch.save(self.autoencoder.state_dict(), ae_path)

        # Save normalizer.
        norm_path = output_dir / "normalizer.pt"
        torch.save(self.normalizer.state_dict(), norm_path)

        # Save resolved config.
        config_path = output_dir / "config.resolved.yaml"
        config_path.write_text(
            yaml.safe_dump(self.config.to_dict(), sort_keys=True),
            encoding="utf-8",
        )

        # Compute hashes for integrity validation.
        hashes = {
            "teacher": _sha256(teacher_path),
            "student": _sha256(student_path),
            "autoencoder": _sha256(ae_path),
            "normalizer": _sha256(norm_path),
        }

        # Save metadata.
        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "shiftad_version": __version__,
            "git_commit": _git_commit(),
            "model_type": "efficientad",
            "is_fixture": self.config.is_fixture,
            "teacher_channels": self.config.teacher_channels,
            "ae_channels": self.config.ae_channels,
            "input_size": list(self.config.input_size),
            "hashes": hashes,
            "build": self.build_metadata,
            "real_pretrained_teacher": (
                not self.config.is_fixture and self.config.teacher_pretrained
            ),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Save manifest for factory dispatch.
        manifest = {
            "model_type": "efficientad",
            "artifact_version": ARTIFACT_VERSION,
            "is_fixture": self.config.is_fixture,
            "real_pretrained_teacher": metadata["real_pretrained_teacher"],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, model_dir: Path, *, device: torch.device | str = "cpu") -> EfficientADDetector:
        """Load a saved EfficientAD model with hash validation.

        Args:
            model_dir: Directory produced by ``save()``.
            device: Target device.

        Returns:
            A loaded and verified EfficientADDetector.

        Raises:
            DetectorError: If artifacts are missing, hashes mismatch, or
                loading fails.
        """

        manifest_path = model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise DetectorError(f"Manifest missing: {manifest_path}.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DetectorError(f"Cannot read manifest: {exc}") from exc

        if manifest.get("model_type") != "efficientad":
            raise DetectorError(
                f"Manifest model_type is {manifest.get('model_type')!r}, expected 'efficientad'."
            )

        # Load metadata and verify hashes.
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.is_file():
            raise DetectorError(f"Metadata missing: {metadata_path}.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DetectorError(f"Cannot read metadata: {exc}") from exc

        hashes = metadata.get("hashes", {})
        for name in ("teacher", "student", "autoencoder", "normalizer"):
            artifact = model_dir / f"{name}.pt"
            if not artifact.is_file():
                raise DetectorError(f"Missing artifact: {artifact}.")
            if name not in hashes:
                raise DetectorError(
                    f"Hash-mismatched artifact {name}.pt: expected hash missing from metadata."
                )
            actual_hash = _sha256(artifact)
            if actual_hash != hashes[name]:
                raise DetectorError(
                    f"Hash-mismatched artifact {name}.pt: "
                    f"expected {hashes[name]}, got {actual_hash}."
                )

        # Load resolved config.
        config_path = model_dir / "config.resolved.yaml"
        if not config_path.is_file():
            raise DetectorError(f"Config missing: {config_path}.")
        saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        # Reconstruct config.
        ckpt_raw = saved_config.get("checkpoint", {})
        config = EfficientADConfig(
            is_fixture=saved_config.get("is_fixture", False),
            teacher_pretrained=saved_config.get("teacher", {}).get("pretrained", True),
            teacher_checkpoint=(
                Path(saved_config["teacher"]["checkpoint"])
                if saved_config.get("teacher", {}).get("checkpoint")
                else None
            ),
            teacher_allow_download=saved_config.get("teacher", {}).get("allow_download", False),
            teacher_use_cache=saved_config.get("teacher", {}).get("use_cache", True),
            teacher_channels=saved_config.get("teacher", {}).get("out_channels", 128),
            ae_channels=saved_config.get("student", {}).get("ae_channels", 128),
            input_size=tuple(saved_config.get("input_size", [256, 256])),
            learning_rate=saved_config.get("training", {}).get("learning_rate", 1e-3),
            weight_decay=saved_config.get("training", {}).get("weight_decay", 1e-5),
            num_steps=saved_config.get("training", {}).get("num_steps", 70000),
            batch_size=saved_config.get("training", {}).get("batch_size", 4),
            lambda_global=saved_config.get("training", {}).get("lambda_global", 1.0),
            local_weight=saved_config.get("scoring", {}).get("local_weight", 1.0),
            global_weight=saved_config.get("scoring", {}).get("global_weight", 1.0),
            smoothing_sigma=saved_config.get("scoring", {}).get("smoothing_sigma", 4.0),
            image_threshold=saved_config.get("scoring", {}).get("image_threshold"),
            pixel_threshold=saved_config.get("scoring", {}).get("pixel_threshold"),
            threshold_source=saved_config.get("scoring", {}).get("threshold_source"),
            checkpoint_enabled=ckpt_raw.get("enabled", True),
            checkpoint_interval=ckpt_raw.get("interval", 10000),
            checkpoint_dir=(Path(ckpt_raw["dir"]) if ckpt_raw.get("dir") else None),
            resume_from=(Path(ckpt_raw["resume_from"]) if ckpt_raw.get("resume_from") else None),
            auto_resume=ckpt_raw.get("auto_resume", True),
            max_to_keep=ckpt_raw.get("max_to_keep", 3),
        )

        detector = cls(config, device=device)

        # Load teacher weights (frozen, needed for exact reproducibility).
        if detector.teacher is not None:
            detector.teacher.load_state_dict(
                torch.load(
                    model_dir / "teacher.pt", map_location=detector.device, weights_only=True
                )
            )
            detector.teacher.to(detector.device)
            detector.teacher.eval()
            detector.teacher.requires_grad_(False)

        # Load student weights.
        if detector.student is not None:
            detector.student.load_state_dict(
                torch.load(
                    model_dir / "student.pt", map_location=detector.device, weights_only=True
                )
            )
            detector.student.to(detector.device)
            detector.student.eval()
        if detector.autoencoder is not None:
            detector.autoencoder.load_state_dict(
                torch.load(
                    model_dir / "autoencoder.pt", map_location=detector.device, weights_only=True
                )
            )
            detector.autoencoder.to(detector.device)
            detector.autoencoder.eval()

        # Load normalizer.
        norm_state = torch.load(
            model_dir / "normalizer.pt",
            map_location=detector.device,
            weights_only=True,
        )
        detector.normalizer = FeatureNormalizer.from_state_dict(norm_state)

        detector.is_fitted = True
        detector.build_metadata = metadata.get("build", {})
        return detector
