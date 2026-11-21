"""Frozen-backbone feature extraction and shifted-image pipeline for K0.

K0 instruction Section 11: K0 must use the accepted Phase 3 frozen DINOv2
feature extractor (``dinov2_vits14``, layers ``[2, 5, 8, 11]``, concat + L2
normalization, 224x224 input). The shift is applied at MVTec AD source
resolution (Phase 5 protocol Section 5.6) and the shifted source image is
resized to 224x224 with PIL bilinear before extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from shiftad.backbones.dinov2 import DinoV2Backbone
from shiftad.backbones.feature_extractor import FeatureFuser
from shiftad.data.preprocessing import load_rgb_tensor
from shiftad.route_k.correspondence import GRID, CorrespondenceMap, build_correspondence
from shiftad.route_k.displacement import PairedDisplacement, build_paired_displacements
from shiftad.shifts.registry import apply_shift


class K0FeatureError(RuntimeError):
    """Report invalid feature extraction."""


class K0FeatureExtractor:
    """Frozen DINOv2 patch feature extractor bound to the K0 protocol.

    The official Phase 3 weights are loaded from the local torch-hub cache
    (``<hub>/checkpoints/<model_name>_pretrain.pth``) when no explicit
    checkpoint is configured; downloads are never performed.
    """

    def __init__(
        self,
        *,
        model_name: str,
        layers: tuple[int, ...],
        input_size: tuple[int, int],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_cache: bool = True,
    ) -> None:
        if input_size[0] % 14 or input_size[1] % 14:
            raise K0FeatureError(f"Input size {input_size} must be divisible by the 14px patch.")
        self.backbone = DinoV2Backbone(
            model_name=model_name,
            layers=layers,
            device=device,
            dtype=dtype,
            checkpoint=self._resolve_checkpoint(model_name),
            allow_download=False,
            use_cache=use_cache,
            expected_input_size=input_size,
        )
        self.fuser = FeatureFuser(mode="concat", normalize=True, target_grid=None)
        self.input_size = input_size
        self.checkpoint_path = self._resolve_checkpoint(model_name)

    @staticmethod
    def _resolve_checkpoint(model_name: str) -> Path | None:
        """Return the cached official checkpoint path or ``None``.

        Looks for ``<torch hub dir>/checkpoints/<model_name>_pretrain.pth``,
        which is the file the official torch-hub entrypoint uses for
        ``pretrained=True``. Never downloads.
        """
        import torch as torch_module

        hub_dir = Path(torch_module.hub.get_dir())
        candidate = hub_dir / "checkpoints" / f"{model_name}_pretrain.pth"
        if candidate.is_file():
            return candidate
        raise K0FeatureError(
            f"Official checkpoint not found in the torch-hub cache: {candidate}. "
            "K0 requires the frozen Phase 3 DINOv2 weights and never downloads."
        )

    @torch.inference_mode()
    def extract(self, image_rgb: np.ndarray) -> Tensor:
        """Extract fused patch features for one float32 HWC image in [0, 1].

        Returns ``[256, D]`` features on CPU.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
            raise K0FeatureError(f"Expected an HWC RGB image, got {image_rgb.shape}.")
        height, width = image_rgb.shape[:2]
        if (height, width) != self.input_size:
            raise K0FeatureError(f"Expected input {self.input_size}, got {(height, width)}.")
        tensor = torch.from_numpy(np.asarray(image_rgb, dtype=np.float32))
        tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
        batch = self.backbone.extract_features(tensor)
        fused = self.fuser(
            layer_features=batch.layer_features,
            patch_grid_shapes=batch.patch_grid_shapes,
        )
        if fused.grid_shape != (GRID, GRID):
            raise K0FeatureError(f"Unexpected patch grid {fused.grid_shape}.")
        return fused.features[0].detach().cpu()

    def extract_image_path(self, path: Path) -> Tensor:
        """Load, resize, and extract features from an image file."""
        height, width = self.input_size
        image = load_image_path(path, (height, width))
        return self.extract(image)

    def extract_shifted_path(
        self,
        path: Path,
        *,
        shift_name: str,
        severity: int,
        derived_seed: int,
    ) -> tuple[Tensor, dict[str, float]]:
        """Apply one shift at source resolution, resize, and extract features.

        Returns ``(features, exact_parameters)``.
        """
        image = load_source_image(path)
        result = apply_shift(
            shift_name,
            severity,
            image,
            None,
            derived_seed,
        )
        height, width = self.input_size
        resized = (
            np.asarray(
                Image.fromarray(
                    (np.clip(result.image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                ).resize((width, height), resample=Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
        parameters = {
            key: float(value)
            for key, value in result.exact_parameters.items()
            if isinstance(value, (int, float))
        }
        return self.extract(resized), parameters


def load_source_image(path: Path) -> np.ndarray:
    """Load an image at its native resolution as float32 HWC in [0, 1]."""
    with Image.open(path) as source:
        source.load()
        rgb = source.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32).copy() / 255.0
    return array


def load_image_path(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load an image resized to ``image_size`` as float32 HWC in [0, 1]."""
    tensor, _ = load_rgb_tensor(path, image_size)
    return tensor.permute(1, 2, 0).numpy()


def shifted_correspondence(
    shift_name: str,
    parameters: dict[str, float],
    source_size: tuple[int, int],
    *,
    tolerance_px: float,
) -> CorrespondenceMap:
    """Build the oracle correspondence map for one shifted image."""
    height, width = source_size
    return build_correspondence(
        shift_name,
        parameters,
        height,
        width,
        tolerance_px=tolerance_px,
    )


def paired_displacements_for_image(
    clean_features: Tensor,
    shifted_features: Tensor,
    correspondence: CorrespondenceMap,
    *,
    raw_identity: str,
    category: str,
    shift_family: str,
    severity: int,
    transform_seed: int,
) -> list[PairedDisplacement]:
    """Build the per-patch displacement records for one image pair."""
    return build_paired_displacements(
        clean_features,
        shifted_features,
        correspondence,
        raw_identity=raw_identity,
        category=category,
        shift_family=shift_family,
        severity=severity,
        transform_seed=transform_seed,
    )
