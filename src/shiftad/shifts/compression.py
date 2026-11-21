"""Compression shift families: jpeg_compression and resolution_reduction.

``jpeg_compression`` is deterministic (no seed consumption) and Level 0
bypasses the encode/decode cycle entirely (protocol Section 3.9).
``resolution_reduction`` downscales and upscales with bilinear interpolation
for the image and nearest-neighbor for the mask (protocol Section 3.10).
"""

from __future__ import annotations

import io
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from shiftad.shifts.base import (
    Shift,
    ShiftError,
    ShiftResult,
    _validate_image,
    _validate_mask,
    identity_result,
    re_binarize,
)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] image to uint8 for PIL encoding."""

    return np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)


def _from_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a uint8 RGB array back to float32 [0, 1]."""

    return array.astype(np.float32) / 255.0


class JpegCompressionShift(Shift):
    """Encode-decode a JPEG cycle at a frozen quality (protocol Section 3.9)."""

    name = "jpeg_compression"
    variant = "cycle"
    _QUALITIES: ClassVar[dict[int, int | None]] = {0: None, 1: 75, 2: 50, 3: 30, 4: 15}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._QUALITIES:
            raise ShiftError(f"Unsupported jpeg_compression severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._QUALITIES:
            raise ShiftError(f"Unsupported jpeg_compression severity {severity}.")
        return {"quality": self._QUALITIES[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        quality = parameters["quality"]
        if quality is None:
            # Level 0 bypass: the image bytes never pass through JPEG.
            return identity_result(image, mask, parameters)
        uint8 = _to_uint8(image)
        buffer = io.BytesIO()
        Image.fromarray(uint8, mode="RGB").save(
            buffer,
            format="JPEG",
            quality=quality,
        )
        buffer.seek(0)
        decoded = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
        shifted = _from_uint8(decoded)
        return ShiftResult(image=shifted, mask=mask, exact_parameters=parameters)


class ResolutionReductionShift(Shift):
    """Downscale-upscale cycle with frozen scale factors (protocol 3.10)."""

    name = "resolution_reduction"
    variant = "downscale_upscale"
    _FACTORS: ClassVar[dict[int, float]] = {0: 1.00, 1: 0.75, 2: 0.50, 3: 0.33, 4: 0.20}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._FACTORS:
            raise ShiftError(f"Unsupported resolution_reduction severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._FACTORS:
            raise ShiftError(f"Unsupported resolution_reduction severity {severity}.")
        return {"scale_factor": self._FACTORS[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        factor = parameters["scale_factor"]
        if factor == 1.0:
            return identity_result(image, mask, parameters)
        height, width = image.shape[0], image.shape[1]
        small_size = (max(1, round(width * factor)), max(1, round(height * factor)))
        uint8 = _to_uint8(image)
        pil = Image.fromarray(uint8, mode="RGB")
        small = pil.resize(small_size, resample=Image.Resampling.BILINEAR)
        restored = small.resize((width, height), resample=Image.Resampling.BILINEAR)
        shifted = _from_uint8(np.asarray(restored))
        before = None
        after = None
        visible = None
        shifted_mask = None
        if mask is not None:
            before = int(np.count_nonzero(mask))
            mask_uint8 = (mask * 255.0).round().astype(np.uint8)
            pil_mask = Image.fromarray(mask_uint8, mode="L")
            small_mask = pil_mask.resize(small_size, resample=Image.Resampling.NEAREST)
            restored_mask = small_mask.resize((width, height), resample=Image.Resampling.NEAREST)
            shifted_mask = re_binarize(np.asarray(restored_mask, dtype=np.float32) / 255.0)
            after = int(np.count_nonzero(shifted_mask))
            visible = after > 0
        return ShiftResult(
            image=shifted,
            mask=shifted_mask,
            exact_parameters=parameters,
            mask_area_before=before,
            mask_area_after=after,
            defect_visible_after_shift=visible,
        )
