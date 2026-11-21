"""Geometric shift families: rotation, translation, scale_change.

All three families apply the same affine transformation matrix to the image
(bilinear) and the mask (nearest-neighbor), use ``BORDER_CONSTANT`` fill 0,
and re-binarize the mask at 0.5 (protocol Sections 3.11-3.13, 5.2). Pixel
parameters are scaled from the nominal 900x900 reference to the actual source
size per protocol Section 5.6.
"""

from __future__ import annotations

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
    scale_translation,
)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] image to uint8 for PIL transforms."""

    return np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)


def _from_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a uint8 RGB array back to float32 [0, 1]."""

    return array.astype(np.float32) / 255.0


def _rotation_coeffs(angle_degrees: float, height: int, width: int) -> tuple[float, ...]:
    """Return PIL affine coefficients for a center rotation.

    PIL maps each output pixel ``(x, y)`` to input ``(a*x + b*y + c,
    d*x + e*y + f)``; the coefficients invert the forward rotation so the
    image rotates counter-clockwise about its center.
    """

    import math

    theta = math.radians(angle_degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    # PIL maps each output pixel to an input position, so the coefficients are
    # the inverse (transpose) of the forward counter-clockwise rotation matrix
    # [[cos, sin], [-sin, cos]] in image coordinates (y down).
    return (cos, -sin, cx - cos * cx + sin * cy, sin, cos, cy - sin * cx - cos * cy)


def _translation_coeffs(dx: int, dy: int) -> tuple[float, ...]:
    """Return PIL affine coefficients shifting content by ``(dx, dy)``."""

    return (1.0, 0.0, float(-dx), 0.0, 1.0, float(-dy))


def _scale_coeffs(factor: float, height: int, width: int) -> tuple[float, ...]:
    """Return PIL affine coefficients for isotropic scaling about the center."""

    if factor <= 0:
        raise ShiftError(f"Scale factor must be positive, got {factor}.")
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    inv = 1.0 / factor
    return (inv, 0.0, cx - cx * inv, 0.0, inv, cy - cy * inv)


def _apply_affine(
    image: np.ndarray,
    coefficients: tuple[float, ...],
    *,
    resample: Image.Resampling,
) -> np.ndarray:
    """Apply PIL affine coefficients to a float32 RGB image with fill 0."""

    uint8 = _to_uint8(image)
    pil = Image.fromarray(uint8, mode="RGB")
    transformed = pil.transform(
        uint8.shape[1::-1],
        Image.Transform.AFFINE,
        coefficients,
        resample=resample,
        fillcolor=0,
    )
    return _from_uint8(np.asarray(transformed))


class RotationShift(Shift):
    """Rotate about the image center by a frozen positive angle."""

    name = "rotation"
    variant = "ccw"
    _ANGLES: ClassVar[dict[int, float]] = {0: 0.0, 1: 5.0, 2: 10.0, 3: 20.0, 4: 35.0}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._ANGLES:
            raise ShiftError(f"Unsupported rotation severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._ANGLES:
            raise ShiftError(f"Unsupported rotation severity {severity}.")
        return {"angle_degrees": self._ANGLES[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        angle = parameters["angle_degrees"]
        if angle == 0.0:
            return identity_result(image, mask, parameters)
        height, width = image.shape[0], image.shape[1]
        coefficients = _rotation_coeffs(angle, height, width)
        shifted_image = _apply_affine(image, coefficients, resample=Image.Resampling.BILINEAR)
        before = None
        after = None
        visible = None
        shifted_mask = None
        if mask is not None:
            before = int(np.count_nonzero(mask))
            mask_uint8 = (mask * 255.0).round().astype(np.uint8)
            pil_mask = Image.fromarray(mask_uint8, mode="L")
            transformed_mask = pil_mask.transform(
                (width, height),
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
            shifted_mask = re_binarize(np.asarray(transformed_mask, dtype=np.float32) / 255.0)
            after = int(np.count_nonzero(shifted_mask))
            visible = after > 0
        return ShiftResult(
            image=shifted_image,
            mask=shifted_mask,
            exact_parameters=parameters,
            mask_area_before=before,
            mask_area_after=after,
            defect_visible_after_shift=visible,
        )


class TranslationShift(Shift):
    """Translate content by frozen pixel offsets (scaled to source size)."""

    name = "translation"
    variant = "positive"
    _DX: ClassVar[dict[int, int]] = {0: 0, 1: 10, 2: 25, 3: 50, 4: 90}
    _DY: ClassVar[dict[int, int]] = {0: 0, 1: 5, 2: 15, 3: 30, 4: 60}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._DX:
            raise ShiftError(f"Unsupported translation severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._DX:
            raise ShiftError(f"Unsupported translation severity {severity}.")
        return {"dx": self._DX[severity], "dy": self._DY[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        reference = self.parameters_for(self.severity)
        height, width = image.shape[0], image.shape[1]
        dx, dy = scale_translation(reference["dx"], reference["dy"], height, width)
        parameters = dict(reference)
        parameters["dx_realized"] = dx
        parameters["dy_realized"] = dy
        if dx == 0 and dy == 0:
            return identity_result(image, mask, parameters)
        coefficients = _translation_coeffs(dx, dy)
        shifted_image = _apply_affine(image, coefficients, resample=Image.Resampling.BILINEAR)
        before = None
        after = None
        visible = None
        shifted_mask = None
        if mask is not None:
            before = int(np.count_nonzero(mask))
            mask_uint8 = (mask * 255.0).round().astype(np.uint8)
            pil_mask = Image.fromarray(mask_uint8, mode="L")
            transformed_mask = pil_mask.transform(
                (width, height),
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
            shifted_mask = re_binarize(np.asarray(transformed_mask, dtype=np.float32) / 255.0)
            after = int(np.count_nonzero(shifted_mask))
            visible = after > 0
        return ShiftResult(
            image=shifted_image,
            mask=shifted_mask,
            exact_parameters=parameters,
            mask_area_before=before,
            mask_area_after=after,
            defect_visible_after_shift=visible,
        )


class ScaleChangeShift(Shift):
    """Zoom out isotropically about the image center by a frozen factor."""

    name = "scale_change"
    variant = "zoom_out"
    _FACTORS: ClassVar[dict[int, float]] = {0: 1.00, 1: 0.90, 2: 0.80, 3: 0.70, 4: 0.55}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._FACTORS:
            raise ShiftError(f"Unsupported scale_change severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._FACTORS:
            raise ShiftError(f"Unsupported scale_change severity {severity}.")
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
        coefficients = _scale_coeffs(factor, height, width)
        shifted_image = _apply_affine(image, coefficients, resample=Image.Resampling.BILINEAR)
        before = None
        after = None
        visible = None
        shifted_mask = None
        if mask is not None:
            before = int(np.count_nonzero(mask))
            mask_uint8 = (mask * 255.0).round().astype(np.uint8)
            pil_mask = Image.fromarray(mask_uint8, mode="L")
            transformed_mask = pil_mask.transform(
                (width, height),
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
            shifted_mask = re_binarize(np.asarray(transformed_mask, dtype=np.float32) / 255.0)
            after = int(np.count_nonzero(shifted_mask))
            visible = after > 0
        return ShiftResult(
            image=shifted_image,
            mask=shifted_mask,
            exact_parameters=parameters,
            mask_area_before=before,
            mask_area_after=after,
            defect_visible_after_shift=visible,
        )


def affine_mask(
    mask: np.ndarray,
    coefficients: tuple[float, ...],
    height: int,
    width: int,
) -> np.ndarray:
    """Apply nearest-neighbor affine coefficients and re-binarize a mask.

    Exposed for fixture tests that verify mask/image matrix alignment.

    Args:
        mask: Binary float32 mask with shape ``[H, W]``.
        coefficients: PIL affine coefficients.
        height: Output height.
        width: Output width.

    Returns:
        The re-binarized transformed mask.
    """

    _validate_mask(mask)
    mask_uint8 = (mask * 255.0).round().astype(np.uint8)
    pil_mask = Image.fromarray(mask_uint8, mode="L")
    transformed = pil_mask.transform(
        (width, height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return re_binarize(np.asarray(transformed, dtype=np.float32) / 255.0)
