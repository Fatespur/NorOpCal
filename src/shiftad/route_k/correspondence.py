"""Oracle patch correspondence for paired benign-shift displacement.

K0 instruction Section 9: for correspondence-preserving transforms ``p' = p``
when patch geometry supports exact alignment; for geometry-altering transforms
the known synthetic transform is used only for K0 oracle correspondence
recovery. The inverse transform / coordinate mapping must be mathematically
explicit and tested (instruction Section 9, 27).

Pipeline facts used by the mapping:

* The shift is applied at the MVTec AD source resolution ``(H_src, W_src)``
  (Phase 5 protocol Section 5.6).
* The shifted source image is resized to ``(224, 224)`` with PIL bilinear
  (``data/preprocessing.load_rgb_tensor``), then DINOv2-S/14 produces a
  ``16 x 16`` patch grid with patch size 14.
* PIL affine coefficients map output pixels to input pixels
  ``(x_in, y_in) = (a*x + b*y + c, d*x + e*y + f)`` (``shifts/geometric.py``).
* PIL ``resize`` uses inverse pixel-center sampling; we model the inverse
  resize as ``x_src = (x_224 + 0.5) * W_src / 224 - 0.5`` and the forward
  resize as ``x_224 = (x_src + 0.5) * 224 / W_src - 0.5``.

The oracle mapping for a transformed patch ``p'`` is therefore::

    p'_center (224 grid) -> inverse resize -> source pixel
        -> inverse affine -> source pixel (pre-shift)
            -> forward resize -> 224 pixel -> nearest patch index ``p``

``correspondence_valid`` is False when the pre-shift source pixel falls
outside the source image (border fill content) or when the inverse mapping
deviates from the nearest patch center by more than half a patch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PATCH_SIZE = 14
GRID = 16
MODEL_SIZE = 224

# Families whose pixels keep their spatial position (photometric, blur,
# compression, resolution). Resolution reduction re-samples but does not
# displace content; the patch grid alignment is preserved at the model size.
CORRESPONDENCE_PRESERVING = frozenset(
    {
        "brightness",
        "contrast",
        "gamma",
        "color_temperature",
        "shadow",
        "gaussian_noise",
        "jpeg_compression",
        "motion_blur",
        "defocus_blur",
        "resolution_reduction",
    }
)

GEOMETRY_ALTERING = frozenset({"rotation", "translation", "scale_change"})


class CorrespondenceError(RuntimeError):
    """Report an invalid correspondence request."""


@dataclass(frozen=True)
class AffineCoefficients:
    """PIL-style output-to-input affine coefficients ``(a, b, c, d, e, f)``."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """Map an output pixel to the input pixel."""
        return (self.a * x + self.b * y + self.c, self.d * x + self.e * y + self.f)


def rotation_coefficients(angle_degrees: float, height: int, width: int) -> AffineCoefficients:
    """Return the frozen rotation coefficients (matches ``shifts.geometric``).

    ``angle`` is the positive counter-clockwise rotation about the image
    center; coefficients invert the forward rotation (output-to-input).
    """
    theta = math.radians(angle_degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return AffineCoefficients(
        a=cos,
        b=-sin,
        c=cx - cos * cx + sin * cy,
        d=sin,
        e=cos,
        f=cy - sin * cx - cos * cy,
    )


def translation_coefficients(dx: float, dy: float) -> AffineCoefficients:
    """Return the frozen translation coefficients (output-to-input)."""
    return AffineCoefficients(a=1.0, b=0.0, c=-dx, d=0.0, e=1.0, f=-dy)


def scale_coefficients(factor: float, height: int, width: int) -> AffineCoefficients:
    """Return the frozen isotropic zoom-out coefficients (output-to-input)."""
    if factor <= 0:
        raise CorrespondenceError(f"Scale factor must be positive, got {factor}.")
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    inv = 1.0 / factor
    return AffineCoefficients(a=inv, b=0.0, c=cx - cx * inv, d=0.0, e=inv, f=cy - cy * inv)


def coefficients_for_family(
    family: str,
    parameters: dict[str, float],
    height: int,
    width: int,
) -> AffineCoefficients | None:
    """Return output-to-input coefficients for a geometry-altering family.

    Args:
        family: One of ``rotation``, ``translation``, ``scale_change``.
        parameters: Realized exact parameters from the shift result
            (``angle_degrees``, ``dx_realized``/``dy_realized``, or
            ``scale_factor``).
        height: Source image height.
        width: Source image width.

    Returns:
        Coefficients, or ``None`` for identity parameters.
    """
    if family == "rotation":
        angle = float(parameters.get("angle_degrees", 0.0))
        if angle == 0.0:
            return None
        return rotation_coefficients(angle, height, width)
    if family == "translation":
        dx = float(parameters.get("dx_realized", parameters.get("dx", 0.0)))
        dy = float(parameters.get("dy_realized", parameters.get("dy", 0.0)))
        if dx == 0 and dy == 0:
            return None
        return translation_coefficients(dx, dy)
    if family == "scale_change":
        factor = float(parameters.get("scale_factor", 1.0))
        if factor == 1.0:
            return None
        return scale_coefficients(factor, height, width)
    raise CorrespondenceError(f"Family {family!r} is not geometry-altering.")


def patch_center(col: int, row: int) -> tuple[float, float]:
    """Return the center pixel of patch ``(col, row)`` in the 224 grid."""
    return (col * PATCH_SIZE + PATCH_SIZE / 2.0 - 0.5, row * PATCH_SIZE + PATCH_SIZE / 2.0 - 0.5)


def inverse_resize_to_source(
    x_224: float, y_224: float, src_height: int, src_width: int
) -> tuple[float, float]:
    """Map a 224-space pixel to the source-resolution pixel (inverse of PIL resize)."""
    return (
        (x_224 + 0.5) * src_width / MODEL_SIZE - 0.5,
        (y_224 + 0.5) * src_height / MODEL_SIZE - 0.5,
    )


def forward_resize_to_model(
    x_src: float, y_src: float, src_height: int, src_width: int
) -> tuple[float, float]:
    """Map a source-resolution pixel to the 224-space pixel (forward PIL resize)."""
    return (
        (x_src + 0.5) * MODEL_SIZE / src_width - 0.5,
        (y_src + 0.5) * MODEL_SIZE / src_height - 0.5,
    )


def nearest_patch(x_224: float, y_224: float) -> tuple[int, int]:
    """Return the containing patch indices for a 224-space pixel.

    Patch ``(col, row)`` covers pixel-center coordinates
    ``[col*14, col*14+14)`` x ``[row*14, row*14+14)`` in the 224 grid.
    """
    col = math.floor(x_224 / PATCH_SIZE)
    row = math.floor(y_224 / PATCH_SIZE)
    return col, row


def _valid_patch(col: int, row: int) -> bool:
    return 0 <= col < GRID and 0 <= row < GRID


@dataclass(frozen=True)
class CorrespondenceMap:
    """Per-family output-to-input patch mapping for one source image.

    ``source_patch[out_index]`` is the corresponding input patch index
    (``None`` when invalid); ``valid`` marks usable correspondences.
    """

    family: str
    source_patch: np.ndarray  # shape [256], int64 or -1
    valid: np.ndarray  # shape [256], bool
    tolerance_px: float

    def __post_init__(self) -> None:
        if self.source_patch.shape != (GRID * GRID,) or self.valid.shape != (GRID * GRID,):
            raise CorrespondenceError("Correspondence maps must cover exactly 256 patches.")


def identity_correspondence() -> CorrespondenceMap:
    """Return the identity mapping used by correspondence-preserving families."""
    source_patch = np.arange(GRID * GRID, dtype=np.int64)
    valid = np.ones(GRID * GRID, dtype=bool)
    return CorrespondenceMap(
        family="identity",
        source_patch=source_patch,
        valid=valid,
        tolerance_px=0.0,
    )


def build_correspondence(
    family: str,
    parameters: dict[str, float],
    src_height: int,
    src_width: int,
    *,
    tolerance_px: float = 7.0,
) -> CorrespondenceMap:
    """Build the oracle correspondence map for one transformed image.

    Args:
        family: Shift family name.
        parameters: Realized exact parameters from the shift result.
        src_height: Source image height.
        src_width: Source image width.
        tolerance_px: Accepted for protocol compatibility; validity is
            determined by the containing patch (a displaced content patch is
            valid wherever it lands inside the source image).

    Returns:
        The correspondence map over all 256 output patches.

    Raises:
        CorrespondenceError: If the family is unknown or parameters invalid.
    """
    if family in CORRESPONDENCE_PRESERVING:
        return identity_correspondence()
    if family not in GEOMETRY_ALTERING:
        raise CorrespondenceError(f"Unknown shift family {family!r}.")
    coefficients = coefficients_for_family(family, parameters, src_height, src_width)
    source_patch = np.full(GRID * GRID, -1, dtype=np.int64)
    valid = np.zeros(GRID * GRID, dtype=bool)
    if coefficients is None:
        return identity_correspondence()
    for row in range(GRID):
        for col in range(GRID):
            out_index = row * GRID + col
            center_x, center_y = patch_center(col, row)
            src_x, src_y = inverse_resize_to_source(center_x, center_y, src_height, src_width)
            in_x, in_y = coefficients.apply(src_x, src_y)
            if not (0.0 <= in_x < src_width and 0.0 <= in_y < src_height):
                continue  # maps outside the source image (border fill region)
            model_x, model_y = forward_resize_to_model(in_x, in_y, src_height, src_width)
            in_col, in_row = nearest_patch(model_x, model_y)
            if not _valid_patch(in_col, in_row):
                continue
            source_patch[out_index] = in_row * GRID + in_col
            valid[out_index] = True
    return CorrespondenceMap(
        family=family,
        source_patch=source_patch,
        valid=valid,
        tolerance_px=tolerance_px,
    )
