"""Shared shift abstractions, frozen protocol constants, and seed derivation.

This module implements the frozen Phase 5 protocol v1.1.0 primitives:
protocol constants, per-sample seed derivation, pixel-parameter scaling from
the nominal 900x900 reference resolution, and the common ``Shift`` contract.
All values here are frozen and must not be adjusted after formal experiments
begin (see ``docs/PHASE5_PROTOCOL.md`` Sections 3-5).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_VERSION = "phase5_v1.1"
BASE_SEED = 42
NOMINAL_SOURCE_SIZE = 900
MASK_REBINARIZE_THRESHOLD = 0.5


class ShiftError(RuntimeError):
    """Report an invalid shift configuration or application failure."""


def derive_seed(
    *,
    protocol_version: str = PROTOCOL_VERSION,
    category: str,
    sample_id: str,
    shift_name: str,
    severity: int,
    base_seed: int = BASE_SEED,
) -> int:
    """Derive the deterministic per-sample RNG seed (protocol Section 4.1).

    Args:
        protocol_version: Frozen protocol version string.
        category: MVTec AD category name, e.g. ``"bottle"``.
        sample_id: Sample identifier from the dataset manifest.
        shift_name: Canonical shift family name, e.g. ``"brightness"``.
        severity: Integer severity level 0-4.
        base_seed: Frozen integer base seed.

    Returns:
        The first 8 bytes of SHA256 as a big-endian unsigned 64-bit integer,
        truncated to the 32-bit NumPy RNG range.

    Raises:
        ShiftError: If ``severity`` is outside 0-4.
    """

    if not 0 <= severity <= 4:
        raise ShiftError(f"Severity must be between 0 and 4, got {severity}.")
    seed_input = f"{protocol_version}:{category}:{sample_id}:{shift_name}:{severity}:{base_seed}"
    digest = hashlib.sha256(seed_input.encode("utf-8")).digest()
    return int.from_bytes(digest[0:8], byteorder="big") % (2**32)


def next_odd(value: int) -> int:
    """Return ``value`` as an odd integer, adding one to even inputs."""

    return value if value % 2 == 1 else value + 1


def scale_translation(dx: int, dy: int, height: int, width: int) -> tuple[int, int]:
    """Scale translation pixels from the 900x900 reference to the source size.

    Args:
        dx: Reference horizontal shift in pixels.
        dy: Reference vertical shift in pixels.
        height: Actual source height ``H_src``.
        width: Actual source width ``W_src``.

    Returns:
        The rounded per-axis scaled ``(dx', dy')``.
    """

    return round(dx * width / NOMINAL_SOURCE_SIZE), round(dy * height / NOMINAL_SOURCE_SIZE)


def scale_kernel_size(kernel_size: int, height: int, width: int) -> int:
    """Scale a blur kernel size from the 900x900 reference to the source size.

    Args:
        kernel_size: Reference kernel size in pixels.
        height: Actual source height ``H_src``.
        width: Actual source width ``W_src``.

    Returns:
        An odd kernel size of at least one pixel.
    """

    scaled = round(kernel_size * min(height, width) / NOMINAL_SOURCE_SIZE)
    return max(1, next_odd(scaled))


def scale_sigma(sigma: float, height: int, width: int) -> float:
    """Scale a Gaussian sigma from the 900x900 reference to the source size.

    Args:
        sigma: Reference sigma in pixels.
        height: Actual source height ``H_src``.
        width: Actual source width ``W_src``.
    """

    return sigma * min(height, width) / NOMINAL_SOURCE_SIZE


@dataclass(frozen=True)
class ShiftResult:
    """Describe one applied shift transform.

    Attributes:
        image: Shifted ``float32`` RGB image with shape ``[H, W, 3]`` in [0, 1].
        mask: Shifted binary mask with shape ``[H, W]``, or ``None`` for normal
            samples.
        exact_parameters: Realized parameters after source-size scaling; this
            is what metadata and fingerprints must record.
        mask_area_before: Anomalous pixel count before a geometric transform,
            or ``None`` when no mask or for photometric shifts.
        mask_area_after: Anomalous pixel count after a geometric transform, or
            ``None`` when no mask or for photometric shifts.
        defect_visible_after_shift: ``True`` if the mask still contains
            anomalous pixels after a geometric transform, otherwise ``False``.
    """

    image: np.ndarray
    mask: np.ndarray | None
    exact_parameters: dict[str, Any] = field(default_factory=dict)
    mask_area_before: int | None = None
    mask_area_after: int | None = None
    defect_visible_after_shift: bool | None = None


class Shift(ABC):
    """Abstract contract implemented by every frozen shift family.

    Subclasses must be deterministic for a fixed ``derived_seed`` and must
    never branch on anomaly labels or defect type. Transforms operate at the
    native source resolution of the input image.
    """

    name: str
    variant: str
    stochastic: bool = False

    def __init__(self, severity: int = 0) -> None:
        if not self.name:
            raise ShiftError("Shift subclasses must define a non-empty name.")

    @abstractmethod
    def parameters_for(self, severity: int) -> dict[str, Any]:
        """Return the frozen reference parameters for one severity level."""

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        """Apply the shift at source resolution.

        Args:
            image: ``float32`` RGB image with shape ``[H, W, 3]`` in [0, 1].
            mask: Binary mask with shape ``[H, W]``, or ``None``.
            derived_seed: Per-sample seed from :func:`derive_seed`.

        Returns:
            The shifted result with realized (scaled) parameters.

        Raises:
            ShiftError: If the image shape or severity parameters are invalid.
        """


def identity_result(
    image: np.ndarray,
    mask: np.ndarray | None,
    parameters: dict[str, Any],
) -> ShiftResult:
    """Build the exact-identity result used by every Level 0 transform.

    Args:
        image: Source image; returned unchanged (same bytes).
        mask: Source mask; returned unchanged or ``None``.
        parameters: Realized parameter record for the cell.
    """

    return ShiftResult(
        image=image,
        mask=mask,
        exact_parameters=parameters,
        mask_area_before=None,
        mask_area_after=None,
        defect_visible_after_shift=None,
    )


def _validate_image(image: np.ndarray) -> None:
    """Validate the common float32 RGB source-image contract."""

    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
        raise ShiftError(
            "Shift input must be a float32 RGB image with shape [H, W, 3]; "
            f"got dtype={image.dtype} shape={image.shape}."
        )


def _validate_mask(mask: np.ndarray | None) -> None:
    """Validate the common binary mask contract when a mask is present."""

    if mask is None:
        return
    if mask.dtype != np.float32 or mask.ndim != 2:
        raise ShiftError(
            "Shift mask must be float32 with shape [H, W]; "
            f"got dtype={mask.dtype} shape={mask.shape}."
        )
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ShiftError("Shift mask must be binary with values in {0.0, 1.0}.")


def re_binarize(mask: np.ndarray) -> np.ndarray:
    """Re-binarize a mask at the frozen 0.5 threshold."""

    return (mask >= MASK_REBINARIZE_THRESHOLD).astype(np.float32)


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Build a normalized 2D Gaussian kernel for the given size and sigma.

    Args:
        kernel_size: Odd kernel side length.
        sigma: Gaussian standard deviation in pixels.

    Returns:
        A ``float32`` kernel with shape ``[kernel_size, kernel_size]`` summing
        to one.
    """

    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ShiftError(f"Gaussian kernel size must be odd and positive, got {kernel_size}.")
    radius = kernel_size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    if sigma <= 0:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[radius, radius] = 1.0
    else:
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    normalized = np.asarray(kernel / kernel.sum(), dtype=np.float32)
    return normalized
