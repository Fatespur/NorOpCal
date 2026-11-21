"""Blur shift families: motion_blur and defocus_blur.

Both families convolve the float32 image with a normalized kernel at native
source resolution and do not change mask geometry (protocol Sections 3.7-3.8,
5.2). Kernel sizes (and sigma for defocus) are scaled from the nominal 900x900
reference to the actual source size per protocol Section 5.6.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as functional

from shiftad.shifts.base import (
    Shift,
    ShiftError,
    ShiftResult,
    _validate_image,
    _validate_mask,
    gaussian_kernel,
    identity_result,
    scale_kernel_size,
    scale_sigma,
)


def _line_kernel(kernel_size: int, angle_degrees: float) -> np.ndarray:
    """Build a normalized 1D line kernel at the given angle.

    Args:
        kernel_size: Odd kernel side length.
        angle_degrees: Line angle in degrees (0 = horizontal).

    Returns:
        A ``float32`` kernel with shape ``[kernel_size, kernel_size]`` summing
        to one.
    """

    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ShiftError(f"Line kernel size must be odd and positive, got {kernel_size}.")
    radius = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    angle = np.deg2rad(angle_degrees)
    direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
    for index in range(kernel_size):
        offset = axis[index] * direction
        column = round(offset[0]) + radius
        row = round(offset[1]) + radius
        if 0 <= row < kernel_size and 0 <= column < kernel_size:
            kernel[row, column] = 1.0
    total = kernel.sum()
    if total <= 0:
        raise ShiftError(f"Line kernel for size {kernel_size} has zero weight.")
    normalized = np.asarray(kernel / total, dtype=np.float32)
    return normalized


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a float32 RGB image with a normalized 2D kernel on CPU.

    Args:
        image: ``float32`` RGB image with shape ``[H, W, 3]``.
        kernel: ``float32`` kernel with shape ``[K, K]``.

    Returns:
        The convolved image with the same shape and dtype.
    """

    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))[None]
    # One shared kernel per RGB channel via grouped convolution.
    kernel_tensor = torch.from_numpy(np.repeat(kernel[None, None], 3, axis=0))
    convolved = functional.conv2d(
        tensor,
        kernel_tensor,
        padding=padding,
        groups=3,
    )
    return np.transpose(convolved[0].numpy(), (1, 2, 0)).astype(np.float32)


class MotionBlurShift(Shift):
    """Apply linear motion blur at a frozen angle (protocol Section 3.7)."""

    name = "motion_blur"
    variant = "linear"
    _KERNELS: ClassVar[dict[int, int]] = {0: 1, 1: 5, 2: 9, 3: 15, 4: 25}
    _ANGLE_DEGREES = 30.0

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._KERNELS:
            raise ShiftError(f"Unsupported motion_blur severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._KERNELS:
            raise ShiftError(f"Unsupported motion_blur severity {severity}.")
        return {
            "kernel_size": self._KERNELS[severity],
            "angle_degrees": self._ANGLE_DEGREES,
        }

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
        kernel_size = scale_kernel_size(reference["kernel_size"], height, width)
        parameters = dict(reference)
        parameters["kernel_size_realized"] = kernel_size
        if kernel_size == 1:
            return identity_result(image, mask, parameters)
        kernel = _line_kernel(kernel_size, reference["angle_degrees"])
        shifted = np.clip(_convolve(image, kernel), 0.0, 1.0).astype(np.float32)
        return ShiftResult(image=shifted, mask=mask, exact_parameters=parameters)


class DefocusBlurShift(Shift):
    """Apply isotropic Gaussian (defocus) blur (protocol Section 3.8)."""

    name = "defocus_blur"
    variant = "gaussian"
    _SIGMAS: ClassVar[dict[int, float]] = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.5, 4: 5.0}
    _KERNELS: ClassVar[dict[int, int]] = {0: 1, 1: 5, 2: 9, 3: 15, 4: 21}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._SIGMAS:
            raise ShiftError(f"Unsupported defocus_blur severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._SIGMAS:
            raise ShiftError(f"Unsupported defocus_blur severity {severity}.")
        return {
            "sigma": self._SIGMAS[severity],
            "kernel_size": self._KERNELS[severity],
        }

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
        kernel_size = scale_kernel_size(reference["kernel_size"], height, width)
        sigma = scale_sigma(reference["sigma"], height, width)
        parameters = dict(reference)
        parameters["kernel_size_realized"] = kernel_size
        parameters["sigma_realized"] = round(float(sigma), 6)
        if kernel_size == 1:
            return identity_result(image, mask, parameters)
        kernel = gaussian_kernel(kernel_size, sigma)
        shifted = np.clip(_convolve(image, kernel), 0.0, 1.0).astype(np.float32)
        return ShiftResult(image=shifted, mask=mask, exact_parameters=parameters)
