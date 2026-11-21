"""Stochastic noise shift family: gaussian_noise (protocol Section 3.6).

Additive Gaussian noise in float32 [0, 1] space, generated with a local
``numpy.random.Generator`` seeded from the derived seed so there is no
implicit global RNG dependence.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from shiftad.shifts.base import (
    Shift,
    ShiftError,
    ShiftResult,
    _validate_image,
    _validate_mask,
    identity_result,
)


class GaussianNoiseShift(Shift):
    """Add zero-mean Gaussian noise with a frozen standard deviation."""

    name = "gaussian_noise"
    variant = "additive"
    stochastic = True
    _SIGMAS: ClassVar[dict[int, float]] = {0: 0.000, 1: 0.010, 2: 0.025, 3: 0.050, 4: 0.100}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._SIGMAS:
            raise ShiftError(f"Unsupported gaussian_noise severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._SIGMAS:
            raise ShiftError(f"Unsupported gaussian_noise severity {severity}.")
        return {"sigma": self._SIGMAS[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        sigma = parameters["sigma"]
        if sigma == 0.0:
            return identity_result(image, mask, parameters)
        generator = np.random.default_rng(derived_seed)
        noise = generator.normal(0.0, sigma, size=image.shape).astype(np.float32)
        shifted = np.clip(image + noise, 0.0, 1.0).astype(np.float32)
        return ShiftResult(image=shifted, mask=mask, exact_parameters=parameters)
