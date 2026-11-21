"""Photometric shift families: brightness, contrast, gamma, color_temperature, shadow.

All transforms operate on float32 RGB images in [0, 1] at native source
resolution and never modify the mask (protocol Section 5.1). Level 0 is an
exact identity for every family. ``shadow`` is stochastic: its center is
jittered with a local ``random.Random`` seeded from the derived seed.
"""

from __future__ import annotations

import random
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

_SHADOW_JITTER = 0.02


class BrightnessShift(Shift):
    """Add a frozen positive brightness offset (protocol Section 3.1)."""

    name = "brightness"
    variant = "positive"
    _DELTAS: ClassVar[dict[int, float]] = {0: 0.00, 1: 0.05, 2: 0.10, 3: 0.18, 4: 0.30}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._DELTAS:
            raise ShiftError(f"Unsupported brightness severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._DELTAS:
            raise ShiftError(f"Unsupported brightness severity {severity}.")
        return {"delta": self._DELTAS[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        delta = parameters["delta"]
        if delta == 0.0:
            return identity_result(image, mask, parameters)
        shifted = np.clip(image + delta, 0.0, 1.0).astype(np.float32)
        return ShiftResult(image=shifted, mask=mask, exact_parameters=parameters)


class ContrastShift(Shift):
    """Scale pixel values around the per-channel mean (protocol Section 3.2)."""

    name = "contrast"
    variant = "reduction"
    _FACTORS: ClassVar[dict[int, float]] = {0: 1.00, 1: 0.85, 2: 0.70, 3: 0.55, 4: 0.35}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._FACTORS:
            raise ShiftError(f"Unsupported contrast severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._FACTORS:
            raise ShiftError(f"Unsupported contrast severity {severity}.")
        return {"factor": self._FACTORS[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        factor = parameters["factor"]
        if factor == 1.0:
            return identity_result(image, mask, parameters)
        mean = image.mean(axis=(0, 1), keepdims=True)
        shifted = np.clip(mean + factor * (image - mean), 0.0, 1.0).astype(np.float32)
        return ShiftResult(image=shifted, mask=mask, exact_parameters=parameters)


class GammaShift(Shift):
    """Apply power-law gamma correction (protocol Section 3.3)."""

    name = "gamma"
    variant = "darkening"
    _GAMMAS: ClassVar[dict[int, float]] = {0: 1.00, 1: 1.20, 2: 1.50, 3: 2.00, 4: 3.00}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._GAMMAS:
            raise ShiftError(f"Unsupported gamma severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._GAMMAS:
            raise ShiftError(f"Unsupported gamma severity {severity}.")
        return {"gamma": self._GAMMAS[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        gamma = parameters["gamma"]
        if gamma == 1.0:
            return identity_result(image, mask, parameters)
        shifted = np.clip(np.power(image, gamma), 0.0, 1.0).astype(np.float32)
        return ShiftResult(image=shifted, mask=mask, exact_parameters=parameters)


class ColorTemperatureShift(Shift):
    """Apply a channel-wise warm white-balance perturbation (protocol 3.4)."""

    name = "color_temperature"
    variant = "warm"
    _RED: ClassVar[dict[int, float]] = {0: 1.00, 1: 1.05, 2: 1.12, 3: 1.22, 4: 1.35}
    _BLUE: ClassVar[dict[int, float]] = {0: 1.00, 1: 0.95, 2: 0.88, 3: 0.78, 4: 0.65}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._RED:
            raise ShiftError(f"Unsupported color_temperature severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._RED:
            raise ShiftError(f"Unsupported color_temperature severity {severity}.")
        return {"red_factor": self._RED[severity], "blue_factor": self._BLUE[severity]}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        red, blue = parameters["red_factor"], parameters["blue_factor"]
        if red == 1.0 and blue == 1.0:
            return identity_result(image, mask, parameters)
        factors = np.asarray([red, 1.0, blue], dtype=np.float32)
        shifted = np.clip(image * factors, 0.0, 1.0).astype(np.float32)
        return ShiftResult(image=shifted, mask=mask, exact_parameters=parameters)


class ShadowShift(Shift):
    """Overlay a deterministic dark region centered on a jittered point.

    The radial gradient mask ``M`` goes from 0 at the center to 1 at the
    edge; the output is ``I_out = I_in * (1 - alpha * (1 - M))`` so the
    center is darkest (protocol Section 3.5).
    """

    name = "shadow"
    variant = "corner"
    stochastic = True
    _ALPHAS: ClassVar[dict[int, float]] = {0: 0.00, 1: 0.15, 2: 0.25, 3: 0.40, 4: 0.55}
    _CENTER = (0.25, 0.75)
    _RADIUS_RATIOS: ClassVar[dict[int, float]] = {0: 0.0, 1: 0.30, 2: 0.35, 3: 0.40, 4: 0.50}

    def __init__(self, severity: int = 0) -> None:
        super().__init__()
        if severity not in self._ALPHAS:
            raise ShiftError(f"Unsupported shadow severity {severity}.")
        self.severity = severity

    def parameters_for(self, severity: int) -> dict[str, Any]:
        if severity not in self._ALPHAS:
            raise ShiftError(f"Unsupported shadow severity {severity}.")
        return {
            "alpha": self._ALPHAS[severity],
            "center": list(self._CENTER),
            "radius_ratio": self._RADIUS_RATIOS[severity],
            "jitter": _SHADOW_JITTER,
        }

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        derived_seed: int,
    ) -> ShiftResult:
        _validate_image(image)
        _validate_mask(mask)
        parameters = self.parameters_for(self.severity)
        alpha = parameters["alpha"]
        if alpha == 0.0:
            return identity_result(image, mask, parameters)
        height, width = image.shape[0], image.shape[1]
        rng = random.Random(derived_seed)
        cx = parameters["center"][0] + rng.uniform(-_SHADOW_JITTER, _SHADOW_JITTER)
        cy = parameters["center"][1] + rng.uniform(-_SHADOW_JITTER, _SHADOW_JITTER)
        radius = parameters["radius_ratio"] * min(height, width)
        if radius <= 0:
            raise ShiftError("Shadow radius must be positive for severity > 0.")
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        distance = np.sqrt((xx - cx * width) ** 2 + (yy - cy * height) ** 2)
        gradient = np.clip(distance / radius, 0.0, 1.0)
        darkening = 1.0 - alpha * (1.0 - gradient)
        shifted = np.clip(image * darkening[..., None], 0.0, 1.0).astype(np.float32)
        realized = dict(parameters)
        realized["realized_center"] = [round(cx, 6), round(cy, 6)]
        return ShiftResult(image=shifted, mask=mask, exact_parameters=realized)
