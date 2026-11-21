"""Deterministic random seed controls for supported numerical libraries."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    """Record seed controls applied to the current process."""

    seed: int
    deterministic_algorithms: bool
    cuda_seeded: bool


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> SeedState:
    """Seed Python, NumPy, CPU PyTorch, and visible CUDA devices.

    Args:
        seed: Non-negative seed supported by NumPy.
        deterministic_algorithms: Request deterministic PyTorch algorithms.

    Returns:
        A record of the applied controls.

    Raises:
        ValueError: If the seed is outside NumPy's supported 32-bit range.
        RuntimeError: If PyTorch cannot enable the requested deterministic mode.

    Assumptions:
        Hash randomization for the already-running Python interpreter cannot be
        changed retroactively and is therefore not claimed here.
    """

    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be between 0 and 2**32 - 1 inclusive.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = torch.cuda.is_available()
    if cuda_seeded:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    return SeedState(
        seed=seed,
        deterministic_algorithms=deterministic_algorithms,
        cuda_seeded=cuda_seeded,
    )
