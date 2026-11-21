"""Tests for deterministic seed controls."""

import random

import numpy as np
import pytest
import torch

from shiftad.utils.seed import seed_everything


def test_seed_everything_reproduces_supported_generators() -> None:
    """Repeated seeding produces identical Python, NumPy, and torch values."""

    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(2))
    state = seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(2))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert state.seed == 123


def test_seed_range_is_validated() -> None:
    """Seeds outside the NumPy range fail before mutating generators."""

    with pytest.raises(ValueError, match="between"):
        seed_everything(-1)
