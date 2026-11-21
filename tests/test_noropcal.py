"""Unit tests for the NorOpCal core package."""

from __future__ import annotations

import numpy as np
import pytest
from noropcal import NorOpCal, load_frozen_spec


def test_frozen_defaults():
    policy = NorOpCal()
    spec = load_frozen_spec()["principal_candidate"]
    assert policy.family == spec["family"]
    assert np.isclose(policy.lam, spec["lambda"])
    assert np.isclose(policy.cutoff, spec["cutoff_value"])
    assert policy.frozen_defaults is True


def test_custom_parameters():
    policy = NorOpCal(lam=0.5, cutoff=0.05, family="A_MEDIAN_RAW")
    assert np.isclose(policy.lam, 0.5)
    assert np.isclose(policy.cutoff, 0.05)
    assert policy.frozen_defaults is False


def test_gate_behavior():
    policy = NorOpCal()
    # Below Q50 cutoff
    assert policy.gate(0.01) is False
    # Above Q50 cutoff
    assert policy.gate(0.02) is True


def test_threshold_adaptation():
    policy = NorOpCal()
    src = np.array([0.10] * 50)
    tgt = np.array([0.30] * 8)
    tau_source = 0.20

    # Delta_m = 0.3 - 0.1 = 0.2 >= Q50 (passes gate)
    # raw_tau = 0.2 + 0.2 = 0.4
    # bounded_tau = 0.2 + 0.25 * (0.4 - 0.2) = 0.25
    adapted = policy.threshold(src, tgt, tau_source)
    assert np.isclose(adapted, 0.25)


def test_abstention_on_negative_or_small_shift():
    policy = NorOpCal()
    src = np.array([0.20] * 50)
    tgt = np.array([0.205] * 8)  # Delta_m = 0.005 < Q50 (0.01589)
    tau_source = 0.35

    # Should abstain and return original tau_source unchanged
    assert np.isclose(policy.threshold(src, tgt, tau_source), tau_source)


def test_empty_array_error():
    policy = NorOpCal()
    empty = np.array([])
    valid = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="Score arrays must not be empty"):
        policy.threshold(empty, valid, 0.5)

    with pytest.raises(ValueError, match="Score arrays must not be empty"):
        policy.threshold(valid, empty, 0.5)
