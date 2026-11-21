"""G2 smoke + parity tests (fixture-based; no training, no dataset, no GPU)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark_unit = pytest.mark.unit


@pytest.mark.unit
def test_import_noropcal_and_shiftad() -> None:
    import noropcal  # noqa: F401
    import shiftad  # noqa: F401


@pytest.mark.unit
def test_frozen_constants_from_sealed_spec() -> None:
    from noropcal import load_frozen_spec

    spec = load_frozen_spec()["principal_candidate"]
    assert spec["family"] == "A_MEDIAN_RAW"
    assert float(spec["lambda"]) == 0.25
    assert abs(float(spec["cutoff_value"]) - 0.015890181064605713) <= 1e-15


@pytest.mark.unit
def test_declarative_protocol_yaml_parses_and_matches(tmp_path) -> None:
    import yaml

    cfg = yaml.safe_load(
        (ROOT / "configs/protocols/noropcal_frozen_protocol.yaml").read_text(encoding="utf-8")
    )
    pc = cfg["method"]["principal_candidate"]
    assert float(pc["lambda"]) == 0.25
    assert abs(float(pc["cutoff_value"]) - 0.015890181064605713) <= 1e-15
    assert cfg["support_sizes"]["principal_k"] == "8"


@pytest.mark.unit
def test_score_cache_readable() -> None:
    import numpy as np

    cache = (ROOT / "reports/p0_dataset_opportunity/_cache/scores"
             / "mv2_walnuts_public_good_regular_dinov2_vits14_scores.npz")
    with np.load(cache) as data:
        scores = np.asarray(data["image_scores"], dtype=float)
    assert scores.size > 0 and np.isfinite(scores).all()


@pytest.mark.unit
def test_facade_matches_original_implementation() -> None:
    import numpy as np
    from noropcal import NorOpCal
    from noropcal._bridge import scientific_impl

    impl = scientific_impl()
    policy = NorOpCal()
    rng = np.random.default_rng(0)
    src = rng.normal(0.5, 0.05, size=40)
    tgt = rng.normal(0.62, 0.05, size=8)
    tau_source = 0.7
    got = policy.threshold(src, tgt, tau_source)
    score = impl.correctability_score(src, tgt, 0.0)
    expected = (
        impl.bounded_tau("A_MEDIAN_RAW", src, tgt, tau_source, policy.lam)
        if impl.gate_adapt(score, policy.cutoff)
        else tau_source
    )
    assert got == expected


@pytest.mark.parity
def test_level1_quick_verification_parity() -> None:
    """Run the shipped Level-1 quick verifier end-to-end (CPU, <2 min)."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_frozen_results.py"), "--quick"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-500:]
    report = json.loads(
        (ROOT / "outputs/verification/verification_quick.json").read_text(encoding="utf-8")
    )
    assert report["verdict"] == "PARITY_PASS"


@pytest.mark.integration
def test_reproduce_main_dry_run_ready() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce_main.py"), "--mode", "dry-run"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "dry-run READY" in proc.stdout
