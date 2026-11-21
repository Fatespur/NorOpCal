"""PB8 shared library — publication benchmark + ablation + paper claim freeze.

PB8 inherits the sealed PB7-P2 terminal
T1 PB7_P2_PUBLICATION_CANDIDATE_CONFIRMED_DINOV2_SPECIFIC and builds a complete,
reviewable, reproducible paper evidence package around the frozen
DINOv2-specific normal-only safe threshold adaptation. It performs NO method
development: the frozen principal policy is never modified.

Frozen principal policy (from PB7-P1 E_FROZEN_METHOD_SPEC.json, PB7-P2 confirmatory):
  family = A_MEDIAN_RAW, lambda = 0.25, cutoff = Q50 (median_shift gate),
  principal K = 8, safety budget mean dFNR <= +0.02, primary utility dBA,
  DINOv2-specific / detector-dependent claim scope.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p2_publication_validation"))

import pb7_p1_common as P1  # type: ignore[import-not-found]  # noqa: E402
import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

OUT = ROOT / "reports" / "pb8_publication_benchmark"
P1_OUT = P1.OUT
P2_OUT = P2.OUT
P0_OUT = P1.PB7P0

# --------------------------------------------------------------------------- #
# Frozen constants (re-export, no redefinition)
# --------------------------------------------------------------------------- #
Q95 = P1.Q95
DBA_MIN = P1.DBA_MIN
DFPR_MAX = P1.DFPR_MAX
DFNR_MAX = P1.DFNR_MAX
SAFE_DFNR_MAX = P1.SAFE_DFNR_MAX
HARM_DBA = P1.HARM_DBA
HARM_DFNR = P1.HARM_DFNR
CATASTROPHIC_DFNR = P1.CATASTROPHIC_DFNR
K_GRID = P1.K_GRID
K_REPEATS = P1.K_REPEATS
PRIMARY_K = P1.PRIMARY_K
DETECTORS = P1.DETECTORS
DETECTOR_NAMES = P1.DETECTOR_NAMES

Firewall = P1.Firewall
load_src = P1.load_src
load_cnd = P1.load_cnd
load_anm = P1.load_anm
source_threshold = P1.source_threshold
entity_list = P1.entity_list
support_indices = P1.support_indices
split_side = P1.split_side
correctability_score = P1.correctability_score
gate_adapt = P1.gate_adapt
bounded_tau = P1.bounded_tau
actuator_raw = P1.actuator_raw
summary_metrics = P1.summary_metrics
sha256_file = P1.sha256_file
evaluate_entity = P1.evaluate_entity
block_bootstrap = P1.block_bootstrap

confirmatory_bootstrap = P2.confirmatory_bootstrap
paired_block_bootstrap = P2.paired_block_bootstrap
evaluate_entity_repeats = P2.evaluate_entity_repeats
CONF_BOOT_B = P2.CONF_BOOT_B
CONF_BOOT_SEED = P2.CONF_BOOT_SEED

# Frozen principal policy
PRINCIPAL_FAMILY = P2.PRINCIPAL_FAMILY
PRINCIPAL_LAMBDA = P2.PRINCIPAL_LAMBDA
PRINCIPAL_CUTOFF_KEY = P2.PRINCIPAL_CUTOFF_KEY
PRINCIPAL_CUTOFF = P2.PRINCIPAL_CUTOFF
PRINCIPAL_K = P2.PRINCIPAL_K

# Frozen safety budget
DFNR_MAX = 0.02

# --------------------------------------------------------------------------- #
# Canonical PB8 baseline registry (frozen; no outcome-driven add/remove)
# --------------------------------------------------------------------------- #
# Each entry: (id, family, lambda, cutoff, gate_mode, description)
# gate_mode: "none" (always adapt), "q50" (frozen median_shift gate), "oracle"
NO_GATE = float("-inf")


def _cutoff(mode: str) -> float:
    if mode == "none":
        return NO_GATE
    if mode == "q50":
        return PRINCIPAL_CUTOFF
    raise ValueError(f"unknown gate mode {mode}")


BASELINES: list[dict[str, Any]] = [
    {
        "id": "B0",
        "family": None,
        "lambda": 0.0,
        "cutoff": 0.0,
        "gate": "none",
        "name": "NoAdapt (tau_source)",
        "role": "reference",
        "source": "frozen",
    },
    {
        "id": "B1",
        "family": "A_T2_RAW",
        "lambda": 1.0,
        "cutoff": NO_GATE,
        "gate": "none",
        "name": "PB7-P0 unsafe strong actuator (A_T2_RAW, lambda=1.0, no gate)",
        "role": "unsafe_strong",
        "source": "frozen",
    },
    {
        "id": "B2",
        "family": "A_MEDIAN_RAW",
        "lambda": 1.0,
        "cutoff": NO_GATE,
        "gate": "none",
        "name": "Raw median actuator (A_MEDIAN_RAW, lambda=1.0, no gate)",
        "role": "raw",
        "source": "frozen",
    },
    {
        "id": "B3",
        "family": "A_MEDIAN_RAW",
        "lambda": 0.25,
        "cutoff": NO_GATE,
        "gate": "none",
        "name": "Bounded-only (A_MEDIAN_RAW, lambda=0.25, no gate)",
        "role": "bounded_only",
        "source": "frozen",
    },
    {
        "id": "B4",
        "family": "A_MEDIAN_RAW",
        "lambda": 1.0,
        "cutoff": PRINCIPAL_CUTOFF,
        "gate": "q50",
        "name": "Gate-only (A_MEDIAN_RAW, lambda=1.0, median_shift>=Q50)",
        "role": "gate_only",
        "source": "fresh",
    },
    {
        "id": "B5",
        "family": "A_MEDIAN_RAW",
        "lambda": 0.25,
        "cutoff": PRINCIPAL_CUTOFF,
        "gate": "q50",
        "name": "Frozen proposed policy (A_MEDIAN_RAW, lambda=0.25, median_shift>=Q50)",
        "role": "principal",
        "source": "frozen",
    },
    {
        "id": "B6_T2",
        "family": "A_T2_RAW",
        "lambda": 0.25,
        "cutoff": PRINCIPAL_CUTOFF,
        "gate": "q50",
        "name": "Alternative family A_T2_RAW (lambda=0.25, Q50)",
        "role": "family_baseline",
        "source": "fresh",
    },
    {
        "id": "B6_TAIL",
        "family": "A_TAIL_RAW",
        "lambda": 0.25,
        "cutoff": PRINCIPAL_CUTOFF,
        "gate": "q50",
        "name": "Alternative family A_TAIL_RAW (lambda=0.25, Q50)",
        "role": "family_baseline",
        "source": "fresh",
    },
]

ORACLE_BASELINE_ID = "B7_ORACLE_SAFE"
LITERATURE_BASELINE_ID = "B8_LITERATURE"


def test_entities() -> list[tuple[str, str, str]]:
    """TEST-side entities (frozen PB7-P1 split)."""
    return [e for e in entity_list() if split_side(e[0]) == "TEST"]


def k8_eligible_entities() -> list[tuple[str, str, str]]:
    """Entities with target-normal FULL support n >= 8 (K8 eligibility)."""
    fw = Firewall()
    out: list[tuple[str, str, str]] = []
    for cat, cond, det in test_entities():
        cnd = load_cnd(cat, cond, det, fw)
        if len(cnd) >= P1.K8_MIN:
            out.append((cat, cond, det))
    return out


def target_normal_counts() -> pd.DataFrame:
    """Per (category, condition, detector) target-normal count."""
    fw = Firewall()
    rows: list[dict[str, Any]] = []
    for cat, cond, det in entity_list():
        cnd = load_cnd(cat, cond, det, fw)
        rows.append(
            {
                "category": cat,
                "condition": cond,
                "detector": det,
                "target_normal_n": len(cnd),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Canonical per-entity policy evaluation (PB8 benchmark / ablation)
# --------------------------------------------------------------------------- #
def evaluate_policy_per_entity(
    family: str | None,
    lam: float,
    cutoff: float,
    k_value: str,
    n_repeats: int = K_REPEATS,
) -> pd.DataFrame:
    """Evaluate one canonical policy on all TEST entities at a K setting.

    Uses the frozen P1.evaluate_entity semantics (K_REPEATS=30 for K in {4,8},
    deterministic for FULL). Returns a per-entity frame with the same columns
    as the sealed PB7-P1 E_TEST_PER_ENTITY.csv plus a baseline tag.
    """
    fw = Firewall()
    rows: list[dict[str, Any]] = []
    for cat, cond, det in test_entities():
        src = load_src(cat, det, fw)
        cnd = load_cnd(cat, cond, det, fw)
        anm = load_anm(cat, det, fw)
        tau_source = source_threshold(src)
        m = evaluate_entity(cat, cond, det, family, lam, cutoff, k_value, src, cnd, anm, tau_source)
        rows.append(
            {
                "category": cat,
                "condition": cond,
                "detector": det,
                "K": k_value,
                **m,
            }
        )
    return pd.DataFrame(rows)


def summary_for(df: pd.DataFrame, det_name: str | None, k_value: str) -> dict[str, float]:
    """Pooled summary metrics for a per-entity frame (optional detector filter)."""
    sub = df[k_value == df.K]
    if det_name is not None:
        sub = sub[sub.detector == det_name]
    return summary_metrics(sub)


# --------------------------------------------------------------------------- #
# SHA256 evidence binding
# --------------------------------------------------------------------------- #
def bind_artifact(rel: str) -> dict[str, str]:
    """SHA256-bind a report artifact by its relative path (from ROOT)."""
    p = ROOT / rel
    return {
        "artifact": rel,
        "exists": str(p.exists()),
        "sha256": sha256_file(p) if p.exists() else "",
    }


def write_json(obj: Any, name: str) -> Path:
    import json

    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    return p
