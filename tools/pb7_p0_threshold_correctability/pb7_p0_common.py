"""PB7-P0 shared library — normal-only threshold correctability identifiability audit.

Frozen constants, data-role registry, code-level DEPLOYABLE / ORACLE firewall,
and the frozen oracle diagnostic estimands (O1/O2/O3) for the PB7-P0
threshold-correctability audit.

The audit asks ONE question: can detector-specific target-normal score
deformation identify the direction and magnitude of a safe operating-point
threshold correction WITHOUT using target anomaly labels?

Everything in this module is frozen at P0 and must not be changed after any
stage outcome is observed (protocol: "禁止看完结果后修改").
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402

OUT = ROOT / "reports" / "pb7_p0_threshold_correctability"
PB6 = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
SCORES_DIR = C.SCORES_DIR

# --------------------------------------------------------------------------- #
# Frozen population (PB6 sealed)
# --------------------------------------------------------------------------- #
DETECTORS: List[str] = ["dinov2_vits14", "wide_resnet50_2", "efficientad"]
DETECTOR_NAMES: Dict[str, str] = {
    "dinov2_vits14": "DINOv2 Memory",
    "wide_resnet50_2": "PatchCore",
    "efficientad": "EfficientAD",
}
MVTEC_CATEGORIES: List[str] = list(C.MVTEC_CATEGORIES)
MVTEC_CONDITIONS: List[str] = list(C.MVTEC_CONDITIONS)
N_UNITS = 39
N_ENTITIES = 117

# --------------------------------------------------------------------------- #
# Frozen operating-point / benefit semantics (PB6 protocol binding)
# --------------------------------------------------------------------------- #
Q95 = 0.95                 # source threshold quantile (frozen threshold_quantile)
Q95_TARGET = 0.95          # BC_RANK operating quantile (frozen)
DBA_MIN = 0.02             # frozen benefit: delta_ba >= +0.02
DFPR_MAX = -0.05           # frozen benefit: delta_fpr <= -0.05
DFNR_MAX = 0.02            # frozen benefit: delta_fnr <= +0.02
SAFE_DFNR_MAX = 0.02       # O2 safety constraint: delta_fnr <= +0.02
CORRECTABLE_DBA_MIN = 0.02  # O3 correctability: delta_ba >= +0.02

# --------------------------------------------------------------------------- #
# Frozen K-support semantics (PB3 / PB6 D8 bridge)
# --------------------------------------------------------------------------- #
K_GRID: List[str] = ["4", "8", "FULL"]
K_REPEATS = 30             # frozen repeats for K in {4, 8}; FULL deterministic
PRIMARY_K = "8"            # principal practical setting
K8_MIN = 8                 # K8 eligibility: FULL_n >= 8
NESTED_SAMPLING_RULE = "NESTED"  # S4 subset S8 subset SFULL


def seed_int(identity: str) -> int:
    """Deterministic uint32 seed derived from the PB1 canonical seed identity."""
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16)


def seed_identity(k_value: Any, category: str, condition: str, detector: str, repeat: int) -> str:
    """Frozen canonical seed identity (PB1/K protocol): PB1|K|{k}|{cat}|{cond}|{det}|{rep}."""
    return f"PB1|K|{k_value}|{category}|{condition}|{detector}|{repeat}"


def base_perm(cat: str, cond: str, det: str, n: int, repeat: int) -> np.ndarray:
    """Full deterministic permutation of target-normal indices (NESTED base draw)."""
    rng = np.random.default_rng(seed_int(seed_identity("BASE", cat, cond, det, repeat)))
    return rng.permutation(n)


def support_indices(cat: str, cond: str, det: str, n: int, k_value: str, repeat: int) -> np.ndarray:
    """Sampled target-normal indices for a (K, repeat) cell (NESTED rule).

    FULL = all indices (deterministic). K in {4, 8}: first K of one base
    permutation per (entity, repeat), sorted. Mirrors the frozen
    pb1ds_pb3_common.support_indices and the PB6 D8 tri_k8_indices extension.
    """
    if k_value == "FULL":
        return np.arange(n)
    k = int(k_value)
    if k > n:
        raise ValueError(f"K={k} > available support {n} for {cat}|{cond}|{det}")
    return np.sort(base_perm(cat, cond, det, n, repeat)[:k])


# --------------------------------------------------------------------------- #
# Data-role registry (P0.2) — DEPLOYABLE vs ORACLE_DIAGNOSTIC_ONLY
# --------------------------------------------------------------------------- #
# DEPLOYABLE: usable by any deployable candidate / feature / actuator.
DEPLOYABLE_ROLES: Dict[str, str] = {
    "source_normal": "DEPLOYABLE",
    "target_normal": "DEPLOYABLE",
    "detector_identity": "DEPLOYABLE",
    "category_identity": "DEPLOYABLE",
    "condition_metadata": "DEPLOYABLE_METADATA_OPTIONAL",
}
# ORACLE_DIAGNOSTIC_ONLY: usable only for oracle targets / diagnostic metrics.
ORACLE_ROLES: Dict[str, str] = {
    "anomaly_scores": "ORACLE_DIAGNOSTIC_ONLY",
    "anomaly_labels": "ORACLE_DIAGNOSTIC_ONLY",
    "ba": "ORACLE_DIAGNOSTIC_ONLY",
    "fpr": "ORACLE_DIAGNOSTIC_ONLY",
    "fnr": "ORACLE_DIAGNOSTIC_ONLY",
    "oracle_threshold": "ORACLE_DIAGNOSTIC_ONLY",
}


class Firewall:
    """Code-level DEPLOYABLE / ORACLE firewall.

    Records every score-file read with its role. Deployable computation must
    call ``assert_deployable_clean()`` before producing any deployable output;
    this raises if any anomaly (oracle-only) file was read on that path.
    """

    def __init__(self) -> None:
        self._reads: List[Tuple[str, str, str, str, str]] = []

    def record(self, kind: str, cat: str, cond: str, det: str, rel: str) -> None:
        self._reads.append((kind, cat, cond, det, rel))

    def assert_deployable_clean(self) -> None:
        bad = [r for r in self._reads if r[0] == "anomaly"]
        if bad:
            raise SystemExit(
                f"FIREWALL_VIOLATION: deployable path read oracle-only anomaly file: {bad[:3]}"
            )


def load_src(cat: str, det: str, fw: Optional[Firewall] = None) -> np.ndarray:
    rel = C.source_rel(cat, det)
    if fw is not None:
        fw.record("normal", cat, "", det, rel)
    return C.load_image_scores(rel)


def load_cnd(cat: str, cond: str, det: str, fw: Optional[Firewall] = None) -> np.ndarray:
    rel = C.cond_normal_rel(cat, cond, det)
    if fw is not None:
        fw.record("normal", cat, cond, det, rel)
    return C.load_image_scores(rel)


def load_anm(cat: str, det: str, fw: Optional[Firewall] = None) -> np.ndarray:
    rel = C.anomaly_rel(cat, det)
    if fw is not None:
        fw.record("anomaly", cat, "", det, rel)
    return C.load_image_scores(rel)


def source_threshold(src: np.ndarray) -> float:
    """Frozen source operating-point threshold: Q95 of source-normal scores."""
    return float(np.percentile(src, 100.0 * Q95))


def entity_list() -> List[Tuple[str, str, str]]:
    """Authoritative 117-entity list from the sealed PB6 D5 OOF per unit."""
    oof = pd.read_csv(PB6 / "PB1DS_PB6_D5_OOF_PER_UNIT.csv")
    ents = oof[["category", "condition", "baseline"]].drop_duplicates()
    return sorted(
        [tuple(r) for r in ents.to_records(index=False)],
        key=lambda e: (e[0], e[1], e[2]),
    )


def unit_list() -> List[Tuple[str, str]]:
    """Authoritative 39-unit list from the sealed PB6 D5 split manifest."""
    ents = entity_list()
    return sorted(set((c, d) for c, d, _ in ents), key=lambda u: (u[0], u[1]))


# --------------------------------------------------------------------------- #
# Frozen oracle diagnostic estimands (P0.3) — O1 / O2 / O3
# --------------------------------------------------------------------------- #
def candidate_grid(src: np.ndarray, cnd: np.ndarray, anm: np.ndarray, tau_source: float) -> np.ndarray:
    """Frozen candidate threshold grid.

    All unique values of the pooled target scores (target-normal U anomaly)
    PLUS the source threshold tau_source, sorted ascending. Detection rule is
    ``score > tau`` (strict), so scanning the unique pooled scores is exact for
    the step-function BA/FPR/FNR curves. tau_source is always a candidate so
    ``delta_tau = 0`` (no change) is always expressible.
    """
    pooled = np.concatenate([cnd, anm])
    return np.sort(np.unique(np.concatenate([pooled, [tau_source]])))


def _tie_break(cands: np.ndarray, tau_source: float) -> float:
    """Frozen tie-break: closest to tau_source; further tie -> smaller tau."""
    dist = np.abs(cands - tau_source)
    best = float(np.min(dist))
    close = cands[np.isclose(dist, best, rtol=0.0, atol=1e-12)]
    return float(np.min(close))


def oracle_ba_optimal(
    src: np.ndarray, cnd: np.ndarray, anm: np.ndarray, tau_source: float
) -> Dict[str, Any]:
    """O1 — Oracle BA-optimal threshold tau*_BA.

    tau*_BA = argmax over the frozen grid of BA(score > tau); tie-break closest
    to tau_source, further tie smaller tau. Returns the chosen tau and its
    detection metrics (BA/FPR/FNR) plus delta vs the source operating point.
    """
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    scores = np.concatenate([cnd, anm])
    grid = candidate_grid(src, cnd, anm, tau_source)
    mb = C.detection_metrics(scores, labels, tau_source)
    best_ba = -np.inf
    best_taus: List[float] = []
    for tau in grid:
        m = C.detection_metrics(scores, labels, float(tau))
        if m["ba"] > best_ba + 1e-12:
            best_ba = m["ba"]
            best_taus = [float(tau)]
        elif np.isclose(m["ba"], best_ba, rtol=0.0, atol=1e-12):
            best_taus.append(float(tau))
    tau_star = _tie_break(np.asarray(best_taus, dtype=float), tau_source)
    ma = C.detection_metrics(scores, labels, tau_star)
    return {
        "tau_star": tau_star,
        "ba": ma["ba"],
        "fpr": ma["fpr"],
        "fnr": ma["fnr"],
        "delta_ba": ma["ba"] - mb["ba"],
        "delta_fpr": ma["fpr"] - mb["fpr"],
        "delta_fnr": ma["fnr"] - mb["fnr"],
        "delta_tau": tau_star - tau_source,
        "n_candidates": int(len(grid)),
    }


def oracle_safe_optimal(
    src: np.ndarray, cnd: np.ndarray, anm: np.ndarray, tau_source: float
) -> Dict[str, Any]:
    """O2 — Oracle safety-constrained threshold tau*_SAFE.

    tau*_SAFE = argmax over the frozen grid of BA subject to
    FNR(tau) - FNR(tau_source) <= +0.02 (delta_fnr <= SAFE_DFNR_MAX).
    Tie-break closest to tau_source, further tie smaller tau.
    tau_source is always feasible (delta_fnr = 0), so the feasible set is
    never empty.
    """
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    scores = np.concatenate([cnd, anm])
    grid = candidate_grid(src, cnd, anm, tau_source)
    mb = C.detection_metrics(scores, labels, tau_source)
    feasible: List[float] = []
    for tau in grid:
        m = C.detection_metrics(scores, labels, float(tau))
        if m["fnr"] - mb["fnr"] <= SAFE_DFNR_MAX + 1e-12:
            feasible.append(float(tau))
    best_ba = -np.inf
    best_taus: List[float] = []
    for tau in feasible:
        m = C.detection_metrics(scores, labels, tau)
        if m["ba"] > best_ba + 1e-12:
            best_ba = m["ba"]
            best_taus = [tau]
        elif np.isclose(m["ba"], best_ba, rtol=0.0, atol=1e-12):
            best_taus.append(tau)
    tau_star = _tie_break(np.asarray(best_taus, dtype=float), tau_source)
    ma = C.detection_metrics(scores, labels, tau_star)
    return {
        "tau_star": tau_star,
        "ba": ma["ba"],
        "fpr": ma["fpr"],
        "fnr": ma["fnr"],
        "delta_ba": ma["ba"] - mb["ba"],
        "delta_fpr": ma["fpr"] - mb["fpr"],
        "delta_fnr": ma["fnr"] - mb["fnr"],
        "delta_tau": tau_star - tau_source,
        "n_feasible": int(len(feasible)),
    }


def correctable_safe(before: Dict[str, float], after: Dict[str, float]) -> int:
    """O3 — Oracle correctability label CORRECTABLE_SAFE.

    1 iff delta_ba >= +0.02 AND delta_fpr <= -0.05 AND delta_fnr <= +0.02
    relative to the source operating point (the frozen benefit definition).
    Diagnostic label only; never a deployment input.
    """
    return int(C.benefit_label(before, after))


def sha256_file(path: Path) -> str:
    return C.sha256_file(path)
