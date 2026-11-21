"""PB1-DS PB3 shared library — Few-Normal K Sensitivity & PRE0 executables.

Sole source of truth for PB3-PRE0 (semantic/estimand integrity gate) and PB3
(few-normal K sensitivity). Everything here is a frozen, result-independent
execution definition reused by the PRE0 sealer and the PB3 runner.

Scope (frozen, publication-locked):
  - K grid {4, 8, FULL} with PB1 30 repeats for 4/8; FULL deterministic.
  - Only frozen methods are executed; D/C selectors reuse the sealed PUB0 OOF
    (adept decisions are frozen; the recalibration transform/QM is re-estimated
    from the K-sampled target-normal support). BC family recomputed from raw
    scores under the K-sampled support. No new candidate/feature/classifier/
    threshold objective; no best-K / best-repeat / best-seed selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import md0sr_common as C
import pb1ds_c1_common as M1
import pb1ds_pb1_common as P
import pb1ds_pb1f_common as F

PRE0_OUT = Path("reports/pb1ds_publication/pb3_pre0_integrity")
PB3_OUT = Path("reports/pb1ds_publication/pb3_k_sensitivity")

# upstream frozen terminals (verified read-only; never re-interpreted)
PB1_TERMINAL = "PB1DS_PB1_COMPLETE_PUBLICATION_BENCHMARK_PROTOCOL_FROZEN"
PB1F_TERMINAL = "PB1DS_PB1F_COMPLETE_FINAL_PROTOCOL_BINDING_VERIFIED"
PB2_TERMINAL = "PB1DS_PB2_COMPLETE_UTILITY_SAFETY_PARETO_SUPPORTED"
ORACLE_TERMINAL_OK = "PB1DS_PB2_COMPLETE_UTILITY_SAFETY_PARETO_SUPPORTED"

N_UNITS = 39
N_ENTITIES = 78
K4_ELIGIBLE = 78
K8_ELIGIBLE = 40
K_FULL_ELIGIBLE = 78
K_REPEATS = 30
K_GRID = ["4", "8", "FULL"]
MAX_SUPPORT = 12

NESTED_SAMPLING_RULE = "NESTED"  # S4 subset S8 subset SFULL for K8-eligible entities

# canonical frozen methods carried into PB3 (publication-locked comparator set)
FROZEN_METHODS: List[str] = [
    "NO_ADAPT",
    "ALWAYS_ON_QM",
    "BC_TARGET_NORMAL_THRESHOLD_FAMILY",
    "D_CATEGORY_ONLY",
    "C_CATEGORY_CONDITIONED",
]
SCIENTIFIC_NEGATIVE_CONTROL: List[str] = ["A_GLOBAL"]  # optional, kept for transparency
DESCRIPTIVE_REFERENCE: List[str] = ["BENEFIT_LABEL_ORACLE"]

# canonical effective external family (PRE0-C outcome, PB2-confirmed behavioral equivalence)
BC_FAMILY_CANONICAL = "BC_TARGET_NORMAL_THRESHOLD_FAMILY"
BC_DEFAULT_REPRESENTATIVE = "BC_RANK_BASED_THRESHOLD"
BC_QUANTILE_IMPLEMENTATION = "BC_QUANTILE_BINNING"

Q95_SOURCE = 0.95
Q95_TARGET = 0.95  # BC_RANK operating quantile
Q_BINNING_N_BINS = 20  # BC_QUANTILE frozen hyperparameter

BOOTSTRAP_REPLICATES = F.BOOTSTRAP_REPLICATES  # 5000
PERMUTATION_REPLICATES = F.PERMUTATION_REPLICATES


def oracle_publishable_name() -> str:
    """PRE0-B resolution: the PB2 ORACLE is a benefit-label oracle, NOT an
    unrestricted BA-optimal upper bound. Future reports must use this name and
    may NOT call it 'unrestricted upper bound / BA-optimal / true utility ceiling'."""
    return "BENEFIT_LABEL_ORACLE"


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def seed_int(identity: str) -> int:
    """Deterministic uint32 seed derived from the PB1 canonical seed identity."""
    return int(sha256_text(identity)[:8], 16)


def seed_identity(k_value: Any, category: str, condition: str, detector: str, repeat: int) -> str:
    """Frozen canonical seed identity (PB1/K protocol); string form.
    Reuses the exact PB1 definition (P.seed_identity)."""
    return P.seed_identity(k_value, category, condition, detector, repeat)


def load_support_reconciliation() -> pd.DataFrame:
    p = F.PB1F_OUT / "PB1DS_PB1F_FULL_SUPPORT_RECONCILIATION.csv"
    return pd.read_csv(p)


def full_n_map() -> Dict[str, int]:
    df = load_support_reconciliation()
    return dict(zip(df["entity_id"], df["FULL_n"].astype(int)))


def k8_eligible_map() -> Dict[str, bool]:
    df = load_support_reconciliation()
    return dict(zip(df["entity_id"], (df["FULL_n"].astype(int) >= 8).to_numpy()))


def entity_parts(entity_id: str) -> Tuple[str, str, str]:
    cat, cond, det = entity_id.split("|")
    return cat, cond, det


def list_entities() -> List[str]:
    df = load_support_reconciliation()
    return sorted(df["entity_id"].tolist())


def unit_manifest_units() -> List[Tuple[str, str]]:
    man = pd.read_csv(F.C1_OUT / "PB1DS_C1_UNIT_MANIFEST.csv")
    return sorted(
        set(map(tuple, man[["category", "condition"]].to_numpy().tolist())),
        key=lambda k: (k[0], k[1]),
    )


def frozen_adapt(entity_id: str, method: str) -> Optional[int]:
    """Sealed PUB0 adapt decision for D/C (frozen; K-independent by lock).
    Returns 0/1, or None for non-C/D methods (caller computes)."""
    cat, cond, det = entity_parts(entity_id)
    if method == "D_CATEGORY_ONLY":
        oof = pd.read_csv(F.PUB0_OUT / "PB1DS_PUB0_CATEGORY_ONLY_OOF.csv")
    elif method == "C_CATEGORY_CONDITIONED":
        oof = pd.read_csv(F.PUB0_OUT / "PB1DS_PUB0_OOF_ALL.csv")
        oof = oof[oof["candidate"] == "C_CATEGORY_CONDITIONED"]
    else:
        return None
    row = oof[(oof["category"] == cat) & (oof["condition"] == cond) & (oof["baseline"] == det)]
    if len(row) != 1:
        raise ValueError(f"non-unique frozen OOF row for {method} {entity_id}: {len(row)}")
    return int(row["adapt"].iloc[0])


def load_row_metrics_available(entity_id: str) -> Dict[str, float]:
    """Per-entity FULL-support frozen detection metrics (from the sealed C1 frame),
    reused as the K-invariant scalar identity check where applicable."""
    cat, cond, det = entity_parts(entity_id)
    # recompute from the frozen cache (deterministic, matches sealed registry)
    src = C.load_image_scores(C.source_rel(cat, det))
    cnd = C.load_image_scores(C.cond_normal_rel(cat, cond, det))
    anm = C.load_image_scores(C.anomaly_rel(cat, det))
    tau = float(np.percentile(src, 95.0))
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    mb = C.detection_metrics(np.concatenate([cnd, anm]), labels, tau)
    return {
        "ba_before": mb["ba"],
        "fpr_before": mb["fpr"],
        "fnr_before": mb["fnr"],
        "f1_before": mb["f1"],
        "tau": tau,
    }


# --------------------------------------------------------------------------- #
# K-sampled target-normal support + nested sampling
# --------------------------------------------------------------------------- #
def _base_perm(entity_id: str, repeat: int, cat: str, cond: str, det: str, n: int) -> np.ndarray:
    """Full deterministic permutation of support indices for an entity+repeat.
    Single base draw => S4 subset S8 subset FULL (NESTED rule)."""
    rng = np.random.default_rng(seed_int(seed_identity("BASE", cat, cond, det, repeat)))
    return rng.permutation(n)


def support_indices(
    entity_id: str, k_value: str, repeat: int, cat: str, cond: str, det: str
) -> np.ndarray:
    """Sampled target-normal score indices for a (K, repeat) cell.

    NESTED rule: S4 subset S8 subset SFULL, derived from one base permutation per
    (entity, repeat). FULL = all indices (deterministic, no repeat re-draw).
    Deterministic from frozen seed identity; must be computed before evaluation
    and sealed in the seed manifest (no re-roll after observing outcomes)."""
    n = full_n_map()[entity_id]
    if k_value == "FULL":
        return np.arange(n)
    k = int(k_value)
    if k > n:
        raise ValueError(f"K={k} > available support {n} for {entity_id}")
    return np.sort(_base_perm(entity_id, repeat, cat, cond, det, n)[:k])


def compute_unit_at_k(
    entity_id: str, cnd, anm, src, tau: float, support_k: np.ndarray, method_after: str
) -> Dict[str, Any]:
    """Per-(entity,K,repeat) downstream for the K-sampled support.

    Few-normal transform semantics (frozen-faithful generalization of the C1
    per-set quantile map): the NORMAL side is mapped through a monotone transform
    whose reference is the K available target normals (``support_k``); the ANOMALY
    side uses the frozen anomaly-self-referenced quantile map (always available).
    At K=FULL (support_k == cnd) this collapses exactly to the sealed C1/PUB0
    after-metrics, so FULL reproduces PB2.

    Parameters
    ----------
    method_after : 'QM'   -> quantile-map cnd via support_k reference, anm via
                             frozen anomaly map; threshold Q95(src)
                   'RANK' -> no remap on normal, threshold Q95(support_k)  (BC rank)
                   'QBIN' -> quantile-binning map from support_k; threshold Q95(src)
                   'NONE' -> before only (NO_ADAPT), threshold Q95(src)
    """
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    before = C.detection_metrics(np.concatenate([cnd, anm]), labels, tau)
    if method_after == "NONE":
        used_n, used_a, used_tau = cnd, anm, tau  # NO_ADAPT
    elif method_after == "QM":
        ref_n = quantile_map_transform(src, support_k)
        used_n = np.asarray([ref_n(x) for x in cnd], dtype=float)
        # anomaly side: frozen anomaly-self-referenced map (method-frozen)
        used_a = C.quantile_mapping(src, anm)
        used_tau = tau
    elif method_after == "RANK":
        used_n, used_a = cnd, anm
        used_tau = float(np.percentile(support_k, 100.0 * Q95_TARGET))
    elif method_after == "QBIN":
        ref_q = qbmap_transform(src, support_k)
        used_n = np.asarray([ref_q(x) for x in cnd], dtype=float)
        used_a = np.asarray([ref_q(x) for x in anm], dtype=float)
        used_tau = tau
    else:
        raise ValueError(method_after)
    after = C.detection_metrics(np.concatenate([used_n, used_a]), labels, used_tau)
    return {"before": before, "after": after, "tau_used": float(used_tau)}


# --------------------------------------------------------------------------- #
# Monotone quantile-map transform estimated from K target-normals
# --------------------------------------------------------------------------- #
def quantile_map_transform(source_scores: np.ndarray, target_k: np.ndarray) -> callable:
    """Return a callable x -> source-quantile using the empirical CDF of the K
    sampled target-normals as reference. Monotone by construction; at K=FULL
    the transform on the normal pool coincides with the frozen quantile_mapping."""
    src_sorted = np.sort(np.asarray(source_scores, dtype=float))
    tk = np.asarray(target_k, dtype=float)
    n_src = len(src_sorted)

    def f(x):
        x = float(x)
        q = float(np.mean(tk <= x))
        idx = max(0, min(int(q * (n_src - 1)), n_src - 1))
        return src_sorted[idx]

    return f


def qbmap_transform(
    source_scores: np.ndarray, target_k: np.ndarray, n_bins: int = Q_BINNING_N_BINS
) -> callable:
    """BC_QUANTILE affine binning map target->source estimated from K normals."""
    src = np.sort(np.asarray(source_scores, dtype=float))
    tk = np.asarray(target_k, dtype=float)
    edges = np.quantile(tk, np.linspace(0.0, 1.0, n_bins + 1))
    src_vals = np.quantile(src, (np.arange(n_bins) + 0.5) / n_bins)

    def f(x):
        x = float(x)
        b = int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1))
        return src_vals[b]

    return f


# --------------------------------------------------------------------------- #
# write helpers
# --------------------------------------------------------------------------- #
def write_csv(df: pd.DataFrame, root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def write_json(obj: Any, root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def write_jsonl(rows: Sequence[Dict[str, Any]], root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    return p


def sha256_file(p: Path) -> str:
    return M1.sha256_file(p)
