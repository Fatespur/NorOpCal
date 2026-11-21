"""PB7-P1 shared library — safety-constrained normal-only threshold actuator development.

This stage (PB7-P1) inherits the sealed PB7-P0 terminal
F4 THRESHOLD_CORRECTABILITY_IDENTIFIED_BUT_SIMPLE_ACTUATORS_INSUFFICIENT and
develops a conservative, bounded, abstaining threshold actuator that converts
identified correctability into SAFE deployable threshold adaptation.

Design principles (frozen at P0, must not change after any outcome):
  - No target-anomaly information in any deployable decision (code firewall).
  - No prediction of oracle threshold direction/magnitude (H2/H3 NOT SUPPORTED).
  - Base actuator families (<= 3) reuse PB7-P0 verified simple formulas.
  - Bounding shrinks the raw correction toward tau_source via lambda in [0,1].
  - Abstention gate uses median_shift (PB7-P0 PASS-B principal signal).
  - DEV is used only to select (family, lambda, cutoff); TEST is one frozen eval.
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
sys.path.insert(0, str(ROOT / "tools" / "pb7_p0_threshold_correctability"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p0_common as P0  # type: ignore[import-not-found]  # noqa: E402

OUT = ROOT / "reports" / "pb7_p1_safety_actuator"
PB7P0 = P0.OUT

# --------------------------------------------------------------------------- #
# Frozen constants (reuse PB7-P0 frozen semantics; no redefinition)
# --------------------------------------------------------------------------- #
Q95 = P0.Q95                       # source threshold quantile
DBA_MIN = P0.DBA_MIN               # frozen benefit: delta_ba >= +0.02
DFPR_MAX = P0.DFPR_MAX             # frozen benefit: delta_fpr <= -0.05
DFNR_MAX = P0.DFNR_MAX             # frozen benefit / safety: delta_fnr <= +0.02
SAFE_DFNR_MAX = P0.SAFE_DFNR_MAX   # O2 safety constraint
HARM_DBA = -0.02                   # frozen harmful-unit: delta_ba < -0.02
HARM_DFNR = 0.02                   # frozen harmful-unit: delta_fnr > +0.02
CATASTROPHIC_DFNR = 0.05           # frozen catastrophic: delta_fnr > +0.05

K_GRID: List[str] = list(P0.K_GRID)          # ["4", "8", "FULL"]
K_REPEATS = P0.K_REPEATS                     # 30
PRIMARY_K = "8"                              # principal practical setting
K8_MIN = P0.K8_MIN                           # K8 eligibility: FULL_n >= 8

DETECTORS: List[str] = list(P0.DETECTORS)
DETECTOR_NAMES: Dict[str, str] = dict(P0.DETECTOR_NAMES)

# --------------------------------------------------------------------------- #
# Frozen PB7-P1 actuator / bounding / gate registries (pre-registered at P0)
# --------------------------------------------------------------------------- #
ACTUATOR_FAMILIES: List[str] = ["A_T2_RAW", "A_TAIL_RAW", "A_MEDIAN_RAW"]
ACTUATOR_NAMES: Dict[str, str] = {
    "A_T2_RAW": "Robust Location-Scale (PB7-P0 T2 exact)",
    "A_TAIL_RAW": "Tail-Preserving Q95 (PB7-P0 T3 exact)",
    "A_MEDIAN_RAW": "Conservative Location-Only (median shift)",
}
LAMBDA_GRID: List[float] = [0.25, 0.50, 0.75, 1.00]
CUTOFF_KEYS: List[str] = ["Q50", "Q60", "Q70", "Q80"]
CUTOFF_QUANTILES: Dict[str, float] = {"Q50": 0.50, "Q60": 0.60, "Q70": 0.70, "Q80": 0.80}

# --------------------------------------------------------------------------- #
# Frozen DEV/TEST split (category-level, deterministic SHA256, pre-registered)
# --------------------------------------------------------------------------- #
SPLIT_SALT = "PB7_P1_DEV_TEST_SPLIT_v1"


def split_side(category: str) -> str:
    """Deterministic category -> DEV/TEST assignment (SHA256 parity, frozen salt)."""
    h = int(hashlib.sha256(f"{SPLIT_SALT}|{category}".encode("utf-8")).hexdigest()[:8], 16)
    return "DEV" if h % 2 == 0 else "TEST"


def seed_int(identity: str) -> int:
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Data-role firewall (reuse PB7-P0 Firewall; deployable path must be clean)
# --------------------------------------------------------------------------- #
Firewall = P0.Firewall
load_src = P0.load_src
load_cnd = P0.load_cnd
load_anm = P0.load_anm
source_threshold = P0.source_threshold
entity_list = P0.entity_list
support_indices = P0.support_indices


def sha256_file(path: Path) -> str:
    return C.sha256_file(path)


# --------------------------------------------------------------------------- #
# Base actuator families (A) — frozen formulas, DEPLOYABLE inputs only
# --------------------------------------------------------------------------- #
def actuator_raw(family: str, src: np.ndarray, cnd: np.ndarray, tau_source: float) -> float:
    """Raw actuator threshold from DEPLOYABLE inputs only (no anomaly use).

    A_T2_RAW:    tau_raw = median_t + (MAD_t/MAD_s) * (tau_source - median_s)
                 (exact PB7-P0 T2 formula; MAD_s ~= 0 guarded to ratio 1.0)
    A_TAIL_RAW:  tau_raw = Q95(target-normal)  (exact PB7-P0 T3 formula)
    A_MEDIAN_RAW: tau_raw = tau_source + (median_t - median_s)
    """
    med_s = float(np.median(src))
    med_t = float(np.median(cnd))
    if family == "A_T2_RAW":
        mad_s = float(np.median(np.abs(src - med_s)))
        mad_t = float(np.median(np.abs(cnd - med_t)))
        ratio = (mad_t / mad_s) if mad_s > 1e-12 else 1.0
        return med_t + ratio * (tau_source - med_s)
    if family == "A_TAIL_RAW":
        return float(np.percentile(cnd, 100.0 * Q95))
    if family == "A_MEDIAN_RAW":
        return tau_source + (med_t - med_s)
    raise ValueError(f"unknown actuator family {family}")


# --------------------------------------------------------------------------- #
# Safety bounding (B) — multiplicative shrinkage toward tau_source
# --------------------------------------------------------------------------- #
def bounded_tau(family: str, src: np.ndarray, cnd: np.ndarray,
                tau_source: float, lam: float) -> float:
    """tau_used = tau_source + lam * (tau_raw - tau_source), lam in [0,1].

    Bounding only shrinks toward the source threshold; it never flips the
    correction direction (B3: no direction switching).
    """
    tau_raw = actuator_raw(family, src, cnd, tau_source)
    return tau_source + lam * (tau_raw - tau_source)


# --------------------------------------------------------------------------- #
# Correctability gate / abstention (C) — median_shift threshold
# --------------------------------------------------------------------------- #
def correctability_score(src: np.ndarray, cnd: np.ndarray, tau_source: float) -> float:
    """Principal correctability signal: median_shift (PB7-P0 PASS-B, AUC 0.712)."""
    return float(np.median(cnd) - np.median(src))


def gate_adapt(score: float, c: float) -> bool:
    """C2 gate form: adapt iff correctability_score >= c, else NoAdapt."""
    return bool(score >= c)


# --------------------------------------------------------------------------- #
# Entity-level evaluation (mirrors PB7-P0 E-stage structure)
# --------------------------------------------------------------------------- #
def evaluate_entity(cat: str, cond: str, det: str, family: str, lam: float, c: float,
                    k_value: str, src: np.ndarray, cnd_full: np.ndarray,
                    anm: np.ndarray, tau_source: float) -> Dict[str, Any]:
    """Evaluate a bounded+abstaining actuator on one entity.

    For K in {4,8}: per frozen repeat, sample K target-normal support, compute
    the K-sampled correctability score and bounded actuator threshold, apply the
    gate, evaluate on the FULL target set, and average metrics over repeats.
    For FULL: deterministic single evaluation on the FULL target-normal support.
    """
    labels = np.concatenate([np.zeros(len(cnd_full)), np.ones(len(anm))])
    scores = np.concatenate([cnd_full, anm])
    before = C.detection_metrics(scores, labels, tau_source)
    n = len(cnd_full)

    def metrics_at(tau: float) -> Dict[str, float]:
        m = C.detection_metrics(scores, labels, float(tau))
        return {
            "ba": m["ba"], "fpr": m["fpr"], "fnr": m["fnr"],
            "delta_ba": m["ba"] - before["ba"],
            "delta_fpr": m["fpr"] - before["fpr"],
            "delta_fnr": m["fnr"] - before["fnr"],
        }

    if k_value == "FULL":
        if family is None:
            tau = tau_source
        else:
            score = correctability_score(src, cnd_full, tau_source)
            tau = bounded_tau(family, src, cnd_full, tau_source, lam) if gate_adapt(score, c) \
                else tau_source
        m = metrics_at(tau)
        mm = C.detection_metrics(scores, labels, tau)
        m["tau"] = tau
        m["benefit"] = int(C.benefit_label(before, mm))
        m["harm"] = int(m["delta_ba"] < HARM_DBA or m["delta_fnr"] > HARM_DFNR)
        m["catastrophic"] = int(m["delta_fnr"] > CATASTROPHIC_DFNR)
        m["coverage"] = 1.0 if abs(tau - tau_source) > 1e-12 else 0.0
        m["n_repeats"] = 1
        return m

    k = int(k_value)
    acc = {"ba": [], "fpr": [], "fnr": [], "delta_ba": [], "delta_fpr": [],
           "delta_fnr": [], "benefit": [], "harm": [], "catastrophic": [], "coverage": []}
    for r in range(K_REPEATS):
        if k > n:
            break
        idx = support_indices(cat, cond, det, n, k_value, r)
        cnd_k = cnd_full[idx]
        if family is None:
            tau = tau_source
        else:
            score = correctability_score(src, cnd_k, tau_source)
            tau = bounded_tau(family, src, cnd_k, tau_source, lam) if gate_adapt(score, c) \
                else tau_source
        m = metrics_at(tau)
        mm = C.detection_metrics(scores, labels, tau)
        acc["ba"].append(m["ba"]); acc["fpr"].append(m["fpr"]); acc["fnr"].append(m["fnr"])
        acc["delta_ba"].append(m["delta_ba"]); acc["delta_fpr"].append(m["delta_fpr"])
        acc["delta_fnr"].append(m["delta_fnr"])
        acc["benefit"].append(int(C.benefit_label(before, mm)))
        acc["harm"].append(int(m["delta_ba"] < HARM_DBA or m["delta_fnr"] > HARM_DFNR))
        acc["catastrophic"].append(int(m["delta_fnr"] > CATASTROPHIC_DFNR))
        acc["coverage"].append(1.0 if abs(tau - tau_source) > 1e-12 else 0.0)
    out = {k_: float(np.mean(v)) for k_, v in acc.items()}
    out["tau"] = float("nan")
    out["n_repeats"] = int(len(acc["ba"]))
    return out


# --------------------------------------------------------------------------- #
# Category-aware block bootstrap (B=5000, same block structure as PB7-P0)
# --------------------------------------------------------------------------- #
BOOT_B = 5000
BOOT_SEED = 20260820


def block_bootstrap(df: pd.DataFrame, seed: int = BOOT_SEED, b: int = BOOT_B) -> Dict[str, Any]:
    """Category-aware block bootstrap of mean delta_ba / delta_fnr / harm.

    df must contain columns: category, delta_ba, delta_fpr, delta_fnr, harm.
    Returns mean and 2.5/97.5 percentiles for delta_ba, delta_fnr, harm.
    """
    valid = df.dropna(subset=["delta_ba", "delta_fpr", "delta_fnr", "harm"])
    n = len(valid)
    if n == 0:
        return {"n_entities": 0, "boot_mean_delta_ba": float("nan"),
                "boot_ci_lo_delta_ba": float("nan"), "boot_ci_hi_delta_ba": float("nan"),
                "boot_mean_delta_fnr": float("nan"), "boot_ci_lo_delta_fnr": float("nan"),
                "boot_ci_hi_delta_fnr": float("nan"),
                "boot_mean_harm_rate": float("nan"), "boot_ci_lo_harm": float("nan"),
                "boot_ci_hi_harm": float("nan")}
    dba = valid["delta_ba"].to_numpy()
    dfnr = valid["delta_fnr"].to_numpy()
    harm = valid["harm"].to_numpy()
    cats = valid["category"].to_numpy()
    uniq = np.unique(cats)
    rng = np.random.default_rng(seed)
    ests_dba = np.empty(b); ests_dfnr = np.empty(b); ests_harm = np.empty(b)
    for i in range(b):
        sample_cats = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.isin(cats, sample_cats)
        ests_dba[i] = dba[idx].mean()
        ests_dfnr[i] = dfnr[idx].mean()
        ests_harm[i] = harm[idx].mean()
    return {
        "n_entities": int(n),
        "boot_mean_delta_ba": float(ests_dba.mean()),
        "boot_ci_lo_delta_ba": float(np.percentile(ests_dba, 2.5)),
        "boot_ci_hi_delta_ba": float(np.percentile(ests_dba, 97.5)),
        "boot_mean_delta_fnr": float(ests_dfnr.mean()),
        "boot_ci_lo_delta_fnr": float(np.percentile(ests_dfnr, 2.5)),
        "boot_ci_hi_delta_fnr": float(np.percentile(ests_dfnr, 97.5)),
        "boot_mean_harm_rate": float(ests_harm.mean()),
        "boot_ci_lo_harm": float(np.percentile(ests_harm, 2.5)),
        "boot_ci_hi_harm": float(np.percentile(ests_harm, 97.5)),
    }


def summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Pooled summary metrics from a per-entity frame (deltas + rates)."""
    g = df.dropna(subset=["delta_ba", "delta_fpr", "delta_fnr", "harm"])
    if len(g) == 0:
        return {k: float("nan") for k in
                ["mean_delta_ba", "mean_delta_fpr", "mean_delta_fnr", "benefit_rate",
                 "harm_rate", "catastrophic_rate", "coverage", "benefit_precision",
                 "benefit_recall", "worst_decile_delta_fnr", "n"]}
    n = len(g)
    benefit_precision = float(g.loc[g["coverage"] > 0, "benefit"].mean()) \
        if (g["coverage"] > 0).any() else float("nan")
    n_benefit = int((g["benefit"] > 0).sum())
    benefit_recall = float(n_benefit / n) if n > 0 else float("nan")
    dfnr_sorted = np.sort(g["delta_fnr"].to_numpy())
    worst_decile = float(dfnr_sorted[int(np.floor(0.9 * len(dfnr_sorted))):].mean()) \
        if len(dfnr_sorted) >= 10 else float(dfnr_sorted.mean())
    return {
        "mean_delta_ba": float(g["delta_ba"].mean()),
        "mean_delta_fpr": float(g["delta_fpr"].mean()),
        "mean_delta_fnr": float(g["delta_fnr"].mean()),
        "benefit_rate": float(g["benefit"].mean()),
        "harm_rate": float(g["harm"].mean()),
        "catastrophic_rate": float(g["catastrophic"].mean()),
        "coverage": float(g["coverage"].mean()),
        "benefit_precision": benefit_precision,
        "benefit_recall": benefit_recall,
        "worst_decile_delta_fnr": worst_decile,
        "n": float(n),
    }
