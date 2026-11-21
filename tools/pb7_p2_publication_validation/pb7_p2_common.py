"""PB7-P2 shared library — publication-readiness confirmatory validation.

This stage (PB7-P2) inherits the sealed PB7-P1 terminal
T4 PB7_P1_DETECTOR_SPECIFIC_SAFE_ACTUATOR_SUPPORTED and performs CONFIRMATORY /
PUBLICATION-READINESS validation of the frozen principal policy:

  family = A_MEDIAN_RAW, lambda = 0.25, cutoff = Q50 (median_shift gate),
  principal K = 8, safety budget mean dFNR <= +0.02, primary utility dBA,
  DINOv2-specific / detector-dependent claim scope.

This is NOT exploratory method development. The frozen method is never
modified here. Any robustness test failure is recorded, not "fixed".

Frozen principal policy (from PB7-P1 E_FROZEN_METHOD_SPEC.json):
  tau_raw = tau_source + (median_t - median_s)
  tau_used = tau_source + 0.25 * (tau_raw - tau_source)
  adapt iff median_shift >= Q50_cutoff else NoAdapt
  Q50_cutoff = 0.015890181064605713 (DEV FULL-support median_shift quantile)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))

import pb7_p1_common as P1  # type: ignore[import-not-found]  # noqa: E402

OUT = ROOT / "reports" / "pb7_p2_publication_validation"
P1_OUT = P1.OUT

# --------------------------------------------------------------------------- #
# Re-export frozen PB7-P1 semantics (no redefinition)
# --------------------------------------------------------------------------- #
Q95 = P1.Q95
DBA_MIN = P1.DBA_MIN
DFPR_MAX = P1.DFPR_MAX
DFNR_MAX = P1.DFNR_MAX
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

# --------------------------------------------------------------------------- #
# Frozen principal policy (PB7-P1 sealed)
# --------------------------------------------------------------------------- #
PRINCIPAL_FAMILY = "A_MEDIAN_RAW"
PRINCIPAL_LAMBDA = 0.25
PRINCIPAL_CUTOFF_KEY = "Q50"
PRINCIPAL_CUTOFF = 0.015890181064605713  # PB7-P1 C_GATE_DEFINITION.json Q50
PRINCIPAL_K = "8"

# Confirmatory bootstrap settings (new pre-registered seed, B=10000)
CONF_BOOT_B = 10000
CONF_BOOT_SEED = 20260821

# Repeat robustness (Stage C): >= 100 independent support repeats / eligible entity
C_REPEATS = 200


# Frozen DEV procedure for cutoff rebuild (Stage B LOO): Q50 of DEV
# FULL-support median_shift distribution, computed from the remaining DEV
# categories only (normal-only; never uses test anomaly information).
def rebuild_cutoff_q50(dev_categories: list[str]) -> float:
    """Rebuild the frozen Q50 cutoff from a DEV category subset (normal-only)."""
    fw = Firewall()
    ents = entity_list()
    scores: list[float] = []
    for cat, cond, det in ents:
        if cat not in dev_categories:
            continue
        src = load_src(cat, det, fw)
        cnd = load_cnd(cat, cond, det, fw)
        tau = source_threshold(src)
        scores.append(correctability_score(src, cnd, tau))
    if len(scores) == 0:
        raise ValueError("rebuild_cutoff_q50: empty DEV subset")
    return float(np.quantile(np.asarray(scores), 0.50))


# --------------------------------------------------------------------------- #
# Per-repeat entity evaluation (Stage C) — exposes the full repeat distribution
# --------------------------------------------------------------------------- #
def evaluate_entity_repeats(
    cat: str,
    cond: str,
    det: str,
    family: str,
    lam: float,
    c: float,
    k_value: str,
    src: np.ndarray,
    cnd_full: np.ndarray,
    anm: np.ndarray,
    tau_source: float,
    n_repeats: int = C_REPEATS,
) -> list[dict[str, Any]]:
    """Per-repeat evaluation of a bounded+abstaining actuator on one entity.

    Returns one dict per repeat (for K in {4,8}) or a single dict (FULL).
    Each dict carries the per-repeat deltas, benefit/harm/catastrophic flags,
    coverage, and the gate decision (selected=1 if adapted).
    """
    labels = np.concatenate([np.zeros(len(cnd_full)), np.ones(len(anm))])
    scores = np.concatenate([cnd_full, anm])
    before = C_detection_metrics(scores, labels, tau_source)
    n = len(cnd_full)

    def one(tau: float, selected: int) -> dict[str, Any]:
        m = C_detection_metrics(scores, labels, float(tau))
        dba = m["ba"] - before["ba"]
        dfpr = m["fpr"] - before["fpr"]
        dfnr = m["fnr"] - before["fnr"]
        return {
            "delta_ba": dba,
            "delta_fpr": dfpr,
            "delta_fnr": dfnr,
            "benefit": int(C_benefit_label(before, m)),
            "harm": int(dba < HARM_DBA or dfnr > HARM_DFNR),
            "catastrophic": int(dfnr > CATASTROPHIC_DFNR),
            "coverage": 1.0 if abs(tau - tau_source) > 1e-12 else 0.0,
            "selected": selected,
            "tau": float(tau),
        }

    if k_value == "FULL":
        if family is None:
            tau = tau_source
            sel = 0
        else:
            score = correctability_score(src, cnd_full, tau_source)
            sel = int(gate_adapt(score, c))
            tau = bounded_tau(family, src, cnd_full, tau_source, lam) if sel else tau_source
        return [one(tau, sel)]

    k = int(k_value)
    out: list[dict[str, Any]] = []
    for r in range(n_repeats):
        if k > n:
            break
        idx = support_indices(cat, cond, det, n, k_value, r)
        cnd_k = cnd_full[idx]
        if family is None:
            tau = tau_source
            sel = 0
        else:
            score = correctability_score(src, cnd_k, tau_source)
            sel = int(gate_adapt(score, c))
            tau = bounded_tau(family, src, cnd_k, tau_source, lam) if sel else tau_source
        out.append(one(tau, sel))
    return out


# --------------------------------------------------------------------------- #
# Paired category-aware block bootstrap (Stage D)
# --------------------------------------------------------------------------- #
def paired_block_bootstrap(
    df_a: pd.DataFrame, df_b: pd.DataFrame, seed: int, b: int = CONF_BOOT_B
) -> dict[str, Any]:
    """Paired category-aware bootstrap of mean(diff) between two policies.

    Within each resample of categories, both policies are restricted to the
    same resampled categories and the mean difference (A - B) is computed for
    delta_ba, delta_fnr, harm. Returns mean/CI of the differences.
    """
    a = df_a.dropna(subset=["delta_ba", "delta_fnr", "harm"]).copy()
    other = df_b.dropna(subset=["delta_ba", "delta_fnr", "harm"]).copy()
    a = a.set_index(["category", "condition", "detector"])
    other = other.set_index(["category", "condition", "detector"])
    common = a.index.intersection(other.index)
    a = a.loc[common]
    other = other.loc[common]
    n = len(a)
    if n == 0:
        return {
            "n_entities": 0,
            "diff_mean_delta_ba": float("nan"),
            "diff_ci_lo_delta_ba": float("nan"),
            "diff_ci_hi_delta_ba": float("nan"),
            "diff_mean_delta_fnr": float("nan"),
            "diff_ci_lo_delta_fnr": float("nan"),
            "diff_ci_hi_delta_fnr": float("nan"),
            "diff_mean_harm": float("nan"),
            "diff_ci_lo_harm": float("nan"),
            "diff_ci_hi_harm": float("nan"),
        }
    dba_a = a["delta_ba"].to_numpy()
    dba_b = other["delta_ba"].to_numpy()
    dfnr_a = a["delta_fnr"].to_numpy()
    dfnr_b = other["delta_fnr"].to_numpy()
    harm_a = a["harm"].to_numpy()
    harm_b = other["harm"].to_numpy()
    cats = a.index.get_level_values("category").to_numpy()
    uniq = np.unique(cats)
    rng = np.random.default_rng(seed)
    est_dba = np.empty(b)
    est_dfnr = np.empty(b)
    est_harm = np.empty(b)
    for i in range(b):
        sample_cats = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.isin(cats, sample_cats)
        est_dba[i] = dba_a[idx].mean() - dba_b[idx].mean()
        est_dfnr[i] = dfnr_a[idx].mean() - dfnr_b[idx].mean()
        est_harm[i] = harm_a[idx].mean() - harm_b[idx].mean()
    return {
        "n_entities": int(n),
        "diff_mean_delta_ba": float(est_dba.mean()),
        "diff_ci_lo_delta_ba": float(np.percentile(est_dba, 2.5)),
        "diff_ci_hi_delta_ba": float(np.percentile(est_dba, 97.5)),
        "diff_mean_delta_fnr": float(est_dfnr.mean()),
        "diff_ci_lo_delta_fnr": float(np.percentile(est_dfnr, 2.5)),
        "diff_ci_hi_delta_fnr": float(np.percentile(est_dfnr, 97.5)),
        "diff_mean_harm": float(est_harm.mean()),
        "diff_ci_lo_harm": float(np.percentile(est_harm, 2.5)),
        "diff_ci_hi_harm": float(np.percentile(est_harm, 97.5)),
    }


# --------------------------------------------------------------------------- #
# Confirmatory category-aware bootstrap (Stage D) — B=10000
# --------------------------------------------------------------------------- #
def confirmatory_bootstrap(
    df: pd.DataFrame, seed: int = CONF_BOOT_SEED, b: int = CONF_BOOT_B
) -> dict[str, Any]:
    """B=10000 category-aware block bootstrap (mean + median + 95% CI)."""
    valid = df.dropna(subset=["delta_ba", "delta_fpr", "delta_fnr", "harm"])
    n = len(valid)
    if n == 0:
        return {"n_entities": 0}
    dba = valid["delta_ba"].to_numpy()
    dfpr = valid["delta_fpr"].to_numpy()
    dfnr = valid["delta_fnr"].to_numpy()
    harm = valid["harm"].to_numpy()
    cov = valid["coverage"].to_numpy()
    cats = valid["category"].to_numpy()
    uniq = np.unique(cats)
    rng = np.random.default_rng(seed)
    est = {k: np.empty(b) for k in ["dba", "dfpr", "dfnr", "harm", "coverage"]}
    for i in range(b):
        sample_cats = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.isin(cats, sample_cats)
        est["dba"][i] = dba[idx].mean()
        est["dfpr"][i] = dfpr[idx].mean()
        est["dfnr"][i] = dfnr[idx].mean()
        est["harm"][i] = harm[idx].mean()
        est["coverage"][i] = cov[idx].mean()
    return {
        "n_entities": int(n),
        "boot_mean_delta_ba": float(est["dba"].mean()),
        "boot_median_delta_ba": float(np.median(est["dba"])),
        "boot_ci_lo_delta_ba": float(np.percentile(est["dba"], 2.5)),
        "boot_ci_hi_delta_ba": float(np.percentile(est["dba"], 97.5)),
        "boot_mean_delta_fpr": float(est["dfpr"].mean()),
        "boot_ci_lo_delta_fpr": float(np.percentile(est["dfpr"], 2.5)),
        "boot_ci_hi_delta_fpr": float(np.percentile(est["dfpr"], 97.5)),
        "boot_mean_delta_fnr": float(est["dfnr"].mean()),
        "boot_ci_lo_delta_fnr": float(np.percentile(est["dfnr"], 2.5)),
        "boot_ci_hi_delta_fnr": float(np.percentile(est["dfnr"], 97.5)),
        "boot_mean_harm_rate": float(est["harm"].mean()),
        "boot_ci_lo_harm": float(np.percentile(est["harm"], 2.5)),
        "boot_ci_hi_harm": float(np.percentile(est["harm"], 97.5)),
        "boot_mean_coverage": float(est["coverage"].mean()),
    }


# --------------------------------------------------------------------------- #
# Small helpers (avoid importing md0sr_common twice)
# --------------------------------------------------------------------------- #
def C_detection_metrics(scores: np.ndarray, labels: np.ndarray, tau: float) -> dict[str, float]:
    import md0sr_common as C  # type: ignore[import-not-found]

    return C.detection_metrics(scores, labels, float(tau))


def C_benefit_label(before: dict[str, float], after: dict[str, float]) -> int:
    import md0sr_common as C  # type: ignore[import-not-found]

    return int(C.benefit_label(before, after))


def load_principal_per_entity(k_value: str = "8", n_repeats: int = 30) -> pd.DataFrame:
    """Evaluate the frozen principal policy on TEST at a K setting (per entity).

    Uses the frozen principal family/lambda/cutoff. For K in {4,8} the
    per-repeat mean over n_repeats is returned (mirrors PB7-P1 E). For FULL a
    single deterministic evaluation is returned.
    """
    fw = Firewall()
    ents = entity_list()
    rows: list[dict[str, Any]] = []
    for cat, cond, det in ents:
        if split_side(cat) != "TEST":
            continue
        src = load_src(cat, det, fw)
        cnd = load_cnd(cat, cond, det, fw)
        anm = load_anm(cat, det, fw)
        tau_source = source_threshold(src)
        reps = evaluate_entity_repeats(
            cat,
            cond,
            det,
            PRINCIPAL_FAMILY,
            PRINCIPAL_LAMBDA,
            PRINCIPAL_CUTOFF,
            k_value,
            src,
            cnd,
            anm,
            tau_source,
            n_repeats=n_repeats,
        )
        agg: dict[str, float] = {}
        for key in [
            "delta_ba",
            "delta_fpr",
            "delta_fnr",
            "benefit",
            "harm",
            "catastrophic",
            "coverage",
        ]:
            agg[key] = float(np.mean([r[key] for r in reps]))
        rows.append({"category": cat, "condition": cond, "detector": det, "K": k_value, **agg})
    return pd.DataFrame(rows)
