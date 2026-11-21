"""PB1-DS-PB6 tri-detector shared library — protocol-frozen 3-detector re-execution.

Extends the frozen PB1DS C1/PUB0 known-category / unknown-condition machinery to a
THIRD detector (EfficientAD) WITHOUT modifying any frozen module. Everything below
is a NEW, result-independent execution definition that reuses the frozen
primitives of md0sr_common (C), pb1ds_c0_common (M0) and pb1ds_c1_common (M):
  - frozen Rule-B 10-feature set and feature preprocessing  (M0)
  - frozen benefit/harm taxonomy                            (M0.classify_change)
  - frozen monotone quantile mapping + detection metrics    (C)
  - frozen selector fit + operating-point rule              (M.fit_selector, M.select_threshold)
  - frozen category one-hot encoding                        (M.cat_onehot)

The ONLY generalized surface is the detector one-hot encoding, which is extended
from the frozen 2-level (reference drop) representation to a 3-level one-hot with
``dinov2_vits14`` as the dropped reference level, exactly as mandated by the sealed
PB6 protocol binding ("detector_encoding: extend to 3-level categorical one-hot; no
reuse of the frozen 2-level coefficient vector").

Evidence level: PROSPECTIVE_KNOWN_CATEGORY_HELD_OUT_CONDITION_TRIDETECTOR.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import md0sr_common as C
import numpy as np
import pandas as pd
import pb1ds_c0_common as M0
import pb1ds_c1_common as M

TRI_DETECTORS: list[str] = ["dinov2_vits14", "wide_resnet50_2", "efficientad"]
REF_DETECTOR: str = "dinov2_vits14"  # dropped reference level of the one-hot
CATS: list[str] = list(M.CATS)  # 8 frozen MVTec AD 2 categories
CONDITIONS: list[str] = (
    list(M.CONDITIONS) if hasattr(M, "CONDITIONS") else list(C.MVTEC_CONDITIONS)
)  # 5 frozen acquisition conditions
TRI_DET_NAMES: dict[str, str] = {
    "dinov2_vits14": "DINOv2 Memory",
    "wide_resnet50_2": "PatchCore",
    "efficientad": "EfficientAD",
}

CANDIDATES: list[str] = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
CONTROLS: list[str] = ["NO_ADAPT", "ALWAYS_ON_QM", "ORACLE"]

# frozen constants reused verbatim from the frozen modules
C_GRID: list[float] = M.C_GRID
THRESH_GRID: np.ndarray = M.THRESH_GRID
MIN_PRECISION: float = M.MIN_PRECISION
SEED: int = M.SEED


# --------------------------------------------------------------------------- #
# detector one-hot for THREE detectors (new 3-level categorical encoding)
# --------------------------------------------------------------------------- #
def det_cols(df: pd.DataFrame) -> np.ndarray:
    """Detector one-hot, reference-coded: dinov2_vits14 dropped -> 2 columns
    (wide_resnet50_2, efficientad). Column order matches TRI_DETECTORS[1:].
    """
    return np.column_stack(
        [(df["baseline"] == det).to_numpy(int) for det in TRI_DETECTORS if det != REF_DETECTOR]
    )


def raw_Z(df: pd.DataFrame) -> np.ndarray:
    return M0.process_features(df)


# --------------------------------------------------------------------------- #
# candidate feature encodings (frozen 10 features / category / detector)
# --------------------------------------------------------------------------- #
def encode_A(df: pd.DataFrame) -> np.ndarray:
    """A_GLOBAL negative control: p = h(Z, d) -- 10 features + tri detector."""
    return np.column_stack([raw_Z(df), det_cols(df)])


def encode_D(df: pd.DataFrame) -> np.ndarray:
    """D_CATEGORY_ONLY attribution control: p = h(c, d) -- category + detector."""
    return np.column_stack([M.cat_onehot(df), det_cols(df)])


def encode_C(df: pd.DataFrame) -> np.ndarray:
    """C_CATEGORY_CONDITIONED publication candidate: p = h(Z, c, d)."""
    return np.column_stack([raw_Z(df), M.cat_onehot(df), det_cols(df)])


def candidate_columns(cand: str) -> list[str]:
    det_cols_names: list[str] = [f"det_{d}" for d in TRI_DETECTORS if d != REF_DETECTOR]
    if cand == "A_GLOBAL":
        return list(M0.C0_FEATURE_COLUMNS) + det_cols_names
    if cand == "D_CATEGORY_ONLY":
        return [f"cat_{c}" for c in sorted(M.CATS)[1:]] + det_cols_names
    if cand == "C_CATEGORY_CONDITIONED":
        return (
            list(M0.C0_FEATURE_COLUMNS) + [f"cat_{c}" for c in sorted(M.CATS)[1:]] + det_cols_names
        )
    raise ValueError(cand)


def encode_candidate(cand: str, df: pd.DataFrame) -> np.ndarray:
    if cand == "A_GLOBAL":
        return encode_A(df)
    if cand == "D_CATEGORY_ONLY":
        return encode_D(df)
    if cand == "C_CATEGORY_CONDITIONED":
        return encode_C(df)
    raise ValueError(cand)


# --------------------------------------------------------------------------- #
# tri-detector per-unit frame (normal + anomaly reads, firewall-aware)
# --------------------------------------------------------------------------- #
def compute_tri_unit_row(
    cat: str, cond: str, bl: str, guard: M.AccessGuard | None
) -> dict[str, Any]:
    """Per-(category, condition, detector) unit, mirroring the frozen
    M.compute_unit_row frame over detector ``bl`` in TRI_DETECTORS."""
    if guard is not None:
        guard.record(cat, cond, bl, "normal")
    src = C.load_image_scores(C.source_rel(cat, bl))
    cnd = C.load_image_scores(C.cond_normal_rel(cat, cond, bl))
    tau = float(np.percentile(src, 95))
    feats = M0.compute_c0_features(src, cnd)
    if guard is not None:
        guard.record(cat, cond, bl, "anomaly")
    anm = C.load_image_scores(C.anomaly_rel(cat, bl))
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    mb = C.detection_metrics(np.concatenate([cnd, anm]), labels, tau)
    cnd_c = C.quantile_mapping(src, cnd)
    anm_c = C.quantile_mapping(src, anm)
    ma = C.detection_metrics(np.concatenate([cnd_c, anm_c]), labels, tau)
    dba = ma["ba"] - mb["ba"]
    dfpr = ma["fpr"] - mb["fpr"]
    dfnr = ma["fnr"] - mb["fnr"]
    benefit, harm, neutral = M0.classify_change(dba, dfpr, dfnr)
    return {
        "category": cat,
        "condition": cond,
        "baseline": bl,
        "baseline_name": TRI_DET_NAMES[bl],
        "tau": tau,
        "n_normal": len(cnd),
        "n_anomaly": len(anm),
        "benefit": benefit,
        "harm": harm,
        "neutral": neutral,
        "delta_ba": dba,
        "delta_fpr": dfpr,
        "delta_fnr": dfnr,
        "ba_before": float(mb["ba"]),
        "ba_after": float(ma["ba"]),
        "fpr_before": float(mb["fpr"]),
        "fpr_after": float(ma["fpr"]),
        "fnr_before": float(mb["fnr"]),
        "fnr_after": float(ma["fnr"]),
        "f1_before": float(mb["f1"]),
        "f1_after": float(ma["f1"]),
        **feats,
    }


def build_tri_frame(units: Sequence[tuple[str, str]], guard: M.AccessGuard | None) -> pd.DataFrame:
    rows = [
        compute_tri_unit_row(cat, cond, bl, guard) for (cat, cond) in units for bl in TRI_DETECTORS
    ]
    return pd.DataFrame(rows)


def build_normal_only_tri_frame(test_unit: tuple[str, str]) -> pd.DataFrame:
    """Deployment-allowed evidence (normal scores only) for the held-out unit."""
    tc, tq = test_unit
    rows = []
    for bl in TRI_DETECTORS:
        src = C.load_image_scores(C.source_rel(tc, bl))
        cnd = C.load_image_scores(C.cond_normal_rel(tc, tq, bl))
        tau = float(np.percentile(src, 95))
        feats = M0.compute_c0_features(src, cnd)
        rows.append(
            {
                "category": tc,
                "condition": tq,
                "baseline": bl,
                "baseline_name": TRI_DET_NAMES[bl],
                "tau": tau,
                "n_normal": len(cnd),
                "n_anomaly": -1,
                **feats,
            }
        )
    return pd.DataFrame(rows)


def unit_groups(df: pd.DataFrame) -> np.ndarray:
    return (df["category"] + "|" + df["condition"]).to_numpy()


# --------------------------------------------------------------------------- #
# reusable frozen selector / threshold / metrics re-exports (unchanged semantics)
# --------------------------------------------------------------------------- #
def fit_selector(X, y, Cval):
    return M.fit_selector(X, y, Cval)


def select_threshold(scores, benefit, harm, ba_before, ba_after, grid):
    return M.select_threshold(scores, benefit, harm, ba_before, ba_after, grid)


def detect_metrics(scores, labels, tau):
    return C.detection_metrics(scores, labels, tau)


def quantile_mapping(source_scores, target_scores) -> np.ndarray:
    return C.quantile_mapping(source_scores, target_scores)


def load_image_scores(rel: str) -> np.ndarray:
    return C.load_image_scores(rel)
