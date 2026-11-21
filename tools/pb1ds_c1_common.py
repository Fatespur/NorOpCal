"""PB1-DS C1 shared library — Category-Conditioned Selective Calibration.

Inherits the frozen C0 canonical definitions (Rule-B 10-feature set, benefit
taxonomy, monotone quantile-mapping actuator) and re-derives them under the
KNOWN-CATEGORY / UNKNOWN-CONDITION deployment assumption.

Primary held-out unit = (category, acquisition condition); both detectors of a
held-out unit are never available for fitting/threshold selection. The different
acquisition conditions of the SAME category ARE allowed in outer development
because product category is deployment-known.

Candidates (Option A detector grouping, frozen):
    A_GLOBAL            : p = h(Z, d)            NEGATIVE_CONTROL (C0-style global)
    B_CATEGORY_CENTERED : p = h(Zcentered, d)    per-category location/scale
    C_CATEGORY_CONDITIONED : p = h(Z, c_id, d)   deformation + category one-hot

Firewall (C1-specific): for a held-out unit (c,q), only target-normal and
source-normal scores may feed the adapt/abstain decision; its anomaly scores and
every outcome metric (benefit label, delta_*) are inaccessible until the fold's
model and operating point are sealed.

Evidence level: PROSPECTIVE_KNOWN_CATEGORY_HELD_OUT_CONDITION_IDENTIFIABILITY.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

import md0sr_common as C
import pb1ds_c0_common as M0  # reuse canonical features / taxonomy / QM primitives

C1_OUT = Path("reports/pb1ds_publication/c1_category_conditioned")
C0_OUT: Path = M0.C0_OUT  # frozen C0 canonical artifact directory

SEED = 42
MAX_ITER = 2000
C_GRID: List[float] = [0.1, 1.0, 10.0]
THRESH_GRID: np.ndarray = np.linspace(0.1, 0.9, 161)  # step 0.005
MIN_PRECISION: float = 0.50
EPS: float = 1e-6

CATS: List[str] = list(C.MVTEC_CATEGORIES)
BASELINES: List[str] = list(C.BASELINES)
BASELINE_NAMES: Dict[str, str] = {"dinov2_vits14": "DINOv2 Memory", "wide_resnet50_2": "PatchCore"}

CANDIDATES: List[str] = ["A_GLOBAL", "B_CATEGORY_CENTERED", "C_CATEGORY_CONDITIONED"]
CONTROLS: List[str] = ["NO_ADAPT", "ALWAYS_ON_QM", "ORACLE"]


# --------------------------------------------------------------------------- #
# unit inventory (auto-discovered from the real cache, never assumed uniform)
# --------------------------------------------------------------------------- #
def _discover_conditions(cat: str) -> List[str]:
    """Non-reference acquisition conditions present in the cache for ``cat``."""
    compat = cat  # cache uses canonical category names for C1 sources
    pat = f"mv2_{compat}_public_good_"
    conds = set()
    for p in (C.SCORES_DIR).glob(f"mv2_{compat}_public_good_*.npz"):
        stem = p.stem
        if not stem.endswith("_scores"):
            continue
        body = stem[len(pat):]
        matched_bl = None
        for bl in BASELINES:
            if body.endswith(f"_{bl}_scores"):
                matched_bl = bl
                break
        if matched_bl is None:
            continue
        cond = body[: -len(f"_{matched_bl}_scores")]
        if cond == "regular":
            continue  # regular = reference/source distribution, not a held-out condition
        conds.add(cond)
    return sorted(conds)


def discover_units() -> List[Tuple[str, str]]:
    """All PRIMARY-PAIRED units (category, condition) with both detectors present."""
    units: List[Tuple[str, str]] = []
    for cat in CATS:
        for cond in _discover_conditions(cat):
            both = all(
                C.unit_files(cat, cond, bl)[0].exists()
                and C.unit_files(cat, cond, bl)[1].exists()
                and C.unit_files(cat, cond, bl)[2].exists()
                for bl in BASELINES)
            if both:
                units.append((cat, cond))
    return units


# --------------------------------------------------------------------------- #
# canonical feature / taxonomy / metrics (reuse frozen C0 definitions)
# --------------------------------------------------------------------------- #
def compute_c0_features(source_scores, cond_scores) -> Dict[str, float]:
    return M0.compute_c0_features(source_scores, cond_scores)


def classify_change(dba: float, dfpr: float, dfnr: float) -> Tuple[int, int, int]:
    return M0.classify_change(dba, dfpr, dfnr)


def load_image_scores(rel: str) -> np.ndarray:
    return C.load_image_scores(rel)


def quantile_mapping(source_scores, target_scores) -> np.ndarray:
    return C.quantile_mapping(source_scores, target_scores)


# --------------------------------------------------------------------------- #
# access guard with normal/anomaly kinds on (category, condition)
# --------------------------------------------------------------------------- #
class AccessGuard:
    """Records file reads keyed (category, condition, baseline, kind). kind in
    {'normal','anomaly'}. A held-out (category, condition) must have ZERO reads
    (of either kind) before the fold seal."""

    def __init__(self) -> None:
        self.pre: List[Tuple[str, str, str, str]] = []
        self.post: List[Tuple[str, str, str, str]] = []
        self.mode: str = "pre"

    def seal(self) -> None:
        self.mode = "post"

    def record(self, cat: str, cond: str, bl: str, kind: str) -> None:
        key = (cat, cond, bl, kind)
        if self.mode == "pre":
            self.pre.append(key)
        else:
            self.post.append(key)

    def pre_reads(self) -> set:
        return set(self.pre)


def audit_fold(test_unit: Tuple[str, str], guard: AccessGuard) -> Dict[str, Any]:
    tc, tq = test_unit
    leaks = sorted({k for k in guard.pre_reads() if k[0] == tc and k[1] == tq})
    pre_units = sorted({(k[0], k[1]) for k in guard.pre_reads()})
    return {
        "test_unit": f"{tc}x{tq}",
        "n_pre_seal_reads": len(guard.pre_reads()),
        "n_post_seal_reads": len(guard.post),
        "test_unit_reads_before_seal": leaks,
        "firewall_pass": len(leaks) == 0,
        "pre_seal_units": pre_units,
    }


# --------------------------------------------------------------------------- #
# per-unit feature + downstream frame (with firewall-aware reads)
# --------------------------------------------------------------------------- #
def compute_unit_row(cat: str, cond: str, bl: str, guard: Optional[AccessGuard]) -> Dict[str, Any]:
    """Read ONE (cat, cond, bl) unit: features on raw normal dists + reference QM
    benefit/harm + before/after metrics. ``anomaly`` read flagged on guard."""
    if guard is not None:
        guard.record(cat, cond, bl, "normal")
    src = C.load_image_scores(C.source_rel(cat, bl))
    cnd = C.load_image_scores(C.cond_normal_rel(cat, cond, bl))
    tau = float(np.percentile(src, 95))
    feats = compute_c0_features(src, cnd)
    if guard is not None:
        guard.record(cat, cond, bl, "anomaly")
    anm = C.load_image_scores(C.anomaly_rel(cat, bl))
    mb = C.detection_metrics(np.concatenate([cnd, anm]),
                             np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))]), tau)
    cnd_c = quantile_mapping(src, cnd)
    anm_c = quantile_mapping(src, anm)
    ma = C.detection_metrics(np.concatenate([cnd_c, anm_c]),
                             np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))]), tau)
    dba = ma["ba"] - mb["ba"]
    dfpr = ma["fpr"] - mb["fpr"]
    dfnr = ma["fnr"] - mb["fnr"]
    benefit, harm, neutral = classify_change(dba, dfpr, dfnr)
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    try:
        auroc_b = float(roc_auc_score(labels, np.concatenate([cnd, anm])))
    except Exception:
        auroc_b = float("nan")
    try:
        aupr_b = float(average_precision_score(labels, np.concatenate([cnd, anm])))
    except Exception:
        aupr_b = float("nan")
    try:
        auroc_a = float(roc_auc_score(labels, np.concatenate([cnd_c, anm_c])))
    except Exception:
        auroc_a = float("nan")
    try:
        aupr_a = float(average_precision_score(labels, np.concatenate([cnd_c, anm_c])))
    except Exception:
        aupr_a = float("nan")
    return {
        "category": cat, "condition": cond, "baseline": bl, "baseline_name": BASELINE_NAMES[bl],
        "tau": tau, "n_normal": int(len(cnd)), "n_anomaly": int(len(anm)),
        "benefit": benefit, "harm": harm, "neutral": neutral,
        "delta_ba": dba, "delta_fpr": dfpr, "delta_fnr": dfnr,
        "ba_before": mb["ba"], "ba_after": ma["ba"],
        "fpr_before": mb["fpr"], "fpr_after": ma["fpr"],
        "fnr_before": mb["fnr"], "fnr_after": ma["fnr"],
        "f1_before": mb["f1"], "f1_after": ma["f1"],
        "auroc_before": auroc_b, "auroc_after": auroc_a,
        "aupr_before": aupr_b, "aupr_after": aupr_a,
        **feats,
    }


def build_frame(units: Sequence[Tuple[str, str]], guard: Optional[AccessGuard]) -> pd.DataFrame:
    rows = [compute_unit_row(cat, cond, bl, guard)
            for (cat, cond) in units for bl in BASELINES]
    return pd.DataFrame(rows)


def build_normal_only_frame(test_unit: Tuple[str, str]) -> pd.DataFrame:
    """Features for the held-out unit from normal scores ONLY (no anomaly read).
    This is the deployment-allowed evidence that feeds the adapt/abstain decision."""
    tc, tq = test_unit
    rows = []
    for bl in BASELINES:
        src = C.load_image_scores(C.source_rel(tc, bl))
        cnd = C.load_image_scores(C.cond_normal_rel(tc, tq, bl))
        tau = float(np.percentile(src, 95))
        feats = compute_c0_features(src, cnd)
        rows.append({"category": tc, "condition": tq, "baseline": bl,
                     "baseline_name": BASELINE_NAMES[bl], "tau": tau,
                     "n_normal": int(len(cnd)), "n_anomaly": -1, **feats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# candidate feature encodings (Option A: detector one-hot)
# --------------------------------------------------------------------------- #
def det_col(df: pd.DataFrame) -> np.ndarray:
    return (df["baseline"] == "wide_resnet50_2").to_numpy(int)


def cat_onehot(df: pd.DataFrame) -> np.ndarray:
    """Reference-coding: drop the first category (can) -> 7 indicator columns."""
    ref = sorted(CATS)[0]
    rows = []
    for c in sorted(CATS):
        if c == ref:
            continue
        rows.append((df["category"] == c).to_numpy(int))
    return np.column_stack(rows) if rows else np.zeros((len(df), 0))


def _raw_Z(df: pd.DataFrame) -> np.ndarray:
    return M0.process_features(df)


def encode_A(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_raw_Z(df), det_col(df)])


def category_stats(Z_raw: np.ndarray, cats: np.ndarray,
                   eps: float = EPS) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]],
                                              Tuple[np.ndarray, np.ndarray]]:
    """Per-category (mu, sigma) from the given rows, plus global fallback.
    For category-centered (C1-B) the stats MUST come from training-only rows."""
    tr_Z = np.asarray(Z_raw, dtype=float)
    tr_cats = np.asarray(cats)
    g = (tr_Z.mean(axis=0), tr_Z.std(axis=0) + eps)
    per: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for c in np.unique(tr_cats):
        m = tr_cats == c
        per[str(c)] = (tr_Z[m].mean(axis=0), tr_Z[m].std(axis=0) + eps)
    return per, g


def center_with(Z_raw: np.ndarray, cats: np.ndarray,
                per: Dict[str, Tuple[np.ndarray, np.ndarray]],
                g: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    Z = np.asarray(Z_raw, dtype=float)
    out = np.zeros_like(Z)
    for ci in range(len(Z)):
        mu, sd = per.get(str(cats[ci]), g)
        out[ci] = (Z[ci] - mu) / sd
    return out


def center_by_category(Z_raw: np.ndarray, cats: np.ndarray, train_idx: np.ndarray,
                       eps: float = EPS) -> np.ndarray:
    """Proxy kept for tests: center every row using stats from train rows only."""
    per, g = category_stats(Z_raw[train_idx], cats[train_idx], eps)
    return center_with(Z_raw, cats, per, g)


def encode_B(df: pd.DataFrame, train_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Category-centered features (train-only stats) + detector one-hot.
    If train_idx is None, the frame itself defines the stats (fit-time/all-dev)."""
    Z = _raw_Z(df)
    cats = df["category"].to_numpy()
    idx = np.arange(len(df)) if train_idx is None else np.asarray(train_idx, dtype=int)
    per, g = category_stats(Z[idx], cats[idx])
    Zc = center_with(Z, cats, per, g)
    return np.column_stack([Zc, det_col(df)])


def category_normalization_stats(train_df: pd.DataFrame) -> Dict[str, Any]:
    """DEV-derived per-category stats used to standardize the held-out unit (C1-B)."""
    Z = _raw_Z(train_df)
    per, g = category_stats(Z, train_df["category"].to_numpy())
    return {"per_category": {k: [mu.tolist(), sd.tolist()] for k, (mu, sd) in per.items()},
            "global": [g[0].tolist(), g[1].tolist()]}


def encode_B_test(test_df: pd.DataFrame, stats: Dict[str, Any]) -> np.ndarray:
    Z = _raw_Z(test_df)
    per = {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in stats["per_category"].items()}
    g = (np.asarray(stats["global"][0]), np.asarray(stats["global"][1]))
    Zc = center_with(Z, test_df["category"].to_numpy(), per, g)
    return np.column_stack([Zc, det_col(test_df)])


def encode_C(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_raw_Z(df), cat_onehot(df), det_col(df)])


def candidate_columns(cand: str, n_z: int = len(M0.C0_FEATURE_COLUMNS)) -> List[str]:
    if cand == "A_GLOBAL":
        return list(M0.C0_FEATURE_COLUMNS) + ["det_patchcore"]
    if cand == "B_CATEGORY_CENTERED":
        return [f"z_{f}" for f in M0.C0_FEATURE_COLUMNS] + ["det_patchcore"]
    if cand == "C_CATEGORY_CONDITIONED":
        return list(M0.C0_FEATURE_COLUMNS) + [f"cat_{c}" for c in sorted(CATS)[1:]] + ["det_patchcore"]
    raise ValueError(cand)


def fit_selector(X: np.ndarray, y: np.ndarray, Cval: float):
    return LogisticRegression(penalty="l2", solver="lbfgs", C=Cval,
                              class_weight="balanced", random_state=SEED,
                              max_iter=MAX_ITER).fit(X, y)


# --------------------------------------------------------------------------- #
# frozen threshold selection (utility=mean dBA; tie -> lower harm, then coverage)
# --------------------------------------------------------------------------- #
def select_threshold(scores: np.ndarray, benefit: np.ndarray, harm: np.ndarray,
                     ba_before: np.ndarray, ba_after: np.ndarray,
                     grid: np.ndarray) -> Optional[float]:
    """Maximize mean ΔBA (= mean(used_ba - before)) subject to
    precision(ADAPT->BENEFIT)>=0.50. Ties -> lower global harm, then lower coverage
    (more conservative). Returns None if no grid point satisfies the constraint
    (fold must then ABSTAIN_ALL)."""
    s = np.asarray(scores, dtype=float)
    b = np.asarray(benefit, dtype=float)
    h = np.asarray(harm, dtype=float)
    bb = np.asarray(ba_before, dtype=float)
    ba = np.asarray(ba_after, dtype=float)
    best_t: Optional[float] = None
    best_util, best_harm, best_cov = -np.inf, np.inf, np.inf
    for t in grid:
        adapt = s > t
        n = int(adapt.sum())
        if n == 0:
            continue
        prec = float(b[adapt].mean())
        if prec < MIN_PRECISION:
            continue
        util = float((np.where(adapt, ba, bb) - bb).mean())
        hgt = float(np.mean(adapt & (h == 1)))
        cov = float(adapt.mean())
        better = (util > best_util + 1e-12
                  or (abs(util - best_util) <= 1e-12 and hgt < best_harm - 1e-12)
                  or (abs(util - best_util) <= 1e-12 and abs(hgt - best_harm) <= 1e-12 and cov < best_cov))
        if better:
            best_t, best_util, best_harm, best_cov = float(t), util, hgt, cov
    return best_t


# --------------------------------------------------------------------------- #
# provenance helpers
# --------------------------------------------------------------------------- #
def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def write_json(obj: Any, name: str) -> Path:
    p = C1_OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def read_json(name: str) -> Any:
    return json.loads((C1_OUT / name).read_text(encoding="utf-8"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())