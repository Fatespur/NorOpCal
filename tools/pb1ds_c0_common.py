"""PB1-DS C0 shared library — canonical BSEL definition and nested-LOCO machinery.

This module defines the CANONICAL BSEL method from first principles (Rule B, the
mechanistic 10-feature set), the frozen benefit/harm taxonomy, the frozen
threshold-selection procedure, and access-guarded unit assembly that enforces the
outer-fold firewall (a held-out category's score files must not be read before the
fold's model/operating point is sealed).

It writes ALL evidence under reports/pb1ds_publication/c0_canonical/ and never
touches MD0-SR or P1RS-HV outputs. Historical fitted instances are NOT reused.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

import md0sr_common as C  # reuse frozen quantile-mapping / detection / benefit primitives

C0_OUT = Path("reports/pb1ds_publication/c0_canonical")

CATS: List[str] = list(C.MVTEC_CATEGORIES)          # 8 MVTec AD 2 categories
CONDITIONS: List[str] = list(C.MVTEC_CONDITIONS)      # 5 acquisition conditions
BASELINES: List[str] = list(C.BASELINES)              # 2 detectors
BL_NAMES: Dict[str, str] = {"dinov2_vits14": "DINOv2 Memory", "wide_resnet50_2": "PatchCore"}

# --------------------------------------------------------------------------- #
# Canonical feature set (Rule B, mechanistic, result-independent, <= 10 features)
# --------------------------------------------------------------------------- #
C0_FEATURE_FAMILIES: Dict[str, List[str]] = {
    "location": ["mean_shift", "median_shift"],
    "scale": ["mad_ratio", "std_ratio"],
    "tail": ["d_q90", "d_q95", "d_q99", "upper_tail_spread_change"],
    "distribution": ["wasserstein", "ks"],
}
C0_FEATURE_COLUMNS: List[str] = [f for fam in C0_FEATURE_FAMILIES.values() for f in fam]


def sealed_canonical_feature_columns(
    freeze_rel: str = "reports/pb1ds_publication/c1_category_conditioned/PB1DS_C1_FEATURE_FREEZE.json",
) -> List[str]:
    """Return the sealed canonical 10 deformation-feature order.

    Authoritative source is the frozen PB1DS_C1_FEATURE_FREEZE.json (reuses the
    PB1DS_C0 feature freeze). We require exact ordered equality with the
    structured ``C0_FEATURE_COLUMNS``; the JSON is NEVER derived from current
    code. Used by PB5 transparency-layer label sourcing (erratum repair).
    """
    p = Path(freeze_rel)
    if p.exists():
        import json

        with open(p, encoding="utf-8") as fh:
            cols = list(json.load(fh).get("feature_columns", []))
        if cols == C0_FEATURE_COLUMNS:
            return cols
        raise ValueError(
            "sealed canonical feature order mismatches C0_FEATURE_COLUMNS; "
            "solvable only by authorized audit, not by guessing"
        )
    # no sealed freeze present -> fall back to the structured canonical set
    # (never a hand-typed copy; strict ordered equality is enforced by callers)
    return list(C0_FEATURE_COLUMNS)

# Frozen hyper-parameter / grid choices (frozen BEFORE any C0 outcome evaluation).
C_GRID: List[float] = [0.1, 1.0, 10.0]
THRESH_GRID: np.ndarray = np.linspace(0.1, 0.9, 161)  # step 0.005
MIN_PRECISION: float = 0.50    # frozen: precision(ADAPT -> BENEFIT) >= 0.50
SEED: int = 42
MAX_ITER: int = 2000


def compute_c0_features(source_scores: np.ndarray, cond_scores: np.ndarray) -> Dict[str, float]:
    """Canonical (Rule B) features on raw source/target normal-score distributions.

    No calibration before feature extraction; all features are monotonically
    invariant transforms of the two raw score distributions only.
    """
    src = np.asarray(source_scores, dtype=float)
    tgt = np.asarray(cond_scores, dtype=float)
    f: Dict[str, float] = {}
    # location
    f["mean_shift"] = float(np.mean(tgt) - np.mean(src))
    f["median_shift"] = float(np.median(tgt) - np.median(src))
    # scale
    f["mad_ratio"] = float(
        np.median(np.abs(tgt - np.median(tgt))) / np.median(np.abs(src - np.median(src)))
    ) if np.median(np.abs(src - np.median(src))) > 0 else 1.0
    f["std_ratio"] = float(np.std(tgt) / np.std(src)) if np.std(src) > 0 else 1.0
    # tail (upper-tail spread change uses q99-q95 gap, requested as upper-tail spread)
    for q in (90, 95, 99):
        f[f"d_q{q}"] = float(np.percentile(tgt, q) - np.percentile(src, q))
    f["upper_tail_spread_change"] = float(
        (np.percentile(tgt, 99) - np.percentile(tgt, 95))
        - (np.percentile(src, 99) - np.percentile(src, 95)))
    # distribution
    f["wasserstein"] = float(stats.wasserstein_distance(src, tgt))
    f["ks"] = float(stats.ks_2samp(src, tgt).statistic)
    return f


def process_features(df: pd.DataFrame) -> np.ndarray:
    """Frozen feature preprocessing: raw values + NaN/inf replacement, NO scaling."""
    X = df[C0_FEATURE_COLUMNS].to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)


def classify_change(dba: float, dfpr: float, dfnr: float) -> Tuple[int, int, int]:
    """Frozen change taxonomy: returns (benefit, harm, neutral) mutually exclusive.

    benefit : full frozen rule (dBA>=+0.02 AND dFPR<=-0.05 AND dFNR<=+0.02).
    harm    : material degradation that is not benefit (dBA<=-0.02 OR dFPR>=+0.05).
    neutral : neither benefit nor harm.
    The benefit/harm dichotomy is guaranteed disjoint by the FPR and BA signs.
    """
    benefit = 1 if (dba >= 0.02 and dfpr <= -0.05 and dfnr <= 0.02) else 0
    if benefit == 1:
        return 1, 0, 0
    harm = 1 if (dba <= -0.02 or dfpr >= 0.05) else 0
    return 0, harm, 1 - harm


def build_unit_frame(categories: Sequence[str], guard: "AccessGuard") -> pd.DataFrame:
    """Assemble a per-unit frame for the given categories (features + reference
    QM benefit/harm + before/after detection metrics). Every score-file read is
    recorded on guard for the fold-access audit."""
    rows: List[Dict[str, Any]] = []
    for cat in categories:
        for cond in CONDITIONS:
            for bl in BASELINES:
                if not C.unit_files(cat, cond, bl)[0].exists() or \
                   not C.unit_files(cat, cond, bl)[1].exists() or \
                   not C.unit_files(cat, cond, bl)[2].exists():
                    continue
                key = (cat, cond, bl)
                guard.record(key)
                src = C.load_image_scores(C.source_rel(cat, bl))
                cnd = C.load_image_scores(C.cond_normal_rel(cat, cond, bl))
                anm = C.load_image_scores(C.anomaly_rel(cat, bl))
                tau = float(np.percentile(src, 95))
                feats = compute_c0_features(src, cnd)
                mb = C.detection_metrics(np.concatenate([cnd, anm]),
                                         np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))]), tau)
                ma = C.detection_metrics(
                    np.concatenate([C.quantile_mapping(src, cnd), C.quantile_mapping(src, anm)]),
                    np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))]), tau)
                dba = ma["ba"] - mb["ba"]
                dfpr = ma["fpr"] - mb["fpr"]
                dfnr = ma["fnr"] - mb["fnr"]
                benefit, harm, neutral = classify_change(dba, dfpr, dfnr)
                labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
                auroc_before = float(roc_auc_score(labels, np.concatenate([cnd, anm])))
                aupr_before = float(average_precision_score(labels, np.concatenate([cnd, anm])))
                auroc_after = float(roc_auc_score(
                    labels, np.concatenate([C.quantile_mapping(src, cnd), C.quantile_mapping(src, anm)])))
                aupr_after = float(average_precision_score(
                    labels, np.concatenate([C.quantile_mapping(src, cnd), C.quantile_mapping(src, anm)])))
                row = {
                    "category": cat, "condition": cond, "baseline": bl, "baseline_name": BL_NAMES[bl],
                    "tau": tau, "n_normal": int(len(cnd)), "n_anomaly": int(len(anm)),
                    "benefit": benefit, "harm": harm, "neutral": neutral,
                    "dba": dba, "dfpr": dfpr, "dfnr": dfnr,
                    "ba_before": mb["ba"], "ba_after": ma["ba"],
                    "fpr_before": mb["fpr"], "fpr_after": ma["fpr"],
                    "fnr_before": mb["fnr"], "fnr_after": ma["fnr"],
                    "f1_before": mb["f1"], "f1_after": ma["f1"],
                    "auroc_before": auroc_before, "auroc_after": auroc_after,
                    "aupr_before": aupr_before, "aupr_after": aupr_after,
                    **feats,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def apply_guard_scores(frame: pd.DataFrame, adapt: np.ndarray,
                       guard: "AccessGuard") -> Dict[str, np.ndarray]:
    """Return per-unit used detection metrics and policy quantities. adapt==1 ->
    calibrated (after) metrics; else raw (before) metrics."""
    adapt = adapt.astype(bool)
    ba_used = np.where(adapt, frame["ba_after"].to_numpy(), frame["ba_before"].to_numpy())
    fpr_used = np.where(adapt, frame["fpr_after"].to_numpy(), frame["fpr_before"].to_numpy())
    fnr_used = np.where(adapt, frame["fnr_after"].to_numpy(), frame["fnr_before"].to_numpy())
    f1_used = np.where(adapt, frame["f1_after"].to_numpy(), frame["f1_before"].to_numpy())
    auroc_used = np.where(adapt, frame["auroc_after"].to_numpy(), frame["auroc_before"].to_numpy())
    aupr_used = np.where(adapt, frame["aupr_after"].to_numpy(), frame["aupr_before"].to_numpy())
    return {"ba_used": ba_used, "fpr_used": fpr_used, "fnr_used": fnr_used,
            "f1_used": f1_used, "auroc_used": auroc_used, "aupr_used": aupr_used}


def policy_utility(adapt: np.ndarray, benefit: np.ndarray,
                   ba_before: np.ndarray, ba_after: np.ndarray) -> Tuple[float, float]:
    """(selective BA, precision(ADAPT->BENEFIT)). adapt decisions already fixed."""
    sel_ba = float(np.where(adapt, ba_after, ba_before).mean())
    n = int(adapt.sum())
    prec = float(benefit[adapt].mean()) if n > 0 else 0.0
    return sel_ba, prec


def select_threshold(scores: np.ndarray, benefit: np.ndarray,
                     ba_before: np.ndarray, ba_after: np.ndarray,
                     grid: np.ndarray) -> Optional[float]:
    """Frozen operating-point selection: maximize policy balanced accuracy
    subject to precision(ADAPT->BENEFIT)>=0.50. Ties -> lower coverage
    (more conservative / higher threshold). Returns None if no grid point
    satisfies the precision constraint (fold must then abstain everywhere)."""
    s = np.asarray(scores, dtype=float)
    b = np.asarray(benefit, dtype=float)
    bb = np.asarray(ba_before, dtype=float)
    ba = np.asarray(ba_after, dtype=float)
    best_t: Optional[float] = None
    best_util, best_cov = -np.inf, np.inf
    for t in grid:
        adapt = s > t
        if int(adapt.sum()) == 0:
            continue
        prec = float(b[adapt].mean())
        if prec < MIN_PRECISION:
            continue
        util = float(np.where(adapt, ba, bb).mean())
        cov = float(adapt.mean())
        if util > best_util + 1e-12 or (abs(util - best_util) <= 1e-12 and cov < best_cov):
            best_t, best_util, best_cov = float(t), util, cov
    return best_t


# --------------------------------------------------------------------------- #
# Access guard — outer-fold firewall audit
# --------------------------------------------------------------------------- #
class AccessGuard:
    """Records every unit whose score files are read, split into PRE_FREEZE and
    POST_FREEZE. A held-out category must never appear in the pre-freeze set."""

    def __init__(self) -> None:
        self.pre: List[Tuple[str, str, str]] = []
        self.post: List[Tuple[str, str, str]] = []
        self.mode: str = "pre"

    def seal_dev(self) -> None:
        self.mode = "post"

    def record(self, key: Tuple[str, str, str]) -> None:
        if self.mode == "pre":
            self.pre.append(key)
        else:
            self.post.append(key)

    def pre_freezed_keys(self) -> set:
        return set(self.pre)

    def post_freezed_keys(self) -> set:
        return set(self.post)


def audit_fold(test_cat: str, guard: AccessGuard) -> Dict[str, Any]:
    pre = guard.pre_freezed_keys()
    leaks = sorted({k for k in pre if k[0] == test_cat})
    return {
        "test_category": test_cat,
        "n_pre_freeze_unit_reads": len(pre),
        "n_post_freeze_unit_reads": len(guard.post_freezed_keys()),
        "test_category_keys_read_before_freeze": leaks,
        "firewall_pass": len(leaks) == 0,
        "pre_freeze_categories_sorted": sorted({k[0] for k in pre}),
    }


# --------------------------------------------------------------------------- #
# block (category x condition) bootstrap, paired across detectors
# --------------------------------------------------------------------------- #
def block_bootstrap_oof(oof: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    blocks: List[pd.DataFrame] = [g for _, g in oof.groupby(["category", "condition"])]
    rng = np.random.default_rng(seed)
    nb = len(blocks)
    boot_dba: List[float] = []
    boot_dfpr: List[float] = []
    boot_dfnr: List[float] = []
    boot_dharm: List[float] = []
    for _ in range(n_boot):
        sel = [blocks[i] for i in rng.integers(0, nb, size=nb)]
        s = pd.concat(sel, ignore_index=True)
        adapt = (s["adapt"] == 1).to_numpy()
        bb = s["ba_before"].to_numpy(); ba = s["ba_after"].to_numpy()
        fprb = s["fpr_before"].to_numpy(); fprb2 = s["fpr_after"].to_numpy()
        fnrb = s["fnr_before"].to_numpy(); fnra = s["fnr_after"].to_numpy()
        harm = s["harm"].to_numpy()
        ba_used = np.where(adapt, ba, bb)
        boot_dba.append(float((ba_used - bb).mean()))
        boot_dfpr.append(float((np.where(adapt, fprb2, fprb) - fprb).mean()))
        boot_dfnr.append(float((np.where(adapt, fnra, fnrb) - fnrb).mean()))
        sel_harm = np.where(adapt, harm, 0.0)
        boot_dharm.append(float((sel_harm - harm).mean()))
    return {
        "delta_ba_vs_noadapt": _ci(boot_dba),
        "delta_fpr": _ci(boot_dfpr),
        "delta_fnr": _ci(boot_dfnr),
        "global_harm_diff_vs_always_on": _ci(boot_dharm),
        "n_boot": n_boot,
    }


def _ci(vals: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()),
            "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5))}


# --------------------------------------------------------------------------- #
# provenance / json helpers
# --------------------------------------------------------------------------- #
def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def write_json(obj: Any, name: str) -> Path:
    p = C0_OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def read_json(name: str) -> Any:
    return json.loads((C0_OUT / name).read_text(encoding="utf-8"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fit_selector(X: np.ndarray, y: np.ndarray, Cval: float):
    return LogisticRegression(penalty="l2", solver="lbfgs", C=Cval,
                              class_weight="balanced", random_state=SEED, max_iter=MAX_ITER).fit(X, y)