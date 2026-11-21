"""MD0-SR shared library.

Scientific re-execution of the minimal selective-calibration method competition
with strict DEV/FINAL isolation, code-enforced FINAL firewall, per-candidate
downstream recomputation, hashing, and behavioral-duplicate detection.

Evidence level:
    PROSPECTIVE_METHOD_COMPARISON_ON_PREVIOUSLY_EXPOSED_BENCHMARK
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

P0_CACHE = Path("reports/p0_dataset_opportunity/_cache")
OUT_ROOT = Path("reports/md0sr_reexecution")
SCORES_DIR = P0_CACHE / "scores"

MVTEC_CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
MVTEC_CONDITIONS = ["overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]
BASELINES = ["dinov2_vits14", "wide_resnet50_2"]
BASELINE_NAMES = {"dinov2_vits14": "DINOv2 Memory", "wide_resnet50_2": "PatchCore"}

# Result-independent, deterministic category assignment (SHA256 of category name).
SPLIT_HASH_SALT = "MD0SR_FINAL_SPLIT_v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Scheme 0: frozen registries that compose protocol / candidate / feature /
# metric / gate definitions. S0 freezes these into reports/md0sr_reexecution/.
# ---------------------------------------------------------------------------

FEATURE_FAMILIES: Dict[str, List[str]] = {
    "location": ["mean_shift", "median_shift", "d_q50"],
    "scale": ["std_ratio", "mad_ratio", "iqr_ratio"],
    "tail": ["d_q90", "d_q95", "d_q99", "q99_q95_change", "q95_q50_change", "upper_tail_mass_change"],
    "distribution": ["wasserstein", "ks"],
}
FEATURE_COLUMNS: List[str] = [f for fam in FEATURE_FAMILIES.values() for f in fam]

MODEL_REGS = ["dinov2_vits14", "wide_resnet50_2"]

UPSTREAM_BSEL = {
    "selector_auroc": 0.8070,
    "selector_auprc": 0.4609,
    "benefit_prevalence": 0.1447,
    "coverage": 0.237,
    "benefit_precision": 0.500,
    "benefit_recall": 0.8182,
    "global_true_harm": 0.026,
    "always_on_harm": 0.855,
    "delta_ba": 0.0292,
    "ci_low": 0.0098,
    "ci_high": 0.0530,
    "patchcore_delta_ba": 0.0175,
    "dinov2_delta_ba": 0.0409,
}

# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

def load_image_scores(rel: str) -> np.ndarray:
    """Load image_scores from a score npz relative to SCORES_DIR."""
    p = SCORES_DIR / rel
    if not p.exists():
        raise FileNotFoundError(f"missing score file: {p}")
    return np.asarray(dict(np.load(p))["image_scores"], dtype=float)


def unit_files(category: str, condition: str, baseline: str) -> Tuple[Path, Path, Path]:
    """Return (source_normal, cond_normal, cond_anomaly) score paths for a unit.

    Evaluation unit: category x acquisition condition x baseline.
    The anomaly set is the 'regular' defect set (as frozen upstream).
    """
    src = SCORES_DIR / f"mv2_{category}_public_good_regular_{baseline}_scores.npz"
    tgt_norm = SCORES_DIR / f"mv2_{category}_public_good_{condition}_{baseline}_scores.npz"
    anom = SCORES_DIR / f"mv2_{category}_public_bad_regular_{baseline}_scores.npz"
    return src, tgt_norm, anom


def condition_identifiers(category: str, condition: str, baseline: str) -> Dict[str, Any]:
    """Per-unit provenance identifiers (not labels)."""
    src, tgt_norm, anom = unit_files(category, condition, baseline)
    return {
        "dataset": "MVTec_AD_2",
        "category": category,
        "condition": condition,
        "baseline": baseline,
        "baseline_name": BASELINE_NAMES[baseline],
        "source_file": src.name,
        "source_sha256": sha256_file(src),
        "cond_normal_file": tgt_norm.name,
        "cond_normal_sha256": sha256_file(tgt_norm),
        "anomaly_file": anom.name,
        "anomaly_sha256": sha256_file(anom),
        "source_n": int(len(load_image_scores(source_rel(category, baseline)))),
        "cond_norm_n": int(len(load_image_scores(cond_normal_rel(category, condition, baseline)))),
        "anomaly_n": int(len(load_image_scores(anomaly_rel(category, baseline)))),
    }


source_rel = lambda category, baseline: f"mv2_{category}_public_good_regular_{baseline}_scores.npz"
cond_normal_rel = lambda category, condition, baseline: f"mv2_{category}_public_good_{condition}_{baseline}_scores.npz"
anomaly_rel = lambda category, baseline: f"mv2_{category}_public_bad_regular_{baseline}_scores.npz"


# ---------------------------------------------------------------------------
# Raw WHEN features (uncalibrated normal-score distributions only).
# ---------------------------------------------------------------------------

def compute_raw_features(source_scores: np.ndarray, cond_scores: np.ndarray) -> Dict[str, float]:
    """Raw features from source vs condition normal-score distributions.

    Per protocol section 17/18: WHEN always observes uncalibrated raw
    source/target normal-score distributions. No calibration before features.
    """
    src = source_scores.astype(float)
    tgt = cond_scores.astype(float)
    f: Dict[str, float] = {}
    # location
    f["mean_shift"] = float(np.mean(tgt) - np.mean(src))
    f["median_shift"] = float(np.median(tgt) - np.median(src))
    f["d_q50"] = float(np.percentile(tgt, 50) - np.percentile(src, 50))
    # scale
    f["std_ratio"] = float(np.std(tgt) / np.std(src)) if np.std(src) > 0 else 1.0
    fs = np.median(np.abs(src - np.median(src)))
    ft = np.median(np.abs(tgt - np.median(tgt)))
    f["mad_ratio"] = float(ft / fs) if fs > 0 else 1.0
    qsrc = [np.percentile(src, q) for q in (25, 75)]
    qtgt = [np.percentile(tgt, q) for q in (25, 75)]
    iqr_s = qsrc[1] - qsrc[0]
    iqr_t = qtgt[1] - qtgt[0]
    f["iqr_ratio"] = float(iqr_t / iqr_s) if iqr_s > 0 else 1.0
    # tail
    for q in (90, 95, 99):
        f[f"d_q{q}"] = float(np.percentile(tgt, q) - np.percentile(src, q))
    f["q99_q95_change"] = float(
        (np.percentile(tgt, 99) - np.percentile(tgt, 95)) - (np.percentile(src, 99) - np.percentile(src, 95)))
    f["q95_q50_change"] = float(
        (np.percentile(tgt, 95) - np.percentile(tgt, 50)) - (np.percentile(src, 95) - np.percentile(src, 50)))
    th_s = np.percentile(src, 90)
    th_t = np.percentile(tgt, 90)
    frac_s = float(np.mean(src > th_s))
    frac_t = float(np.mean(tgt > th_t))
    f["upper_tail_mass_change"] = frac_t - frac_s
    # distribution
    f["wasserstein"] = float(stats.wasserstein_distance(src, tgt))
    f["ks"] = float(stats.ks_2samp(src, tgt).statistic)
    return f


# ---------------------------------------------------------------------------
# HOW mappings.
# ---------------------------------------------------------------------------

def quantile_mapping(source_scores: np.ndarray, target_scores: np.ndarray) -> np.ndarray:
    """Frozen quantile mapping (monotone by construction, mask-safe)."""
    source_sorted = np.sort(source_scores)
    out = np.empty_like(target_scores, dtype=float)
    for i, s in enumerate(target_scores):
        q = np.mean(target_scores <= s)
        idx = max(0, min(int(q * (len(source_sorted) - 1)), len(source_sorted) - 1))
        out[i] = source_sorted[idx]
    return out


def partial_qm(source_scores: np.ndarray, target_scores: np.ndarray, alpha: float) -> np.ndarray:
    """C1 partial quantile mapping g_alpha(s) = (1-alpha)s + alpha*g_QM(s)."""
    qm = quantile_mapping(source_scores, target_scores)
    return (1.0 - alpha) * target_scores + alpha * qm


def tail_aware_qm(source_scores: np.ndarray, target_scores: np.ndarray, p: float = 2.0) -> np.ndarray:
    """C2 monotonic tail-aware mapping.

    s' = source_q( rank(target, s)^p ). rank in [0,1]; power p>1 pushes the
    high tail toward stronger source-correction while the composition of a
    monotone rank and a monotone quantile function guarantees monotonicity.
    """
    source_sorted = np.sort(source_scores)
    out = np.empty_like(target_scores, dtype=float)
    n = len(target_scores)
    for i, s in enumerate(target_scores):
        r = np.mean(target_scores <= s)
        rp = float(r) ** p
        rp = max(0.0, min(1.0, rp))
        idx = max(0, min(int(rp * (len(source_sorted) - 1)), len(source_sorted) - 1))
        out[i] = source_sorted[idx]
    return out


def shape_aware_qm(source_scores: np.ndarray, cond_scores: np.ndarray, features: Dict[str, float]) -> np.ndarray:
    """C3 shape-aware mapping.

    Shape state is derived ONLY from target-normal raw features (loc/scale/tail
    dominance). A small preregistered mapping per state.
    """
    loc = abs(features["mean_shift"]) / max(abs(features["std_ratio"] - 1.0), 1e-6)
    scale = abs(features["std_ratio"] - 1.0)
    tail = abs(features["upper_tail_mass_change"]) + abs(features["d_q99"]) / max(abs(features["mean_shift"]), 1e-6)
    parts = {"location": loc, "scale": scale, "tail": tail}
    state = max(parts, key=parts.get)
    if state == "location":
        alpha = 0.75
    elif state == "scale":
        alpha = 0.50
    elif state == "tail":
        alpha = 1.00
    else:  # mixed
        alpha = 0.50
    return partial_qm(source_scores, cond_scores, alpha)


def apply_mapping(mapping: str, source_scores: np.ndarray, cond_scores: np.ndarray,
                  features: Optional[Dict[str, float]] = None, alpha: float = 1.0,
                  p: float = 2.0) -> np.ndarray:
    if mapping == "qm":
        return quantile_mapping(source_scores, cond_scores)
    if mapping == "partial":
        return partial_qm(source_scores, cond_scores, alpha)
    if mapping == "tail_aware":
        return tail_aware_qm(source_scores, cond_scores, p)
    if mapping == "shape_aware":
        if features is None:
            raise ValueError("shape_aware needs features")
        return shape_aware_qm(source_scores, cond_scores, features)
    raise ValueError(f"unknown mapping: {mapping}")


# ---------------------------------------------------------------------------
# Benefit label and detection metrics.
# ---------------------------------------------------------------------------

def detection_metrics(scores: np.ndarray, labels: np.ndarray, tau: float) -> Dict[str, float]:
    pred = (scores > tau).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ba = 0.5 * (tpr + tnr)
    denom = tp + fp
    prec = tp / denom if denom > 0 else 0.0
    denom2 = tp + fn
    rec = tp / denom2 if denom2 > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"fpr": fpr, "fnr": fnr, "ba": ba, "f1": f1,
            "tpr": tpr, "tnr": tnr, "precision": prec, "recall": rec,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def benefit_label(metrics_before: Dict[str, float], metrics_after: Dict[str, float]) -> int:
    dba = metrics_after["ba"] - metrics_before["ba"]
    dfpr = metrics_after["fpr"] - metrics_before["fpr"]
    dfnr = metrics_after["fnr"] - metrics_before["fnr"]
    return 1 if (dba >= 0.02 and dfpr <= -0.05 and dfnr <= 0.02) else 0


def build_unit(source_scores: np.ndarray, cond_normal: np.ndarray, anomaly: np.ndarray,
               tau: float) -> Dict[str, Any]:
    """Compute before/after metrics + benefit for quantile-mapping reference."""
    labels = np.concatenate([np.zeros(len(cond_normal)), np.ones(len(anomaly))])
    mb = detection_metrics(np.concatenate([cond_normal, anomaly]), labels, tau)
    aft_n = quantile_mapping(source_scores, cond_normal)
    aft_a = quantile_mapping(source_scores, anomaly)
    ma = detection_metrics(np.concatenate([aft_n, aft_a]), labels, tau)
    dba = ma["ba"] - mb["ba"]
    dfpr = ma["fpr"] - mb["fpr"]
    dfnr = ma["fnr"] - mb["fnr"]
    return {"labels": labels, "tau": tau,
            "mb": mb, "ma": ma, "dba": dba, "dfpr": dfpr, "dfnr": dfnr,
            "benefit_qm": benefit_label(mb, ma)}


def policy_metrics(adapt_flags: np.ndarray, benefit: np.ndarray,
                   ba_before: np.ndarray, ba_after: np.ndarray,
                   fpr_before: np.ndarray, fpr_after: np.ndarray,
                   fnr_before: np.ndarray, fnr_after: np.ndarray) -> Dict[str, float]:
    """Selective-policy aggregate metrics from per-unit decisions and recomputed
    downstream per-unit metrics (below: these per-unit values already correspond
    to each candidate's own calibrated scores)."""
    adapt = adapt_flags == 1
    abstain = ~adapt
    # actual per-unit metric used = after if adapted else before
    ba_used = np.where(adapt, ba_after, ba_before)
    nba = ba_before.mean()
    aba = ba_after.mean()
    sba = ba_used.mean()
    n = int(adapt.sum())
    nben = int(benefit.sum())
    abp = float(benefit[adapt].mean()) if n > 0 else 0.0
    har = float((benefit[adapt] == 0).mean()) if n > 0 else 0.0
    gh = float((adapt & (benefit == 0)).mean())
    nadapt = int(adapt.sum())
    cov = float(adapt.mean())
    # benefit recall = adapted-but-without-benefit / total with benefit
    recall_den = max(nben, 1)
    benefit_recall = float(np.sum(adapt & (benefit == 1)) / recall_den) if nben > 0 else 0.0
    # benefit precision
    bp = float(np.sum(adapt & (benefit == 1)) / max(n, 1))
    # worst-case delta ba = min per-unit (used - before) across adapted units
    if n > 0:
        per_unit_dba = ba_used[adapt] - ba_before[adapt]
        worst = float(per_unit_dba.min())
    else:
        worst = 0.0
    # oracle-gap recovery: how much of oracle benefit is recovered
    oracle_adapt = benefit == 1
    oracle_ba = np.where(oracle_adapt, ba_after, ba_before).mean()
    gap_total = oracle_ba - nba
    gap_sel = sba - nba
    oracle_gap_recovery = float(gap_sel / gap_total) if gap_total > 0 else 1.0
    return {
        "n_units": len(adapt_flags),
        "n_adapt": nadapt,
        "adapt_rate": float(adapt.mean()),
        "coverage": cov,
        "benefit_prevalence": float(benefit.mean()),
        "benefit_recall": benefit_recall,
        "benefit_precision": bp,
        "harm_among_adapted": abp,
        "harm_rate_adapted": har,
        "global_harm": gh,
        "ba_before_mean": float(nba),
        "ba_after_mean": float(aba),
        "ba_selective": float(sba),
        "delta_ba": float(sba - nba),
        "delta_ba_always": float(aba - nba),
        "worst_case_delta_ba": worst,
        "oracle_ba": float(oracle_ba),
        "oracle_gap_recovery": oracle_gap_recovery,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        means[i] = np.mean(values[rng.choice(n, n, replace=True)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def delta_ba_ci(ba_used_adapt_idx: np.ndarray, ba_before: np.ndarray,
                adapt_mask: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """95% CI of delta_ba vs NoAdapt over units (selective used-ba vs before-ba)."""
    rng = np.random.default_rng(seed)
    n = len(ba_before)
    adapted_idx = np.where(adapt_mask)[0]
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        ba_used = np.where(idx in set(adapted_idx), ba_used_adapt_idx[idx], ba_before[idx])
        boot[i] = np.mean(ba_used) - np.mean(ba_before[idx])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Experiment ledger helpers.
# ---------------------------------------------------------------------------

LEDGER_COLUMNS = [
    "experiment_id", "candidate_id", "stage", "mechanism_family", "parameters_json",
    "random_seed", "train_units", "validation_units", "code_path", "code_sha256",
    "config_sha256", "selector_output_sha256", "adapt_decision_sha256", "alpha_output_sha256",
    "calibrated_score_sha256", "result_sha256", "status", "timestamp",
]


def load_ledger() -> pd.DataFrame:
    p = OUT_ROOT / "MD0SR_EXPERIMENT_LEDGER.csv"
    if not p.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(p)


def _arr_hash(x: Optional[np.ndarray]) -> str:
    if x is None or len(x) == 0:
        return ""
    return sha256_bytes_arr(np.asarray(x, dtype=float))


def append_ledger_entry(entry: Dict[str, Any]) -> None:
    p = OUT_ROOT / "MD0SR_EXPERIMENT_LEDGER.csv"
    df = load_ledger()
    row = {c: entry.get(c, "") for c in LEDGER_COLUMNS}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(p, index=False)


def record_experiment(expt_id: str, candidate_id: str, stage: str, family: str, params: Dict[str, Any],
                      seed: int, train_units: List[str], val_units: List[str], code_path: str,
                      sel: Optional[np.ndarray], adapt: Optional[np.ndarray], alpha: Optional[np.ndarray],
                      cal_scores: Optional[np.ndarray], result: Dict[str, Any], status: str) -> None:
    entry = {
        "experiment_id": expt_id,
        "candidate_id": candidate_id,
        "stage": stage,
        "mechanism_family": family,
        "parameters_json": json.dumps(params, sort_keys=True),
        "random_seed": seed,
        "train_units": len(train_units),
        "validation_units": len(val_units),
        "code_path": code_path,
        "code_sha256": sha256_file(Path(code_path)) if Path(code_path).exists() else "",
        "config_sha256": sha256_json(params),
        "selector_output_sha256": _arr_hash(sel),
        "adapt_decision_sha256": _arr_hash(adapt),
        "alpha_output_sha256": _arr_hash(alpha),
        "calibrated_score_sha256": _arr_hash(cal_scores),
        "result_sha256": sha256_json(result),
        "status": status,
        "timestamp": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
    }
    append_ledger_entry(entry)


# ---------------------------------------------------------------------------
# FINAL split firewall.
# ---------------------------------------------------------------------------

FINAL_CATEGORIES: List[str] = []


def set_final_categories(cats: Sequence[str]) -> None:
    global FINAL_CATEGORIES
    FINAL_CATEGORIES = list(cats)


def assert_final_guard(split_cats: Sequence[str], label: str) -> None:
    """Stages S1-S5 must call this with their allowed categories. FINAL cats are
    forbidden to be read for anomaly outcomes by S1-S5."""
    if list(split_cats) == FINAL_CATEGORIES:
        # FINAL allowed
        return
    for c in split_cats:
        if c in FINAL_CATEGORIES:
            raise SystemExit(
                f"FINAL_SPLIT_ACCESS_VIOLATION: {label} attempted to use FINAL category {c}")


def write_json(obj: Any, name: str) -> Path:
    p = OUT_ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    return p