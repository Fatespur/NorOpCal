"""PB1-DS PB2 statistics — category-condition block bootstrap and block
benefit-label permutation, both at the frozen B=5000 with empirical p v
=(k+1)/(B+1).

Block invariant (frozen PB1): a resampling/permutation unit is a
(category, condition) block; the two detector rows of a unit are always
resampled / reshuffled together. Row-level (iid) bootstrap is forbidden.

Targets:
  - method-wise mean ΔBA (vs NO_ADAPT) 95% CI + global-harm CI;
  - paired (category-condition block) bootstrap over frozen practical
    comparators (D/C vs NO_ADAPT, vs ALWAYS_ON, C vs D, vs strongest external);
  - block-preserving benefit-label permutation (B=5000) for the selective
    methods (A_GLOBAL, D_CATEGORY_ONLY, C_CATEGORY_CONDITIONED).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def _ci(vals: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    return {"mean": float(a.mean()),
            "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5))}


def canonical_block_order(frame: pd.DataFrame) -> List[Tuple[str, str]]:
    return sorted(set(map(tuple, frame[["category", "condition"]].values)), key=lambda k: (k[0], k[1]))


def _blocks_by_unit(frame: pd.DataFrame) -> List[pd.DataFrame]:
    return [g for _, g in frame.groupby(["category", "condition"])]


def _metric_parts(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    adapt = (frame["adapt"] == 1).to_numpy(bool)
    return {
        "adapt": adapt,
        "ba_before": frame["ba_before"].to_numpy(float),
        "ba_after": frame["ba_after"].to_numpy(float),
        "harm": frame["harm"].to_numpy(int),
        "benefit": frame["benefit"].to_numpy(int),
        "prob": frame["selector_prob"].to_numpy(float),
    }


def _concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k in parts[0]:
        out[k] = np.concatenate([p[k] for p in parts])
    return out


def _dba(p: Dict[str, np.ndarray]) -> float:
    used = np.where(p["adapt"], p["ba_after"], p["ba_before"])
    return float(np.mean(used - p["ba_before"]))


def _harm(p: Dict[str, np.ndarray]) -> float:
    return float(np.mean(p["adapt"] & (p["harm"] == 1)))


def _auprc(benefit: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(benefit, prob))
    except Exception:
        return float("nan")


def _auroc(benefit: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(benefit)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(benefit, prob))
    except Exception:
        return float("nan")


def method_bootstrap(frame: pd.DataFrame, n_boot: int = 5000, seed: int = 42
                     ) -> Dict[str, Any]:
    """Block bootstrap (category-condition) of mean ΔBA and global harm."""
    blocks = _blocks_by_unit(frame)
    nb = len(blocks)
    rng = np.random.default_rng(seed)
    dba_vals: List[float] = []
    harm_vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, nb, size=nb)
        p = _concat([_metric_parts(blocks[i]) for i in idx])
        dba_vals.append(_dba(p))
        harm_vals.append(_harm(p))
    return {"n_boot": n_boot, "block": "category-condition",
            "delta_ba_ci": _ci(dba_vals), "global_harm_ci": _ci(harm_vals)}


def paired_bootstrap(frame_a: pd.DataFrame, frame_b: pd.DataFrame,
                     n_boot: int = 5000, seed: int = 42) -> Dict[str, Any]:
    """Paired block bootstrap of (A - B) on the same block order."""
    order = canonical_block_order(frame_a)
    ob = canonical_block_order(frame_b)
    assert order == ob, "block order mismatch between paired frames"
    ba = {k: _metric_parts(fr) for k, fr in zip(order, _blocks_by_unit(frame_a))}
    bb = {k: _metric_parts(fr) for k, fr in zip(order, _blocks_by_unit(frame_b))}
    rng = np.random.default_rng(seed)
    nb = len(order)
    d_dba: List[float] = []
    d_harm: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, nb, size=nb)
        pa = _concat([ba[order[i]] for i in idx])
        pb = _concat([bb[order[i]] for i in idx])
        d_dba.append(_dba(pa) - _dba(pb))
        d_harm.append(_harm(pa) - _harm(pb))
    return {"n_boot": n_boot, "block": "category-condition",
            "delta_ba_A_minus_B": _ci(d_dba), "global_harm_A_minus_B": _ci(d_harm)}


def benefit_permutation(frame: pd.DataFrame, n_perm: int = 5000,
                        seed: int = 7) -> Dict[str, Any]:
    """Block-preserving benefit-label permutation against fixed selector probs.
    p = (k+1)/(B+1); never reports p=0.000."""
    blocks = []
    for _, g in frame.groupby(["category", "condition"]):
        blocks.append((g["benefit"].to_numpy(int), g["selector_prob"].to_numpy(float)))
    obs_labels = np.concatenate([b for b, _ in blocks])
    probs = np.concatenate([p for _, p in blocks])
    obs_pairs = [b for b, _ in blocks]
    obs_auprc = _auprc(obs_labels, probs)
    obs_auroc = _auroc(obs_labels, probs)
    rng = np.random.default_rng(seed)
    n_blk = len(blocks)
    perm_auprc: List[float] = []
    perm_auroc: List[float] = []
    for _ in range(n_perm):
        order = rng.permutation(n_blk)
        lab = np.concatenate([obs_pairs[i] for i in order])
        if len(np.unique(lab)) < 2:
            perm_auprc.append(float("nan"))
            perm_auroc.append(float("nan"))
            continue
        perm_auprc.append(_auprc(lab, probs))
        perm_auroc.append(_auroc(lab, probs))
    apr = [v for v in perm_auprc if not np.isnan(v)]
    aro = [v for v in perm_auroc if not np.isnan(v)]
    k_prc = int(np.sum(np.asarray(apr) >= obs_auprc)) if apr else 1
    p_prc = float((k_prc + 1) / (n_perm + 1))
    if aro and not np.isnan(obs_auroc):
        k_roc = int(np.sum(np.asarray(aro) >= obs_auroc))
        p_roc = float((k_roc + 1) / (n_perm + 1))
    else:
        p_roc = float("nan")
    return {
        "n_perm": n_perm, "block": "category-condition",
        "observed_auprc": obs_auprc, "observed_auroc": obs_auroc,
        "perm_null_auprc": _ci(apr) if apr else {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")},
        "perm_null_auroc": _ci(aro) if aro else {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")},
        "k_plus_1_over_B_plus_1": True,
        "p_value_auprc": p_prc, "p_value_auroc": p_roc,
    }