"""MD0-SR shared DEV dataset assembly.

Builds the per-unit DEV table (features on RAW uncalibrated normal-score
distributions + quantile-mapping reference benefit label + before/after
reference metrics) for stages S1-S5. FINAL categories are never read here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import md0sr_common as C


def available_categories() -> Tuple[List[str], List[str]]:
    """Return (dev_cats, final_cats) from the frozen protocol + split freeze."""
    proto = json.loads((C.OUT_ROOT / "MD0SR_PROTOCOL.json").read_text(encoding="utf-8"))
    return list(proto["dev_categories"]), list(proto["final_categories"])


def build_dev_frame(categories: List[str]) -> pd.DataFrame:
    """Assemble the DEV per-unit frame with raw features + reference metrics."""
    rows: List[Dict[str, Any]] = []
    for cat in categories:
        for cond in C.MVTEC_CONDITIONS:
            for bl in C.BASELINES:
                src, tn, an = C.unit_files(cat, cond, bl)
                if not (src.exists() and tn.exists() and an.exists()):
                    continue
                source_scores = C.load_image_scores(C.source_rel(cat, bl))
                cond_scores = C.load_image_scores(C.cond_normal_rel(cat, cond, bl))
                anomaly = C.load_image_scores(C.anomaly_rel(cat, bl))
                feats = C.compute_raw_features(source_scores, cond_scores)
                tau = np.percentile(source_scores, 95)
                unit = C.build_unit(source_scores, cond_scores, anomaly, tau)
                row = {
                    "category": cat, "condition": cond, "baseline": bl, "baseline_name": C.BASELINE_NAMES[bl],
                    "tau": tau,
                    "n_normal": int(len(cond_scores)), "n_anomaly": int(len(anomaly)),
                    "benefit_ref": unit["benefit_qm"],
                    "dba_ref": unit["dba"], "dfpr_ref": unit["dfpr"], "dfnr_ref": unit["dfnr"],
                    "ba_before": unit["mb"]["ba"], "ba_after_ref": unit["ma"]["ba"],
                    "fpr_before": unit["mb"]["fpr"], "fpr_after_ref": unit["ma"]["fpr"],
                    "fnr_before": unit["mb"]["fnr"], "fnr_after_ref": unit["ma"]["fnr"],
                    "f1_before": unit["mb"]["f1"], "f1_after_ref": unit["ma"]["f1"],
                    **feats,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def per_unit_arrays(df: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, np.ndarray]]:
    """Per-unit raw score arrays for downstream recomputation: labels, tau."""
    out = {}
    for _, r in df.iterrows():
        src = C.load_image_scores(C.source_rel(r["category"], r["baseline"]))
        cond = C.load_image_scores(C.cond_normal_rel(r["category"], r["condition"], r["baseline"]))
        anom = C.load_image_scores(C.anomaly_rel(r["category"], r["baseline"]))
        out[(r["category"], r["condition"], r["baseline"])] = {
            "source": src, "cond": cond, "anomaly": anom, "tau": float(r["tau"])}
    return out


def unit_key(df_row: pd.Series) -> Tuple[str, str, str]:
    return (df_row["category"], df_row["condition"], df_row["baseline"])


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[C.FEATURE_COLUMNS].to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)


def loco_score(out_of_fold_clf_factory: Any, df: pd.DataFrame, target: str,
               seed: int = 42) -> np.ndarray:
    """Leave-one-category-out out-of-fold selector scores, aligned to df rows.

    clf_factory: callable() -> fresh sklearn classifier. target: column name.
    Returns score array in df row order (probability of positive class if proba
    available, else decision). Internal CV uses only DEV data.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    X = feature_matrix(df)
    y = df[target].to_numpy()
    groups = df["category"].to_numpy()
    scores = np.full(len(df), np.nan, dtype=float)
    loo = LeaveOneGroupOut()
    for tr_idx, te_idx in loo.split(X, y, groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        clf = out_of_fold_clf_factory()
        clf.fit(X[tr_idx], y[tr_idx])
        if hasattr(clf, "predict_proba"):
            scores[te_idx] = clf.predict_proba(X[te_idx])[:, 1]
        else:
            scores[te_idx] = clf.predict(X[te_idx])
    if np.isnan(scores).any():
        raise RuntimeError(f"LOCO produced NaN for target {target}")
    return scores