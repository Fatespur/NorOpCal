"""PB7-P0 Stage C — Few-Normal Robustness.

Evaluates whether the B-stage normal-only deformation signal is preserved at
small target-normal support sizes. Frozen K semantics (P0): K in {4, 8, FULL},
K_REPEATS = 30, NESTED rule (S4 subset S8 subset SFULL), fixed seed list.

For every entity and every (K, repeat) cell the B-stage features are recomputed
from the K-sampled target-normal support (DEPLOYABLE path). Oracle targets
(delta_tau*_SAFE, CORRECTABLE_SAFE) are the FULL-support oracle diagnostics from
Stage A and are never used to fit anything here.

C2 metrics per feature per K: bias vs FULL, within-entity variance, rank
correlation with FULL, sign consistency, and correctability-discrimination
stability (AUC of the K-sampled feature vs CORRECTABLE_SAFE).

C3 K8 gate (pre-registered):
  1. principal-feature sign consistency at K8 >= 0.80
  2. deformation signal not swamped: between-entity std / mean within-entity std >= 2.0
  3. detector-level pattern preserved: Spearman of per-detector means (K8 vs FULL) >= 0.8
  4. correctability discrimination retained at K8: mean AUC >= 0.55
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p0_threshold_correctability"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p0_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT = P.OUT

# Frozen C3 K8 gate criteria (pre-registered)
K8_SIGN_CONSISTENCY_MIN = 0.80
K8_SIGNAL_SWAMP_RATIO_MIN = 2.0
K8_DETECTOR_PATTERN_RHO_MIN = 0.80
K8_CORR_AUC_MIN = 0.55

# B-stage key features (the PASS-B / signal features; all B features computed)
KEY_FEATURES = ["median_shift", "mean_shift", "mad_ratio", "std_ratio",
                "d_q90", "d_q95", "d_q99", "q99_q95_change",
                "wasserstein", "ks", "T1", "T2", "T3"]
# Principal feature for the K8 gate (strongest PASS-B signal)
PRINCIPAL = "median_shift"


def compute_features(src: np.ndarray, cnd: np.ndarray, tau_source: float) -> Dict[str, float]:
    raw = C.compute_raw_features(src, cnd)
    f = {name: float(raw[name]) for name in KEY_FEATURES if name in raw}
    f["T1"] = float(np.mean(cnd <= tau_source))
    f["T2"] = float(np.percentile(cnd, 95) - np.percentile(src, 95))
    f["T3"] = float(np.mean(cnd > tau_source))
    return f


def sign_auc(feat: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(C.roc_auc_score(labels, feat))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fw = P.Firewall()
    ents = P.entity_list()
    head = pd.read_csv(OUT / "A2_ORACLE_HEADROOM.csv")

    # FULL-support features (B-stage reference) + per-K sampled features
    full_rows: List[Dict[str, Any]] = []
    sampled_rows: List[Dict[str, Any]] = []

    for cat, cond, det in ents:
        src = P.load_src(cat, det, fw)
        cnd_full = P.load_cnd(cat, cond, det, fw)
        tau_src = P.source_threshold(src)
        n = len(cnd_full)
        h = head[(head.category == cat) & (head.condition == cond) & (head.detector == det)].iloc[0]

        f_full = compute_features(src, cnd_full, tau_src)
        full_rows.append({
            "category": cat, "condition": cond, "detector": det,
            "tau_source": tau_src, "n_full": n,
            "delta_tau_safe": h["delta_tau_safe"], "correctable_safe": int(h["correctable_safe"]),
            **f_full,
        })

        for k_value in ["4", "8"]:
            k = int(k_value)
            if k > n:
                continue  # entity ineligible for this K (FULL_n < K)
            for r in range(P.K_REPEATS):
                idx = P.support_indices(cat, cond, det, n, k_value, r)
                cnd_k = cnd_full[idx]
                f_k = compute_features(src, cnd_k, tau_src)
                sampled_rows.append({
                    "category": cat, "condition": cond, "detector": det,
                    "K": k_value, "repeat": r, "n_k": int(len(idx)),
                    "correctable_safe": int(h["correctable_safe"]),
                    **f_k,
                })

    df_full = pd.DataFrame(full_rows)
    df_sampled = pd.DataFrame(sampled_rows)
    df_full.to_csv(OUT / "C_FULL_FEATURES.csv", index=False)
    df_sampled.to_csv(OUT / "C_SAMPLED_FEATURES.csv", index=False)

    # ---- C2: stability metrics per feature per K --------------------------- #
    stab_rows: List[Dict[str, Any]] = []
    for feat in KEY_FEATURES:
        for k_value in ["4", "8"]:
            sub = df_sampled[df_sampled["K"] == k_value]
            # per-entity mean/std across repeats
            g = sub.groupby(["category", "condition", "detector"])[feat]
            mean_k = g.mean()
            std_k = g.std()
            full_map = df_full.set_index(["category", "condition", "detector"])[feat]
            aligned = pd.DataFrame({"mean_k": mean_k, "std_k": std_k,
                                    "full": full_map.reindex(mean_k.index)}).dropna()
            if len(aligned) == 0:
                continue
            bias = float((aligned["mean_k"] - aligned["full"]).mean())
            mean_std = float(aligned["std_k"].mean())
            between_std = float(aligned["mean_k"].std())
            rank_corr = float(stats.spearmanr(aligned["mean_k"], aligned["full"]).statistic)
            # sign consistency (for features with a meaningful sign)
            if feat in ("median_shift", "mean_shift", "d_q90", "d_q95", "d_q99",
                        "q99_q95_change", "T2"):
                sign_consistent = float(np.mean(
                    np.sign(aligned["mean_k"]) == np.sign(aligned["full"])))
            elif feat in ("mad_ratio", "std_ratio"):
                sign_consistent = float(np.mean(
                    np.sign(aligned["mean_k"] - 1) == np.sign(aligned["full"] - 1)))
            else:
                sign_consistent = float("nan")
            # correctability discrimination stability
            aucs = []
            for r in range(P.K_REPEATS):
                rr = sub[sub["repeat"] == r]
                if len(np.unique(rr["correctable_safe"])) >= 2:
                    aucs.append(sign_auc(rr[feat].to_numpy(), rr["correctable_safe"].to_numpy()))
            auc_mean = float(np.mean(aucs)) if aucs else float("nan")
            auc_std = float(np.std(aucs)) if aucs else float("nan")
            stab_rows.append({
                "feature": feat, "K": k_value,
                "bias_vs_full": bias, "mean_within_std": mean_std,
                "between_std": between_std, "swamp_ratio": between_std / mean_std if mean_std > 0 else float("nan"),
                "rank_corr_with_full": rank_corr, "sign_consistency": sign_consistent,
                "corr_auc_mean": auc_mean, "corr_auc_std": auc_std,
                "n_entities": int(len(aligned)),
            })
    df_stab = pd.DataFrame(stab_rows)
    df_stab.to_csv(OUT / "C_K_STABILITY.csv", index=False)

    # ---- C3: K8 gate ------------------------------------------------------- #
    sub8 = df_sampled[df_sampled["K"] == "8"]
    g8 = sub8.groupby(["category", "condition", "detector"])[PRINCIPAL]
    mean8 = g8.mean()
    std8 = g8.std()
    full_p = df_full.set_index(["category", "condition", "detector"])[PRINCIPAL]
    aligned8 = pd.DataFrame({"mean8": mean8, "std8": std8, "full": full_p.reindex(mean8.index)}).dropna()

    sign_consistency_k8 = float(np.mean(
        np.sign(aligned8["mean8"]) == np.sign(aligned8["full"])))
    mean_within_std8 = float(aligned8["std8"].mean())
    between_std8 = float(aligned8["mean8"].std())
    swamp_ratio8 = between_std8 / mean_within_std8 if mean_within_std8 > 0 else float("nan")

    det_mean_full = df_full.groupby("detector")[PRINCIPAL].mean()
    det_mean_k8 = sub8.groupby("detector")[PRINCIPAL].mean()
    det_rho = float(stats.spearmanr(det_mean_full, det_mean_k8).statistic)

    aucs8 = []
    for r in range(P.K_REPEATS):
        rr = sub8[sub8["repeat"] == r]
        if len(np.unique(rr["correctable_safe"])) >= 2:
            aucs8.append(sign_auc(rr[PRINCIPAL].to_numpy(), rr["correctable_safe"].to_numpy()))
    corr_auc_k8 = float(np.mean(aucs8)) if aucs8 else float("nan")

    checks = {
        "sign_consistency_k8": sign_consistency_k8,
        "sign_consistency_k8_ok": sign_consistency_k8 >= K8_SIGN_CONSISTENCY_MIN,
        "swamp_ratio_k8": swamp_ratio8,
        "swamp_ratio_k8_ok": (not np.isnan(swamp_ratio8)) and swamp_ratio8 >= K8_SIGNAL_SWAMP_RATIO_MIN,
        "detector_pattern_rho_k8": det_rho,
        "detector_pattern_rho_k8_ok": (not np.isnan(det_rho)) and det_rho >= K8_DETECTOR_PATTERN_RHO_MIN,
        "corr_auc_k8": corr_auc_k8,
        "corr_auc_k8_ok": (not np.isnan(corr_auc_k8)) and corr_auc_k8 >= K8_CORR_AUC_MIN,
    }
    gate_pass = all(checks[k + "_k8_ok"] for k in
                    ["sign_consistency", "swamp_ratio", "detector_pattern_rho", "corr_auc"])
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "C",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "principal_feature": PRINCIPAL,
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_D" if gate_pass else
                    "PRACTICALLY_INSUFFICIENT_FEW_NORMAL_SUPPORT",
    }
    (OUT / "C_GATE.json").write_text(json.dumps(gate, indent=2, sort_keys=False), encoding="utf-8")

    # ---- ledger ------------------------------------------------------------ #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame(
        columns=["stage", "step", "status", "timestamp", "artifacts", "note"])
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "C",
        "step": "few_normal_robustness_k4_k8_full",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": "C_FULL_FEATURES.csv, C_SAMPLED_FEATURES.csv, "
                     "C_K_STABILITY.csv, C_GATE.json",
        "note": f"C gate {'PASS -> PROCEED_TO_D' if gate_pass else 'FAIL -> terminal'}; "
                f"K8 sign_consistency={sign_consistency_k8:.3f}, swamp_ratio={swamp_ratio8:.2f}, "
                f"corr_auc={corr_auc_k8:.3f}",
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print("\nK stability (key features):")
    print(df_stab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
