"""PB7-P0 Stage E — Strict OOF / Held-Out Evaluation of deployable actuators.

Evaluates the pre-registered simple threshold actuators (T0-T4) under strict
held-out conditions. All candidates are parameter-free pre-registered formulas
(T4 delta is a fixed pre-registered constant), so evaluating on all 117
entities is a strict OOF evaluation: each entity's target scores are used only
as that entity's evaluation set and never to fit any candidate.

Settings: primary K=8, secondary K=4, FULL. For K in {4,8} the actuator
threshold is computed from the K-sampled target-normal support (30 frozen
repeats) and evaluated on the FULL target set; metrics are averaged over
repeats. Oracle tau*_SAFE (Stage A) is reported as DIAGNOSTIC ONLY.

Metrics: BA/FPR/FNR and deltas; safety (harm rate, catastrophic harm rate,
coverage/abstention); ranking (AUROC/AUPR, expected unchanged by threshold-only
actuators). Bootstrap: B=5000 category-aware block bootstrap.

E Gate (G1-G6) is evaluated and recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p0_threshold_correctability"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p0_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT = P.OUT

BOOT_B = 5000
BOOT_SEED = 20260820
DBA_MIN = P.DBA_MIN
DFPR_MAX = P.DFPR_MAX
DFNR_MAX = P.DFNR_MAX
HARM_DBA = -0.02
HARM_DFNR = 0.02
CATASTROPHIC_DFNR = 0.05

CANDIDATES = ["T0", "T1", "T2", "T3", "T4", "ORACLE"]
K_SETTINGS = ["FULL", "8", "4"]
PRIMARY_K = "8"


def actuator_threshold(cand: str, src: np.ndarray, cnd: np.ndarray, tau_source: float) -> float:
    """Pre-registered actuator threshold from DEPLOYABLE inputs only."""
    if cand == "T0":
        return tau_source
    med_s = float(np.median(src))
    med_t = float(np.median(cnd))
    mad_s = float(np.median(np.abs(src - med_s)))
    mad_t = float(np.median(np.abs(cnd - med_t)))
    q95s = float(np.percentile(src, 95))
    q95t = float(np.percentile(cnd, 95))
    if cand == "T1":
        return tau_source + (q95t - q95s)
    if cand == "T2":
        ratio = (mad_t / mad_s) if mad_s > 1e-12 else 1.0
        return med_t + ratio * (tau_source - med_s)
    if cand == "T3":
        return q95t
    if cand == "T4":
        t3_feat = float(np.mean(cnd > tau_source))
        if t3_feat > 0.10:  # pre-registered delta = 2 * nominal exceedance
            return q95t
        return tau_source
    raise ValueError(f"unknown candidate {cand}")


def evaluate_entity(cat: str, cond: str, det: str, cand: str, k_value: str,
                    src: np.ndarray, cnd_full: np.ndarray, anm: np.ndarray,
                    tau_source: float, oracle_tau: float) -> Dict[str, Any]:
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

    if cand == "ORACLE":
        m = metrics_at(oracle_tau)
        m["tau"] = oracle_tau
        mm = C.detection_metrics(scores, labels, oracle_tau)
        m["benefit"] = int(C.benefit_label(before, mm))
        m["harm"] = int(m["delta_ba"] < HARM_DBA or m["delta_fnr"] > HARM_DFNR)
        m["catastrophic"] = int(m["delta_fnr"] > CATASTROPHIC_DFNR)
        m["coverage"] = 1.0 if abs(oracle_tau - tau_source) > 1e-12 else 0.0
        return m

    if k_value == "FULL":
        tau = actuator_threshold(cand, src, cnd_full, tau_source)
        m = metrics_at(tau)
        m["tau"] = tau
        m["benefit"] = int(C.benefit_label(before, C.detection_metrics(scores, labels, tau)))
        m["harm"] = int(m["delta_ba"] < HARM_DBA or m["delta_fnr"] > HARM_DFNR)
        m["catastrophic"] = int(m["delta_fnr"] > CATASTROPHIC_DFNR)
        m["coverage"] = 1.0 if abs(tau - tau_source) > 1e-12 else 0.0
        return m

    # K in {4, 8}: average metrics over frozen repeats
    k = int(k_value)
    acc = {"ba": [], "fpr": [], "fnr": [], "delta_ba": [], "delta_fpr": [],
           "delta_fnr": [], "benefit": [], "harm": [], "catastrophic": [], "coverage": []}
    for r in range(P.K_REPEATS):
        if k > n:
            break
        idx = P.support_indices(cat, cond, det, n, k_value, r)
        cnd_k = cnd_full[idx]
        tau = actuator_threshold(cand, src, cnd_k, tau_source)
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fw = P.Firewall()
    ents = P.entity_list()
    head = pd.read_csv(OUT / "A2_ORACLE_HEADROOM.csv")

    rows: List[Dict[str, Any]] = []
    for cat, cond, det in ents:
        src = P.load_src(cat, det, fw)
        cnd_full = P.load_cnd(cat, cond, det, fw)
        anm = P.load_anm(cat, det, fw)
        tau_source = P.source_threshold(src)
        h = head[(head.category == cat) & (head.condition == cond) & (head.detector == det)].iloc[0]
        oracle_tau = float(h["tau_star_safe"])
        for cand in CANDIDATES:
            for k_value in K_SETTINGS:
                r = evaluate_entity(cat, cond, det, cand, k_value,
                                    src, cnd_full, anm, tau_source, oracle_tau)
                rows.append({"category": cat, "condition": cond, "detector": det,
                             "candidate": cand, "K": k_value, **r})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E_PER_ENTITY.csv", index=False)

    # ---- E5: detector-stratified summary ---------------------------------- #
    summary_rows: List[Dict[str, Any]] = []
    for cand in CANDIDATES:
        for k_value in K_SETTINGS:
            sub = df[(df.candidate == cand) & (df.K == k_value)]
            for group_name, g in [("POOLED", sub)] + \
                    [(d, sub[sub.detector == d]) for d in P.DETECTORS]:
                n = len(g)
                summary_rows.append({
                    "candidate": cand, "K": k_value, "group": group_name,
                    "n": n,
                    "mean_delta_ba": float(g["delta_ba"].mean()),
                    "mean_delta_fpr": float(g["delta_fpr"].mean()),
                    "mean_delta_fnr": float(g["delta_fnr"].mean()),
                    "benefit_rate": float(g["benefit"].mean()),
                    "harm_rate": float(g["harm"].mean()),
                    "catastrophic_rate": float(g["catastrophic"].mean()),
                    "coverage": float(g["coverage"].mean()),
                    "mean_ba": float(g["ba"].mean()),
                    "mean_fpr": float(g["fpr"].mean()),
                    "mean_fnr": float(g["fnr"].mean()),
                })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT / "E_DETECTOR_SUMMARY.csv", index=False)

    # ---- E7: bootstrap (category-aware block, B=5000) --------------------- #
    rng = np.random.default_rng(BOOT_SEED)
    boot_rows: List[Dict[str, Any]] = []
    for cand in CANDIDATES:
        for k_value in K_SETTINGS:
            sub = df[(df.candidate == cand) & (df.K == k_value)]
            # drop entities ineligible for this K (NaN metrics)
            valid = sub.dropna(subset=["delta_ba", "delta_fpr", "delta_fnr", "harm"])
            if len(valid) == 0:
                boot_rows.append({"candidate": cand, "K": k_value,
                                  "boot_mean_delta_ba": float("nan"),
                                  "boot_ci_lo_delta_ba": float("nan"),
                                  "boot_ci_hi_delta_ba": float("nan"),
                                  "boot_mean_delta_fpr": float("nan"),
                                  "boot_mean_delta_fnr": float("nan"),
                                  "boot_mean_harm_rate": float("nan"),
                                  "boot_ci_lo_harm": float("nan"),
                                  "boot_ci_hi_harm": float("nan"),
                                  "n_entities": 0})
                continue
            dba = valid["delta_ba"].to_numpy()
            dfpr = valid["delta_fpr"].to_numpy()
            dfnr = valid["delta_fnr"].to_numpy()
            harm = valid["harm"].to_numpy()
            cat_arr = valid["category"].to_numpy()
            uniq = np.unique(cat_arr)
            ests_dba = np.empty(BOOT_B); ests_dfpr = np.empty(BOOT_B)
            ests_dfnr = np.empty(BOOT_B); ests_harm = np.empty(BOOT_B)
            for b in range(BOOT_B):
                sample_cats = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.isin(cat_arr, sample_cats)
                ests_dba[b] = dba[idx].mean()
                ests_dfpr[b] = dfpr[idx].mean()
                ests_dfnr[b] = dfnr[idx].mean()
                ests_harm[b] = harm[idx].mean()
            boot_rows.append({
                "candidate": cand, "K": k_value,
                "boot_mean_delta_ba": float(ests_dba.mean()),
                "boot_ci_lo_delta_ba": float(np.percentile(ests_dba, 2.5)),
                "boot_ci_hi_delta_ba": float(np.percentile(ests_dba, 97.5)),
                "boot_mean_delta_fpr": float(ests_dfpr.mean()),
                "boot_mean_delta_fnr": float(ests_dfnr.mean()),
                "boot_mean_harm_rate": float(ests_harm.mean()),
                "boot_ci_lo_harm": float(np.percentile(ests_harm, 2.5)),
                "boot_ci_hi_harm": float(np.percentile(ests_harm, 97.5)),
                "n_entities": int(len(valid)),
            })
    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(OUT / "E_BOOTSTRAP.csv", index=False)

    # ---- E4: ranking check (threshold-only should not change AUROC/AUPR) --- #
    rank_rows: List[Dict[str, Any]] = []
    for _, r in df[(df.candidate == "T0") & (df.K == "FULL")].iterrows():
        cat, cond, det = r["category"], r["condition"], r["detector"]
        cnd = P.load_cnd(cat, cond, det)
        anm = P.load_anm(cat, det)
        scores = np.concatenate([cnd, anm])
        labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
        rank_rows.append({
            "category": cat, "condition": cond, "detector": det,
            "auroc": float(C.roc_auc_score(labels, scores)),
            "aupr": float(C.average_precision_score(labels, scores)),
        })
    df_rank = pd.DataFrame(rank_rows)
    df_rank.to_csv(OUT / "E_RANKING.csv", index=False)

    # ---- E Gate ------------------------------------------------------------ #
    # Use FULL (principal deployable setting for the gate; K8 reported separately)
    def gate_metrics(cand: str, k_value: str) -> Dict[str, float]:
        sub = df[(df.candidate == cand) & (df.K == k_value)]
        b = df_boot[(df_boot.candidate == cand) & (df_boot.K == k_value)].iloc[0]
        return {
            "mean_delta_ba": float(sub["delta_ba"].mean()),
            "ci_lo_delta_ba": float(b["boot_ci_lo_delta_ba"]),
            "mean_delta_fnr": float(sub["delta_fnr"].mean()),
            "harm_rate": float(sub["harm"].mean()),
            "benefit_rate": float(sub["benefit"].mean()),
            "coverage": float(sub["coverage"].mean()),
        }

    # best Always-On simple actuator among T1/T2/T3 (FULL)
    best_always = max(["T1", "T2", "T3"], key=lambda c: gate_metrics(c, "FULL")["mean_delta_ba"])
    always = gate_metrics(best_always, "FULL")
    t4 = gate_metrics("T4", "FULL")
    t0 = gate_metrics("T0", "FULL")
    oracle = gate_metrics("ORACLE", "FULL")

    g1 = always["mean_delta_ba"] >= DBA_MIN
    g2 = always["ci_lo_delta_ba"] > 0
    g3 = (always["mean_delta_fnr"] <= DFNR_MAX) and (t4["harm_rate"] < always["harm_rate"])
    g4 = True  # detector validity: check below
    g5 = True  # K8 retains main effect: check below
    g6 = t4["harm_rate"] < always["harm_rate"] and t4["mean_delta_ba"] >= always["mean_delta_ba"] - 1e-9

    # G4: not single-detector driven (best Always-On benefit in >= 2 detectors)
    sub_always = df[(df.candidate == best_always) & (df.K == "FULL")]
    det_benefit = {d: int((sub_always[sub_always.detector == d]["benefit"] > 0).sum())
                   for d in P.DETECTORS}
    g4 = sum(1 for v in det_benefit.values() if v > 0) >= 2

    # G5: K8 retains main effect (best Always-On mean delta_ba at K8 > 0)
    always_k8 = gate_metrics(best_always, "8")
    g5 = always_k8["mean_delta_ba"] > 0

    checks = {
        "G1_benefit_delta_ba_ge_002": g1,
        "G1_detail": {"best_always_on": best_always, "mean_delta_ba": always["mean_delta_ba"]},
        "G2_ci_lower_bound_gt_0": g2,
        "G2_detail": {"ci_lo_delta_ba": always["ci_lo_delta_ba"]},
        "G3_safety_dfnr_le_002_and_harm_below_always": g3,
        "G3_detail": {"always_mean_delta_fnr": always["mean_delta_fnr"],
                      "always_harm_rate": always["harm_rate"], "t4_harm_rate": t4["harm_rate"]},
        "G4_detector_validity": g4,
        "G4_detail": det_benefit,
        "G5_k8_retains_effect": g5,
        "G5_detail": {"k8_mean_delta_ba": always_k8["mean_delta_ba"]},
        "G6_selectivity_value": g6,
        "G6_detail": {"t4_harm_rate": t4["harm_rate"], "always_harm_rate": always["harm_rate"],
                      "t4_mean_delta_ba": t4["mean_delta_ba"],
                      "always_mean_delta_ba": always["mean_delta_ba"]},
        "oracle_upper_bound_diagnostic_only": oracle,
        "t0_baseline": t0,
    }
    gate_pass = all([g1, g2, g3, g4, g5, g6])
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "E",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "DETECTOR_DEPENDENT_SAFE_THRESHOLD_ADAPTATION_SUPPORTED" if gate_pass else
                    "THRESHOLD_CORRECTABILITY_IDENTIFIED_BUT_SIMPLE_ACTUATORS_INSUFFICIENT",
    }
    (OUT / "E_GATE.json").write_text(json.dumps(gate, indent=2, sort_keys=False), encoding="utf-8")

    # ---- ledger ------------------------------------------------------------ #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame(
        columns=["stage", "step", "status", "timestamp", "artifacts", "note"])
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "E",
        "step": "strict_oof_heldout_evaluation",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": "E_PER_ENTITY.csv, E_DETECTOR_SUMMARY.csv, E_BOOTSTRAP.csv, "
                     "E_RANKING.csv, E_GATE.json",
        "note": f"E gate {'PASS' if gate_pass else 'FAIL'}; best Always-On={best_always} "
                f"mean_delta_ba={always['mean_delta_ba']:.4f} CI_lo={always['ci_lo_delta_ba']:.4f} "
                f"harm_rate={always['harm_rate']:.3f}",
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print("\nDetector summary (FULL):")
    print(df_summary[(df_summary.K == "FULL") & (df_summary.group == "POOLED")]
          [["candidate", "mean_delta_ba", "mean_delta_fpr", "mean_delta_fnr",
            "benefit_rate", "harm_rate", "catastrophic_rate", "coverage"]].round(4).to_string(index=False))
    print("\nBootstrap (FULL):")
    print(df_boot[df_boot.K == "FULL"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
