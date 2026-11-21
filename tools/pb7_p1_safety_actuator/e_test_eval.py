"""PB7-P1 Stage E — Frozen TEST Evaluation.

Freezes the single principal candidate selected on DEV (family, lambda, cutoff)
and performs ONE frozen evaluation on the TEST split at the principal K=8
setting (auxiliary K=4, FULL). Mandatory baselines:

  B0 NoAdapt (tau_source)
  B1 PB7-P0 best Always-On T2 (A_T2_RAW, lambda=1.0, no gate)
  B2 Frozen base actuator without shrinkage (A_MEDIAN_RAW, lambda=1.0, no gate)
  B3 Bounded actuator Always-On (A_MEDIAN_RAW, lambda=0.25, no gate)
  B4 Bounded + Abstaining actuator (A_MEDIAN_RAW, lambda=0.25, cutoff=Q50)  [principal]
  B5 Oracle safe threshold (DIAGNOSTIC ONLY)

Metrics: delta_BA, 95% CI (B=5000 category-aware block bootstrap), delta_FPR,
delta_FNR, harm rate, catastrophic harm, coverage, benefit precision/recall,
per-detector results.

Gates G1-G7 are evaluated and recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p1_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT = P.OUT
PB7P0 = P.PB7P0
SAFE_DFNR = P.DFNR_MAX  # +0.02
HARM_REDUCTION_MIN = 0.25  # G3: relative harm reduction vs PB7-P0 T2
COVERAGE_FLOOR = 0.10      # G5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- read DEV-selected candidate --------------------------------------- #
    dgate = json.loads((OUT / "D_GATE.json").read_text(encoding="utf-8"))
    sel = dgate["selected"]
    if sel is None:
        raise SystemExit("D_NO_SAFE_DEV_CANDIDATE: no principal candidate to evaluate")
    cdef = json.loads((OUT / "C_GATE_DEFINITION.json").read_text(encoding="utf-8"))
    cutoff_value = float(cdef["cutoff_values"][sel["cutoff_key"]])

    # ---- freeze method spec ------------------------------------------------- #
    method_spec = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "E",
        "timestamp": ts,
        "principal_candidate": {
            "family": sel["family"],
            "lambda": sel["lambda"],
            "cutoff_key": sel["cutoff_key"],
            "cutoff_value": cutoff_value,
            "gate": "adapt iff median_shift >= cutoff_value else NoAdapt",
            "bounding": "tau_used = tau_source + lambda * (tau_raw - tau_source)",
        },
        "k_settings": {"principal": "8", "auxiliary": ["4", "FULL"]},
        "detector_handling": "single global operating setting; no detector-specific tuning",
        "baselines": {
            "B0": "NoAdapt (tau_source)",
            "B1": "PB7-P0 best Always-On T2 (A_T2_RAW lambda=1.0 no gate)",
            "B2": "Frozen base actuator without shrinkage (A_MEDIAN_RAW lambda=1.0 no gate)",
            "B3": "Bounded actuator Always-On (A_MEDIAN_RAW lambda=0.25 no gate)",
            "B4": "Bounded + Abstaining (A_MEDIAN_RAW lambda=0.25 cutoff=Q50) [principal]",
            "B5": "Oracle safe threshold (DIAGNOSTIC ONLY)",
        },
        "frozen": True,
    }
    (OUT / "E_FROZEN_METHOD_SPEC.json").write_text(json.dumps(method_spec, indent=2), encoding="utf-8")

    # ---- load TEST entities ------------------------------------------------ #
    fw = P.Firewall()
    ents = P.entity_list()
    test_ents = [e for e in ents if P.split_side(e[0]) == "TEST"]
    head = pd.read_csv(PB7P0 / "A2_ORACLE_HEADROOM.csv")

    # ---- baseline definitions ---------------------------------------------- #
    # each baseline: (family, lambda, cutoff_value or None, gate bool, oracle bool)
    baselines = {
        "B0": {"family": None, "lambda": None, "cutoff": None, "gate": False, "oracle": False},
        "B1": {"family": "A_T2_RAW", "lambda": 1.00, "cutoff": None, "gate": False, "oracle": False},
        "B2": {"family": "A_MEDIAN_RAW", "lambda": 1.00, "cutoff": None, "gate": False, "oracle": False},
        "B3": {"family": "A_MEDIAN_RAW", "lambda": 0.25, "cutoff": None, "gate": False, "oracle": False},
        "B4": {"family": "A_MEDIAN_RAW", "lambda": 0.25, "cutoff": cutoff_value, "gate": True, "oracle": False},
        "B5": {"family": None, "lambda": None, "cutoff": None, "gate": False, "oracle": True},
    }

    # ---- evaluate per entity ------------------------------------------------ #
    rows: List[Dict[str, Any]] = []
    for cat, cond, det in test_ents:
        src = P.load_src(cat, det, fw)
        cnd_full = P.load_cnd(cat, cond, det, fw)
        anm = P.load_anm(cat, det, fw)
        tau_source = P.source_threshold(src)
        h = head[(head.category == cat) & (head.condition == cond) & (head.detector == det)].iloc[0]
        oracle_tau = float(h["tau_star_safe"])
        for bl, spec in baselines.items():
            for k_value in P.K_GRID:
                if spec["oracle"]:
                    # oracle: evaluate at tau*_SAFE directly (DIAGNOSTIC ONLY)
                    labels = np.concatenate([np.zeros(len(cnd_full)), np.ones(len(anm))])
                    scores = np.concatenate([cnd_full, anm])
                    before = C.detection_metrics(scores, labels, tau_source)
                    m = C.detection_metrics(scores, labels, oracle_tau)
                    r = {
                        "ba": m["ba"], "fpr": m["fpr"], "fnr": m["fnr"],
                        "delta_ba": m["ba"] - before["ba"],
                        "delta_fpr": m["fpr"] - before["fpr"],
                        "delta_fnr": m["fnr"] - before["fnr"],
                        "tau": oracle_tau,
                        "benefit": int(C.benefit_label(before, m)),
                        "harm": int((m["ba"] - before["ba"]) < P.HARM_DBA
                                    or (m["fnr"] - before["fnr"]) > P.HARM_DFNR),
                        "catastrophic": int((m["fnr"] - before["fnr"]) > P.CATASTROPHIC_DFNR),
                        "coverage": 1.0 if abs(oracle_tau - tau_source) > 1e-12 else 0.0,
                        "n_repeats": 1,
                    }
                else:
                    r = P.evaluate_entity(cat, cond, det, spec["family"], spec["lambda"],
                                          spec["cutoff"] if spec["gate"] else 0.0,
                                          k_value, src, cnd_full, anm, tau_source)
                rows.append({"category": cat, "condition": cond, "detector": det,
                             "baseline": bl, "K": k_value, **r})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E_TEST_PER_ENTITY.csv", index=False)

    # ---- pooled + detector summary ------------------------------------------ #
    summary_rows: List[Dict[str, Any]] = []
    for bl in baselines:
        for k_value in P.K_GRID:
            sub = df[(df.baseline == bl) & (df.K == k_value)]
            for group_name, g in [("POOLED", sub)] + \
                    [(d, sub[sub.detector == d]) for d in P.DETECTORS]:
                s = P.summary_metrics(g)
                summary_rows.append({"baseline": bl, "K": k_value, "group": group_name,
                                     "n": int(len(g)), **s})
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT / "E_TEST_DETECTOR_SUMMARY.csv", index=False)

    # ---- bootstrap (B=5000, category-aware block) --------------------------- #
    boot_rows: List[Dict[str, Any]] = []
    for bl in baselines:
        for k_value in P.K_GRID:
            sub = df[(df.baseline == bl) & (df.K == k_value)]
            b = P.block_bootstrap(sub, seed=P.BOOT_SEED, b=P.BOOT_B)
            boot_rows.append({"baseline": bl, "K": k_value, **b})
    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(OUT / "E_TEST_BOOTSTRAP.csv", index=False)

    # ---- E3: principal gates (at K=8) --------------------------------------- #
    def gate_metrics(bl: str, k_value: str) -> Dict[str, float]:
        sub = df[(df.baseline == bl) & (df.K == k_value)]
        b = df_boot[(df_boot.baseline == bl) & (df_boot.K == k_value)].iloc[0]
        s = P.summary_metrics(sub)
        return {**s,
                "ci_lo_delta_ba": float(b["boot_ci_lo_delta_ba"]),
                "ci_hi_delta_ba": float(b["boot_ci_hi_delta_ba"]),
                "boot_mean_delta_ba": float(b["boot_mean_delta_ba"]),
                "boot_mean_delta_fnr": float(b["boot_mean_delta_fnr"]),
                "boot_mean_harm_rate": float(b["boot_mean_harm_rate"])}

    b4 = gate_metrics("B4", "8")
    b3 = gate_metrics("B3", "8")
    b1 = gate_metrics("B1", "8")
    b0 = gate_metrics("B0", "8")

    # G1 utility
    g1 = b4["mean_delta_ba"] > 0 and b4["ci_lo_delta_ba"] > 0
    g1_strong = b4["mean_delta_ba"] >= P.DBA_MIN
    # G2 FNR safety
    g2 = b4["mean_delta_fnr"] <= SAFE_DFNR + 1e-12
    # G3 harm reduction vs PB7-P0 Always-On T2 (FULL historical 0.573; use TEST B1 as reference)
    ref_harm = b1["harm_rate"]
    rel_reduction = (ref_harm - b4["harm_rate"]) / ref_harm if ref_harm > 0 else float("nan")
    g3 = b4["harm_rate"] < ref_harm and rel_reduction >= HARM_REDUCTION_MIN
    # G4 selectivity value: B4 vs B3
    g4 = (b4["mean_delta_fnr"] < b3["mean_delta_fnr"] - 1e-12 or
          b4["harm_rate"] < b3["harm_rate"] - 1e-12)
    g4_utility_retained = b4["mean_delta_ba"] >= b3["mean_delta_ba"] - 0.01
    # G5 coverage
    g5 = b4["coverage"] >= COVERAGE_FLOOR
    # G6 K8 principal (evaluated at K=8 by construction)
    g6 = True
    # G7 detector specificity (report; not a pass/fail gate)
    det_b4 = df_summary[(df_summary.baseline == "B4") & (df_summary.K == "8") &
                        (df_summary.group != "POOLED")]
    det_dba = {r["group"]: float(r["mean_delta_ba"]) for _, r in det_b4.iterrows()}
    det_dfnr = {r["group"]: float(r["mean_delta_fnr"]) for _, r in det_b4.iterrows()}
    det_cov = {r["group"]: float(r["coverage"]) for _, r in det_b4.iterrows()}

    checks = {
        "G1_utility_dba_gt_0_ci_lo_gt_0": g1,
        "G1_detail": {"mean_delta_ba": b4["mean_delta_ba"], "ci_lo": b4["ci_lo_delta_ba"],
                      "strong_dba_ge_002": g1_strong},
        "G2_fnr_safety_le_002": g2,
        "G2_detail": {"mean_delta_fnr": b4["mean_delta_fnr"]},
        "G3_harm_reduction": g3,
        "G3_detail": {"ref_harm_B1": ref_harm, "b4_harm": b4["harm_rate"],
                      "relative_reduction": rel_reduction},
        "G4_selectivity_value": g4,
        "G4_detail": {"b4_dfnr": b4["mean_delta_fnr"], "b3_dfnr": b3["mean_delta_fnr"],
                      "b4_harm": b4["harm_rate"], "b3_harm": b3["harm_rate"],
                      "b4_dba": b4["mean_delta_ba"], "b3_dba": b3["mean_delta_ba"],
                      "utility_retained": g4_utility_retained},
        "G5_coverage_ge_10pct": g5,
        "G5_detail": {"coverage": b4["coverage"], "detector_coverage": det_cov},
        "G6_k8_principal": g6,
        "G7_detector_specificity": det_dba,
    }
    gate_pass = all([g1, g2, g3, g4, g5, g6])
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "E",
        "timestamp": ts,
        "principal_candidate": method_spec["principal_candidate"],
        "checks": checks,
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_F" if gate_pass else "PROCEED_TO_F_WITH_FAILED_GATES",
    }
    (OUT / "E_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger ------------------------------------------------------------- #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path)
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "E",
        "step": "frozen_test_evaluation",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": ts,
        "artifacts": "E_FROZEN_METHOD_SPEC.json, E_TEST_PER_ENTITY.csv, "
                     "E_TEST_DETECTOR_SUMMARY.csv, E_TEST_BOOTSTRAP.csv, E_GATE.json",
        "note": (f"E gate {'PASS' if gate_pass else 'FAIL'}; B4 K8 dBA={b4['mean_delta_ba']:.4f} "
                 f"CI_lo={b4['ci_lo_delta_ba']:.4f} dFNR={b4['mean_delta_fnr']:.4f} "
                 f"harm={b4['harm_rate']:.3f} cov={b4['coverage']:.3f}"),
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print("\nTEST benchmark (K=8, POOLED):")
    print(df_summary[(df_summary.K == "8") & (df_summary.group == "POOLED")]
          [["baseline", "mean_delta_ba", "mean_delta_fpr", "mean_delta_fnr",
            "benefit_rate", "harm_rate", "catastrophic_rate", "coverage"]].round(4).to_string(index=False))
    print("\nTEST bootstrap (K=8):")
    print(df_boot[df_boot.K == "8"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
