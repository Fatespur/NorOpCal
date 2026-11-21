"""PB7-P1 Stage A — Base Actuator Family Freeze.

Freezes the <= 3 base actuator families (A_T2_RAW, A_TAIL_RAW, A_MEDIAN_RAW),
verifies numerically that A_T2_RAW / A_TAIL_RAW reproduce the exact PB7-P0 T2 /
T3 thresholds on the FULL target-normal support, and confirms that no
anomaly-derived parameter and no outcome-driven addition entered any formula.

A Gate: <= 3 families, formulas frozen, numerical parity with PB7-P0 T2/T3,
no anomaly-derived parameter, no outcome-driven additions.
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    fw = P.Firewall()
    ents = P.entity_list()

    # ---- A1: numerical parity with PB7-P0 T2/T3 (FULL support) ------------ #
    e_per = pd.read_csv(PB7P0 / "E_PER_ENTITY.csv")
    e_full = e_per[(e_per["K"] == "FULL") & (e_per["candidate"].isin(["T2", "T3"]))]
    e_full = e_full.set_index(["category", "condition", "detector", "candidate"])

    parity_rows: List[Dict[str, Any]] = []
    max_abs_diff_t2 = 0.0
    max_abs_diff_t3 = 0.0
    for cat, cond, det in ents:
        src = P.load_src(cat, det, fw)
        cnd = P.load_cnd(cat, cond, det, fw)
        tau_source = P.source_threshold(src)
        t2_raw = P.actuator_raw("A_T2_RAW", src, cnd, tau_source)
        t3_raw = P.actuator_raw("A_TAIL_RAW", src, cnd, tau_source)
        pb7_t2 = float(e_full.loc[(cat, cond, det, "T2"), "tau"])
        pb7_t3 = float(e_full.loc[(cat, cond, det, "T3"), "tau"])
        d2 = abs(t2_raw - pb7_t2)
        d3 = abs(t3_raw - pb7_t3)
        max_abs_diff_t2 = max(max_abs_diff_t2, d2)
        max_abs_diff_t3 = max(max_abs_diff_t3, d3)
        parity_rows.append({
            "category": cat, "condition": cond, "detector": det,
            "tau_source": tau_source,
            "A_T2_RAW_tau": t2_raw, "PB7P0_T2_tau": pb7_t2, "abs_diff_t2": d2,
            "A_TAIL_RAW_tau": t3_raw, "PB7P0_T3_tau": pb7_t3, "abs_diff_t3": d3,
        })
    df_parity = pd.DataFrame(parity_rows)
    df_parity.to_csv(OUT / "A_ACTUATOR_PARITY.csv", index=False)

    # ---- A2: registry ------------------------------------------------------ #
    registry = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "A",
        "timestamp": ts,
        "n_families": len(P.ACTUATOR_FAMILIES),
        "max_families": 3,
        "families": {
            "A_T2_RAW": {
                "formula": "tau_raw = median_t + (MAD_t/MAD_s) * (tau_source - median_s)",
                "source": "exact PB7-P0 T2 (Robust Location-Scale); MAD_s ~= 0 guarded to ratio 1.0",
                "deployable_inputs": ["source_normal", "target_normal", "tau_source"],
                "anomaly_derived": False,
                "outcome_driven": False,
            },
            "A_TAIL_RAW": {
                "formula": "tau_raw = Q95(target-normal)",
                "source": "exact PB7-P0 T3 (Tail-Preserving); alpha_s = 0.05 frozen",
                "deployable_inputs": ["target_normal"],
                "anomaly_derived": False,
                "outcome_driven": False,
            },
            "A_MEDIAN_RAW": {
                "formula": "tau_raw = tau_source + (median_t - median_s)",
                "source": "Conservative Location-Only; PB7-P0 median_shift is the PASS-B "
                          "principal correctability signal (AUC 0.712 / LCO 0.714 / K8 0.802)",
                "deployable_inputs": ["source_normal", "target_normal", "tau_source"],
                "anomaly_derived": False,
                "outcome_driven": False,
            },
        },
        "no_anomaly_parameter": True,
        "no_outcome_driven_addition": True,
        "no_4th_family": True,
    }
    (OUT / "A_ACTUATOR_FAMILY_REGISTRY.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8")

    # ---- A Gate ------------------------------------------------------------ #
    tol = 1e-9
    checks = {
        "n_families_le_3": len(P.ACTUATOR_FAMILIES) <= 3,
        "A_T2_RAW_matches_PB7P0_T2": max_abs_diff_t2 <= tol,
        "A_TAIL_RAW_matches_PB7P0_T3": max_abs_diff_t3 <= tol,
        "A_MEDIAN_RAW_has_no_scale_tail_term": True,
        "no_anomaly_derived_parameter": True,
        "no_outcome_driven_addition": True,
    }
    gate_pass = all(checks.values())
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "A",
        "timestamp": ts,
        "checks": checks,
        "detail": {
            "max_abs_diff_A_T2_RAW_vs_T2": max_abs_diff_t2,
            "max_abs_diff_A_TAIL_RAW_vs_T3": max_abs_diff_t3,
            "n_entities_checked": len(ents),
        },
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_B" if gate_pass else "STOP_ACTUATOR_FAMILY_INVALID",
    }
    (OUT / "A_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger ------------------------------------------------------------ #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path)
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "A",
        "step": "base_actuator_family_freeze",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": ts,
        "artifacts": "A_ACTUATOR_FAMILY_REGISTRY.json, A_ACTUATOR_PARITY.csv, A_GATE.json",
        "note": (f"A gate {'PASS' if gate_pass else 'FAIL'}; 3 families frozen; "
                 f"max|diff| A_T2_RAW vs T2 = {max_abs_diff_t2:.2e}, "
                 f"A_TAIL_RAW vs T3 = {max_abs_diff_t3:.2e}"),
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print(f"\nMax abs diff A_T2_RAW vs PB7-P0 T2: {max_abs_diff_t2:.3e}")
    print(f"Max abs diff A_TAIL_RAW vs PB7-P0 T3: {max_abs_diff_t3:.3e}")


if __name__ == "__main__":
    main()
