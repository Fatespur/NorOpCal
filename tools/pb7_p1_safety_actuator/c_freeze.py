"""PB7-P1 Stage C — Correctability Gate / Abstention Freeze.

Registers the abstention gate:
    adapt iff correctability_score (median_shift) >= c, else NoAdapt
with the pre-registered cutoff grid c in {Q50, Q60, Q70, Q80} of the DEV
FULL-support median_shift distribution (Option A, DEV-selected in Stage D).

C1: principal signal = median_shift (PB7-P0 PASS-B, AUC 0.712 / LCO 0.714 / K8 0.802).
C2: gate form = simple threshold; no redefinition after outcomes.
C3: cutoff grid frozen before any test evaluation; values computed from DEV
    normal-only score distribution only.
C4: coverage floor = 0.10 (pooled); verified on DEV K8 for every cutoff.

C Gate: signal frozen, gate form frozen, cutoff grid frozen, DEV-only cutoff
source, coverage floor feasible on DEV K8.
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
COVERAGE_FLOOR = 0.10


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    fw = P.Firewall()
    ents = P.entity_list()
    dev_ents = [e for e in ents if P.split_side(e[0]) == "DEV"]

    # ---- C1/C3: DEV FULL-support median_shift distribution ----------------- #
    full_scores: Dict[tuple, float] = {}
    k8_scores: Dict[tuple, np.ndarray] = {}
    for cat, cond, det in dev_ents:
        src = P.load_src(cat, det, fw)
        cnd = P.load_cnd(cat, cond, det, fw)
        tau = P.source_threshold(src)
        full_scores[(cat, cond, det)] = P.correctability_score(src, cnd, tau)
        n = len(cnd)
        if n >= 8:
            vals = []
            for r in range(P.K_REPEATS):
                idx = P.support_indices(cat, cond, det, n, "8", r)
                vals.append(P.correctability_score(src, cnd[idx], tau))
            k8_scores[(cat, cond, det)] = np.asarray(vals)

    fs = np.asarray(list(full_scores.values()))
    cutoffs: Dict[str, float] = {
        k: float(np.quantile(fs, P.CUTOFF_QUANTILES[k])) for k in P.CUTOFF_KEYS
    }

    # ---- C4: DEV K8 coverage per cutoff ------------------------------------ #
    cov_rows: List[Dict[str, Any]] = []
    coverage_ok = True
    for k in P.CUTOFF_KEYS:
        c = cutoffs[k]
        covs = np.asarray([(v >= c).mean() for v in k8_scores.values()])
        pooled_cov = float(covs.mean())
        if pooled_cov < COVERAGE_FLOOR:
            coverage_ok = False
        cov_rows.append({
            "cutoff_key": k, "cutoff_value": c,
            "dev_k8_pooled_coverage": pooled_cov,
            "dev_k8_n_entities_adapted_gt_half": int((covs > 0.5).sum()),
            "dev_k8_n_entities": int(len(covs)),
        })
    df_cov = pd.DataFrame(cov_rows)
    df_cov.to_csv(OUT / "C_CUTOFF_COVERAGE.csv", index=False)

    # ---- Gate direction check on DEV (PB7-P0 correctability evidence) ------ #
    head = pd.read_csv(PB7P0 / "A2_ORACLE_HEADROOM.csv")
    dev_corr = {}
    for (cat, cond, det), s in full_scores.items():
        h = head[(head.category == cat) & (head.condition == cond) & (head.detector == det)].iloc[0]
        dev_corr[(cat, cond, det)] = int(h["correctable_safe"])
    corr_arr = np.asarray([dev_corr[k] for k in full_scores])
    # correctable concentration among high median_shift (DEV)
    hi = fs >= np.median(fs)
    frac_corr_hi = float(corr_arr[hi].mean()) if hi.sum() > 0 else float("nan")
    frac_corr_lo = float(corr_arr[~hi].mean()) if (~hi).sum() > 0 else float("nan")
    direction_ok = frac_corr_hi > frac_corr_lo

    definition = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "C",
        "timestamp": ts,
        "signal": "median_shift (PB7-P0 PASS-B principal)",
        "signal_evidence": {"correctable_auc": 0.712, "lco_auc": 0.714, "k8_corr_auc": 0.802},
        "gate_form": "adapt iff median_shift >= c else NoAdapt",
        "cutoff_selection": "Option A (DEV-selected from pre-registered grid)",
        "cutoff_grid": P.CUTOFF_KEYS,
        "cutoff_quantiles": P.CUTOFF_QUANTILES,
        "cutoff_source": "DEV FULL-support median_shift distribution (39 entities)",
        "cutoff_values": cutoffs,
        "coverage_floor": COVERAGE_FLOOR,
        "no_score_redefinition": True,
        "no_test_outcome_use": True,
    }
    (OUT / "C_GATE_DEFINITION.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")

    # ---- C Gate ------------------------------------------------------------ #
    checks = {
        "signal_frozen_median_shift": True,
        "gate_form_simple_threshold": True,
        "cutoff_grid_frozen": list(cutoffs.keys()) == P.CUTOFF_KEYS,
        "cutoff_source_dev_only": True,
        "coverage_floor_feasible_on_dev_k8": coverage_ok,
        "gate_direction_prefers_correctable": direction_ok,
    }
    gate_pass = all(checks.values())
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "C",
        "timestamp": ts,
        "checks": checks,
        "detail": {
            "cutoff_values": cutoffs,
            "dev_k8_coverage": {r["cutoff_key"]: r["dev_k8_pooled_coverage"]
                                for r in cov_rows},
            "dev_frac_correctable_high_median_shift": frac_corr_hi,
            "dev_frac_correctable_low_median_shift": frac_corr_lo,
        },
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_D" if gate_pass else "STOP_GATE_INVALID",
    }
    (OUT / "C_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger ------------------------------------------------------------ #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path)
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "C",
        "step": "correctability_gate_abstention_freeze",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": ts,
        "artifacts": "C_GATE_DEFINITION.json, C_CUTOFF_COVERAGE.csv, C_GATE.json",
        "note": (f"C gate {'PASS' if gate_pass else 'FAIL'}; cutoffs "
                 f"{json.dumps({k: round(v, 6) for k, v in cutoffs.items()})}; "
                 f"DEV K8 coverage {[round(r['dev_k8_pooled_coverage'], 3) for r in cov_rows]}"),
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print(f"\nCutoffs: {json.dumps({k: round(v, 6) for k, v in cutoffs.items()})}")
    print(f"DEV K8 coverage: {[round(r['dev_k8_pooled_coverage'], 3) for r in cov_rows]}")


if __name__ == "__main__":
    main()
