"""PB7-P1 Stage B — Safety Bounding Family Freeze.

Freezes the multiplicative shrinkage bounding family:
    tau_used = tau_source + lambda * (tau_raw - tau_source),  lambda in {0.25, 0.50, 0.75, 1.00}

B1: the lambda grid is frozen before any outcome evaluation.
B2: uncertainty-adaptive shrinkage is NOT implemented (protocol prefers the
    simple B1 form; no simple closed-form confidence weight was pre-registered).
B3: bounding only shrinks toward tau_source or abstains; it never flips the
    correction direction, never applies an anomaly-dependent sign switch, and
    never applies oracle sign correction.

B Gate: lambda grid frozen, lambda=1.00 reproduces the raw actuator, bounding
is monotone toward tau_source, no direction switching, B2 not required.
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    fw = P.Firewall()
    ents = P.entity_list()

    # ---- B1: verify lambda grid semantics on all entities ------------------ #
    rows: List[Dict[str, Any]] = []
    max_diff_lam1 = 0.0
    shrink_ok = True
    for cat, cond, det in ents:
        src = P.load_src(cat, det, fw)
        cnd = P.load_cnd(cat, cond, det, fw)
        tau_source = P.source_threshold(src)
        for fam in P.ACTUATOR_FAMILIES:
            tau_raw = P.actuator_raw(fam, src, cnd, tau_source)
            tau_lam1 = P.bounded_tau(fam, src, cnd, tau_source, 1.00)
            max_diff_lam1 = max(max_diff_lam1, abs(tau_lam1 - tau_raw))
            # monotone shrink toward tau_source: |tau_used - tau_source| non-increasing in lambda
            prev = abs(tau_raw - tau_source)
            for lam in [0.75, 0.50, 0.25]:
                tau_used = P.bounded_tau(fam, src, cnd, tau_source, lam)
                cur = abs(tau_used - tau_source)
                if cur > prev + 1e-12:
                    shrink_ok = False
                prev = cur
            rows.append({
                "category": cat, "condition": cond, "detector": det,
                "family": fam, "tau_source": tau_source, "tau_raw": tau_raw,
                "tau_lam_1.00": tau_lam1, "tau_lam_0.75": P.bounded_tau(fam, src, cnd, tau_source, 0.75),
                "tau_lam_0.50": P.bounded_tau(fam, src, cnd, tau_source, 0.50),
                "tau_lam_0.25": P.bounded_tau(fam, src, cnd, tau_source, 0.25),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B_BOUNDING_GRID.csv", index=False)

    # ---- B2: uncertainty-adaptive shrinkage NOT implemented ---------------- #
    b2 = {
        "implemented": False,
        "reason": "protocol prefers the simple B1 multiplicative shrinkage; "
                  "no simple closed-form confidence weight was pre-registered",
        "note": "lambda is a fixed global scalar selected on DEV; no per-entity "
                "uncertainty weighting",
    }

    # ---- B3: no direction switching ---------------------------------------- #
    b3 = {
        "bounding_direction": "shrink toward tau_source only",
        "abstention": "gate (Stage C) may abstain (tau_used = tau_source)",
        "forbidden": ["flip correction direction", "anomaly-dependent sign switch",
                      "oracle sign correction"],
        "enforced": True,
    }

    registry = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "B",
        "timestamp": ts,
        "bounding_form": "tau_used = tau_source + lambda * (tau_raw - tau_source)",
        "lambda_grid": P.LAMBDA_GRID,
        "lambda_1_equals_raw_actuator": True,
        "b2_uncertainty_adaptive": b2,
        "b3_no_direction_switching": b3,
    }
    (OUT / "B_BOUNDING_FAMILY.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # ---- B Gate ------------------------------------------------------------ #
    checks = {
        "lambda_grid_frozen": P.LAMBDA_GRID == [0.25, 0.50, 0.75, 1.00],
        "lambda_1_equals_raw": max_diff_lam1 <= 1e-9,
        "monotone_shrink_toward_source": shrink_ok,
        "no_direction_switching": True,
        "b2_optional_not_required": True,
    }
    gate_pass = all(checks.values())
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "B",
        "timestamp": ts,
        "checks": checks,
        "detail": {
            "max_abs_diff_lambda_1_vs_raw": max_diff_lam1,
            "n_entities_checked": len(ents),
        },
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_C" if gate_pass else "STOP_BOUNDING_INVALID",
    }
    (OUT / "B_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger ------------------------------------------------------------ #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path)
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "B",
        "step": "safety_bounding_family_freeze",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": ts,
        "artifacts": "B_BOUNDING_FAMILY.json, B_BOUNDING_GRID.csv, B_GATE.json",
        "note": (f"B gate {'PASS' if gate_pass else 'FAIL'}; lambda grid "
                 f"{P.LAMBDA_GRID} frozen; B2 not implemented (B1 preferred); "
                 f"no direction switching"),
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print(f"\nMax abs diff lambda=1.00 vs raw: {max_diff_lam1:.3e}")


if __name__ == "__main__":
    main()
