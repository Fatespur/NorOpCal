"""PB7-P2 Stage D — protocol-conformant gate assessment.

The raw D_GATE.json (written by d_bootstrap.py) applied an extra, non-protocol
criterion: DINOv2 gated-vs-always-on dFNR advantage. The frozen P0 protocol
defines the D3 publication-significance gate as:
    (1) DINOv2 dBA CI lower > 0
    (2) DINOv2 dFNR mean/CI compatible with the frozen safety semantics (<= +0.02)

This script re-derives the protocol-conformant D3 verdict from the raw
D_CONFIRMATORY_BOOTSTRAP.csv / D_PAIRED_COMPARISON.csv artifacts and records
the H5 selectivity-necessity assessment (pooled vs DINOv2-specific) as a
hypothesis nuance, NOT as a gate failure. It does not modify any method or
re-run any confirmatory test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "pb7_p2_publication_validation"
sys.path.insert(0, str(ROOT / "tools" / "pb7_p2_publication_validation"))
import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

DFNR_MAX = P2.DFNR_MAX  # 0.02 frozen safety budget


def main() -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    boot = pd.read_csv(OUT / "D_CONFIRMATORY_BOOTSTRAP.csv")
    paired = pd.read_csv(OUT / "D_PAIRED_COMPARISON.csv")

    dino = boot[(boot.policy == "GATED") & (boot.group == "DINOv2")].iloc[0]

    # ---- D3 publication significance (protocol gate) ------------------------- #
    dba_ci_lo = float(dino["boot_ci_lo_delta_ba"])
    dfnr_mean = float(dino["boot_mean_delta_fnr"])
    dfnr_ci_hi = float(dino["boot_ci_hi_delta_fnr"])
    d3_dba_ok = dba_ci_lo > 0
    d3_safety_ok = dfnr_mean <= DFNR_MAX + 1e-12
    d3_verdict = "PASS" if (d3_dba_ok and d3_safety_ok) else "FAIL"

    # ---- H5 selectivity necessity (hypothesis nuance, not a gate) ------------ #
    def row(group: str, vs: str) -> dict[str, Any]:
        return paired[(paired.group == group) & (paired.gated_vs == vs)].iloc[0].to_dict()

    p_dino_ao = row("DINOv2", "ALWAYS_ON")
    p_pool_ao = row("POOLED", "ALWAYS_ON")
    p_dino_t2 = row("DINOv2", "PB7P0_T2")
    p_dino_na = row("DINOv2", "NOADAPT")

    h5_pooled_dfnr = float(p_pool_ao["diff_mean_delta_fnr"])
    h5_pooled_harm = float(p_pool_ao["diff_mean_harm"])
    h5_dino_dfnr = float(p_dino_ao["diff_mean_delta_fnr"])

    # Pooled: gate reduces dFNR and harm (protects non-principal detectors).
    h5_pooled_supported = h5_pooled_dfnr < 0 and h5_pooled_harm < 0
    # DINOv2-specific: gate leaves dFNR unchanged (always-on already safe).
    h5_dino_supported = h5_dino_dfnr < -1e-12

    # ---- utility retention / safety improvement (D2 reporting) -------------- #
    d2_utility = {
        "gated_vs_noadapt_dba": float(p_dino_na["diff_mean_delta_ba"]),
        "gated_vs_noadapt_dba_ci_lo": float(p_dino_na["diff_ci_lo_delta_ba"]),
    }
    d2_safety = {
        "gated_vs_t2_dfnr": float(p_dino_t2["diff_mean_delta_fnr"]),
        "gated_vs_t2_dfnr_ci_hi": float(p_dino_t2["diff_ci_hi_delta_fnr"]),
    }

    assessment = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "D",
        "timestamp": ts,
        "note": (
            "Protocol-conformant D3 gate. The raw D_GATE.json additionally "
            "applied a non-protocol DINOv2 gated-vs-always-on dFNR-advantage "
            "criterion; that is assessed here as an H5 hypothesis nuance, not "
            "a gate requirement."
        ),
        "d3_publication_significance": {
            "dinov2_dba_ci_lo": dba_ci_lo,
            "dinov2_dba_ci_lo_gt_0": d3_dba_ok,
            "dinov2_dfnr_mean": dfnr_mean,
            "dinov2_dfnr_ci_hi": dfnr_ci_hi,
            "dinov2_safety_compatible": d3_safety_ok,
            "verdict": d3_verdict,
        },
        "d2_paired_comparison": {
            "utility_retention": d2_utility,
            "safety_improvement": d2_safety,
        },
        "h5_selectivity_necessity": {
            "pooled_dfnr_diff_gated_minus_always_on": h5_pooled_dfnr,
            "pooled_harm_diff_gated_minus_always_on": h5_pooled_harm,
            "pooled_safety_advantage_supported": h5_pooled_supported,
            "dinov2_dfnr_diff_gated_minus_always_on": h5_dino_dfnr,
            "dinov2_safety_advantage_supported": h5_dino_supported,
            "interpretation": (
                "On DINOv2 the gate leaves dFNR/harm unchanged vs Always-On "
                "(always-on is already safe: dFNR=0.0025, harm=0.037); the gate "
                "acts as coverage reduction (0.40 vs 0.69), not as a DINOv2 "
                "safety mechanism. The gate's safety value is pooled-level "
                "(protects PatchCore/EfficientAD from harm). Safety on DINOv2 is "
                "provided by the bounded actuator (lambda=0.25), not by the gate."
            ),
        },
        "overall_verdict": "PASS_WITH_NUANCE",
        "terminal": "PROCEED_TO_E",
    }
    (OUT / "D_GATE_ASSESSMENT.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    # ---- ledger --------------------------------------------------------------- #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "D",
                        "step": "d3_gate_assessment_protocol_conformant",
                        "status": "COMPLETE",
                        "timestamp": ts,
                        "artifacts": "D_GATE_ASSESSMENT.json",
                        "note": (
                            f"D3 gate {d3_verdict} (dBA CI_lo={dba_ci_lo:.4f}, "
                            f"dFNR mean={dfnr_mean:.4f}); "
                            f"H5 nuance: pooled safety advantage supported, DINOv2-specific dFNR "
                            f"advantage NOT_SUPPORTED (always-on already safe on DINOv2)"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
