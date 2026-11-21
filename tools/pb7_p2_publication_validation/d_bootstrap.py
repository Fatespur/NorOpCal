"""PB7-P2 Stage D — Confirmatory Bootstrap + Uncertainty Confirmation.

Independent confirmatory category-aware bootstrap at B=10000 with a new
pre-registered seed (20260821), for pooled and DINOv2, on the frozen principal
policy (A_MEDIAN_RAW lambda=0.25 cutoff=Q50, K=8).

Paired comparisons (paired category-aware bootstrap):
  - Frozen Gated vs NoAdapt
  - Frozen Gated vs Frozen Always-On (lambda=0.25, no gate)
  - Frozen Gated vs PB7-P0 unsafe T2 (A_T2_RAW lambda=1.0 no gate)

Publication significance (DINOv2 principal):
  - delta_BA CI lower > 0
  - delta_FNR mean/CI compatible with the frozen safety semantics (<= +0.02)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p2_publication_validation"))

import pb7_p1_common as P1  # type: ignore[import-not-found]  # noqa: E402
import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

OUT = P2.OUT
DET = "dinov2_vits14"
K = "8"
N_REPEATS = 30  # principal evaluation protocol (matches PB7-P1 E)


def eval_policy(family: str, lam: float, cutoff: float, gate: bool) -> pd.DataFrame:
    """Evaluate a policy on TEST at K=8 (per entity, mean over N_REPEATS)."""
    fw = P2.Firewall()
    ents = P2.entity_list()
    rows: list[dict[str, Any]] = []
    for cat, cond, det in ents:
        if P2.split_side(cat) != "TEST":
            continue
        src = P2.load_src(cat, det, fw)
        cnd = P2.load_cnd(cat, cond, det, fw)
        anm = P2.load_anm(cat, det, fw)
        tau_source = P2.source_threshold(src)
        c = cutoff if gate else 0.0
        r = P1.evaluate_entity(cat, cond, det, family, lam, c, K, src, cnd, anm, tau_source)
        rows.append(
            {
                "category": cat,
                "condition": cond,
                "detector": det,
                "delta_ba": r["delta_ba"],
                "delta_fpr": r["delta_fpr"],
                "delta_fnr": r["delta_fnr"],
                "benefit": r["benefit"],
                "harm": r["harm"],
                "catastrophic": r["catastrophic"],
                "coverage": r["coverage"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    gated = eval_policy("A_MEDIAN_RAW", 0.25, P2.PRINCIPAL_CUTOFF, True)
    noadapt = eval_policy(None, 0.0, 0.0, False)
    always_on = eval_policy("A_MEDIAN_RAW", 0.25, 0.0, False)
    t2 = eval_policy("A_T2_RAW", 1.0, 0.0, False)

    policies = {
        "GATED": gated,
        "NOADAPT": noadapt,
        "ALWAYS_ON": always_on,
        "PB7P0_T2": t2,
    }
    for name, df in policies.items():
        df.to_csv(OUT / f"D_POLICY_{name}.csv", index=False)

    # ---- confirmatory bootstrap (B=10000) ------------------------------------ #
    boot_rows: list[dict[str, Any]] = []
    for name, df in policies.items():
        for group, det_filter in [("POOLED", None), ("DINOv2", DET)]:
            sub = df if det_filter is None else df[df.detector == det_filter]
            b = P2.confirmatory_bootstrap(sub, seed=P2.CONF_BOOT_SEED, b=P2.CONF_BOOT_B)
            boot_rows.append({"policy": name, "group": group, **b})
    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(OUT / "D_CONFIRMATORY_BOOTSTRAP.csv", index=False)

    # ---- paired comparisons --------------------------------------------------- #
    paired_rows: list[dict[str, Any]] = []
    for group, det_filter in [("POOLED", None), ("DINOv2", DET)]:
        ga = gated if det_filter is None else gated[gated.detector == det_filter]
        for other_name, other_df in [
            ("NOADAPT", noadapt),
            ("ALWAYS_ON", always_on),
            ("PB7P0_T2", t2),
        ]:
            ob = other_df if det_filter is None else other_df[other_df.detector == det_filter]
            p = P2.paired_block_bootstrap(ga, ob, seed=P2.CONF_BOOT_SEED, b=P2.CONF_BOOT_B)
            paired_rows.append({"group": group, "gated_vs": other_name, **p})
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(OUT / "D_PAIRED_COMPARISON.csv", index=False)

    # ---- publication significance (DINOv2 principal) ------------------------- #
    dino_gated = gated[gated.detector == DET]
    b = P2.confirmatory_bootstrap(dino_gated, seed=P2.CONF_BOOT_SEED, b=P2.CONF_BOOT_B)
    dba_ci_lo = float(b["boot_ci_lo_delta_ba"])
    dfnr_mean = float(b["boot_mean_delta_fnr"])
    dfnr_ci_hi = float(b["boot_ci_hi_delta_fnr"])
    dba_ci_lo_gt_0 = dba_ci_lo > 0
    safety_compatible = dfnr_mean <= P2.DFNR_MAX + 1e-12

    # paired significance: Gated vs Always-On on DINOv2 (safety advantage)
    pa = df_paired[(df_paired.group == "DINOv2") & (df_paired.gated_vs == "ALWAYS_ON")].iloc[0]
    safety_advantage = float(pa["diff_ci_hi_delta_fnr"]) < 0  # Gated dFNR < Always-On dFNR

    checks = {
        "dinov2_dba_ci_lo": dba_ci_lo,
        "dinov2_dba_ci_lo_gt_0": dba_ci_lo_gt_0,
        "dinov2_dfnr_mean": dfnr_mean,
        "dinov2_dfnr_ci_hi": dfnr_ci_hi,
        "dinov2_safety_compatible": safety_compatible,
        "dinov2_gated_vs_always_on_dfnr_advantage": safety_advantage,
    }
    ok = dba_ci_lo_gt_0 and safety_compatible and safety_advantage
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "D",
        "timestamp": ts,
        "bootstrap": {"B": P2.CONF_BOOT_B, "seed": P2.CONF_BOOT_SEED},
        "checks": checks,
        "verdict": "PASS" if ok else "FAIL",
        "terminal": "PROCEED_TO_E" if ok else "PROCEED_TO_E_WITH_UNCERTAINTY_NOT_CONFIRMED",
    }
    (OUT / "D_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger --------------------------------------------------------------- #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "D",
                        "step": "confirmatory_bootstrap_and_paired_comparison",
                        "status": "COMPLETE",
                        "timestamp": ts,
                        "artifacts": "D_CONFIRMATORY_BOOTSTRAP.csv, D_PAIRED_COMPARISON.csv, "
                        "D_POLICY_*.csv, D_GATE.json",
                        "note": (
                            f"D gate {'PASS' if ok else 'FAIL'}; DINOv2 dBA CI_lo={dba_ci_lo:.4f}, "
                            f"dFNR mean={dfnr_mean:.4f}, "
                            f"gated-vs-always-on safety adv={safety_advantage}"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(gate, indent=2))
    print("\nConfirmatory bootstrap (B=10000):")
    print(df_boot.round(4).to_string(index=False))
    print("\nPaired comparisons:")
    print(df_paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
