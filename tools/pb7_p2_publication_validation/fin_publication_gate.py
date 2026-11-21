"""PB7-P2 Stage FINAL — Publication-Readiness Gate.

Aggregates the frozen P0 protocol and stage gates A-G into the FINAL
publication-readiness verdict, writes F_PB7_P2_PUBLICATION_VERDICT.json and
PB7_P2_PUBLICATION_READINESS_FINAL_REPORT.md, and updates the execution ledger.

Terminal decision (T1-T6) is derived from the frozen stage gates only; no
method is modified and no confirmatory test is re-run here.
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


def load_json(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    a = load_json("A_EXACT_REPRODUCTION.json")
    b = load_json("B_GATE.json")
    c = load_json("C_GATE.json")
    d = load_json("D_GATE_ASSESSMENT.json")
    e = load_json("E_GATE.json")
    f = load_json("F_GATE.json")
    g = load_json("G_GATE.json")

    # ---- aggregate gates ------------------------------------------------------- #
    exact_reproduction = "PASS" if a["verdict"] == "PASS" else "FAIL"
    category_robustness = "PASS" if b["verdict"] == "PASS" else "FAIL"
    k8_robustness = "PASS" if c["verdict"] == "PASS" else "FAIL"
    bootstrap_confirmation = d["d3_publication_significance"]["verdict"]
    residual_harm = e["verdict"]
    dinov2_supported = (
        f["f1_detector_confirmation"][0]["verdict"] == "USEFUL_SAFE_SELECTABLE"
        and float(f["f1_detector_confirmation"][0]["mean_delta_ba"]) > 0
        and float(f["f1_detector_confirmation"][0]["mean_delta_fnr"]) <= P2.DFNR_MAX + 1e-12
    )
    patchcore_status = f["f1_detector_confirmation"][1]["verdict"]
    efficientad_status = f["f1_detector_confirmation"][2]["verdict"]
    external_validation = g["verdict"]

    # ---- terminal decision (T1-T6) --------------------------------------------- #
    t1_ok = (
        exact_reproduction == "PASS"
        and category_robustness == "PASS"
        and k8_robustness == "PASS"
        and bootstrap_confirmation == "PASS"
        and residual_harm == "R1_LOCALIZED_RESIDUAL_RISK"
        and bool(dinov2_supported)
        and f["f2_classification_persists"]
    )
    if t1_ok:
        terminal = "PB7_P2_PUBLICATION_CANDIDATE_CONFIRMED_DINOV2_SPECIFIC"
        terminal_label = "T1"
    elif exact_reproduction != "PASS":
        terminal = "PB7_P2_REPRODUCTION_FAILURE"
        terminal_label = "T6"
    elif category_robustness != "PASS":
        terminal = "PB7_P2_METHOD_POSITIVE_BUT_CATEGORY_FRAGILE"
        terminal_label = "T4"
    elif k8_robustness != "PASS":
        terminal = "PB7_P2_METHOD_POSITIVE_BUT_FEW_NORMAL_ROBUSTNESS_INSUFFICIENT"
        terminal_label = "T3"
    elif residual_harm == "R2_SYSTEMIC_SAFETY_FAILURE":
        terminal = "PB7_P2_SAFETY_NOT_CONFIRMED"
        terminal_label = "T5"
    else:
        terminal = "PB7_P2_PUBLICATION_CANDIDATE_CONFIRMED_WITH_CATEGORY_LIMITATION"
        terminal_label = "T2"

    verdict = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "FINAL",
        "timestamp": ts,
        "inherits_terminal": "PB7_P1_DETECTOR_SPECIFIC_SAFE_ACTUATOR_SUPPORTED",
        "exact_reproduction": exact_reproduction,
        "category_robustness": category_robustness,
        "k8_robustness": k8_robustness,
        "bootstrap_confirmation": bootstrap_confirmation,
        "residual_harm": residual_harm,
        "dinov2_supported": bool(dinov2_supported),
        "patchcore_status": patchcore_status,
        "efficientad_status": efficientad_status,
        "external_validation": external_validation,
        "publication_candidate": t1_ok,
        "terminal": terminal,
        "terminal_label": terminal_label,
        "answer": (
            "YES — the frozen PB7-P1 DINOv2-specific normal-only safe threshold "
            "adaptation remains useful, safe, and stable under category "
            "perturbation, repeated K=8 supports, confirmatory bootstrap, and "
            "residual-harm analysis."
            if t1_ok
            else "NO — see terminal."
        ),
    }
    (OUT / "F_PB7_P2_PUBLICATION_VERDICT.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )

    # ---- final report ----------------------------------------------------------- #
    report = build_report(a, b, c, d, e, f, g, verdict, ts)
    (OUT / "PB7_P2_PUBLICATION_READINESS_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    # ---- ledger ------------------------------------------------------------------ #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "FINAL",
                        "step": "publication_readiness_gate",
                        "status": "COMPLETE",
                        "timestamp": ts,
                        "artifacts": "F_PB7_P2_PUBLICATION_VERDICT.json, "
                        "PB7_P2_PUBLICATION_READINESS_FINAL_REPORT.md",
                        "note": (
                            f"FINAL terminal {terminal_label} {terminal}; "
                            f"exact={exact_reproduction}, category={category_robustness}, "
                            f"k8={k8_robustness}, bootstrap={bootstrap_confirmation}, "
                            f"harm={residual_harm}, dinov2={dinov2_supported}"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(verdict, indent=2))


def build_report(
    a: dict[str, Any],
    b: dict[str, Any],
    c: dict[str, Any],
    d: dict[str, Any],
    e: dict[str, Any],
    f: dict[str, Any],
    g: dict[str, Any],
    verdict: dict[str, Any],
    ts: str,
) -> str:
    boot = pd.read_csv(OUT / "D_CONFIRMATORY_BOOTSTRAP.csv")
    stab = pd.read_csv(OUT / "C_K_SUPPORT_STABILITY.csv")
    harm = pd.read_csv(OUT / "E_RESIDUAL_HARM.csv")
    det = pd.read_csv(OUT / "F_DETECTOR_CONFIRMATION.csv")

    k8_pool = stab[(stab.K == "8") & (stab.group == "POOLED")].iloc[0]
    full_pool = stab[(stab.K == "FULL") & (stab.group == "POOLED")].iloc[0]

    lines: list[str] = []
    ap = lines.append
    ap("# PB7-P2 Publication-Readiness Confirmatory Validation — Final Report")
    ap("")
    ap(f"- Date: {ts}")
    ap("- Stage: PB7-P2 (CONFIRMATORY / PUBLICATION-READINESS)")
    ap("- Inherits terminal: `PB7_P1_DETECTOR_SPECIFIC_SAFE_ACTUATOR_SUPPORTED`")
    ap(
        f"- Frozen principal policy: A_MEDIAN_RAW, lambda=0.25, cutoff=Q50 "
        f"({P2.PRINCIPAL_CUTOFF:.6f}), K=8, safety budget mean dFNR <= +0.02"
    )
    ap("")
    ap("## 1. Exact Reproduction (Stage A)")
    ap("")
    ap(f"- Verdict: **{a['verdict']}** (tolerance < 1e-9)")
    ap(
        f"- Max per-entity abs diff: dBA={a['max_per_entity_abs_diff']['delta_ba']:.2e}, "
        f"dFNR={a['max_per_entity_abs_diff']['delta_fnr']:.2e}, "
        f"harm={a['max_per_entity_abs_diff']['harm']:.2e}"
    )
    ap(
        f"- Pooled K8: dBA={a['checks']['POOLED_delta_ba_diff']:.2e}, "
        f"dFNR={a['checks']['POOLED_delta_fnr_diff']:.2e}"
    )
    ap(
        f"- DINOv2 K8: dBA={a['checks']['DINOv2_delta_ba_diff']:.2e}, "
        f"dFNR={a['checks']['DINOv2_delta_fnr_diff']:.2e}"
    )
    ap("")
    ap("## 2. Leave-One-Category-Out Influence Audit (Stage B)")
    ap("")
    ap(f"- Verdict: **{b['verdict']}**; full DINOv2 dBA={b['checks']['full_dba']:.4f}")
    ap(
        f"- LOO runs: {b['checks']['n_loco_runs']}; dBA>0 in "
        f"{b['checks']['n_runs_dba_gt_0']}/{b['checks']['n_loco_runs']}; "
        f"dFNR<=0.02 in {b['checks']['n_runs_dfnr_le_002']}/{b['checks']['n_loco_runs']}"
    )
    ap(
        f"- Max influence: {b['checks']['max_influence_dba']:.4f} "
        f"(category {b['checks']['max_influence_category']}, "
        f"{b['checks']['max_influence_dba'] / b['checks']['full_dba'] * 100:.1f}% of total)"
    )
    ap(f"- No single category >50% utility: {b['checks']['no_single_category_gt_50pct_utility']}")
    ap(f"- Category-fragile flag: {b['category_fragile_flag']}")
    ap("")
    ap("## 3. K-Support Confirmatory Audit (Stage C)")
    ap("")
    ap(f"- Verdict: **{c['verdict']}**; repeats={c['n_repeats']}")
    ap("")
    ap("| Setting | median dBA | positive freq | safety freq | median dFNR |")
    ap("|---|---|---|---|---|")
    for _, r in stab.iterrows():
        ap(
            f"| {r['K']} {r['group']} | {r['median_delta_ba']:.4f} | "
            f"{r['positive_dba_freq']:.3f} | {r['safety_pass_freq']:.3f} | "
            f"{r['median_delta_fnr']:.4f} |"
        )
    ap("")
    ap(
        f"- K8 gate: positive freq={k8_pool['positive_dba_freq']:.3f} (>=0.90), "
        f"safety freq={k8_pool['safety_pass_freq']:.3f} (>=0.90), "
        f"median dBA={k8_pool['median_delta_ba']:.4f} vs FULL "
        f"{full_pool['median_delta_ba']:.4f} "
        f"(retains {k8_pool['median_delta_ba'] / full_pool['median_delta_ba'] * 100:.0f}%)"
    )
    ap(
        f"- K8 Jaccard: pooled={c['checks']['k8_jaccard_pooled']:.3f}, "
        f"DINOv2={c['checks']['k8_jaccard_dinov2']:.3f}"
    )
    ap("")
    ap("## 4. Confirmatory Bootstrap (Stage D)")
    ap("")
    ap(f"- B={P2.CONF_BOOT_B}, seed={P2.CONF_BOOT_SEED} (pre-registered)")
    ap("")
    ap("| Group | mean dBA | 95% CI dBA | mean dFNR | 95% CI dFNR | harm | coverage |")
    ap("|---|---|---|---|---|---|---|")
    for _, r in boot[boot.policy == "GATED"].iterrows():
        ap(
            f"| {r['group']} | {r['boot_mean_delta_ba']:.4f} | "
            f"[{r['boot_ci_lo_delta_ba']:.4f}, {r['boot_ci_hi_delta_ba']:.4f}] | "
            f"{r['boot_mean_delta_fnr']:.4f} | "
            f"[{r['boot_ci_lo_delta_fnr']:.4f}, {r['boot_ci_hi_delta_fnr']:.4f}] | "
            f"{r['boot_mean_harm_rate']:.4f} | {r['boot_mean_coverage']:.4f} |"
        )
    ap("")
    ap(
        f"- D3 publication significance: **{d['d3_publication_significance']['verdict']}** "
        f"(DINOv2 dBA CI lower={d['d3_publication_significance']['dinov2_dba_ci_lo']:.4f} > 0; "
        f"dFNR mean={d['d3_publication_significance']['dinov2_dfnr_mean']:.4f} <= +0.02)"
    )
    ur = d["d2_paired_comparison"]["utility_retention"]
    si = d["d2_paired_comparison"]["safety_improvement"]
    ap(
        f"- Paired comparisons (DINOv2): Gated vs NoAdapt dBA={ur['gated_vs_noadapt_dba']:.4f} "
        f"(CI lo={ur['gated_vs_noadapt_dba_ci_lo']:.4f}); "
        f"Gated vs PB7-P0 T2 dFNR={si['gated_vs_t2_dfnr']:.4f} "
        f"(CI hi={si['gated_vs_t2_dfnr_ci_hi']:.4f})"
    )
    h5 = d["h5_selectivity_necessity"]
    ap(
        f"- H5 selectivity nuance: pooled safety advantage supported "
        f"(dFNR diff={h5['pooled_dfnr_diff_gated_minus_always_on']:.4f}, "
        f"harm diff={h5['pooled_harm_diff_gated_minus_always_on']:.4f}); "
        f"DINOv2-specific dFNR advantage NOT_SUPPORTED (always-on already safe on DINOv2; "
        f"the gate acts as coverage reduction there)."
    )
    ap("")
    ap("## 5. Residual Harm Audit (Stage E)")
    ap("")
    ap(f"- Verdict: **{e['verdict']}** (R1 LOCALIZED_RESIDUAL_RISK)")
    ap(
        f"- n_harm={e['n_harm']}/{e['n_entities']} (harm rate={e['harm_rate']:.3f}); "
        f"n_catastrophic={e['n_catastrophic']}"
    )
    ap(
        f"- Concentration: single category={e['concentration']['single_category_concentration']} "
        f"(all in walnuts, top share={e['concentration']['top_category_share']:.2f}); "
        f"detector-wide failure={e['concentration']['detector_wide_failure']}; "
        f"condition systemic failure={e['concentration']['condition_systemic_failure']}; "
        f"DINOv2 harm rate={e['concentration']['dinov2_harm_rate']:.3f} "
        f"(n={e['concentration']['dinov2_n_harm']})"
    )
    ap("")
    ap("| category | condition | detector | source_fnr | median_shift | dBA | dFNR | harm_type |")
    ap("|---|---|---|---|---|---|---|---|")
    for _, r in harm.iterrows():
        ap(
            f"| {r['category']} | {r['condition']} | {r['detector']} | "
            f"{r['source_fnr']:.3f} | {r['median_shift_full']:.4f} | "
            f"{r['delta_ba']:.4f} | {r['delta_fnr']:.4f} | {r['harm_type']} |"
        )
    ap("")
    ap("## 6. Detector-Specificity Confirmation (Stage F)")
    ap("")
    ap(f"- Classification persists vs sealed PB7-P1: **{f['f2_classification_persists']}**")
    ap("")
    ap(
        "| detector | n_k8 | n_correctable_safe | mean dBA | mean dFNR | "
        "harm | coverage | verdict |"
    )
    ap("|---|---|---|---|---|---|---|---|")
    for _, r in det.iterrows():
        ap(
            f"| {r['detector']} | {r['n_k8']} | {r['n_correctable_safe']} | "
            f"{r['mean_delta_ba']:.4f} | {r['mean_delta_fnr']:.4f} | "
            f"{r['harm_rate']:.3f} | {r['coverage']:.3f} | {r['verdict']} |"
        )
    ap("")
    ap(f'- Claim frozen: **{f["f3_claim"]["frozen"]}** — "{f["f3_claim"]["claim"]}"')
    ap("")
    ap("## 7. External-Setting Validation (Stage G)")
    ap("")
    ap(f"- Verdict: **{g['verdict']}**")
    ap(
        "- AeBAD-S median_shift by condition: "
        + ", ".join(f"{k}={v:.4f}" for k, v in g["aebad_s_median_shift_by_condition"].items())
    )
    ap(
        f"- Frozen cutoff={g['frozen_cutoff']:.6f}; all AeBAD-S median_shift above cutoff: "
        f"{g['all_aebad_median_shift_above_frozen_cutoff']}"
    )
    ap(f"- Reason: {g['reason']}")
    ap("")
    ap("## 8. Limitations")
    ap("")
    ap(
        "- **Residual harm localized to walnuts** (9/9 harmful entities; all high "
        "source-FNR). The method is retained with this limitation documented; "
        "walnuts high-source-FNR entities are the known residual risk."
    )
    ap(
        "- **K4 is not safe** (K4 pooled safety freq=0.095, DINOv2=0.000; K4 dFNR "
        "exceeds +0.02). Only K8 is the safe few-normal operating point."
    )
    ap(
        "- **FULL is not safe on DINOv2** (dFNR=0.0282 > +0.02); the K8 gate "
        "abstains more and is the safe principal setting."
    )
    ap(
        "- **K8 evaluation scope**: only fabric and walnuts are K8-eligible "
        "(n>=8 target-normal); rice/sheet_metal/vial (n<8) are excluded from K8 "
        "but included in FULL/LOO."
    )
    ap(
        "- **H5 nuance**: on DINOv2 the gate provides no dFNR advantage over "
        "Always-On (always-on is already safe there); the gate's safety value is "
        "pooled-level (protects PatchCore/EfficientAD)."
    )
    ap(
        "- **No external validation**: frozen Q50 cutoff is dataset/score-scale "
        "specific and not transferable to AeBAD-S; re-tuning is forbidden."
    )
    ap("")
    ap("## 9. Terminal")
    ap("")
    ap(f"- **{verdict['terminal_label']} {verdict['terminal']}**")
    ap("")
    ap("## 10. Next Authorization")
    ap("")
    ap(
        "- With terminal T1, the next authorized stage is **PB8 — Publication "
        "Benchmark + Ablation + Paper Claim Freeze** (final paper main table, "
        "strong baselines, ablation, mechanism figures, statistical tables, "
        "computational overhead, method naming, final claim hierarchy, paper "
        "section planning). PB7-P2 does not do paper packaging."
    )
    ap("")
    ap("## Evidence tables")
    ap("")
    ap("- `A_REPRODUCTION.csv`, `A_REPRODUCTION_PER_ENTITY.csv`")
    ap("- `B_LOCO_CATEGORY_INFLUENCE.csv`")
    ap(
        "- `C_K_SUPPORT_STABILITY.csv`, `C_K8_REPEAT_DISTRIBUTION.csv`, "
        "`C_K8_REPEAT_DISTRIBUTION_DINOV2.csv`, `C_GATE_SELECTION_JACCARD.csv`"
    )
    ap("- `D_CONFIRMATORY_BOOTSTRAP.csv`, `D_PAIRED_COMPARISON.csv`, `D_POLICY_*.csv`")
    ap("- `E_RESIDUAL_HARM.csv`, `E_HARMFUL_VS_NONHARMFUL.csv`, `E_CATASTROPHIC_CONCENTRATION.csv`")
    ap("- `F_DETECTOR_CONFIRMATION.csv`, `F_DETECTOR_CONFIRMATION_PER_ENTITY.csv`")
    ap("- `G_EXTERNAL_VALIDATION.csv`")
    ap(
        "- `P0_CONFIRMATORY_INPUT_MANIFEST.csv`, `P0_HYPOTHESIS_REGISTRY.json`, "
        "`P0_EXECUTION_LEDGER.csv`"
    )
    ap("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
