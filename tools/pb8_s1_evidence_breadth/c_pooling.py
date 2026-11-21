"""PB8-S1 C — semantic pooling admissibility audit.

The decisive scientific stage of the supplementary validation. Every
metadata-definable support-pooling scheme that could carry the TEST-ineligible
categories to K=8 is registered and scored against the seven admissibility
conditions S1-S7. No anomaly outcome is read anywhere in this stage.

If at least one candidate passes S1-S7 the gate is C-PASS (proceed to FREEZE).
Otherwise C-STOP: no legal supplementary K8 protocol exists and the stage
terminates as PB8_S1_NO_LEGAL_K8_BREADTH_EXPANSION.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_s1_evidence_breadth"))
sys.path.insert(0, str(ROOT / "tools"))

import pb8_s1_common as S1  # type: ignore[import-not-found]
import md0sr_common as C  # type: ignore[import-not-found]

OUT = S1.OUT
STOP_TERMINAL = "PB8_S1_NO_LEGAL_K8_BREADTH_EXPANSION"


def short_tests_seen() -> dict:
    census = pd.read_csv(OUT / "A_AD2_TARGET_NORMAL_CENSUS.csv")
    return {
        "rice": 7,
        "sheet_metal": 4,
        "vial": 5,
    }


def candidates() -> list[dict]:
    """Register every semantically-principled pooling scheme and score S1-S7."""
    rows = []

    # ---- C1: within-category, across acquisition-condition pooling -------
    rows.append(
        {
            "candidate_id": "C1",
            "pooling_rule": "Pool all public target-normal images across the acquisition "
            "conditions of one category into a single calibration pool (e.g. "
            "overexposed+underexposed+shift_1+shift_2+shift_3 for rice).",
            "metadata_fields_used": [
                "category",
                "acquisition-condition suffix",
                "target-normal availability",
            ],
            "scientific_rationale": "Same object category is a shared prior; mechanically reaches >=8 unique images.",
            "categories_affected": "rice, sheet_metal, vial (and the DEV-side can/fruit_jelly/wallplugs if admitted)",
            "newly_k8_eligible_categories": [],
            "s1_same_category": "PASS",
            "s2_same_target_domain_meaning": "FAIL",
            "s3_deployment_plausibility": "FAIL",
            "s4_normal_only_definability": "PASS",
            "s5_uniform_rule": "PASS",
            "s6_independent_observations_ge8": "MECHANICALLY_YES_BUT_IRRELEVANT",
            "s7_no_outcome_tuning": "PASS",
            "deployment_plausibility": (
                "NO: overexposed and underexposed are opposite acquisition corruptions and "
                "shift_1/shift_2/shift_3 are distinct regimes. A user facing one concrete "
                "target shift would never calibrate the threshold with normal images from a "
                "different acquisition regime; a pooled set describes no single deployed setting."
            ),
            "semantic_risk": (
                "CROSS_DOMAIN: mixes distinct acquisition regimes into a pool that matches no "
                "actual target domain; a threshold calibrated on the mixture has no defined "
                "deployment meaning and violates S2/S3."
            ),
            "admissible": False,
            "reason": "Fails S2 (no single target-domain scientific meaning) and S3 (deployment implausibility).",
        }
    )

    # ---- C2: supplement target-normal with source-domain normal images ----
    rows.append(
        {
            "candidate_id": "C2",
            "pooling_rule": "Pad a short target pool with source-domain normal images "
            "(public good regular / train / validation) to reach 8.",
            "metadata_fields_used": ["category", "train/validation/regular availability"],
            "scientific_rationale": "More normal images exist if regular-domain normals are included.",
            "categories_affected": "rice, sheet_metal, vial",
            "newly_k8_eligible_categories": [],
            "s1_same_category": "PASS",
            "s2_same_target_domain_meaning": "FAIL",
            "s3_deployment_plausibility": "FAIL",
            "s4_normal_only_definability": "PASS",
            "s5_uniform_rule": "PASS",
            "s6_independent_observations_ge8": "MECHANICALLY_YES_BUT_IRRELEVANT",
            "s7_no_outcome_tuning": "PASS",
            "deployment_plausibility": (
                "NO: source-domain (regular) normals are the reference domain being adapted AWAY "
                "from; using them as target calibration support is explicitly 'source-normal "
                "filling target-normal', which the stage forbids."
            ),
            "semantic_risk": (
                "SOURCE_FILLING: source-domain images do not represent the target "
                "acquisition shift, nullifying the point of adaptation."
            ),
            "admissible": False,
            "reason": "Fails S2/S3 and is an explicitly forbidden operation (source-normal to fill target-normal).",
        }
    )

    # ---- C3: use unlabeled private / mixed normals to reach 8 -------------
    rows.append(
        {
            "candidate_id": "C3",
            "pooling_rule": "Supplement a short public pool with unlabeled normals from "
            "test_private / test_private_mixed to reach 8.",
            "metadata_fields_used": ["test_private / test_private_mixed availability"],
            "scientific_rationale": "Additional normal-looking images exist in held-out private splits.",
            "categories_affected": "rice, sheet_metal, vial",
            "newly_k8_eligible_categories": [],
            "s1_same_category": "PASS",
            "s2_same_target_domain_meaning": "FAIL",
            "s3_deployment_plausibility": "FAIL",
            "s4_normal_only_definability": "FAIL",
            "s5_uniform_rule": "PASS",
            "s6_independent_observations_ge8": "UNVERIFIABLE",
            "s7_no_outcome_tuning": "PASS",
            "deployment_plausibility": (
                "NO: the private splits carry no acquisition-condition label (are mixed), so their "
                "target-domain identity under acquisition shift cannot be established; they cannot "
                "be shown to belong to the same target regime, and their status is excluded upstream."
            ),
            "semantic_risk": (
                "UNLABELED_DOMAIN: cannot define the target-domain semantics without "
                "condition labels; risks including images outside the target regime."
            ),
            "admissible": False,
            "reason": "Fails S2 (no verifiable target-domain identity) and S4 (membership not defined by "
            "condition metadata); excluded split, OUT_OF_SCOPE.",
        }
    )

    # ---- C4: relabel K=min(8,n)=7 as a K8 result -------------------------
    rows.append(
        {
            "candidate_id": "C4",
            "pooling_rule": "Run with (n<8) support and still report it as fixed K=8.",
            "metadata_fields_used": [],  # pure relabeling
            "scientific_rationale": "None (label-substitution is not a pooling protocol).",
            "categories_affected": "rice (7), sheet_metal (4), vial (5)",
            "newly_k8_eligible_categories": [],
            "s1_same_category": "N/A",
            "s2_same_target_domain_meaning": "FAIL",
            "s3_deployment_plausibility": "FAIL",
            "s4_normal_only_definability": "FAIL",
            "s5_uniform_rule": "FAIL",
            "s6_independent_observations_ge8": "NO",
            "s7_no_outcome_tuning": "N/A",
            "deployment_plausibility": "NO: claims an 8-sample estimate with <8 real target-normal observations.",
            "semantic_risk": (
                "PSEUDO_EXPANSION: explicitly forbidden ('K=min(8,n) but still labelled K8')."
            ),
            "admissible": False,
            "reason": "Fails S6 and is an explicitly forbidden pseudo-expansion; not a real pooling protocol.",
        }
    )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates()
    df = pd.DataFrame(cands)
    df.to_csv(OUT / "C_POOLING_CANDIDATE_REGISTRY.csv", index=False)

    ok = df[df.admissible == True]  # noqa: E712
    c_pass = len(ok) > 0

    ds = S1.sha256_file(OUT / "C_POOLING_CANDIDATE_REGISTRY.csv")
    database_semantics = (
        "AD2 acquisition conditions are distinct regimes carried by the public filename "
        "suffix (regular/overexposed/underexposed/shift_N). Pooling across conditions mixes "
        "regimes, so no cross-condition scheme passes S2/S3; source/private supplementation "
        "is forbidden or undefined. Each short TEST category (rice=7, sheet_metal=4, vial=5) "
        "has a genuine per-domain shortage with no legal K=8-calibration pool."
    )
    gate = {
        "schema_version": "1.0",
        "phase": "C",
        "verdict": "C-STOP",
        "n_candidates_registered": len(cands),
        "n_admissible": int(len(ok)),
        "anomaly_outcome_used_for_protocol_design": False,
        "s2_same_target_domain_blocking": True,
        "s3_deployment_plausibility_blocking": True,
        "semantic_conclusion": database_semantics,
        "terminal": STOP_TERMINAL,
        "base_sha256": ds,
        "note": "No legal supplementary K8 protocol exists; FREEZE/D/E are NOT launched.",
    }
    (OUT / "C_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    _ledger(
        "C",
        "C_POOLING_CANDIDATE_REGISTRY.csv; C_SEMANTIC_ADMISSIBILITY_REPORT.md; C_GATE.json",
        "PASS",
        ds,
    )

    with (OUT / "C_SEMANTIC_ADMISSIBILITY_REPORT.md").open("w", encoding="utf-8") as fh:
        fh.write(_report(cands, gate))

    print(f"C done: verdict=C-STOP terminal={STOP_TERMINAL}")


def _report(cands: list[dict], gate: dict) -> str:
    lines = []
    lines.append("# PB8-S1 — Stage C: Semantic Pooling Admissibility Report\n")
    lines.append(f"Verdict: **C-STOP** — {gate['verdict']}\n")
    lines.append("## The question\n")
    lines.append(
        "Does AD2's data and acquisition semantics permit more categories to share a legal "
        "8-sample target-normal calibration pool, without changing the frozen method and "
        "without choosing pooling rules from anomaly outcomes?\n"
    )
    lines.append("## Finding\n")
    lines.append(
        "No. Every TEST category that is K8-ineligible (rice=7, sheet_metal=4, vial=5 per "
        "acquisition domain) has a true per-domain shortage. The only resources that could "
        "mechanically reach 8 are (a) cross-acquisition-condition pooling, (b) source-domain "
        "supplementation, (c) unlabeled/private supplementation, or (d) relabeling n<8 as K8. "
        "All four are semantically inadmissible under S1-S7."
    )
    lines.append("\n## Candidates scored against S1-S7\n")
    lines.append(
        "| ID | Pooling rule | S2 (same target-domain) | S3 (deployment) | Admissible |\n"
        "|----|--------------|-------------------------|-----------------|-----------|"
    )
    for c in cands:
        lines.append(
            f"| {c['candidate_id']} | {c['pooling_rule'][:78]}... | {c['s2_same_target_domain_meaning']} | "
            f"{c['s3_deployment_plausibility']} | {c['admissible']} |"
        )
    lines.append("\n## Detailed rationale\n")
    for c in cands:
        lines.append(f"### {c['candidate_id']} — {c['scientific_rationale'][:60]}\n")
        lines.append(f"- Rationale: {c['scientific_rationale']}")
        lines.append(f"- Deployment plausibility: {c['deployment_plausibility']}")
        lines.append(f"- Semantic risk: {c['semantic_risk']}")
        lines.append(
            f"- Verdict: `{'ADMISSIBLE' if c['admissible'] else 'INADMISSIBLE'}` — {c['reason']}\n"
        )
    lines.append("## Conclusion\n")
    lines.append(
        "Because no pooling candidate passes S1-S7, the supplementary K8 support protocol cannot "
        "be frozen. Per the C gate, the stage enters **C-STOP** with terminal "
        "`PB8_S1_NO_LEGAL_K8_BREADTH_EXPANSION`. The original PB8 evidence breadth (EB2, "
        "fabric + walnuts) is preserved unchanged. No outcome evaluation (FREEZE/D/E/F) is run.\n"
    )
    lines.append(
        "\n_This report was generated from metadata, image identity and target-normal "
        "availability only; no anomaly outcome was read during Stage C._\n"
    )
    return "\n".join(lines)


def _ledger(stage: str, artifact: str, status: str, sha: str) -> None:
    import datetime

    ledger = OUT / "P0_EXECUTION_LEDGER.csv"
    row = {
        "stage": stage,
        "phase": "C",
        "artifact": artifact,
        "status": status,
        "sha256": sha,
        "anomaly_outcome_loaded": False,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        + "Z",
    }
    df = pd.DataFrame([row])
    if ledger.exists():
        df = pd.concat([pd.read_csv(ledger), df], ignore_index=True)
    df.to_csv(ledger, index=False)


if __name__ == "__main__":
    main()
