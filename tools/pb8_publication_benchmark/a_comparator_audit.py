"""PB8 A — comparator eligibility audit.

Searches the closest literature for a published comparator that is truly
isomorphic to the frozen problem setting (visual industrial anomaly detection
under acquisition shift; frozen detector; threshold-only normal-only
adaptation with K=8 target-normal support; safety-constrained mean dFNR<=+0.02;
selective abstention; no target anomaly labels). Records each candidate's
eligibility and emits A_EXTERNAL_COMPARATOR_ELIGIBILITY.csv + A_GATE.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_publication_benchmark"))

import pb8_common as P8  # type: ignore[import-not-found]  # noqa: E402

OUT = P8.OUT
OUT.mkdir(parents=True, exist_ok=True)

# Frozen eligibility criteria (PB8 A1): all five must hold.
CRITERIA = [
    "no_target_anomaly_label",
    "problem_semantics_close",
    "fair_on_same_detector_score",
    "no_test_outcome_tuning",
    "verifiable_implementation",
]

# Candidates surfaced by the PB8 literature audit (2026-08-20 web search).
# Each row records the audit findings; eligibility is assessed against the
# frozen deployment semantics, not against any outcome.
CANDIDATES = [
    {
        "candidate": "NC-TTDA (Normal-Class Test-Time Domain Adaptation)",
        "publication": "Practical Test-Time Domain Adaptation for Industrial Condition Monitoring by Leveraging Normal-Class Data (PMC12737140, 2025)",
        "target_normal_required": "yes",
        "target_anomaly_required": "no",
        "retraining_required": "yes (adapts pretrained model)",
        "modifies_representation": "yes",
        "threshold_only": "no",
        "same_deployment_semantics": "no (model adaptation, not threshold-only; sensor/time-series domain)",
        "reproducible_from_official": "not verified",
        "eligible": "NO",
        "reason": "Modifies the model/representation; not threshold-only; time-series condition-monitoring domain, not visual acquisition-shift operating-point calibration.",
    },
    {
        "candidate": "TARD (Test-time Domain Adaptation for Robust Fault Detection)",
        "publication": "TARD: Test-time Domain Adaptation for Robust Fault Detection under Evolving Operating Conditions (arXiv 2507.16354, 2025)",
        "target_normal_required": "yes",
        "target_anomaly_required": "no",
        "retraining_required": "yes (domain adaptation module)",
        "modifies_representation": "yes",
        "threshold_only": "no",
        "same_deployment_semantics": "no (feature/domain adaptation; fault-detection on system parameters + sensor measurements)",
        "reproducible_from_official": "not verified",
        "eligible": "NO",
        "reason": "Adapts features via a dedicated domain-adaptation module; not threshold-only; different domain and no safety-constrained selective semantics.",
    },
    {
        "candidate": "When Model Meets New Normals (test-time adaptation for unsupervised time-series AD)",
        "publication": "When Model Meets New Normals: Test-Time Adaptation for Unsupervised Time-Series Anomaly Detection (arXiv 2312.11976)",
        "target_normal_required": "yes",
        "target_anomaly_required": "no",
        "retraining_required": "yes",
        "modifies_representation": "yes",
        "threshold_only": "no",
        "same_deployment_semantics": "no (time-series; representation adaptation)",
        "reproducible_from_official": "not verified",
        "eligible": "NO",
        "reason": "Time-series anomaly detection with representation adaptation; not visual threshold-only calibration.",
    },
    {
        "candidate": "AnoPLe (few-shot AD via bi-directional prompt learning, normal-only)",
        "publication": "AnoPLe: Few-Shot Anomaly Detection via Bi-directional Prompt Learning with Only Normal Samples (arXiv 2408.13516)",
        "target_normal_required": "yes (few normal samples)",
        "target_anomaly_required": "no",
        "retraining_required": "yes (prompt learning)",
        "modifies_representation": "yes",
        "threshold_only": "no",
        "same_deployment_semantics": "no (learns prompts / modifies scoring; not a frozen-detector threshold update)",
        "reproducible_from_official": "not verified",
        "eligible": "NO",
        "reason": "Modifies the scoring representation via prompt learning; not threshold-only operating-point adaptation.",
    },
    {
        "candidate": "VisionAD / CAReg / SubspaceAD (training-free few-shot AD)",
        "publication": "VisionAD (arXiv 2504.11895); CAReg; SubspaceAD (CVPR 2026)",
        "target_normal_required": "yes (few normal samples)",
        "target_anomaly_required": "no",
        "retraining_required": "no (training-free)",
        "modifies_representation": "yes (memory bank / subspace modeling / registration)",
        "threshold_only": "no",
        "same_deployment_semantics": "no (builds new detectors from few normals; not a frozen-detector threshold update under acquisition shift)",
        "reproducible_from_official": "not verified",
        "eligible": "NO",
        "reason": "Construct new few-shot detectors (memory bank / subspace), not a threshold-only adaptation of a frozen detector.",
    },
    {
        "candidate": "Online Adaptive Anomaly Thresholding with Confidence Sequences",
        "publication": "Online Adaptive Anomaly Thresholding with Confidence Sequences (ICML 2024, sun24h)",
        "target_normal_required": "no (online score stream)",
        "target_anomaly_required": "no",
        "retraining_required": "no",
        "modifies_representation": "no",
        "threshold_only": "yes",
        "same_deployment_semantics": "no (online streaming threshold with abstention; no acquisition-shift normal-support calibration, no safety budget on FNR)",
        "reproducible_from_official": "not verified",
        "eligible": "NO",
        "reason": "Online streaming thresholding, not batch acquisition-shift calibration with a frozen detector and K=8 normal support.",
    },
    {
        "candidate": "The Threshold Paradox (threshold calibration critique)",
        "publication": "The Threshold Paradox: Why Calibrating on the Test Set Introduces Bias in Anomaly Detection (SciTePress 2026)",
        "target_normal_required": "n/a",
        "target_anomaly_required": "n/a",
        "retraining_required": "n/a",
        "modifies_representation": "n/a",
        "threshold_only": "n/a",
        "same_deployment_semantics": "no (a critique of test-set threshold calibration, not a deployable method)",
        "reproducible_from_official": "n/a",
        "eligible": "NO",
        "reason": "A methodological critique, not a deployable comparator.",
    },
]


def main() -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for cand in CANDIDATES:
        rows.append(
            {
                "method": cand["candidate"],
                "publication": cand["publication"],
                "target_normal_required": cand["target_normal_required"],
                "target_anomaly_required": cand["target_anomaly_required"],
                "retraining_required": cand["retraining_required"],
                "modifies_representation": cand["modifies_representation"],
                "threshold_only": cand["threshold_only"],
                "same_deployment_semantics": cand["same_deployment_semantics"],
                "reproducible_from_official_public": cand["reproducible_from_official"],
                "eligible_comparator": cand["eligible"],
                "reason": cand["reason"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "A_EXTERNAL_COMPARATOR_ELIGIBILITY.csv", index=False)

    n_eligible = int((df["eligible_comparator"] == "YES").sum())
    verdict = "NO_DIRECT_LITERATURE_COMPARATOR_WITH_MATCHED_DEPLOYMENT_SEMANTICS"
    if n_eligible > 0:
        verdict = "ELIGIBLE_COMPARATOR_FOUND"

    gate = {
        "schema_version": "1.0",
        "stage": "PB8_PUBLICATION_BENCHMARK_ABLATION_CLAIM_FREEZE",
        "phase": "A",
        "timestamp": ts,
        "n_candidates": len(df),
        "n_eligible": n_eligible,
        "criteria": CRITERIA,
        "verdict": verdict,
        "benchmark_rule": (
            "No eligible literature comparator; the main benchmark uses the strongest "
            "protocol-compatible frozen baselines (B0-B7); B8 literature comparator is "
            "NOT included. No SOTA comparison is fabricated."
        ),
        "terminal": "PROCEED_TO_B",
    }
    (OUT / "A_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"A done: {n_eligible}/{len(df)} eligible, verdict {verdict}")


if __name__ == "__main__":
    main()
