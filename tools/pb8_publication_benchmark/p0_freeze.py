"""PB8 P0 — publication evidence freeze.

Binds the sealed PB7-P0/P1/P2 evidence and the frozen method spec into a
SHA256 manifest, freezes the primary estimand and the report hierarchy, and
emits P0_GATE.json.
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


def main() -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- P0.1 bind artifacts (SHA256) -------------------------------------- #
    artifacts = [
        "reports/pb7_p0_threshold_correctability/PB7_P0_THRESHOLD_CORRECTABILITY_FINAL_REPORT.md",
        "reports/pb7_p0_threshold_correctability/F_THRESHOLD_MECHANISM_VERDICT.json",
        "reports/pb7_p0_threshold_correctability/A2_ORACLE_HEADROOM.csv",
        "reports/pb7_p0_threshold_correctability/A3_DETECTOR_HEADROOM_SUMMARY.csv",
        "reports/pb7_p0_threshold_correctability/E_PER_ENTITY.csv",
        "reports/pb7_p0_threshold_correctability/B_FEATURES.csv",
        "reports/pb7_p1_safety_actuator/PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_FINAL_REPORT.md",
        "reports/pb7_p1_safety_actuator/F_PB7_P1_MECHANISM_VERDICT.json",
        "reports/pb7_p1_safety_actuator/E_FROZEN_METHOD_SPEC.json",
        "reports/pb7_p1_safety_actuator/E_TEST_PER_ENTITY.csv",
        "reports/pb7_p1_safety_actuator/E_TEST_DETECTOR_SUMMARY.csv",
        "reports/pb7_p1_safety_actuator/E_TEST_BOOTSTRAP.csv",
        "reports/pb7_p1_safety_actuator/P0_SPLIT.json",
        "reports/pb7_p1_safety_actuator/A_ACTUATOR_FAMILY_REGISTRY.json",
        "reports/pb7_p1_safety_actuator/C_GATE_DEFINITION.json",
        "reports/pb7_p1_safety_actuator/F_SAFETY_UTILITY_CURVE.csv",
        "reports/pb7_p1_safety_actuator/F_DETECTOR_DECOMPOSITION.csv",
        "reports/pb7_p1_safety_actuator/F_RESIDUAL_HARM_ENTITIES.csv",
        "reports/pb7_p2_publication_validation/PB7_P2_PUBLICATION_READINESS_FINAL_REPORT.md",
        "reports/pb7_p2_publication_validation/F_PB7_P2_PUBLICATION_VERDICT.json",
        "reports/pb7_p2_publication_validation/P0_CONFIRMATORY_INPUT_MANIFEST.csv",
        "reports/pb7_p2_publication_validation/P0_CONFIRMATORY_PROTOCOL.json",
        "reports/pb7_p2_publication_validation/A_EXACT_REPRODUCTION.json",
        "reports/pb7_p2_publication_validation/B_LOCO_CATEGORY_INFLUENCE.csv",
        "reports/pb7_p2_publication_validation/C_K_SUPPORT_STABILITY.csv",
        "reports/pb7_p2_publication_validation/D_CONFIRMATORY_BOOTSTRAP.csv",
        "reports/pb7_p2_publication_validation/D_PAIRED_COMPARISON.csv",
        "reports/pb7_p2_publication_validation/E_RESIDUAL_HARM.csv",
        "reports/pb7_p2_publication_validation/F_DETECTOR_CONFIRMATION.csv",
        "reports/pb7_p2_publication_validation/G_EXTERNAL_VALIDATION.csv",
    ]
    rows = [P8.bind_artifact(a) for a in artifacts]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "P0_PUBLICATION_EVIDENCE_MANIFEST.csv", index=False)
    n_bound = int(manifest["exists"].eq("True").sum())

    # ---- P0.2 freeze primary estimand -------------------------------------- #
    estimand = {
        "schema_version": "1.0",
        "stage": "PB8_PUBLICATION_BENCHMARK_ABLATION_CLAIM_FREEZE",
        "phase": "P0",
        "detector": "DINOv2 (dinov2_vits14)",
        "setting": "K8",
        "policy": "frozen PB7-P1/P2 principal (A_MEDIAN_RAW, lambda=0.25, median_shift>=Q50)",
        "primary_metric": "delta_ba (balanced accuracy change vs NoAdapt)",
        "primary_safety_metric": "delta_fnr (mean, budget <= +0.02)",
        "secondary_safety": "harm rate (delta_ba < -0.02 or delta_fnr > +0.02)",
        "tertiary": "coverage, benefit precision, benefit recall, catastrophic rate",
        "safety_budget": 0.02,
        "report_hierarchy": {
            "primary": "DINOv2 K8",
            "secondary": "Pooled K8",
            "mechanism_negative_controls": ["PatchCore K8", "EfficientAD K8"],
            "rule": "pooled results must not precede DINOv2 to avoid a universal impression",
        },
    }
    (OUT / "P0_PUBLICATION_ESTIMAND.json").write_text(
        json.dumps(estimand, indent=2), encoding="utf-8"
    )

    # ---- P0.3 gate ---------------------------------------------------------- #
    gate = {
        "schema_version": "1.0",
        "stage": "PB8_PUBLICATION_BENCHMARK_ABLATION_CLAIM_FREEZE",
        "phase": "P0",
        "timestamp": ts,
        "checks": {
            "n_artifacts_bound": len(artifacts),
            "n_artifacts_exist": n_bound,
            "all_artifacts_exist": n_bound == len(artifacts),
            "primary_estimand_frozen": True,
            "report_hierarchy_frozen": True,
            "safety_budget": 0.02,
            "firewall_declared": True,
        },
        "verdict": "PASS" if n_bound == len(artifacts) else "FAIL",
        "terminal": "PROCEED_TO_A",
    }
    (OUT / "P0_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"P0 done: bound {n_bound}/{len(artifacts)} artifacts, verdict {gate['verdict']}")


if __name__ == "__main__":
    main()
