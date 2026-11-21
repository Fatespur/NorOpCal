"""PB8-S1 P0 — evidence freeze (metadata-only).

Binds the predecessor artifacts that the supplementary stage inherits and
freezes the original PB8 principal scope + support semantics. NO anomaly
outcome is read or bound here; oracle-only artifacts are recorded as excluded
from Phase-I rather than hashed.
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
PB8 = ROOT / "reports" / "pb8_publication_benchmark"
P1 = ROOT / "reports" / "pb7_p1_safety_actuator"
P2 = ROOT / "reports" / "pb7_p2_publication_validation"
P0D = ROOT / "reports" / "p0_dataset_opportunity"


def _bind(rel: str, role: str) -> dict:
    p = ROOT / rel
    return {
        "path": rel,
        "size": p.stat().st_size if p.exists() else "",
        "sha256": S1.sha256_file(p) if p.exists() else "",
        "role": role,
    }


def _bind_deployable_normal_scores() -> list[dict]:
    """Phase-I binds the normal-only score caches used to measure support
    availability. Anomaly score files are oracle-only and excluded from
    Phase-I, exactly as the outcome firewall requires."""
    rows = []
    for cat in C.MVTEC_CATEGORIES:
        rows.append(
            _bind(
                f"reports/p0_dataset_opportunity/_cache/scores/{C.source_rel(cat, S1.DETECTOR)}",
                "deployable source-normal (DINOv2)",
            )
        )
    for cat, cond in S1.unit_list():
        rows.append(
            _bind(
                f"reports/p0_dataset_opportunity/_cache/scores/"
                f"{C.cond_normal_rel(cat, cond, S1.DETECTOR)}",
                "deployable target-normal (DINOv2)",
            )
        )
    return rows


def _scope() -> dict:
    split = {c: S1.split_side(c) for c in C.MVTEC_CATEGORIES}
    return {
        "schema_version": "1.0",
        "stage": "PB8_S1_AD2_K8_EVIDENCE_BREADTH",
        "phase": "P0",
        "note": "Original PB8 principal scope frozen verbatim; PB8-S1 may not alter it.",
        "n_total_categories": 8,
        "categories": C.MVTEC_CATEGORIES,
        "dev_test_split": split,
        "original_k8_eligible_categories": ["fabric", "walnuts"],
        "original_test_k8_dinov2_units": 10,
        "original_test_k8_pooled_units": 30,
        "original_support_definition_semantics": (
            "unit = category x acquisition-condition x detector; target-normal = "
            "test_public good images under that acquisition-condition filename suffix; "
            "source = test_public good_regular; K8 eligibility = TEST-side unit with "
            "target-normal FULL support n >= 8; anomaly = public_bad_regular (frozen)."
        ),
        "frozen_q50": S1.Q50_CUTOFF,
        "frozen_lambda": S1.LAMBDA,
        "frozen_safety_budget_dfNR": S1.DFNR_MAX,
        "frozen_principal_k": "8",
        "frozen_principal_result_preserved": {
            "fabric_plus_walnuts": True,
            "n_DINOv2_K8_units": 10,
            "delta_ba_pp": 3.23,
            "ci_95_pp": [2.70, 3.76],
            "mean_delta_fnr_pp": 0.24,
            "harm_pct": 3.67,
            "coverage_pct": 40.33,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    binds = [
        _bind(
            "reports/pb8_publication_benchmark/PB8_PUBLICATION_BENCHMARK_FINAL_REPORT.md",
            "predecessor PB8 final report",
        ),
        _bind(
            "reports/pb8_publication_benchmark/F_PB8_PUBLICATION_VERDICT.json",
            "predecessor PB8 terminal verdict",
        ),
        _bind(
            "reports/pb8_publication_benchmark/FINAL_CLAIM_HIERARCHY.md",
            "predecessor PB8 claim hierarchy (frozen)",
        ),
        _bind(
            "reports/pb7_p1_safety_actuator/E_FROZEN_METHOD_SPEC.json",
            "frozen NorOpCal method spec",
        ),
        _bind(
            "reports/pb8_publication_benchmark/D_K_SUPPORT_ELIGIBILITY.csv",
            "original K-support eligibility table",
        ),
        _bind(
            "reports/pb8_publication_benchmark/D_K_SUPPORT_FINAL.csv",
            "original K-support final (frozen)",
        ),
        _bind(
            "reports/pb8_publication_benchmark/D_EVIDENCE_BREADTH_VERDICT.json",
            "PB8 EB2 evidence-breadth verdict",
        ),
        _bind(
            "reports/pb8_publication_benchmark/P0_PUBLICATION_ESTIMAND.json",
            "PB8 publication estimand",
        ),
        _bind(
            "reports/p0_dataset_opportunity/P0_DATASET_MANIFEST.json",
            "MVTec AD 2 dataset manifest (metadata source)",
        ),
        _bind(
            "reports/pb1ds_publication/pb6_real_third_detector/PB1DS_PB6_D5_OOF_PER_UNIT.csv",
            "PB6 authoritative entity/unit table (39 units x 3 detectors)",
        ),
    ]
    binds += _bind_deployable_normal_scores()
    # Explicitly record the oracle-only exclusion (not hashed in Phase-I).
    binds.append(
        {
            "path": "reports/p0_dataset_opportunity/_cache/scores/*_public_bad_*_dinov2_vits14_scores.npz",
            "size": "(excluded)",
            "sha256": "(excluded from Phase-I; oracle-only anomaly outcomes)",
            "role": "ORACLE_DIAGNOSTIC_ONLY — not read or bound during Phase-I",
        }
    )

    manifest = pd.DataFrame(binds)
    manifest.to_csv(OUT / "P0_S1_INPUT_MANIFEST.csv", index=False)

    scope = _scope()
    (OUT / "P0_ORIGINAL_PB8_SCOPE.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")

    ds_hash = S1.sha256_json(scope)
    gate = {
        "schema_version": "1.0",
        "phase": "P0",
        "verdict": "PASS",
        "n_bound_artifacts": len(binds),
        "deployable_normal_score_files_bound": len(binds) - 1,
        "oracle_only_excluded_from_phase1": True,
        "anomaly_outcome_loaded": False,
        "frozen_scope_sha256": ds_hash,
        "terminal": "PROCEED_TO_A",
    }
    (OUT / "P0_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    _ledger(
        "P0", "P0_S1_INPUT_MANIFEST.csv; P0_ORIGINAL_PB8_SCOPE.json; P0_GATE.json", "PASS", ds_hash
    )
    print(f"P0 done: {len(binds)} artifacts bound; scope sha={ds_hash[:12]}")


def _ledger(stage: str, artifact: str, status: str, sha: str) -> None:
    import datetime

    ledger = OUT / "P0_EXECUTION_LEDGER.csv"
    row = {
        "stage": stage,
        "phase": "P0",
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
