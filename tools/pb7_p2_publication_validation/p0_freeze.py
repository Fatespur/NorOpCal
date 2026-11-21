"""PB7-P2 Stage P0 — Confirmatory Protocol Freeze.

Binds the sealed PB7-P1 frozen method spec and inputs, registers the
confirmatory hypotheses H1-H7, records the frozen DEV/TEST split and the
frozen principal policy, and produces the confirmatory input manifest
(SHA256-bound) and the execution ledger.

This stage makes no scientific claims; it only freezes the confirmatory
protocol before any robustness test runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p2_publication_validation"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p1_common as P1  # type: ignore[import-not-found]  # noqa: E402
import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

OUT = P2.OUT


def sha256_file(path: Path) -> str:
    return C.sha256_file(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- frozen principal policy (from sealed PB7-P1 spec) ------------------ #
    e_spec = json.loads((P1.OUT / "E_FROZEN_METHOD_SPEC.json").read_text(encoding="utf-8"))
    principal = e_spec["principal_candidate"]
    split = json.loads((P1.OUT / "P0_SPLIT.json").read_text(encoding="utf-8"))
    assert principal["family"] == "A_MEDIAN_RAW"
    assert principal["lambda"] == 0.25
    assert principal["cutoff_key"] == "Q50"

    # ---- input manifest (SHA256-bound) -------------------------------------- #
    rows: list[dict[str, str]] = []

    def add(path: Path, role: str, kind: str, note: str) -> None:
        p = path if path.is_absolute() else (ROOT / path)
        if not p.exists():
            raise FileNotFoundError(f"missing frozen input: {p}")
        rows.append(
            {
                "path": str(p.relative_to(ROOT)),
                "sha256": sha256_file(p),
                "role": role,
                "kind": kind,
                "note": note,
            }
        )

    # PB7-P1 sealed artifacts
    add(
        P1.OUT / "E_FROZEN_METHOD_SPEC.json",
        "DEPLOYABLE",
        "method_spec",
        "PB7-P1 frozen principal method spec",
    )
    add(
        P1.OUT / "C_GATE_DEFINITION.json",
        "DEPLOYABLE",
        "gate_definition",
        "PB7-P1 frozen correctability gate / cutoff definition",
    )
    add(P1.OUT / "P0_SPLIT.json", "DEPLOYABLE", "split", "PB7-P1 frozen DEV/TEST split")
    add(
        P1.OUT / "E_TEST_PER_ENTITY.csv",
        "ORACLE_DIAGNOSTIC_ONLY",
        "test_entities",
        "PB7-P1 frozen TEST per-entity results (sealed reference for exact reproduction)",
    )
    add(
        P1.OUT / "E_TEST_DETECTOR_SUMMARY.csv",
        "ORACLE_DIAGNOSTIC_ONLY",
        "test_summary",
        "PB7-P1 frozen TEST summary (sealed reference)",
    )
    add(
        P1.OUT / "E_TEST_BOOTSTRAP.csv",
        "ORACLE_DIAGNOSTIC_ONLY",
        "test_bootstrap",
        "PB7-P1 frozen TEST bootstrap (sealed reference)",
    )
    add(
        P1.OUT / "F_PB7_P1_MECHANISM_VERDICT.json",
        "ORACLE_DIAGNOSTIC_ONLY",
        "verdict",
        "PB7-P1 sealed mechanism verdict (T4)",
    )
    add(P1.OUT / "P0_PROTOCOL.json", "DEPLOYABLE", "protocol", "PB7-P1 frozen protocol")

    # PB7-P0 evidence
    add(
        P1.PB7P0 / "A2_ORACLE_HEADROOM.csv",
        "ORACLE_DIAGNOSTIC_ONLY",
        "oracle_headroom",
        "PB7-P0 oracle headroom / tau_star_safe (diagnostic)",
    )
    add(
        P1.PB7P0 / "P0_INPUT_MANIFEST.csv",
        "DEPLOYABLE",
        "p0_manifest",
        "PB7-P0 frozen input manifest",
    )

    # score caches (deployable + oracle)
    seen_src: set = set()
    seen_cnd: set = set()
    seen_anm: set = set()
    for cat, cond, det in P1.entity_list():
        src_rel = C.source_rel(cat, det)
        if src_rel not in seen_src:
            add(
                C.SCORES_DIR / src_rel,
                "DEPLOYABLE",
                "source_normal",
                f"source-normal scores {cat}|{det}",
            )
            seen_src.add(src_rel)
        cnd_rel = C.cond_normal_rel(cat, cond, det)
        if cnd_rel not in seen_cnd:
            add(
                C.SCORES_DIR / cnd_rel,
                "DEPLOYABLE",
                "target_normal",
                f"target-normal scores {cat}|{cond}|{det}",
            )
            seen_cnd.add(cnd_rel)
        anm_rel = C.anomaly_rel(cat, det)
        if anm_rel not in seen_anm:
            add(
                C.SCORES_DIR / anm_rel,
                "ORACLE_DIAGNOSTIC_ONLY",
                "anomaly",
                f"target anomaly scores {cat}|{det}",
            )
            seen_anm.add(anm_rel)

    df_manifest = pd.DataFrame(rows)
    df_manifest.to_csv(OUT / "P0_CONFIRMATORY_INPUT_MANIFEST.csv", index=False)

    # ---- hypothesis registry (H1-H7) ---------------------------------------- #
    hypotheses = {
        "H1_category_robustness": {
            "statement": "DINOv2 positive utility is not driven by a single category.",
            "evidence": "LOO category influence audit (Stage B)",
        },
        "H2_safety_robustness": {
            "statement": "DINOv2 satisfies mean dFNR <= +0.02 under reasonable "
            "resampling / category perturbation.",
            "evidence": "LOO audit + confirmatory bootstrap (Stages B, D)",
        },
        "H3_few_normal_robustness": {
            "statement": "K8 results retain the principal utility/safety relative to FULL.",
            "evidence": "K-support confirmatory audit (Stage C)",
        },
        "H4_repeat_robustness": {
            "statement": "K8 results are not driven by a few support realizations.",
            "evidence": "repeated-support stability (Stage C)",
        },
        "H5_selectivity_necessity": {
            "statement": "The frozen gated policy keeps a clear safety advantage over "
            "its frozen Always-On counterpart.",
            "evidence": "paired comparison Gated vs Always-On (Stage D)",
        },
        "H6_detector_dependence": {
            "statement": "DINOv2 vs PatchCore/EfficientAD differences persist under "
            "confirmatory analysis.",
            "evidence": "detector-specificity confirmation (Stage F)",
        },
        "H7_no_single_category_failure_invalidates_mechanism": {
            "statement": "Removing any single category does not systematically flip the "
            "DINOv2 overall conclusion.",
            "evidence": "LOO category influence audit (Stage B)",
        },
    }
    (OUT / "P0_HYPOTHESIS_REGISTRY.json").write_text(
        json.dumps(hypotheses, indent=2), encoding="utf-8"
    )

    # ---- protocol json ------------------------------------------------------- #
    protocol = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "P0",
        "timestamp": ts,
        "inherits_terminal": "PB7_P1_DETECTOR_SPECIFIC_SAFE_ACTUATOR_SUPPORTED",
        "nature": "CONFIRMATORY / PUBLICATION-READINESS (no method development)",
        "frozen_principal_policy": {
            "family": principal["family"],
            "lambda": principal["lambda"],
            "cutoff_key": principal["cutoff_key"],
            "cutoff_value": principal["cutoff_value"],
            "gate": principal["gate"],
            "bounding": principal["bounding"],
            "principal_k": "8",
            "safety_budget": "mean dFNR <= +0.02",
            "primary_utility": "dBA",
            "claim_scope": "DINOv2-specific / detector-dependent",
        },
        "deployment_firewall": {
            "allowed": [
                "source-normal",
                "target-normal",
                "frozen detector",
                "frozen category metadata",
            ],
            "forbidden": [
                "target anomaly labels",
                "target anomaly score distribution",
                "anomaly prevalence",
                "oracle threshold",
                "outcome-derived walnuts indicator",
                "per-test-category tuned parameters",
            ],
        },
        "forbidden_actions": [
            "modify lambda",
            "change Q50 to Q40/Q60",
            "add median+MAD",
            "add high-source-FNR rule",
            "walnuts special abstention",
            "retrain selector for DINOv2",
            "delete failing category",
            "modify BENEFIT",
            "modify safety threshold",
            "change primary metric",
            "outcome-driven detector selection",
            "feature search",
            "XGBoost/RF/NN",
            "reverse-modify PB7-P1 from new test outcomes",
            "claim representation improvement",
            "detector-agnostic claim",
        ],
        "execution_order": ["P0", "A", "B", "C", "D", "E", "F", "G", "FINAL"],
        "split": split,
        "confirmatory_bootstrap": {"B": P2.CONF_BOOT_B, "seed": P2.CONF_BOOT_SEED},
        "repeat_robustness": {"n_repeats": P2.C_REPEATS},
    }
    (OUT / "P0_CONFIRMATORY_PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    # ---- P0 gate ------------------------------------------------------------- #
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "P0",
        "timestamp": ts,
        "checks": {
            "principal_policy_frozen": True,
            "manifest_sha256_bound": True,
            "n_manifest_rows": len(df_manifest),
            "n_source_normal_files": len(seen_src),
            "n_target_normal_files": len(seen_cnd),
            "n_anomaly_files": len(seen_anm),
            "hypotheses_registered": len(hypotheses),
            "firewall_declared": True,
        },
        "verdict": "PASS",
        "terminal": "PROCEED_TO_A",
    }
    (OUT / "P0_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger -------------------------------------------------------------- #
    ledger = pd.DataFrame(
        [
            {
                "stage": "P0",
                "step": "confirmatory_protocol_freeze",
                "status": "COMPLETE",
                "timestamp": ts,
                "artifacts": "P0_CONFIRMATORY_PROTOCOL.json, P0_CONFIRMATORY_INPUT_MANIFEST.csv, "
                "P0_HYPOTHESIS_REGISTRY.json, P0_GATE.json",
                "note": f"P0 gate PASS; manifest rows={len(df_manifest)}; H1-H7 registered",
            }
        ]
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
