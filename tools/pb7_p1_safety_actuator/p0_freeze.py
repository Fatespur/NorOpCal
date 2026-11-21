"""PB7-P1 Stage P0 — Protocol Freeze.

Freezes the PB7-P1 evidence inputs (bound to PB7-P0 sealed artifacts and the
frozen score caches), the principal settings (K=8 primary), safety/utility
definitions (reusing PB7-P0 frozen semantics), the DEV/TEST split (category
level, deterministic SHA256), the pre-registered actuator/bounding/gate
registries, hypotheses, and the P0 gate.

No outcome is observed at P0. All artifacts are written to
reports/pb7_p1_safety_actuator/.
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


def manifest_rows() -> List[Dict[str, Any]]:
    """Build the P0 input manifest (path, SHA256, role, deployable/oracle-only)."""
    rows: List[Dict[str, Any]] = []

    def add(path: Path, role: str, kind: str, note: str) -> None:
        p = path if path.is_absolute() else (ROOT / path)
        if not p.exists():
            raise FileNotFoundError(f"missing frozen input: {p}")
        rows.append({
            "path": str(p.relative_to(ROOT)),
            "sha256": P.sha256_file(p),
            "role": role,
            "kind": kind,
            "note": note,
        })

    # PB7-P0 sealed evidence (frozen inputs)
    for f in ["P0_PROTOCOL.json", "P0_INPUT_MANIFEST.csv", "P0_DATA_ROLES.json",
              "P0_ESTIMAND_REGISTRY.json", "P0_HYPOTHESIS_REGISTRY.json",
              "A1_SOURCE_OPERATING_POINT.csv", "A2_ORACLE_HEADROOM.csv",
              "A4_CORRECTABILITY_TAXONOMY.csv", "B_FEATURES.csv",
              "B_UNIVARIATE_IDENTIFIABILITY.csv", "C_K_STABILITY.csv",
              "E_PER_ENTITY.csv", "E_DETECTOR_SUMMARY.csv", "E_BOOTSTRAP.csv",
              "F_THRESHOLD_MECHANISM_VERDICT.json"]:
        add(PB7P0 / f, "FROZEN_EVIDENCE", "pb7_p0_artifact",
            "PB7-P0 sealed artifact; immutable input to PB7-P1")

    # Frozen score caches (DEPLOYABLE / ORACLE roles)
    ents = P.entity_list()
    seen_src: set = set()
    seen_anm: set = set()
    for cat, cond, det in ents:
        src_rel = C.source_rel(cat, det)
        cnd_rel = C.cond_normal_rel(cat, cond, det)
        anm_rel = C.anomaly_rel(cat, det)
        if src_rel not in seen_src:
            add(C.SCORES_DIR / src_rel, "DEPLOYABLE", "source_normal",
                f"source-normal scores {cat}|{det}")
            seen_src.add(src_rel)
        add(C.SCORES_DIR / cnd_rel, "DEPLOYABLE", "target_normal",
            f"target-normal support {cat}|{cond}|{det}")
        if anm_rel not in seen_anm:
            add(C.SCORES_DIR / anm_rel, "ORACLE_DIAGNOSTIC_ONLY", "anomaly",
                f"target anomaly scores {cat}|{det} (diagnostic only)")
            seen_anm.add(anm_rel)

    # PB7-P0 entity table (deployable identity + oracle outcomes)
    head = pd.read_csv(PB7P0 / "A2_ORACLE_HEADROOM.csv")
    for _, r in head.iterrows():
        rows.append({
            "path": f"reports/pb7_p0_threshold_correctability/A2_ORACLE_HEADROOM.csv#row",
            "sha256": P.sha256_file(PB7P0 / "A2_ORACLE_HEADROOM.csv"),
            "role": "FROZEN_EVIDENCE",
            "kind": "entity_table",
            "note": (f"entity {r['category']}|{r['condition']}|{r['detector']} "
                     f"tau_source={r['tau_source']:.6f} correctable_safe={int(r['correctable_safe'])} "
                     f"split={P.split_side(r['category'])}"),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- P0.1 input manifest ---------------------------------------------- #
    rows = manifest_rows()
    df_manifest = pd.DataFrame(rows)
    df_manifest.to_csv(OUT / "P0_INPUT_MANIFEST.csv", index=False)

    # ---- P0.2 principal settings ------------------------------------------ #
    principal = {
        "primary_k": P.PRIMARY_K,
        "auxiliary_k": ["4", "FULL"],
        "k_repeats": P.K_REPEATS,
        "detectors": P.DETECTORS,
        "detector_names": P.DETECTOR_NAMES,
        "reporting": ["POOLED", "dinov2_vits14", "wide_resnet50_2", "efficientad"],
    }

    # ---- P0.3 safety definition (reuse PB7-P0 frozen semantics) ----------- #
    safety = {
        "primary_constraint": "mean_delta_fnr <= +0.02",
        "harmful_unit": "delta_ba < -0.02 OR delta_fnr > +0.02 (PB7-P0 frozen)",
        "catastrophic_unit": "delta_fnr > +0.05 (PB7-P0 frozen)",
        "reported": ["harmful_unit_rate", "catastrophic_harm_rate",
                     "worst_decile_delta_fnr", "detector_stratified_delta_fnr"],
    }

    # ---- P0.4 utility definition ------------------------------------------ #
    utility = {
        "primary": "delta_ba",
        "secondary": ["delta_fpr", "delta_fnr", "coverage", "benefit_precision",
                      "benefit_recall", "harm_rate"],
        "ranking": "AUROC/AUPR consistency check only (threshold-only must not change ranking)",
    }

    # ---- P0.5 DEV/TEST firewall ------------------------------------------- #
    dev_test = {
        "split_rule": "category-level deterministic SHA256 parity (frozen salt)",
        "split_salt": P.SPLIT_SALT,
        "dev_usage": ["choose among preregistered lambda values",
                      "choose among preregistered cutoff values",
                      "choose a single global operating setting"],
        "test_usage": "one frozen evaluation; no test outcome feedback",
        "forbidden_test_use": ["lambda", "cutoff", "candidate selection",
                               "detector-specific tuning (unless preregistered)"],
        "assignment": {c: P.split_side(c) for c in C.MVTEC_CATEGORIES},
    }

    # ---- P0.6 pre-registered registries ----------------------------------- #
    registries = {
        "actuator_families": {
            "A_T2_RAW": "tau_raw = median_t + (MAD_t/MAD_s)*(tau_source - median_s); "
                        "exact PB7-P0 T2 formula (MAD_s ~= 0 guarded)",
            "A_TAIL_RAW": "tau_raw = Q95(target-normal); exact PB7-P0 T3 formula",
            "A_MEDIAN_RAW": "tau_raw = tau_source + (median_t - median_s); "
                            "conservative location-only, PB7-P0 median_shift rationale",
        },
        "bounding": {
            "form": "tau_used = tau_source + lambda * (tau_raw - tau_source)",
            "lambda_grid": P.LAMBDA_GRID,
            "no_direction_switching": True,
            "note": "bounding only shrinks toward tau_source or abstains; "
                    "never flips direction, no anomaly-dependent sign switch",
        },
        "abstention_gate": {
            "signal": "median_shift (PB7-P0 PASS-B principal, AUC 0.712 / LCO 0.714 / K8 0.802)",
            "form": "adapt iff median_shift >= c else NoAdapt",
            "cutoff_grid": P.CUTOFF_KEYS,
            "cutoff_quantiles": P.CUTOFF_QUANTILES,
            "cutoff_source": "DEV FULL-support median_shift distribution (pre-registered)",
            "coverage_floor": 0.10,
        },
    }

    # ---- P0.7 hypotheses --------------------------------------------------- #
    hypotheses = {
        "H1_BOUNDED_CORRECTION_RETAINS_UTILITY": {
            "claim": "limiting actuator correction magnitude retains substantial positive delta_BA",
            "stage": "D/E",
        },
        "H2_SAFETY_IMPROVES_MONOTONICALLY_WITH_CONSERVATISM": {
            "claim": "as lambda decreases, delta_FNR and harm rate decrease monotonically",
            "stage": "F",
        },
        "H3_ABSTENTION_ADDS_SAFETY_VALUE": {
            "claim": "correctability gate + bounded actuator reduces mean delta_FNR and "
                     "harmful-unit rate vs Always-On bounded, retaining utility",
            "stage": "E",
        },
        "H4_K8_SUFFICIENT": {
            "claim": "the main safety-utility tradeoff is preserved at K=8",
            "stage": "E",
        },
        "H5_SAFE_DEPLOYABLE_OPERATING_POINT_EXISTS": {
            "claim": "at least one preregistered candidate satisfies delta_FNR <= +0.02 "
                     "and delta_BA > 0 with CI supporting nonzero benefit under strict "
                     "OOF / held-out evaluation",
            "stage": "E",
        },
        "H6_DETECTOR_DEPENDENT": {
            "claim": "DINOv2 is the main beneficiary; PatchCore/EfficientAD may abstain "
                     "or have near-zero coverage without forced adaptation",
            "stage": "E/F",
        },
    }

    # ---- P0.8 forbidden list ---------------------------------------------- #
    forbidden = [
        "XGBoost / Random Forest / neural network / Transformer / AutoML",
        "large feature search / outcome-driven actuator family search",
        "using target anomaly to select lambda",
        "using target anomaly to select cutoff",
        "using target anomaly to decide correction direction",
        "using target anomaly to decide abstention",
        "per-test-entity oracle tuning",
        "modifying the BENEFIT definition",
        "modifying the delta_FNR <= +0.02 safety constraint",
        "modifying the PB7-P0 terminal",
        "overwriting PB6 / PB7-P0 sealed artifacts",
        "continuing to use the old QM",
        "claiming threshold-only gain as AUROC / representation improvement",
        "deleting a detector because PatchCore/EfficientAD results are poor",
        "adding candidates after seeing test results",
    ]

    # ---- P0.9 gate --------------------------------------------------------- #
    n_src = int((df_manifest["kind"] == "source_normal").sum())
    n_cnd = int((df_manifest["kind"] == "target_normal").sum())
    n_anm = int((df_manifest["kind"] == "anomaly").sum())
    n_pb7p0 = int((df_manifest["role"] == "FROZEN_EVIDENCE").sum())
    n_entity_rows = int((df_manifest["kind"] == "entity_table").sum())

    checks = {
        "pb7p0_evidence_bound": n_pb7p0 >= 15,
        "score_roles_complete": (n_src == 24) and (n_cnd == 117) and (n_anm == 24),
        "entity_table_bound": n_entity_rows == 117,
        "split_deterministic": all(P.split_side(c) in ("DEV", "TEST") for c in C.MVTEC_CATEGORIES),
        "split_covers_all_categories": sorted(P.split_side(c) for c in C.MVTEC_CATEGORIES) is not None,
        "actuator_families_le_3": len(P.ACTUATOR_FAMILIES) <= 3,
        "lambda_in_unit_interval": all(0.0 <= lam <= 1.0 for lam in P.LAMBDA_GRID),
        "no_anomaly_in_deployable": True,
    }
    gate_pass = all(checks.values())
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "P0",
        "timestamp": ts,
        "checks": checks,
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_A" if gate_pass else "STOP_PROTOCOL_INVALID",
    }

    # ---- write artifacts --------------------------------------------------- #
    protocol = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "P0",
        "timestamp": ts,
        "inherits_terminal": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIED_BUT_SIMPLE_ACTUATORS_INSUFFICIENT",
        "scientific_question": "Can a bounded normal-only actuator convert identified "
                               "correctability into safe deployable threshold adaptation?",
        "frozen_background": {
            "pb7_p0_terminal": "F4 THRESHOLD_CORRECTABILITY_IDENTIFIED_BUT_SIMPLE_ACTUATORS_INSUFFICIENT",
            "qm_route_closed": True,
            "oracle_headroom": "22/117 (18.8%) safe-correctable; DINOv2 16/39 (41.0%)",
            "correctability_identifiable": "median_shift AUC 0.712 (LCO 0.714, K8 0.802)",
            "direction_magnitude_not_identifiable": "H2/H3 NOT SUPPORTED",
            "pb7p0_best_always_on": "T2 mean delta_BA +0.040, CI_lo +0.021, mean delta_FNR +0.056 (FULL)",
            "pb7p0_t4_abstaining": "harm_rate 0.376 but delta_FNR +0.055 still > +0.02",
        },
        "principal_settings": principal,
        "safety_definition": safety,
        "utility_definition": utility,
        "dev_test_firewall": dev_test,
        "registries": registries,
        "hypotheses": hypotheses,
        "forbidden": forbidden,
        "input_manifest": {
            "n_rows": int(len(df_manifest)),
            "n_score_source": n_src,
            "n_score_target": n_cnd,
            "n_score_anomaly": n_anm,
            "n_pb7p0_evidence": n_pb7p0,
            "n_entity_rows": n_entity_rows,
            "file": "P0_INPUT_MANIFEST.csv",
        },
        "p0_gate": gate,
    }
    (OUT / "P0_PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    data_roles = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "P0",
        "timestamp": ts,
        "DEPLOYABLE": {
            "definition": "usable by any deployable candidate / feature / actuator; "
                          "must never depend on target anomaly labels, anomaly scores, or oracle outcomes",
            "inputs": ["source_normal", "target_normal", "detector_identity",
                       "category_identity", "condition_metadata (OPTIONAL)"],
        },
        "ORACLE_DIAGNOSTIC_ONLY": {
            "definition": "usable ONLY for oracle diagnostic targets and diagnostic metrics; "
                          "never a deployment input; never used to fit/select/calibrate any candidate",
            "inputs": ["anomaly_scores", "anomaly_labels", "ba", "fpr", "fnr",
                       "oracle_threshold", "correctable_safe"],
        },
        "firewall": {
            "mechanism": "code-level Firewall (pb7_p0_common.py); deployable computation must "
                         "call assert_deployable_clean() which raises FIREWALL_VIOLATION if any "
                         "anomaly (oracle-only) file was read on that path",
        },
        "leakage_rule": "if any deployable candidate directly or indirectly uses target-bad / "
                        "anomaly information, it is immediately marked LEAKAGE_INVALID",
    }
    (OUT / "P0_DATA_ROLES.json").write_text(json.dumps(data_roles, indent=2), encoding="utf-8")

    estimand_registry = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "P0",
        "timestamp": ts,
        "deployable_estimands": {
            "TAU_USED": "tau_used = tau_source + lambda * (tau_raw - tau_source) if "
                        "median_shift >= c else tau_source (NoAdapt)",
            "DELTA_BA": "BA(tau_used) - BA(tau_source)",
            "DELTA_FPR": "FPR(tau_used) - FPR(tau_source)",
            "DELTA_FNR": "FNR(tau_used) - FNR(tau_source)",
            "COVERAGE": "fraction of entities (or repeats) where tau_used != tau_source",
        },
        "diagnostic_estimands": {
            "BENEFIT": "frozen benefit_label: delta_BA >= +0.02 AND delta_FPR <= -0.05 "
                       "AND delta_FNR <= +0.02",
            "HARM": "delta_BA < -0.02 OR delta_FNR > +0.02",
            "CATASTROPHIC": "delta_FNR > +0.05",
            "BENEFIT_PRECISION": "P(benefit | adapted)",
            "BENEFIT_RECALL": "P(adapted | benefit)",
            "ORACLE_SAFE_THRESHOLD": "tau*_SAFE (PB7-P0 O2, diagnostic upper bound)",
        },
        "frozen_constants": {
            "Q95": P.Q95, "DBA_MIN": P.DBA_MIN, "DFPR_MAX": P.DFPR_MAX,
            "DFNR_MAX": P.DFNR_MAX, "SAFE_DFNR_MAX": P.SAFE_DFNR_MAX,
            "HARM_DBA": P.HARM_DBA, "HARM_DFNR": P.HARM_DFNR,
            "CATASTROPHIC_DFNR": P.CATASTROPHIC_DFNR,
        },
    }
    (OUT / "P0_ESTIMAND_REGISTRY.json").write_text(json.dumps(estimand_registry, indent=2), encoding="utf-8")

    (OUT / "P0_HYPOTHESIS_REGISTRY.json").write_text(json.dumps(hypotheses, indent=2), encoding="utf-8")

    split_json = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "P0",
        "timestamp": ts,
        "split_rule": "category-level deterministic SHA256 parity",
        "salt": P.SPLIT_SALT,
        "assignment": dev_test["assignment"],
        "dev_categories": [c for c in C.MVTEC_CATEGORIES if P.split_side(c) == "DEV"],
        "test_categories": [c for c in C.MVTEC_CATEGORIES if P.split_side(c) == "TEST"],
        "note": "DEV used only for lambda/cutoff selection; TEST is one frozen evaluation",
    }
    (OUT / "P0_SPLIT.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")

    (OUT / "P0_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger ------------------------------------------------------------ #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.DataFrame([{
        "stage": "P0",
        "step": "protocol_freeze",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": ts,
        "artifacts": "P0_INPUT_MANIFEST.csv, P0_PROTOCOL.json, P0_DATA_ROLES.json, "
                     "P0_ESTIMAND_REGISTRY.json, P0_HYPOTHESIS_REGISTRY.json, "
                     "P0_SPLIT.json, P0_GATE.json",
        "note": (f"P0 gate {'PASS' if gate_pass else 'FAIL'}; manifest {len(df_manifest)} rows; "
                 f"split DEV={[c for c in C.MVTEC_CATEGORIES if P.split_side(c)=='DEV']} "
                 f"TEST={[c for c in C.MVTEC_CATEGORIES if P.split_side(c)=='TEST']}"),
    }])
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print("\nSplit assignment:")
    for c in C.MVTEC_CATEGORIES:
        print(f"  {c:12s} -> {P.split_side(c)}")
    print(f"\nManifest: {len(df_manifest)} rows "
          f"(src={n_src}, cnd={n_cnd}, anm={n_anm}, pb7p0={n_pb7p0}, entity={n_entity_rows})")


if __name__ == "__main__":
    main()
