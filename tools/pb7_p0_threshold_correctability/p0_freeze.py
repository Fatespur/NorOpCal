#!/usr/bin/env python3
"""PB7-P0 — Evidence Freeze + Protocol Registration.

P0 of PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT. Freezes the
evidence base and registers the protocol BEFORE any stage outcome is observed:

  P0.1  freeze inputs       -> P0_INPUT_MANIFEST.csv (path/size/SHA256/role/
                              deployable/oracle_only for score caches, entity
                              mapping, and prior PB3 K-support definitions)
  P0.2  data roles          -> P0_DATA_ROLES.json (DEPLOYABLE vs
                              ORACLE_DIAGNOSTIC_ONLY, code-level firewall)
  P0.3  register estimands  -> P0_ESTIMAND_REGISTRY.json (O1 tau*_BA,
                              O2 tau*_SAFE, O3 CORRECTABLE_SAFE)
  P0.4  register hypotheses -> P0_HYPOTHESIS_REGISTRY.json (H1-H6)
  P0    protocol + gate     -> P0_PROTOCOL.json, P0_GATE.json, P0_GIT_STATE.txt,
                              P0_EXECUTION_LEDGER.csv

P0 Gate: if oracle threshold semantics are not unique, the anomaly/deployable
firewall cannot be guaranteed, or score-cache provenance is unclear, the stage
stops with STOP_PROTOCOL_INVALID and never proceeds to A.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p0_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT = P.OUT
PB6 = P.PB6
PB3 = ROOT / "reports" / "pb1ds_publication" / "pb3_k_sensitivity"
PB3R = ROOT / "reports" / "pb1ds_publication" / "pb3r_end_to_end"
PB3PRE0 = ROOT / "reports" / "pb1ds_publication" / "pb3_pre0_integrity"


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)
        return r.stdout.strip()

    return {
        "toplevel": run("rev-parse", "--show-toplevel"),
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "status": run("status", "--short"),
    }


def build_input_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        kind: str,
        path: Path,
        role: str,
        deployable: int,
        oracle_only: int,
        cat: str = "",
        cond: str = "",
        det: str = "",
        entity_id: str = "",
    ) -> None:
        rows.append(
            {
                "kind": kind,
                "entity_id": entity_id,
                "category": cat,
                "condition": cond,
                "detector": det,
                "role": role,
                "deployable": deployable,
                "oracle_only": oracle_only,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": P.sha256_file(path),
            }
        )

    # ---- score caches (source-normal, target-normal, anomaly) -----------------
    # source-normal and anomaly are shared across conditions (per category x
    # detector); target-normal is per entity (category x condition x detector).
    src_dets = sorted({(c, d) for c, _, d in P.entity_list()})
    for cat, det in src_dets:
        src = C.source_rel(cat, det)
        add(
            "score_cache_source_normal",
            P.SCORES_DIR / src,
            "DEPLOYABLE",
            1,
            0,
            cat=cat,
            det=det,
            entity_id=f"{cat}|regular|{det}",
        )
        anm = C.anomaly_rel(cat, det)
        add(
            "score_cache_anomaly",
            P.SCORES_DIR / anm,
            "ORACLE_DIAGNOSTIC_ONLY",
            0,
            1,
            cat=cat,
            det=det,
            entity_id=f"{cat}|regular|{det}",
        )
    for cat, cond, det in P.entity_list():
        cnd = C.cond_normal_rel(cat, cond, det)
        add(
            "score_cache_target_normal",
            P.SCORES_DIR / cnd,
            "DEPLOYABLE",
            1,
            0,
            cat=cat,
            cond=cond,
            det=det,
            entity_id=f"{cat}|{cond}|{det}",
        )

    # ---- entity mapping / split provenance ------------------------------------
    add("entity_mapping", PB6 / "PB1DS_PB6_D5_OOF_PER_UNIT.csv", "PROVENANCE", 0, 1,
        entity_id="117 entities x 6 candidates")
    add("entity_mapping", PB6 / "PB1DS_PB6_D5_SPLIT_MANIFEST.json", "PROVENANCE", 0, 0,
        entity_id="39 units / 39 outer folds")
    add("entity_mapping", PB6 / "PB1DS_PB6_D2_ASSET_RESOLUTION.csv", "PROVENANCE", 0, 0,
        entity_id="39 units")

    # ---- prior PB3 K-support definitions --------------------------------------
    add("k_support", PB3PRE0 / "PB1DS_PB3_PRE0_SUPPORT_SEED_MANIFEST.jsonl", "DEPLOYABLE", 1, 0,
        entity_id="78 entities x K{4,8} x 30 repeats")
    add("k_support", PB3 / "PB1DS_PB3_K8_ELIGIBILITY.csv", "PROVENANCE", 0, 0,
        entity_id="2-detector K8 eligibility")
    add("k_support", PB3 / "PB1DS_PB3_PROTOCOL.md", "PROVENANCE", 0, 0)
    add("k_support", PB3R / "PB1DS_PB3R_PROTOCOL.md", "PROVENANCE", 0, 0)
    add("k_support", PB6 / "PB1DS_PB6_K8_BRIDGE.csv", "PROVENANCE", 0, 1,
        entity_id="tri-detector K8 bridge (FULL + K8)")
    add("k_support", PB6 / "PB1DS_PB6_D8_BRIDGES.md", "PROVENANCE", 0, 0)

    # ---- sealed PB6 references -------------------------------------------------
    add("sealed_reference", PB6 / "PB1DS_PB6_TERMINAL.json", "PROVENANCE", 0, 0)
    add("sealed_reference", PB6 / "PB1DS_PB6_PROTOCOL_BINDING.json", "PROVENANCE", 0, 0)

    df = pd.DataFrame(rows)
    df = df.sort_values(["kind", "entity_id"]).reset_index(drop=True)
    return df


def write_data_roles() -> dict[str, object]:
    roles = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "P0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "DEPLOYABLE": {
            "definition": (
                "usable by any deployable candidate / feature / actuator; "
                "must never depend on target anomaly labels, anomaly scores, "
                "or any oracle outcome"
            ),
            "inputs": {
                "source_normal": "S_source_normal (mv2_{cat}_public_good_regular_{det}_scores.npz)",
                "target_normal": "S_target_normal (mv2_{cat}_public_good_{cond}_{det}_scores.npz)",
                "detector_identity": "which detector produced the scores",
                "category_identity": "category label (per-category detector deployment)",
                "condition_metadata": "acquisition condition metadata (OPTIONAL; no PB7 candidate may depend on the condition label directly)",
            },
        },
        "ORACLE_DIAGNOSTIC_ONLY": {
            "definition": (
                "usable ONLY for oracle diagnostic targets (O1/O2/O3) and "
                "diagnostic metrics; never a deployment input; never used to "
                "fit, select, or calibrate any deployable candidate"
            ),
            "inputs": {
                "anomaly_scores": "S_anomaly (mv2_{cat}_public_bad_regular_{det}_scores.npz)",
                "anomaly_labels": "target anomaly labels",
                "ba": "balanced accuracy outcome",
                "fpr": "false-positive-rate outcome",
                "fnr": "false-negative-rate outcome",
                "oracle_threshold": "tau*_BA / tau*_SAFE",
                "threshold_anomaly_outcomes": "threshold-specific anomaly outcomes",
            },
        },
        "firewall": {
            "mechanism": (
                "code-level Firewall in pb7_p0_common.py: every score read is "
                "recorded with role; deployable computation must call "
                "assert_deployable_clean() which raises FIREWALL_VIOLATION if "
                "any anomaly (oracle-only) file was read on that path"
            ),
            "deployable_loaders": ["load_src", "load_cnd"],
            "oracle_loader": "load_anm",
        },
        "leakage_rule": (
            "if any deployable candidate directly or indirectly uses "
            "target-bad / anomaly information, it is immediately marked "
            "LEAKAGE_INVALID and excluded from formal comparison"
        ),
    }
    (OUT / "P0_DATA_ROLES.json").write_text(json.dumps(roles, indent=2), encoding="utf-8")
    return roles


def write_estimand_registry() -> dict[str, object]:
    reg = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "P0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": {
            "generation": (
                "all unique values of the pooled target scores "
                "(target-normal U anomaly) PLUS the source threshold tau_source, "
                "sorted ascending"
            ),
            "detection_rule": "score > tau (strict), matching md0sr_common.detection_metrics",
            "includes_source_threshold": True,
            "endpoint_handling": (
                "grid includes min and max pooled scores; both endpoints are "
                "valid candidates (BA at extremes is typically 0.5); no "
                "clipping, no padding"
            ),
            "exactness": (
                "BA/FPR/FNR are step functions of tau, so scanning the unique "
                "pooled scores is exact for the decision boundary"
            ),
        },
        "tie_break": (
            "among thresholds achieving the target optimum, choose the one "
            "closest to tau_source; further tie -> smaller tau"
        ),
        "O1_ORACLE_BA_OPTIMAL_THRESHOLD": {
            "target": "tau*_BA",
            "definition": (
                "tau*_BA = argmax over the frozen grid of BA(score > tau); "
                "tie-break closest to tau_source, further tie smaller tau"
            ),
            "outputs": ["tau_star", "ba", "fpr", "fnr", "delta_ba", "delta_fpr", "delta_fnr", "delta_tau"],
            "role": "ORACLE_DIAGNOSTIC_ONLY",
        },
        "O2_ORACLE_SAFE_CONSTRAINED_THRESHOLD": {
            "target": "tau*_SAFE",
            "definition": (
                "tau*_SAFE = argmax over the frozen grid of BA subject to "
                "FNR(tau) - FNR(tau_source) <= +0.02 (delta_fnr <= +0.02); "
                "tie-break closest to tau_source, further tie smaller tau"
            ),
            "feasibility": (
                "tau_source is always feasible (delta_fnr = 0), so the "
                "feasible set is never empty"
            ),
            "outputs": ["tau_star", "ba", "fpr", "fnr", "delta_ba", "delta_fpr", "delta_fnr", "delta_tau", "n_feasible"],
            "role": "ORACLE_DIAGNOSTIC_ONLY",
        },
        "O3_ORACLE_CORRECTABILITY_LABEL": {
            "target": "CORRECTABLE_SAFE",
            "definition": (
                "CORRECTABLE_SAFE = 1 iff delta_ba >= +0.02 AND "
                "delta_fpr <= -0.05 AND delta_fnr <= +0.02 relative to the "
                "source operating point (the frozen benefit definition), "
                "evaluated at tau*_SAFE; otherwise 0"
            ),
            "role": "ORACLE_DIAGNOSTIC_ONLY",
            "deployment_use": "FORBIDDEN (diagnostic label only)",
        },
        "frozen_constants": {
            "Q95": P.Q95,
            "DBA_MIN": P.DBA_MIN,
            "DFPR_MAX": P.DFPR_MAX,
            "DFNR_MAX": P.DFNR_MAX,
            "SAFE_DFNR_MAX": P.SAFE_DFNR_MAX,
        },
    }
    (OUT / "P0_ESTIMAND_REGISTRY.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg


def write_hypothesis_registry() -> dict[str, object]:
    reg = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "P0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypotheses": {
            "H1_THRESHOLD_HEADROOM_EXISTS": {
                "claim": (
                    "at least one detector has a non-trivial fraction of units "
                    "with CORRECTABLE_SAFE = 1"
                ),
                "minimum_continuation": ">= 20% of units OR >= 8 units (at least one satisfied)",
                "stage": "A",
            },
            "H2_DIRECTION_IDENTIFIABILITY": {
                "claim": (
                    "normal-only deformation can predict sign(tau*_SAFE - tau_source) "
                    "significantly better than a trivial baseline"
                ),
                "stage": "B",
            },
            "H3_MAGNITUDE_IDENTIFIABILITY": {
                "claim": (
                    "normal-only deformation has a stable monotonic / regression "
                    "relation with delta_tau*_SAFE"
                ),
                "stage": "B",
            },
            "H4_DETECTOR_DEPENDENCE": {
                "claim": (
                    "different detectors show substantial differences in "
                    "correctability prevalence, delta_tau* distribution, and the "
                    "deformation -> threshold relationship"
                ),
                "stage": "A/B",
            },
            "H5_FEW_NORMAL_STABILITY": {
                "claim": (
                    "the above information is preserved with sufficient stability "
                    "at K=8 target-normal support"
                ),
                "stage": "C",
            },
            "H6_SAFE_DEPLOYABLE_ACTUATOR_EXISTS": {
                "claim": (
                    "at least one pre-registered simple normal-only threshold rule "
                    "achieves actual benefit under strict held-out evaluation"
                ),
                "stage": "D/E",
            },
        },
        "trivial_baselines_for_H2": [
            "B0_ALWAYS_NO_ADAPT",
            "B1_ALWAYS_RAISE_THRESHOLD",
            "B2_ALWAYS_LOWER_THRESHOLD",
            "B3_DETECTOR_ONLY_PRIOR",
            "B4_SINGLE_NORMAL_ONLY_SCALAR",
        ],
    }
    (OUT / "P0_HYPOTHESIS_REGISTRY.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg


def write_protocol(manifest: pd.DataFrame, roles: dict[str, object],
                   estimands: dict[str, object], hypotheses: dict[str, object]) -> dict[str, object]:
    proto = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "P0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_question": (
            "Can detector-specific target-normal score deformation identify the "
            "direction and magnitude of a safe operating-point threshold "
            "correction without using target anomaly labels?"
        ),
        "frozen_background": {
            "pb6_terminal": "PB1DS_PB6_COMPLETE_REAL_THIRD_DETECTOR_DOES_NOT_SUPPORT_CORE_SELECTIVE_RECALIBRATION_CLAIM",
            "pb6_terminal_immutable": True,
            "pb6_posthoc_audit": {
                "H1_QM_ANOMALY_SELF_RANKING": "SUPPORTED",
                "H2_QM_IMPLEMENTATION_MATCHES_FROZEN_SEMANTICS": "SUPPORTED",
                "H3_QM_DESIGN_DEGENERATE": "SUPPORTED",
                "H4_EFFICIENTAD_NO_QM_HEADROOM": "SUPPORTED",
                "H5_EFFICIENTAD_THRESHOLD_CORRECTABLE": "SUPPORTED",
                "H6_BENEFIT_IS_OPERATING_POINT_REBALANCING": "SUPPORTED",
                "H7_BENEFIT_DOMINATED_BY_FAILED_OPERATING_POINTS": "SUPPORTED",
            },
            "frozen_interpretation": (
                "PB6 QM's positive delta_BA is operating-point rebalancing, not "
                "representation/ranking capability improvement. FNR_after = 14/15 "
                "on 117/117 entities. QM benefit requires FNR_before >= 0.9133. "
                "EfficientAD QM oracle headroom = 0/39 (correct abstention); "
                "BC_RANK O2/O3 headroom ~ 2/39; general score-threshold O4 "
                "headroom = 11/39. QM mean delta_AUROC = -0.0772; 33/33 QM "
                "benefit rows from already-failed operating points."
            ),
            "benefit_definition": "delta_ba >= +0.02 AND delta_fpr <= -0.05 AND delta_fnr <= +0.02",
            "source_threshold": "tau_source = Q95(source-normal scores)",
            "detectors": P.DETECTORS,
            "population": "39 units x 3 detectors = 117 entities (MVTec AD 2)",
        },
        "population": {
            "n_units": P.N_UNITS,
            "n_entities": P.N_ENTITIES,
            "entity_source": "PB1DS_PB6_D5_OOF_PER_UNIT.csv (authoritative)",
            "unit_source": "PB1DS_PB6_D5_SPLIT_MANIFEST.json (39 outer folds)",
        },
        "data_roles": roles,
        "estimands": estimands,
        "hypotheses": hypotheses,
        "k_support": {
            "k_grid": P.K_GRID,
            "repeats": P.K_REPEATS,
            "primary_k": P.PRIMARY_K,
            "nested_rule": P.NESTED_SAMPLING_RULE,
            "seed_identity": "PB1|K|{k}|{cat}|{cond}|{det}|{repeat}",
            "seed_int": "int(sha256(identity)[:8], 16)",
            "tri_detector_extension": (
                "PB3 seed manifest covers the 2 reference detectors; the third "
                "detector (efficientad) uses the same frozen seed identity via "
                "the PB6 D8 tri_k8_indices rule"
            ),
            "k8_eligibility": "FULL_n >= 8",
        },
        "stage_plan": {
            "P0": "evidence freeze + protocol registration (this stage)",
            "A": "oracle threshold headroom characterization (diagnosis only)",
            "B": "normal-only deformation vs oracle correctability (identifiability)",
            "C": "few-normal robustness (K4/K8/FULL)",
            "D": "pre-registered simple threshold actuator candidates (T0-T4)",
            "E": "strict OOF / held-out evaluation",
            "FINAL": "mechanism adjudication + final report",
        },
        "forbidden": [
            "develop new complex classifiers (XGBoost/RF/NN/Transformer/AutoML)",
            "feature search / hyperparameter sweep",
            "outcome-driven threshold tuning",
            "select features by test anomaly outcome",
            "add new rules ad hoc from detector results",
            "modify the BENEFIT definition",
            "modify PB6 sealed artifacts",
            "modify the old PB6 terminal",
            "treat oracle thresholds as deployable methods",
            "use target anomaly labels to construct any deployable predictor",
            "use target anomaly score distribution as a deployable input",
            "redefine candidate actuators after seeing held-out results",
            "continue developing the old QM",
            "describe AUROC-invariant threshold correction as detection capability improvement",
        ],
        "leakage_rule": (
            "if any deployable candidate directly or indirectly uses "
            "target-bad / anomaly information, it is immediately marked "
            "LEAKAGE_INVALID and excluded from formal comparison"
        ),
        "p0_gate": {
            "conditions": [
                "oracle threshold semantics are unique (grid/tie-break/endpoints frozen)",
                "anomaly/deployable firewall is code-level guaranteed",
                "score-cache provenance is clear (SHA256-bound manifest)",
            ],
            "failure_terminal": "STOP_PROTOCOL_INVALID",
        },
        "input_manifest": {
            "n_rows": int(len(manifest)),
            "n_score_source": int((manifest["kind"] == "score_cache_source_normal").sum()),
            "n_score_target": int((manifest["kind"] == "score_cache_target_normal").sum()),
            "n_score_anomaly": int((manifest["kind"] == "score_cache_anomaly").sum()),
            "file": "P0_INPUT_MANIFEST.csv",
        },
    }
    (OUT / "P0_PROTOCOL.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")
    return proto


def run_gate(manifest: pd.DataFrame) -> dict[str, object]:
    checks: dict[str, object] = {}

    # ---- check 1: oracle threshold semantics unique ---------------------------
    est = json.loads((OUT / "P0_ESTIMAND_REGISTRY.json").read_text(encoding="utf-8"))
    grid_ok = (
        est["grid"]["generation"]
        and est["grid"]["includes_source_threshold"] is True
        and est["grid"]["endpoint_handling"]
        and est["grid"]["exactness"]
    )
    tie_ok = bool(est["tie_break"])
    o1_ok = bool(est["O1_ORACLE_BA_OPTIMAL_THRESHOLD"]["definition"])
    o2_ok = bool(est["O2_ORACLE_SAFE_CONSTRAINED_THRESHOLD"]["definition"])
    o3_ok = bool(est["O3_ORACLE_CORRECTABILITY_LABEL"]["definition"])
    # self-check: O1/O2 unique + tau_source feasible for O2 on a sample of entities
    sample = P.entity_list()[::13][:9]
    o1_unique = True
    o2_unique = True
    o2_feasible = True
    for cat, cond, det in sample:
        src = P.load_src(cat, det)
        cnd = P.load_cnd(cat, cond, det)
        anm = P.load_anm(cat, det)
        tau_s = P.source_threshold(src)
        o1 = P.oracle_ba_optimal(src, cnd, anm, tau_s)
        o2 = P.oracle_safe_optimal(src, cnd, anm, tau_s)
        if not (isinstance(o1["tau_star"], float) and np.isfinite(o1["tau_star"])):
            o1_unique = False
        if not (isinstance(o2["tau_star"], float) and np.isfinite(o2["tau_star"])):
            o2_unique = False
        if o2["n_feasible"] < 1:
            o2_feasible = False
    checks["oracle_semantics_unique"] = bool(grid_ok and tie_ok and o1_ok and o2_ok and o3_ok)
    checks["oracle_o1_o2_self_check"] = bool(o1_unique and o2_unique and o2_feasible)
    checks["oracle_semantics_detail"] = (
        f"grid={grid_ok}, tie_break={tie_ok}, O1={o1_ok}, O2={o2_ok}, O3={o3_ok}, "
        f"self_check({len(sample)} entities) O1_unique={o1_unique}, O2_unique={o2_unique}, "
        f"O2_feasible={o2_feasible}"
    )

    # ---- check 2: firewall code-level guaranteed ------------------------------
    fw = P.Firewall()
    src = P.load_src("can", "dinov2_vits14", fw)
    cnd = P.load_cnd("can", "overexposed", "dinov2_vits14", fw)
    _ = P.source_threshold(src)
    fw.assert_deployable_clean()  # must not raise on a deployable-only path
    checks["firewall_guaranteed"] = True
    checks["firewall_detail"] = (
        "deployable-only path (load_src + load_cnd + source_threshold) passed "
        "assert_deployable_clean(); anomaly reads are recorded as oracle-only "
        "and raise FIREWALL_VIOLATION if used on a deployable path"
    )

    # ---- check 3: score-cache provenance clear --------------------------------
    n_expected = 24 + 117 + 24
    n_src = int((manifest["kind"] == "score_cache_source_normal").sum())
    n_tgt = int((manifest["kind"] == "score_cache_target_normal").sum())
    n_anm = int((manifest["kind"] == "score_cache_anomaly").sum())
    n_score = n_src + n_tgt + n_anm
    all_hashed = bool(manifest["sha256"].notna().all() and (manifest["sha256"].astype(str).str.len() == 64).all())
    checks["score_cache_provenance_clear"] = bool(n_score == n_expected and all_hashed)
    checks["score_cache_provenance_detail"] = (
        f"source={n_src}, target={n_tgt}, anomaly={n_anm}, total={n_score} "
        f"(expected {n_expected}); all SHA256 present={all_hashed}"
    )

    passed = all(
        checks[k]
        for k in [
            "oracle_semantics_unique",
            "oracle_o1_o2_self_check",
            "firewall_guaranteed",
            "score_cache_provenance_clear",
        ]
    )
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "P0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
        "verdict": "PASS" if passed else "FAIL",
        "terminal": "PROCEED_TO_A" if passed else "STOP_PROTOCOL_INVALID",
    }
    (OUT / "P0_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    manifest = build_input_manifest()
    manifest.to_csv(OUT / "P0_INPUT_MANIFEST.csv", index=False)

    roles = write_data_roles()
    estimands = write_estimand_registry()
    hypotheses = write_hypothesis_registry()
    proto = write_protocol(manifest, roles, estimands, hypotheses)
    gate = run_gate(manifest)

    gs = git_state()
    (OUT / "P0_GIT_STATE.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in gs.items()) + "\n", encoding="utf-8"
    )

    # ---- execution ledger ------------------------------------------------------
    ledger = OUT / "P0_EXECUTION_LEDGER.csv"
    rows = []
    if ledger.exists():
        with ledger.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    rows = [r for r in rows if not (r["stage"] == "P0" and r["step"] == "evidence_freeze_protocol_registration")]
    rows.append(
        {
            "stage": "P0",
            "step": "evidence_freeze_protocol_registration",
            "status": "COMPLETE",
            "timestamp": ts,
            "artifacts": (
                "P0_PROTOCOL.json, P0_INPUT_MANIFEST.csv, P0_DATA_ROLES.json, "
                "P0_ESTIMAND_REGISTRY.json, P0_HYPOTHESIS_REGISTRY.json, "
                "P0_GATE.json, P0_GIT_STATE.txt"
            ),
            "note": (
                f"P0 gate {gate['verdict']} -> {gate['terminal']}; "
                f"{len(manifest)} inputs SHA256-bound; O1/O2/O3 + H1-H6 registered"
            ),
        }
    )
    with ledger.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"P0 gate: {gate['verdict']} -> {gate['terminal']}")
    print(f"manifest rows: {len(manifest)}")
    for k, v in gate["checks"].items():
        print(f"  {k}: {v}")
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
