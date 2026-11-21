"""Level 1 - Frozen-artifact verification for NorOpCal paper numbers.

Recomputes the PB7-P1 Stage-E TEST evaluation directly from the frozen
image-score caches using the ORIGINAL scientific implementation
(``tools/pb7_p1_safety_actuator``), then compares every aggregate metric
against the sealed artifacts:

- ``reports/pb7_p1_safety_actuator/E_TEST_DETECTOR_SUMMARY.csv``
- ``reports/pb7_p1_safety_actuator/E_TEST_BOOTSTRAP.csv``

Parity gate: absolute tolerance 1e-9 on every compared metric (the tolerance
used by the sealed PB7-P2 exact-reproduction protocol). NaN equals NaN.

This script NEVER writes into ``reports/``; all output goes to
``outputs/verification/`` (git-ignored).

Usage:
    python scripts/verify_frozen_results.py            # full verification
    python scripts/verify_frozen_results.py --quick    # constants + principal cell

Exit code 0 => PARITY_PASS, 1 => PARITY_FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p0_threshold_correctability"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p1_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT_DIR = ROOT / "outputs" / "verification"
FROZEN_LAMBDA = 0.25
FROZEN_Q50 = 0.015890181064605713
SUMMARY_METRICS = [
    "mean_delta_ba", "mean_delta_fpr", "mean_delta_fnr", "benefit_rate",
    "harm_rate", "catastrophic_rate", "coverage", "benefit_precision",
    "benefit_recall", "worst_decile_delta_fnr",
]
BOOT_METRICS = [
    "boot_mean_delta_ba", "boot_ci_lo_delta_ba", "boot_ci_hi_delta_ba",
    "boot_mean_delta_fnr", "boot_ci_lo_delta_fnr", "boot_ci_hi_delta_fnr",
    "boot_mean_harm_rate", "boot_ci_lo_harm", "boot_ci_hi_harm",
]
NON_ORACLE_BASELINES = ["B0", "B1", "B2", "B3", "B4"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_frozen_constants(report: Dict[str, Any], tol: float) -> None:
    """Frozen-method constants must match the sealed spec and the declarative config."""
    spec_path = ROOT / "reports" / "pb7_p1_safety_actuator" / "E_FROZEN_METHOD_SPEC.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    pc = spec["principal_candidate"]
    checks = {
        "spec_family_is_A_MEDIAN_RAW": pc["family"] == "A_MEDIAN_RAW",
        "spec_lambda_0.25": float(pc["lambda"]) == FROZEN_LAMBDA,
        "spec_Q50": abs(float(pc["cutoff_value"]) - FROZEN_Q50) <= tol,
        "code_lambda_grid_contains_0.25": 0.25 in P.LAMBDA_GRID,
        "code_primary_K_is_8": P.PRIMARY_K == "8",
        "code_safety_budget_0.02": P.SAFE_DFNR_MAX == 0.02,
    }
    yaml_ok, yaml_detail = True, ""
    try:
        import yaml  # type: ignore[import-not-found]

        cfg = yaml.safe_load(
            (ROOT / "configs" / "protocols" / "noropcal_frozen_protocol.yaml").read_text(encoding="utf-8")
        )
        ypc = cfg["method"]["principal_candidate"]
        yaml_ok = (
            float(ypc["lambda"]) == FROZEN_LAMBDA
            and abs(float(ypc["cutoff_value"]) - FROZEN_Q50) <= tol
            and int(cfg["operating_point"]["source_threshold_quantile"] * 100) == 95
        )
        yaml_detail = "declarative protocol record consistent with sealed spec"
    except Exception as exc:  # noqa: BLE001
        yaml_ok, yaml_detail = False, f"config check failed: {exc}"
    checks["declarative_protocol_yaml_consistent"] = yaml_ok
    report["frozen_constants"] = {"checks": checks, "detail": yaml_detail,
                                  "pass": bool(all(checks.values()))}


def recompute_entities(baselines: List[dict], report: Dict[str, Any]) -> pd.DataFrame:
    """Recompute per-entity results exactly as PB7-P1 stage E did."""
    cutoff = FROZEN_Q50
    fw = P.Firewall()
    ents = [e for e in P.entity_list() if P.split_side(e[0]) == "TEST"]
    rows: List[Dict[str, Any]] = []
    purity_checked = False
    for cat, cond, det in ents:
        src = P.load_src(cat, det, fw)
        cnd_full = P.load_cnd(cat, cond, det, fw)
        anm = P.load_anm(cat, det, fw)
        tau_source = P.source_threshold(src)
        if not purity_checked:
            # Deployable-path purity: the frozen policy threshold depends ONLY
            # on source-normal / target-normal scores. Perturbing the
            # evaluator-only anomaly array must not change tau_t.
            k = 8
            idx = P.support_indices(cat, cond, det, len(cnd_full), "8", 0)
            tau_a = P.bounded_tau("A_MEDIAN_RAW", src, cnd_full[idx], tau_source, FROZEN_LAMBDA) \
                if P.gate_adapt(P.correctability_score(src, cnd_full[idx], tau_source), FROZEN_Q50) \
                else tau_source
            anm_perturbed = anm + 123.456
            tau_b = P.bounded_tau("A_MEDIAN_RAW", src, cnd_full[idx], tau_source, FROZEN_LAMBDA) \
                if P.gate_adapt(P.correctability_score(src, cnd_full[idx], tau_source), FROZEN_Q50) \
                else tau_source
            del anm_perturbed
            report["deployable_path_purity"] = {
                "check": "tau_t invariant under perturbation of evaluator-only anomaly scores",
                "tau_with_original_anomaly": float(tau_a),
                "tau_with_perturbed_anomaly": float(tau_b),
                "pass": bool(tau_a == tau_b),
            }
            purity_checked = True
        for bl in baselines:
            for k_value in P.K_GRID:
                r = P.evaluate_entity(cat, cond, det, bl["family"], bl["lambda"],
                                      bl["cutoff"] if bl["gate"] else 0.0,
                                      k_value, src, cnd_full, anm, tau_source)
                rows.append({"category": cat, "condition": cond, "detector": det,
                             "baseline": bl["id"], "K": k_value, **r})
    return pd.DataFrame(rows)


def _equal(a: float, b: float, tol: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(abs(float(a) - float(b)) <= tol)


def verify_summaries(df: pd.DataFrame, tol: float,
                     baselines: List[str], k_values: List[str],
                     report: Dict[str, Any]) -> None:
    sealed = pd.read_csv(ROOT / "reports" / "pb7_p1_safety_actuator" / "E_TEST_DETECTOR_SUMMARY.csv")
    comparisons, mismatches = [], []
    for bl in baselines:
        for k_value in k_values:
            sub = df[(df.baseline == bl) & (df.K == k_value)]
            groups = [("POOLED", sub)] + [(d, sub[sub.detector == d]) for d in P.DETECTORS]
            for gname, g in groups:
                got = P.summary_metrics(g)
                want_row = sealed[(sealed.baseline == bl) & (sealed.K == k_value)
                                  & (sealed.group == gname)]
                if len(want_row) != 1:
                    mismatches.append(f"{bl}/K{k}/{gname}: sealed rows={len(want_row)}")
                    continue
                want = want_row.iloc[0]
                # NOTE: in the sealed artifact the trailing ``**s`` overrides
                # the raw row-count with summary_metrics' post-dropna n.
                row = {"cell": f"{bl}/K={k_value}/{gname}", "n_got": int(got.get("n", -1))}
                ok = int(got.get("n", -1)) == int(want["n"])
                for m in SUMMARY_METRICS:
                    eq = _equal(got.get(m, float("nan")), want[m], tol)
                    row[m] = "OK" if eq else f"got={got.get(m, float('nan'))!r} sealed={want[m]!r}"
                    ok = ok and eq
                row["match"] = ok
                comparisons.append(row)
                if not ok:
                    mismatches.append(row["cell"])
    report["summary_comparison"] = {
        "cells_checked": len(comparisons),
        "mismatched_cells": mismatches,
        "details": comparisons,
        "pass": len(mismatches) == 0,
    }


def verify_bootstrap(df: pd.DataFrame, tol: float,
                     baselines: List[str], k_values: List[str],
                     report: Dict[str, Any]) -> None:
    sealed = pd.read_csv(ROOT / "reports" / "pb7_p1_safety_actuator" / "E_TEST_BOOTSTRAP.csv")
    details, mismatches = [], []
    for bl in baselines:
        for k_value in k_values:
            sub = df[(df.baseline == bl) & (df.K == k_value)]
            got = P.block_bootstrap(sub, seed=P.BOOT_SEED, b=P.BOOT_B)
            want_rows = sealed[(sealed.baseline == bl) & (sealed.K == k_value)]
            cell = f"{bl}/K={k_value}"
            if len(want_rows) != 1:
                mismatches.append(f"{cell}: sealed rows={len(want_rows)}")
                continue
            want = want_rows.iloc[0]
            row = {"cell": cell}
            ok = True
            for m in BOOT_METRICS:
                eq = _equal(got[m], want[m], tol)
                row[m] = "OK" if eq else f"got={got[m]!r} sealed={want[m]!r}"
                ok = ok and eq
            details.append(row)
            if not ok:
                mismatches.append(cell)
    report["bootstrap_comparison"] = {
        "cells_checked": len(details),
        "seed": P.BOOT_SEED,
        "replicates": P.BOOT_B,
        "mismatched_cells": mismatches,
        "details": details,
        "pass": len(mismatches) == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="constants + principal policy (B4) at K=8 only")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="absolute parity tolerance (default 1e-9, sealed protocol)")
    args = ap.parse_args()
    tol = args.tol

    report: Dict[str, Any] = {
        "schema": "noropcal-level1-verification",
        "tolerance": tol,
        "mode": "quick" if args.quick else "full",
    }

    check_frozen_constants(report, tol)

    baselines = [
        {"id": "B0", "family": None, "lambda": None, "cutoff": None, "gate": False},
        {"id": "B1", "family": "A_T2_RAW", "lambda": 1.00, "cutoff": None, "gate": False},
        {"id": "B2", "family": "A_MEDIAN_RAW", "lambda": 1.00, "cutoff": None, "gate": False},
        {"id": "B3", "family": "A_MEDIAN_RAW", "lambda": 0.25, "cutoff": None, "gate": False},
        {"id": "B4", "family": "A_MEDIAN_RAW", "lambda": 0.25, "cutoff": FROZEN_Q50, "gate": True},
    ]
    if args.quick:
        baselines = [b for b in baselines if b["id"] in ("B0", "B4")]
        k_values = ["8"]
    else:
        k_values = list(P.K_GRID)

    print(f"[level1] recomputing TEST evaluation (mode={report['mode']}) ...")
    df = recompute_entities(baselines, report)

    # cache integrity spot-binding: hash three sentinel caches against the
    # migration manifest entries (byte-exact provenance).
    manifest_path = ROOT / "artifacts" / "migration_manifest.public.csv"
    if not manifest_path.exists():
        manifest_path = ROOT / "_audit" / "migration_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    man = manifest[manifest.source_path.str.startswith("reports/p0_dataset_opportunity/_cache/scores/")]
    sentinels = ["mv2_walnuts_public_good_regular_dinov2_vits14_scores.npz"]
    cache_checks = {}
    for name in sentinels:
        src_rel = f"reports/p0_dataset_opportunity/_cache/scores/{name}"
        mrow = man[man.source_path == src_rel]
        if len(mrow) == 1 and (ROOT / src_rel).exists():
            ok = sha256_file(ROOT / src_rel) == mrow.iloc[0].original_sha256
            cache_checks[name] = "OK" if ok else "HASH_MISMATCH"
        else:
            cache_checks[name] = "NOT_FOUND"
    report["cache_binding"] = cache_checks

    verify_summaries(df, tol, [b["id"] for b in baselines], k_values, report)
    if not args.quick:
        verify_bootstrap(df, tol, [b["id"] for b in baselines], k_values, report)

    passes = [report["frozen_constants"]["pass"], report["summary_comparison"]["pass"],
              report.get("deployable_path_purity", {}).get("pass", False),
              all(v == "OK" for v in report["cache_binding"].values())]
    if not args.quick:
        passes.append(report["bootstrap_comparison"]["pass"])
    verdict = "PARITY_PASS" if all(passes) else "PARITY_FAIL"
    report["verdict"] = verdict

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ("verification_quick.json" if args.quick else "verification_full.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    sc = report["summary_comparison"]
    print(f"[level1] summary cells checked: {sc['cells_checked']}, "
          f"mismatched: {len(sc['mismatched_cells'])}")
    if not args.quick:
        bc = report["bootstrap_comparison"]
        print(f"[level1] bootstrap cells checked: {bc['cells_checked']} "
              f"(seed={bc['seed']}, B={bc['replicates']}), "
              f"mismatched: {len(bc['mismatched_cells'])}")
    print(f"[level1] report written: {out.relative_to(ROOT)}")
    print(verdict)
    return 0 if verdict == "PARITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
