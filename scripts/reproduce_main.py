"""Level 2 - Full-reproduction orchestrator for NorOpCal.

Two modes:

* ``--mode dry-run`` (DEFAULT) - validates the full reproduction environment
  and prints the exact frozen stage commands, without executing anything.

* ``--mode execute`` - runs the frozen stage chain inside an ISOLATED SCRATCH
  COPY of the repository so sealed artifacts in the checkout are never
  overwritten. Requires ``--scratch-dir``. The chain is the sealed PB7-P0 ->
  PB7-P1 -> PB7-P2 -> PB8 order; each stage script verifies its inputs'
  hashes itself.

Honesty notes:
- Level 2 scoring-from-raw-images additionally requires the MVTec AD 2 dataset
  and GPU training (EfficientAD 70k steps). This orchestrator does NOT hide
  that: pass ``--with-scoring`` only when both are available.
- G2 validated the dry-run path end-to-end; the execute path was implemented
  but NOT executed during G2 (END_TO_END_FULL_RUN_NOT_EXECUTED_IN_G2).

Example:
    python scripts/reproduce_main.py --dry-run
    python scripts/reproduce_main.py --mode execute \
        --scratch-dir /path/to/empty/scratch_dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGE_CHAIN = [
    ("PB7-P0 protocol freeze", "tools/pb7_p0_threshold_correctability/p0_freeze.py"),
    ("PB7-P0 A oracle headroom", "tools/pb7_p0_threshold_correctability/a_oracle_headroom.py"),
    ("PB7-P0 B identifiability", "tools/pb7_p0_threshold_correctability/b_identifiability.py"),
    ("PB7-P0 C few-normal", "tools/pb7_p0_threshold_correctability/c_few_normal.py"),
    ("PB7-P0 E evaluation", "tools/pb7_p0_threshold_correctability/e_evaluate.py"),
    ("PB7-P1 P0 freeze", "tools/pb7_p1_safety_actuator/p0_freeze.py"),
    ("PB7-P1 A actuator family", "tools/pb7_p1_safety_actuator/a_freeze.py"),
    ("PB7-P1 B bounding", "tools/pb7_p1_safety_actuator/b_freeze.py"),
    ("PB7-P1 C gate definition", "tools/pb7_p1_safety_actuator/c_freeze.py"),
    ("PB7-P1 D DEV selection", "tools/pb7_p1_safety_actuator/d_dev_select.py"),
    ("PB7-P1 E TEST evaluation", "tools/pb7_p1_safety_actuator/e_test_eval.py"),
    ("PB7-P1 F mechanism", "tools/pb7_p1_safety_actuator/f_mechanism.py"),
    ("PB7-P2 P0 freeze", "tools/pb7_p2_publication_validation/p0_freeze.py"),
    ("PB7-P2 A exact reproduction", "tools/pb7_p2_publication_validation/a_reproduction.py"),
    ("PB7-P2 B LOCO", "tools/pb7_p2_publication_validation/b_loco.py"),
    ("PB7-P2 C K-support", "tools/pb7_p2_publication_validation/c_k_support.py"),
    ("PB7-P2 D bootstrap", "tools/pb7_p2_publication_validation/d_bootstrap.py"),
    ("PB7-P2 E residual harm", "tools/pb7_p2_publication_validation/e_residual_harm.py"),
    ("PB7-P2 F detector specificity", "tools/pb7_p2_publication_validation/f_detector_specificity.py"),
    ("PB8 benchmark", "tools/pb8_publication_benchmark/b_benchmark.py"),
    ("PB8 ablations", "tools/pb8_publication_benchmark/c_ablation.py"),
    ("PB8 statistics", "tools/pb8_publication_benchmark/f_statistics.py"),
]
SCORE_CACHE_DIR = ROOT / "reports" / "p0_dataset_opportunity" / "_cache" / "scores"
EXPECTED_CACHES = 165


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_environment(with_scoring: bool) -> dict:
    checks: dict = {}
    try:
        import numpy  # noqa: F401

        checks["numpy"] = "OK"
    except ImportError:
        checks["numpy"] = "MISSING"
    try:
        import pandas  # noqa: F401

        checks["pandas"] = "OK"
    except ImportError:
        checks["pandas"] = "MISSING"
    n_caches = len(list(SCORE_CACHE_DIR.glob("mv2_*_scores.npz"))) if SCORE_CACHE_DIR.is_dir() else 0
    checks["frozen_score_caches"] = f"{n_caches}/{EXPECTED_CACHES}"
    checks["score_caches_complete"] = n_caches >= EXPECTED_CACHES
    missing_scripts = [rel for _, rel in STAGE_CHAIN if not (ROOT / rel).exists()]
    checks["stage_scripts_present"] = (len(missing_scripts) == 0)
    if missing_scripts:
        checks["missing_stage_scripts"] = missing_scripts[:5]
    spec = ROOT / "reports" / "pb7_p1_safety_actuator" / "E_FROZEN_METHOD_SPEC.json"
    checks["sealed_method_spec_present"] = spec.exists()

    if with_scoring:
        for var in ("NOROPCAL_MVTEC_AD2_ROOT", "NOROPCAL_PB6_WORK_ROOT"):
            checks[f"env:{var}"] = "SET" if os.environ.get(var) else "MISSING (required for scoring)"
        try:
            import torch  # noqa: F401

            checks["torch"] = "OK"
        except ImportError:
            checks["torch"] = "MISSING (required for scoring)"
    return checks


def print_chain() -> None:
    print("[level2] frozen stage chain:")
    for i, (name, rel) in enumerate(STAGE_CHAIN, 1):
        print(f"  {i:2d}. {name:34s} python {rel}")


def run_execute(scratch: Path, with_scoring: bool) -> int:
    """Copy the repo payload into scratch and run the chain there."""
    if scratch.exists() and any(scratch.iterdir()):
        print(f"[level2] refusing non-empty scratch dir: {scratch}")
        return 2
    scratch.mkdir(parents=True)
    print(f"[level2] copying repository payload into {scratch} ...")
    for item in ("tools", "src", "configs", "reports", "_audit"):
        src = ROOT / item
        if src.exists():
            shutil.copytree(src, scratch / item)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(scratch / "src"))
    for i, (name, rel) in enumerate(STAGE_CHAIN, 1):
        print(f"[level2] ({i}/{len(STAGE_CHAIN)}) {name}")
        proc = subprocess.run(
            [sys.executable, str(scratch / rel)],
            cwd=scratch, capture_output=True, text=True, env=env,
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            print(f"[level2] stage FAILED: {name}")
            result = {"status": "CHAIN_FAILED", "failed_stage": name}
            (scratch / "reproduce_result.json").write_text(json.dumps(result, indent=1))
            return 1
    print("[level2] chain complete; running in-scratch Level-1 verification ...")
    proc = subprocess.run(
        [sys.executable, str(scratch / "scripts" / "verify_frozen_results.py")],
        cwd=scratch, capture_output=True, text=True, env=env,
    )
    verdict = "PARITY_PASS" if proc.returncode == 0 else "PARITY_FAIL"
    result = {"status": f"EXECUTED_CHAIN_{verdict}",
              "note": "executed inside isolated scratch copy"}
    (scratch / "reproduce_result.json").write_text(json.dumps(result, indent=1))
    print(f"[level2] {verdict} (scratch outputs under {scratch})")
    return 0 if verdict == "PARITY_PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/protocols/noropcal_frozen_protocol.yaml",
                    help="frozen protocol record (declarative; code remains authoritative)")
    ap.add_argument("--data-root", default=None,
                    help="MVTec AD 2 root (sets NOROPCAL_MVTEC_AD2_ROOT for scoring)")
    ap.add_argument("--work-root", default=None,
                    help="EfficientAD work dir (sets NOROPCAL_PB6_WORK_ROOT for scoring)")
    ap.add_argument("--with-scoring", action="store_true",
                    help="include raw-image detector scoring stages (dataset + GPU needed)")
    ap.add_argument("--mode", choices=["dry-run", "execute"], default="dry-run")
    ap.add_argument("--scratch-dir", default=None,
                    help="empty target directory for isolated execute mode")
    args = ap.parse_args()

    if args.data_root:
        os.environ["NOROPCAL_MVTEC_AD2_ROOT"] = args.data_root
    if args.work_root:
        os.environ["NOROPCAL_PB6_WORK_ROOT"] = args.work_root

    checks = check_environment(args.with_scoring)
    print("[level2] environment checks:")
    for k, v in checks.items():
        print(f"   {k}: {v}")

    cfg = ROOT / args.config
    print(f"[level2] protocol record: {args.config} "
          f"({'present' if cfg.exists() else 'MISSING'}) "
          "(declarative mirror; authoritative values live in tools/pb7_p1_safety_actuator)")

    ok_env = (
        checks.get("numpy") == "OK" and checks.get("pandas") == "OK"
        and checks.get("score_caches_complete", False)
        and checks.get("stage_scripts_present", False)
    )
    if args.with_scoring:
        ok_env = ok_env and all(
            checks.get(f"env:{v}") == "SET"
            for v in ("NOROPCAL_MVTEC_AD2_ROOT", "NOROPCAL_PB6_WORK_ROOT")
        ) and checks.get("torch") == "OK"
    if args.mode == "dry-run":
        print_chain()
        if args.with_scoring:
            print("[level2] + raw-image scoring stages would run BEFORE the chain:")
            print("      python tools/r03_d1_extract_features.py")
            print("      python tools/r03_d0_extract_patchcore.py")
            print("      python tools/p0_scores.py")
            print("      python tools/pb6/d2_asset_resolve.py")
            print("      python tools/pb6/d2_efficientad_score.py   # EfficientAD 70k-step training")
        print(f"[level2] dry-run {'READY' if ok_env else 'BLOCKED'} "
              "(nothing was executed; sealed artifacts untouched)")
        return 0 if ok_env else 1

    # execute mode
    if not ok_env:
        print("[level2] environment BLOCKED; fix checks before executing.")
        return 1
    if not args.scratch_dir:
        print("[level2] execute mode requires --scratch-dir (isolated copy; "
              "sealed artifacts in the checkout are never overwritten).")
        return 2
    return run_execute(Path(args.scratch_dir), args.with_scoring)


if __name__ == "__main__":
    raise SystemExit(main())
