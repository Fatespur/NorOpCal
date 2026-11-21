#!/usr/bin/env python3
"""PB1-DS-PB6 post-D2 orchestrator — drives the authorized scientific chain.

Runs the frozen protocol stages IN ORDER after the real D2 run completes, and
enforces the PB6 hard-stop rules: on ANY condition below the chain STOPS and no
further scientific execution or auto-repair is performed under this authorization.

  D2  -> D3  -> D4 gate  -> D5 -> D6 -> D7 -> D8 -> B=5000 bootstrap ->
         self-audit -> scientific terminal -> final report

Gating (each FAIL terminates immediately):
  G1. D2 not complete (per D4 gate: not all 8 categories at exactly 70000 steps,
      incomplete/unverifiable checkpoints/caches, or teacher SHA256 != frozen);
  G2. D3 raw-score integrity NOT PASS (any of finite/identity/39-unit mapping FAIL);
  G3. D4 formal gate FAIL when the protocol binds FAIL to terminate downstream;
  G4. any required stage script missing (HARD_STOP, never silently skipped);
  G5. any stage subprocess returns non-zero (HARD_STOP on FAIL).

A stage that reports scientific negative results is passed through verbatim to the
terminal; the driver never re-tunes, re-selects, or re-runs to force a PASS.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"

STAGES: list[str] = [
    "d3_score_integrity",
    "d4_gate",
    "d5_tri_fit",
    "d6_benefit",
    "d7_benchmark",
    "d8_k8_contamination",
    "stat_bootstrap",
    "self_audit",
    "terminal",
    "final_report",
]


def _require(script: str) -> Path:
    p = ROOT / "tools" / "pb6" / f"{script}.py"
    if not p.is_file():
        raise SystemExit(f"HARD_STOP: missing stage script {p} — cannot proceed.")
    return p


def main() -> int:
    import json

    log: list[str] = []
    # G1/G2/G3 first: D4 gate already verifies D2 completeness + teacher hash +
    # freeze consistency (R1-R4) and consumes the D3 audit (R5).
    for script in STAGES:
        path = _require(script)
        print(f"\n=== {script} ===", flush=True)
        r = subprocess.run([str(PY), str(path)], cwd=str(ROOT))
        if r.returncode != 0:
            log.append(f"{script}: FAIL (exit {r.returncode})")
            print(f"HARD_STOP at {script}", flush=True)
            break
        log.append(f"{script}: PASS")
    summary = {"chain_complete": all(x.endswith(": PASS") for x in log), "steps": log}
    out = PB6_DIR / "PB1DS_PB6_POST_D2_DRIVER.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "\n".join(log))
    print("DRIVER verdict:", "COMPLETE" if summary["chain_complete"] else "STOPPED")
    return 0 if summary["chain_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
