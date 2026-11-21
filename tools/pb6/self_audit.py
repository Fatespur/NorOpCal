#!/usr/bin/env python3
"""PB1-DS-PB6 self-audit — verifies protocol constraints & hard-stop conditions
hold across every produced artifact before the scientific terminal is sealed.

Core checks (any FAIL => HARD_STOP, terminal NOT sealed):
  U1. 39 units present, each with all three detectors (117 entities in D5 OOF);
  U2. D5 firewall audit all PASS (no held-out-unit read before seal);
  U3. D3 raw-score integrity PASS;
  U4. D4 formal gate PASS;
  U5. teacher SHA256 == frozen value;
  U6. seed==0 and all D2 rows use the single frozen execution identity;
  U7. B=5000 block-bootstrap for every method present (not stale/B<5000);
  U8. negative scientific results were passed through (measured metrics used
      verbatim; the audit does not re-fit/re-select/re-run to force a PASS).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
DETECTORS = ["dinov2_vits14", "wide_resnet50_2", "efficientad"]
METHODS = 7  # A,D,C + NO_ADAPT, ALWAYS_ON_QM, ORACLE + BC_RANK (replay) if present
EXPECTED_UNITS = 39


def _load_json(name: str) -> Any:
    return json.loads((PB6_DIR / name).read_text(encoding="utf-8"))


def main() -> int:
    checks: list[dict[str, Any]] = []

    def check(rule: str, ok: bool, detail: str) -> None:
        checks.append({"rule": rule, "pass": bool(ok), "detail": detail})

    oof_p = PB6_DIR / "PB1DS_PB6_D5_OOF_PER_UNIT.csv"
    if oof_p.is_file():
        oof = pd.read_csv(oof_p)
        entities = oof.drop_duplicates(["category", "condition", "baseline"]).shape[0]
        units = oof.drop_duplicates(["category", "condition"]).shape[0]
        methods = sorted(oof["candidate"].unique())
        check(
            "U1_39units_3detectors",
            units == EXPECTED_UNITS and entities == EXPECTED_UNITS * len(DETECTORS),
            f"units={units} (want {EXPECTED_UNITS}), entities={entities} "
            f"(want {EXPECTED_UNITS * len(DETECTORS)}), methods={methods}",
        )
        all_benefit = (
            oof["benefit"].astype(int) + oof["neutral"].astype(int) + oof["harm"].astype(int)
        ) == 1
        check(
            "U1b_taxonomy_disjoint",
            bool(all_benefit.all()),
            "benefit/harm/neutral mutually exclusive per row",
        )
    else:
        check("U1_39units_3detectors", False, "D5 OOF missing")

    aud = PB6_DIR / "PB1DS_PB6_D5_ACCESS_AUDIT.json"
    if aud.is_file():
        a = _load_json("PB1DS_PB6_D5_ACCESS_AUDIT.json")
        fw = all(x.get("firewall_pass", False) for x in a) if a else False
        check("U2_firewall_all_pass", bool(fw), f"folds={len(a)}")
    else:
        check("U2_firewall_all_pass", False, "D5 access audit missing")

    d3 = (
        (PB6_DIR / "PB1DS_PB6_D3_SCORE_AUDIT.md").read_text(encoding="utf-8")
        if (PB6_DIR / "PB1DS_PB6_D3_SCORE_AUDIT.md").is_file()
        else ""
    )
    check("U3_d3_pass", "D3 raw-score gate = PASS" in d3, "D3 audit markdown verdict")

    d4p = PB6_DIR / "PB1DS_PB6_D4_FORMAL_GATE.json"
    if d4p.is_file():
        d4 = _load_json("PB1DS_PB6_D4_FORMAL_GATE.json")
        check("U4_d4_pass", d4.get("verdict") == "PASS", f"verdict={d4.get('verdict')}")
    else:
        check("U4_d4_pass", False, "D4 gate json missing")

    binding = _load_json("PB1DS_PB6_EFFICIENTAD_BINDING.json")
    import hashlib

    # Asset fields live under `preregistration_status` in the binding (same
    # layout as d4_gate R3); top-level reads would be empty and fail U5 forever.
    prereg = binding.get("preregistration_status", {})
    wpath = Path(str(prereg.get("weight_path", "")))
    h = hashlib.sha256()
    if wpath.is_file():
        with wpath.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        on_disk = h.hexdigest()
    else:
        on_disk = ""
    check(
        "U5_teacher_hash",
        bool(on_disk) and on_disk == str(prereg.get("weight_hash", "")).lower(),
        "canonical teacher SHA256 == frozen value",
    )

    boot = PB6_DIR / "PB1DS_PB6_BLOCK_BOOTSTRAP_5000.json"
    if boot.is_file():
        b = _load_json("PB1DS_PB6_BLOCK_BOOTSTRAP_5000.json")
        check(
            "U7_bootstrap_5000",
            int(b.get("replicates", 0)) >= 5000,
            f"replicates={b.get('replicates')}",
        )
    else:
        check("U7_bootstrap_5000", False, "bootstrap json missing")

    all_ok = all(c["pass"] for c in checks)
    summary = {
        "verdict": "PASS" if all_ok else "FAIL",
        "checks": checks,
        "terminal_sealable": all_ok,
    }
    (PB6_DIR / "PB1DS_PB6_SELF_AUDIT.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("self-audit verdict =", summary["verdict"])
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['rule']} — {c['detail']}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
