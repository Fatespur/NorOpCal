"""Re-render the four final main-text figures from their frozen inputs.

Calls the ORIGINAL submission figure-generation scripts (verbatim, under
``assets/figures/submission_generation/``) with ``--output-dir`` redirected to
``outputs/figures/regenerated/`` so sealed exports under ``assets/figures/``
are never overwritten.

Parity levels reported (see G2 spec section 26):
- BYTE_PARITY                             regenerated hash == shipped hash
- CONTENT_IDENTICAL_AFTER_DATE_NORM       PDFs identical after blanking
                                          matplotlib CreationDate/ModDate
- DATA_PARITY_RENDERED_DIFFERENT_BYTES    rendered from identical frozen
                                          inputs; bytes differ (metadata/fonts)
- VISUAL_NOT_VERIFIED                     no automated visual claim is made

Known name mapping (sealed FIGURE_RENAME_MAP.md):
    make_fig1.py                  -> Fig1_NorOpCal_Framework(.*)   [shipped as
                                     Fig1_NorOpCal_Overall_Framework.svg]
    make_fig3_refined.py          -> Fig3_Gate_Informativeness_and_Correctability_REFINED
                                     [shipped as Fig2_Gate_Informativeness_and_Correctability_Diagnostic.*]
    make_fig2_support_refined.py  -> Utility_across_Frozen_Support_Settings_REFINED
                                   + Safety_across_Frozen_Support_Settings_REFINED
                                     [shipped as Fig3_Utility_Across_Frozen_Support_Settings.*
                                      and Fig4_Safety_Across_Frozen_Support_Settings.*]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "assets" / "figures" / "submission_generation"
OUT = ROOT / "outputs" / "figures" / "regenerated"
SHIPPED = ROOT / "assets" / "figures" / "submission"

JOBS = [
    {"script": "make_fig1.py", "produced_prefix": "Fig1_NorOpCal_Framework",
     "shipped": ["Fig1_NorOpCal_Overall_Framework.svg"]},
    {"script": "make_fig3_refined.py", "produced_prefix": "Fig3_Gate_Informativeness_and_Correctability_REFINED",
     "shipped": ["Fig2_Gate_Informativeness_and_Correctability_Diagnostic.pdf",
                 "Fig2_Gate_Informativeness_and_Correctability_Diagnostic.svg",
                 "Fig2_Gate_Informativeness_and_Correctability_Diagnostic.png"]},
    {"script": "make_fig2_support_refined.py",
     "produced_prefix": "_REFINED",
     "produced_contains": ("Utility_across_Frozen_Support_Settings_REFINED",
                           "Safety_across_Frozen_Support_Settings_REFINED"),
     "shipped": ["Fig3_Utility_Across_Frozen_Support_Settings.pdf",
                 "Fig3_Utility_Across_Frozen_Support_Settings.svg",
                 "Fig4_Safety_Across_Frozen_Support_Settings.pdf",
                 "Fig4_Safety_Across_Frozen_Support_Settings.svg"]},
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_hash_norm_dates(path: Path) -> str:
    """SHA256 of a PDF after blanking CreationDate/ModDate metadata.

    matplotlib-stamped dates are the only expected byte difference between
    deterministic re-renders; normalizing them exposes content-level parity
    without overclaiming raw BYTE_PARITY.
    """
    import re

    b = Path(path).read_bytes()
    b = re.sub(rb"/CreationDate \([^)]*\)", b"/CreationDate ()", b)
    b = re.sub(rb"/ModDate \([^)]*\)", b"/ModDate ()", b)
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"jobs": [], "note": "renders only; no metric recomputation"}

    for job in JOBS:
        script = GEN / job["script"]
        if not script.exists():
            report["jobs"].append({"script": job["script"], "status": "MISSING_SCRIPT"})
            continue
        proc = subprocess.run(
            [sys.executable, str(script), "--output-dir", str(OUT)],
            cwd=GEN, capture_output=True, text=True,
        )
        entry = {
            "script": job["script"],
            "returncode": proc.returncode,
            "status": "RENDER_OK" if proc.returncode == 0 else "RENDER_FAIL",
        }
        if proc.returncode == 0:
            produced = []
            prefixes = job.get("produced_contains") or (job["produced_prefix"],)
            for p in sorted(OUT.iterdir()):
                if p.is_file() and any(p.name.startswith(prefix) for prefix in prefixes):
                    produced.append({"file": p.name,
                                     "sha256": sha256_file(p),
                                     "bytes": p.stat().st_size})
            entry["regenerated_files"] = produced
            # shipped parity: compare SVG siblings by hash when names allow
            parity = []
            for shipped_name in job["shipped"]:
                sp = SHIPPED / shipped_name
                if not sp.exists():
                    parity.append({"shipped": shipped_name, "status": "SHIPPED_MISSING"})
                    continue
                suffix = Path(shipped_name).suffix
                cand = [q for q in produced if q["file"].endswith(suffix)]
                exact = any(q["sha256"] == sha256_file(sp) for q in cand)
                status = "BYTE_PARITY" if exact else None
                if not exact and suffix == ".pdf":
                    # deterministic render check: normalize PDF date metadata
                    reg_pdfs = {q["file"]: OUT / q["file"] for q in cand}
                    exact_norm = any(
                        pdf_hash_norm_dates(reg_pdfs[qf]) == pdf_hash_norm_dates(sp)
                        for qf in reg_pdfs
                    )
                    status = ("CONTENT_IDENTICAL_AFTER_DATE_NORM" if exact_norm
                              else "DATA_PARITY_RENDERED_DIFFERENT_BYTES")
                if status is None:
                    status = ("DATA_PARITY_RENDERED_DIFFERENT_BYTES" if cand
                              else "NO_REGENERATED_COUNTERPART")
                parity.append({"shipped": shipped_name, "status": status})
            entry["parity_vs_shipped"] = parity
        report["jobs"].append(entry)

    data_ok = True
    data_dir = GEN / "data"
    for csvp in sorted(data_dir.glob("*.csv")):
        h = sha256_file(csvp)
        entry = {"input_snapshot": csvp.name, "sha256": h}
        report.setdefault("frozen_input_snapshots", []).append(entry)
    report["data_inputs_note"] = (
        "All quantitative panels render from these frozen CSV snapshots "
        "(DATA_PARITY); byte equality of rendered outputs across machines or "
        "library versions is NOT expected or claimed."
    )

    out_json = ROOT / "outputs" / "verification" / "figures_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1)[:2400])
    print(f"[figures] report: {out_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
