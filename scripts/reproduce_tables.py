"""Reproduce the paper tables (registry Tables I-V + S1) from sealed artifacts.

This script performs a verbatim, value-preserving export of the sealed PB8 /
PB8-S1 table artifacts into human-readable Markdown under
``outputs/verification/tables/``. It recomputes NOTHING: every number is
copied byte-faithfully from the sealed CSVs bound by the G1 migration
manifest.

Table registry source: ``docs/reference/PUBLICATION_GROUND_TRUTH.md`` section 12.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "verification" / "tables"

# registry entry -> sealed artifact (paths relative to repo root)
TABLE_REGISTRY = [
    ("TABLE_1_MAIN_BENCHMARK",
     "reports/pb8_publication_benchmark/TABLE_1_MAIN_BENCHMARK.csv",
     "Main DINOv2 K8 benchmark B0-B6 (paper Table I)"),
    ("TABLE_2_DETECTOR_SPECIFICITY",
     "reports/pb8_publication_benchmark/TABLE_2_DETECTOR_SPECIFICITY.csv",
     "Detector specificity verdicts (paper Table II)"),
    ("TABLE_3_COMPONENT_ABLATION",
     "reports/pb8_publication_benchmark/TABLE_3_COMPONENT_ABLATION.csv",
     "Component ablation gate x shrinkage (paper Table III)"),
    ("TABLE_4_K_SUPPORT",
     "reports/pb8_publication_benchmark/TABLE_4_K_SUPPORT.csv",
     "K-support K4/K8/FULL (paper Table IV)"),
    ("TABLE_5_SCOPE_AND_FAILURES",
     "reports/pb8_publication_benchmark/TABLE_5_SCOPE_AND_FAILURES.csv",
     "Scope and failures L1-L7 (paper Table V)"),
    ("TABLE_S1_AD2_K8_EVIDENCE_BREADTH",
     "reports/pb8_s1_evidence_breadth/TABLE_S1_AD2_K8_EVIDENCE_BREADTH.csv",
     "AD2 K8 evidence breadth (supplementary Table S1)"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> int:
    manifest_path = ROOT / "_audit" / "migration_manifest.csv"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                manifest[row["source_path"]] = row["original_sha256"]

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"tables": [], "all_source_hashes_verified": True}
    ok_all = True
    for name, rel, role in TABLE_REGISTRY:
        src = ROOT / rel
        if not src.exists():
            print(f"[tables] MISSING sealed artifact: {rel}")
            ok_all = False
            continue
        src_hash = sha256_file(src)
        hash_ok = (manifest.get(rel) == src_hash) if manifest else None
        if hash_ok is False:
            ok_all = False
        with open(src, newline="", encoding="utf-8-sig") as fh:
            reader = list(csv.reader(fh))
        header, body = reader[0], reader[1:]
        md = (f"# {name}\n\nRole: {role}\n\n"
              f"Source artifact: `{rel}`  \n"
              f"SHA256: `{src_hash}`"
              f"{'' if hash_ok is None else (' (verified against migration manifest)' if hash_ok else ' **MISMATCH vs manifest**')}\n\n"
              + md_table(header, body) + "\n")
        dst = OUT / f"{name}.md"
        dst.write_text(md, encoding="utf-8")
        summary["tables"].append({"table": name, "source": rel,
                                  "sha256": src_hash,
                                  "manifest_verified": hash_ok,
                                  "rows": len(body),
                                  "markdown": str(dst.relative_to(ROOT))})
        print(f"[tables] {name}: {len(body)} data rows -> {dst.relative_to(ROOT)}")

    summary["all_source_hashes_verified"] = ok_all
    (OUT / "tables_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("[tables]", "SOURCE_HASH_VERIFIED" if ok_all else "HASH_MISMATCH_DETECTED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
