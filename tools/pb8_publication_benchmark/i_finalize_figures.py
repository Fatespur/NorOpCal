"""Finalize the PB8 S5 manifest after format and human visual QA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "pb8_publication_benchmark"
FIG = OUT / "figures"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "role": role,
        "source_stage": "PB8-I",
        "sha256": _sha256(path),
    }


def main() -> None:
    manifest_path = FIG / "pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_files: list[Path] = []
    for number, stem in [
        (1, "FIGURE_1_METHOD_OVERVIEW"),
        (2, "FIGURE_2_SAFETY_UTILITY"),
        (3, "FIGURE_3_DETECTOR_DECOMPOSITION"),
        (4, "FIGURE_4_SUPPORT_ROBUSTNESS"),
        (5, "FIGURE_5_RESIDUAL_HARM_MAP"),
    ]:
        del number
        final_files.extend(FIG / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png"))
        final_files.append(FIG / f"{stem}_grayscale.png")
    missing = [path for path in final_files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing final figure artifacts: {missing}")

    manifest["status"] = "completed"
    manifest["artifacts"] = [_artifact(path, "publication_figure") for path in final_files]
    manifest["artifacts"].extend(
        _artifact(FIG / f"FIGURE_{number}_{name}_SOURCE.csv", "figure_source_snapshot")
        for number, name in [
            (1, "METHOD_OVERVIEW"),
            (2, "SAFETY_UTILITY"),
            (3, "DETECTOR_DECOMPOSITION"),
            (4, "SUPPORT_ROBUSTNESS"),
            (5, "RESIDUAL_HARM"),
        ]
    )
    manifest["metrics"].update(
        {
            "publication_figures": 5,
            "formats_per_figure": 4,
            "strict_raster_checks_passed": 5,
            "strict_vector_checks_passed": 5,
            "human_visual_qa": "PASS",
        }
    )
    manifest["warnings"].extend(
        [
            {
                "code": "FIGURE5_SVG_RASTER_LAYER",
                "message": "The heatmap color layer is embedded raster in SVG; labels and axes remain vector, with 600-DPI PNG and PDF alternatives.",
                "blocking": False,
                "owner_stage": "PB8-I",
            },
            {
                "code": "FLOWCHART_CANDIDATE_REJECTED",
                "message": "The cumcm-academic-flowchart candidate passed machine gates but failed human English-label/layout QA; it is retained only as diagnostic evidence and is not the formal Figure 1.",
                "blocking": False,
                "owner_stage": "PB8-I",
            },
        ]
    )
    manifest["provenance"].update(
        {
            "figure1_script": "tools/pb8_publication_benchmark/plot_method_overview.py",
            "figure2_5_script": "tools/pb8_publication_benchmark/plot_figures.py",
            "qa": "strict check_figure.py plus rendered contact-sheet human review",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    gate = {
        "schema_version": "1.0",
        "phase": "I",
        "publication_figures": 5,
        "source_snapshots": 5,
        "raster_dpi": 600,
        "strict_format_qa": "PASS",
        "human_visual_qa": "PASS",
        "formal_figure_1": "code-native Python vector schematic",
        "rejected_candidate": "cumcm-academic-flowchart output retained as diagnostic only",
        "verdict": "PASS",
        "terminal": "PROCEED_TO_J",
    }
    (OUT / "I_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
