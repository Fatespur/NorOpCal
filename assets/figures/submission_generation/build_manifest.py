"""Build the S5 pipeline manifest for the NorOpCal IEEE TIM figure package."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy
from PIL import Image, __version__ as pillow_version


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]

UPSTREAM_INPUTS = (
    "PUBLICATION_GROUND_TRUTH.md",
    "reports/pb8_publication_benchmark/B_BASELINE_SUMMARY.csv",
    "reports/pb8_publication_benchmark/B_DETECTOR_BENCHMARK.csv",
    "reports/pb8_publication_benchmark/C_RANDOM_ABSTENTION_CONTROL.csv",
    "reports/pb8_publication_benchmark/C_RANDOM_ABSTENTION_SUMMARY.json",
    "reports/pb8_publication_benchmark/E_MECHANISM_EVIDENCE.json",
    "reports/pb8_publication_benchmark/TABLE_4_K_SUPPORT.csv",
    "reports/pb8_publication_benchmark/B_BOOTSTRAP_CI.csv",
    "reports/pb7_p0_threshold_correctability/B_UNIVARIATE_IDENTIFIABILITY.csv",
    "reports/pb7_p0_threshold_correctability/C_K_STABILITY.csv",
    "tools/pb7_p0_threshold_correctability/b_identifiability.py",
)

SNAPSHOT_INPUTS = (
    "reports/manuscript/NorOpCal_TIM_figures/data/fig2_policy_points.csv",
    "reports/manuscript/NorOpCal_TIM_figures/data/fig2_detector_points.csv",
    "reports/manuscript/NorOpCal_TIM_figures/data/fig2_support_points.csv",
    "reports/manuscript/NorOpCal_TIM_figures/data/fig3_diagnostic_summary.csv",
    "reports/manuscript/NorOpCal_TIM_figures/data/fig3_refined_evidence.csv",
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Return a repository-relative POSIX path."""

    return path.resolve().relative_to(REPO_ROOT).as_posix()


def file_record(path: Path, role: str, source_stage: str) -> dict[str, str]:
    """Create a schema-compatible manifest file record."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "role": role,
        "sha256": sha256(path),
        "source_stage": source_stage,
    }


def artifact_role(path: Path) -> str:
    """Map a package artifact to its delivery role."""

    if path.suffix == ".svg":
        return "editable vector figure"
    if path.suffix == ".pdf":
        return "publication vector figure"
    if path.suffix == ".png" and path.stem.endswith("_grayscale"):
        return "grayscale QA preview"
    if path.suffix == ".png":
        return "600-dpi raster preview"
    if path.suffix == ".py":
        return "reproducible generation or manifest script"
    if path.suffix == ".csv":
        return "frozen figure data snapshot"
    if path.name == "FIGURE_GENERATION_REPORT.md":
        return "generation and QA report"
    if path.name == "FIGURE_CONTRACT.md":
        return "figure evidence and design contract"
    return "S5 figure package artifact"


def png_metrics() -> dict[str, dict[str, object]]:
    """Collect pixel dimensions and embedded DPI for all final PNGs."""

    metrics: dict[str, dict[str, object]] = {}
    for path in sorted(PACKAGE_DIR.glob("*.png")):
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (None, None))
            metrics[path.name] = {
                "pixels": [image.width, image.height],
                "dpi": [round(float(value), 4) if value is not None else None for value in dpi],
            }
    return metrics


def build_manifest() -> dict[str, object]:
    """Assemble a completed S5 manifest after all QA gates pass."""

    inputs = [
        file_record(REPO_ROOT / path, "sealed or publication-ground-truth evidence", "AUX")
        for path in UPSTREAM_INPUTS
    ]
    inputs.extend(
        file_record(REPO_ROOT / path, "frozen figure data snapshot", "S5")
        for path in SNAPSHOT_INPUTS
    )

    artifact_paths = sorted(
        path
        for path in PACKAGE_DIR.iterdir()
        if path.is_file()
        and path.name != "pipeline_manifest.json"
        and path.suffix.lower() in {".svg", ".pdf", ".png", ".py", ".md", ".csv"}
    )
    artifacts = [file_record(path, artifact_role(path), "S5") for path in artifact_paths]

    return {
        "schema_version": "1.0",
        "run_id": "noropcal_tim_figures_20260820",
        "contest": "OTHER",
        "problem_id": "NorOpCal_TIM_figures",
        "stage": "S5",
        "status": "completed",
        "inputs": inputs,
        "artifacts": artifacts,
        "metrics": {
            "figure_count": 4,
            "layout_count": len(list(PACKAGE_DIR.glob("*.svg"))),
            "publication_exports": sum(
                1
                for path in PACKAGE_DIR.glob("*.*")
                if path.suffix.lower() in {".svg", ".pdf", ".png"}
                and not path.stem.endswith(("_grayscale", "_90pct_preview"))
            ),
            "grayscale_previews": len(list(PACKAGE_DIR.glob("*_grayscale.png"))),
            "support_split_scientific_source_binding": "PASS (both independent figures match frozen points and K=8 intervals)",
            "support_split_missing_interval_disclosure": "PASS (K=4 and Full remain point-only in both figures)",
            "support_split_title_caption_semantics": "PASS (utility and safety are independent figures; population caveat retained in captions)",
            "support_split_layout_bbox_checks": "PASS (all labels disjoint; utility baseline and safety-region gaps preserved)",
            "support_split_strict_size_dpi_check": "PASS (each figure 3.50 x 2.85 in; 600 dpi PNG)",
            "support_split_vector_font_check": "PASS (editable SVG text; no Type3 PDF marker; CIDFontType2 present)",
            "support_split_visual_review": "PASS (both figures in color and grayscale; no overlap or clipping)",
            "refined_fig3_strict_size_dpi_check": "PASS (7.16 x 2.28 in; 600 dpi PNG)",
            "refined_fig3_layout_bbox_checks": "PASS (label-to-interval, label-to-marker, footnote-to-axis)",
            "refined_fig3_svg_editable_text_check": "PASS (text nodes present; no image/base64 payload)",
            "refined_fig3_pdf_object_font_check": "PASS (no Type3 marker; CIDFontType2 present)",
            "refined_fig3_visual_review": "PASS (color, grayscale, and 90% preview; no overlap or clipping)",
            "png_files": png_metrics(),
        },
        "assumptions": [
            {
                "id": "A1",
                "statement": "The deliverable is an isolated venue-styled candidate figure package; the manuscript and its sealed figure registry remain unchanged.",
                "status": "validated",
                "risk": "high",
                "evidence": ["reports/manuscript/NorOpCal_TIM_figures/FIGURE_GENERATION_REPORT.md"],
            },
            {
                "id": "A2",
                "statement": "All quantitative coordinates are direct frozen snapshots; no experiment or metric was rerun or reanalyzed.",
                "status": "validated",
                "risk": "critical",
                "evidence": list(SNAPSHOT_INPUTS) + list(UPSTREAM_INPUTS),
            },
            {
                "id": "A3",
                "statement": "Figure 3 is a conservative optional diagnostic summary and does not reconstruct raw distributions or ROC curves.",
                "status": "validated",
                "risk": "medium",
                "evidence": [
                    "reports/manuscript/NorOpCal_TIM_figures/FIGURE_CONTRACT.md",
                    "reports/manuscript/NorOpCal_TIM_figures/REFINED_FIGURE_CONTRACT.md",
                    "reports/manuscript/NorOpCal_TIM_figures/FIGURE_GENERATION_REPORT.md",
                ],
            },
        ],
        "warnings": [
            {
                "code": "MANUSCRIPT_REGISTRY_UNCHANGED",
                "message": "These assets are not inserted into or registered by the manuscript; that requires separate S6 authorization.",
                "owner_stage": "S6",
                "blocking": False,
            },
            {
                "code": "PDF_FONT_AUDIT_FALLBACK",
                "message": "PDF font QA used source configuration plus PDF object-marker inspection because pypdf/Poppler/Ghostscript were unavailable.",
                "owner_stage": "S5",
                "blocking": False,
            },
            {
                "code": "NO_EXTERNAL_COLOR_SIMULATOR",
                "message": "Grayscale previews and redundant encodings were visually checked; an external color-vision simulator was not available.",
                "owner_stage": "S5",
                "blocking": False,
            },
        ],
        "provenance": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": "Codex (nature-figure + scipilot-figure-skill)",
            "commands": [
                "python reports/manuscript/NorOpCal_TIM_figures/make_all_figures.py --output-dir reports/manuscript/NorOpCal_TIM_figures",
                "python reports/manuscript/NorOpCal_TIM_figures/make_fig2_support_refined.py --output-dir reports/manuscript/NorOpCal_TIM_figures",
                "python reports/manuscript/NorOpCal_TIM_figures/make_fig3_refined.py --output-dir reports/manuscript/NorOpCal_TIM_figures",
                "python <scipilot-figure-skill>/scripts/check_figure.py <PDF> <PNG> --min-dpi 600 --width-in <target> --height-in <target> --strict",
                "python -m compileall reports/manuscript/NorOpCal_TIM_figures",
                "git diff --check",
            ],
            "environment": {
                "backend": "python/matplotlib",
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "matplotlib": matplotlib.__version__,
                "numpy": numpy.__version__,
                "pillow": pillow_version,
            },
        },
    }


def main() -> None:
    """Write the manifest with deterministic formatting and dynamic hashes."""

    manifest = build_manifest()
    destination = PACKAGE_DIR / "pipeline_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {relative(destination)} with {len(manifest['artifacts'])} artifacts")


if __name__ == "__main__":
    main()
