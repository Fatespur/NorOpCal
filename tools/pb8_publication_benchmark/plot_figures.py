"""PB8 I — render publication figures 2–5 from frozen source snapshots."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "pb8_publication_benchmark"
FIG = OUT / "figures"
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # type: ignore[import-not-found]  # noqa: E402
from setup_style import setup_style  # type: ignore[import-not-found]  # noqa: E402

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
GRAY = "#6B7280"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(fig: plt.Figure, stem: str, size: tuple[float, float]) -> list[Path]:
    paths = export_figure(
        fig,
        str(FIG / stem),
        formats=("pdf", "svg", "png"),
        dpi=600,
        size_inches=size,
        grayscale_preview=True,
        tight=True,
        pad_inches=0.04,
        transparent=False,
    )
    plt.close(fig)
    return [Path(path) for path in paths]


def figure_2() -> list[Path]:
    data = pd.read_csv(FIG / "FIGURE_2_SAFETY_UTILITY_SOURCE.csv")
    fig, ax = plt.subplots()
    x = 100 * data["delta_fnr"].to_numpy()
    y = 100 * data["delta_ba"].to_numpy()
    ax.plot(x, y, color=GRAY, linewidth=1.2, zorder=1)
    colors = [GRAY, GREEN, BLUE, ORANGE, VERMILION]
    markers = ["s", "o", "o", "o", "o"]
    for idx, row in data.iterrows():
        ax.scatter(
            x[idx],
            y[idx],
            s=58,
            color=colors[idx],
            marker=markers[idx],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset = (5, 7) if idx not in {3, 4} else (5, -13)
        ax.annotate(
            str(row["policy"]),
            (x[idx], y[idx]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.axvline(2.0, color=VERMILION, linestyle="--", linewidth=1.0)
    ax.text(
        2.08,
        0.25,
        "Safety limit: ΔFNR = +2 pp",
        color=VERMILION,
        fontsize=7.5,
        rotation=90,
        va="bottom",
    )
    ax.axhline(0, color="#B8BDC5", linewidth=0.8)
    ax.set_xlabel("Safety cost, ΔFNR (percentage points; lower is safer)")
    ax.set_ylabel("Utility, Δ balanced accuracy (percentage points)")
    ax.set_title("Frozen shrinkage path at Q50 gate (DINOv2, K=8)", loc="left")
    ax.set_xlim(-0.35, 8.25)
    ax.set_ylim(-0.35, 5.85)
    return _save(fig, "FIGURE_2_SAFETY_UTILITY", (7.1, 4.3))


def figure_3() -> list[Path]:
    data = pd.read_csv(FIG / "FIGURE_3_DETECTOR_DECOMPOSITION_SOURCE.csv")
    labels = {
        "dinov2_vits14": "DINOv2 ViT-S/14",
        "wide_resnet50_2": "PatchCore (WRN50-2)",
        "efficientad": "EfficientAD",
    }
    data["label"] = data["detector"].map(labels)
    data = data.set_index("label").loc[list(labels.values())].reset_index()
    y = np.arange(len(data))[::-1]
    colors = [BLUE, ORANGE, GREEN]
    specs = [
        ("safe_headroom_fraction", "Safe-correctable headroom", (0, 1), "{:.0f}%", 100),
        ("delta_ba", "Δ balanced accuracy", (-0.012, 0.038), "{:+.1f} pp", 100),
        ("harm_rate", "Harm rate", (0, 0.29), "{:.1f}%", 100),
        ("coverage", "Adaptation coverage", (0, 0.47), "{:.1f}%", 100),
    ]
    fig, axes = plt.subplots(1, 4, sharey=True)
    for ax, (column, title, limits, fmt, scale) in zip(axes, specs, strict=True):
        values = data[column].to_numpy() * scale
        scaled_limits = (limits[0] * scale, limits[1] * scale)
        ax.axvline(0, color="#C7CBD1", linewidth=0.8)
        for idx, value in enumerate(values):
            edge = VERMILION if bool(data.loc[idx, "safety_fail"]) else "white"
            ax.scatter(
                value, y[idx], s=58, color=colors[idx], edgecolor=edge, linewidth=1.3, zorder=3
            )
            ax.annotate(
                fmt.format(value),
                (value, y[idx]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.2,
            )
        ax.set_xlim(*scaled_limits)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axes[0].set_yticks(y, data["label"])
    axes[1].set_xlabel("Percentage points")
    axes[0].set_xlabel("Fraction")
    axes[2].set_xlabel("Fraction")
    axes[3].set_xlabel("Fraction")
    fig.suptitle(
        "Frozen policy decomposes differently across detectors (K=8)",
        x=0.02,
        ha="left",
        fontsize=10,
    )
    return _save(fig, "FIGURE_3_DETECTOR_DECOMPOSITION", (9.2, 3.4))


def figure_4() -> list[Path]:
    data = pd.read_csv(FIG / "FIGURE_4_SUPPORT_ROBUSTNESS_SOURCE.csv", dtype={"K": str})
    order = ["4", "8", "FULL"]
    x = np.arange(len(order))
    metrics = [
        ("mean_delta_ba", "Mean Δ balanced accuracy", "Percentage points", 100),
        ("mean_delta_fnr", "Mean ΔFNR", "Percentage points", 100),
        ("sealed_safety_frequency", "Safety frequency", "Fraction", 1),
    ]
    fig, axes = plt.subplots(1, 3)
    styles = {"DINOv2": (BLUE, "o", -0.06), "POOLED": (ORANGE, "s", 0.06)}
    for ax, (metric, title, ylabel, scale) in zip(axes, metrics, strict=True):
        for group, (color, marker, shift) in styles.items():
            subset = data.loc[data["group"].eq(group)].set_index("K").loc[order]
            values = scale * subset[metric].to_numpy()
            ax.plot(
                x + shift,
                values,
                color=color,
                marker=marker,
                linewidth=1.2,
                markersize=5.5,
                label=group,
            )
        if metric == "mean_delta_fnr":
            ax.axhline(2.0, color=VERMILION, linestyle="--", linewidth=0.9, label="Safety limit")
        if metric == "sealed_safety_frequency":
            ax.axhline(1.0, color="#AEB4BC", linestyle=":", linewidth=0.9)
            ax.set_ylim(-0.06, 1.08)
        ax.set_xticks(x, ["K=4", "K=8", "FULL"])
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    handles, labels = axes[1].get_legend_handles_labels()
    # Keep the shared legend compact and make the safety threshold explicit in panel b.
    axes[0].legend(loc="best", frameon=False, fontsize=7.5)
    if "Safety limit" in labels:
        axes[1].legend(handles[-1:], labels[-1:], loc="best", frameon=False, fontsize=7.2)
    fig.suptitle("Support-size robustness under the frozen policy", x=0.02, ha="left", fontsize=10)
    return _save(fig, "FIGURE_4_SUPPORT_ROBUSTNESS", (9.0, 3.5))


def figure_5() -> list[Path]:
    data = pd.read_csv(FIG / "FIGURE_5_RESIDUAL_HARM_SOURCE.csv")
    order = ["overexposed", "shift_1", "shift_2", "shift_3", "underexposed"]
    matrix = data.pivot(index="category", columns="condition", values="harm").loc[
        ["fabric", "walnuts"], order
    ]
    fig, ax = plt.subplots()
    image = ax.imshow(
        100 * matrix.to_numpy(),
        cmap="magma",
        vmin=0,
        vmax=40,
        aspect="auto",
        interpolation="nearest",
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = 100 * float(matrix.iloc[row, col])
            color = "white" if value > 18 else "#20242A"
            ax.text(col, row, f"{value:.1f}%", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(
        np.arange(len(order)), ["Overexposed", "Shift 1", "Shift 2", "Shift 3", "Underexposed"]
    )
    ax.set_yticks(np.arange(2), ["Fabric", "Walnuts"])
    ax.set_xlabel("Acquisition-shift condition")
    ax.set_ylabel("Frozen TEST K=8 category")
    ax.set_title("Residual harm is localized to walnuts / shift 1 (DINOv2)", loc="left")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Harm rate (%)")
    return _save(fig, "FIGURE_5_RESIDUAL_HARM_MAP", (7.1, 3.2))


def _write_manifest(outputs: list[Path]) -> None:
    inputs = sorted(FIG.glob("FIGURE_*_SOURCE.csv"))
    flowchart = FIG / "figure1_method_overview" / "diagram_manifest.json"
    artifacts: list[dict[str, Any]] = []
    for path in sorted(outputs):
        artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "role": "publication_figure",
                "source_stage": "PB8-I",
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "run_id": "PB8-PUBLICATION-FIGURES-20260820",
        "contest": "OTHER",
        "problem_id": "ShiftAD-PB8",
        "stage": "S5",
        "status": "in_progress",
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)} for path in inputs
        ]
        + [{"path": flowchart.relative_to(ROOT).as_posix(), "sha256": _sha256(flowchart)}],
        "artifacts": artifacts,
        "metrics": {
            "quantitative_figures": 4,
            "formats_per_figure": 4,
            "raster_dpi": 600,
            "flowchart_primary_score": 100.0,
            "flowchart_layout_diversity": 0.9357,
        },
        "assumptions": [
            "All quantitative panels use frozen PB8 source snapshots without refitting.",
            "Percentage-point axes are raw fractions multiplied by 100.",
            "Figure 5 displays entity-level mean harm over 30 sealed support repeats.",
        ],
        "warnings": [
            {
                "code": "EVIDENCE_BREADTH",
                "message": "K=8 figures contain only two eligible TEST categories.",
                "blocking": False,
                "owner_stage": "PB8-D",
            }
        ],
        "provenance": {
            "script": "tools/pb8_publication_benchmark/plot_figures.py",
            "style_skill": "scipilot-figure-skill",
            "flowchart_skill": "cumcm-academic-flowchart",
        },
    }
    (FIG / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    setup_style(journal="general", lang="en", use_sciplots=False, constrained_layout=True)
    outputs: list[Path] = []
    for renderer in (figure_2, figure_3, figure_4, figure_5):
        outputs.extend(renderer())
    _write_manifest(outputs)


if __name__ == "__main__":
    main()
