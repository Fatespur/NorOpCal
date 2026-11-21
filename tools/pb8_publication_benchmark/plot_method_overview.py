"""Render PB8 Figure 1 as a reproducible publication-grade vector schematic."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "pb8_publication_benchmark" / "figures"
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # type: ignore[import-not-found]  # noqa: E402
from setup_style import setup_style  # type: ignore[import-not-found]  # noqa: E402

COLORS = {
    "input": ("#DCEAF7", "#3B6F9D"),
    "signal": ("#E8E3F5", "#65539A"),
    "decision": ("#F9E4C8", "#B66C18"),
    "abstain": ("#ECEFF3", "#667085"),
    "adapt": ("#E2F0E9", "#2E7D5B"),
    "output": ("#CFE8DD", "#277356"),
    "validation": ("#F8DFDC", "#A84F47"),
}


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    role: str,
    *,
    fontsize: float = 8.2,
    weight: str = "normal",
) -> None:
    x, y = xy
    width, height = size
    fill, edge = COLORS[role]
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.09",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172033",
        fontweight=weight,
        linespacing=1.25,
    )


def _diamond(
    ax: plt.Axes, center: tuple[float, float], size: tuple[float, float], text: str
) -> None:
    cx, cy = center
    width, height = size
    fill, edge = COLORS["decision"]
    points = [
        (cx, cy + height / 2),
        (cx + width / 2, cy),
        (cx, cy - height / 2),
        (cx - width / 2, cy),
    ]
    ax.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=8.3,
        color="#172033",
        fontweight="bold",
        linespacing=1.2,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    color: str = "#526071",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.15,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    if label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            midpoint[0],
            midpoint[1] + 0.13,
            label,
            fontsize=7.5,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
        )


def _source_snapshot() -> None:
    rows = [
        ("input_source", "node", "Frozen source threshold", "input", "No target-label use"),
        (
            "input_support",
            "node",
            "K=8 target normal support",
            "input",
            "Known-normal samples only",
        ),
        ("shift_signal", "node", "Raw median score offset", "signal", "Outcome-free shift signal"),
        ("policy_constants", "node", "Q50 gate and lambda=0.25", "signal", "Frozen in PB7-P2"),
        ("gate", "decision", "Median shift >= Q50?", "decision", "Abstain when false"),
        ("abstain", "node", "Preserve source threshold", "abstain", "NoAdapt branch"),
        ("adapt", "node", "Bounded threshold update", "adapt", "Shrink raw offset by lambda=0.25"),
        ("threshold", "node", "Selective target threshold", "output", "Operating-point adaptation"),
        ("utility", "validation", "Delta BA / Delta FPR", "validation", "Versus NoAdapt"),
        ("safety", "validation", "Delta FNR / harm", "validation", "Frozen safety gates"),
        ("scope", "validation", "DINOv2 / TEST K=8", "validation", "Two eligible categories"),
    ]
    pd.DataFrame(rows, columns=["id", "kind", "label", "role", "evidence_semantics"]).to_csv(
        FIG / "FIGURE_1_METHOD_OVERVIEW_SOURCE.csv", index=False
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    _source_snapshot()
    setup_style(journal="general", lang="en", use_sciplots=False, constrained_layout=False)
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(
        0.2,
        5.94,
        "Conservative normal-only operating-point adaptation",
        fontsize=12,
        fontweight="bold",
        color="#172033",
        va="top",
    )
    ax.text(
        0.2,
        5.62,
        "Fixed anomaly scores; no retraining, gradients, target anomalies, or final-test labels",
        fontsize=8.2,
        color="#526071",
        va="top",
    )

    _box(ax, (0.35, 4.25), (1.85, 0.78), "Frozen source\nthreshold", "input", weight="bold")
    _box(ax, (0.35, 2.95), (1.85, 0.78), "K=8 target-normal\nsupport", "input", weight="bold")
    _box(ax, (2.75, 4.25), (1.9, 0.78), "Frozen Q50 gate\n+ λ=0.25 bound", "signal")
    _box(ax, (2.75, 2.95), (1.9, 0.78), "Raw median\nscore offset", "signal")
    _diamond(ax, (5.6, 3.98), (1.45, 1.25), "Shift\n≥ Q50?")
    _box(ax, (6.75, 4.72), (1.85, 0.78), "Abstain\nsource threshold", "abstain")
    _box(ax, (6.75, 2.55), (1.85, 0.78), "Adapt\nbounded update", "adapt")
    _box(
        ax,
        (8.92, 3.63),
        (1.15, 0.84),
        "Selective\nthreshold",
        "output",
        fontsize=8.0,
        weight="bold",
    )

    _arrow(ax, (2.2, 4.64), (2.75, 4.64))
    _arrow(ax, (2.2, 3.34), (2.75, 3.34))
    _arrow(ax, (4.65, 4.64), (5.05, 4.25))
    _arrow(ax, (4.65, 3.34), (5.05, 3.72))
    _arrow(ax, (6.18, 4.35), (6.75, 4.94), label="No")
    _arrow(ax, (6.18, 3.62), (6.75, 3.02), label="Yes")
    _arrow(ax, (8.6, 5.1), (8.92, 4.16), connectionstyle="arc3,rad=-0.15")
    _arrow(ax, (8.6, 2.94), (8.92, 3.93), connectionstyle="arc3,rad=0.15")

    _box(
        ax,
        (0.65, 0.65),
        (2.55, 0.98),
        "UTILITY\nΔBA +3.23 pp\n95% CI [2.70, 3.76]",
        "validation",
        fontsize=7.7,
        weight="bold",
    )
    _box(
        ax,
        (3.82, 0.65),
        (2.55, 0.98),
        "SAFETY\nΔFNR +0.24 pp\nHarm 3.67%",
        "validation",
        fontsize=7.7,
        weight="bold",
    )
    _box(
        ax,
        (6.98, 0.65),
        (2.55, 0.98),
        "SCOPE\nDINOv2 · frozen TEST K=8\n2 eligible categories",
        "validation",
        fontsize=7.7,
        weight="bold",
    )
    # One orthogonal evidence spine avoids crossing the method branches.
    _arrow(ax, (9.5, 3.63), (9.5, 1.93))
    ax.plot([1.92, 9.5], [1.93, 1.93], color="#526071", linewidth=1.15)
    for x in (1.92, 5.10, 8.25):
        _arrow(ax, (x, 1.93), (x, 1.63))
    ax.text(
        5.1,
        2.08,
        "Frozen evaluation and claim boundaries",
        fontsize=7.6,
        color="#526071",
        ha="center",
        va="bottom",
    )

    export_figure(
        fig,
        str(FIG / "FIGURE_1_METHOD_OVERVIEW"),
        formats=("pdf", "svg", "png"),
        dpi=600,
        size_inches=(10.2, 6.2),
        grayscale_preview=True,
        tight=True,
        pad_inches=0.04,
        transparent=False,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
