"""Generate the NorOpCal utility-safety and detector-specificity figure."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_common import COLORS, apply_ieee_style, export_figure, panel_label


LOGGER = logging.getLogger(__name__)
DATA_DIR: Final[Path] = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Point:
    """One frozen utility-safety point."""

    name: str
    delta_fnr_pp: float
    delta_ba_pp: float
    safety_class: str
    detail: str


def _load_points(path: Path, *, name_key: str, detail_key: str) -> list[Point]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Point(
            name=row[name_key],
            delta_fnr_pp=float(row["delta_fnr_pp"]),
            delta_ba_pp=float(row["delta_ba_pp"]),
            safety_class=row["safety_class"],
            detail=row[detail_key],
        )
        for row in rows
    ]


def _prepare_axis(ax: plt.Axes) -> None:
    """Apply shared axes, zones, and the preregistered safety budget."""

    ax.set_xlim(-0.9, 8.25)
    ax.set_ylim(-1.45, 8.85)
    ax.axvspan(-0.9, 2.0, color=COLORS["blue_soft"], alpha=0.42, zorder=0)
    ax.axvspan(2.0, 8.25, color=COLORS["orange_soft"], alpha=0.42, zorder=0)
    ax.axvline(2.0, color=COLORS["ink"], linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.7, zorder=1)
    ax.text(0.55, 8.48, "Admissible region", color=COLORS["muted"], fontsize=5.9, ha="center")
    ax.text(5.15, 8.48, "Safety-budget violation", color=COLORS["muted"], fontsize=5.9, ha="center")
    ax.text(
        2.12,
        7.82,
        "Aggregate safety budget\n+2 pp",
        color=COLORS["ink"],
        fontsize=5.8,
        ha="left",
        va="top",
    )
    ax.set_xlabel(r"Mean $\Delta$FNR (pp)")
    ax.set_ylabel(r"Mean $\Delta$BA (pp)")
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 2, 4, 6, 8])


def _annotate(
    ax: plt.Axes,
    point: Point,
    label: str,
    offset: tuple[float, float],
    *,
    color: str,
    bold: bool = False,
) -> None:
    ax.annotate(
        label,
        xy=(point.delta_fnr_pp, point.delta_ba_pp),
        xytext=offset,
        textcoords="offset points",
        fontsize=6.0,
        fontweight="bold" if bold else "normal",
        color=color,
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["line"],
            "linewidth": 0.55,
            "shrinkA": 1.5,
            "shrinkB": 3.0,
        },
        zorder=6,
    )


def _plot_policy(ax: plt.Axes, points: list[Point]) -> None:
    markers = {"NoAdapt": "o", "B1": "^", "B2": "s", "B3": "P", "B4": "X", "NorOpCal": "D"}
    offsets = {
        "NoAdapt": (6, -8),
        "B1": (-5, 7),
        "B2": (-16, 7),
        "B3": (6, 7),
        "B4": (-18, -9),
        "NorOpCal": (7, -12),
    }
    for point in points:
        if point.name == "NoAdapt":
            face, edge, size, line = COLORS["white"], COLORS["grey"], 32, 0.9
        elif point.name == "NorOpCal":
            face, edge, size, line = COLORS["blue"], COLORS["ink"], 58, 0.8
        elif point.safety_class == "violation":
            face, edge, size, line = COLORS["orange_soft"], COLORS["orange"], 42, 1.0
        else:
            face, edge, size, line = COLORS["green_soft"], COLORS["green"], 42, 1.0
        ax.scatter(
            point.delta_fnr_pp,
            point.delta_ba_pp,
            marker=markers[point.name],
            s=size,
            facecolor=face,
            edgecolor=edge,
            linewidth=line,
            zorder=5,
        )
        _annotate(
            ax,
            point,
            point.name,
            offsets[point.name],
            color=COLORS["blue"] if point.name == "NorOpCal" else COLORS["ink"],
            bold=point.name == "NorOpCal",
        )
    ax.set_title("Policy-level utility–safety tradeoff", pad=8)


def _plot_detectors(ax: plt.Axes, points: list[Point]) -> None:
    styles = {
        "DINOv2": ("D", COLORS["blue"], COLORS["ink"], 58),
        "PatchCore": ("X", COLORS["red_soft"], COLORS["red"], 52),
        "EfficientAD": ("^", COLORS["teal_soft"], COLORS["teal"], 48),
    }
    offsets = {"DINOv2": (8, 7), "PatchCore": (-8, 7), "EfficientAD": (-8, 8)}
    labels = {
        "DINOv2": "DINOv2",
        "PatchCore": "PatchCore\n(safety violation)",
        "EfficientAD": "EfficientAD",
    }
    for point in points:
        marker, face, edge, size = styles[point.name]
        ax.scatter(
            point.delta_fnr_pp,
            point.delta_ba_pp,
            marker=marker,
            s=size,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=5,
        )
        _annotate(
            ax,
            point,
            labels[point.name],
            offsets[point.name],
            color=COLORS["blue"] if point.name == "DINOv2" else COLORS["ink"],
            bold=point.name == "DINOv2",
        )
    ax.set_title("Detector-specific utility–safety response", pad=8)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor=COLORS["grey"], markersize=4.6, label="Reference"),
        Line2D([], [], marker="P", linestyle="None", markerfacecolor=COLORS["green_soft"], markeredgecolor=COLORS["green"], markersize=5.1, label="Within budget"),
        Line2D([], [], marker="X", linestyle="None", markerfacecolor=COLORS["orange_soft"], markeredgecolor=COLORS["orange"], markersize=5.1, label="Budget violation"),
        Line2D([], [], marker="D", linestyle="None", markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["ink"], markersize=5.1, label="NorOpCal / principal"),
    ]


def _draw(layout: str, policy: list[Point], detectors: list[Point]) -> plt.Figure:
    if layout == "double":
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35), constrained_layout=False)
        legend_cols = 4
        top_margin, bottom_margin, vertical_spacing = 0.76, 0.18, 0.34
    else:
        fig, axes = plt.subplots(2, 1, figsize=(3.50, 6.25), constrained_layout=False)
        legend_cols = 2
        top_margin, bottom_margin, vertical_spacing = 0.86, 0.09, 0.46
    for ax in axes:
        _prepare_axis(ax)
    _plot_policy(axes[0], policy)
    _plot_detectors(axes[1], detectors)
    panel_label(axes[0], "a", x=-0.16)
    panel_label(axes[1], "b", x=-0.16)
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=legend_cols,
        fontsize=6.2,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    fig.subplots_adjust(
        top=top_margin, bottom=bottom_margin, wspace=0.34, hspace=vertical_spacing
    )
    return fig


def generate(output_dir: Path) -> list[Path]:
    """Generate double- and single-column variants from frozen snapshots."""

    apply_ieee_style()
    policy = _load_points(DATA_DIR / "fig2_policy_points.csv", name_key="policy", detail_key="role")
    detectors = _load_points(DATA_DIR / "fig2_detector_points.csv", name_key="detector", detail_key="verdict")
    generated: list[Path] = []
    generated.extend(export_figure(_draw("double", policy, detectors), output_dir, "Fig2_Utility_Safety_and_Detector_Specificity"))
    generated.extend(export_figure(_draw("single", policy, detectors), output_dir, "Fig2_Utility_Safety_and_Detector_Specificity_singlecol"))
    return generated


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Generated %d Fig. 2 exports", len(generate(args.output_dir)))


if __name__ == "__main__":
    main()
