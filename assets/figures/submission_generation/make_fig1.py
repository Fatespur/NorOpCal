"""Generate the NorOpCal framework schematic in IEEE TIM layouts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figure_common import COLORS, apply_ieee_style, export_figure


LOGGER = logging.getLogger(__name__)


def _box(
    ax: plt.Axes,
    xywh: tuple[float, float, float, float],
    title: str,
    lines: Sequence[str],
    *,
    face: str,
    edge: str,
    title_size: float = 7.0,
    text_size: float = 6.15,
    linewidth: float = 1.0,
    linestyle: str = "-",
    title_color: str | None = None,
) -> None:
    """Draw one compact rounded module with a colored title rule."""

    x, y, width, height = xywh
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.35,rounding_size=1.15",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        joinstyle="round",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.plot(
        [x + 1.0, x + width - 1.0],
        [y + height - 6.1, y + height - 6.1],
        color=edge,
        linewidth=0.7,
        zorder=3,
    )
    ax.text(
        x + 1.2,
        y + height - 3.15,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=title_color or edge,
        va="center",
        ha="left",
        zorder=4,
    )
    if lines:
        ax.text(
            x + 1.25,
            y + height - 8.4,
            "\n".join(lines),
            fontsize=text_size,
            color=COLORS["ink"],
            linespacing=1.38,
            va="top",
            ha="left",
            zorder=4,
        )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    connectionstyle: str = "arc3,rad=0",
    linewidth: float = 1.15,
    linestyle: str = "-",
) -> None:
    """Draw a clean process arrow."""

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def _score_glyph(ax: plt.Axes, x: float, y: float, width: float) -> None:
    """Draw a deliberately schematic pair of normal-score profiles."""

    xs = [x + width * i / 30 for i in range(31)]
    source = [y + 2.4 + 4.0 * (1 - ((i - 11) / 11) ** 2) ** 2 if abs(i - 11) <= 11 else y + 2.4 for i in range(31)]
    target = [y + 2.4 + 3.4 * (1 - ((i - 20) / 9) ** 2) ** 2 if abs(i - 20) <= 9 else y + 2.4 for i in range(31)]
    ax.plot(xs, source, color=COLORS["blue"], linewidth=1.15, zorder=4)
    ax.plot(xs, target, color=COLORS["orange"], linewidth=1.15, linestyle="--", zorder=4)
    ax.vlines(
        [x + width * 11 / 30, x + width * 20 / 30],
        y + 2.1,
        y + 7.3,
        colors=[COLORS["blue"], COLORS["orange"]],
        linestyles=["-", "--"],
        linewidth=0.65,
        zorder=3,
    )
    ax.text(x, y + 0.1, "schematic score profiles", fontsize=5.1, color=COLORS["muted"])


def _draw_double(ax: plt.Axes) -> None:
    """Draw the horizontal double-column framework."""

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 90)
    ax.axis("off")

    ax.text(1, 87.2, "FROZEN DEPLOYMENT CONTEXT", fontsize=6.2, fontweight="bold", color=COLORS["muted"])
    ax.text(32, 87.2, "NOROPCAL", fontsize=7.0, fontweight="bold", color=COLORS["blue"])
    ax.text(107, 87.2, "DECISION", fontsize=6.2, fontweight="bold", color=COLORS["muted"])

    _box(
        ax,
        (1, 57, 27, 27),
        "A. Frozen anomaly detector",
        ("detector and representation frozen", "score function frozen", "ranking unchanged"),
        face=COLORS["blue_soft"],
        edge=COLORS["blue"],
    )
    _box(
        ax,
        (1, 31, 27, 21),
        "B. Available information",
        (r"source-normal statistics; threshold $\tau_s$", "known-normal target images (K = 8)"),
        face=COLORS["green_soft"],
        edge=COLORS["green"],
    )
    _box(
        ax,
        (1, 3, 27, 23),
        "Not available",
        (
            "target anomaly images / labels / masks",
            "target anomaly-score distribution",
            "retraining or fine-tuning",
        ),
        face=COLORS["grey_soft"],
        edge=COLORS["grey"],
        linestyle="--",
        text_size=5.75,
    )

    _box(
        ax,
        (32, 48, 16, 31),
        "1  Target scoring",
        (r"frozen detector $s(\cdot)$", r"$S_s^N$ and $S_t^N$"),
        face=COLORS["panel"],
        edge=COLORS["blue"],
        title_size=6.4,
        text_size=6.4,
    )

    _box(
        ax,
        (52, 48, 17, 31),
        "2  Deformation",
        (r"$m_s=\mathrm{median}(S_s^N)$", r"$m_t=\mathrm{median}(S_t^N)$", r"$\Delta m=m_t-m_s$"),
        face=COLORS["panel"],
        edge=COLORS["teal"],
        title_size=6.4,
        text_size=6.15,
    )
    _score_glyph(ax, 54, 50.3, 13)

    _box(
        ax,
        (73, 42, 16, 42),
        "3A  Selective gate",
        (r"authorize iff $\Delta m \geq c$", r"frozen $c=Q_{50}$", "otherwise abstain"),
        face=COLORS["teal_soft"],
        edge=COLORS["teal"],
        title_size=7.2,
        text_size=6.35,
        linewidth=1.45,
    )
    ax.text(81, 47.3, "selective, not always-on", fontsize=5.4, color=COLORS["teal"], ha="center", fontweight="bold")

    _box(
        ax,
        (93, 61, 12, 23),
        "3B  Actuator",
        ("conservative correction", "shrinkage update", "threshold only"),
        face=COLORS["orange_soft"],
        edge=COLORS["orange"],
        title_size=6.5,
        text_size=5.55,
    )
    ax.text(99, 63.4, r"$\tau_t=\tau_s+0.25\,\Delta m$", fontsize=6.3, color=COLORS["orange"], ha="center", fontweight="bold")

    _box(
        ax,
        (93, 35, 12, 18),
        "Reject / abstain",
        ("retain source threshold",),
        face=COLORS["grey_soft"],
        edge=COLORS["grey"],
        title_size=6.3,
        text_size=5.55,
    )
    ax.text(99, 37.4, r"$\tau_t=\tau_s$", fontsize=6.4, color=COLORS["ink"], ha="center", fontweight="bold")

    _box(
        ax,
        (108, 42, 11, 34),
        "Threshold decision",
        ("authorized: corrected", "rejected: unchanged"),
        face=COLORS["blue_soft"],
        edge=COLORS["blue"],
        title_size=6.15,
        text_size=5.35,
    )
    ax.text(113.5, 46.2, r"anomaly if $s(x)>\tau_t$", fontsize=5.75, color=COLORS["blue"], ha="center", fontweight="bold")

    _arrow(ax, (28.5, 66), (31.2, 66), color=COLORS["blue"])
    _arrow(ax, (28.5, 42), (31.2, 58), color=COLORS["green"], connectionstyle="arc3,rad=-0.12")
    _arrow(ax, (48.7, 63.5), (51.2, 63.5), color=COLORS["teal"])
    _arrow(ax, (69.7, 63.5), (72.2, 63.5), color=COLORS["teal"])
    _arrow(ax, (89.7, 68), (92.2, 72), color=COLORS["orange"], connectionstyle="arc3,rad=-0.10")
    _arrow(ax, (89.7, 57), (92.2, 45), color=COLORS["grey"], connectionstyle="arc3,rad=0.10")
    _arrow(ax, (105.7, 72), (107.2, 65), color=COLORS["orange"], connectionstyle="arc3,rad=0.16")
    _arrow(ax, (105.7, 44), (107.2, 52), color=COLORS["grey"], connectionstyle="arc3,rad=-0.16")

    ax.text(91.3, 76.8, "AUTHORIZE", fontsize=5.2, color=COLORS["orange"], fontweight="bold", ha="right")
    ax.text(91.3, 42.2, "REJECT", fontsize=5.2, color=COLORS["grey"], fontweight="bold", ha="right")
    ax.text(
        60,
        1.0,
        "Normal-only, threshold-only, selective operating-point calibration under acquisition shift",
        fontsize=6.0,
        color=COLORS["muted"],
        ha="center",
        va="bottom",
    )


def _draw_single(ax: plt.Axes) -> None:
    """Draw a stacked single-column version without shrinking the horizontal layout."""

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 182)
    ax.axis("off")

    ax.text(2, 179, "FROZEN DEPLOYMENT CONTEXT", fontsize=6.2, fontweight="bold", color=COLORS["muted"])
    _box(
        ax,
        (2, 145, 46, 30),
        "A. Frozen detector",
        ("detector / representation frozen", "score function frozen", "ranking unchanged"),
        face=COLORS["blue_soft"],
        edge=COLORS["blue"],
        text_size=5.75,
    )
    _box(
        ax,
        (52, 145, 46, 30),
        "B. Available information",
        (r"source-normal statistics; $\tau_s$", "known-normal target images", "K = 8"),
        face=COLORS["green_soft"],
        edge=COLORS["green"],
        text_size=5.75,
    )
    _box(
        ax,
        (2, 119, 96, 21),
        "Not available",
        (
            "target anomalies, labels, masks, or anomaly-score distribution",
            "retraining or fine-tuning",
        ),
        face=COLORS["grey_soft"],
        edge=COLORS["grey"],
        linestyle="--",
        text_size=5.7,
    )

    _box(
        ax,
        (10, 94, 80, 20),
        "1  Target scoring",
        (r"known-normal target images $\rightarrow$ frozen $s(\cdot)$ $\rightarrow$ $S_t^N$; source scores $S_s^N$ retained" ,),
        face=COLORS["panel"],
        edge=COLORS["blue"],
        title_size=6.4,
        text_size=5.75,
    )
    _box(
        ax,
        (10, 69, 80, 20),
        "2  Normal-score deformation",
        (r"$m_s=\mathrm{median}(S_s^N)$; $m_t=\mathrm{median}(S_t^N)$; $\Delta m=m_t-m_s$",),
        face=COLORS["panel"],
        edge=COLORS["teal"],
        text_size=5.75,
    )
    _box(
        ax,
        (4, 39, 44, 25),
        "3A  Selective gate",
        (r"authorize iff $\Delta m\geq c$", r"frozen $c=Q_{50}$", "otherwise abstain"),
        face=COLORS["teal_soft"],
        edge=COLORS["teal"],
        text_size=5.7,
        linewidth=1.4,
    )
    _box(
        ax,
        (54, 47, 42, 17),
        "Authorize calibration",
        (r"$\tau_t=\tau_s+0.25\,\Delta m$", "threshold only"),
        face=COLORS["orange_soft"],
        edge=COLORS["orange"],
        title_size=6.1,
        text_size=5.6,
    )
    _box(
        ax,
        (54, 24, 42, 17),
        "Reject / abstain",
        (r"$\tau_t=\tau_s$", "threshold unchanged"),
        face=COLORS["grey_soft"],
        edge=COLORS["grey"],
        title_size=6.1,
        text_size=5.6,
    )
    _box(
        ax,
        (10, 3, 80, 16),
        "Final decision rule",
        (r"anomaly if $s(x)>\tau_t$; detector and ranking remain frozen",),
        face=COLORS["blue_soft"],
        edge=COLORS["blue"],
        text_size=5.75,
    )

    _arrow(ax, (25, 145), (25, 141), color=COLORS["blue"])
    _arrow(ax, (75, 145), (75, 141), color=COLORS["blue"])
    ax.plot([25, 99], [141, 141], color=COLORS["blue"], lw=0.9, solid_capstyle="round")
    ax.plot([99, 99], [141, 104], color=COLORS["blue"], lw=0.9, solid_capstyle="round")
    _arrow(ax, (99, 104), (90.5, 104), color=COLORS["blue"])
    _arrow(ax, (50, 93.5), (50, 89.5), color=COLORS["teal"])
    _arrow(ax, (50, 68.5), (28, 64.5), color=COLORS["teal"], connectionstyle="arc3,rad=0.08")
    _arrow(ax, (48.7, 55), (53.3, 56), color=COLORS["orange"])
    _arrow(ax, (48.7, 48), (53.3, 32), color=COLORS["grey"], connectionstyle="arc3,rad=0.10")
    _arrow(ax, (75, 23.5), (66, 19.5), color=COLORS["grey"], connectionstyle="arc3,rad=-0.10")
    _arrow(ax, (75, 46.5), (70, 19.5), color=COLORS["orange"], connectionstyle="arc3,rad=0.18")


def generate(output_dir: Path) -> list[Path]:
    """Generate double- and single-column variants."""

    apply_ieee_style()
    generated: list[Path] = []

    fig = plt.figure(figsize=(7.16, 3.55), constrained_layout=True)
    _draw_double(fig.add_subplot(111))
    generated.extend(export_figure(fig, output_dir, "Fig1_NorOpCal_Framework"))

    fig_single = plt.figure(figsize=(3.50, 6.35), constrained_layout=True)
    _draw_single(fig_single.add_subplot(111))
    generated.extend(export_figure(fig_single, output_dir, "Fig1_NorOpCal_Framework_singlecol"))
    return generated


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = generate(args.output_dir)
    LOGGER.info("Generated %d Fig. 1 exports", len(paths))


if __name__ == "__main__":
    main()
