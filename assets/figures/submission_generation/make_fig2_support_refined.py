"""Create two independent IEEE TIM frozen-setting result figures."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt

from figure_refined_common import (
    PALETTE,
    apply_refined_ieee_style,
    export_refined,
    minimal_axes,
)


LOGGER = logging.getLogger(__name__)
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_PATH: Final[Path] = PACKAGE_DIR / "data" / "fig2_support_points.csv"
SAFETY_BUDGET_PP: Final[float] = 2.0
SUPPORT_ORDER: Final[tuple[str, ...]] = ("K = 4", "K = 8", "Full")
EXPECTED_POINTS: Final[dict[str, tuple[float, float]]] = {
    "K = 4": (3.7593795093795, 2.85470085470085),
    "K = 8": (3.22626262626262, 0.244444444444444),
    "Full": (3.71586746586746, 2.82051282051282),
}
EXPECTED_INTERVALS: Final[
    dict[str, tuple[tuple[float, float], tuple[float, float]]]
] = {
    "K = 8": (
        (2.69696969696970, 3.75555555555556),
        (0.0, 0.488888888888889),
    )
}
EXPECTED_POPULATIONS: Final[dict[str, tuple[int, int]]] = {
    "K = 4": (26, 5),
    "K = 8": (10, 2),
    "Full": (26, 5),
}
FIGURE_SIZE: Final[tuple[float, float]] = (3.5, 2.85)


@dataclass(frozen=True)
class SupportPoint:
    """One frozen DINOv2 support-setting operating point."""

    support: str
    n_units: int
    n_categories: int
    delta_ba_pp: float
    delta_ba_ci_lo_pp: float | None
    delta_ba_ci_hi_pp: float | None
    delta_fnr_pp: float
    delta_fnr_ci_lo_pp: float | None
    delta_fnr_ci_hi_pp: float | None
    interval_status: str
    safety_class: str
    support_repeats: int


def _optional_float(value: str) -> float | None:
    """Parse a possibly empty numeric cell without inventing a value."""

    return float(value) if value.strip() else None


def _load_points(path: Path = DATA_PATH) -> list[SupportPoint]:
    """Load and validate the frozen support-setting snapshot."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = [
        SupportPoint(
            support=row["support"],
            n_units=int(row["n_units"]),
            n_categories=int(row["n_categories"]),
            delta_ba_pp=float(row["delta_ba_pp"]),
            delta_ba_ci_lo_pp=_optional_float(row["delta_ba_ci_lo_pp"]),
            delta_ba_ci_hi_pp=_optional_float(row["delta_ba_ci_hi_pp"]),
            delta_fnr_pp=float(row["delta_fnr_pp"]),
            delta_fnr_ci_lo_pp=_optional_float(row["delta_fnr_ci_lo_pp"]),
            delta_fnr_ci_hi_pp=_optional_float(row["delta_fnr_ci_hi_pp"]),
            interval_status=row["interval_status"],
            safety_class=row["safety_class"],
            support_repeats=int(row["support_repeats"]),
        )
        for row in rows
    ]
    if tuple(point.support for point in points) != SUPPORT_ORDER:
        raise ValueError(f"Unexpected support order: {[point.support for point in points]}")
    for point in points:
        expected_ba, expected_fnr = EXPECTED_POINTS[point.support]
        if abs(point.delta_ba_pp - expected_ba) > 1e-12:
            raise ValueError(f"Frozen delta BA mismatch for {point.support}")
        if abs(point.delta_fnr_pp - expected_fnr) > 1e-12:
            raise ValueError(f"Frozen delta FNR mismatch for {point.support}")
        has_ba_interval = point.delta_ba_ci_lo_pp is not None
        has_fnr_interval = point.delta_fnr_ci_lo_pp is not None
        if has_ba_interval != (point.delta_ba_ci_hi_pp is not None):
            raise ValueError(f"Incomplete delta BA interval for {point.support}")
        if has_fnr_interval != (point.delta_fnr_ci_hi_pp is not None):
            raise ValueError(f"Incomplete delta FNR interval for {point.support}")
        if point.support != "K = 8" and (has_ba_interval or has_fnr_interval):
            raise ValueError(f"Unsupported stored interval for {point.support}")
        expected_units, expected_categories = EXPECTED_POPULATIONS[point.support]
        if (point.n_units, point.n_categories) != (
            expected_units,
            expected_categories,
        ):
            raise ValueError(f"Frozen population mismatch for {point.support}")
        if point.support == "K = 8":
            expected_ba_ci, expected_fnr_ci = EXPECTED_INTERVALS[point.support]
            observed_ba_ci = (point.delta_ba_ci_lo_pp, point.delta_ba_ci_hi_pp)
            observed_fnr_ci = (point.delta_fnr_ci_lo_pp, point.delta_fnr_ci_hi_pp)
            if observed_ba_ci != expected_ba_ci:
                raise ValueError("Frozen delta BA interval mismatch for K = 8")
            if observed_fnr_ci != expected_fnr_ci:
                raise ValueError("Frozen delta FNR interval mismatch for K = 8")
    return points


def _prepare_axis(ax: plt.Axes) -> None:
    """Apply shared categorical geometry to one independent result figure."""

    minimal_axes(ax)
    ax.set_xlim(-0.45, 2.45)
    ax.set_xticks(range(len(SUPPORT_ORDER)))
    ax.set_xticklabels(SUPPORT_ORDER)
    ax.tick_params(axis="x", pad=4.0)
    ax.grid(False)


def _draw_metric_point(
    ax: plt.Axes,
    *,
    x: int,
    point: SupportPoint,
    value: float,
    ci_lo: float | None,
    ci_hi: float | None,
) -> None:
    """Draw one estimate and only its stored interval, when available."""

    principal = point.support == "K = 8"
    color = PALETTE["principal"] if principal else PALETTE["medium_gray"]
    edge = PALETTE["ink"] if principal else PALETTE["dark_gray"]
    marker = "D" if principal else "o"
    size = 6.8 if principal else 5.0
    if ci_lo is not None and ci_hi is not None:
        ax.errorbar(
            x,
            value,
            yerr=[[value - ci_lo], [ci_hi - value]],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=3.2,
            capthick=0.9,
            zorder=3,
        )
    ax.plot(
        x,
        value,
        marker=marker,
        markersize=size,
        linestyle="None",
        markerfacecolor=color,
        markeredgecolor=edge,
        markeredgewidth=0.75 if principal else 0.6,
        zorder=4,
    )


def _annotate_selected(
    ax: plt.Axes,
    *,
    value: float,
    value_label: str,
    value_offset: tuple[float, float],
    note_offset: tuple[float, float],
) -> None:
    """Label the frozen K=8 estimate without claiming global optimality."""

    ax.annotate(
        value_label,
        xy=(1, value),
        xytext=value_offset,
        textcoords="offset points",
        color=PALETTE["principal"],
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["principal_light"],
            "linewidth": 0.55,
            "shrinkA": 2.0,
            "shrinkB": 4.0,
        },
        zorder=5,
    )
    ax.annotate(
        "Frozen operating point",
        xy=(1, value),
        xytext=note_offset,
        textcoords="offset points",
        color=PALETTE["principal"],
        fontsize=6.1,
        alpha=0.84,
        ha="left",
        va="top",
        zorder=5,
    )


def draw_utility() -> plt.Figure:
    """Assemble the independent utility figure at IEEE single-column size."""

    points = _load_points()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=False)
    _prepare_axis(ax)
    ax.set_ylim(-0.35, 4.55)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.axhline(
        0.0,
        color=PALETTE["boundary"],
        linestyle=(0, (3, 3)),
        linewidth=0.75,
        zorder=1,
    )
    ax.text(
        -0.38,
        0.10,
        "No utility gain",
        color=PALETTE["medium_gray"],
        fontsize=6.35,
        ha="left",
        va="bottom",
    )
    for x, point in enumerate(points):
        _draw_metric_point(
            ax,
            x=x,
            point=point,
            value=point.delta_ba_pp,
            ci_lo=point.delta_ba_ci_lo_pp,
            ci_hi=point.delta_ba_ci_hi_pp,
        )
    _annotate_selected(
        ax,
        value=points[1].delta_ba_pp,
        value_label="+3.23 pp",
        value_offset=(13, 8),
        note_offset=(13, -8),
    )
    ax.set_ylabel(r"Mean $\Delta$BA (pp)")
    ax.set_title("Utility across frozen support settings", loc="left", pad=7)
    fig.subplots_adjust(left=0.19, right=0.965, bottom=0.22, top=0.86)
    return fig


def draw_safety() -> plt.Figure:
    """Assemble the independent safety figure at IEEE single-column size."""

    points = _load_points()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=False)
    _prepare_axis(ax)
    ax.set_ylim(-0.35, 3.55)
    ax.set_yticks([0, 1, 2, 3])
    ax.axhspan(
        SAFETY_BUDGET_PP,
        3.55,
        color=PALETTE["unsafe_tint"],
        alpha=0.65,
        zorder=0,
    )
    ax.axhline(
        SAFETY_BUDGET_PP,
        color=PALETTE["boundary"],
        linestyle=(0, (4, 3)),
        linewidth=0.85,
        zorder=1,
    )
    ax.text(
        -0.38,
        SAFETY_BUDGET_PP + 0.10,
        r"Safety budget: $\Delta$FNR $\leq$ 2 pp",
        color=PALETTE["dark_gray"],
        fontsize=6.45,
        ha="left",
        va="bottom",
    )
    ax.text(
        2.36,
        1.55,
        "Safety-feasible region",
        color=PALETTE["medium_gray"],
        fontsize=6.15,
        fontstyle="italic",
        alpha=0.76,
        ha="right",
        va="center",
    )
    ax.text(
        1.70,
        3.37,
        "Outside safety budget",
        color=PALETTE["medium_gray"],
        fontsize=6.15,
        fontstyle="italic",
        alpha=0.76,
        ha="center",
        va="top",
    )
    for x, point in enumerate(points):
        _draw_metric_point(
            ax,
            x=x,
            point=point,
            value=point.delta_fnr_pp,
            ci_lo=point.delta_fnr_ci_lo_pp,
            ci_hi=point.delta_fnr_ci_hi_pp,
        )
    _annotate_selected(
        ax,
        value=points[1].delta_fnr_pp,
        value_label="+0.24 pp",
        value_offset=(13, 8),
        note_offset=(13, -6),
    )
    ax.set_ylabel(r"Mean $\Delta$FNR (pp)")
    ax.set_title("Safety across frozen support settings", loc="left", pad=7)
    fig.subplots_adjust(left=0.19, right=0.965, bottom=0.22, top=0.86)
    return fig


def generate(output_dir: Path) -> list[Path]:
    """Generate two independent frozen-setting figures without manuscript edits."""

    apply_refined_ieee_style()
    outputs = export_refined(
        draw_utility(),
        output_dir,
        "Utility_across_Frozen_Support_Settings_REFINED",
    )
    outputs.extend(
        export_refined(
            draw_safety(),
            output_dir,
            "Safety_across_Frozen_Support_Settings_REFINED",
        )
    )
    return outputs

def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    LOGGER.info("Generated %d independent frozen-setting figure files", len(generate(args.output_dir)))


if __name__ == "__main__":
    main()
