"""Create the refined IEEE TIM utility-safety and detector-dependence figure."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt

from figure_refined_common import PALETTE, apply_refined_ieee_style, export_refined, minimal_axes


LOGGER = logging.getLogger(__name__)
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PACKAGE_DIR / "data"
SAFETY_BUDGET_PP: Final[float] = 2.0


@dataclass(frozen=True)
class OperatingPoint:
    """One frozen utility-safety operating point."""

    name: str
    delta_fnr_pp: float
    delta_ba_pp: float
    safety_class: str


def _load_points(path: Path, name_key: str) -> list[OperatingPoint]:
    """Read frozen coordinates without deriving or modifying values."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        OperatingPoint(
            name=row[name_key],
            delta_fnr_pp=float(row["delta_fnr_pp"]),
            delta_ba_pp=float(row["delta_ba_pp"]),
            safety_class=row["safety_class"],
        )
        for row in rows
    ]


def _prepare_axis(ax: plt.Axes, *, setting_note: str | None = None) -> None:
    """Set shared geometry and the frozen safety boundary."""

    minimal_axes(ax)
    ax.set_xlim(-1.0, 8.45)
    ax.set_ylim(-1.55, 9.40)
    ax.axvspan(SAFETY_BUDGET_PP, 8.45, color=PALETTE["unsafe_tint"], zorder=0)
    ax.axvline(
        SAFETY_BUDGET_PP,
        color=PALETTE["boundary"],
        linestyle=(0, (4, 3)),
        linewidth=0.8,
        zorder=1,
    )
    ax.axhline(0.0, color=PALETTE["light_gray"], linewidth=0.65, zorder=1)
    ax.text(
        SAFETY_BUDGET_PP + 1.20,
        9.10,
        r"Safety budget: $\Delta$FNR $\leq$ 2 pp",
        color=PALETTE["dark_gray"],
        fontsize=7.05,
        ha="left",
        va="top",
    )
    region_y = 8.24
    ax.text(
        -0.82,
        region_y,
        "Safety-feasible region",
        color=PALETTE["dark_gray"],
        fontsize=6.75,
        fontstyle="italic",
        ha="left",
        va="top",
    )
    ax.text(
        4.55,
        region_y,
        "Outside safety budget",
        color=PALETTE["dark_gray"],
        fontsize=6.75,
        fontstyle="italic",
        ha="left",
        va="top",
    )
    if setting_note is not None:
        ax.text(
            -0.82,
            8.80,
            setting_note,
            color=PALETTE["dark_gray"],
            fontsize=6.6,
            fontweight="normal",
            ha="left",
            va="top",
        )
    ax.set_xlabel(r"Mean $\Delta$FNR (pp)")
    ax.set_ylabel(r"Mean $\Delta$BA (pp)")
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 2, 4, 6, 8])


def _label(
    ax: plt.Axes,
    point: OperatingPoint,
    offset: tuple[float, float],
    *,
    principal: bool = False,
    label_text: str | None = None,
) -> None:
    """Add a compact direct label with a quiet leader line."""

    ax.annotate(
        label_text or point.name,
        xy=(point.delta_fnr_pp, point.delta_ba_pp),
        xytext=offset,
        textcoords="offset points",
        color=PALETTE["principal"] if principal else PALETTE["ink"],
        fontsize=7.6,
        fontweight="bold" if principal else "normal",
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["medium_gray"],
            "linewidth": 0.55,
            "shrinkA": 2.0,
            "shrinkB": 3.0,
        },
        zorder=5,
    )


def _plot_policy(ax: plt.Axes, points: list[OperatingPoint]) -> None:
    """Plot policy-level points with one marker grammar per scientific role."""

    offsets = {
        "NoAdapt": (8, -9),
        "B1": (8, -7),
        "B2": (-10, 8),
        "B3": (8, 9),
        "B4": (-10, -9),
        "NorOpCal": (10, -12),
    }
    for point in points:
        if point.name == "NorOpCal":
            ax.plot(
                point.delta_fnr_pp,
                point.delta_ba_pp,
                marker="D",
                markersize=6.4,
                linestyle="None",
                markerfacecolor=PALETTE["principal"],
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.75,
                zorder=4,
            )
            _label(ax, point, offsets[point.name], principal=True)
            ax.annotate(
                "Frozen selected policy",
                xy=(point.delta_fnr_pp, point.delta_ba_pp),
                xytext=(10, -25),
                textcoords="offset points",
                color=PALETTE["principal"],
                fontsize=6.45,
                ha="left",
                va="top",
                zorder=5,
            )
        elif point.name == "NoAdapt":
            ax.plot(
                point.delta_fnr_pp,
                point.delta_ba_pp,
                marker="o",
                markersize=4.8,
                linestyle="None",
                markerfacecolor=PALETTE["white"],
                markeredgecolor=PALETTE["medium_gray"],
                markeredgewidth=0.9,
                zorder=4,
            )
            _label(ax, point, offsets[point.name])
        else:
            ax.plot(
                point.delta_fnr_pp,
                point.delta_ba_pp,
                marker="o",
                markersize=4.8,
                linestyle="None",
                markerfacecolor=PALETTE["medium_gray"],
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.6,
                zorder=4,
            )
            _label(ax, point, offsets[point.name], label_text="B3 (bounded-only)" if point.name == "B3" else None)
    ax.set_title("(a) Utility–safety operating points", loc="left", pad=7)


def _plot_detectors(ax: plt.Axes, points: list[OperatingPoint]) -> None:
    """Plot detector dependence with direct labels and no redundant legend."""

    offsets = {"DINOv2": (5, 8), "PatchCore": (10, -7), "EfficientAD": (-10, 8)}
    for point in points:
        principal = point.name == "DINOv2"
        ax.plot(
            point.delta_fnr_pp,
            point.delta_ba_pp,
            marker="D" if principal else "o",
            markersize=6.4 if principal else 4.8,
            linestyle="None",
            markerfacecolor=PALETTE["principal"] if principal else PALETTE["medium_gray"],
            markeredgecolor=PALETTE["ink"] if principal else PALETTE["dark_gray"],
            markeredgewidth=0.75 if principal else 0.6,
            zorder=4,
        )
        _label(ax, point, offsets[point.name], principal=principal)
    ax.set_title("(b) Detector dependence", loc="left", pad=7)


def draw() -> plt.Figure:
    """Assemble the double-column refined Figure 2."""

    policy = _load_points(DATA_DIR / "fig2_policy_points.csv", "policy")
    detectors = _load_points(DATA_DIR / "fig2_detector_points.csv", "detector")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.18), constrained_layout=False)
    _prepare_axis(axes[0], setting_note="Normal-only deployable setting")
    _prepare_axis(axes[1])
    _plot_policy(axes[0], policy)
    _plot_detectors(axes[1], detectors)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.89, wspace=0.27)
    return fig


def generate(output_dir: Path) -> list[Path]:
    """Generate the refined Figure 2 candidates without replacing prior assets."""

    apply_refined_ieee_style()
    return export_refined(
        draw(),
        output_dir,
        "Fig2_Utility_Safety_and_Detector_Specificity_REFINED",
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    LOGGER.info("Generated %d refined Figure 2 files", len(generate(args.output_dir)))


if __name__ == "__main__":
    main()
