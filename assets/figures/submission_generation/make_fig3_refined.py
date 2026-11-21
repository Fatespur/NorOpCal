"""Create the refined IEEE TIM gate and correctability diagnostic figure."""

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
DATA_PATH: Final[Path] = PACKAGE_DIR / "data" / "fig3_refined_evidence.csv"


@dataclass(frozen=True)
class EvidencePoint:
    """One frozen diagnostic estimate and its optional stored interval."""

    estimate: float
    lower: float | None
    upper: float | None
    interval_type: str


def _optional_float(value: str) -> float | None:
    """Parse an optional numeric field without deriving missing evidence."""

    return float(value) if value.strip() else None


def _load_evidence(path: Path) -> dict[str, EvidencePoint]:
    """Load the audit snapshot used by the two panels."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["quantity"]: EvidencePoint(
            estimate=float(row["estimate"]),
            lower=_optional_float(row["lower"]),
            upper=_optional_float(row["upper"]),
            interval_type=row["interval_type"],
        )
        for row in rows
    }


def _plot_gate(ax: plt.Axes, point: EvidencePoint) -> None:
    """Show the stored randomization interval without inventing a distribution."""

    if point.lower is None or point.upper is None:
        raise ValueError("Gate-advantage interval is required by the refined figure contract")

    minimal_axes(ax, keep_left=False)
    ax.set_title("(a) Gate informativeness", loc="left", pad=7)
    ax.set_xlim(-0.12, 2.18)
    ax.set_ylim(-0.48, 0.48)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.axvline(
        0.0,
        color=PALETTE["boundary"],
        linestyle=(0, (4, 3)),
        linewidth=0.8,
        zorder=1,
    )
    ax.text(
        0.025,
        0.39,
        "No advantage",
        color=PALETTE["dark_gray"],
        fontsize=7.1,
        ha="left",
        va="top",
    )

    ax.hlines(0.0, point.lower, point.upper, color=PALETTE["principal"], linewidth=1.35, zorder=3)
    ax.vlines([point.lower, point.upper], -0.075, 0.075, color=PALETTE["principal"], linewidth=1.0, zorder=3)
    ax.plot(
        point.estimate,
        0.0,
        marker="D",
        markersize=6.1,
        linestyle="None",
        markerfacecolor=PALETTE["principal"],
        markeredgecolor=PALETTE["ink"],
        markeredgewidth=0.7,
        zorder=4,
    )
    ax.text(
        point.estimate,
        0.17,
        f"+{point.estimate:.2f} pp",
        color=PALETTE["principal"],
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.text(
        (point.lower + point.upper) / 2.0,
        -0.17,
        f"[{point.lower:.2f}, {point.upper:.2f}] pp",
        color=PALETTE["dark_gray"],
        fontsize=7.2,
        ha="center",
        va="top",
    )
    ax.text(
        0.98,
        0.97,
        "1,000 preregistered random-control seeds",
        transform=ax.transAxes,
        color=PALETTE["dark_gray"],
        fontsize=7.0,
        ha="right",
        va="top",
    )
    ax.set_xlabel("Utility gain over random abstention (pp)")


def _plot_correctability(
    ax: plt.Axes,
    full: EvidencePoint,
    k8: EvidencePoint,
) -> None:
    """Plot one real bootstrap interval and one point-only K=8 estimate."""

    if full.lower is None or full.upper is None:
        raise ValueError("Full-population AUC interval is required by the refined figure contract")
    if k8.lower is not None or k8.upper is not None:
        raise ValueError("K=8 must remain point-only unless a stored interval is available")

    minimal_axes(ax)
    ax.set_title("(b) Correctability diagnostic", loc="left", pad=7)
    ax.set_xlim(0.48, 0.95)
    ax.set_ylim(-0.95, 1.55)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticks([1.0, 0.0], labels=["Full population", r"$K=8$ subset"])
    ax.axvline(
        0.5,
        color=PALETTE["boundary"],
        linestyle=(0, (4, 3)),
        linewidth=0.8,
        zorder=1,
    )
    ax.text(
        0.505,
        1.43,
        "Chance",
        color=PALETTE["dark_gray"],
        fontsize=7.1,
        ha="left",
        va="top",
    )

    ax.hlines(1.0, full.lower, full.upper, color=PALETTE["medium_gray"], linewidth=1.2, zorder=2)
    ax.vlines([full.lower, full.upper], 0.91, 1.09, color=PALETTE["medium_gray"], linewidth=0.9, zorder=2)
    ax.plot(
        full.estimate,
        1.0,
        marker="o",
        markersize=5.1,
        linestyle="None",
        markerfacecolor=PALETTE["medium_gray"],
        markeredgecolor=PALETTE["dark_gray"],
        markeredgewidth=0.65,
        zorder=3,
    )
    ax.text(
        0.942,
        1.22,
        f"{full.estimate:.3f} [{full.lower:.3f}, {full.upper:.3f}]",
        color=PALETTE["dark_gray"],
        fontsize=7.1,
        ha="right",
        va="bottom",
    )

    ax.plot(
        k8.estimate,
        0.0,
        marker="D",
        markersize=5.8,
        linestyle="None",
        markerfacecolor=PALETTE["principal"],
        markeredgecolor=PALETTE["ink"],
        markeredgewidth=0.7,
        zorder=3,
    )
    ax.text(
        k8.estimate + 0.025,
        0.0,
        f"{k8.estimate:.3f}\n(point estimate)",
        color=PALETTE["principal"],
        fontsize=7.1,
        fontweight="normal",
        ha="left",
        va="center",
        linespacing=0.95,
    )
    ax.text(
        0.515,
        -0.58,
        "Full: category-aware bootstrap 95% interval",
        color=PALETTE["dark_gray"],
        fontsize=6.35,
        ha="left",
        va="center",
    )
    ax.text(
        0.515,
        -0.76,
        r"$K=8$: no stored interval; point estimate only",
        color=PALETTE["dark_gray"],
        fontsize=6.35,
        ha="left",
        va="center",
    )
    ax.set_xlabel("Correctability AUC")


def draw() -> plt.Figure:
    """Assemble the compact double-column refined Figure 3."""

    evidence = _load_evidence(DATA_PATH)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.28),
        gridspec_kw={"width_ratios": [0.98, 1.02]},
        constrained_layout=False,
    )
    _plot_gate(axes[0], evidence["gate_advantage_pp"])
    _plot_correctability(
        axes[1],
        evidence["correctability_auc_full"],
        evidence["correctability_auc_k8"],
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.86, wspace=0.34)
    return fig


def generate(output_dir: Path) -> list[Path]:
    """Generate the refined Figure 3 without replacing prior assets."""

    apply_refined_ieee_style()
    return export_refined(
        draw(),
        output_dir,
        "Fig3_Gate_Informativeness_and_Correctability_REFINED",
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    LOGGER.info("Generated %d refined Figure 3 files", len(generate(args.output_dir)))


if __name__ == "__main__":
    main()
