"""Shared IEEE TIM figure style and export helpers for NorOpCal assets."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


DOUBLE_COLUMN_WIDTH: Final[float] = 7.16
SINGLE_COLUMN_WIDTH: Final[float] = 3.50
PNG_DPI: Final[int] = 600

COLORS: Final[dict[str, str]] = {
    "ink": "#263238",
    "muted": "#65747B",
    "line": "#87969D",
    "grid": "#DCE3E6",
    "panel": "#F7F9FA",
    "blue": "#2E5F85",
    "blue_soft": "#E7EFF5",
    "teal": "#2F7774",
    "teal_soft": "#E5F1F0",
    "green": "#4F7659",
    "green_soft": "#EAF1EB",
    "orange": "#B46A3C",
    "orange_soft": "#F7ECE5",
    "red": "#9C534B",
    "red_soft": "#F5E9E7",
    "grey": "#7C878C",
    "grey_soft": "#EFF2F3",
    "white": "#FFFFFF",
}


def apply_ieee_style() -> None:
    """Apply a restrained IEEE-compatible style with editable vector text."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.75,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["white"],
        }
    )


def export_figure(fig: Figure, output_dir: Path, basename: str) -> list[Path]:
    """Export editable SVG/PDF and a 600-DPI PNG plus grayscale QA preview."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{basename}.{suffix}" for suffix in ("svg", "pdf", "png")]
    fig.savefig(paths[0], format="svg")
    fig.savefig(paths[1], format="pdf")
    fig.savefig(paths[2], format="png", dpi=PNG_DPI)
    with Image.open(paths[2]) as image:
        image.convert("L").save(
            output_dir / f"{basename}_grayscale.png",
            dpi=(PNG_DPI, PNG_DPI),
        )
    plt.close(fig)
    return paths


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.13) -> None:
    """Add an IEEE-style parenthesized panel label."""

    ax.text(
        x,
        1.055,
        f"({label})",
        transform=ax.transAxes,
        color=COLORS["ink"],
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
