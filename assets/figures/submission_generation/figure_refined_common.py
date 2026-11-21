"""Shared IEEE TIM styling and export helpers for refined NorOpCal figures."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


DOUBLE_COLUMN_WIDTH: Final[float] = 7.16
PNG_DPI: Final[int] = 600

PALETTE: Final[dict[str, str]] = {
    "ink": "#242424",
    "dark_gray": "#555555",
    "medium_gray": "#858585",
    "light_gray": "#D6D6D6",
    "very_light_gray": "#F2F2F2",
    "boundary": "#747474",
    "principal": "#2F5597",
    "principal_light": "#AAB8D3",
    "unsafe_tint": "#FBF8F6",
    "muted_orange": "#C55A3C",
    "white": "#FFFFFF",
}


def apply_refined_ieee_style() -> None:
    """Apply the shared Times New Roman visual grammar for refined figures."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.75,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.25,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.facecolor": PALETTE["white"],
            "savefig.transparent": False,
        }
    )


def export_refined(fig: Figure, output_dir: Path, basename: str) -> list[Path]:
    """Export SVG/PDF/PNG plus grayscale and 90%-scale QA previews."""

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{basename}.svg"
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    grayscale_path = output_dir / f"{basename}_grayscale.png"
    scale_path = output_dir / f"{basename}_90pct_preview.png"

    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=PNG_DPI)
    with Image.open(png_path) as image:
        image.convert("L").save(grayscale_path, dpi=(PNG_DPI, PNG_DPI))
        scaled_size = (round(image.width * 0.9), round(image.height * 0.9))
        image.resize(scaled_size, Image.Resampling.LANCZOS).save(
            scale_path,
            dpi=(PNG_DPI, PNG_DPI),
        )
    plt.close(fig)
    return [svg_path, pdf_path, png_path, grayscale_path, scale_path]


def minimal_axes(ax: plt.Axes, *, keep_left: bool = True) -> None:
    """Remove non-essential borders and grids without changing data geometry."""

    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_linewidth(0.8)
    if keep_left:
        ax.spines["left"].set_linewidth(0.8)
