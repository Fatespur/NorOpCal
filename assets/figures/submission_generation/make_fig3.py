"""Generate conservative gate-informativeness and correctability diagnostics."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt

from figure_common import COLORS, apply_ieee_style, export_figure, panel_label


LOGGER = logging.getLogger(__name__)
DATA_PATH: Final[Path] = Path(__file__).resolve().parent / "data" / "fig3_diagnostic_summary.csv"


def _load_summary() -> dict[str, dict[str, str]]:
    with DATA_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["quantity"]: row for row in csv.DictReader(handle)}


def _plot_gate_advantage(ax: plt.Axes, summary: dict[str, dict[str, str]]) -> None:
    row = summary["gate_advantage_pp"]
    estimate = float(row["estimate"])
    lower = float(row["lower"])
    upper = float(row["upper"])
    ax.axhline(0.0, color=COLORS["grey"], linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    ax.errorbar(
        [0],
        [estimate],
        yerr=[[estimate - lower], [upper - estimate]],
        fmt="D",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        markeredgecolor=COLORS["ink"],
        markeredgewidth=0.75,
        markersize=6.0,
        elinewidth=1.5,
        capsize=4.0,
        capthick=1.0,
        zorder=4,
    )
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.12, 2.25)
    ax.set_xticks([0], ["Real selective gate"])
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylabel("Utility gain over coverage-matched\nrandom abstention (pp)")
    ax.set_title("Gate informativeness over random abstention", pad=8)
    ax.text(0.06, estimate + 0.02, f"+{estimate:.2f} pp", color=COLORS["blue"], fontsize=7.2, fontweight="bold", va="center")
    ax.text(0.06, lower - 0.05, f"Randomization interval\n[{lower:.2f}, {upper:.2f}] pp", color=COLORS["muted"], fontsize=5.8, va="top")
    ax.text(0.0, 0.08, "1000 preregistered control seeds", color=COLORS["muted"], fontsize=5.5, ha="center", va="bottom")


def _plot_auc_summary(ax: plt.Axes, summary: dict[str, dict[str, str]]) -> None:
    labels = ["Full population", "K = 8 subset"]
    values = [
        float(summary["correctability_auc_full"]["estimate"]),
        float(summary["correctability_auc_k8"]["estimate"]),
    ]
    y_positions = [1, 0]
    colors = [COLORS["teal"], COLORS["blue"]]
    markers = ["o", "D"]
    ax.axvline(0.5, color=COLORS["grey"], linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    for y_pos, value, color, marker in zip(y_positions, values, colors, markers, strict=True):
        ax.hlines(y_pos, 0.5, value, color=color, linewidth=2.0, zorder=2)
        ax.scatter(value, y_pos, marker=marker, s=48, facecolor=color, edgecolor=COLORS["ink"], linewidth=0.75, zorder=4)
        ax.text(value + 0.012, y_pos, f"{value:.3f}", color=color, fontsize=7.0, fontweight="bold", va="center")
    ax.set_xlim(0.45, 0.90)
    ax.set_ylim(-0.75, 1.75)
    ax.set_yticks(y_positions, labels)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Correctability diagnostic AUC")
    ax.set_title("Correctability diagnostic evidence", pad=8)
    ax.text(0.505, 1.53, "chance", color=COLORS["muted"], fontsize=5.5, va="top")
    ax.text(
        0.675,
        -0.52,
        "Oracle-conditioned diagnostic only\nnot a deployment oracle",
        color=COLORS["muted"],
        fontsize=5.8,
        ha="center",
        va="center",
    )


def _draw(layout: str, summary: dict[str, dict[str, str]]) -> plt.Figure:
    if layout == "double":
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85), constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(3.50, 5.25), constrained_layout=True)
    _plot_gate_advantage(axes[0], summary)
    _plot_auc_summary(axes[1], summary)
    panel_label(axes[0], "a", x=-0.17)
    panel_label(axes[1], "b", x=-0.17)
    return fig


def generate(output_dir: Path) -> list[Path]:
    """Generate the evidence-conservative double- and single-column variants."""

    apply_ieee_style()
    summary = _load_summary()
    generated: list[Path] = []
    generated.extend(export_figure(_draw("double", summary), output_dir, "Fig3_Gate_Informativeness_and_Correctability"))
    generated.extend(export_figure(_draw("single", summary), output_dir, "Fig3_Gate_Informativeness_and_Correctability_singlecol"))
    return generated


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Generated %d Fig. 3 exports", len(generate(args.output_dir)))


if __name__ == "__main__":
    main()
