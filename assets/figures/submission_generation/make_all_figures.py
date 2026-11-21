"""Generate all NorOpCal IEEE TIM figure assets without modifying the manuscript."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import make_fig1
import make_fig2
import make_fig3


LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Generate all registered assets into one output directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    paths = [
        *make_fig1.generate(args.output_dir),
        *make_fig2.generate(args.output_dir),
        *make_fig3.generate(args.output_dir),
    ]
    LOGGER.info("Generated %d vector/raster publication exports", len(paths))


if __name__ == "__main__":
    main()
