"""Generate the refined IEEE TIM NorOpCal result figures and QA previews."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from make_fig2_support_refined import generate as generate_fig2
from make_fig3_refined import generate as generate_fig3


PACKAGE_DIR = Path(__file__).resolve().parent


def main() -> None:
    """CLI entry point for deterministic refined-figure generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    outputs = generate_fig2(args.output_dir) + generate_fig3(args.output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
