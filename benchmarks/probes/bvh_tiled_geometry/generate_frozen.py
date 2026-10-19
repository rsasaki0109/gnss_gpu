"""Run the unchanged D294 projector against its immutable original source."""

from pathlib import Path

import generate

generate.ROOT = Path(__file__).resolve().parents[1] / "bvh_baseline_20261019"

if __name__ == "__main__":
    generate.main()
