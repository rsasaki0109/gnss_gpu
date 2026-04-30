#!/usr/bin/env python3
"""Augment a PPC window CSV with curated solver-state runtime features.

Reads a window CSV produced by the PPC augmentation pipeline, validates that
the curated solver-state columns are present, sanitises non-finite values to
a neutral default, and writes a copy that opt-in research training scripts
can consume.  The output schema is the input schema with the six curated
columns rewritten in place; all other columns pass through unchanged.

The default product training and inference paths
(``experiments/train_ppc_solver_transition_surrogate_stack.py``,
``experiments/product_inference_model.py``) do not invoke this script.  See
``experiments/solver_state_wrapper.py`` for the lifted-contract caveat about
target leakage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from solver_state_wrapper import SolverStateWrapper


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--neutral-value",
        type=float,
        default=0.0,
        help="Replacement for non-finite values in curated columns (default: 0.0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)

    df = pd.read_csv(args.input_csv)
    wrapper = SolverStateWrapper()
    curated = wrapper.curate(df, neutral_value=args.neutral_value)

    out = df.copy()
    for col in wrapper.runtime_feature_columns():
        out[col] = curated[col].to_numpy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(
        f"wrote: {args.output_csv} ({len(out)} rows, "
        f"{len(wrapper.runtime_feature_columns())} curated columns sanitised)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
