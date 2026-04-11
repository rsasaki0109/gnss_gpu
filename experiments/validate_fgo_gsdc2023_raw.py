#!/usr/bin/env python3
"""CLI wrapper for the GSDC2023 raw-data bridge."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import (
    BridgeConfig,
    DEFAULT_ROOT,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    POSITION_SOURCES,
    _build_trip_arrays,
    _export_bridge_outputs,
    _fit_state_with_clock_bias,
    validate_raw_gsdc2023_trip,
)


_DEFAULT_ROOT = DEFAULT_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--trip", type=str, required=True, help="relative trip path under data root")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--start-epoch", type=int, default=0)
    p.add_argument("--motion-sigma-m", type=float, default=3.0)
    p.add_argument("--fgo-iters", type=int, default=8)
    p.add_argument("--signal-type", type=str, default="GPS_L1_CA")
    p.add_argument("--constellation-type", type=int, default=1, help="Kaggle enum; GPS=1")
    p.add_argument("--weight-mode", choices=("sin2el", "cn0"), default="sin2el")
    p.add_argument("--position-source", choices=POSITION_SOURCES, default="baseline")
    p.add_argument("--chunk-epochs", type=int, default=0, help="if >0, solve FGO in chunks of this many epochs")
    p.add_argument(
        "--gated-threshold",
        type=float,
        default=GATED_BASELINE_THRESHOLD_DEFAULT,
        help="baseline_mse_pr threshold for gated source fallback",
    )
    p.add_argument(
        "--export-bridge-dir",
        type=Path,
        default=None,
        help="optional output directory for bridge_positions.csv and bridge_metrics.json",
    )
    args = p.parse_args()

    result = validate_raw_gsdc2023_trip(
        args.data_root,
        args.trip,
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        config=BridgeConfig(
            motion_sigma_m=args.motion_sigma_m,
            fgo_iters=args.fgo_iters,
            signal_type=args.signal_type,
            constellation_type=args.constellation_type,
            weight_mode=args.weight_mode,
            position_source=args.position_source,
            chunk_epochs=args.chunk_epochs,
            gated_baseline_threshold=args.gated_threshold,
        ),
    )
    for line in result.summary_lines():
        print(line)
    if args.export_bridge_dir is not None:
        _export_bridge_outputs(args.export_bridge_dir, result)
        print(f"  bridge out  : {args.export_bridge_dir}")


if __name__ == "__main__":
    main()
