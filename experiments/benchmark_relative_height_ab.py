#!/usr/bin/env python3
"""A/B: postprocess vs graph relative-height vs both (+ optional sigma sweep)."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import (
    BridgeConfig,
    DEFAULT_ROOT,
    validate_raw_gsdc2023_trip,
)


def _row(label: str, r) -> str:
    m = r.metrics_fgo
    if m is None:
        return f"{label:32s}  truth N/A"
    return (
        f"{label:32s}  RMS2D={m['rms_2d']:.3f}m  RMS3D={m['rms_3d']:.3f}m  "
        f"P50={m['p50']:.3f}  P95={m['p95']:.3f}  wMSE_pr={r.fgo_mse_pr:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--trip",
        type=str,
        default="train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4",
    )
    p.add_argument("--max-epochs", type=int, default=-1, help="use all epochs when <=0")
    args = p.parse_args()

    base = BridgeConfig(
        position_source="fgo",
        chunk_epochs=200,
        motion_sigma_m=0.3,
        clock_drift_sigma_m=1.0,
        fgo_iters=8,
        use_vd=True,
        multi_gnss=True,
        tdcp_enabled=False,
        apply_relative_height=False,
        graph_relative_height=False,
        relative_height_sigma_m=0.5,
    )

    runs: list[tuple[str, BridgeConfig]] = [
        ("A neither (off/off)", replace(base)),
        ("B postprocess only (smoothing)", replace(base, apply_relative_height=True)),
        ("C graph σ=0.5", replace(base, graph_relative_height=True, relative_height_sigma_m=0.5)),
        ("D graph σ=1.0", replace(base, graph_relative_height=True, relative_height_sigma_m=1.0)),
        ("E graph σ=2.0", replace(base, graph_relative_height=True, relative_height_sigma_m=2.0)),
        ("F both (post + graph σ=1.0)", replace(base, apply_relative_height=True, graph_relative_height=True, relative_height_sigma_m=1.0)),
    ]

    print(f"data_root={args.data_root}")
    print(f"trip={args.trip}")
    print(f"max_epochs={args.max_epochs}")
    print("-" * 100)

    for label, cfg in runs:
        t0 = time.perf_counter()
        result = validate_raw_gsdc2023_trip(
            args.data_root,
            args.trip,
            max_epochs=args.max_epochs,
            config=cfg,
        )
        dt = time.perf_counter() - t0
        print(f"{_row(label, result)}  ({dt:.1f}s)")
    print("-" * 100)

    print("Notes: FGO metrics vs ground_truth.csv. Graph factor uses Kaggle WLS for loop detection.")


if __name__ == "__main__":
    main()
