#!/usr/bin/env python3
"""Baseline vs graph relative-height (weak sigma) on several test trips (no GT: use wMSE_pr)."""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_TEST_TRIPS = [
    "test/2020-12-11-19-30-us-ca-mtv-e/pixel4xl",
    "test/2021-09-14-20-32-us-ca-mtv-k/pixel4",
    "test/2022-04-27-21-55-us-ca-ebf-ww/mi8",
]


def trips_from_settings_csv(path: Path, limit: int) -> list[str]:
    out: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            course = (row.get("Course") or "").strip()
            phone = (row.get("Phone") or "").strip()
            if not course or not phone:
                continue
            out.append(f"test/{course}/{phone}")
            if len(out) >= limit:
                break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--max-epochs", type=int, default=400, help="cap epochs per trip for runtime")
    p.add_argument(
        "--settings-csv",
        type=Path,
        default=None,
        help="if set, load trips as test/{Course}/{Phone} from this file (e.g. settings_test.csv)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=8,
        help="max rows to take from --settings-csv (ignored when --settings-csv not set)",
    )
    p.add_argument(
        "--graph-sigma-m",
        type=float,
        default=2.0,
        help="graph relative-height sigma (m); previous sweep showed 2.0 least harmful on train",
    )
    args = p.parse_args()

    if args.settings_csv is not None:
        trip_list = trips_from_settings_csv(args.settings_csv, args.limit)
    else:
        trip_list = DEFAULT_TEST_TRIPS

    base = BridgeConfig(
        position_source="fgo",
        chunk_epochs=200,
        motion_sigma_m=0.3,
        clock_drift_sigma_m=1.0,
        fgo_iters=8,
        use_vd=True,
        multi_gnss=True,
        tdcp_enabled=False,
    )

    pairs = [
        ("baseline (no graph)", replace(base, graph_relative_height=False)),
        (f"graph σ={args.graph_sigma_m:g} m", replace(base, graph_relative_height=True, relative_height_sigma_m=args.graph_sigma_m)),
    ]

    print(f"data_root={args.data_root}")
    print(f"max_epochs={args.max_epochs} (cap)")
    print(f"trip_source={'settings_csv ' + str(args.settings_csv) if args.settings_csv else 'built-in default (3)'}")
    print(f"trips={len(trip_list)}")
    print("-" * 95)

    graph_wins = 0
    base_wins = 0
    ties = 0

    for trip in trip_list:
        td = args.data_root / trip
        if not td.is_dir():
            print(f"{trip:50s}  SKIP (missing dir)")
            continue
        print(f"\n## {trip}")
        mse: list[float] = []
        for label, cfg in pairs:
            t0 = time.perf_counter()
            r = validate_raw_gsdc2023_trip(args.data_root, trip, max_epochs=args.max_epochs, config=cfg)
            dt = time.perf_counter() - t0
            mse.append(float(r.fgo_mse_pr))
            gt = "yes" if r.metrics_fgo is not None else "no"
            if r.metrics_fgo is not None:
                m = r.metrics_fgo
                extra = f"  RMS2D={m['rms_2d']:.3f}  RMS3D={m['rms_3d']:.3f}"
            else:
                extra = ""
            print(
                f"  {label:28s}  wMSE_pr={r.fgo_mse_pr:.4f}  iters={r.fgo_iters}  GT={gt}{extra}  ({dt:.1f}s)"
            )
        if len(mse) == 2:
            if mse[1] < mse[0] - 1e-9:
                graph_wins += 1
            elif mse[0] < mse[1] - 1e-9:
                base_wins += 1
            else:
                ties += 1
    print("-" * 95)
    print(
        f"Summary (lower wMSE_pr better): graph_win={graph_wins}  baseline_win={base_wins}  tie={ties}  "
        f"(among trips with both runs)"
    )


if __name__ == "__main__":
    main()
