"""CLI for the unified scenario engine (:mod:`gnss_gpu.scenario`).

Example::

    python -m gnss_gpu.scenario_cli --nav base.nav \\
        --lat 35.619 --lon 139.779 --alt 30.0 \\
        --start 2021-01-01T00:00:00 --duration 60 --step 1 \\
        --constellations G --out out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from gnss_gpu.scenario import ScenarioConfig, run_scenario


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gnss_gpu.scenario_cli",
        description="Simulate per-epoch GNSS observables from a receiver "
        "location, a time window, and a RINEX NAV file (with an optional "
        "PLATEAU city mesh for LOS/NLOS + multipath).",
    )
    p.add_argument("--nav", required=True, help="RINEX NAV file path")
    p.add_argument("--lat", type=float, required=True, help="receiver latitude [deg]")
    p.add_argument("--lon", type=float, required=True, help="receiver longitude [deg]")
    p.add_argument("--alt", type=float, required=True, help="receiver altitude [m]")
    p.add_argument("--start", required=True, help="ISO-8601 start time, e.g. 2021-01-01T00:00:00")
    p.add_argument("--duration", type=float, required=True, help="scenario duration [s]")
    p.add_argument("--step", type=float, default=1.0, help="epoch step [s] (default: 1.0)")
    p.add_argument(
        "--constellations", default="G",
        help="constellation letters, e.g. 'G', 'GEJ', or 'G,E,J' (default: G)",
    )
    p.add_argument("--plateau-dir", default=None, help="PLATEAU CityGML directory (optional)")
    p.add_argument(
        "--diffraction-model", default="knife_edge", choices=["knife_edge", "utd", "none"],
        help="NLOS diffraction amplitude model, or 'none' to disable (default: knife_edge)",
    )
    p.add_argument("--elevation-mask-deg", type=float, default=10.0)
    p.add_argument("--cn0-zenith-dbhz", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="output CSV path (default: print summary only)")
    p.add_argument(
        "--rinex-out", default=None,
        help="output RINEX 3.04 OBS path (C1C/D1C/S1C only, no carrier phase)",
    )
    return p


def _write_csv(arrays: dict, out_path: str) -> None:
    keys = list(arrays.keys())
    n_rows = len(arrays[keys[0]]) if keys else 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n_rows):
            writer.writerow([arrays[k][i] for k in keys])


def _print_summary(result, arrays: dict) -> None:
    n_epochs = result.n_epochs
    n_sat_per_epoch = np.array([ep.n_sat for ep in result.epochs], dtype=np.float64)
    mean_visible = float(n_sat_per_epoch.mean()) if n_sat_per_epoch.size else 0.0

    is_los = arrays.get("is_los", np.zeros(0, dtype=bool))
    los_fraction = float(np.mean(is_los)) if is_los.size else float("nan")

    print(f"epochs: {n_epochs}")
    print(f"mean visible satellites per epoch: {mean_visible:.2f}")
    print(f"LOS fraction: {los_fraction:.3f}")
    print(f"total satellite-epoch rows: {is_los.size}")


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    diffraction_model = None if args.diffraction_model == "none" else args.diffraction_model

    config = ScenarioConfig(
        nav_file=args.nav,
        lat_deg=args.lat,
        lon_deg=args.lon,
        alt_m=args.alt,
        start_time=args.start,
        duration_s=args.duration,
        step_s=args.step,
        constellations=args.constellations,
        plateau_dir=args.plateau_dir,
        diffraction_model=diffraction_model,
        elevation_mask_deg=args.elevation_mask_deg,
        cn0_zenith_dbhz=args.cn0_zenith_dbhz,
        seed=args.seed,
    )

    result = run_scenario(config)
    arrays = result.to_arrays()

    if args.out:
        _write_csv(arrays, args.out)
        print(f"wrote {len(arrays['epoch_index'])} rows to {args.out}")

    if args.rinex_out:
        result.to_rinex(args.rinex_out)
        print(f"wrote {result.n_epochs} epochs to {args.rinex_out}")

    _print_summary(result, arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
