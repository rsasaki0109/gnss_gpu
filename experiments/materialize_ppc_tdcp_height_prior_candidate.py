#!/usr/bin/env python3
"""Materialize a window candidate with a TDCP-derived height prior.

This diagnostic keeps the RTK candidate's ECEF X/Y, estimates a low-frequency
ellipsoidal height prior by GPS L1 TDCP integration from the rover RINEX header
position, and solves only ECEF Z so the candidate lies on that height surface.
It is intended for segment-local CT-RBPF/FGO candidate-pool experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics  # noqa: E402
from exp_ppc_ctrbpf_fgo import _build_tdcp_measurements  # noqa: E402


_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    b = _WGS84_A * (1.0 - _WGS84_F)
    ep2 = (_WGS84_A * _WGS84_A - b * b) / (b * b)
    p = math.hypot(float(x), float(y))
    theta = math.atan2(float(z) * _WGS84_A, p * b)
    lon = math.atan2(float(y), float(x))
    lat = math.atan2(
        float(z) + ep2 * b * math.sin(theta) ** 3,
        p - _WGS84_E2 * _WGS84_A * math.cos(theta) ** 3,
    )
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return lat, lon, h


def _lla_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    return np.array(
        [
            (n + h) * math.cos(lat) * math.cos(lon),
            (n + h) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - _WGS84_E2) + h) * math.sin(lat),
        ],
        dtype=np.float64,
    )


def _solve_z_for_height(x: float, y: float, z0: float, height_m: float) -> float:
    lo = float(z0) - 200.0
    hi = float(z0) + 200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        _lat, _lon, h = _ecef_to_lla(float(x), float(y), mid)
        if h < float(height_m):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _parse_header_approx_position(obs_path: Path) -> np.ndarray:
    with obs_path.open(errors="replace") as fh:
        for line in fh:
            if "APPROX POSITION XYZ" in line:
                return np.array([float(line[0:14]), float(line[14:28]), float(line[28:42])])
            if "END OF HEADER" in line:
                break
    raise ValueError(f"APPROX POSITION XYZ not found: {obs_path}")


def _estimate_tdcp_height_series(
    run_dir: Path,
    max_epochs: int,
    postfit_max_m: float,
) -> tuple[dict[float, float], int]:
    header_pos = _parse_header_approx_position(run_dir / "rover.obs")
    data = PPCDatasetLoader(run_dir).load_experiment_data(
        max_epochs=max_epochs,
        include_sat_velocity=True,
        systems=("G",),
    )
    times = np.asarray(data["times"], dtype=np.float64)
    pos = header_pos.astype(np.float64).copy()
    heights: dict[float, float] = {
        round(float(times[0]), 1): _ecef_to_lla(float(pos[0]), float(pos[1]), float(pos[2]))[2]
    }
    accepted = 0

    for i in range(len(times) - 1):
        dt = float(times[i + 1] - times[i])
        if not np.isfinite(dt) or dt <= 0.0:
            continue
        prev_meas = _build_tdcp_measurements(
            np.asarray(data["sat_ecef"][i], dtype=np.float64),
            np.asarray(data["system_ids"][i], dtype=np.int32),
            list(data["used_prns"][i]),
            np.asarray(data["carrier_phase"][i], dtype=np.float64),
            np.asarray(data["sat_velocity"][i], dtype=np.float64),
            np.asarray(data["clock_drift"][i], dtype=np.float64),
            pos,
        )
        cur_meas = _build_tdcp_measurements(
            np.asarray(data["sat_ecef"][i + 1], dtype=np.float64),
            np.asarray(data["system_ids"][i + 1], dtype=np.int32),
            list(data["used_prns"][i + 1]),
            np.asarray(data["carrier_phase"][i + 1], dtype=np.float64),
            np.asarray(data["sat_velocity"][i + 1], dtype=np.float64),
            np.asarray(data["clock_drift"][i + 1], dtype=np.float64),
            pos,
        )
        velocity, _rms = estimate_velocity_from_tdcp_with_metrics(
            pos,
            prev_meas,
            cur_meas,
            dt,
            min_sats=4,
            max_postfit_rms_m=float(postfit_max_m),
            elevation_weight=True,
            receiver_motion_sign=-1.0,
        )
        if velocity is not None:
            pos = pos + np.asarray(velocity, dtype=np.float64) * dt
            accepted += 1
        tow = round(float(times[i + 1]), 1)
        heights[tow] = _ecef_to_lla(float(pos[0]), float(pos[1]), float(pos[2]))[2]

    if not heights:
        raise ValueError("no TDCP height estimates")
    return heights, accepted


def _estimate_tdcp_height_prior(
    run_dir: Path,
    start_tow: float,
    end_tow: float,
    max_epochs: int,
    postfit_max_m: float,
) -> tuple[float, int]:
    series, accepted = _estimate_tdcp_height_series(
        run_dir,
        max_epochs,
        postfit_max_m,
    )
    heights = [
        height
        for tow, height in series.items()
        if float(start_tow) <= float(tow) <= float(end_tow)
    ]
    if not heights:
        raise ValueError("no TDCP height estimates in requested window")
    return float(np.median(heights)), accepted


def _rewrite_pos_window(src: Path, dst: Path, start_tow: float, end_tow: float, height_m: float) -> int:
    kept = 0
    with src.open() as inf, dst.open("w", newline="") as outf:
        for line in inf:
            if line.startswith("%") or not line.strip():
                outf.write(line)
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            tow = round(float(parts[1]), 1)
            if not (float(start_tow) <= tow <= float(end_tow)):
                continue
            x = float(parts[2])
            y = float(parts[3])
            z = _solve_z_for_height(x, y, float(parts[4]), float(height_m))
            lat, lon, h = _ecef_to_lla(x, y, z)
            parts[4] = f"{z:.4f}"
            parts[5] = f"{math.degrees(lat):.9f}"
            parts[6] = f"{math.degrees(lon):.9f}"
            parts[7] = f"{h:.4f}"
            outf.write(
                f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} "
                f"{parts[5]} {parts[6]} {parts[7]} {parts[8]} {parts[9]} "
                f"{parts[10]} {parts[11]} {parts[12]}\n"
            )
            kept += 1
    return kept


def _copy_csv_window(src: Path, dst: Path, start_tow: float, end_tow: float) -> int:
    kept = 0
    with src.open(newline="") as inf, dst.open("w", newline="") as outf:
        reader = csv.DictReader(inf)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {src}")
        writer = csv.DictWriter(outf, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            tow = round(float(row["tow"]), 1)
            if float(start_tow) <= tow <= float(end_tow):
                writer.writerow(row)
                kept += 1
    return kept


def _batch_materialize(args: argparse.Namespace) -> None:
    if args.batch_csv is None:
        return
    series, accepted = _estimate_tdcp_height_series(
        args.run_dir,
        int(args.max_epochs),
        float(args.tdcp_postfit_max_m),
    )
    first_tow = min(series)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, object]] = []
    with args.batch_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                start_tow = float(row.get("start_tow", "nan"))
                end_tow = float(row.get("end_tow", "nan"))
            except ValueError:
                start_idx = int(row["start_idx"])
                end_idx = int(row["end_idx"])
                start_tow = round(first_tow + 0.2 * start_idx, 1)
                end_tow = round(first_tow + 0.2 * end_idx, 1)
            if not np.isfinite(start_tow) or not np.isfinite(end_tow):
                start_idx = int(row["start_idx"])
                end_idx = int(row["end_idx"])
                start_tow = round(first_tow + 0.2 * start_idx, 1)
                end_tow = round(first_tow + 0.2 * end_idx, 1)
            config = str(row.get("config", "cand")).strip() or "cand"
            if args.only_config and config != args.only_config:
                continue
            source_pos = Path(row["out_pos"])
            source_csv = Path(row["out_csv"])
            if not source_pos.is_file() or not source_csv.is_file():
                continue
            heights = [
                height
                for tow, height in series.items()
                if float(start_tow) <= float(tow) <= float(end_tow)
            ]
            if not heights:
                continue
            height_m = float(np.median(heights))
            dir_name = f"{args.out_prefix}_{int(row['start_idx'])}_{int(row['end_idx'])}_{config}"
            out_dir = args.out_dir / dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{args.city}_{args.run}_full"
            out_pos = out_dir / f"{prefix}.pos"
            out_csv = out_dir / f"{prefix}.csv"
            n_pos = _rewrite_pos_window(source_pos, out_pos, start_tow, end_tow, height_m)
            n_csv = _copy_csv_window(source_csv, out_csv, start_tow, end_tow)
            rows_out.append({
                "dir": dir_name,
                "config": config,
                "start_idx": row.get("start_idx", ""),
                "end_idx": row.get("end_idx", ""),
                "start_tow": f"{start_tow:.1f}",
                "end_tow": f"{end_tow:.1f}",
                "height_prior_m": f"{height_m:.4f}",
                "tdcp_pairs_accepted": accepted,
                "pos_rows": n_pos,
                "csv_rows": n_csv,
                "source_pos": str(source_pos),
                "source_csv": str(source_csv),
            })
    manifest = args.out_dir / f"{args.out_prefix}_manifest.csv"
    with manifest.open("w", newline="") as fh:
        fieldnames = [
            "dir",
            "config",
            "start_idx",
            "end_idx",
            "start_tow",
            "end_tow",
            "height_prior_m",
            "tdcp_pairs_accepted",
            "pos_rows",
            "csv_rows",
            "source_pos",
            "source_csv",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"saved manifest: {manifest} rows={len(rows_out)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-pos", type=Path, default=None)
    parser.add_argument("--source-csv", type=Path, default=None)
    parser.add_argument("--start-tow", type=float, default=None)
    parser.add_argument("--end-tow", type=float, default=None)
    parser.add_argument("--batch-csv", type=Path, default=None)
    parser.add_argument("--out-prefix", default="tdcp_height_prior")
    parser.add_argument("--only-config", default="")
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--tdcp-postfit-max-m", type=float, default=2.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_csv is not None:
        _batch_materialize(args)
        return
    if args.source_pos is None or args.source_csv is None:
        raise SystemExit("--source-pos and --source-csv are required outside --batch-csv")
    if args.start_tow is None or args.end_tow is None:
        raise SystemExit("--start-tow and --end-tow are required outside --batch-csv")

    height_m, accepted = _estimate_tdcp_height_prior(
        args.run_dir,
        float(args.start_tow),
        float(args.end_tow),
        int(args.max_epochs),
        float(args.tdcp_postfit_max_m),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.city}_{args.run}_full"
    out_pos = args.out_dir / f"{prefix}.pos"
    out_csv = args.out_dir / f"{prefix}.csv"
    n_pos = _rewrite_pos_window(args.source_pos, out_pos, args.start_tow, args.end_tow, height_m)
    n_csv = _copy_csv_window(args.source_csv, out_csv, args.start_tow, args.end_tow)
    print(f"height_prior_m={height_m:.4f} tdcp_pairs_accepted={accepted}")
    print(f"saved pos: {out_pos} rows={n_pos}")
    print(f"saved csv: {out_csv} rows={n_csv}")


if __name__ == "__main__":
    main()
