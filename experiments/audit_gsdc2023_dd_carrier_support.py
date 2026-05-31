#!/usr/bin/env python3
"""Audit GSDC2023 double-differenced carrier support against local base RINEX."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.audit_gsdc2023_base_readiness import discover_gsdc_trips  # noqa: E402
from experiments.gsdc2023_base_correction import (  # noqa: E402
    GPS_WEEK_SECONDS,
    base_setting,
    course_base_obs_path,
    read_base_station_xyz,
    trip_course_phone,
    unix_ms_to_gps_abs_seconds,
)
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    build_trip_arrays,
)
from experiments.gsdc2023_tdcp import valid_adr_state  # noqa: E402
from gnss_gpu.dd_carrier import (  # noqa: E402
    BEIDOU_B1I_WAVELENGTH,
    BEIDOU_B2A_WAVELENGTH,
    DDCarrierComputer,
    GALILEO_E1_WAVELENGTH,
    GALILEO_E5A_WAVELENGTH,
    GPS_L1_WAVELENGTH,
    GPS_L5_WAVELENGTH,
    QZSS_L1_WAVELENGTH,
    QZSS_L5_WAVELENGTH,
)
from gnss_gpu.spp import _elevation_azimuth  # noqa: E402


DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_dd_carrier_support_20260519.csv")
ANDROID_TO_DD_SYSTEM_ID = {
    1: 0,  # GPS
    3: 1,  # GLONASS
    6: 2,  # Galileo
    5: 3,  # BeiDou
    4: 4,  # QZSS
}


def carrier_wavelength_m(constellation: int, signal_type: str) -> float:
    signal = str(signal_type).upper()
    is_l5 = "L5" in signal or "E5" in signal
    if int(constellation) == 1:
        return GPS_L5_WAVELENGTH if is_l5 else GPS_L1_WAVELENGTH
    if int(constellation) == 6:
        return GALILEO_E5A_WAVELENGTH if is_l5 else GALILEO_E1_WAVELENGTH
    if int(constellation) == 4:
        return QZSS_L5_WAVELENGTH if is_l5 else QZSS_L1_WAVELENGTH
    if int(constellation) == 5:
        return BEIDOU_B2A_WAVELENGTH if is_l5 else BEIDOU_B1I_WAVELENGTH
    return GPS_L1_WAVELENGTH


def rover_measurements_for_epoch(batch, epoch_idx: int) -> list[SimpleNamespace]:
    if batch.adr is None or batch.adr_state is None:
        return []
    rx = np.asarray(batch.kaggle_wls[epoch_idx, :3], dtype=np.float64)
    out: list[SimpleNamespace] = []
    for slot_idx, (constellation, svid, signal_type) in enumerate(batch.slot_keys):
        adr_m = float(batch.adr[epoch_idx, slot_idx])
        state = int(batch.adr_state[epoch_idx, slot_idx])
        sat = np.asarray(batch.sat_ecef[epoch_idx, slot_idx], dtype=np.float64)
        if not (np.isfinite(adr_m) and adr_m != 0.0 and valid_adr_state(state)):
            continue
        if sat.shape != (3,) or not np.isfinite(sat).all() or np.linalg.norm(sat) < 1.0e6:
            continue
        wavelength = carrier_wavelength_m(int(constellation), str(signal_type))
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            continue
        elevation = 0.3
        if np.isfinite(rx).all() and np.linalg.norm(rx) > 1.0e6:
            elevation, _ = _elevation_azimuth(rx, sat)
        out.append(
            SimpleNamespace(
                system_id=ANDROID_TO_DD_SYSTEM_ID.get(int(constellation), 0),
                prn=int(svid),
                satellite_ecef=sat,
                carrier_phase=adr_m / wavelength,
                elevation=float(elevation),
                snr=30.0,
            )
        )
    return out


def _computer_for_trip(
    data_root: Path,
    trip: str,
    cache: dict[Path, DDCarrierComputer],
    *,
    base_obs_template: str | None = None,
    require_base_obs_template: bool = False,
) -> DDCarrierComputer:
    split, course, phone = trip_course_phone(trip)
    if split is None or course is None or phone is None:
        raise ValueError(f"trip must be split/course/phone: {trip}")
    base_name, rinex_type = base_setting(data_root, split, course, phone)
    obs_path = course_base_obs_path_for_template(
        data_root,
        split,
        course,
        base_name,
        rinex_type,
        base_obs_template=base_obs_template,
        require_template=require_base_obs_template,
    )
    key = obs_path.resolve()
    if key not in cache:
        base_xyz = read_base_station_xyz(data_root, course, base_name, apply_offset=False)
        cache[key] = DDCarrierComputer(
            obs_path,
            base_position=base_xyz,
            allowed_systems=("G", "E", "J", "C"),
            interpolate_base_epochs=True,
        )
    return cache[key]


def course_base_obs_path_for_template(
    data_root: Path,
    split: str,
    course: str,
    base_name: str,
    rinex_type: str | None,
    *,
    base_obs_template: str | None,
    require_template: bool = False,
) -> Path:
    if not base_obs_template:
        return course_base_obs_path(data_root, split, course, base_name, rinex_type)
    suffix = "rnx3" if rinex_type == "V3" else "rnx2" if rinex_type == "V2" else "rnx3"
    rendered = base_obs_template.format(base=base_name, rinex_type=rinex_type or "", suffix=suffix)
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = data_root / split / course / candidate
    if candidate.is_file() or require_template:
        return candidate
    return course_base_obs_path(data_root, split, course, base_name, rinex_type)


def snap_tow_to_base_epoch(computer: DDCarrierComputer, tow: float, tolerance_s: float) -> float | None:
    base_tows = np.asarray(getattr(computer, "_base_tow_keys", np.array([], dtype=np.float64)), dtype=np.float64)
    if base_tows.size == 0:
        return None
    idx = int(np.searchsorted(base_tows, float(tow)))
    candidates = []
    if idx < base_tows.size:
        candidates.append(float(base_tows[idx]))
    if idx > 0:
        candidates.append(float(base_tows[idx - 1]))
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - float(tow)))
    if abs(nearest - float(tow)) <= float(tolerance_s):
        return nearest
    return None


def dd_carrier_support_row(
    data_root: Path,
    trip: str,
    args: argparse.Namespace,
    cache: dict[Path, DDCarrierComputer] | None = None,
) -> dict[str, object]:
    cache = cache if cache is not None else {}
    batch = build_trip_arrays(
        data_root / trip,
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        use_tdcp=True,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        data_root=data_root,
        trip=trip,
        dual_frequency=args.dual_frequency,
    )
    split, course, phone = trip_course_phone(trip)
    if split is None or course is None or phone is None:
        raise ValueError(f"trip must be split/course/phone: {trip}")
    base_name, rinex_type = base_setting(data_root, split, course, phone)
    obs_path = course_base_obs_path_for_template(
        data_root,
        split,
        course,
        base_name,
        rinex_type,
        base_obs_template=getattr(args, "base_obs_template", None),
        require_template=bool(getattr(args, "require_base_obs_template", False)),
    )
    computer = _computer_for_trip(
        data_root,
        trip,
        cache,
        base_obs_template=getattr(args, "base_obs_template", None),
        require_base_obs_template=bool(getattr(args, "require_base_obs_template", False)),
    )
    tows = np.mod(unix_ms_to_gps_abs_seconds(batch.times_ms), GPS_WEEK_SECONDS)
    dd_pairs: list[int] = []
    rover_counts: list[int] = []
    snapped_epochs = 0
    for epoch_idx, tow in enumerate(tows):
        measurements = rover_measurements_for_epoch(batch, epoch_idx)
        rover_counts.append(len(measurements))
        tow_for_dd = snap_tow_to_base_epoch(computer, float(tow), args.tow_snap_tolerance_s)
        if tow_for_dd is None:
            continue
        snapped_epochs += 1
        dd = computer.compute_dd(
            tow_for_dd,
            measurements,
            rover_position_approx=batch.kaggle_wls[epoch_idx, :3],
            min_common_sats=args.min_common_sats,
        )
        if dd is not None:
            dd_pairs.append(int(dd.n_dd))

    base_tows = getattr(computer, "_base_tow_keys", np.array([], dtype=np.float64))
    base_dt = np.diff(np.asarray(base_tows, dtype=np.float64))
    return {
        "trip": trip,
        "phone": Path(trip).name,
        "base_obs_path": str(obs_path),
        "base_obs_file": obs_path.name,
        "n_epochs": int(batch.times_ms.size),
        "base_epochs": int(len(base_tows)),
        "base_dt_median_s": float(np.median(base_dt)) if base_dt.size else float("nan"),
        "rover_carrier_mean_per_epoch": float(np.mean(rover_counts)) if rover_counts else 0.0,
        "base_snapped_epochs": int(snapped_epochs),
        "base_snap_coverage_frac": float(snapped_epochs / max(batch.times_ms.size, 1)),
        "dd_epochs": int(len(dd_pairs)),
        "dd_coverage_frac": float(len(dd_pairs) / max(batch.times_ms.size, 1)),
        "dd_pairs_mean": float(np.mean(dd_pairs)) if dd_pairs else 0.0,
        "dd_pairs_p50": float(np.percentile(dd_pairs, 50)) if dd_pairs else 0.0,
        "dd_pairs_p95": float(np.percentile(dd_pairs, 95)) if dd_pairs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", action="append", choices=("train", "test"), default=[])
    parser.add_argument("--trip", action="append", default=[], help="train/.../phone; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-common-sats", type=int, default=4)
    parser.add_argument("--tow-snap-tolerance-s", type=float, default=0.6)
    parser.add_argument(
        "--base-obs-template",
        default=None,
        help="optional course-relative template such as '{base}_1hz.obs'; falls back to standard base obs unless required",
    )
    parser.add_argument("--require-base-obs-template", action="store_true")
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument("--tdcp-geometry-correction", action=argparse.BooleanOptionalAction, default=DEFAULT_TDCP_GEOMETRY_CORRECTION)
    args = parser.parse_args()

    splits = tuple(args.split) if args.split else ("train",)
    trips = discover_gsdc_trips(args.data_root, splits=splits)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no train trips found")

    rows = []
    cache: dict[Path, DDCarrierComputer] = {}
    for idx, trip in enumerate(trips, start=1):
        rows.append(dd_carrier_support_row(args.data_root, trip, args, cache))
        print(f"[{idx}/{len(trips)}] {trip}", flush=True)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote: {args.output}")
    print(f"mean DD coverage: {frame['dd_coverage_frac'].mean():.3f}")
    print(f"mean DD pairs when available: {frame['dd_pairs_mean'].mean():.2f}")


if __name__ == "__main__":
    main()
