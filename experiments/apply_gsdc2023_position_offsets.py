#!/usr/bin/env python3
"""Apply train-backed phone position offsets to a GSDC2023 submission CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.evaluate import ecef_to_lla, lla_to_ecef
from experiments.gsdc2023_height_constraints import phone_position_offset
from experiments.gsdc2023_imu import (
    ecef_to_enu_relative,
    enu_to_ecef_relative,
    estimate_rpy_from_velocity,
    wrap_to_180_deg,
)
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


SUPPORTED_PHONE_SCALES_1X = {
    "mi8": 1.0,
    "xiaomimi8": 1.0,
    "pixel6pro": 1.0,
    "pixel7pro": 1.0,
    "samsunga32": 1.0,
    "samsunga325g": 1.0,
    "sm-a205u": 1.0,
    "sm-a325f": 1.0,
    "sm-g988b": 1.0,
    "sm-s908b": 1.0,
}

PHONE_TUNED_SCALES = {
    "mi8": 2.5,
    "xiaomimi8": 2.5,
    "pixel6pro": 3.0,
    "pixel7pro": 4.0,
    "samsunga32": 2.5,
    "samsunga325g": 2.5,
    "sm-a205u": 2.5,
    "sm-a325f": 2.5,
    "sm-g988b": 4.0,
    "sm-s908b": 2.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phone_from_trip(trip_id: str) -> str:
    return str(trip_id).rstrip("/").split("/")[-1].lower()


def lla_to_ecef_rows(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            lla_to_ecef(math.radians(float(lat)), math.radians(float(lon)), 0.0)
            for lat, lon in zip(lat_deg, lon_deg)
        ],
        dtype=np.float64,
    )


def ecef_to_latlon_rows(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat: list[float] = []
    lon: list[float] = []
    for x, y, z in np.asarray(xyz, dtype=np.float64).reshape(-1, 3):
        lat_rad, lon_rad, _ = ecef_to_lla(float(x), float(y), float(z))
        lat.append(math.degrees(lat_rad))
        lon.append(math.degrees(lon_rad))
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def apply_scaled_phone_offset(xyz_ecef: np.ndarray, phone: str, scale: float) -> np.ndarray:
    xyz = np.asarray(xyz_ecef, dtype=np.float64).reshape(-1, 3)
    if scale <= 0.0:
        return xyz.copy()
    offset = phone_position_offset(phone)
    if offset is None or xyz.size == 0:
        return xyz.copy()
    finite_rows = np.isfinite(xyz).all(axis=1)
    if not finite_rows.any():
        return xyz.copy()

    origin_xyz = xyz[np.flatnonzero(finite_rows)[0]]
    enu = ecef_to_enu_relative(xyz, origin_xyz)
    times_s = np.arange(enu.shape[0], dtype=np.float64)
    vel_enu = np.zeros_like(enu)
    for axis in range(3):
        vel_enu[:, axis] = (
            np.gradient(enu[:, axis], times_s, edge_order=1) if enu.shape[0] > 1 else 0.0
        )
    heading = wrap_to_180_deg(np.rad2deg(estimate_rpy_from_velocity(vel_enu)[:, 2]) - 180.0)
    heading_rad = np.deg2rad(heading)
    offset_rl, offset_ud = offset
    offset_rl *= float(scale)
    offset_ud *= float(scale)
    offset_enu = np.column_stack(
        [
            np.cos(heading_rad) * offset_ud - np.sin(heading_rad) * offset_rl,
            np.sin(heading_rad) * offset_ud + np.cos(heading_rad) * offset_rl,
            np.zeros(heading_rad.size, dtype=np.float64),
        ],
    )
    return enu_to_ecef_relative(enu + offset_enu, origin_xyz)


def scale_map_for_policy(policy: str, uniform_scale: float) -> dict[str, float]:
    if policy == "supported":
        return {phone: float(uniform_scale) for phone in SUPPORTED_PHONE_SCALES_1X}
    if policy == "phone-tuned":
        return dict(PHONE_TUNED_SCALES)
    raise ValueError(f"unsupported policy: {policy}")


def parse_phone_scale_override(value: str) -> tuple[str, float]:
    phone, sep, scale_text = value.partition("=")
    if sep == "":
        raise argparse.ArgumentTypeError("expected PHONE=SCALE")
    phone = phone.strip().lower()
    if not phone:
        raise argparse.ArgumentTypeError("phone name must not be empty")
    try:
        scale = float(scale_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid scale for {phone!r}: {scale_text!r}") from exc
    if scale < 0.0:
        raise argparse.ArgumentTypeError("scale must be non-negative")
    return phone, scale


def parse_trip_scale_override(value: str) -> tuple[str, float]:
    trip_id, sep, scale_text = value.partition("=")
    if sep == "":
        raise argparse.ArgumentTypeError("expected TRIP=SCALE")
    trip_id = trip_id.strip()
    if not trip_id:
        raise argparse.ArgumentTypeError("trip id must not be empty")
    try:
        scale = float(scale_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid scale for {trip_id!r}: {scale_text!r}") from exc
    if scale < 0.0:
        raise argparse.ArgumentTypeError("scale must be non-negative")
    return trip_id, scale


def apply_offsets(
    source: pd.DataFrame,
    scales: dict[str, float],
    trip_scales: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trip_scales = trip_scales or {}
    output = source.copy()
    trip_rows: list[dict[str, Any]] = []
    for trip_id, index in source.groupby("tripId", sort=False).groups.items():
        trip_id = str(trip_id)
        phone = phone_from_trip(str(trip_id))
        scale = float(trip_scales.get(trip_id, scales.get(phone, 0.0)))
        indices = np.asarray(list(index), dtype=np.int64)
        if scale > 0.0:
            xyz = lla_to_ecef_rows(
                source.loc[indices, "LatitudeDegrees"].to_numpy(),
                source.loc[indices, "LongitudeDegrees"].to_numpy(),
            )
            lat, lon = ecef_to_latlon_rows(apply_scaled_phone_offset(xyz, phone, scale))
            output.loc[indices, "LatitudeDegrees"] = lat
            output.loc[indices, "LongitudeDegrees"] = lon
            delta_m = haversine_m(
                source.loc[indices, "LatitudeDegrees"].to_numpy(),
                source.loc[indices, "LongitudeDegrees"].to_numpy(),
                lat,
                lon,
            )
        else:
            delta_m = np.zeros(indices.size, dtype=np.float64)
        trip_rows.append(
            {
                "tripId": trip_id,
                "phone": phone,
                "scale": scale,
                "rows": int(indices.size),
                "mean_delta_m": float(np.mean(delta_m)) if delta_m.size else 0.0,
                "p95_delta_m": float(np.percentile(delta_m, 95)) if delta_m.size else 0.0,
                "max_delta_m": float(np.max(delta_m)) if delta_m.size else 0.0,
            },
        )
    return output, pd.DataFrame(trip_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--trip-summary", type=Path, default=None)
    parser.add_argument("--policy", choices=("supported", "phone-tuned"), default="supported")
    parser.add_argument("--scale", type=float, default=1.0, help="uniform scale for --policy supported")
    parser.add_argument(
        "--phone-scale",
        action="append",
        default=[],
        type=parse_phone_scale_override,
        metavar="PHONE=SCALE",
        help="override one phone scale after applying --policy; can be repeated",
    )
    parser.add_argument(
        "--trip-scale",
        action="append",
        default=[],
        type=parse_trip_scale_override,
        metavar="TRIP=SCALE",
        help="override one exact tripId scale after phone scales; can be repeated",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    required = {"tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
    missing = required.difference(source.columns)
    if missing:
        raise SystemExit(f"{args.input} is missing required columns: {sorted(missing)}")

    scales = scale_map_for_policy(args.policy, args.scale)
    for phone, scale in args.phone_scale:
        scales[phone] = scale
    trip_scales = dict(args.trip_scale)
    output, trip_summary = apply_offsets(source, scales, trip_scales)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    trip_summary_path = args.trip_summary
    if trip_summary_path is None:
        trip_summary_path = args.output.with_name(args.output.stem + "_trip_summary.csv")
    trip_summary_path.parent.mkdir(parents=True, exist_ok=True)
    trip_summary.to_csv(trip_summary_path, index=False)

    delta_m = haversine_m(
        source["LatitudeDegrees"].to_numpy(),
        source["LongitudeDegrees"].to_numpy(),
        output["LatitudeDegrees"].to_numpy(),
        output["LongitudeDegrees"].to_numpy(),
    )
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "policy": args.policy,
        "scale": float(args.scale),
        "phone_scale_overrides": {
            phone: scale for phone, scale in sorted(dict(args.phone_scale).items()) if scale > 0.0
        },
        "trip_scale_overrides": {
            trip_id: scale for trip_id, scale in sorted(trip_scales.items()) if scale > 0.0
        },
        "scales": {phone: scale for phone, scale in sorted(scales.items()) if scale > 0.0},
        "rows": int(len(output)),
        "nan_lat_lon_rows": int(output[["LatitudeDegrees", "LongitudeDegrees"]].isna().any(axis=1).sum()),
        "changed_rows_gt_0p01m": int(np.count_nonzero(delta_m > 0.01)),
        "delta_vs_input": gsdc_score_m(delta_m),
        "trip_summary": str(trip_summary_path),
    }
    summary_path = args.summary
    if summary_path is None:
        summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
