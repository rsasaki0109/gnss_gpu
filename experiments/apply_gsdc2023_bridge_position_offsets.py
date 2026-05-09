"""Apply phone position offsets to every source coordinate in a GSDC bridge CSV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_gsdc2023_target_trip_source_delta import source_coordinate_columns
from experiments.apply_gsdc2023_position_offsets import (
    apply_scaled_phone_offset,
    phone_from_trip,
    sha256_file,
)
from experiments.evaluate import ecef_to_lla, lla_to_ecef
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


def _altitude_column_for(latitude_column: str, frame: pd.DataFrame) -> str | None:
    if latitude_column == "LatitudeDegrees":
        return "AltitudeMeters" if "AltitudeMeters" in frame.columns else None
    candidate = latitude_column.removesuffix("LatitudeDegrees") + "AltitudeMeters"
    return candidate if candidate in frame.columns else None


def _latlon_alt_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            lla_to_ecef(math.radians(float(lat)), math.radians(float(lon)), float(alt))
            for lat, lon, alt in zip(lat_deg, lon_deg, alt_m)
        ],
        dtype=np.float64,
    )


def _ecef_to_latlon_alt(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitudes: list[float] = []
    longitudes: list[float] = []
    altitudes: list[float] = []
    for x, y, z in np.asarray(xyz, dtype=np.float64).reshape(-1, 3):
        lat_rad, lon_rad, alt_m = ecef_to_lla(float(x), float(y), float(z))
        latitudes.append(math.degrees(lat_rad))
        longitudes.append(math.degrees(lon_rad))
        altitudes.append(float(alt_m))
    return (
        np.asarray(latitudes, dtype=np.float64),
        np.asarray(longitudes, dtype=np.float64),
        np.asarray(altitudes, dtype=np.float64),
    )


def _infer_phone(frame: pd.DataFrame, explicit_phone: str | None) -> str:
    if explicit_phone:
        return explicit_phone.lower()
    if "tripId" not in frame.columns:
        raise ValueError("--phone is required when bridge_positions.csv has no tripId column")
    phones = {phone_from_trip(trip_id) for trip_id in frame["tripId"].dropna().astype(str).unique()}
    if len(phones) != 1:
        raise ValueError(f"--phone is required when bridge_positions.csv contains phones: {sorted(phones)}")
    return next(iter(phones))


def apply_bridge_position_offsets(
    frame: pd.DataFrame,
    *,
    phone: str | None,
    scale: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "UnixTimeMillis" not in frame.columns:
        raise ValueError("bridge_positions.csv is missing UnixTimeMillis")
    if scale < 0.0:
        raise ValueError("scale must be non-negative")

    sources = source_coordinate_columns(frame)
    if not sources:
        raise ValueError("bridge_positions.csv contains no source coordinate column pairs")

    resolved_phone = _infer_phone(frame, phone)
    output = frame.copy()
    source_summary: dict[str, object] = {}
    for source, (lat_column, lon_column) in sources.items():
        alt_column = _altitude_column_for(lat_column, frame)
        alt = frame[alt_column].to_numpy(dtype=np.float64) if alt_column else np.zeros(len(frame), dtype=np.float64)
        lat0 = frame[lat_column].to_numpy(dtype=np.float64)
        lon0 = frame[lon_column].to_numpy(dtype=np.float64)
        xyz = _latlon_alt_to_ecef(lat0, lon0, alt)
        shifted = apply_scaled_phone_offset(xyz, resolved_phone, scale)
        lat1, lon1, alt1 = _ecef_to_latlon_alt(shifted)
        output[lat_column] = lat1
        output[lon_column] = lon1
        if alt_column:
            output[alt_column] = alt1

        delta_m = haversine_m(lat0, lon0, lat1, lon1)
        source_summary[source] = {
            "latitude_column": lat_column,
            "longitude_column": lon_column,
            "altitude_column": alt_column,
            "changed_rows_gt_0p01m": int(np.count_nonzero(delta_m > 0.01)),
            "delta_vs_input": gsdc_score_m(delta_m),
        }

    selected_delta = None
    if "selected" in sources:
        lat_column, lon_column = sources["selected"]
        selected_delta = haversine_m(
            frame[lat_column].to_numpy(dtype=np.float64),
            frame[lon_column].to_numpy(dtype=np.float64),
            output[lat_column].to_numpy(dtype=np.float64),
            output[lon_column].to_numpy(dtype=np.float64),
        )

    summary = {
        "phone": resolved_phone,
        "scale": float(scale),
        "rows": int(len(output)),
        "source_count": int(len(sources)),
        "source_columns": {source: [lat, lon] for source, (lat, lon) in sources.items()},
        "source_summary": source_summary,
        "selected_delta_vs_input": gsdc_score_m(selected_delta) if selected_delta is not None else None,
    }
    return output, summary


def write_outputs(
    output_dir: Path,
    output: pd.DataFrame,
    summary: dict[str, object],
    *,
    input_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bridge_positions.csv"
    output.to_csv(output_path, index=False)
    payload = {
        **summary,
        "input": str(input_path),
        "output": str(output_path),
        "sha256": sha256_file(output_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-positions", type=Path, required=True)
    parser.add_argument("--phone", help="phone name; inferred from tripId when available")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.bridge_positions)
    output, summary = apply_bridge_position_offsets(frame, phone=args.phone, scale=args.scale)
    write_outputs(args.output_dir, output, summary, input_path=args.bridge_positions)
    print(f"wrote bridge position offsets: rows={summary['rows']} sources={summary['source_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
