#!/usr/bin/env python3
"""Inject deterministic, truth-free faults into a PPC IMU CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


IMU_HEADER = (
    "GPS TOW (s)",
    "GPS Week",
    "Acc X (m/s^2)",
    "Acc Y (m/s^2)",
    "Acc Z (m/s^2)",
    "Ang Rate X (deg/s)",
    "Ang Rate Y (deg/s)",
    "Ang Rate Z (deg/s)",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inject_imu_fault(
    rows: list[dict[str, Any]],
    *,
    fault: str,
    start_tow: float,
    end_tow: float,
    time_offset_s: float = 0.05,
    accel_bias_mps2: tuple[float, float, float] = (0.2, 0.0, 0.0),
    gyro_bias_degps: tuple[float, float, float] = (0.0, 0.0, 1.0),
    vibration_frequency_hz: float = 20.0,
    vibration_accel_mps2: tuple[float, float, float] = (1.0, 0.0, 0.0),
    vibration_gyro_degps: tuple[float, float, float] = (0.0, 0.0, 2.0),
) -> tuple[list[dict[str, float | int]], int]:
    """Return a copied faulted stream and number of source rows affected."""

    supported = {
        "dropout",
        "time_offset",
        "accel_bias_jump",
        "gyro_bias_jump",
        "bias_jump",
        "vibration",
    }
    if fault not in supported:
        raise ValueError("unsupported fault")
    if not math.isfinite(start_tow) or not math.isfinite(end_tow):
        raise ValueError("fault interval must be finite")
    if end_tow < start_tow:
        raise ValueError("invalid fault interval")
    vectors = (
        accel_bias_mps2,
        gyro_bias_degps,
        vibration_accel_mps2,
        vibration_gyro_degps,
    )
    if any(len(vector) != 3 for vector in vectors) or not all(
        math.isfinite(float(value)) for vector in vectors for value in vector
    ):
        raise ValueError("fault vectors must contain three finite values")
    if not math.isfinite(time_offset_s):
        raise ValueError("time offset must be finite")
    if not math.isfinite(vibration_frequency_hz) or vibration_frequency_hz <= 0:
        raise ValueError("vibration frequency must be > 0")

    output: list[dict[str, float | int]] = []
    affected = 0
    for source in rows:
        row: dict[str, float | int] = {
            "week": int(source["week"]),
            "tow": float(source["tow"]),
            "ax": float(source["ax"]),
            "ay": float(source["ay"]),
            "az": float(source["az"]),
            "gx": float(source["gx"]),
            "gy": float(source["gy"]),
            "gz": float(source["gz"]),
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("IMU row contains a non-finite value")
        active = start_tow <= float(row["tow"]) <= end_tow
        if not active:
            output.append(row)
            continue
        affected += 1
        if fault == "dropout":
            continue
        if fault == "time_offset":
            tow = float(row["tow"]) + time_offset_s
            week = int(row["week"])
            while tow >= 604800.0:
                tow -= 604800.0
                week += 1
            while tow < 0.0:
                tow += 604800.0
                week -= 1
            row["tow"] = tow
            row["week"] = week
        if fault in {"accel_bias_jump", "bias_jump"}:
            for key, delta in zip(("ax", "ay", "az"), accel_bias_mps2):
                row[key] = float(row[key]) + float(delta)
        if fault in {"gyro_bias_jump", "bias_jump"}:
            for key, delta in zip(("gx", "gy", "gz"), gyro_bias_degps):
                row[key] = float(row[key]) + float(delta)
        if fault == "vibration":
            phase = 2.0 * math.pi * vibration_frequency_hz * (
                float(source["tow"]) - start_tow
            )
            wave = math.sin(phase)
            for key, amplitude in zip(
                ("ax", "ay", "az"), vibration_accel_mps2
            ):
                row[key] = float(row[key]) + float(amplitude) * wave
            for key, amplitude in zip(
                ("gx", "gy", "gz"), vibration_gyro_degps
            ):
                row[key] = float(row[key]) + float(amplitude) * wave
        output.append(row)

    output.sort(key=lambda row: (int(row["week"]), float(row["tow"])))
    for previous, current in zip(output, output[1:]):
        if (int(current["week"]), float(current["tow"])) <= (
            int(previous["week"]),
            float(previous["tow"]),
        ):
            raise ValueError("fault creates duplicate or non-increasing IMU times")
    return output, affected


def _read_imu(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, skipinitialspace=True)
        header = next(reader, None)
        if header is None or len(header) != 8:
            raise ValueError("invalid PPC IMU header")
        for line_number, fields in enumerate(reader, start=2):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) != 8:
                raise ValueError(f"invalid PPC IMU row {line_number}")
            try:
                values = [float(field.strip()) for field in fields]
            except ValueError as exc:
                raise ValueError(f"invalid PPC IMU row {line_number}") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite PPC IMU row {line_number}")
            if values[1] != round(values[1]):
                raise ValueError(f"non-integral GPS week on row {line_number}")
            rows.append(
                dict(
                    tow=values[0],
                    week=int(values[1]),
                    ax=values[2],
                    ay=values[3],
                    az=values[4],
                    gx=values[5],
                    gy=values[6],
                    gz=values[7],
                )
            )
    if not rows:
        raise ValueError("PPC IMU input contains no samples")
    return rows


def _write_imu(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(IMU_HEADER)
        for row in rows:
            writer.writerow(
                (
                    f"{float(row['tow']):.12g}",
                    int(row["week"]),
                    f"{float(row['ax']):.12g}",
                    f"{float(row['ay']):.12g}",
                    f"{float(row['az']):.12g}",
                    f"{float(row['gx']):.12g}",
                    f"{float(row['gy']):.12g}",
                    f"{float(row['gz']):.12g}",
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--fault",
        choices=(
            "dropout",
            "time_offset",
            "accel_bias_jump",
            "gyro_bias_jump",
            "bias_jump",
            "vibration",
        ),
        required=True,
    )
    parser.add_argument("--start-tow", type=float, required=True)
    parser.add_argument("--end-tow", type=float, required=True)
    parser.add_argument("--time-offset-s", type=float, default=0.05)
    parser.add_argument("--accel-bias-mps2", nargs=3, type=float, default=(0.2, 0, 0))
    parser.add_argument("--gyro-bias-degps", nargs=3, type=float, default=(0, 0, 1))
    parser.add_argument("--vibration-frequency-hz", type=float, default=20.0)
    parser.add_argument("--vibration-accel-mps2", nargs=3, type=float, default=(1, 0, 0))
    parser.add_argument("--vibration-gyro-degps", nargs=3, type=float, default=(0, 0, 2))
    args = parser.parse_args(argv)

    source = _read_imu(args.input)
    output, affected = inject_imu_fault(
        source,
        fault=args.fault,
        start_tow=args.start_tow,
        end_tow=args.end_tow,
        time_offset_s=args.time_offset_s,
        accel_bias_mps2=tuple(args.accel_bias_mps2),
        gyro_bias_degps=tuple(args.gyro_bias_degps),
        vibration_frequency_hz=args.vibration_frequency_hz,
        vibration_accel_mps2=tuple(args.vibration_accel_mps2),
        vibration_gyro_degps=tuple(args.vibration_gyro_degps),
    )
    _write_imu(args.output, output)
    manifest = {
        "schema": "gnss_gpu_ppc_imu_fault_v1",
        "truth_usage": "none",
        "fault": args.fault,
        "start_tow": args.start_tow,
        "end_tow": args.end_tow,
        "time_offset_s": args.time_offset_s,
        "accel_bias_mps2": list(args.accel_bias_mps2),
        "gyro_bias_degps": list(args.gyro_bias_degps),
        "vibration_frequency_hz": args.vibration_frequency_hz,
        "vibration_accel_mps2": list(args.vibration_accel_mps2),
        "vibration_gyro_degps": list(args.vibration_gyro_degps),
        "input_rows": len(source),
        "output_rows": len(output),
        "affected_source_rows": affected,
        "input_sha256": _sha256(args.input),
        "output_sha256": _sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
