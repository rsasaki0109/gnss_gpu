#!/usr/bin/env python3
"""Build a truth-free IMU/Doppler route between accepted static GNSS anchors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_wp29_imu_heading_route_seed_trace import (
    _positions,
    _read_csv,
    build_endpoint_closed_route,
    integrate_gyro_intervals,
)
from build_wp31_tdcp_gyro_gap_fill import load_joint_position_overrides
from run_wp29_tdcp_anchor_smoother import _load_static_position_override


def _accepted_anchor(
    static_path: Path, fusion_path: Path
) -> tuple[int, int, np.ndarray, int, str]:
    static = json.loads(static_path.read_text(encoding="utf-8"))
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    candidate_id = fusion.get("selected_candidate_id")
    reason = str(fusion.get("reason", ""))
    if candidate_id is None or reason not in (
        "clear_widelane",
        "temporal_widelane_consensus",
        "high_evidence_temporal_widelane_consensus",
    ):
        raise RuntimeError("static endpoint fusion is not accepted")
    matches = [
        row
        for row in static.get("candidates", [])
        if int(row.get("candidate_id", -1)) == int(candidate_id)
    ]
    if len(matches) != 1 or not bool(matches[0].get("applied", False)):
        raise RuntimeError("accepted static endpoint candidate is absent or invalid")
    start, end = (int(value) for value in static["segment"])
    if end <= start:
        raise RuntimeError("static endpoint segment is invalid")
    position = np.asarray(matches[0]["position_ecef"], dtype=np.float64).reshape(3)
    if not np.isfinite(position).all():
        raise RuntimeError("static endpoint position is not finite")
    return start, end, position, int(candidate_id), reason


def resolve_accepted_anchor(
    *,
    static_path: Path | None = None,
    fusion_path: Path | None = None,
    position_path: Path | None = None,
    joint_path: Path | None = None,
    joint_side: str = "left",
) -> tuple[int, int, np.ndarray, int, str]:
    """Resolve exactly one supported, already accepted endpoint artifact."""

    modes = [
        static_path is not None or fusion_path is not None,
        position_path is not None,
        joint_path is not None,
    ]
    if sum(modes) != 1:
        raise ValueError("endpoint requires exactly one accepted anchor source")
    if modes[0]:
        if static_path is None or fusion_path is None:
            raise ValueError("static endpoint requires both static and fusion artifacts")
        return _accepted_anchor(static_path, fusion_path)
    if position_path is not None:
        return _load_static_position_override(position_path)
    if joint_side not in ("left", "right"):
        raise ValueError("joint endpoint side must be left or right")
    spans = load_joint_position_overrides(joint_path)
    return spans[0 if joint_side == "left" else 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("displacements", type=Path)
    parser.add_argument("imu_csv", type=Path)
    parser.add_argument("--start-static", type=Path)
    parser.add_argument("--start-fusion", type=Path)
    parser.add_argument("--start-position-anchor", type=Path)
    parser.add_argument("--start-joint-position-anchor", type=Path)
    parser.add_argument("--start-joint-side", choices=("left", "right"), default="left")
    parser.add_argument("--end-static", type=Path)
    parser.add_argument("--end-fusion", type=Path)
    parser.add_argument("--end-position-anchor", type=Path)
    parser.add_argument("--end-joint-position-anchor", type=Path)
    parser.add_argument("--end-joint-side", choices=("left", "right"), default="left")
    parser.add_argument("--start-epoch", type=int)
    parser.add_argument("--end-epoch", type=int)
    parser.add_argument("--speed-cap-ratio", type=float, default=1.15)
    parser.add_argument(
        "--heading-correction-stride-epochs",
        type=int,
        default=25,
        help="Five-second low-frequency correction at the 5 Hz PPC rate",
    )
    parser.add_argument("--out-seeds", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    trajectory_rows = _read_csv(args.trajectory)
    displacement_rows = _read_csv(args.displacements)
    if len(trajectory_rows) != len(displacement_rows):
        raise RuntimeError("trajectory/displacement row counts differ")
    try:
        start_anchor = resolve_accepted_anchor(
            static_path=args.start_static,
            fusion_path=args.start_fusion,
            position_path=args.start_position_anchor,
            joint_path=args.start_joint_position_anchor,
            joint_side=args.start_joint_side,
        )
        end_anchor = resolve_accepted_anchor(
            static_path=args.end_static,
            fusion_path=args.end_fusion,
            position_path=args.end_position_anchor,
            joint_path=args.end_joint_position_anchor,
            joint_side=args.end_joint_side,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    start = (
        int(args.start_epoch)
        if args.start_epoch is not None
        else (start_anchor[0] + start_anchor[1] - 1) // 2
    )
    end = (
        int(args.end_epoch)
        if args.end_epoch is not None
        else (end_anchor[0] + end_anchor[1] - 1) // 2
    )
    if not start_anchor[0] <= start < start_anchor[1]:
        raise RuntimeError("start epoch is outside its accepted static segment")
    if not end_anchor[0] <= end < end_anchor[1] or end <= start:
        raise RuntimeError("end epoch is outside its accepted static segment")

    tows = np.asarray([float(row["tow"]) for row in trajectory_rows])
    imu_rows = _read_csv(args.imu_csv)
    imu_tows = np.asarray([float(row["GPS TOW (s)"]) for row in imu_rows])
    gyro_z = np.deg2rad(
        np.asarray([float(row["  Ang Rate Z (deg/s)"]) for row in imu_rows])
    )
    gyro_increments = integrate_gyro_intervals(tows, imu_tows, gyro_z)
    dt = np.diff(tows)
    doppler = np.asarray(
        [
            [float(row[f"doppler_d{axis}_m"]) for axis in "xyz"]
            for row in displacement_rows
        ]
    )
    doppler_norm = np.linalg.norm(doppler, axis=1)
    stationary = (doppler_norm[1:] < 0.05) & np.isfinite(gyro_increments)
    if np.count_nonzero(stationary) < 100:
        raise RuntimeError("insufficient Doppler-stationary intervals for gyro bias")
    gyro_bias = float(np.median(gyro_increments[stationary] / dt[stationary]))
    if abs(np.rad2deg(gyro_bias)) > 0.5:
        raise RuntimeError("gyro bias exceeds fail-closed gate")

    def segment_bias(segment_start: int, segment_end: int) -> tuple[float, int]:
        indices = np.arange(max(segment_start, 0), min(segment_end - 1, len(dt)))
        valid = indices[stationary[indices]]
        if len(valid) < 20:
            raise RuntimeError("accepted static segment has insufficient gyro-bias rows")
        value = float(np.median(gyro_increments[valid] / dt[valid]))
        if abs(np.rad2deg(value)) > 0.5:
            raise RuntimeError("static-segment gyro bias exceeds fail-closed gate")
        return value, len(valid)

    start_bias, start_bias_rows = segment_bias(start_anchor[0], start_anchor[1])
    end_bias, end_bias_rows = segment_bias(end_anchor[0], end_anchor[1])
    start_midpoint = 0.5 * (start_anchor[0] + start_anchor[1] - 1)
    end_midpoint = 0.5 * (end_anchor[0] + end_anchor[1] - 1)
    gyro_bias_profile = np.interp(
        np.arange(len(dt), dtype=np.float64),
        [start_midpoint, end_midpoint],
        [start_bias, end_bias],
    )

    positions = _positions(trajectory_rows)
    positions[start] = start_anchor[2]
    positions[end] = end_anchor[2]
    tdcp_speed = np.asarray([float(row["norm_m"]) for row in displacement_rows])
    route, metrics = build_endpoint_closed_route(
        positions,
        doppler,
        tdcp_speed,
        gyro_increments,
        start=start,
        end=end,
        gyro_bias_radps=gyro_bias_profile,
        epoch_dt_s=dt,
        tdcp_doppler_cap_ratio=float(args.speed_cap_ratio),
        heading_correction_stride_epochs=int(args.heading_correction_stride_epochs),
    )
    if metrics["doppler_heading_p95_deg"] > 15.0:
        raise RuntimeError(
            "gyro/Doppler heading coherence exceeds gate: "
            f"{metrics['doppler_heading_p95_deg']:.3f} deg"
        )
    if not 0.8 <= metrics["speed_scale"] <= 1.2:
        raise RuntimeError("route speed scale exceeds gate")

    output: list[dict[str, Any]] = []
    for offset, position in enumerate(route):
        epoch = start + offset
        output.append(
            {
                "epoch": epoch,
                "tow": float(trajectory_rows[epoch]["tow"]),
                "log_weight": 0.0,
                "ecef_x": float(position[0]),
                "ecef_y": float(position[1]),
                "ecef_z": float(position[2]),
                "source": "static_anchor_imu_route_seed",
            }
        )
    summary = {
        "segment": [start, end + 1],
        "seed_epochs": len(output),
        "start_static_segment": list(start_anchor[:2]),
        "end_static_segment": list(end_anchor[:2]),
        "start_candidate_id": start_anchor[3],
        "end_candidate_id": end_anchor[3],
        "start_reason": start_anchor[4],
        "end_reason": end_anchor[4],
        "gyro_bias_dps": float(np.rad2deg(gyro_bias)),
        "start_gyro_bias_dps": float(np.rad2deg(start_bias)),
        "end_gyro_bias_dps": float(np.rad2deg(end_bias)),
        "start_gyro_bias_rows": int(start_bias_rows),
        "end_gyro_bias_rows": int(end_bias_rows),
        "stationary_bias_intervals": int(np.count_nonzero(stationary)),
        "speed_cap_ratio": float(args.speed_cap_ratio),
        **metrics,
    }
    args.out_seeds.parent.mkdir(parents=True, exist_ok=True)
    with args.out_seeds.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
