#!/usr/bin/env python3
"""Build a truth-free route seed from PPC gyro heading and GNSS speed evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _positions(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"ecef_{axis}"]) for axis in "xyz"] for row in rows],
        dtype=np.float64,
    )


def local_enu_basis(position_ecef: np.ndarray) -> np.ndarray:
    """Return rows [East, North, Up] for a local, truth-free ECEF position."""

    position = np.asarray(position_ecef, dtype=np.float64).reshape(3)
    east = np.asarray([-position[1], position[0], 0.0], dtype=np.float64)
    east /= np.linalg.norm(east)
    up = position / np.linalg.norm(position)
    north = np.cross(up, east)
    north /= np.linalg.norm(north)
    return np.vstack([east, north, up])


def select_pre_jump_anchor(
    rows: list[dict[str, str]],
    *,
    min_epoch: int,
    max_epoch: int,
    min_margin: float = 5.0,
    min_history: int = 5,
    jump_residual_m: float = 2.0,
    guard_steps: int = 1,
) -> tuple[int, dict[str, float]]:
    """Select a high-margin anchor just before an assignment-path mode jump."""

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        if min_epoch <= epoch <= max_epoch:
            grouped.setdefault(epoch, []).append(row)
    epochs = sorted(grouped)
    records: list[tuple[int, float, float]] = []
    for epoch in epochs:
        candidates = sorted(grouped[epoch], key=lambda row: int(row["max_marginal_rank"]))
        selected = next((row for row in candidates if int(row["selected"]) == 1), None)
        if selected is None or len(candidates) < 2 or int(selected["max_marginal_rank"]) != 1:
            continue
        margin = float(candidates[0]["max_marginal_relative"]) - float(
            candidates[1]["max_marginal_relative"]
        )
        next_residual = float(selected["next_selected_transition_residual_m"])
        records.append((epoch, margin, next_residual))
    for index, (epoch, _margin, residual) in enumerate(records):
        if residual < jump_residual_m or index < min_history - 1:
            continue
        history = records[index - min_history + 1 : index + 1]
        if min(item[1] for item in history) < min_margin:
            continue
        selected_index = index - int(guard_steps)
        if selected_index < 0:
            break
        anchor_epoch = records[selected_index][0]
        return anchor_epoch, {
            "jump_epoch": float(epoch),
            "jump_residual_m": float(residual),
            "history_min_margin": float(min(item[1] for item in history)),
        }
    raise RuntimeError("no guarded high-margin assignment jump was found")


def select_pre_jump_seed_support_anchor(
    rows: list[dict[str, str]],
    *,
    min_epoch: int,
    max_epoch: int,
    min_support: int = 2,
    min_history: int = 3,
    jump_residual_m: float = 2.0,
    guard_steps: int = 1,
) -> tuple[int, dict[str, float]]:
    """Select a route-supported anchor immediately before a detected mode jump."""

    selected = sorted(
        (
            row
            for row in rows
            if min_epoch <= int(row["epoch"]) <= max_epoch
            and int(row["selected"]) == 1
        ),
        key=lambda row: int(row["epoch"]),
    )
    for index, row in enumerate(selected):
        residual = float(row["next_selected_transition_residual_m"])
        if residual < jump_residual_m or index < min_history - 1:
            continue
        history = selected[index - min_history + 1 : index + 1]
        support = [int(item["current_seed_support"]) for item in history]
        if min(support) < min_support:
            continue
        selected_index = index - int(guard_steps)
        if selected_index < 0:
            break
        return int(selected[selected_index]["epoch"]), {
            "jump_epoch": float(row["epoch"]),
            "jump_residual_m": residual,
            "history_min_seed_support": float(min(support)),
        }
    raise RuntimeError("no guarded route-supported assignment jump was found")


def integrate_gyro_intervals(
    epoch_tows: np.ndarray, imu_tows: np.ndarray, gyro_z_radps: np.ndarray
) -> np.ndarray:
    """Integrate gyro-Z over each consecutive GNSS interval."""

    output = np.zeros(len(epoch_tows) - 1, dtype=np.float64)
    for index, (left, right) in enumerate(zip(epoch_tows[:-1], epoch_tows[1:])):
        mask = (imu_tows >= left) & (imu_tows < right)
        if np.count_nonzero(mask) < 2:
            raise RuntimeError(f"insufficient IMU samples in GNSS interval {index}")
        output[index] = float(np.trapezoid(gyro_z_radps[mask], imu_tows[mask]))
    return output


def build_endpoint_closed_route(
    positions_ecef: np.ndarray,
    doppler_displacements_ecef: np.ndarray,
    tdcp_speed_m: np.ndarray,
    gyro_increments_rad: np.ndarray,
    *,
    start: int,
    end: int,
    gyro_bias_radps: float,
    epoch_dt_s: np.ndarray,
    tdcp_doppler_cap_ratio: float = 1.15,
    heading_correction_stride_epochs: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Use gyro turn shape, robust GNSS speed, and two position endpoints."""

    if not (0 <= start < end < len(positions_ecef)):
        raise ValueError("route segment is invalid")
    basis = local_enu_basis(positions_ecef[start])
    doppler_enu = np.asarray(doppler_displacements_ecef, dtype=np.float64) @ basis.T
    doppler_speed = np.linalg.norm(doppler_enu[:, :2], axis=1)
    gyro_bias = np.asarray(gyro_bias_radps, dtype=np.float64)
    if gyro_bias.ndim == 0:
        gyro_bias = np.full_like(gyro_increments_rad, float(gyro_bias))
    if gyro_bias.shape != np.asarray(gyro_increments_rad).shape:
        raise ValueError("gyro bias profile must match gyro increments")
    corrected_increments = gyro_increments_rad - gyro_bias * epoch_dt_s
    cumulative = np.r_[0.0, np.cumsum(corrected_increments)]
    indices = np.arange(start + 1, end + 1)

    moving = indices[
        np.isfinite(doppler_speed[indices]) & (doppler_speed[indices] > 0.2)
    ]
    if len(moving) < 20:
        raise RuntimeError("insufficient moving Doppler intervals")
    observed = np.arctan2(doppler_enu[moving, 0], doppler_enu[moving, 1])
    sign_scores: list[tuple[float, int, np.ndarray, int]] = []
    for sign in (-1, 1):
        base_full = sign * cumulative
        correction = np.zeros_like(base_full)
        correction_knots = 0
        stride = int(heading_correction_stride_epochs)
        if stride > 0:
            knot_epochs: list[float] = []
            knot_offsets: list[float] = []
            for left in range(start + 1, end + 1, stride):
                window = moving[(moving >= left) & (moving < min(left + stride, end + 1))]
                if len(window) < 5:
                    continue
                residual = observed[np.isin(moving, window)] - base_full[window]
                knot_epochs.append(float(np.mean(window)))
                knot_offsets.append(float(np.angle(np.sum(np.exp(1j * residual)))))
            if len(knot_epochs) >= 2:
                unwrapped = np.unwrap(np.asarray(knot_offsets, dtype=np.float64))
                correction = np.interp(
                    np.arange(len(base_full), dtype=np.float64),
                    np.asarray(knot_epochs),
                    unwrapped,
                )
                correction_knots = len(knot_epochs)
        base = base_full[moving] + correction[moving]
        offset = np.angle(np.sum(np.exp(1j * (observed - base))))
        residual = np.abs(np.angle(np.exp(1j * (base + offset - observed))))
        sign_scores.append(
            (float(np.percentile(residual, 95.0)), sign, correction, correction_knots)
        )
    sign_scores.sort(key=lambda item: item[0])
    heading_p95, gyro_sign, heading_correction, correction_knots = sign_scores[0]

    tdcp_segment_speed = np.asarray(tdcp_speed_m, dtype=np.float64)[indices]
    doppler_segment_speed = doppler_speed[indices]
    tdcp_finite = np.isfinite(tdcp_segment_speed)
    doppler_finite = np.isfinite(doppler_segment_speed)
    speed = np.full_like(tdcp_segment_speed, np.nan)
    both = tdcp_finite & doppler_finite
    speed[both] = np.minimum(
        tdcp_segment_speed[both],
        float(tdcp_doppler_cap_ratio) * doppler_segment_speed[both],
    )
    speed[tdcp_finite & ~doppler_finite] = tdcp_segment_speed[
        tdcp_finite & ~doppler_finite
    ]
    speed[~tdcp_finite & doppler_finite] = doppler_segment_speed[
        ~tdcp_finite & doppler_finite
    ]
    if not np.isfinite(speed).all():
        raise RuntimeError("route speed evidence contains non-finite intervals")
    base_heading = gyro_sign * cumulative[indices] + heading_correction[indices]
    target_enu = (positions_ecef[end] - positions_ecef[start]) @ basis.T
    target_complex = target_enu[1] + 1j * target_enu[0]
    raw_complex = np.sum(speed * np.exp(1j * base_heading))
    if abs(raw_complex) <= 1.0 or abs(target_complex) <= 1.0:
        raise RuntimeError("route endpoint displacement is too short")
    speed_scale = abs(target_complex) / abs(raw_complex)
    heading_offset = np.angle(target_complex) - np.angle(raw_complex)
    heading = base_heading + heading_offset
    horizontal_steps = np.column_stack(
        [speed_scale * speed * np.sin(heading), speed_scale * speed * np.cos(heading)]
    )
    progress = np.r_[0.0, np.cumsum(speed)]
    progress /= progress[-1]
    route_enu = np.zeros((len(indices) + 1, 3), dtype=np.float64)
    route_enu[1:, :2] = np.cumsum(horizontal_steps, axis=0)
    route_enu[:, 2] = progress * target_enu[2]
    route = positions_ecef[start] + route_enu @ basis
    return route, {
        "gyro_sign": float(gyro_sign),
        "doppler_heading_p95_deg": float(np.rad2deg(heading_p95)),
        "doppler_heading_runner_p95_deg": float(np.rad2deg(sign_scores[1][0])),
        "heading_correction_stride_epochs": int(heading_correction_stride_epochs),
        "heading_correction_knots": int(correction_knots),
        "speed_scale": float(speed_scale),
        "tdcp_only_speed_intervals": int(
            np.count_nonzero(tdcp_finite & ~doppler_finite)
        ),
        "doppler_only_speed_intervals": int(
            np.count_nonzero(~tdcp_finite & doppler_finite)
        ),
        "heading_offset_deg": float(np.rad2deg(heading_offset)),
        "endpoint_error_m": float(np.linalg.norm(route[-1] - positions_ecef[end])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("displacements", type=Path)
    parser.add_argument("imu_csv", type=Path)
    parser.add_argument("assignment_candidates", type=Path)
    parser.add_argument("late_anchor_result", type=Path)
    parser.add_argument("--anchor-search-start", type=int, default=800)
    parser.add_argument("--anchor-search-end", type=int, default=950)
    parser.add_argument(
        "--anchor-mode", choices=("margin", "seed-support"), default="margin"
    )
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--speed-cap-ratio", type=float, default=1.15)
    parser.add_argument(
        "--base-seeds",
        type=Path,
        help="Optional existing external seeds to retain in the output",
    )
    parser.add_argument("--out-seeds", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    trajectory_rows = _read_csv(args.trajectory)
    displacement_rows = _read_csv(args.displacements)
    if len(trajectory_rows) != len(displacement_rows):
        raise RuntimeError("trajectory/displacement row counts differ")
    end = int(args.end)
    late_result = json.loads(args.late_anchor_result.read_text(encoding="utf-8"))
    if late_result.get("reason") != "temporal_widelane_consensus":
        raise RuntimeError("late endpoint does not have accepted static fusion evidence")
    if trajectory_rows[end].get("source") != "static_fusion_override":
        raise RuntimeError("late endpoint is not a static-fusion trajectory position")

    assignment_rows = _read_csv(args.assignment_candidates)
    selector = (
        select_pre_jump_anchor
        if args.anchor_mode == "margin"
        else select_pre_jump_seed_support_anchor
    )
    start, jump_metrics = selector(
        assignment_rows,
        min_epoch=int(args.anchor_search_start),
        max_epoch=int(args.anchor_search_end),
    )
    tows = np.asarray([float(row["tow"]) for row in trajectory_rows], dtype=np.float64)
    imu_rows = _read_csv(args.imu_csv)
    imu_tows = np.asarray([float(row["GPS TOW (s)"]) for row in imu_rows])
    gyro_z = np.deg2rad(
        np.asarray([float(row["  Ang Rate Z (deg/s)"]) for row in imu_rows])
    )
    gyro_increments = integrate_gyro_intervals(tows, imu_tows, gyro_z)
    dt = np.diff(tows)
    doppler = np.asarray(
        [[float(row[f"doppler_d{axis}_m"]) for axis in "xyz"] for row in displacement_rows]
    )
    doppler_norm = np.linalg.norm(doppler, axis=1)
    stationary = (doppler_norm[1:] < 0.05) & np.isfinite(gyro_increments)
    if np.count_nonzero(stationary) < 100:
        raise RuntimeError("insufficient Doppler-stationary intervals for gyro bias")
    gyro_bias = float(np.median(gyro_increments[stationary] / dt[stationary]))
    if abs(np.rad2deg(gyro_bias)) > 0.5:
        raise RuntimeError("gyro bias exceeds fail-closed gate")
    tdcp_speed = np.asarray([float(row["norm_m"]) for row in displacement_rows])
    route, route_metrics = build_endpoint_closed_route(
        _positions(trajectory_rows),
        doppler,
        tdcp_speed,
        gyro_increments,
        start=start,
        end=end,
        gyro_bias_radps=gyro_bias,
        epoch_dt_s=dt,
        tdcp_doppler_cap_ratio=float(args.speed_cap_ratio),
    )
    if route_metrics["doppler_heading_p95_deg"] > 15.0:
        raise RuntimeError("gyro/Doppler heading coherence exceeds gate")
    if not 0.8 <= route_metrics["speed_scale"] <= 1.2:
        raise RuntimeError("route speed scale exceeds gate")

    output: list[dict[str, Any]] = []
    if args.base_seeds is not None:
        output.extend(_read_csv(args.base_seeds))
    imu_seed_rows = 0
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
                "source": "imu_heading_route_seed",
            }
        )
        imu_seed_rows += 1
    summary = {
        "segment": [start, end + 1],
        "seed_epochs": imu_seed_rows,
        "total_seed_rows": len(output),
        "base_seed_rows": len(output) - imu_seed_rows,
        "imu_seed_rows": imu_seed_rows,
        "gyro_bias_dps": float(np.rad2deg(gyro_bias)),
        "stationary_bias_intervals": int(np.count_nonzero(stationary)),
        "speed_cap_ratio": float(args.speed_cap_ratio),
        "anchor_mode": args.anchor_mode,
        **jump_metrics,
        **route_metrics,
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
