#!/usr/bin/env python3
"""Fill weak TDCP intervals with gyro-shaped Doppler-speed dead reckoning."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from build_wp29_imu_heading_route_seed_trace import (  # noqa: E402
    build_endpoint_closed_route,
    integrate_gyro_intervals,
    local_enu_basis,
)
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from run_wp29_tdcp_anchor_smoother import (  # noqa: E402
    _close_static_anchor_gaps,
    _load_fusion_static_override,
    _load_static_position_override,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _vector(row: dict[str, str], prefix: str = "") -> np.ndarray:
    return np.asarray(
        [float(row[f"{prefix}{axis}_m"]) for axis in ("dx", "dy", "dz")],
        dtype=np.float64,
    )


def _circular(value: np.ndarray | float) -> np.ndarray | float:
    return np.angle(np.exp(1j * value))


def load_joint_position_overrides(
    path: Path,
) -> list[tuple[int, int, np.ndarray, int, str]]:
    """Load a fail-closed two-stop anchor selected by independent joint gates."""

    source = json.loads(path.read_text(encoding="utf-8"))
    if not source.get("selected"):
        raise ValueError(f"joint position anchor is not selected: {path}")
    reason = str(source.get("reason", ""))
    if reason != "tdcp_gsi_road_continuity_unique":
        raise ValueError(f"unsupported joint position anchor reason {reason!r}: {path}")
    spans = []
    for side in ("left", "right"):
        segment = source.get(f"{side}_segment")
        position = source.get(f"{side}_position_ecef")
        candidate_id = source.get(f"{side}_selected_candidate_id")
        if (
            not isinstance(segment, list)
            or len(segment) != 2
            or int(segment[0]) >= int(segment[1])
            or not isinstance(position, list)
            or len(position) != 3
            or candidate_id is None
        ):
            raise ValueError(f"invalid {side} joint position anchor: {path}")
        vector = np.asarray(position, dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError(f"nonfinite {side} joint position anchor: {path}")
        spans.append(
            (
                int(segment[0]),
                int(segment[1]),
                vector,
                int(candidate_id),
                reason,
            )
        )
    if spans[1][0] <= spans[0][1]:
        raise ValueError(f"joint position anchor segments overlap: {path}")
    return spans


def _gyro_sign_and_bias(
    tdcp_enu: np.ndarray,
    reliable: np.ndarray,
    gyro_increments: np.ndarray,
    dt_s: np.ndarray,
) -> tuple[int, float, float]:
    horizontal_speed = np.linalg.norm(tdcp_enu[:, :2], axis=1) / np.maximum(dt_s, 1e-6)
    stationary = reliable & (horizontal_speed < 0.05)
    bias_rows = np.flatnonzero(stationary[1:])
    bias = (
        0.0
        if len(bias_rows) < 5
        else float(np.median(gyro_increments[bias_rows] / dt_s[bias_rows + 1]))
    )
    corrected = gyro_increments - bias * dt_s[1:]
    cumulative = np.r_[0.0, np.cumsum(corrected)]
    moving = np.flatnonzero(reliable & (horizontal_speed > 0.5))
    headings = np.arctan2(tdcp_enu[:, 0], tdcp_enu[:, 1])
    pairs = [(a, b) for a, b in zip(moving[:-1], moving[1:]) if 1 <= b - a <= 25]
    if len(pairs) < 10:
        raise RuntimeError("insufficient reliable TDCP heading pairs for gyro sign")
    scores = []
    for sign in (-1, 1):
        residuals = [
            abs(
                float(
                    _circular(
                        headings[b]
                        - headings[a]
                        - sign * (cumulative[b] - cumulative[a])
                    )
                )
            )
            for a, b in pairs
        ]
        scores.append((float(np.percentile(residuals, 95.0)), sign))
    scores.sort()
    return int(scores[0][1]), bias, float(np.rad2deg(scores[0][0]))


def build_hybrid_displacements(
    rows: list[dict[str, str]],
    gyro_increments: np.ndarray,
    basis: np.ndarray,
    *,
    min_tdcp_sats: int,
    max_tdcp_postfit_rms_m: float,
    max_tdcp_norm_m: float,
    max_speed_mps: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    n = len(rows)
    times = np.asarray([float(row["tow"]) for row in rows])
    dt_s = np.r_[0.2, np.diff(times)]
    dt_s[~np.isfinite(dt_s) | (dt_s <= 0.0)] = 0.2
    tdcp = np.asarray([_vector(row) for row in rows])
    doppler = np.asarray([_vector(row, "doppler_") for row in rows])
    reliable = np.asarray(
        [
            row.get("source") == "tdcp"
            and int(row.get("n_used", 0)) >= int(min_tdcp_sats)
            and float(row.get("postfit_rms_m", "inf")) <= float(max_tdcp_postfit_rms_m)
            and np.isfinite(tdcp[index]).all()
            and float(np.linalg.norm(tdcp[index])) <= float(max_tdcp_norm_m)
            for index, row in enumerate(rows)
        ],
        dtype=bool,
    )
    reliable[0] = False
    tdcp_enu = tdcp @ basis.T
    doppler_enu = doppler @ basis.T
    gyro_sign, gyro_bias, gyro_heading_p95_deg = _gyro_sign_and_bias(
        tdcp_enu, reliable, gyro_increments, dt_s
    )
    corrected_gyro = gyro_sign * (gyro_increments - gyro_bias * dt_s[1:])
    cumulative = np.r_[0.0, np.cumsum(corrected_gyro)]
    headings = np.arctan2(tdcp_enu[:, 0], tdcp_enu[:, 1])
    output_enu = tdcp_enu.copy()
    sources = np.where(reliable, "tdcp", "unfilled").astype(object)
    gap_reports: list[dict[str, Any]] = []
    index = 1
    while index < n:
        if reliable[index]:
            index += 1
            continue
        start = index
        while index < n and not reliable[index]:
            index += 1
        end = index - 1
        left = start - 1 if start > 1 and reliable[start - 1] else None
        right = index if index < n and reliable[index] else None
        if left is None or right is None:
            for epoch in range(start, end + 1):
                if np.isfinite(doppler_enu[epoch]).all():
                    output_enu[epoch] = doppler_enu[epoch]
                    sources[epoch] = "doppler_unbounded"
                else:
                    output_enu[epoch] = 0.0
                    sources[epoch] = "hold_unbounded"
            continue
        base_right = headings[left] + cumulative[right] - cumulative[left]
        correction = float(_circular(headings[right] - base_right))
        span_time = max(times[right] - times[left], 1e-6)
        for epoch in range(start, end + 1):
            progress = (times[epoch] - times[left]) / span_time
            heading = (
                headings[left]
                + cumulative[epoch]
                - cumulative[left]
                + correction * progress
            )
            doppler_speed = float(np.linalg.norm(doppler_enu[epoch, :2])) / max(
                dt_s[epoch], 1e-6
            )
            left_speed = float(np.linalg.norm(tdcp_enu[left, :2])) / dt_s[left]
            right_speed = float(np.linalg.norm(tdcp_enu[right, :2])) / dt_s[right]
            boundary_speed = (1.0 - progress) * left_speed + progress * right_speed
            speed = doppler_speed if np.isfinite(doppler_speed) else boundary_speed
            speed = float(np.clip(speed, 0.0, min(max_speed_mps, 1.5 * boundary_speed + 2.0)))
            vertical_rate = (
                (1.0 - progress) * tdcp_enu[left, 2] / dt_s[left]
                + progress * tdcp_enu[right, 2] / dt_s[right]
            )
            output_enu[epoch] = np.asarray(
                [
                    speed * dt_s[epoch] * np.sin(heading),
                    speed * dt_s[epoch] * np.cos(heading),
                    vertical_rate * dt_s[epoch],
                ]
            )
            sources[epoch] = "gyro_doppler_gap_fill"
        gap_reports.append(
            {
                "start": start,
                "end": end + 1,
                "epochs": end - start + 1,
                "duration_s": float(times[end] - times[start] + dt_s[start]),
                "heading_endpoint_correction_deg": float(np.rad2deg(correction)),
            }
        )
    output = output_enu @ basis
    audit_rows = [
        {
            "epoch": epoch,
            "tow": float(times[epoch]),
            "dx_m": float(output[epoch, 0]),
            "dy_m": float(output[epoch, 1]),
            "dz_m": float(output[epoch, 2]),
            "norm_m": float(np.linalg.norm(output[epoch])),
            "source": str(sources[epoch]),
            "tdcp_reliable": int(reliable[epoch]),
            "interval_dt_s": float(dt_s[epoch]),
        }
        for epoch in range(n)
    ]
    return list(output), audit_rows, {
        "tdcp_reliable_intervals": int(np.count_nonzero(reliable)),
        "gyro_gap_fill_intervals": int(np.count_nonzero(sources == "gyro_doppler_gap_fill")),
        "unbounded_fallback_intervals": int(
            np.count_nonzero(np.isin(sources, ["doppler_unbounded", "hold_unbounded"]))
        ),
        "gyro_sign": gyro_sign,
        "gyro_bias_dps": float(np.rad2deg(gyro_bias)),
        "gyro_heading_p95_deg": gyro_heading_p95_deg,
        "gap_count": len(gap_reports),
        "gaps": gap_reports,
    }


def close_anchor_motion_gaps(
    displacements: list[np.ndarray],
    displacement_rows: list[dict[str, Any]],
    spans: list[tuple[int, int, np.ndarray, int, str]],
    *,
    mode: str = "longest",
    duration_exponent: float = 2.0,
    fragmentation_max_dominant_share: float = 0.5,
) -> list[dict[str, Any]]:
    """Endpoint-close anchors over uncertain gyro-filled displacement rows."""

    if mode not in (
        "longest",
        "all_filled",
        "duration_weighted",
        "all_intervals",
        "fragmentation_gated",
    ):
        raise ValueError(f"unsupported motion gap closure mode: {mode}")
    if float(duration_exponent) <= 0.0:
        raise ValueError("duration exponent must be positive")
    if not 0.0 < float(fragmentation_max_dominant_share) <= 1.0:
        raise ValueError("fragmentation dominant-run share must be in (0, 1]")

    reports: list[dict[str, Any]] = []
    for left, right in zip(spans[:-1], spans[1:]):
        start, stop = int(left[1]), int(right[0])
        runs: list[tuple[float, int, int]] = []
        epoch = start
        while epoch <= stop:
            if displacement_rows[epoch]["source"] != "gyro_doppler_gap_fill":
                epoch += 1
                continue
            run_start = epoch
            while (
                epoch <= stop
                and displacement_rows[epoch]["source"] == "gyro_doppler_gap_fill"
            ):
                epoch += 1
            duration = sum(
                float(displacement_rows[item]["interval_dt_s"])
                for item in range(run_start, epoch)
            )
            runs.append((duration, run_start, epoch))
        if not runs and mode != "all_intervals":
            continue
        total_filled_duration = float(sum(run[0] for run in runs))
        dominant_run_share = (
            0.0
            if not runs
            else float(max(run[0] for run in runs) / total_filled_duration)
        )
        effective_mode = mode
        if mode == "fragmentation_gated":
            effective_mode = (
                "all_filled"
                if dominant_run_share <= float(fragmentation_max_dominant_share)
                else "duration_weighted"
            )
        if mode == "all_intervals":
            interval_duration = sum(
                float(displacement_rows[item]["interval_dt_s"])
                for item in range(start, stop + 1)
            )
            selected_runs = [(interval_duration, start, stop + 1)]
        else:
            selected_runs = [max(runs)] if effective_mode == "longest" else runs
        duration = float(sum(run[0] for run in selected_runs))
        selected_epochs = [
            item
            for _run_duration, run_start, run_end in selected_runs
            for item in range(run_start, run_end)
        ]
        raw_delta = np.sum(np.asarray(displacements[start : stop + 1]), axis=0)
        target_delta = np.asarray(right[2]) - np.asarray(left[2])
        correction = target_delta - raw_delta
        run_weight_denominator = sum(
            run_duration ** float(duration_exponent)
            for run_duration, _run_start, _run_end in selected_runs
        )
        epoch_weights: dict[int, float] = {}
        for run_duration, run_start, run_end in selected_runs:
            for item in range(run_start, run_end):
                dt = float(displacement_rows[item]["interval_dt_s"])
                epoch_weights[item] = (
                    dt / duration
                    if effective_mode != "duration_weighted"
                    else dt
                    * run_duration ** (float(duration_exponent) - 1.0)
                    / run_weight_denominator
                )
        for item in selected_epochs:
            weight = epoch_weights[item]
            displacements[item] = np.asarray(displacements[item]) + correction * weight
            vector = np.asarray(displacements[item])
            displacement_rows[item].update(
                {
                    "dx_m": float(vector[0]),
                    "dy_m": float(vector[1]),
                    "dz_m": float(vector[2]),
                    "norm_m": float(np.linalg.norm(vector)),
                    "source": (
                        "anchor_closed_interval_bias"
                        if mode == "all_intervals"
                        else "anchor_closed_gyro_gap_fill"
                    ),
                }
            )
        reports.append(
            {
                "left_candidate_id": int(left[3]),
                "right_candidate_id": int(right[3]),
                "left_epoch": int(left[1] - 1),
                "right_epoch": int(right[0]),
                "closure_mode": mode,
                "effective_closure_mode": effective_mode,
                "duration_exponent": (
                    float(duration_exponent)
                    if effective_mode == "duration_weighted"
                    else None
                ),
                "dominant_filled_run_share": dominant_run_share,
                "fragmentation_max_dominant_share": (
                    float(fragmentation_max_dominant_share)
                    if mode == "fragmentation_gated"
                    else None
                ),
                "gap_start": min(selected_epochs),
                "gap_end": max(selected_epochs) + 1,
                "selected_gap_runs": len(selected_runs),
                "selected_gap_epochs": len(selected_epochs),
                "gap_duration_s": duration,
                "endpoint_residual_m": float(np.linalg.norm(correction)),
                "correction_ecef_m": correction.tolist(),
            }
        )
    return reports


def bridge_long_gyro_routes(
    displacements: list[np.ndarray],
    displacement_rows: list[dict[str, Any]],
    spans: list[tuple[int, int, np.ndarray, int, str]],
    *,
    times: np.ndarray,
    gyro_increments: np.ndarray,
    gyro_bias_radps: float,
    doppler_displacements: np.ndarray,
    min_gap_duration_s: float = 30.0,
    min_longest_runner_ratio: float = 2.0,
    heading_correction_stride_epochs: int = 25,
    max_heading_p95_deg: float = 15.0,
    min_speed_scale: float = 0.8,
    max_speed_scale: float = 1.2,
) -> list[dict[str, Any]]:
    """Replace one dominant long gyro gap with a truth-free curved route.

    The two gap-boundary positions are derived independently: forward from the
    accepted left anchor and backward from the accepted right anchor.  Only a
    duration-dominant gap with coherent gyro/Doppler heading and a bounded
    endpoint speed scale is modified.
    """

    if min_gap_duration_s <= 0.0 or min_longest_runner_ratio <= 1.0:
        raise ValueError("positive gap duration and runner ratio > 1 are required")
    times = np.asarray(times, dtype=np.float64)
    doppler = np.asarray(doppler_displacements, dtype=np.float64)
    if len(displacements) != len(displacement_rows) or doppler.shape != (len(displacements), 3):
        raise ValueError("route-bridge displacement inputs are inconsistent")
    dt = np.diff(times)
    reports: list[dict[str, Any]] = []
    for left, right in zip(spans[:-1], spans[1:]):
        pair_start, pair_stop = int(left[1]), int(right[0])
        runs: list[tuple[float, int, int]] = []
        epoch = pair_start
        while epoch <= pair_stop:
            if displacement_rows[epoch]["source"] != "gyro_doppler_gap_fill":
                epoch += 1
                continue
            run_start = epoch
            while epoch <= pair_stop and displacement_rows[epoch]["source"] == "gyro_doppler_gap_fill":
                epoch += 1
            duration = sum(float(displacement_rows[item]["interval_dt_s"]) for item in range(run_start, epoch))
            runs.append((duration, run_start, epoch))
        if not runs:
            continue
        ranked = sorted(runs, reverse=True)
        duration, run_start, run_end = ranked[0]
        runner_duration = ranked[1][0] if len(ranked) > 1 else 0.0
        ratio = float("inf") if runner_duration <= 0.0 else duration / runner_duration
        base = {
            "left_candidate_id": int(left[3]),
            "right_candidate_id": int(right[3]),
            "gap_start": int(run_start),
            "gap_end": int(run_end),
            "gap_duration_s": float(duration),
            "runner_gap_duration_s": float(runner_duration),
            "longest_runner_duration_ratio": float(ratio),
            "applied": False,
        }
        if duration < float(min_gap_duration_s):
            reports.append({**base, "reason": "long_gap_too_short"})
            continue
        if ratio < float(min_longest_runner_ratio):
            reports.append({**base, "reason": "long_gap_not_duration_unique"})
            continue

        left_boundary = np.asarray(left[2], dtype=np.float64) + np.sum(
            np.asarray(displacements[pair_start:run_start]), axis=0
        )
        right_boundary = np.asarray(right[2], dtype=np.float64) - np.sum(
            np.asarray(displacements[run_end : pair_stop + 1]), axis=0
        )
        start_epoch, end_epoch = run_start - 1, run_end - 1
        endpoint_positions = np.zeros((len(displacements), 3), dtype=np.float64)
        endpoint_positions[start_epoch] = left_boundary
        endpoint_positions[end_epoch] = right_boundary
        hybrid_speed = np.asarray(
            [float(np.linalg.norm(vector)) for vector in displacements],
            dtype=np.float64,
        )
        try:
            route, metrics = build_endpoint_closed_route(
                endpoint_positions,
                doppler,
                hybrid_speed,
                np.asarray(gyro_increments, dtype=np.float64),
                start=start_epoch,
                end=end_epoch,
                gyro_bias_radps=float(gyro_bias_radps),
                epoch_dt_s=dt,
                heading_correction_stride_epochs=int(heading_correction_stride_epochs),
            )
        except (RuntimeError, ValueError) as error:
            reports.append({**base, "reason": "route_build_failed", "detail": str(error)})
            continue
        diagnostics = {
            **base,
            **metrics,
            "boundary_displacement_m": float(np.linalg.norm(right_boundary - left_boundary)),
        }
        if float(metrics["doppler_heading_p95_deg"]) > float(max_heading_p95_deg):
            reports.append({**diagnostics, "reason": "route_heading_incoherent"})
            continue
        if not float(min_speed_scale) <= float(metrics["speed_scale"]) <= float(max_speed_scale):
            reports.append({**diagnostics, "reason": "route_speed_scale_out_of_bounds"})
            continue
        steps = np.diff(route, axis=0)
        for offset, item in enumerate(range(run_start, run_end)):
            vector = steps[offset]
            displacements[item] = vector
            displacement_rows[item].update(
                {
                    "dx_m": float(vector[0]),
                    "dy_m": float(vector[1]),
                    "dz_m": float(vector[2]),
                    "norm_m": float(np.linalg.norm(vector)),
                    "source": "anchor_closed_gyro_route",
                }
            )
        reports.append({**diagnostics, "applied": True, "reason": "dominant_long_gyro_route_closed"})
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("displacements", type=Path)
    parser.add_argument("imu_csv", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--static-anchor", type=Path, nargs=2, action="append", required=True)
    parser.add_argument("--position-anchor", type=Path, action="append", default=[])
    parser.add_argument(
        "--development-position-anchor", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--joint-position-anchor", type=Path, action="append", default=[]
    )
    parser.add_argument("--min-tdcp-sats", type=int, default=8)
    parser.add_argument("--max-tdcp-postfit-rms-m", type=float, default=0.05)
    parser.add_argument("--max-tdcp-norm-m", type=float, default=3.5)
    parser.add_argument("--max-speed-mps", type=float, default=20.0)
    parser.add_argument("--static-gap-endpoint-closure", action="store_true")
    parser.add_argument("--motion-gap-endpoint-closure", action="store_true")
    parser.add_argument("--long-gyro-route-bridge", action="store_true")
    parser.add_argument("--long-gyro-route-min-duration-s", type=float, default=30.0)
    parser.add_argument("--long-gyro-route-min-runner-ratio", type=float, default=2.0)
    parser.add_argument("--long-gyro-route-heading-stride", type=int, default=25)
    parser.add_argument("--long-gyro-route-max-heading-p95-deg", type=float, default=15.0)
    parser.add_argument("--long-gyro-route-min-speed-scale", type=float, default=0.8)
    parser.add_argument("--long-gyro-route-max-speed-scale", type=float, default=1.2)
    parser.add_argument(
        "--motion-gap-closure-mode",
        choices=(
            "longest",
            "all_filled",
            "duration_weighted",
            "all_intervals",
            "fragmentation_gated",
        ),
        default="longest",
    )
    parser.add_argument("--motion-gap-duration-exponent", type=float, default=2.0)
    parser.add_argument(
        "--motion-gap-fragmentation-max-dominant-share",
        type=float,
        default=0.5,
    )
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    parser.add_argument("--out-displacements", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_csv(args.displacements)
    imu = _read_csv(args.imu_csv)
    times = np.asarray([float(row["tow"]) for row in rows])
    imu_times = np.asarray([float(row["GPS TOW (s)"]) for row in imu])
    gyro_z = np.deg2rad(np.asarray([float(row["  Ang Rate Z (deg/s)"]) for row in imu]))
    gyro_increments = integrate_gyro_intervals(times, imu_times, gyro_z)
    spans = [
        _load_fusion_static_override(static_path, fusion_path)
        for static_path, fusion_path in args.static_anchor
    ]
    spans.extend(_load_static_position_override(path) for path in args.position_anchor)
    for path in args.development_position_anchor:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("production_promoted") is not False:
            raise RuntimeError("development position anchor must be explicitly unpromoted")
        if result.get("reason") != "compact_multimode_rank_cluster_development":
            raise RuntimeError("unsupported development position anchor reason")
        selected_ids = [int(value) for value in result.get("selected_candidate_ids", [])]
        if len(selected_ids) < 3:
            raise RuntimeError("development multimode anchor has insufficient members")
        start, end = (int(value) for value in result["segment"])
        position = np.asarray(result["position_ecef"], dtype=np.float64).reshape(3)
        if end <= start or not np.isfinite(position).all():
            raise RuntimeError("development position anchor is invalid")
        spans.append((start, end, position, -1, str(result["reason"])))
    for path in args.joint_position_anchor:
        spans.extend(load_joint_position_overrides(path))
    spans.sort(key=lambda item: item[0])
    basis = local_enu_basis(spans[0][2])
    displacements, displacement_rows, motion_summary = build_hybrid_displacements(
        rows,
        gyro_increments,
        basis,
        min_tdcp_sats=args.min_tdcp_sats,
        max_tdcp_postfit_rms_m=args.max_tdcp_postfit_rms_m,
        max_tdcp_norm_m=args.max_tdcp_norm_m,
        max_speed_mps=args.max_speed_mps,
    )
    route_bridge_reports = []
    if args.long_gyro_route_bridge:
        doppler = np.asarray([_vector(row, "doppler_") for row in rows])
        route_bridge_reports = bridge_long_gyro_routes(
            displacements,
            displacement_rows,
            spans,
            times=times,
            gyro_increments=gyro_increments,
            gyro_bias_radps=np.deg2rad(float(motion_summary["gyro_bias_dps"])),
            doppler_displacements=doppler,
            min_gap_duration_s=args.long_gyro_route_min_duration_s,
            min_longest_runner_ratio=args.long_gyro_route_min_runner_ratio,
            heading_correction_stride_epochs=args.long_gyro_route_heading_stride,
            max_heading_p95_deg=args.long_gyro_route_max_heading_p95_deg,
            min_speed_scale=args.long_gyro_route_min_speed_scale,
            max_speed_scale=args.long_gyro_route_max_speed_scale,
        )
    closure_reports = []
    if args.static_gap_endpoint_closure:
        closure_reports = _close_static_anchor_gaps(
            displacements, times, spans, nominal_dt_s=0.2
        )
    motion_closure_reports = []
    if args.motion_gap_endpoint_closure:
        motion_closure_reports = close_anchor_motion_gaps(
            displacements,
            displacement_rows,
            spans,
            mode=args.motion_gap_closure_mode,
            duration_exponent=args.motion_gap_duration_exponent,
            fragmentation_max_dominant_share=(
                args.motion_gap_fragmentation_max_dominant_share
            ),
        )
    predicted = np.full((len(rows), 3), np.nan)
    for start, end, position, _candidate_id, _reason in spans:
        predicted[start:end] = position
    for epoch in range(spans[0][0] - 1, -1, -1):
        predicted[epoch] = predicted[epoch + 1] - displacements[epoch + 1]
    for left, right in zip(spans[:-1], spans[1:]):
        for epoch in range(left[1], right[0] + 1):
            predicted[epoch] = predicted[epoch - 1] + displacements[epoch]
        predicted[right[0] : right[1]] = right[2]
    for epoch in range(spans[-1][1], len(rows)):
        predicted[epoch] = predicted[epoch - 1] + displacements[epoch]

    truth_times, truth_positions = PPCDatasetLoader(args.data_dir).load_ground_truth()
    truth = np.asarray(
        [truth_positions[int(np.argmin(np.abs(truth_times - tow)))] for tow in times]
    )
    errors = np.linalg.norm(predicted - truth, axis=1)
    trajectory = [
        {
            "epoch": epoch,
            "tow": float(times[epoch]),
            "ecef_x": float(predicted[epoch, 0]),
            "ecef_y": float(predicted[epoch, 1]),
            "ecef_z": float(predicted[epoch, 2]),
            "error_m": float(errors[epoch]),
            "sub50cm": int(errors[epoch] < 0.5),
            "fix": 0,
            "false_fix": 0,
        }
        for epoch in range(len(rows))
    ]
    summary = {
        "n_epochs_full_denominator": len(rows),
        "development_anchor_used": bool(args.development_position_anchor),
        "production_promoted": not bool(args.development_position_anchor),
        **motion_summary,
        "static_anchor_spans": [
            {"start": s, "end": e, "candidate_id": cid, "reason": reason}
            for s, e, _p, cid, reason in spans
        ],
        "static_gap_closure_reports": closure_reports,
        "long_gyro_route_bridge_reports": route_bridge_reports,
        "motion_gap_closure_reports": motion_closure_reports,
        "sub50cm_full_epochs": int(np.count_nonzero(errors < 0.5)),
        "sub50cm_full_pct": float(100.0 * np.mean(errors < 0.5)),
        "declared_fix_epochs": 0,
        "false_fix_epochs": 0,
        "false_fix_pct": 0.0,
    }
    for path, records in (
        (args.out_trajectory, trajectory),
        (args.out_displacements, displacement_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
