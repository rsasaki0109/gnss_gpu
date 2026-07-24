#!/usr/bin/env python3
"""Build truth-free RBPF position seeds from a cross-run temporal route trace."""

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
        [
            [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])]
            for row in rows
        ],
        dtype=np.float64,
    )


def _displacements(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [
            [float(row["dx_m"]), float(row["dy_m"]), float(row["dz_m"])]
            for row in rows
        ],
        dtype=np.float64,
    )


def integrate_static_anchored_template(
    displacements: np.ndarray,
    anchor_position: np.ndarray,
    *,
    start: int,
    static_start: int,
    static_end: int,
    end: int,
) -> np.ndarray:
    """Integrate temporal displacement on both sides of an accepted static stop."""

    displacements = np.asarray(displacements, dtype=np.float64)
    anchor_position = np.asarray(anchor_position, dtype=np.float64).reshape(3)
    if not (0 <= start < static_start < static_end <= end < len(displacements)):
        raise ValueError("template segment/static ordering is invalid")
    if not np.isfinite(displacements[start : end + 1]).all():
        raise ValueError("template displacements must be finite")
    positions = np.tile(anchor_position, (end - start + 1, 1))
    for epoch in range(static_start - 1, start - 1, -1):
        positions[epoch - start] = (
            positions[epoch + 1 - start] - displacements[epoch + 1]
        )
    for epoch in range(static_end, end + 1):
        positions[epoch - start] = (
            positions[epoch - 1 - start] + displacements[epoch]
        )
    return positions


def retime_and_close_template(
    target_positions: np.ndarray,
    target_doppler_steps: np.ndarray,
    template_positions: np.ndarray,
    *,
    max_endpoint_closure_m: float,
    max_arc_relative_error: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Re-time a template by Doppler progress and close it to trusted endpoints."""

    target_positions = np.asarray(target_positions, dtype=np.float64)
    target_doppler_steps = np.asarray(target_doppler_steps, dtype=np.float64)
    template_positions = np.asarray(template_positions, dtype=np.float64)
    if len(target_positions) != len(target_doppler_steps) + 1:
        raise ValueError("target Doppler steps must span consecutive target positions")
    if len(target_positions) < 2 or len(template_positions) < 2:
        raise ValueError("route seed needs at least two positions")
    if not np.isfinite(target_doppler_steps).all() or np.any(target_doppler_steps < 0.0):
        raise ValueError("target Doppler progress must be finite and non-negative")
    target_arc = float(np.sum(target_doppler_steps))
    template_steps = np.linalg.norm(np.diff(template_positions, axis=0), axis=1)
    template_arc = float(np.sum(template_steps))
    if target_arc <= 0.0 or template_arc <= 0.0:
        raise ValueError("route arc must be positive")
    start_closure = float(np.linalg.norm(template_positions[0] - target_positions[0]))
    end_closure = float(np.linalg.norm(template_positions[-1] - target_positions[-1]))
    arc_relative_error = abs(template_arc - target_arc) / target_arc
    if max(start_closure, end_closure) > float(max_endpoint_closure_m):
        raise RuntimeError("route template endpoint closure exceeds gate")
    if arc_relative_error > float(max_arc_relative_error):
        raise RuntimeError("route template Doppler arc exceeds gate")
    progress = np.r_[0.0, np.cumsum(target_doppler_steps)] / target_arc
    template_cumulative = np.r_[0.0, np.cumsum(template_steps)]
    query = progress * template_arc
    seed = np.column_stack(
        [
            np.interp(query, template_cumulative, template_positions[:, axis])
            for axis in range(3)
        ]
    )
    alpha = progress.reshape(-1, 1)
    seed += (1.0 - alpha) * (target_positions[0] - seed[0])
    seed += alpha * (target_positions[-1] - seed[-1])
    return seed, {
        "target_doppler_arc_m": target_arc,
        "template_temporal_arc_m": template_arc,
        "arc_relative_error": arc_relative_error,
        "start_endpoint_closure_m": start_closure,
        "end_endpoint_closure_m": end_closure,
    }


def expand_axis_position_seeds(
    position: np.ndarray, radii_m: tuple[float, ...]
) -> list[tuple[np.ndarray, float]]:
    """Return a center plus deterministic ECEF-axis route uncertainty seeds."""

    center = np.asarray(position, dtype=np.float64).reshape(3)
    if any(not np.isfinite(radius) or radius <= 0.0 for radius in radii_m):
        raise ValueError("route seed radii must be finite and positive")
    output = [(center.copy(), 0.0)]
    for radius in radii_m:
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = float(radius)
            output.append((center + offset, -float(radius)))
            output.append((center - offset, -float(radius)))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_trajectory", type=Path)
    parser.add_argument("target_displacements", type=Path)
    parser.add_argument("template_displacements", type=Path)
    parser.add_argument("template_static_anchor", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--template-start", type=int, required=True)
    parser.add_argument("--template-static-start", type=int, required=True)
    parser.add_argument("--template-static-end", type=int, required=True)
    parser.add_argument("--template-end", type=int, required=True)
    parser.add_argument("--max-endpoint-closure-m", type=float, default=20.0)
    parser.add_argument("--max-arc-relative-error", type=float, default=0.05)
    parser.add_argument(
        "--seed-radii-m",
        default="",
        help="Optional comma-separated ECEF-axis uncertainty radii",
    )
    parser.add_argument("--out-seeds", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    target_rows = _read_csv(args.target_trajectory)
    target_displacement_rows = _read_csv(args.target_displacements)
    template_displacement_rows = _read_csv(args.template_displacements)
    if len(target_rows) != len(target_displacement_rows):
        raise RuntimeError("target trajectory/displacement row counts differ")
    start, end = int(args.start), int(args.end)
    if not (0 <= start < end < len(target_rows)):
        raise RuntimeError("target segment is invalid")
    anchor_result = json.loads(args.template_static_anchor.read_text(encoding="utf-8"))
    if str(anchor_result.get("reason")) != "height_temporal_road_consensus":
        raise RuntimeError("template static anchor is not accepted")
    anchor_position = np.asarray(anchor_result["position_ecef"], dtype=np.float64)
    template = integrate_static_anchored_template(
        _displacements(template_displacement_rows),
        anchor_position,
        start=int(args.template_start),
        static_start=int(args.template_static_start),
        static_end=int(args.template_static_end),
        end=int(args.template_end),
    )
    doppler_steps = np.asarray(
        [
            float(row["doppler_norm_m"])
            for row in target_displacement_rows[start + 1 : end + 1]
        ],
        dtype=np.float64,
    )
    seeds, metrics = retime_and_close_template(
        _positions(target_rows)[start : end + 1],
        doppler_steps,
        template,
        max_endpoint_closure_m=float(args.max_endpoint_closure_m),
        max_arc_relative_error=float(args.max_arc_relative_error),
    )
    seed_radii = tuple(
        float(value)
        for value in str(args.seed_radii_m).split(",")
        if value.strip()
    )
    output: list[dict[str, Any]] = []
    for offset, position in enumerate(seeds):
        epoch = start + offset
        for seed_index, (expanded, log_weight) in enumerate(
            expand_axis_position_seeds(position, seed_radii)
        ):
            output.append(
                {
                    "epoch": epoch,
                    "tow": float(target_rows[epoch]["tow"]),
                    "log_weight": log_weight,
                    "ecef_x": float(expanded[0]),
                    "ecef_y": float(expanded[1]),
                    "ecef_z": float(expanded[2]),
                    "source": f"tdcp_route_seed:{seed_index}",
                }
            )
    summary = {
        "segment": [start, end + 1],
        "template_segment": [int(args.template_start), int(args.template_end) + 1],
        "template_static_segment": [
            int(args.template_static_start),
            int(args.template_static_end),
        ],
        "seed_epochs": len(seeds),
        "seeds_per_epoch": 1 + 6 * len(seed_radii),
        "seed_rows": len(output),
        "seed_radii_m": list(seed_radii),
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
