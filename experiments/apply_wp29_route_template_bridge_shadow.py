#!/usr/bin/env python3
"""Bridge an outage with a prior PF-only route template and target progress."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def _positions(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])] for row in rows]
    )


def route_template_bridge(
    target: np.ndarray,
    template: np.ndarray,
    *,
    start: int,
    end: int,
    endpoint_candidates: int,
    max_endpoint_distance_m: float,
    max_arc_relative_error: float,
) -> tuple[np.ndarray, dict]:
    target = np.asarray(target, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    target_arc = float(np.linalg.norm(np.diff(target[start : end + 1], axis=0), axis=1).sum())
    template_cumulative = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(template, axis=0), axis=1))
    ]
    left = np.argsort(np.linalg.norm(template - target[start], axis=1))[
        : int(endpoint_candidates)
    ]
    right = np.argsort(np.linalg.norm(template - target[end], axis=1))[
        : int(endpoint_candidates)
    ]
    options = []
    for i in left:
        for j in right:
            if j <= i:
                continue
            arc = float(template_cumulative[j] - template_cumulative[i])
            arc_relative_error = abs(arc - target_arc) / max(target_arc, 1.0e-9)
            endpoint_distance = float(
                np.linalg.norm(template[i] - target[start])
                + np.linalg.norm(template[j] - target[end])
            )
            options.append((endpoint_distance + target_arc * arc_relative_error, i, j, arc))
    if not options:
        raise RuntimeError("route template has no forward endpoint pair")
    _score, i, j, template_arc = min(options)
    start_distance = float(np.linalg.norm(template[i] - target[start]))
    end_distance = float(np.linalg.norm(template[j] - target[end]))
    relative_error = abs(template_arc - target_arc) / max(target_arc, 1.0e-9)
    if max(start_distance, end_distance) > float(max_endpoint_distance_m):
        raise RuntimeError("route template endpoints are too far from target")
    if relative_error > float(max_arc_relative_error):
        raise RuntimeError("route template arc length is inconsistent")
    target_progress = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(target[start : end + 1], axis=0), axis=1)),
    ]
    template_segment = template[i : j + 1]
    template_progress = template_cumulative[i : j + 1] - template_cumulative[i]
    query = target_progress / max(target_progress[-1], 1.0e-9) * template_progress[-1]
    bridge = np.column_stack(
        [np.interp(query, template_progress, template_segment[:, axis]) for axis in range(3)]
    )
    return bridge, {
        "template_start_index": int(i),
        "template_end_index": int(j),
        "target_arc_m": target_arc,
        "template_arc_m": template_arc,
        "arc_relative_error": relative_error,
        "start_endpoint_distance_m": start_distance,
        "end_endpoint_distance_m": end_distance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_trajectory", type=Path)
    parser.add_argument("--template-trajectory", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--endpoint-candidates", type=int, default=300)
    parser.add_argument("--max-endpoint-distance-m", type=float, default=1.5)
    parser.add_argument("--max-arc-relative-error", type=float, default=0.02)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    args = parser.parse_args()
    with args.target_trajectory.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    with args.template_trajectory.open(newline="", encoding="utf-8-sig") as fh:
        template_rows = list(csv.DictReader(fh))
    positions = _positions(rows)
    bridge, metrics = route_template_bridge(
        positions,
        _positions(template_rows),
        start=int(args.start),
        end=int(args.end),
        endpoint_candidates=int(args.endpoint_candidates),
        max_endpoint_distance_m=float(args.max_endpoint_distance_m),
        max_arc_relative_error=float(args.max_arc_relative_error),
    )
    times, truth_ecef = PPCDatasetLoader(args.data_dir).load_ground_truth()
    output = []
    for row in rows:
        epoch = int(row["epoch"])
        position = (
            bridge[epoch - int(args.start)]
            if int(args.start) <= epoch <= int(args.end)
            else positions[epoch]
        )
        truth = truth_ecef[int(np.argmin(np.abs(times - float(row["tow"]))))]
        error = float(np.linalg.norm(position - truth))
        fix = int(row.get("fix", "0"))
        output.append(
            {
                **row,
                "ecef_x": float(position[0]),
                "ecef_y": float(position[1]),
                "ecef_z": float(position[2]),
                "source": (
                    "route_template_bridge_shadow"
                    if int(args.start) <= epoch <= int(args.end)
                    else row.get("source", "")
                ),
                "error_m": error,
                "sub50cm": int(error < 0.5),
                "false_fix": int(bool(fix) and error >= 0.5),
                "route_template_applied": int(
                    int(args.start) <= epoch <= int(args.end)
                ),
            }
        )
    fixed = [row for row in output if int(row["fix"])]
    summary = {
        "n_epochs_full_denominator": len(output),
        "segment": [int(args.start), int(args.end) + 1],
        **metrics,
        "bridge_sub50cm_epochs": sum(
            int(row["sub50cm"])
            for row in output[int(args.start) : int(args.end) + 1]
        ),
        "sub50cm_full_epochs": sum(int(row["sub50cm"]) for row in output),
        "sub50cm_full_pct": 100.0
        * sum(int(row["sub50cm"]) for row in output)
        / max(len(output), 1),
        "declared_fix_epochs": len(fixed),
        "false_fix_epochs": sum(int(row["false_fix"]) for row in fixed),
        "false_fix_pct": 100.0
        * sum(int(row["false_fix"]) for row in fixed)
        / max(len(fixed), 1),
    }
    args.out_trajectory.parent.mkdir(parents=True, exist_ok=True)
    with args.out_trajectory.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
