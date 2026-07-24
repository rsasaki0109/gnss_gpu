#!/usr/bin/env python3
"""Select two static candidates jointly with a trusted TDCP-only edge."""

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

from gnss_gpu.io.nmea_writer import _ecef_to_lla_py  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def select_tdcp_joint_pair(
    left_candidates: list[dict[str, Any]],
    right_candidates: list[dict[str, Any]],
    edge_displacements: np.ndarray,
    edge_sources: list[str],
    *,
    left_target_height_m: float | None = None,
    right_target_height_m: float | None = None,
    max_height_residual_m: float = 0.15,
    min_tdcp_fraction: float = 1.0,
    max_edge_residual_m: float = 0.5,
    min_runner_gap_m: float = 0.1,
    left_road_candidates: list[dict[str, Any]] | None = None,
    right_road_candidates: list[dict[str, Any]] | None = None,
    max_road_continuity_travel_m: float = 50.0,
    max_road_distance_change_m: float = 0.5,
    min_road_runner_gap_m: float = 0.1,
) -> dict[str, Any]:
    base = {"selected": False, "reason": "tdcp_joint_pair_rejected"}
    displacements = np.asarray(edge_displacements, dtype=np.float64).reshape(-1, 3)
    if len(displacements) == 0 or len(edge_sources) != len(displacements):
        return {**base, "reason": "invalid_edge_intervals"}
    if not np.isfinite(displacements).all():
        return {**base, "reason": "nonfinite_edge_displacement"}
    tdcp_fraction = float(sum(source == "tdcp" for source in edge_sources) / len(edge_sources))
    if tdcp_fraction < float(min_tdcp_fraction):
        return {
            **base,
            "reason": "insufficient_tdcp_edge_fraction",
            "edge_interval_count": len(displacements),
            "tdcp_fraction": tdcp_fraction,
        }

    def eligible(
        rows: list[dict[str, Any]], target_height: float | None
    ) -> list[dict[str, Any]]:
        if target_height is None:
            return rows
        return [
            row
            for row in rows
            if abs(_ecef_to_lla_py(*row["position_ecef"])[2] - target_height)
            <= float(max_height_residual_m)
        ]

    left = eligible(left_candidates, left_target_height_m)
    right = eligible(right_candidates, right_target_height_m)
    common = {
        "edge_interval_count": len(displacements),
        "tdcp_fraction": tdcp_fraction,
        "left_eligible_candidate_ids": [int(row["candidate_id"]) for row in left],
        "right_eligible_candidate_ids": [int(row["candidate_id"]) for row in right],
    }
    if not left or not right:
        return {**base, **common, "reason": "no_height_eligible_pair"}
    measured_delta = np.sum(displacements, axis=0)
    pairs = []
    for left_row in left:
        left_position = np.asarray(left_row["position_ecef"], dtype=np.float64)
        for right_row in right:
            right_position = np.asarray(right_row["position_ecef"], dtype=np.float64)
            residual = float(np.linalg.norm((right_position - left_position) - measured_delta))
            pairs.append(
                {
                    "left_candidate_id": int(left_row["candidate_id"]),
                    "right_candidate_id": int(right_row["candidate_id"]),
                    "edge_residual_m": residual,
                    "left_position_ecef": left_position.tolist(),
                    "right_position_ecef": right_position.tolist(),
                }
            )
    pairs.sort(key=lambda row: float(row["edge_residual_m"]))
    if len(pairs) < 2:
        return {**base, **common, "reason": "insufficient_pair_count"}
    best, runner = pairs[:2]
    runner_gap = float(runner["edge_residual_m"] - best["edge_residual_m"])
    diagnostics = {
        **common,
        "measured_delta_ecef_m": measured_delta.tolist(),
        "best_pair": {k: v for k, v in best.items() if not k.endswith("position_ecef")},
        "runner_pair": {k: v for k, v in runner.items() if not k.endswith("position_ecef")},
        "runner_gap_m": runner_gap,
    }
    if (left_road_candidates is None) != (right_road_candidates is None):
        return {**base, **diagnostics, "reason": "incomplete_road_continuity_inputs"}
    if left_road_candidates is not None and right_road_candidates is not None:
        travel_m = float(np.linalg.norm(measured_delta))
        if travel_m > float(max_road_continuity_travel_m):
            return {
                **base,
                **diagnostics,
                "reason": "road_continuity_edge_too_long",
                "edge_travel_m": travel_m,
            }
        left_road = {int(row["candidate_id"]): row for row in left_road_candidates}
        right_road = {int(row["candidate_id"]): row for row in right_road_candidates}
        road_pairs = []
        for pair in pairs:
            if float(pair["edge_residual_m"]) > float(max_edge_residual_m):
                continue
            left_id = int(pair["left_candidate_id"])
            right_id = int(pair["right_candidate_id"])
            if left_id not in left_road or right_id not in right_road:
                return {**base, **diagnostics, "reason": "road_candidate_missing"}
            left_road_position = np.asarray(left_road[left_id]["position_ecef"])
            right_road_position = np.asarray(right_road[right_id]["position_ecef"])
            if (
                np.linalg.norm(left_road_position - np.asarray(pair["left_position_ecef"]))
                > 1e-6
                or np.linalg.norm(
                    right_road_position - np.asarray(pair["right_position_ecef"])
                )
                > 1e-6
            ):
                return {**base, **diagnostics, "reason": "road_candidate_position_mismatch"}
            road_change = abs(
                float(left_road[left_id]["road_distance_m"])
                - float(right_road[right_id]["road_distance_m"])
            )
            road_pairs.append({**pair, "road_distance_change_m": road_change})
        road_pairs.sort(
            key=lambda row: (
                float(row["road_distance_change_m"]),
                float(row["edge_residual_m"]),
            )
        )
        if len(road_pairs) < 2:
            return {**base, **diagnostics, "reason": "insufficient_road_pair_count"}
        road_best, road_runner = road_pairs[:2]
        road_gap = float(
            road_runner["road_distance_change_m"]
            - road_best["road_distance_change_m"]
        )
        road_diagnostics = {
            **diagnostics,
            "edge_travel_m": travel_m,
            "road_best_pair": {
                key: value
                for key, value in road_best.items()
                if not key.endswith("position_ecef")
            },
            "road_runner_pair": {
                key: value
                for key, value in road_runner.items()
                if not key.endswith("position_ecef")
            },
            "road_runner_gap_m": road_gap,
        }
        if float(road_best["road_distance_change_m"]) > float(
            max_road_distance_change_m
        ):
            return {
                **base,
                **road_diagnostics,
                "reason": "weak_road_offset_continuity",
            }
        if road_gap < float(min_road_runner_gap_m):
            return {
                **base,
                **road_diagnostics,
                "reason": "road_offset_continuity_not_unique",
            }
        return {
            **base,
            **road_diagnostics,
            "selected": True,
            "reason": "tdcp_gsi_road_continuity_unique",
            "left_selected_candidate_id": int(road_best["left_candidate_id"]),
            "right_selected_candidate_id": int(road_best["right_candidate_id"]),
            "left_position_ecef": road_best["left_position_ecef"],
            "right_position_ecef": road_best["right_position_ecef"],
        }
    if float(best["edge_residual_m"]) > float(max_edge_residual_m):
        return {**base, **diagnostics, "reason": "weak_absolute_edge_match"}
    if runner_gap < float(min_runner_gap_m):
        return {**base, **diagnostics, "reason": "joint_pair_not_unique"}
    return {
        **base,
        **diagnostics,
        "selected": True,
        "reason": "tdcp_joint_pair_unique",
        "left_selected_candidate_id": int(best["left_candidate_id"]),
        "right_selected_candidate_id": int(best["right_candidate_id"]),
        "left_position_ecef": best["left_position_ecef"],
        "right_position_ecef": best["right_position_ecef"],
    }


def _read_edge(path: Path, start: int, end: int) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = rows[start : end + 1]
    return (
        np.asarray(
            [[float(row["dx_m"]), float(row["dy_m"]), float(row["dz_m"])] for row in selected]
        ),
        [str(row["source"]) for row in selected],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_candidates_json", type=Path)
    parser.add_argument("right_candidates_json", type=Path)
    parser.add_argument("--displacements", type=Path, required=True)
    parser.add_argument("--left-target-height-m", type=float)
    parser.add_argument("--right-target-height-m", type=float)
    parser.add_argument("--left-road-json", type=Path)
    parser.add_argument("--right-road-json", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left_source = json.loads(args.left_candidates_json.read_text(encoding="utf-8"))
    right_source = json.loads(args.right_candidates_json.read_text(encoding="utf-8"))
    left_segment = [int(value) for value in left_source["segment"]]
    right_segment = [int(value) for value in right_source["segment"]]
    if right_segment[0] <= left_segment[1]:
        parser.error("right segment must start after left segment")
    edge, sources = _read_edge(args.displacements, left_segment[1], right_segment[0])
    if (args.left_road_json is None) != (args.right_road_json is None):
        parser.error("--left-road-json and --right-road-json must be used together")
    left_road = (
        json.loads(args.left_road_json.read_text(encoding="utf-8"))["candidates"]
        if args.left_road_json is not None
        else None
    )
    right_road = (
        json.loads(args.right_road_json.read_text(encoding="utf-8"))["candidates"]
        if args.right_road_json is not None
        else None
    )
    result = select_tdcp_joint_pair(
        list(left_source["candidates"]),
        list(right_source["candidates"]),
        edge,
        sources,
        left_target_height_m=args.left_target_height_m,
        right_target_height_m=args.right_target_height_m,
        left_road_candidates=left_road,
        right_road_candidates=right_road,
    )
    result["left_segment"] = left_segment
    result["right_segment"] = right_segment
    if result.get("selected") and args.data_dir is not None:
        _times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
        for side, segment in (("left", left_segment), ("right", right_segment)):
            values = np.asarray(truth[segment[0] : segment[1]], dtype=np.float64)
            values = values[np.isfinite(values).all(axis=1)]
            reference = np.median(values, axis=0)
            result[f"{side}_selected_audit_error_m"] = float(
                np.linalg.norm(np.asarray(result[f"{side}_position_ecef"]) - reference)
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
