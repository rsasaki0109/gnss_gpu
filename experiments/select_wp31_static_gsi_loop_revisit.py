#!/usr/bin/env python3
"""Select a GSI-corrected revisit using an accepted anchor's OSM offset fingerprint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.nmea_writer import _ecef_to_lla_py  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def select_loop_revisit(
    reference_candidates: list[dict[str, Any]],
    reference_selected_id: int,
    reference_road_candidates: list[dict[str, Any]],
    corrected_parents: list[dict[str, Any]],
    target_road_candidates: list[dict[str, Any]],
    height_result: dict[str, Any],
    refined_result: dict[str, Any],
    *,
    min_height_runner_gap_m: float = 1.0,
    max_height_correction_m: float = 2.0,
    max_reference_distance_m: float = 10.0,
    max_road_feature_distance_m: float = 2.0,
    max_road_offset_delta_m: float = 3.0,
    min_road_offset_runner_gap_m: float = 2.0,
    max_refined_update_m: float = 0.25,
    max_refined_norm_rms: float = 0.30,
    min_refined_observations: int = 400,
    max_final_height_residual_m: float = 0.15,
) -> dict[str, Any]:
    base = {"selected_candidate_id": None, "reason": "gsi_loop_revisit_rejected"}
    if height_result.get("reason") != "weak_absolute_height_match":
        return {**base, "reason": "unsupported_height_proposal"}
    parent_id = int(height_result["best_candidate_id"])
    height_gap = float(height_result["runner_gap_m"])
    correction = float(height_result["best_height_residual_m"])
    if height_gap < float(min_height_runner_gap_m):
        return {**base, "reason": "height_parent_not_unique"}
    if correction > float(max_height_correction_m):
        return {**base, "reason": "height_correction_too_large"}

    def one(rows: list[dict[str, Any]], candidate_id: int, label: str) -> dict[str, Any]:
        matches = [row for row in rows if int(row["candidate_id"]) == candidate_id]
        if len(matches) != 1:
            raise ValueError(f"{label} candidate must be unique")
        return matches[0]

    try:
        reference = one(reference_candidates, int(reference_selected_id), "reference")
        reference_road = one(reference_road_candidates, int(reference_selected_id), "reference road")
        target = one(corrected_parents, parent_id, "target")
        target_road = one(target_road_candidates, parent_id, "target road")
    except ValueError:
        return {**base, "reason": "candidate_linkage_invalid"}
    reference_position = np.asarray(reference["position_ecef"], dtype=np.float64)
    target_position = np.asarray(target["position_ecef"], dtype=np.float64)
    if (
        np.linalg.norm(reference_position - np.asarray(reference_road["position_ecef"])) > 1e-6
        or np.linalg.norm(target_position - np.asarray(target_road["position_ecef"])) > 1e-6
    ):
        return {**base, "reason": "road_candidate_position_mismatch"}

    def offset_delta(row: dict[str, Any]) -> float:
        return float(
            np.hypot(
                float(row["road_offset_east_m"]) - float(reference_road["road_offset_east_m"]),
                float(row["road_offset_north_m"]) - float(reference_road["road_offset_north_m"]),
            )
        )

    ranked = sorted(
        (
            {"candidate_id": int(row["candidate_id"]), "road_offset_delta_m": offset_delta(row)}
            for row in target_road_candidates
        ),
        key=lambda row: float(row["road_offset_delta_m"]),
    )
    if len(ranked) < 2:
        return {**base, "reason": "insufficient_road_candidates"}
    best, runner = ranked[:2]
    road_gap = float(runner["road_offset_delta_m"] - best["road_offset_delta_m"])
    reference_distance = float(np.linalg.norm(target_position - reference_position))
    road_feature_distance = float(
        np.hypot(
            float(target_road["nearest_road_east_m"]) - float(reference_road["nearest_road_east_m"]),
            float(target_road["nearest_road_north_m"]) - float(reference_road["nearest_road_north_m"]),
        )
    )
    diagnostics = {
        "height_parent_candidate_id": parent_id,
        "height_runner_gap_m": height_gap,
        "height_correction_m": correction,
        "road_best_candidate_id": int(best["candidate_id"]),
        "road_best_offset_delta_m": float(best["road_offset_delta_m"]),
        "road_runner_candidate_id": int(runner["candidate_id"]),
        "road_runner_offset_delta_m": float(runner["road_offset_delta_m"]),
        "road_offset_runner_gap_m": road_gap,
        "reference_position_distance_m": reference_distance,
        "nearest_road_feature_distance_m": road_feature_distance,
    }
    if int(best["candidate_id"]) != parent_id:
        return {**base, **diagnostics, "reason": "height_road_parent_disagree"}
    if reference_distance > float(max_reference_distance_m):
        return {**base, **diagnostics, "reason": "revisit_position_too_far"}
    if road_feature_distance > float(max_road_feature_distance_m):
        return {**base, **diagnostics, "reason": "nearest_road_feature_changed"}
    if float(best["road_offset_delta_m"]) > float(max_road_offset_delta_m):
        return {**base, **diagnostics, "reason": "road_offset_revisit_too_far"}
    if road_gap < float(min_road_offset_runner_gap_m):
        return {**base, **diagnostics, "reason": "road_offset_revisit_not_unique"}

    refined = list(refined_result.get("candidates", []))
    if len(refined) != 1:
        return {**base, **diagnostics, "reason": "refined_candidate_count_invalid"}
    row = refined[0]
    if np.linalg.norm(np.asarray(refined_result["seed_center_ecef"]) - target_position) > 1e-6:
        return {**base, **diagnostics, "reason": "refined_seed_center_mismatch"}
    if not bool(row.get("applied")) or str(row.get("reason")) != "converged":
        return {**base, **diagnostics, "reason": "dd_refinement_failed"}
    if float(row["update_norm_m"]) > float(max_refined_update_m):
        return {**base, **diagnostics, "reason": "dd_refinement_update_too_large"}
    if float(row["final_norm_rms"]) > float(max_refined_norm_rms):
        return {**base, **diagnostics, "reason": "dd_refinement_residual_too_large"}
    if int(row["n_observations"]) < int(min_refined_observations):
        return {**base, **diagnostics, "reason": "dd_refinement_evidence_too_low"}
    position = np.asarray(row["position_ecef"], dtype=np.float64)
    final_height = float(_ecef_to_lla_py(*position)[2])
    final_height_residual = abs(
        final_height - float(height_result["predicted_antenna_ellipsoid_height_m"])
    )
    refined_diagnostics = {
        **diagnostics,
        "refined_candidate_id": int(row["candidate_id"]),
        "refined_update_m": float(row["update_norm_m"]),
        "refined_norm_rms": float(row["final_norm_rms"]),
        "refined_observations": int(row["n_observations"]),
        "final_height_residual_m": final_height_residual,
    }
    if final_height_residual > float(max_final_height_residual_m):
        return {**base, **refined_diagnostics, "reason": "refined_height_mismatch"}
    return {
        **base,
        **refined_diagnostics,
        "selected_candidate_id": parent_id,
        "reason": "gsi_height_osm_loop_revisit_unique",
        "position_ecef": position.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-candidates", type=Path, required=True)
    parser.add_argument("--reference-fusion", type=Path, required=True)
    parser.add_argument("--reference-road", type=Path, required=True)
    parser.add_argument("--corrected-parents", type=Path, required=True)
    parser.add_argument("--target-road", type=Path, required=True)
    parser.add_argument("--height-result", type=Path, required=True)
    parser.add_argument("--refined-result", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    reference, fusion, reference_road = load(args.reference_candidates), load(args.reference_fusion), load(args.reference_road)
    corrected, target_road, height, refined = load(args.corrected_parents), load(args.target_road), load(args.height_result), load(args.refined_result)
    if fusion.get("reason") != "clear_widelane" or fusion.get("selected_candidate_id") is None:
        parser.error("reference fusion is not an accepted clear-wide-lane anchor")
    result = select_loop_revisit(
        list(reference["candidates"]), int(fusion["selected_candidate_id"]), list(reference_road["candidates"]),
        list(corrected["candidates"]), list(target_road["candidates"]), height, refined,
    )
    result["segment"] = [int(value) for value in corrected["segment"]]
    if result.get("position_ecef") is not None and args.data_dir is not None:
        start, end = result["segment"]
        _times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
        values = np.asarray(truth[start:end], dtype=np.float64)
        values = values[np.isfinite(values).all(axis=1)]
        result["selected_audit_error_m"] = float(np.linalg.norm(np.asarray(result["position_ecef"]) - np.median(values, axis=0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
