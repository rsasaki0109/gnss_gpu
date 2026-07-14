#!/usr/bin/env python3
"""Evaluate candidate-centred 3DMA likelihood on a PPC trajectory span.

This is an offline, truth-free selector: reference positions are read only for
the final evaluation metrics.  Candidate scores use rover pseudoranges, C/N0,
PLATEAU LOS predictions and an optional source trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD_PYTHON = _REPO_ROOT / "build" / "python"
sys.path.insert(0, str(_REPO_ROOT))
if _BUILD_PYTHON.exists():
    sys.path.insert(0, str(_BUILD_PYTHON))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from gnss_gpu.bvh import BVHAccelerator  # noqa: E402
from gnss_gpu.candidate_3dma import (  # noqa: E402
    cn0_to_los_probability,
    horizontal_candidates_ecef,
    multipivot_consensus_scores,
    recurrence_vector_scores,
    road_mode_trigger,
    robust_subset_consensus_scores,
    score_candidate_positions,
    temporal_bias_consistency_scores,
    visibility_mode_cluster_scores,
)
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.spp import _ecef_to_llh, correct_pseudoranges  # noqa: E402
from ppc_window_geometry import load_ppc_window_geometry  # noqa: E402


def _read_pos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    positions: list[list[float]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text or text.startswith(("%", "#")):
                continue
            parts = text.split()
            if len(parts) < 5:
                continue
            try:
                tow, x, y, z = map(float, parts[1:5])
            except ValueError:
                continue
            if np.all(np.isfinite([tow, x, y, z])):
                times.append(tow)
                positions.append([x, y, z])
    if not times:
        raise ValueError(f"no usable positions in {path}")
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(positions, dtype=np.float64)[order]


def _nearest_source(
    times: np.ndarray, positions: np.ndarray, tow: float, tolerance_s: float
) -> np.ndarray | None:
    insertion = int(np.searchsorted(times, tow))
    indices = [i for i in (insertion - 1, insertion) if 0 <= i < len(times)]
    if not indices:
        return None
    best = min(indices, key=lambda idx: abs(float(times[idx]) - tow))
    if abs(float(times[best]) - tow) > tolerance_s:
        return None
    return positions[best]


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def evaluate(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    systems = tuple(part.strip() for part in args.systems.split(",") if part.strip())
    if args.start_tow is not None:
        data = load_ppc_window_geometry(
            args.data_dir,
            start_tow=float(args.start_tow),
            end_tow=float(args.end_tow),
            systems=systems,
            transmit_time_iterations=int(args.transmit_time_iterations),
        )
    else:
        loader = PPCDatasetLoader(args.data_dir)
        data = loader.load_experiment_data(
            max_epochs=args.max_epochs,
            start_epoch=args.start_epoch,
            systems=systems,
        )
    source_times = source_positions = None
    if args.source_pos is not None:
        source_times, source_positions = _read_pos(args.source_pos)
    with np.load(args.triangle_cache_npz) as cache:
        triangles = np.asarray(cache["triangles"], dtype=np.float64)
    bvh = BVHAccelerator(triangles)

    road_union = road_forward = None
    if args.osm_road:
        try:
            from shapely.geometry import Point  # noqa: PLC0415
            from sim_phase68_osm_road_centerline_correction import (  # noqa: PLC0415
                _road_union_from_osm,
            )
        except ImportError as exc:
            raise RuntimeError(
                "--osm-road requires the optional osmnx, pyproj and shapely packages"
            ) from exc
        approximate_llh = [
            _ecef_to_llh(*np.asarray(row, dtype=np.float64))
            for row in np.asarray(data["ground_truth"], dtype=np.float64)
        ]
        margin = float(args.osm_bbox_margin_deg)
        north = max(np.degrees(lat) for lat, _lon, _h in approximate_llh) + margin
        south = min(np.degrees(lat) for lat, _lon, _h in approximate_llh) - margin
        east = max(np.degrees(lon) for _lat, lon, _h in approximate_llh) + margin
        west = min(np.degrees(lon) for _lat, lon, _h in approximate_llh) - margin
        road_union, road_forward, _road_inverse, _edge_count = _road_union_from_osm(
            north=north,
            south=south,
            east=east,
            west=west,
            epsg=int(args.osm_epsg),
        )

    offsets = np.arange(
        -float(args.radius_m),
        float(args.radius_m) + 0.5 * float(args.spacing_m),
        float(args.spacing_m),
        dtype=np.float64,
    )
    zero_offset_index = int(np.argmin(np.abs(offsets)))
    source_candidate_index = zero_offset_index * len(offsets) + zero_offset_index
    east_grid, north_grid = np.meshgrid(offsets, offsets, indexing="xy")
    candidate_displacements = np.hypot(east_grid.ravel(), north_grid.ravel())
    rows: list[dict[str, object]] = []
    baseline_errors: list[float] = []
    selected_errors: list[float] = []
    oracle_errors: list[float] = []
    score_history: list[np.ndarray] = []
    error_history: list[np.ndarray] = []
    los_history: list[np.ndarray] = []
    innovation_history: list[np.ndarray] = []
    satellite_id_history: list[list[str]] = []
    road_distance_history: list[np.ndarray | None] = []
    improved = worsened = skipped = 0

    for local_idx in range(int(data["n_epochs"])):
        tow = float(data["times"][local_idx])
        truth = np.asarray(data["ground_truth"][local_idx], dtype=np.float64)
        if source_times is not None and source_positions is not None:
            source = _nearest_source(
                source_times, source_positions, tow, float(args.source_time_tolerance_s)
            )
            if source is None:
                skipped += 1
                continue
        else:
            source = horizontal_candidates_ecef(
                truth,
                [float(args.reference_offset_east_m)],
                [float(args.reference_offset_north_m)],
                grid=False,
            )[0]
        satellites = np.asarray(data["sat_ecef"][local_idx], dtype=np.float64)
        pseudoranges = np.asarray(data["pseudoranges"][local_idx], dtype=np.float64)
        cn0 = np.asarray(data["weights"][local_idx], dtype=np.float64)
        system_ids = np.asarray(data["system_ids"][local_idx], dtype=np.int32)
        candidates = horizontal_candidates_ecef(source, offsets, offsets)
        satellite_weights = None
        if args.atmosphere_model == "broadcast":
            pseudoranges, satellite_weights = correct_pseudoranges(
                satellites,
                pseudoranges,
                source,
                tow,
            )
        observed_los = cn0_to_los_probability(
            cn0,
            midpoint_dbhz=float(args.cn0_midpoint_dbhz),
            scale_db=float(args.cn0_scale_db),
        )
        recurrence_precheck = None
        recurrence_precheck_abstained = False
        if (
            args.strategy == "recurrence_vector"
            and float(args.recurrence_max_source_error_m) > 0.0
        ):
            source_los = np.asarray(
                bvh.check_los_batch(
                    source[None, :], satellites[None, :, :].copy()
                ),
                dtype=bool,
            )
            recurrence_precheck = recurrence_vector_scores(
                source[None, :],
                satellites,
                pseudoranges,
                source_los,
                source,
                observed_los_probability=observed_los,
                satellite_weights=satellite_weights,
                clock_group_ids=system_ids,
                max_satellites_per_group=int(args.recurrence_max_satellites),
                sigma_los_m=float(args.recurrence_sigma_los_m),
                nlos_bias_m=float(args.recurrence_nlos_bias_m),
                sigma_nlos_m=float(args.recurrence_sigma_nlos_m),
            )
            precheck_error = float(
                np.median(np.abs(recurrence_precheck.ranging_errors_m[0]))
            )
            recurrence_precheck_abstained = precheck_error > float(
                args.recurrence_max_source_error_m
            )
        if recurrence_precheck_abstained:
            # The selected output is provably the source after the gate. Keep
            # candidate-shaped diagnostics without paying for 169 redundant
            # BVH ray batches whose scores cannot be emitted.
            predicted_los = np.broadcast_to(
                source_los, (len(candidates), len(satellites))
            ).copy()
        else:
            repeated_satellites = np.broadcast_to(
                satellites[None, :, :], (len(candidates), len(satellites), 3)
            ).copy()
            predicted_los = np.asarray(
                bvh.check_los_batch(candidates, repeated_satellites), dtype=bool
            )
        road_outside_distance = None
        road_distances_array = None
        if road_union is not None and road_forward is not None:
            road_distances: list[float] = []
            for candidate in candidates:
                lat, lon, _height = _ecef_to_llh(*candidate)
                x, y = road_forward.transform(np.degrees(lon), np.degrees(lat))
                road_distances.append(float(Point(x, y).distance(road_union)))
            road_distances_array = np.asarray(road_distances, dtype=np.float64)
            road_outside_distance = np.maximum(
                0.0,
                road_distances_array - float(args.road_corridor_half_width_m),
            )
        result = score_candidate_positions(
            candidates,
            satellites,
            pseudoranges,
            predicted_los,
            satellite_weights=satellite_weights,
            clock_group_ids=system_ids,
            observed_los_probability=observed_los,
            sigma_los_m=float(args.sigma_los_m),
            nlos_bias_m=float(args.nlos_bias_m),
            sigma_nlos_negative_m=float(args.sigma_nlos_negative_m),
            sigma_nlos_positive_m=float(args.sigma_nlos_positive_m),
            visibility_weight=float(args.visibility_weight),
            road_outside_distance_m=road_outside_distance,
            road_sigma_m=float(args.road_sigma_m),
            road_weight=float(args.road_weight),
        )
        selection_scores = result.scores
        recurrence_subset_position_median_m = float("nan")
        recurrence_subset_position_p95_m = float("nan")
        recurrence_source_error_median_m = float("nan")
        recurrence_raw_selected_probability = float("nan")
        recurrence_abstained = False
        recurrence_abstain_reason = ""
        if args.strategy == "multipivot":
            selection_scores = multipivot_consensus_scores(
                result.innovations_m,
                predicted_los,
                observed_los_probability=observed_los,
                satellite_weights=satellite_weights,
                scale_m=float(args.multipivot_scale_m),
                max_pivots=int(args.max_pivots),
            ) + result.visibility_scores + result.road_scores
        elif args.strategy == "robust_subset":
            selection_scores = robust_subset_consensus_scores(
                result.innovations_m,
                predicted_los,
                observed_los_probability=observed_los,
                satellite_weights=satellite_weights,
                scale_m=float(args.subset_scale_m),
                subset_size=int(args.subset_size),
                max_satellites=int(args.subset_max_satellites),
                subset_quantile=float(args.subset_quantile),
            ) + result.visibility_scores + result.road_scores
        elif args.strategy == "recurrence_vector":
            recurrence = (
                recurrence_precheck
                if recurrence_precheck_abstained
                else recurrence_vector_scores(
                    candidates,
                    satellites,
                    pseudoranges,
                    predicted_los,
                    source,
                    observed_los_probability=observed_los,
                    satellite_weights=satellite_weights,
                    clock_group_ids=system_ids,
                    max_satellites_per_group=int(args.recurrence_max_satellites),
                    sigma_los_m=float(args.recurrence_sigma_los_m),
                    nlos_bias_m=float(args.recurrence_nlos_bias_m),
                    sigma_nlos_m=float(args.recurrence_sigma_nlos_m),
                )
            )
            assert recurrence is not None
            selection_scores = (
                np.full(len(candidates), -np.inf, dtype=np.float64)
                if recurrence_precheck_abstained
                else recurrence.scores + result.road_scores
            )
            if recurrence_precheck_abstained:
                selection_scores[source_candidate_index] = 0.0
            subset_distances = np.linalg.norm(
                recurrence.subset_positions_ecef - source[None, :], axis=1
            )
            recurrence_subset_position_median_m = float(np.median(subset_distances))
            recurrence_subset_position_p95_m = float(np.quantile(subset_distances, 0.95))
            recurrence_source_error_median_m = float(
                np.median(
                    np.abs(
                        recurrence.ranging_errors_m[
                            0 if recurrence_precheck_abstained else source_candidate_index
                        ]
                    )
                )
            )
            if recurrence_precheck_abstained or (
                float(args.recurrence_max_source_error_m) > 0.0
                and recurrence_source_error_median_m
                > float(args.recurrence_max_source_error_m)
            ):
                selection_scores = np.full(len(candidates), -np.inf, dtype=np.float64)
                selection_scores[source_candidate_index] = 0.0
                recurrence_abstained = True
                recurrence_abstain_reason = "source_projection_error"
            if not recurrence_abstained and not bool(args.recurrence_allow_boundary):
                recurrence_raw_best = int(np.argmax(selection_scores))
                recurrence_north, recurrence_east = divmod(
                    recurrence_raw_best, len(offsets)
                )
                if recurrence_north in (0, len(offsets) - 1) or recurrence_east in (
                    0,
                    len(offsets) - 1,
                ):
                    selection_scores = np.full(
                        len(candidates), -np.inf, dtype=np.float64
                    )
                    selection_scores[source_candidate_index] = 0.0
                    recurrence_abstained = True
                    recurrence_abstain_reason = "grid_boundary"
            if not recurrence_abstained:
                recurrence_shifted = selection_scores - float(
                    np.max(selection_scores)
                )
                recurrence_probability = np.exp(
                    np.clip(recurrence_shifted, -745.0, 0.0)
                )
                recurrence_probability /= float(np.sum(recurrence_probability))
                recurrence_raw_selected_probability = float(
                    np.max(recurrence_probability)
                )
                if (
                    float(args.recurrence_min_selected_probability) > 0.0
                    and recurrence_raw_selected_probability
                    < float(args.recurrence_min_selected_probability)
                ):
                    selection_scores = np.full(
                        len(candidates), -np.inf, dtype=np.float64
                    )
                    selection_scores[source_candidate_index] = 0.0
                    recurrence_abstained = True
                    recurrence_abstain_reason = "low_selected_probability"
        if args.source_prior_sigma_m > 0.0:
            selection_scores = selection_scores - 0.5 * (
                candidate_displacements / float(args.source_prior_sigma_m)
            ) ** 2
        if args.visibility_cluster:
            selection_scores = visibility_mode_cluster_scores(
                selection_scores,
                predicted_los,
                (len(offsets), len(offsets)),
                score_margin=float(args.cluster_score_margin),
                max_hamming=int(args.cluster_max_hamming),
                outside_penalty=float(args.cluster_outside_penalty),
            )
        selection_probability = np.exp(
            np.clip(selection_scores - float(np.max(selection_scores)), -745.0, 0.0)
        )
        selection_probability /= float(np.sum(selection_probability))
        selected_index = int(np.argmax(selection_scores))
        errors = np.linalg.norm(candidates - truth[None, :], axis=1)
        score_history.append(selection_scores.copy())
        error_history.append(errors)
        los_history.append(predicted_los)
        innovation_history.append(result.innovations_m.copy())
        satellite_id_history.append(list(data["used_prns"][local_idx]))
        road_distance_history.append(
            None if road_distances_array is None else road_distances_array.copy()
        )
        baseline_error = float(np.linalg.norm(source - truth))
        selected_error = float(errors[selected_index])
        oracle_error = float(np.min(errors))
        baseline_errors.append(baseline_error)
        selected_errors.append(selected_error)
        oracle_errors.append(oracle_error)
        improved += int(selected_error + 1.0e-9 < baseline_error)
        worsened += int(selected_error > baseline_error + 1.0e-9)
        north_index, east_index = divmod(selected_index, len(offsets))
        rows.append(
            {
                "epoch": args.start_epoch + local_idx,
                "tow": tow,
                "satellites": len(satellites),
                "baseline_error_m": baseline_error,
                "selected_error_m": selected_error,
                "oracle_error_m": oracle_error,
                "selected_east_m": float(offsets[east_index]),
                "selected_north_m": float(offsets[north_index]),
                "selected_probability": float(selection_probability[selected_index]),
                "selected_score": float(selection_scores[selected_index]),
                "selected_los_count": int(np.count_nonzero(predicted_los[selected_index])),
                "source_score": float(selection_scores[source_candidate_index]),
                "selected_score_gain": float(
                    selection_scores[selected_index]
                    - selection_scores[source_candidate_index]
                ),
                "recurrence_subset_position_median_m": recurrence_subset_position_median_m,
                "recurrence_subset_position_p95_m": recurrence_subset_position_p95_m,
                "recurrence_source_error_median_m": recurrence_source_error_median_m,
                "recurrence_raw_selected_probability": recurrence_raw_selected_probability,
                "recurrence_abstained": recurrence_abstained,
                "recurrence_abstain_reason": recurrence_abstain_reason,
                "source_road_distance_m": (
                    float(road_distances_array[source_candidate_index])
                    if road_distances_array is not None
                    else float("nan")
                ),
                "selected_road_distance_m": (
                    float(road_distances_array[selected_index])
                    if road_distances_array is not None
                    else float("nan")
                ),
            }
        )

    source_road_distances = [
        float(distances[source_candidate_index]) if distances is not None else float("nan")
        for distances in road_distance_history
    ]
    closest_candidate_road_distances = [
        float(np.min(distances)) if distances is not None else float("nan")
        for distances in road_distance_history
    ]
    road_trigger_enabled = float(args.road_trigger_source_distance_m) > 0.0
    road_trigger_applied = not road_trigger_enabled or road_mode_trigger(
        source_road_distances,
        closest_candidate_road_distances_m=closest_candidate_road_distances,
        min_distance_m=float(args.road_trigger_source_distance_m),
        max_candidate_distance_m=float(
            args.road_trigger_max_candidate_distance_m
        ),
        min_contiguous_epochs=int(args.road_trigger_min_contiguous_epochs),
    )
    road_trigger_flags = [
        np.isfinite(distance)
        and distance >= float(args.road_trigger_source_distance_m)
        and np.isfinite(closest_candidate_road_distances[index])
        and closest_candidate_road_distances[index]
        <= float(args.road_trigger_max_candidate_distance_m)
        for index, distance in enumerate(source_road_distances)
    ]

    if (args.selection_mode == "window" or args.strategy == "temporal") and rows:
        if args.strategy == "temporal":
            aggregate_scores = temporal_bias_consistency_scores(
                innovation_history,
                satellite_id_history,
                scale_m=float(args.temporal_scale_m),
                min_epochs_per_satellite=int(args.temporal_min_epochs),
            )
        else:
            aggregate_scores = np.sum(
                np.vstack([scores - float(np.max(scores)) for scores in score_history]),
                axis=0,
            )
        aggregate_probability = np.exp(
            np.clip(aggregate_scores - float(np.max(aggregate_scores)), -745.0, 0.0)
        )
        aggregate_probability /= float(np.sum(aggregate_probability))
        shared_best = int(np.argmax(aggregate_scores))
        if not road_trigger_applied:
            shared_best = source_candidate_index
        north_index, east_index = divmod(shared_best, len(offsets))
        selected_errors = []
        improved = worsened = 0
        for idx, row in enumerate(rows):
            selected_error = float(error_history[idx][shared_best])
            baseline_error = baseline_errors[idx]
            selected_errors.append(selected_error)
            improved += int(selected_error + 1.0e-9 < baseline_error)
            worsened += int(selected_error > baseline_error + 1.0e-9)
            row.update(
                {
                    "selected_error_m": selected_error,
                    "selected_east_m": float(offsets[east_index]),
                    "selected_north_m": float(offsets[north_index]),
                    "selected_probability": float(aggregate_probability[shared_best]),
                    "selected_score": float(aggregate_scores[shared_best]),
                    "selected_los_count": int(
                        np.count_nonzero(los_history[idx][shared_best])
                    ),
                    "source_score": float(aggregate_scores[source_candidate_index]),
                    "selected_score_gain": float(
                        aggregate_scores[shared_best]
                        - aggregate_scores[source_candidate_index]
                    ),
                    "selected_road_distance_m": (
                        float(road_distance_history[idx][shared_best])
                        if road_distance_history[idx] is not None
                        else float("nan")
                    ),
                }
            )
    elif rows and not road_trigger_applied:
        selected_errors = list(baseline_errors)
        improved = worsened = 0
        for idx, row in enumerate(rows):
            source_score = float(score_history[idx][source_candidate_index])
            row.update(
                {
                    "selected_error_m": baseline_errors[idx],
                    "selected_east_m": 0.0,
                    "selected_north_m": 0.0,
                    "selected_probability": float("nan"),
                    "selected_score": source_score,
                    "selected_los_count": int(
                        np.count_nonzero(los_history[idx][source_candidate_index])
                    ),
                    "source_score": source_score,
                    "selected_score_gain": 0.0,
                    "selected_road_distance_m": source_road_distances[idx],
                }
            )

    summary: dict[str, object] = {
        "data_dir": str(args.data_dir),
        "systems": args.systems,
        "source_pos": str(args.source_pos or ""),
        "reference_offset_east_m": args.reference_offset_east_m,
        "reference_offset_north_m": args.reference_offset_north_m,
        "triangle_cache_npz": str(args.triangle_cache_npz),
        "start_epoch": args.start_epoch,
        "start_tow": args.start_tow,
        "end_tow": args.end_tow,
        "transmit_time_iterations": args.transmit_time_iterations,
        "atmosphere_model": args.atmosphere_model,
        "osm_road": args.osm_road,
        "road_corridor_half_width_m": args.road_corridor_half_width_m,
        "road_trigger_source_distance_m": args.road_trigger_source_distance_m,
        "road_trigger_max_candidate_distance_m": (
            args.road_trigger_max_candidate_distance_m
        ),
        "road_trigger_min_contiguous_epochs": args.road_trigger_min_contiguous_epochs,
        "road_trigger_longest_run_epochs": _longest_true_run(road_trigger_flags),
        "road_trigger_applied": road_trigger_applied,
        "requested_epochs": args.max_epochs,
        "evaluated_epochs": len(rows),
        "skipped_epochs": skipped,
        "candidate_count_per_epoch": int(len(offsets) ** 2),
        "selection_mode": args.selection_mode,
        "strategy": args.strategy,
        "visibility_cluster": args.visibility_cluster,
        "source_prior_sigma_m": args.source_prior_sigma_m,
        "radius_m": args.radius_m,
        "spacing_m": args.spacing_m,
        "baseline_p50_m": _percentile(baseline_errors, 50.0),
        "selected_p50_m": _percentile(selected_errors, 50.0),
        "oracle_p50_m": _percentile(oracle_errors, 50.0),
        "baseline_p95_m": _percentile(baseline_errors, 95.0),
        "selected_p95_m": _percentile(selected_errors, 95.0),
        "oracle_p95_m": _percentile(oracle_errors, 95.0),
        "baseline_rms_m": float(np.sqrt(np.mean(np.square(baseline_errors)))) if baseline_errors else float("nan"),
        "selected_rms_m": float(np.sqrt(np.mean(np.square(selected_errors)))) if selected_errors else float("nan"),
        "oracle_rms_m": float(np.sqrt(np.mean(np.square(oracle_errors)))) if oracle_errors else float("nan"),
        "improved_epochs": improved,
        "worsened_epochs": worsened,
        "recurrence_abstained_epochs": int(
            sum(bool(row.get("recurrence_abstained", False)) for row in rows)
        ),
        "recurrence_min_selected_probability": float(
            args.recurrence_min_selected_probability
        ),
        "recurrence_max_source_error_m": float(
            args.recurrence_max_source_error_m
        ),
        "recurrence_allow_boundary": bool(args.recurrence_allow_boundary),
        "recurrence_acceptance_rate": (
            float(
                np.mean(
                    [not bool(row.get("recurrence_abstained", False)) for row in rows]
                )
            )
            if rows and args.strategy == "recurrence_vector"
            else float("nan")
        ),
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-pos", type=Path)
    source.add_argument("--source-from-reference-offset", action="store_true")
    parser.add_argument("--triangle-cache-npz", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-tow", type=float)
    parser.add_argument("--end-tow", type=float)
    parser.add_argument("--transmit-time-iterations", type=int, default=2)
    parser.add_argument("--atmosphere-model", choices=("broadcast", "off"), default="broadcast")
    parser.add_argument("--systems", default="G,R,E,C,J")
    parser.add_argument("--radius-m", type=float, default=3.0)
    parser.add_argument("--spacing-m", type=float, default=0.5)
    parser.add_argument("--source-time-tolerance-s", type=float, default=0.11)
    parser.add_argument("--reference-offset-east-m", type=float, default=0.0)
    parser.add_argument("--reference-offset-north-m", type=float, default=1.7)
    parser.add_argument("--sigma-los-m", type=float, default=3.0)
    parser.add_argument("--nlos-bias-m", type=float, default=15.0)
    parser.add_argument("--sigma-nlos-negative-m", type=float, default=8.0)
    parser.add_argument("--sigma-nlos-positive-m", type=float, default=25.0)
    parser.add_argument("--visibility-weight", type=float, default=1.0)
    parser.add_argument("--selection-mode", choices=("epoch", "window"), default="epoch")
    parser.add_argument(
        "--strategy",
        choices=("absolute", "multipivot", "robust_subset", "recurrence_vector", "temporal"),
        default="absolute",
    )
    parser.add_argument(
        "--recurrence-allow-boundary",
        action="store_true",
        help="allow a recurrence-vector maximum on the candidate-grid boundary",
    )
    parser.add_argument("--multipivot-scale-m", type=float, default=5.0)
    parser.add_argument("--max-pivots", type=int, default=6)
    parser.add_argument("--subset-scale-m", type=float, default=3.0)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--subset-max-satellites", type=int, default=10)
    parser.add_argument("--subset-quantile", type=float, default=0.2)
    parser.add_argument("--recurrence-max-satellites", type=int, default=9)
    parser.add_argument("--recurrence-sigma-los-m", type=float, default=3.0)
    parser.add_argument("--recurrence-nlos-bias-m", type=float, default=15.0)
    parser.add_argument("--recurrence-sigma-nlos-m", type=float, default=20.0)
    parser.add_argument(
        "--recurrence-max-source-error-m",
        type=float,
        default=20.0,
        help="abstain when median projected source ranging error exceeds this; <=0 disables",
    )
    parser.add_argument(
        "--recurrence-min-selected-probability",
        type=float,
        default=0.05,
        help="abstain when the recurrence argmax posterior mass is below this; <=0 disables",
    )
    parser.add_argument("--temporal-scale-m", type=float, default=2.0)
    parser.add_argument("--temporal-min-epochs", type=int, default=8)
    parser.add_argument("--visibility-cluster", action="store_true")
    parser.add_argument("--cluster-score-margin", type=float, default=4.0)
    parser.add_argument("--cluster-max-hamming", type=int, default=1)
    parser.add_argument("--cluster-outside-penalty", type=float, default=5.0)
    parser.add_argument("--osm-road", action="store_true")
    parser.add_argument("--osm-epsg", type=int, default=32653)
    parser.add_argument("--osm-bbox-margin-deg", type=float, default=0.002)
    parser.add_argument("--road-corridor-half-width-m", type=float, default=1.5)
    parser.add_argument("--road-sigma-m", type=float, default=0.75)
    parser.add_argument("--road-weight", type=float, default=1.0)
    parser.add_argument("--road-trigger-source-distance-m", type=float, default=0.0)
    parser.add_argument(
        "--road-trigger-max-candidate-distance-m", type=float, default=0.5
    )
    parser.add_argument("--road-trigger-min-contiguous-epochs", type=int, default=10)
    parser.add_argument("--source-prior-sigma-m", type=float, default=0.0)
    parser.add_argument("--cn0-midpoint-dbhz", type=float, default=32.0)
    parser.add_argument("--cn0-scale-db", type=float, default=4.0)
    args = parser.parse_args()
    if (args.start_tow is None) != (args.end_tow is None):
        parser.error("start-tow and end-tow must be provided together")
    if args.max_epochs <= 0 or args.radius_m <= 0.0 or args.spacing_m <= 0.0:
        parser.error("max-epochs, radius-m and spacing-m must be positive")
    if (
        args.road_trigger_source_distance_m < 0.0
        or args.road_trigger_max_candidate_distance_m < 0.0
        or args.road_trigger_min_contiguous_epochs <= 0
    ):
        parser.error("road trigger distances must be non-negative and epochs positive")
    if not 0.0 <= args.recurrence_min_selected_probability <= 1.0:
        parser.error("recurrence-min-selected-probability must lie in [0, 1]")

    started = time.perf_counter()
    summary, rows = evaluate(args)
    runtime_s = time.perf_counter() - started
    summary["runtime_s"] = float(runtime_s)
    summary["runtime_ms_per_evaluated_epoch"] = (
        1000.0 * float(runtime_s) / len(rows) if rows else float("nan")
    )
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.json")
    epochs_path = args.out_prefix.with_name(args.out_prefix.name + "_epochs.csv")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with epochs_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["epoch"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved: {summary_path}")
    print(f"saved: {epochs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
