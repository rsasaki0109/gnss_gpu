#!/usr/bin/env python3
"""Run the trusted-FIX TDCP Viterbi smoother over a saved RBPF basin trace."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp_ppc_tdcp_velocity import _epoch_measurements  # noqa: E402
from exp_wp23b_float_seed import _doppler_velocity  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.tdcp_anchor_smoother import (  # noqa: E402
    AnchorCandidateEpoch,
    anchored_viterbi_path,
    constrained_assignment_greedy_path,
    constrained_assignment_viterbi_path,
    constrained_assignment_viterbi_audit,
    constrained_greedy_path,
    constrained_viterbi_audit,
    constrained_viterbi_path,
    interpolate_path_position,
)
from gnss_gpu.tdcp_velocity import estimate_displacement_from_tdcp  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
        dtype=np.float64,
    )


def _current_epoch_seed_support(row: dict[str, str], epoch: int) -> int:
    prefix = f"{int(epoch)}:"
    seeds = set()
    for token in str(row.get("proposal_sources", "")).split("|"):
        parts = token.strip().split(":")
        if len(parts) == 2 and token.startswith(prefix) and parts[1].isdigit():
            seeds.add(int(parts[1]))
    return len(seeds)


def _has_recent_external_position_seed(
    row: dict[str, str], epoch: int, max_age_epochs: int
) -> bool:
    """Return whether a basin has recent truth-free external-seed provenance."""

    for token in str(row.get("proposal_sources", "")).split("|"):
        match = re.fullmatch(r"(\d+):external_position:\d+:\d+", token.strip())
        if match is None:
            continue
        age = int(epoch) - int(match.group(1))
        if 0 <= age <= int(max_age_epochs):
            return True
    return False


def _assignment_map(row: dict[str, str]) -> dict[tuple[str, str, int], int]:
    return {
        (str(ref_sat), str(sat_id), int(wavelength_nm)): int(integer)
        for ref_sat, sat_id, wavelength_nm, _generation, integer in json.loads(
            row["assignment_json"]
        )
    }


def _is_snapshot_proposal(row: dict[str, str], epoch: int) -> bool:
    tokens = tuple(
        token for token in str(row.get("proposal_sources", "")).split("|") if token
    )
    return any(
        re.fullmatch(r"\d+:snapshot:\d+", token) is not None
        or token == f"{int(epoch)}:1"
        for token in tokens
    )


def _is_reacquisition_proposal(row: dict[str, str], epoch: int) -> bool:
    if _is_snapshot_proposal(row, epoch):
        return True
    return any(
        ":trusted_float_line:" in token
        for token in str(row.get("proposal_sources", "")).split("|")
    )


def _build_anchor_path_audit(
    by_epoch: dict[int, list[dict[str, str]]],
    path: dict[int, int],
    ground_truth: np.ndarray,
    interval_displacements: dict[tuple[int, int], np.ndarray],
    *,
    static_path_offset: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Describe chosen and oracle anchor branches; truth is audit-only."""

    output: list[dict[str, Any]] = []
    previous_epoch: int | None = None
    for epoch in sorted(path):
        rows = by_epoch[epoch]
        selected_index = int(path[epoch])
        selected = rows[selected_index]
        offset = (
            np.zeros(3, dtype=np.float64)
            if static_path_offset is None
            else np.asarray(static_path_offset, dtype=np.float64).reshape(3)
        )
        positions = np.asarray([_position(row) + offset for row in rows])
        truth = np.asarray(ground_truth[epoch], dtype=np.float64)
        errors = np.linalg.norm(positions - truth.reshape(1, 3), axis=1)
        oracle_index = int(np.argmin(errors))
        selected_assignment = _assignment_map(selected)
        shared_pairs = matches = conflicts = 0
        transition_residual_m = float("nan")
        if previous_epoch is not None:
            previous = by_epoch[previous_epoch][int(path[previous_epoch])]
            previous_assignment = _assignment_map(previous)
            shared = set(previous_assignment) & set(selected_assignment)
            shared_pairs = len(shared)
            matches = sum(
                previous_assignment[key] == selected_assignment[key] for key in shared
            )
            conflicts = shared_pairs - matches
            predicted = _position(previous) + np.asarray(
                interval_displacements[(previous_epoch, epoch)], dtype=np.float64
            ).reshape(3)
            transition_residual_m = float(
                np.linalg.norm(_position(selected) - predicted)
            )
        log_weight = float(selected["log_weight"])
        weight_rank = 1 + sum(float(row["log_weight"]) > log_weight for row in rows)
        oracle = rows[oracle_index]
        output.append(
            {
                "epoch": epoch,
                "selected_index": selected_index,
                "selected_basin_id": selected["basin_id"],
                "selected_assignment_id": selected["assignment_id"],
                "selected_proposal_sources": selected.get("proposal_sources", ""),
                "selected_log_weight_rank": weight_rank,
                "selected_error_m": float(errors[selected_index]),
                "oracle_index": oracle_index,
                "oracle_basin_id": oracle["basin_id"],
                "oracle_assignment_id": oracle["assignment_id"],
                "oracle_proposal_sources": oracle.get("proposal_sources", ""),
                "oracle_error_m": float(errors[oracle_index]),
                "oracle_sub50cm": int(errors[oracle_index] < 0.5),
                "selected_is_oracle_assignment": int(
                    selected["assignment_id"] == oracle["assignment_id"]
                ),
                "shared_pairs_from_previous": shared_pairs,
                "matching_integers_from_previous": matches,
                "conflicting_integers_from_previous": conflicts,
                "transition_residual_m": transition_residual_m,
            }
        )
        previous_epoch = epoch
    return output


def _select_static_anchor_candidate(
    static_result: dict[str, Any],
    *,
    max_norm_rms: float,
    max_runner_up_ratio: float,
    min_bootstrap_wins: int = 4,
    selected_candidate_id: int | None = None,
) -> dict[str, Any]:
    ranked = list(static_result.get("candidates", []))
    if len(ranked) < 2:
        raise RuntimeError("static anchor result needs at least two candidates")
    if selected_candidate_id is not None:
        selected = [
            row
            for row in ranked
            if int(row.get("candidate_id", -1)) == int(selected_candidate_id)
        ]
        if len(selected) != 1:
            raise RuntimeError("fusion-selected static candidate is absent or duplicated")
        if not bool(selected[0].get("applied", False)):
            raise RuntimeError("fusion-selected static candidate was not applied")
        return selected[0]
    best = ranked[0]
    if not bool(best.get("applied", False)):
        raise RuntimeError("best static anchor candidate was not applied")
    best_rms = float(best["final_norm_rms"])
    runner_up_rms = float(ranked[1]["final_norm_rms"])
    bootstrap = list(best.get("bootstrap_norm_rms", []))
    if bootstrap:
        if int(best.get("bootstrap_wins", 0)) < int(min_bootstrap_wins):
            raise RuntimeError("best static anchor candidate fails bootstrap-win gate")
        best_rms = float(np.median(bootstrap))
        runner_up_rms = min(
            float(np.median(row["bootstrap_norm_rms"]))
            for row in ranked[1:]
            if row.get("bootstrap_norm_rms")
        )
    elif best_rms > float(max_norm_rms):
        raise RuntimeError("best static anchor candidate fails normalized-RMS gate")
    ratio = best_rms / max(runner_up_rms, np.finfo(np.float64).eps)
    if ratio > float(max_runner_up_ratio):
        raise RuntimeError("best static anchor candidate is not separated from runner-up")
    return best


def _load_fusion_static_override(
    static_path: Path, fusion_path: Path
) -> tuple[int, int, np.ndarray, int, str]:
    static_result = json.loads(static_path.read_text(encoding="utf-8"))
    fusion_result = json.loads(fusion_path.read_text(encoding="utf-8"))
    selected_id = fusion_result.get("selected_candidate_id")
    reason = str(fusion_result.get("reason", ""))
    if selected_id is None or reason not in (
        "clear_widelane",
        "temporal_widelane_consensus",
        "high_evidence_temporal_widelane_consensus",
    ):
        raise RuntimeError("static fusion override is not accepted")
    matches = [
        row
        for row in static_result.get("candidates", [])
        if int(row.get("candidate_id", -1)) == int(selected_id)
    ]
    if len(matches) != 1 or not bool(matches[0].get("applied", False)):
        raise RuntimeError("static fusion override candidate is absent or invalid")
    start, end = (int(value) for value in static_result["segment"])
    return (
        start,
        end,
        np.asarray(matches[0]["position_ecef"], dtype=np.float64).reshape(3),
        int(selected_id),
        reason,
    )


def _load_static_position_override(
    path: Path,
) -> tuple[int, int, np.ndarray, int, str]:
    result = json.loads(path.read_text(encoding="utf-8"))
    reason = str(result.get("reason", ""))
    if reason not in (
        "height_temporal_road_consensus",
        "motion_supported_child_cluster",
        "compact_widelane_parent_marginal",
        "gsi_ground_height_calibrated",
        "gsi_height_osm_unique_gate",
        "gsi_height_osm_loop_revisit_unique",
        "gsi_osm_carrier_temporal_direction_consensus",
        "gsi_osm_carrier_temporal_cube_consensus",
        "multimode_ddpr_consensus",
        "unique_secondary_topk_posterior",
        "unique_relative_secondary_parent_primary_compact",
        "unique_trifrequency_ddpr_rank_consensus",
    ):
        raise RuntimeError("static position override is not accepted")
    selected_id = result.get("selected_candidate_id")
    if selected_id is None:
        raise RuntimeError("static position override has no selected candidate")
    start, end = (int(value) for value in result["segment"])
    position = np.asarray(result["position_ecef"], dtype=np.float64).reshape(3)
    if end <= start or not np.isfinite(position).all():
        raise RuntimeError("static position override is invalid")
    return start, end, position, int(selected_id), reason


def _robust_static_velocity_bias(samples: list[np.ndarray]) -> np.ndarray | None:
    """Estimate a constant Doppler velocity bias from a known-static segment."""

    if len(samples) < 5:
        return None
    values = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 5:
        return None
    center = np.median(values, axis=0)
    distances = np.linalg.norm(values - center.reshape(1, 3), axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    limit = median_distance + 4.0 * max(mad, 1.0e-3)
    inliers = values[distances <= limit]
    if len(inliers) < 5:
        return None
    return np.median(inliers, axis=0)


def _robust_trusted_fix_velocity_bias(
    samples: list[np.ndarray],
) -> np.ndarray | None:
    """Estimate Doppler bias from guarded-FIX finite-difference velocities."""

    if len(samples) < 5:
        return None
    values = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 5:
        return None
    center = np.median(values, axis=0)
    distances = np.linalg.norm(values - center.reshape(1, 3), axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    limit = median_distance + 4.0 * max(mad, 0.05)
    inliers = values[distances <= limit]
    if len(inliers) < 3:
        return None
    return np.median(inliers, axis=0)


def _resolve_tdcp_fallback(requested: str, anchor_source: str) -> str:
    if requested != "doppler-calibrated-auto":
        return requested
    if anchor_source.startswith("static_stop"):
        return "doppler-calibrated-static"
    return "doppler-calibrated-trusted-fix"


def _tdcp_doppler_gate_reason(
    tdcp_displacement: np.ndarray,
    doppler_displacement: np.ndarray | None,
    *,
    max_vector_difference_m: float,
) -> str | None:
    """Return a truth-free rejection reason for a gross TDCP/Doppler conflict.

    The two carrier-derived estimators have different failure modes.  This gate is
    intentionally loose: at the default 0.75 m per 0.2 s it only rejects a TDCP
    increment when the independent calibrated-Doppler vector disagrees by more
    than 3.75 m/s.  Missing/non-finite Doppler evidence never rejects TDCP.
    """

    if doppler_displacement is None or not np.isfinite(max_vector_difference_m):
        return None
    tdcp = np.asarray(tdcp_displacement, dtype=np.float64).reshape(3)
    doppler = np.asarray(doppler_displacement, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(tdcp)) or not np.all(np.isfinite(doppler)):
        return None
    if float(np.linalg.norm(tdcp - doppler)) > float(max_vector_difference_m):
        return "tdcp_doppler_vector_conflict"
    return None


def _interval_dt_s(times: np.ndarray, epoch: int, nominal_dt_s: float) -> float:
    """Use recorded GNSS time across dropouts, with nominal cadence as fallback."""

    if epoch <= 0:
        return float(nominal_dt_s)
    dt_s = float(times[epoch]) - float(times[epoch - 1])
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return float(nominal_dt_s)
    return dt_s


def _close_static_anchor_gaps(
    displacements: list[np.ndarray],
    times: np.ndarray,
    static_spans: list[tuple[int, int, np.ndarray, int, str]],
    *,
    nominal_dt_s: float,
) -> list[dict[str, Any]]:
    """Put endpoint residual on the largest observed-time gap between static anchors."""

    reports: list[dict[str, Any]] = []
    ordered = sorted(static_spans, key=lambda item: item[0])
    for left, right in zip(ordered[:-1], ordered[1:]):
        left_start, left_end, left_position, left_id, _left_reason = left
        right_start, right_end, right_position, right_id, _right_reason = right
        if left_end > right_start:
            raise RuntimeError("accepted static anchor segments overlap")
        gap_epochs = [
            epoch
            for epoch in range(left_end, right_start + 1)
            if _interval_dt_s(times, epoch, nominal_dt_s) > 1.5 * nominal_dt_s
        ]
        if not gap_epochs:
            continue
        bridge_epoch = max(
            gap_epochs,
            key=lambda epoch: _interval_dt_s(times, epoch, nominal_dt_s),
        )
        raw_delta = np.sum(
            np.asarray(displacements[left_end : right_start + 1]), axis=0
        )
        target_delta = np.asarray(right_position) - np.asarray(left_position)
        correction = target_delta - raw_delta
        displacements[bridge_epoch] = (
            np.asarray(displacements[bridge_epoch], dtype=np.float64) + correction
        )
        reports.append(
            {
                "left_candidate_id": int(left_id),
                "right_candidate_id": int(right_id),
                "left_epoch": int(left_end - 1),
                "right_epoch": int(right_start),
                "bridge_epoch": int(bridge_epoch),
                "bridge_dt_s": _interval_dt_s(times, bridge_epoch, nominal_dt_s),
                "raw_endpoint_residual_m": float(np.linalg.norm(correction)),
                "correction_ecef_m": correction.tolist(),
            }
        )
    return reports


def _resolve_path_mode(
    requested: str, anchor_source: str, *, has_external_route_seed: bool
) -> str:
    """Resolve one production selector from truth-free evidence provenance."""

    if requested != "auto":
        return requested
    if has_external_route_seed:
        return "assignment-viterbi"
    if anchor_source.startswith("static_stop"):
        return "assignment-reacquisition-greedy"
    return "viterbi"


def _resolve_static_anchor_offset(
    requested: bool,
    auto_requested: bool,
    anchor_source: str,
    *,
    has_external_route_seed: bool,
) -> bool:
    """Apply the static offset unless an external route already owns geometry."""

    if not anchor_source.startswith("static_stop"):
        return False
    if requested:
        return True
    return bool(auto_requested and not has_external_route_seed)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics = {int(row["epoch"]): row for row in _read_csv(args.epoch_diagnostics)}
    n_epochs = max(diagnostics) + 1
    static_override = None
    static_override_json = getattr(args, "static_override_json", None)
    static_override_fusion_json = getattr(args, "static_override_fusion_json", None)
    if (static_override_json is None) != (static_override_fusion_json is None):
        raise RuntimeError("static override requires both result and fusion JSON")
    if static_override_json is not None:
        static_override = _load_fusion_static_override(
            static_override_json, static_override_fusion_json
        )
    static_position_override = None
    static_position_override_json = getattr(
        args, "static_position_override_json", None
    )
    if static_position_override_json is not None:
        static_position_override = _load_static_position_override(
            static_position_override_json
        )
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    trace_paths = [args.basin_trace, *args.additional_basin_trace]
    for branch_index, trace_path in enumerate(trace_paths):
        for row in _read_csv(trace_path):
            if int(row["epoch"]) >= n_epochs:
                continue
            prepared = dict(row)
            prepared["branch_id"] = str(branch_index)
            by_epoch[int(row["epoch"])].append(prepared)
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=n_epochs,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )

    first_fix_epoch = next(
        (epoch for epoch in sorted(diagnostics) if diagnostics[epoch]["fix"] == "1"),
        None,
    )
    anchor_stride = int(args.anchor_stride_epochs)
    anchor_epochs = [
        epoch for epoch in sorted(by_epoch) if epoch % anchor_stride == 0
    ]
    anchor_source = "trusted_fix"
    if args.static_anchor_json is not None:
        static_result = json.loads(args.static_anchor_json.read_text(encoding="utf-8"))
        fusion_selected_id = None
        static_anchor_fusion_json = getattr(args, "static_anchor_fusion_json", None)
        if static_anchor_fusion_json is not None:
            fusion_result = json.loads(
                static_anchor_fusion_json.read_text(encoding="utf-8")
            )
            fusion_selected_id = fusion_result.get("selected_candidate_id")
            if fusion_selected_id is None:
                raise RuntimeError("static fusion result did not select a candidate")
        best = _select_static_anchor_candidate(
            static_result,
            max_norm_rms=float(args.static_anchor_max_norm_rms),
            max_runner_up_ratio=float(args.static_anchor_max_runner_up_ratio),
            min_bootstrap_wins=int(args.static_anchor_min_bootstrap_wins),
            selected_candidate_id=(
                None if fusion_selected_id is None else int(fusion_selected_id)
            ),
        )
        static_position = np.asarray(best["position_ecef"], dtype=np.float64)
        segment_start, segment_end = (int(x) for x in static_result["segment"])
        anchor_options = []
        for epoch in anchor_epochs:
            if not (segment_start <= epoch < segment_end):
                continue
            for index, row in enumerate(by_epoch[epoch]):
                distance = float(np.linalg.norm(_position(row) - static_position))
                anchor_options.append((distance, epoch, index))
        if not anchor_options:
            raise RuntimeError("static anchor segment contains no basin anchor epochs")
        distance, trusted_anchor_epoch, trusted_anchor_index = min(anchor_options)
        if distance > float(args.static_anchor_max_candidate_distance_m):
            raise RuntimeError("static anchor is too far from all basin candidates")
        anchor_source = (
            "static_stop" if fusion_selected_id is None else "static_stop_fusion"
        )
    else:
        if first_fix_epoch is None:
            raise RuntimeError("basin replay contains no trusted FIX anchor")
        assignment_id = diagnostics[first_fix_epoch]["map_assignment_id"]
        anchor_options = []
        for epoch in anchor_epochs:
            if abs(epoch - first_fix_epoch) > int(args.anchor_search_epochs):
                continue
            for index, row in enumerate(by_epoch[epoch]):
                if row["assignment_id"] == assignment_id:
                    anchor_options.append((abs(epoch - first_fix_epoch), epoch, index))
        if not anchor_options:
            raise RuntimeError("trusted FIX assignment is absent from nearby anchor epochs")
        _distance, trusted_anchor_epoch, trusted_anchor_index = min(anchor_options)
    trusted_constraints: dict[int, int] = {}
    for fix_epoch in sorted(diagnostics):
        if diagnostics[fix_epoch]["fix"] != "1":
            continue
        fix_assignment = diagnostics[fix_epoch]["map_assignment_id"]
        options: list[tuple[int, int, float, int]] = []
        for epoch in anchor_epochs:
            distance = abs(epoch - fix_epoch)
            if distance > int(args.anchor_search_epochs):
                continue
            for index, row in enumerate(by_epoch[epoch]):
                if row["assignment_id"] == fix_assignment:
                    options.append((distance, epoch, -float(row["log_weight"]), index))
        if options:
            _distance, epoch, _negative_weight, index = min(options)
            trusted_constraints[epoch] = index
    if anchor_source.startswith("static_stop"):
        for epoch in anchor_epochs:
            if not (segment_start <= epoch < segment_end):
                continue
            distances = np.asarray(
                [np.linalg.norm(_position(row) - static_position) for row in by_epoch[epoch]]
            )
            index = int(np.argmin(distances))
            if distances[index] <= float(args.static_anchor_max_candidate_distance_m):
                trusted_constraints[epoch] = index
    static_override_constraints = 0
    if static_override is not None and bool(
        getattr(args, "static_override_constrain_path", False)
    ):
        for epoch in anchor_epochs:
            if not (static_override[0] <= epoch < static_override[1]):
                continue
            distances = np.asarray(
                [
                    np.linalg.norm(_position(row) - static_override[2])
                    for row in by_epoch[epoch]
                ]
            )
            index = int(np.argmin(distances))
            if distances[index] <= float(args.static_override_max_candidate_distance_m):
                trusted_constraints[epoch] = index
                static_override_constraints += 1

    static_anchor_spans: list[tuple[int, int, np.ndarray, int, str]] = []
    if anchor_source.startswith("static_stop"):
        primary_candidate_id = (
            int(fusion_selected_id)
            if fusion_selected_id is not None
            else int(best.get("candidate_id", -1))
        )
        static_anchor_spans.append(
            (
                int(segment_start),
                int(segment_end),
                np.asarray(static_position, dtype=np.float64),
                primary_candidate_id,
                "primary_static_anchor",
            )
        )
    additional_static_anchors = getattr(args, "additional_static_anchor", [])
    if additional_static_anchors and not anchor_source.startswith("static_stop"):
        raise RuntimeError("additional static anchors require a primary static anchor")
    for static_path, fusion_path in additional_static_anchors:
        static_anchor_spans.append(
            _load_fusion_static_override(Path(static_path), Path(fusion_path))
        )
    static_anchor_spans.sort(key=lambda item: item[0])

    map_positions: dict[int, np.ndarray] = {}
    for epoch, rows in by_epoch.items():
        map_positions[epoch] = _position(max(rows, key=lambda row: float(row["log_weight"])))
    effective_tdcp_fallback = _resolve_tdcp_fallback(
        str(args.tdcp_fallback), anchor_source
    )
    doppler_static_bias = None
    doppler_static_bias_samples = 0
    if effective_tdcp_fallback == "doppler-calibrated-static":
        if not anchor_source.startswith("static_stop"):
            raise RuntimeError("static Doppler calibration requires --static-anchor-json")
        bias_samples: list[np.ndarray] = []
        for epoch in range(segment_start, segment_end):
            velocity, _rms = _doppler_velocity(data, epoch, static_position)
            if velocity is not None:
                bias_samples.append(np.asarray(velocity, dtype=np.float64))
        doppler_static_bias = _robust_static_velocity_bias(bias_samples)
        doppler_static_bias_samples = len(bias_samples)
        if doppler_static_bias is None:
            raise RuntimeError("not enough static Doppler samples for calibration")
    trusted_fix_bias_samples: list[np.ndarray] = []
    tdcp_displacements: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
    temporal_displacement_rows: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "tow": float(data["times"][0]),
            "interval_dt_s": 0.0,
            "dx_m": 0.0,
            "dy_m": 0.0,
            "dz_m": 0.0,
            "norm_m": 0.0,
            "source": "origin",
            "postfit_rms_m": float("nan"),
            "n_used": 0,
            "n_rejected": 0,
            "doppler_dx_m": float("nan"),
            "doppler_dy_m": float("nan"),
            "doppler_dz_m": float("nan"),
            "doppler_norm_m": float("nan"),
            "doppler_rms_mps": float("nan"),
            "tdcp_doppler_vector_difference_m": float("nan"),
            "gate_reason": "",
        }
    ]
    previous = [
        measurement
        for measurement in _epoch_measurements(data, 0)
        if int(measurement.system_id) in (0, 2, 4)
    ]
    n_tdcp = 0
    n_tdcp_accepted = 0
    n_tdcp_doppler_rejected = 0
    n_doppler_fallback = 0
    n_time_gap_intervals = 0
    max_interval_dt_s = 0.0
    for epoch in range(1, n_epochs):
        interval_dt_s = _interval_dt_s(
            np.asarray(data["times"]), epoch, float(args.epoch_dt_s)
        )
        max_interval_dt_s = max(max_interval_dt_s, interval_dt_s)
        n_time_gap_intervals += int(interval_dt_s > 1.5 * float(args.epoch_dt_s))
        current = [
            measurement
            for measurement in _epoch_measurements(data, epoch)
            if int(measurement.system_id) in (0, 2, 4)
        ]
        approximate = map_positions.get(
            epoch, _position(diagnostics[epoch])
        )
        if (
            effective_tdcp_fallback == "doppler-calibrated-trusted-fix"
            and diagnostics[epoch]["fix"] == "1"
            and diagnostics[epoch - 1]["fix"] == "1"
        ):
            dt = float(diagnostics[epoch]["tow"]) - float(
                diagnostics[epoch - 1]["tow"]
            )
            velocity, _rms = _doppler_velocity(data, epoch, approximate)
            if velocity is not None and dt > 0.0:
                trusted_velocity = (
                    _position(diagnostics[epoch])
                    - _position(diagnostics[epoch - 1])
                ) / dt
                trusted_fix_bias_samples.append(
                    np.asarray(velocity, dtype=np.float64) - trusted_velocity
                )
                doppler_static_bias = _robust_trusted_fix_velocity_bias(
                    trusted_fix_bias_samples
                )
                doppler_static_bias_samples = len(trusted_fix_bias_samples)
        estimate = estimate_displacement_from_tdcp(
            approximate,
            previous,
            current,
            interval_dt_s,
            min_sats=int(args.tdcp_min_sats),
            max_postfit_rms_m=float(args.tdcp_max_postfit_rms_m),
            slip_residual_threshold_m=float(args.tdcp_slip_threshold_m),
        )
        displacement = (
            None
            if estimate is None
            else np.asarray(estimate.displacement_ecef_m, dtype=np.float64)
        )
        displacement_source = "tdcp" if displacement is not None else "hold"
        postfit_rms_m = (
            float("nan") if estimate is None else float(estimate.postfit_rms_m)
        )
        n_used = 0 if estimate is None else int(estimate.n_used)
        n_rejected = 0 if estimate is None else int(estimate.n_rejected)
        doppler_displacement = None
        doppler_rms_mps = float("nan")
        if effective_tdcp_fallback in (
            "doppler",
            "doppler-calibrated-static",
            "doppler-calibrated-trusted-fix",
        ):
            velocity, doppler_rms_mps = _doppler_velocity(data, epoch, approximate)
            calibrated_ready = (
                effective_tdcp_fallback != "doppler-calibrated-trusted-fix"
                or doppler_static_bias is not None
            )
            if velocity is not None and calibrated_ready:
                if doppler_static_bias is not None:
                    velocity = np.asarray(velocity) - doppler_static_bias
                doppler_displacement = np.asarray(velocity, dtype=np.float64) * float(
                    interval_dt_s
                )
        tdcp_doppler_vector_difference_m = (
            float("nan")
            if displacement is None or doppler_displacement is None
            else float(np.linalg.norm(displacement - doppler_displacement))
        )
        gate_reason = None
        if displacement is not None:
            gate_reason = _tdcp_doppler_gate_reason(
                displacement,
                doppler_displacement,
                max_vector_difference_m=float(
                    getattr(args, "tdcp_doppler_max_vector_difference_m", float("inf"))
                ),
            )
            if gate_reason is not None:
                displacement = None
                displacement_source = "hold"
                n_tdcp_doppler_rejected += 1
        if displacement is None and effective_tdcp_fallback in (
            "doppler",
            "doppler-calibrated-static",
            "doppler-calibrated-trusted-fix",
        ):
            if doppler_displacement is not None:
                displacement = doppler_displacement
                displacement_source = "doppler"
                n_doppler_fallback += 1
        committed_displacement = (
            np.zeros(3, dtype=np.float64) if displacement is None else displacement
        )
        tdcp_displacements.append(committed_displacement)
        temporal_displacement_rows.append(
            {
                "epoch": epoch,
                "tow": float(data["times"][epoch]),
                "interval_dt_s": interval_dt_s,
                "dx_m": float(committed_displacement[0]),
                "dy_m": float(committed_displacement[1]),
                "dz_m": float(committed_displacement[2]),
                "norm_m": float(np.linalg.norm(committed_displacement)),
                "source": displacement_source,
                "postfit_rms_m": postfit_rms_m,
                "n_used": n_used,
                "n_rejected": n_rejected,
                "doppler_dx_m": (
                    float("nan")
                    if doppler_displacement is None
                    else float(doppler_displacement[0])
                ),
                "doppler_dy_m": (
                    float("nan")
                    if doppler_displacement is None
                    else float(doppler_displacement[1])
                ),
                "doppler_dz_m": (
                    float("nan")
                    if doppler_displacement is None
                    else float(doppler_displacement[2])
                ),
                "doppler_norm_m": (
                    float("nan")
                    if doppler_displacement is None
                    else float(np.linalg.norm(doppler_displacement))
                ),
                "doppler_rms_mps": doppler_rms_mps,
                "tdcp_doppler_vector_difference_m": (
                    tdcp_doppler_vector_difference_m
                ),
                "gate_reason": "" if gate_reason is None else gate_reason,
            }
        )
        n_tdcp += int(estimate is not None)
        n_tdcp_accepted += int(estimate is not None and gate_reason is None)
        previous = current

    static_gap_closure_reports: list[dict[str, Any]] = []
    if bool(getattr(args, "static_gap_endpoint_closure", False)):
        if len(static_anchor_spans) < 2:
            raise RuntimeError("static gap endpoint closure needs at least two anchors")
        static_gap_closure_reports = _close_static_anchor_gaps(
            tdcp_displacements,
            np.asarray(data["times"], dtype=np.float64),
            static_anchor_spans,
            nominal_dt_s=float(args.epoch_dt_s),
        )
        if not static_gap_closure_reports:
            raise RuntimeError("no recorded-time gap exists between static anchors")
        for report in static_gap_closure_reports:
            epoch = int(report["bridge_epoch"])
            committed = np.asarray(tdcp_displacements[epoch], dtype=np.float64)
            row = temporal_displacement_rows[epoch]
            row["dx_m"] = float(committed[0])
            row["dy_m"] = float(committed[1])
            row["dz_m"] = float(committed[2])
            row["norm_m"] = float(np.linalg.norm(committed))
            row["source"] = "static_endpoint_gap_closure"
            row["gate_reason"] = "static_anchor_endpoint_residual"


    candidate_epochs = [
        AnchorCandidateEpoch(
            epoch=epoch,
            positions_ecef=np.asarray([_position(row) for row in by_epoch[epoch]]),
            log_weights=np.asarray(
                [
                    float(row["log_weight"])
                    + (
                        float(args.current_seed_consensus_bonus)
                        if _current_epoch_seed_support(row, epoch)
                        >= int(args.current_seed_consensus_min_support)
                        else 0.0
                    )
                    + (
                        float(getattr(args, "external_seed_bonus", 0.0))
                        if _has_recent_external_position_seed(
                            row,
                            epoch,
                            int(getattr(args, "external_seed_max_age_epochs", 5)),
                        )
                        else 0.0
                    )
                    for row in by_epoch[epoch]
                ],
                dtype=np.float64,
            ),
        )
        for epoch in anchor_epochs
    ]
    candidate_assignments = {
        epoch: [_assignment_map(row) for row in by_epoch[epoch]]
        for epoch in anchor_epochs
    }
    candidate_reacquisition_flags = {
        epoch: [_is_reacquisition_proposal(row, epoch) for row in by_epoch[epoch]]
        for epoch in anchor_epochs
    }
    interval_displacements = {
        (left, right): np.sum(
            np.asarray(tdcp_displacements[left + 1 : right + 1]), axis=0
        )
        for left, right in zip(anchor_epochs[:-1], anchor_epochs[1:])
    }
    has_external_route_seed = any(
        _has_recent_external_position_seed(row, epoch, 0)
        for epoch, rows in by_epoch.items()
        for row in rows
    )
    effective_path_mode = _resolve_path_mode(
        str(args.path_mode),
        anchor_source,
        has_external_route_seed=has_external_route_seed,
    )
    if effective_path_mode in (
        "assignment-greedy",
        "assignment-reacquisition-greedy",
    ):
        path = constrained_assignment_greedy_path(
            candidate_epochs,
            interval_displacements,
            candidate_assignments,
            constrained_indices=trusted_constraints,
            transition_sigma_m=float(args.transition_sigma_m),
            emission_weight=float(args.emission_weight),
            transition_loss=str(args.transition_loss),
            assignment_match_bonus=float(args.assignment_match_bonus),
            assignment_conflict_penalty=float(args.assignment_conflict_penalty),
            candidate_reacquisition_flags=(
                candidate_reacquisition_flags
                if effective_path_mode == "assignment-reacquisition-greedy"
                else None
            ),
            reacquisition_min_exact_pairs=int(args.reacquisition_min_exact_pairs),
            reacquisition_min_stable_anchors=int(
                args.reacquisition_min_stable_anchors
            ),
            reacquisition_window_anchors=int(args.reacquisition_window_anchors),
            reacquisition_ignore_assignment=bool(
                args.reacquisition_ignore_assignment
            ),
            reacquisition_dead_reckon=bool(args.reacquisition_dead_reckon),
        )
    elif effective_path_mode == "assignment-viterbi":
        path = constrained_assignment_viterbi_path(
            candidate_epochs,
            interval_displacements,
            candidate_assignments,
            constrained_indices=trusted_constraints,
            transition_sigma_m=float(args.transition_sigma_m),
            emission_weight=float(args.emission_weight),
            transition_loss=str(args.transition_loss),
            assignment_match_bonus=float(args.assignment_match_bonus),
            assignment_conflict_penalty=float(args.assignment_conflict_penalty),
        )
    elif effective_path_mode == "assignment-max-marginal":
        assignment_audit = constrained_assignment_viterbi_audit(
            candidate_epochs,
            interval_displacements,
            candidate_assignments,
            constrained_indices=trusted_constraints,
            transition_sigma_m=float(args.transition_sigma_m),
            emission_weight=float(args.emission_weight),
            transition_loss=str(args.transition_loss),
            assignment_match_bonus=float(args.assignment_match_bonus),
            assignment_conflict_penalty=float(args.assignment_conflict_penalty),
        )
        path = {
            epoch: int(np.argmax(scores))
            for epoch, scores in assignment_audit.max_marginal_relative.items()
        }
    elif effective_path_mode == "greedy":
        path = constrained_greedy_path(
            candidate_epochs,
            interval_displacements,
            constrained_indices=trusted_constraints,
            transition_sigma_m=float(args.transition_sigma_m),
            emission_weight=float(args.emission_weight),
            transition_loss=str(args.transition_loss),
        )
    elif (
        args.use_all_trusted_anchors
        or anchor_source.startswith("static_stop")
        or args.path_mode == "auto"
    ):
        path = constrained_viterbi_path(
            candidate_epochs,
            interval_displacements,
            constrained_indices=trusted_constraints,
            transition_sigma_m=float(args.transition_sigma_m),
            emission_weight=float(args.emission_weight),
            transition_loss=str(args.transition_loss),
        )
    else:
        path = anchored_viterbi_path(
            candidate_epochs,
            interval_displacements,
            anchor_epoch=trusted_anchor_epoch,
            anchor_index=trusted_anchor_index,
            transition_sigma_m=float(args.transition_sigma_m),
            emission_weight=float(args.emission_weight),
            transition_loss=str(args.transition_loss),
        )
    path_candidate_audit: list[dict[str, Any]] = []
    if getattr(args, "out_path_candidate_audit", None) is not None:
        if not trusted_constraints:
            raise RuntimeError("path candidate audit requires constrained anchors")
        if effective_path_mode.startswith("assignment-"):
            score_audit = constrained_assignment_viterbi_audit(
                candidate_epochs,
                interval_displacements,
                candidate_assignments,
                constrained_indices=trusted_constraints,
                transition_sigma_m=float(args.transition_sigma_m),
                emission_weight=float(args.emission_weight),
                transition_loss=str(args.transition_loss),
                assignment_match_bonus=float(args.assignment_match_bonus),
                assignment_conflict_penalty=float(
                    args.assignment_conflict_penalty
                ),
            )
        else:
            score_audit = constrained_viterbi_audit(
                candidate_epochs,
                interval_displacements,
                constrained_indices=trusted_constraints,
                transition_sigma_m=float(args.transition_sigma_m),
                emission_weight=float(args.emission_weight),
                transition_loss=str(args.transition_loss),
            )
        epoch_lookup = {item.epoch: item for item in candidate_epochs}
        for epoch_index, epoch in enumerate(anchor_epochs):
            max_marginal = score_audit.max_marginal_relative[epoch]
            forward = score_audit.forward_relative[epoch]
            backward = score_audit.backward_relative[epoch]
            previous_epoch = anchor_epochs[epoch_index - 1] if epoch_index > 0 else None
            next_epoch = (
                anchor_epochs[epoch_index + 1]
                if epoch_index + 1 < len(anchor_epochs)
                else None
            )
            for index, row in enumerate(by_epoch[epoch]):
                position = _position(row)
                previous_residual = float("nan")
                if previous_epoch is not None:
                    previous_position = epoch_lookup[previous_epoch].positions_ecef[
                        path[previous_epoch]
                    ]
                    previous_residual = float(
                        np.linalg.norm(
                            position
                            - previous_position
                            - interval_displacements[(previous_epoch, epoch)]
                        )
                    )
                next_residual = float("nan")
                if next_epoch is not None:
                    next_position = epoch_lookup[next_epoch].positions_ecef[path[next_epoch]]
                    next_residual = float(
                        np.linalg.norm(
                            next_position
                            - position
                            - interval_displacements[(epoch, next_epoch)]
                        )
                    )
                truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
                error = float(np.linalg.norm(position - truth))
                path_candidate_audit.append(
                    {
                        "epoch": epoch,
                        "basin_id": row["basin_id"],
                        "ecef_x": float(position[0]),
                        "ecef_y": float(position[1]),
                        "ecef_z": float(position[2]),
                        "interval_from_previous_dx_m": (
                            float(interval_displacements[(previous_epoch, epoch)][0])
                            if previous_epoch is not None
                            else float("nan")
                        ),
                        "interval_from_previous_dy_m": (
                            float(interval_displacements[(previous_epoch, epoch)][1])
                            if previous_epoch is not None
                            else float("nan")
                        ),
                        "interval_from_previous_dz_m": (
                            float(interval_displacements[(previous_epoch, epoch)][2])
                            if previous_epoch is not None
                            else float("nan")
                        ),
                        "selected": int(path[epoch] == index),
                        "error_m": error,
                        "sub50cm": int(error < 0.5),
                        "log_weight": float(row["log_weight"]),
                        "current_seed_support": _current_epoch_seed_support(row, epoch),
                        "forward_relative": float(forward[index]),
                        "backward_relative": float(backward[index]),
                        "max_marginal_relative": float(max_marginal[index]),
                        "max_marginal_rank": int(
                            1 + np.count_nonzero(max_marginal > max_marginal[index])
                        ),
                        "previous_selected_transition_residual_m": previous_residual,
                        "next_selected_transition_residual_m": next_residual,
                    }
                )
    candidate_by_epoch = {item.epoch: item for item in candidate_epochs}
    dead_reckon_anchor_positions = None
    if effective_path_mode in ("absolute-tdcp", "absolute-tdcp-dead-reckon"):
        trusted_position = (
            np.asarray(static_position, dtype=np.float64)
            if (
                effective_path_mode == "absolute-tdcp-dead-reckon"
                and anchor_source.startswith("static_stop")
            )
            else candidate_by_epoch[trusted_anchor_epoch].positions_ecef[
                trusted_anchor_index
            ]
        )
        predicted: dict[int, np.ndarray] = {
            trusted_anchor_epoch: np.asarray(trusted_position).copy()
        }
        if (
            effective_path_mode == "absolute-tdcp-dead-reckon"
            and static_anchor_spans
        ):
            for start, end, position, _candidate_id, _reason in static_anchor_spans:
                for epoch in range(start, end):
                    predicted[epoch] = np.asarray(position, dtype=np.float64).copy()
            first_start = int(static_anchor_spans[0][0])
            for epoch in range(first_start - 1, -1, -1):
                predicted[epoch] = predicted[epoch + 1] - tdcp_displacements[epoch + 1]
            for left, right in zip(static_anchor_spans[:-1], static_anchor_spans[1:]):
                left_end = int(left[1])
                right_start = int(right[0])
                for epoch in range(left_end, right_start + 1):
                    predicted[epoch] = predicted[epoch - 1] + tdcp_displacements[epoch]
                for epoch in range(right_start, int(right[1])):
                    predicted[epoch] = np.asarray(right[2], dtype=np.float64).copy()
            last_end = int(static_anchor_spans[-1][1])
            for epoch in range(last_end, n_epochs):
                predicted[epoch] = predicted[epoch - 1] + tdcp_displacements[epoch]
        else:
            for epoch in range(trusted_anchor_epoch + 1, n_epochs):
                predicted[epoch] = predicted[epoch - 1] + tdcp_displacements[epoch]
            for epoch in range(trusted_anchor_epoch - 1, -1, -1):
                predicted[epoch] = predicted[epoch + 1] - tdcp_displacements[epoch + 1]
        if effective_path_mode == "absolute-tdcp-dead-reckon":
            dead_reckon_anchor_positions = {
                epoch: predicted[epoch] for epoch in anchor_epochs
            }
        else:
            path = {
                epoch: int(
                    np.argmin(
                        np.linalg.norm(
                            candidate_by_epoch[epoch].positions_ecef
                            - predicted[epoch][None, :],
                            axis=1,
                        )
                    )
                )
                for epoch in anchor_epochs
            }
    anchor_positions = (
        dead_reckon_anchor_positions
        if dead_reckon_anchor_positions is not None
        else {
            epoch: candidate_by_epoch[epoch].positions_ecef[index]
            for epoch, index in path.items()
        }
    )
    apply_static_anchor_offset = _resolve_static_anchor_offset(
        bool(args.apply_static_anchor_offset),
        bool(getattr(args, "apply_static_anchor_offset_auto", False)),
        anchor_source,
        has_external_route_seed=has_external_route_seed,
    )
    static_path_offset = None
    if apply_static_anchor_offset:
        supported = [
            anchor_positions[epoch]
            for epoch in trusted_constraints
            if epoch in anchor_positions and segment_start <= epoch < segment_end
        ]
        if not supported:
            raise RuntimeError("static anchor offset has no constrained path support")
        static_path_offset = static_position - np.median(
            np.asarray(supported, dtype=np.float64), axis=0
        )

    output: list[dict[str, Any]] = []
    for epoch in range(n_epochs):
        diagnostic = diagnostics[epoch]
        original = _position(diagnostic)
        smoothed = interpolate_path_position(epoch, anchor_positions)
        if smoothed is not None and static_path_offset is not None:
            smoothed = smoothed + static_path_offset
        declared_fix = diagnostic["fix"] == "1"
        if declared_fix:
            selected = original
            source = "trusted_fix"
        elif (
            static_position_override is not None
            and static_position_override[0] <= epoch < static_position_override[1]
        ):
            selected = static_position_override[2]
            source = "static_position_override"
        elif (
            static_override is not None
            and static_override[0] <= epoch < static_override[1]
        ):
            selected = static_override[2]
            source = "static_fusion_override"
        elif smoothed is not None:
            selected = smoothed
            source = "tdcp_anchor_smoother"
        else:
            selected = original
            source = "baseline_fallback"
        truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
        error = float(np.linalg.norm(selected - truth))
        output.append(
            {
                "epoch": epoch,
                "tow": float(data["times"][epoch]),
                "ecef_x": float(selected[0]),
                "ecef_y": float(selected[1]),
                "ecef_z": float(selected[2]),
                "source": source,
                "fix": int(declared_fix),
                "error_m": error,
                "sub50cm": int(error < 0.5),
                "false_fix": int(declared_fix and error >= 0.5),
            }
        )
    fixed = [row for row in output if row["fix"]]
    summary = {
        "n_epochs_full_denominator": n_epochs,
        "transition_sigma_m": float(args.transition_sigma_m),
        "transition_loss": str(args.transition_loss),
        "emission_weight": float(args.emission_weight),
        "current_seed_consensus_bonus": float(args.current_seed_consensus_bonus),
        "current_seed_consensus_min_support": int(args.current_seed_consensus_min_support),
        "external_seed_bonus": float(getattr(args, "external_seed_bonus", 0.0)),
        "external_seed_max_age_epochs": int(
            getattr(args, "external_seed_max_age_epochs", 5)
        ),
        "anchor_stride_epochs": anchor_stride,
        "path_mode": str(args.path_mode),
        "effective_path_mode": effective_path_mode,
        "tdcp_fallback": str(args.tdcp_fallback),
        "effective_tdcp_fallback": effective_tdcp_fallback,
        "candidate_branches": len(trace_paths),
        "first_fix_epoch": first_fix_epoch,
        "anchor_source": anchor_source,
        "trusted_anchor_epoch": trusted_anchor_epoch,
        "trusted_anchor_index": trusted_anchor_index,
        "trusted_anchor_constraints": len(trusted_constraints),
        "use_all_trusted_anchors": bool(args.use_all_trusted_anchors),
        "apply_static_anchor_offset_auto": bool(
            getattr(args, "apply_static_anchor_offset_auto", False)
        ),
        "effective_apply_static_anchor_offset": apply_static_anchor_offset,
        "static_path_offset_ecef_m": (
            None if static_path_offset is None else static_path_offset.tolist()
        ),
        "tdcp_intervals": n_tdcp,
        "tdcp_accepted_intervals": n_tdcp_accepted,
        "tdcp_doppler_rejected_intervals": n_tdcp_doppler_rejected,
        "tdcp_doppler_max_vector_difference_m": float(
            getattr(args, "tdcp_doppler_max_vector_difference_m", float("inf"))
        ),
        "time_gap_intervals": n_time_gap_intervals,
        "max_interval_dt_s": max_interval_dt_s,
        "static_anchor_spans": [
            {
                "start": int(start),
                "end": int(end),
                "candidate_id": int(candidate_id),
                "reason": reason,
            }
            for start, end, _position_ecef, candidate_id, reason in static_anchor_spans
        ],
        "static_gap_endpoint_closure": bool(
            getattr(args, "static_gap_endpoint_closure", False)
        ),
        "static_gap_closure_reports": static_gap_closure_reports,
        "doppler_fallback_intervals": n_doppler_fallback,
        "doppler_static_bias_samples": doppler_static_bias_samples,
        "doppler_static_bias_ecef_mps": (
            None if doppler_static_bias is None else doppler_static_bias.tolist()
        ),
        "smoother_epochs": sum(row["source"] == "tdcp_anchor_smoother" for row in output),
        "static_fusion_override_epochs": sum(
            row["source"] == "static_fusion_override" for row in output
        ),
        "static_fusion_override_candidate_id": (
            None if static_override is None else static_override[3]
        ),
        "static_fusion_override_reason": (
            None if static_override is None else static_override[4]
        ),
        "static_fusion_path_constraints": static_override_constraints,
        "static_position_override_epochs": sum(
            row["source"] == "static_position_override" for row in output
        ),
        "static_position_override_candidate_id": (
            None if static_position_override is None else static_position_override[3]
        ),
        "static_position_override_reason": (
            None if static_position_override is None else static_position_override[4]
        ),
        "sub50cm_full_epochs": sum(row["sub50cm"] for row in output),
        "sub50cm_full_pct": 100.0 * sum(row["sub50cm"] for row in output) / n_epochs,
        "declared_fix_epochs": len(fixed),
        "false_fix_epochs": sum(row["false_fix"] for row in fixed),
        "false_fix_pct": 100.0
        * sum(row["false_fix"] for row in fixed)
        / max(len(fixed), 1),
    }
    out_anchor_audit = getattr(args, "out_anchor_audit", None)
    if out_anchor_audit is not None:
        anchor_audit = _build_anchor_path_audit(
            by_epoch,
            path,
            np.asarray(data["ground_truth"]),
            interval_displacements,
            static_path_offset=static_path_offset,
        )
        out_anchor_audit.parent.mkdir(parents=True, exist_ok=True)
        with out_anchor_audit.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(anchor_audit[0]))
            writer.writeheader()
            writer.writerows(anchor_audit)
    out_path_candidate_audit = getattr(args, "out_path_candidate_audit", None)
    if out_path_candidate_audit is not None:
        out_path_candidate_audit.parent.mkdir(parents=True, exist_ok=True)
        with out_path_candidate_audit.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(path_candidate_audit[0]))
            writer.writeheader()
            writer.writerows(path_candidate_audit)
    out_temporal_displacements = getattr(args, "out_temporal_displacements", None)
    if out_temporal_displacements is not None:
        out_temporal_displacements.parent.mkdir(parents=True, exist_ok=True)
        with out_temporal_displacements.open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(temporal_displacement_rows[0])
            )
            writer.writeheader()
            writer.writerows(temporal_displacement_rows)
    return summary, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument(
        "--additional-basin-trace", type=Path, action="append", default=[]
    )
    parser.add_argument("epoch_diagnostics", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--transition-sigma-m", type=float, default=0.5)
    parser.add_argument(
        "--transition-loss", choices=("gaussian", "huber", "cauchy"), default="gaussian"
    )
    parser.add_argument("--emission-weight", type=float, default=0.03)
    parser.add_argument("--current-seed-consensus-bonus", type=float, default=0.0)
    parser.add_argument("--current-seed-consensus-min-support", type=int, default=2)
    parser.add_argument("--external-seed-bonus", type=float, default=0.0)
    parser.add_argument("--external-seed-max-age-epochs", type=int, default=5)
    parser.add_argument("--assignment-match-bonus", type=float, default=2.0)
    parser.add_argument("--assignment-conflict-penalty", type=float, default=4.0)
    parser.add_argument("--reacquisition-min-exact-pairs", type=int, default=4)
    parser.add_argument("--reacquisition-min-stable-anchors", type=int, default=10)
    parser.add_argument("--reacquisition-window-anchors", type=int, default=0)
    parser.add_argument("--reacquisition-ignore-assignment", action="store_true")
    parser.add_argument("--reacquisition-dead-reckon", action="store_true")
    parser.add_argument("--anchor-stride-epochs", type=int, default=5)
    parser.add_argument(
        "--path-mode",
        choices=(
            "viterbi",
            "greedy",
            "assignment-greedy",
            "assignment-reacquisition-greedy",
            "assignment-viterbi",
            "assignment-max-marginal",
            "absolute-tdcp",
            "absolute-tdcp-dead-reckon",
            "auto",
        ),
        default="viterbi",
    )
    parser.add_argument("--anchor-search-epochs", type=int, default=4)
    parser.add_argument("--use-all-trusted-anchors", action="store_true")
    parser.add_argument("--static-anchor-json", type=Path)
    parser.add_argument("--static-anchor-fusion-json", type=Path)
    parser.add_argument(
        "--additional-static-anchor",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("STATIC_JSON", "FUSION_JSON"),
    )
    parser.add_argument("--static-gap-endpoint-closure", action="store_true")
    parser.add_argument("--static-override-json", type=Path)
    parser.add_argument("--static-override-fusion-json", type=Path)
    parser.add_argument("--static-position-override-json", type=Path)
    parser.add_argument("--static-override-constrain-path", action="store_true")
    parser.add_argument(
        "--static-override-max-candidate-distance-m", type=float, default=0.5
    )
    parser.add_argument("--static-anchor-max-norm-rms", type=float, default=0.015)
    parser.add_argument("--static-anchor-max-runner-up-ratio", type=float, default=0.95)
    parser.add_argument("--static-anchor-min-bootstrap-wins", type=int, default=4)
    parser.add_argument("--apply-static-anchor-offset", action="store_true")
    parser.add_argument("--apply-static-anchor-offset-auto", action="store_true")
    parser.add_argument(
        "--static-anchor-max-candidate-distance-m", type=float, default=0.5
    )
    parser.add_argument("--epoch-dt-s", type=float, default=0.2)
    parser.add_argument("--tdcp-min-sats", type=int, default=5)
    parser.add_argument("--tdcp-max-postfit-rms-m", type=float, default=0.5)
    parser.add_argument("--tdcp-slip-threshold-m", type=float, default=0.25)
    parser.add_argument(
        "--tdcp-doppler-max-vector-difference-m",
        type=float,
        default=float("inf"),
        help=(
            "reject TDCP and use the configured Doppler fallback when the two "
            "displacement vectors differ by more than this many metres"
        ),
    )
    parser.add_argument(
        "--tdcp-fallback",
        choices=(
            "zero",
            "doppler",
            "doppler-calibrated-static",
            "doppler-calibrated-trusted-fix",
            "doppler-calibrated-auto",
        ),
        default="zero",
    )
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    parser.add_argument("--out-anchor-audit", type=Path)
    parser.add_argument("--out-path-candidate-audit", type=Path)
    parser.add_argument("--out-temporal-displacements", type=Path)
    args = parser.parse_args()
    summary, rows = run(args)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_trajectory.parent.mkdir(parents=True, exist_ok=True)
    with args.out_trajectory.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
