#!/usr/bin/env python3
"""Track native MultiSD integer basins without opening PPC reference truth."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.ambiguity_basin_pf import AmbiguityBasinParticleFilter  # noqa: E402
from gnss_gpu.basin_imu_bridge import (  # noqa: E402
    CausalBasinImuPredictor,
    load_ppc_imu_csv,
)
from gnss_gpu.basin_ffbsi import FixedLagBasinFFBSi  # noqa: E402
from gnss_gpu.basin_fgo_bridge import (  # noqa: E402
    NativeFGOHypothesisCandidate,
    parse_native_fgo_hypotheses,
    transition_native_fgo_candidates,
)
from gnss_gpu.pf_imu_preint_adapter import (  # noqa: E402
    ecef_to_enu_rotation,
    ecef_to_lla_rad,
)


@dataclass(frozen=True)
class NativeImuFgoState:
    position_ecef_m: np.ndarray
    velocity_ecef_mps: np.ndarray


@dataclass(frozen=True)
class CausalImuOffsetAnchor:
    tow_s: float
    fixed_position_ecef_m: np.ndarray
    imu_position_ecef_m: np.ndarray


def _native_imu_fgo_state(native_rows: list[dict]) -> NativeImuFgoState | None:
    """Parse one causal native IMU-FGO proposal without treating it as evidence."""

    if not native_rows:
        return None
    payload = native_rows[0].get("imu_fgo")
    if not isinstance(payload, dict):
        return None
    if payload.get("available") is not True or payload.get("converged") is not True:
        return None
    # A recovered smoother can provide a numerically valid trajectory, but it
    # was re-anchored after an indeterminate update.  Keep that epoch as
    # telemetry only instead of allowing it to influence a PF transition or
    # accelerated FIX decision.
    recovery_epochs = payload.get("recovery_epochs", 0)
    if (
        isinstance(recovery_epochs, bool)
        or not isinstance(recovery_epochs, (int, float))
        or not np.isfinite(recovery_epochs)
        or recovery_epochs != 0
    ):
        return None
    fault_reason = payload.get("fault_reason", "ok")
    if fault_reason != "ok":
        return None
    try:
        position = np.asarray(payload["position_ecef"], dtype=np.float64).reshape(3)
        velocity_nav = np.asarray(
            payload["velocity_nav_mps"], dtype=np.float64
        ).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not np.all(np.isfinite(position))
        or not np.all(np.isfinite(velocity_nav))
        or np.linalg.norm(position) < 1.0e6
        or np.linalg.norm(velocity_nav) > 100.0
    ):
        return None
    lat, lon = ecef_to_lla_rad(position)
    velocity_ecef = ecef_to_enu_rotation(lat, lon).T @ velocity_nav
    return NativeImuFgoState(position.copy(), velocity_ecef)


def _native_imu_process_covariance(dt: float) -> np.ndarray:
    accel_sigma = 0.5
    position_sigma = 0.10 + 0.5 * accel_sigma * dt * dt
    velocity_sigma = 0.20 + accel_sigma * dt
    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[:3, :3] = np.eye(3) * position_sigma**2
    covariance[3:, 3:] = np.eye(3) * velocity_sigma**2
    return covariance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_basin_rows(path: Path) -> dict[int, list[dict]]:
    by_epoch: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL line {line_number}") from exc
            if row.get("schema") != "gnsspp_multisd_basin_v1":
                raise ValueError(f"invalid basin schema on line {line_number}")
            by_epoch[int(row["epoch_index"])].append(row)
    return dict(by_epoch)


def write_validated_pf_feedback(
    path: Path,
    tracker_rows: list[dict[str, object]],
    basin_rows_by_epoch: dict[int, list[dict]],
    *,
    group_index: int,
) -> int:
    """Write only PF FIX modes whose selected native candidate passed GNSS holdout."""

    fields = (
        "gps_week",
        "tow",
        "source_epoch_index",
        "satellite",
        "reference_satellite",
        "signal",
        "segment_index",
        "reference_segment_index",
        "wavelength_m",
        "fixed_cycles",
        "unique_holdout_pass",
        "disjoint_holdout_selected",
        "causal_imu_motion_selected",
        "imu_aperture_selected",
        "imu_accelerated_fix",
        "selected_native_holdout_pass",
        "schema",
    )
    output_rows: list[dict[str, object]] = []
    for tracker in tracker_rows:
        if int(tracker.get("shadow_fixed", 0)) != 1:
            continue
        epoch_index = int(tracker["epoch_index"])
        rank = int(tracker.get("selected_rank", -1))
        selected = [
            row
            for row in basin_rows_by_epoch.get(epoch_index, ())
            if row.get("group_index") == int(group_index)
            and row.get("rank") == rank
            and row.get("evaluated") is True
            and row.get("pass") is True
        ]
        if len(selected) != 1:
            continue
        native = selected[0]
        fixed_integers = native.get("fixed_integers")
        if not isinstance(fixed_integers, list) or not fixed_integers:
            continue
        for fixed in fixed_integers:
            output_rows.append(
                {
                    "gps_week": int(native["gps_week"]),
                    "tow": float(native["tow"]),
                    "source_epoch_index": epoch_index,
                    "satellite": str(fixed["satellite"]),
                    "reference_satellite": str(fixed["reference_satellite"]),
                    "signal": int(fixed["signal"]),
                    "segment_index": int(fixed["segment_index"]),
                    "reference_segment_index": int(fixed["reference_segment_index"]),
                    "wavelength_m": float(fixed["wavelength_m"]),
                    "fixed_cycles": int(fixed["fixed_cycles"]),
                    "unique_holdout_pass": int(tracker.get("unique_holdout_pass", 0)),
                    "disjoint_holdout_selected": int(
                        tracker.get("disjoint_holdout_selected", 0)
                    ),
                    "causal_imu_motion_selected": int(
                        tracker.get("causal_imu_motion_selected", 0)
                    ),
                    "imu_aperture_selected": int(
                        tracker.get("imu_aperture_selected", 0)
                    ),
                    "imu_accelerated_fix": int(tracker.get("imu_accelerated_fix", 0)),
                    "selected_native_holdout_pass": 1,
                    "schema": "gnss_gpu_pf_fgo_feedback_v1",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def _assignments_compatible(
    left: dict[object, int] | None,
    right: dict[object, int],
    minimum_common: int = 6,
) -> bool:
    if left is None:
        return False
    common = set(left).intersection(right)
    return len(common) >= minimum_common and all(
        left[key] == right[key] for key in common
    )


def _disjoint_carrier_holdout_winner(
    native_rows: list[dict],
    candidates: list[NativeFGOHypothesisCandidate],
    *,
    group_index: int,
    minimum_satellites_per_partition: int,
    minimum_cost_margin: float,
    residual_clip: float,
    minimum_carrier_pass_fraction: float,
) -> tuple[NativeFGOHypothesisCandidate | None, float]:
    """Select only when two disjoint carrier partitions prefer the same basin."""

    if len(candidates) < 2:
        return None, float("nan")
    rows_by_rank = {
        int(row.get("rank", -1)): row
        for row in native_rows
        if row.get("group_index") == int(group_index)
        and row.get("evaluated") is True
        and row.get("pass") is True
    }
    candidate_rows = []
    key_sets: list[set[tuple[str, str, int]]] = []
    for candidate in candidates:
        row = rows_by_rank.get(int(candidate.rank))
        if row is None:
            return None, float("nan")
        residuals = row.get("validation_residuals")
        if not isinstance(residuals, list):
            return None, float("nan")
        keys: set[tuple[str, str, int]] = set()
        for residual in residuals:
            if not isinstance(residual, dict) or residual.get("kind") != "carrier":
                continue
            try:
                key = (
                    str(residual["satellite"]),
                    str(residual["reference_satellite"]),
                    int(residual["signal"]),
                )
                normalized = abs(float(residual["normalized_residual"]))
            except (KeyError, TypeError, ValueError):
                return None, float("nan")
            if not key[0] or not key[1] or not np.isfinite(normalized):
                return None, float("nan")
            keys.add(key)
        carrier_rows = [
            residual
            for residual in residuals
            if isinstance(residual, dict) and residual.get("kind") == "carrier"
        ]
        if not carrier_rows or (
            sum(residual.get("pass") is True for residual in carrier_rows)
            / len(carrier_rows)
            < float(minimum_carrier_pass_fraction)
        ):
            # A top-level pass inconsistent with its carrier details is
            # faulted evidence. Abort consensus instead of dropping one row.
            return None, float("nan")
        candidate_rows.append((candidate, residuals))
        key_sets.append(keys)
    common_keys = sorted(set.intersection(*key_sets)) if key_sets else []
    partitions = (set(common_keys[::2]), set(common_keys[1::2]))
    if any(
        len(partition) < int(minimum_satellites_per_partition)
        for partition in partitions
    ):
        return None, float("nan")

    winners: list[NativeFGOHypothesisCandidate] = []
    margins: list[float] = []
    for partition in partitions:
        scores = []
        for candidate, residuals in candidate_rows:
            values = []
            for residual in residuals:
                if not isinstance(residual, dict) or residual.get("kind") != "carrier":
                    continue
                key = (
                    str(residual.get("satellite", "")),
                    str(residual.get("reference_satellite", "")),
                    int(residual.get("signal", -1)),
                )
                if key in partition:
                    value = abs(float(residual.get("normalized_residual", np.nan)))
                    if not np.isfinite(value):
                        return None, float("nan")
                    values.append(min(value, float(residual_clip)) ** 2)
            if not values:
                return None, float("nan")
            scores.append((float(np.mean(values)), int(candidate.rank), candidate))
        scores.sort(key=lambda item: (item[0], item[1]))
        winners.append(scores[0][2])
        margins.append(scores[1][0] - scores[0][0])
    minimum_margin = min(margins)
    if winners[0].rank != winners[1].rank or minimum_margin < float(
        minimum_cost_margin
    ):
        return None, minimum_margin
    return winners[0], minimum_margin


def _carrier_details_consistent(
    native_rows: list[dict],
    candidates: list[NativeFGOHypothesisCandidate],
    *,
    group_index: int,
    minimum_pass_fraction: float,
) -> bool:
    """Require top-level passes to agree with detailed carrier decisions."""

    rows_by_rank = {
        int(row.get("rank", -1)): row
        for row in native_rows
        if row.get("group_index") == int(group_index)
        and row.get("evaluated") is True
        and row.get("pass") is True
    }
    for candidate in candidates:
        row = rows_by_rank.get(int(candidate.rank))
        residuals = row.get("validation_residuals") if row is not None else None
        if not isinstance(residuals, list):
            return False
        carrier_rows = [
            residual
            for residual in residuals
            if isinstance(residual, dict) and residual.get("kind") == "carrier"
        ]
        if not carrier_rows or (
            sum(residual.get("pass") is True for residual in carrier_rows)
            / len(carrier_rows)
            < float(minimum_pass_fraction)
        ):
            return False
    return True


def _causal_imu_motion_consensus_winner(
    native_rows: list[dict],
    candidates: list[NativeFGOHypothesisCandidate],
    anchors: list[CausalImuOffsetAnchor],
    *,
    tow_s: float,
    imu_position_ecef_m: np.ndarray,
    group_index: int,
    window_s: float,
    minimum_anchor_spacing_s: float,
    minimum_anchors_per_partition: int,
    maximum_residual_m: float,
    minimum_margin_m: float,
    minimum_carrier_pass_fraction: float,
) -> tuple[NativeFGOHypothesisCandidate | None, float, float, int]:
    """Select only when two causal anchor partitions prefer the same basin."""

    if len(candidates) < 2 or not _carrier_details_consistent(
        native_rows,
        candidates,
        group_index=int(group_index),
        minimum_pass_fraction=float(minimum_carrier_pass_fraction),
    ):
        return None, float("nan"), float("nan"), 0
    selected_anchors: list[CausalImuOffsetAnchor] = []
    for anchor in reversed(anchors):
        age_s = float(tow_s) - float(anchor.tow_s)
        if age_s < 0.0:
            return None, float("nan"), float("nan"), 0
        if age_s > float(window_s):
            break
        if not selected_anchors or selected_anchors[-1].tow_s - anchor.tow_s >= float(
            minimum_anchor_spacing_s
        ):
            selected_anchors.append(anchor)
    partitions = (selected_anchors[::2], selected_anchors[1::2])
    if any(
        len(partition) < int(minimum_anchors_per_partition) for partition in partitions
    ):
        return None, float("nan"), float("nan"), len(selected_anchors)

    current_imu = np.asarray(imu_position_ecef_m, dtype=np.float64).reshape(3)
    winners: list[NativeFGOHypothesisCandidate] = []
    winner_residuals: list[float] = []
    margins: list[float] = []
    for partition in partitions:
        offsets = np.asarray(
            [
                anchor.fixed_position_ecef_m - anchor.imu_position_ecef_m
                for anchor in partition
            ],
            dtype=np.float64,
        )
        if (
            offsets.ndim != 2
            or offsets.shape[1] != 3
            or not np.all(np.isfinite(offsets))
        ):
            return None, float("nan"), float("nan"), len(selected_anchors)
        predicted_offset = np.median(offsets, axis=0)
        scores = sorted(
            (
                float(
                    np.linalg.norm(
                        candidate.position_ecef_m - current_imu - predicted_offset
                    )
                ),
                int(candidate.rank),
                candidate,
            )
            for candidate in candidates
        )
        if not all(np.isfinite(score[0]) for score in scores):
            return None, float("nan"), float("nan"), len(selected_anchors)
        winners.append(scores[0][2])
        winner_residuals.append(scores[0][0])
        margins.append(scores[1][0] - scores[0][0])

    maximum_winner_residual = max(winner_residuals)
    minimum_margin = min(margins)
    if (
        winners[0].rank != winners[1].rank
        or maximum_winner_residual > float(maximum_residual_m)
        or minimum_margin < float(minimum_margin_m)
    ):
        return None, maximum_winner_residual, minimum_margin, len(selected_anchors)
    return winners[0], maximum_winner_residual, minimum_margin, len(selected_anchors)


def track_basin_rows(
    rows_by_epoch: dict[int, list[dict]],
    *,
    group_index: int = 0,
    likelihood_temperature: float = 0.1,
    max_basins: int = 64,
    parents_per_candidate: int = 2,
    fix_gamma_threshold: float = 0.99,
    fix_min_streak: int = 3,
    validation_conditioning: bool = True,
    validation_gap_tolerance_epochs: int = 0,
    imu_predictor: CausalBasinImuPredictor | None = None,
    native_imu_fgo: bool = False,
    native_imu_aperture_m: float = 0.0,
    native_imu_aperture_margin_m: float = 0.05,
    native_imu_fix_min_streak: int = 0,
    native_imu_motion_gate_m: float = 0.30,
    disjoint_holdout_consensus: bool = False,
    disjoint_holdout_margin: float = 0.02,
    disjoint_holdout_min_satellites: int = 2,
    disjoint_holdout_residual_clip: float = 3.0,
    disjoint_holdout_min_carrier_fraction: float = 0.75,
    causal_imu_motion_consensus: bool = False,
    causal_imu_motion_window_s: float = 10.0,
    causal_imu_motion_anchor_spacing_s: float = 0.4,
    causal_imu_motion_min_anchors_per_partition: int = 3,
    causal_imu_motion_gate_m: float = 0.75,
    causal_imu_motion_margin_m: float = 0.02,
    causal_imu_motion_min_carrier_fraction: float = 0.75,
    ffbsi_lag_epochs: int = 0,
    ffbsi_backward_samples: int = 128,
    ffbsi_seed: int = 0,
) -> list[dict[str, object]]:
    """Return truth-free causal shadow rows for one native basin stream."""

    if native_imu_aperture_m < 0.0:
        raise ValueError("native_imu_aperture_m must be non-negative")
    if native_imu_aperture_margin_m <= 0.0:
        raise ValueError("native_imu_aperture_margin_m must be positive")
    if native_imu_fix_min_streak < 0:
        raise ValueError("native_imu_fix_min_streak must be non-negative")
    if native_imu_fix_min_streak == 1:
        raise ValueError("native_imu_fix_min_streak must be 0 or at least 2")
    if native_imu_motion_gate_m <= 0.0:
        raise ValueError("native_imu_motion_gate_m must be positive")
    if disjoint_holdout_margin < 0.0:
        raise ValueError("disjoint_holdout_margin must be non-negative")
    if disjoint_holdout_min_satellites < 1:
        raise ValueError("disjoint_holdout_min_satellites must be positive")
    if disjoint_holdout_residual_clip <= 0.0:
        raise ValueError("disjoint_holdout_residual_clip must be positive")
    if not 0.0 < disjoint_holdout_min_carrier_fraction <= 1.0:
        raise ValueError("disjoint_holdout_min_carrier_fraction must be in (0, 1]")
    if causal_imu_motion_window_s <= 0.0:
        raise ValueError("causal_imu_motion_window_s must be positive")
    if causal_imu_motion_anchor_spacing_s <= 0.0:
        raise ValueError("causal_imu_motion_anchor_spacing_s must be positive")
    if causal_imu_motion_min_anchors_per_partition < 1:
        raise ValueError("causal_imu_motion_min_anchors_per_partition must be positive")
    if causal_imu_motion_gate_m <= 0.0:
        raise ValueError("causal_imu_motion_gate_m must be positive")
    if causal_imu_motion_margin_m < 0.0:
        raise ValueError("causal_imu_motion_margin_m must be non-negative")
    if not 0.0 < causal_imu_motion_min_carrier_fraction <= 1.0:
        raise ValueError("causal_imu_motion_min_carrier_fraction must be in (0, 1]")

    particle_filter = AmbiguityBasinParticleFilter(
        max_basins=int(max_basins),
        min_fixed_ambiguities=6,
        fix_gamma_threshold=float(fix_gamma_threshold),
        fix_min_streak=int(fix_min_streak),
        dedup_position_radius_m=0.05,
        diversity_reserve_fraction=0.25,
        diversity_radius_m=0.5,
    )
    output: list[dict[str, object]] = []
    validated_assignment: dict[object, int] | None = None
    validated_position_ecef: np.ndarray | None = None
    validated_imu_position_ecef: np.ndarray | None = None
    validated_fix_streak = 0
    validation_gap_epochs = 0
    native_validation_streak = 0
    previous_tow: float | None = None
    fallback_velocity = np.zeros(3, dtype=np.float64)
    previous_position: np.ndarray | None = None
    previous_native_imu: NativeImuFgoState | None = None
    causal_imu_anchors: list[CausalImuOffsetAnchor] = []
    smoother = (
        FixedLagBasinFFBSi(ffbsi_lag_epochs, ffbsi_backward_samples)
        if ffbsi_lag_epochs > 0
        else None
    )
    if imu_predictor is not None and rows_by_epoch:
        first_epoch = min(rows_by_epoch)
        imu_predictor.calibrate_before(float(rows_by_epoch[first_epoch][0]["tow"]))
    for epoch_index in sorted(rows_by_epoch):
        epoch_started = time.perf_counter()
        native_rows = rows_by_epoch[epoch_index]
        tow = float(native_rows[0]["tow"])
        dt = 0.0 if previous_tow is None else max(0.0, tow - previous_tow)
        imu_prediction = None
        native_imu_state = (
            _native_imu_fgo_state(native_rows) if native_imu_fgo else None
        )
        native_imu_motion_used = False
        native_validation_motion_residual_m = float("nan")
        if previous_tow is not None:
            prior_map = particle_filter.map_basin()
            if (
                native_imu_state is not None
                and previous_native_imu is not None
                and dt > 0.0
                and prior_map is not None
            ):
                displacement = (
                    native_imu_state.position_ecef_m
                    - previous_native_imu.position_ecef_m
                )
                particle_filter.predict_inertial(
                    dt,
                    cv_position_correction_ecef_m=(
                        displacement - previous_native_imu.velocity_ecef_mps * dt
                    ),
                    delta_velocity_ecef_mps=(
                        native_imu_state.velocity_ecef_mps
                        - previous_native_imu.velocity_ecef_mps
                    ),
                    process_covariance=_native_imu_process_covariance(dt),
                )
                native_imu_motion_used = True
            elif imu_predictor is not None and prior_map is not None:
                imu_prediction = imu_predictor.predict_interval(
                    previous_tow,
                    tow,
                    position_ecef_m=prior_map.conditional.mean[:3],
                    velocity_ecef_mps=prior_map.conditional.mean[3:6],
                )
            if native_imu_motion_used:
                pass
            elif imu_prediction is not None:
                particle_filter.predict_inertial(
                    dt,
                    cv_position_correction_ecef_m=(
                        imu_prediction.cv_position_correction_ecef_m
                    ),
                    delta_velocity_ecef_mps=(imu_prediction.delta_velocity_ecef_mps),
                    process_covariance=imu_prediction.process_covariance,
                )
            else:
                particle_filter.predict(dt)
        has_evaluated_group = any(
            row.get("group_index") == int(group_index) and row.get("evaluated") is True
            for row in native_rows
        )
        if has_evaluated_group:
            candidates = parse_native_fgo_hypotheses(
                {"multisd_validation_hypothesis_details": native_rows},
                group_index=int(group_index),
            )
            if native_imu_state is not None:
                # Proposal-only continuous feedback.  Relative likelihood and
                # the independent held-out pass stay exactly GNSS-derived.
                candidates = tuple(
                    replace(
                        candidate,
                        velocity_ecef_mps=(native_imu_state.velocity_ecef_mps.copy()),
                    )
                    for candidate in candidates
                )
            strict_passing = [
                candidate for candidate in candidates if candidate.validation_pass
            ]
            passing = strict_passing
            disjoint_holdout_selected = False
            disjoint_holdout_cost_margin = float("nan")
            if disjoint_holdout_consensus and len(passing) > 1:
                winner, disjoint_holdout_cost_margin = _disjoint_carrier_holdout_winner(
                    native_rows,
                    passing,
                    group_index=int(group_index),
                    minimum_satellites_per_partition=int(
                        disjoint_holdout_min_satellites
                    ),
                    minimum_cost_margin=float(disjoint_holdout_margin),
                    residual_clip=float(disjoint_holdout_residual_clip),
                    minimum_carrier_pass_fraction=float(
                        disjoint_holdout_min_carrier_fraction
                    ),
                )
                if winner is not None:
                    passing = [winner]
                    disjoint_holdout_selected = True
            imu_aperture_selected = False
            if (
                len(passing) > 1
                and native_imu_state is not None
                and native_imu_aperture_m > 0.0
                and validated_position_ecef is not None
                and validated_imu_position_ecef is not None
            ):
                inertial_target = (
                    validated_position_ecef
                    + native_imu_state.position_ecef_m
                    - validated_imu_position_ecef
                )
                ranked = sorted(
                    (
                        float(
                            np.linalg.norm(candidate.position_ecef_m - inertial_target)
                        ),
                        candidate,
                    )
                    for candidate in strict_passing
                )
                if ranked[0][0] <= float(native_imu_aperture_m) and ranked[1][
                    0
                ] - ranked[0][0] >= float(native_imu_aperture_margin_m):
                    passing = [ranked[0][1]]
                    imu_aperture_selected = True
            causal_imu_motion_selected = False
            causal_imu_motion_residual_m = float("nan")
            causal_imu_motion_margin = float("nan")
            causal_imu_motion_anchor_count = 0
            if (
                causal_imu_motion_consensus
                and len(passing) > 1
                and native_imu_state is not None
            ):
                (
                    winner,
                    causal_imu_motion_residual_m,
                    causal_imu_motion_margin,
                    causal_imu_motion_anchor_count,
                ) = _causal_imu_motion_consensus_winner(
                    native_rows,
                    strict_passing,
                    causal_imu_anchors,
                    tow_s=tow,
                    imu_position_ecef_m=native_imu_state.position_ecef_m,
                    group_index=int(group_index),
                    window_s=float(causal_imu_motion_window_s),
                    minimum_anchor_spacing_s=float(causal_imu_motion_anchor_spacing_s),
                    minimum_anchors_per_partition=int(
                        causal_imu_motion_min_anchors_per_partition
                    ),
                    maximum_residual_m=float(causal_imu_motion_gate_m),
                    minimum_margin_m=float(causal_imu_motion_margin_m),
                    minimum_carrier_pass_fraction=float(
                        causal_imu_motion_min_carrier_fraction
                    ),
                )
                if winner is not None:
                    passing = [winner]
                    causal_imu_motion_selected = True
            if len(passing) == 1:
                passing_assignment = dict(passing[0].assignment)
                if _assignments_compatible(validated_assignment, passing_assignment):
                    validated_fix_streak += 1
                else:
                    validated_fix_streak = 1
                if (
                    native_imu_state is not None
                    and validated_position_ecef is not None
                    and validated_imu_position_ecef is not None
                ):
                    predicted = (
                        validated_position_ecef
                        + native_imu_state.position_ecef_m
                        - validated_imu_position_ecef
                    )
                    native_validation_motion_residual_m = float(
                        np.linalg.norm(passing[0].position_ecef_m - predicted)
                    )
                    native_validation_streak = (
                        native_validation_streak + 1
                        if native_validation_motion_residual_m
                        <= float(native_imu_motion_gate_m)
                        else 1
                    )
                else:
                    native_validation_streak = 1 if native_imu_state is not None else 0
                validated_assignment = passing_assignment
                validated_position_ecef = passing[0].position_ecef_m.copy()
                validated_imu_position_ecef = (
                    native_imu_state.position_ecef_m.copy()
                    if native_imu_state is not None
                    else None
                )
                validation_gap_epochs = 0
            else:
                validation_gap_epochs += 1
                if validation_gap_epochs > int(validation_gap_tolerance_epochs):
                    validated_assignment = None
                    validated_position_ecef = None
                    validated_imu_position_ecef = None
                    validated_fix_streak = 0
                    native_validation_streak = 0
            transition_candidates = (
                passing if validation_conditioning and len(passing) == 1 else candidates
            )
            transition = transition_native_fgo_candidates(
                particle_filter,
                transition_candidates,
                fallback_velocity_ecef_mps=fallback_velocity,
                parents_per_candidate=int(parents_per_candidate),
                likelihood_temperature=float(likelihood_temperature),
            )
            posterior = particle_filter.posterior()
            if validation_conditioning and len(passing) != 1:
                particle_filter.invalidate_fix()
                posterior = particle_filter.posterior_snapshot()
        else:
            # A native epoch may have no independently evaluable group. This
            # is a normal fail-closed abstention, not malformed input.
            candidates = ()
            strict_passing = []
            passing = []
            imu_aperture_selected = False
            disjoint_holdout_selected = False
            disjoint_holdout_cost_margin = float("nan")
            causal_imu_motion_selected = False
            causal_imu_motion_residual_m = float("nan")
            causal_imu_motion_margin = float("nan")
            causal_imu_motion_anchor_count = 0
            validation_gap_epochs += 1
            if validation_gap_epochs > int(validation_gap_tolerance_epochs):
                validated_assignment = None
                validated_position_ecef = None
                validated_imu_position_ecef = None
                validated_fix_streak = 0
                native_validation_streak = 0
            particle_filter.invalidate_fix()
            posterior = particle_filter.posterior_snapshot()
            transition = None
        selected = particle_filter.map_basin()
        selected_assignment = selected.assignment_dict if selected else None
        unique_validation_pass = (
            len(passing) == 1
            and selected_assignment is not None
            and dict(passing[0].assignment) == selected_assignment
        )
        unique_holdout_pass = len(strict_passing) == 1 and unique_validation_pass
        selected_candidate = (
            passing[0]
            if unique_validation_pass
            else next(
                (
                    candidate
                    for candidate in candidates
                    if dict(candidate.assignment) == selected_assignment
                ),
                None,
            )
        )
        selected_rank = selected_candidate.rank if selected_candidate else -1
        imu_accelerated_fix = bool(
            native_imu_fix_min_streak >= 2
            and unique_validation_pass
            and (
                unique_holdout_pass
                or imu_aperture_selected
                or disjoint_holdout_selected
                or causal_imu_motion_selected
            )
            and posterior.gamma >= float(fix_gamma_threshold)
            and validated_fix_streak >= int(native_imu_fix_min_streak)
            and native_validation_streak >= int(native_imu_fix_min_streak)
        )
        if int(validation_gap_tolerance_epochs) > 0:
            fixed = (
                bool(
                    unique_holdout_pass
                    or imu_aperture_selected
                    or disjoint_holdout_selected
                    or causal_imu_motion_selected
                )
                and bool(
                    unique_validation_pass
                    and posterior.gamma >= float(fix_gamma_threshold)
                    and validated_fix_streak >= int(fix_min_streak)
                )
                or imu_accelerated_fix
            )
        else:
            fixed = (
                bool(
                    posterior.fixed
                    and unique_validation_pass
                    and (
                        unique_holdout_pass
                        or imu_aperture_selected
                        or disjoint_holdout_selected
                        or causal_imu_motion_selected
                    )
                )
                or imu_accelerated_fix
            )
        position = (
            passing[0].position_ecef_m.copy()
            if fixed
            else selected.conditional.mean[:3].copy()
            if selected
            else np.full(3, np.nan)
        )
        if previous_position is not None and dt > 0.0 and np.all(np.isfinite(position)):
            fallback_velocity = (position - previous_position) / dt
        if np.all(np.isfinite(position)):
            previous_position = position
        if fixed and native_imu_state is not None and np.all(np.isfinite(position)):
            causal_imu_anchors.append(
                CausalImuOffsetAnchor(
                    tow_s=tow,
                    fixed_position_ecef_m=position.copy(),
                    imu_position_ecef_m=native_imu_state.position_ecef_m.copy(),
                )
            )
            minimum_tow = tow - float(causal_imu_motion_window_s)
            causal_imu_anchors = [
                anchor for anchor in causal_imu_anchors if anchor.tow_s >= minimum_tow
            ]
        previous_tow = tow
        smoothed = None
        if smoother is not None:
            smoother.capture(particle_filter, tow)
            smoothed = smoother.estimate(seed=int(ffbsi_seed) + int(epoch_index))
        row = {
            "epoch_index": epoch_index,
            "tow": tow,
            "shadow_fixed": int(fixed),
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "posterior_gamma": posterior.gamma,
            "posterior_ess": posterior.ess,
            "basins": posterior.n_basins,
            "fix_streak": posterior.fix_streak,
            "validated_fix_streak": validated_fix_streak,
            "validation_gap_epochs": validation_gap_epochs,
            "selected_rank": selected_rank,
            "unique_holdout_pass": int(unique_holdout_pass),
            "unique_validation_pass": int(unique_validation_pass),
            "imu_aperture_selected": int(imu_aperture_selected),
            "disjoint_holdout_selected": int(disjoint_holdout_selected),
            "disjoint_holdout_cost_margin": disjoint_holdout_cost_margin,
            "causal_imu_motion_selected": int(causal_imu_motion_selected),
            "causal_imu_motion_residual_m": causal_imu_motion_residual_m,
            "causal_imu_motion_margin_m": causal_imu_motion_margin,
            "causal_imu_motion_anchor_count": causal_imu_motion_anchor_count,
            "strict_passing_candidates": len(strict_passing),
            "native_validation_streak": native_validation_streak,
            "native_validation_motion_residual_m": (
                native_validation_motion_residual_m
            ),
            "imu_accelerated_fix": int(imu_accelerated_fix),
            "candidate_count": len(candidates),
            "transition_branches": (
                transition.parent_child_branches if transition else 0
            ),
            "minimum_conflicts": transition.minimum_conflicts if transition else 0,
            "maximum_conflicts": transition.maximum_conflicts if transition else 0,
            "imu_used": int(imu_prediction is not None or native_imu_motion_used),
            "imu_source": (
                "native_fgo"
                if native_imu_motion_used
                else "legacy_bridge"
                if imu_prediction is not None
                else "none"
            ),
            "native_imu_fgo_available": int(native_imu_state is not None),
            "native_imu_motion_used": int(native_imu_motion_used),
            "imu_samples": (
                imu_prediction.sample_count if imu_prediction is not None else 0
            ),
            "imu_covered_duration_s": (
                imu_prediction.covered_duration_s if imu_prediction is not None else 0.0
            ),
            "imu_position_correction_m": (
                float(np.linalg.norm(imu_prediction.cv_position_correction_ecef_m))
                if imu_prediction is not None
                else 0.0
            ),
            "imu_delta_velocity_mps": (
                float(np.linalg.norm(imu_prediction.delta_velocity_ecef_mps))
                if imu_prediction is not None
                else 0.0
            ),
            "ffbsi_valid": int(smoothed is not None),
            "ffbsi_tow": smoothed.target_tow_s if smoothed else np.nan,
            "ffbsi_x": smoothed.position_ecef_m[0] if smoothed else np.nan,
            "ffbsi_y": smoothed.position_ecef_m[1] if smoothed else np.nan,
            "ffbsi_z": smoothed.position_ecef_m[2] if smoothed else np.nan,
            "ffbsi_assignment_probability": (
                smoothed.assignment_probability if smoothed else 0.0
            ),
            "ffbsi_effective_samples": (smoothed.effective_samples if smoothed else 0),
        }
        row["epoch_runtime_ms"] = (time.perf_counter() - epoch_started) * 1000.0
        output.append(row)
        previous_native_imu = native_imu_state
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basin-jsonl", type=Path, required=True)
    parser.add_argument("--imu-csv", type=Path)
    parser.add_argument(
        "--native-imu-fgo",
        action="store_true",
        help="consume embedded native IMU-FGO states as PF proposals only",
    )
    parser.add_argument(
        "--native-imu-aperture",
        type=float,
        default=0.0,
        metavar="M",
        help="select among GNSS-passing basins only when relative IMU-FGO motion innovation is within M metres",
    )
    parser.add_argument(
        "--native-imu-aperture-margin",
        type=float,
        default=0.05,
        metavar="M",
        help="required nearest-vs-runner-up IMU-FGO separation margin",
    )
    parser.add_argument(
        "--native-imu-fix-min-streak",
        type=int,
        default=0,
        help="allow IMU-motion-consistent FIX after this many GNSS-passing epochs (0 disables)",
    )
    parser.add_argument(
        "--native-imu-motion-gate",
        type=float,
        default=0.30,
        metavar="M",
        help="maximum candidate-vs-relative-IMU displacement residual for accelerated FIX",
    )
    parser.add_argument("--disjoint-holdout-consensus", action="store_true")
    parser.add_argument("--disjoint-holdout-margin", type=float, default=0.02)
    parser.add_argument("--disjoint-holdout-min-satellites", type=int, default=2)
    parser.add_argument("--disjoint-holdout-residual-clip", type=float, default=3.0)
    parser.add_argument(
        "--disjoint-holdout-min-carrier-fraction", type=float, default=0.75
    )
    parser.add_argument("--causal-imu-motion-consensus", action="store_true")
    parser.add_argument("--causal-imu-motion-window", type=float, default=10.0)
    parser.add_argument("--causal-imu-motion-anchor-spacing", type=float, default=0.4)
    parser.add_argument(
        "--causal-imu-motion-min-anchors-per-partition", type=int, default=3
    )
    parser.add_argument("--causal-imu-motion-gate", type=float, default=0.75)
    parser.add_argument("--causal-imu-motion-margin", type=float, default=0.02)
    parser.add_argument(
        "--causal-imu-motion-min-carrier-fraction", type=float, default=0.75
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--pf-feedback-csv",
        type=Path,
        help="write validated PF integer modes for a strictly causal next solver pass",
    )
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--likelihood-temperature", type=float, default=0.1)
    parser.add_argument("--max-basins", type=int, default=64)
    parser.add_argument("--parents-per-candidate", type=int, default=2)
    parser.add_argument("--fix-gamma", type=float, default=0.99)
    parser.add_argument("--fix-min-streak", type=int, default=3)
    parser.add_argument("--validation-gap-tolerance", type=int, default=0)
    parser.add_argument("--ffbsi-lag", type=int, default=0)
    parser.add_argument("--ffbsi-samples", type=int, default=128)
    parser.add_argument("--ffbsi-seed", type=int, default=0)
    parser.add_argument(
        "--no-validation-conditioning",
        action="store_false",
        dest="validation_conditioning",
        help="retain failed/multiple holdout candidates in the PF (diagnostic only)",
    )
    args = parser.parse_args(argv)

    total_started = time.perf_counter()
    setup_started = time.perf_counter()
    imu_predictor = (
        CausalBasinImuPredictor(load_ppc_imu_csv(args.imu_csv))
        if args.imu_csv is not None
        else None
    )
    basin_rows = load_basin_rows(args.basin_jsonl)
    setup_ms = (time.perf_counter() - setup_started) * 1000.0
    tracking_started = time.perf_counter()
    rows = track_basin_rows(
        basin_rows,
        group_index=args.group_index,
        likelihood_temperature=args.likelihood_temperature,
        max_basins=args.max_basins,
        parents_per_candidate=args.parents_per_candidate,
        fix_gamma_threshold=args.fix_gamma,
        fix_min_streak=args.fix_min_streak,
        validation_conditioning=args.validation_conditioning,
        validation_gap_tolerance_epochs=args.validation_gap_tolerance,
        imu_predictor=imu_predictor,
        native_imu_fgo=args.native_imu_fgo,
        native_imu_aperture_m=args.native_imu_aperture,
        native_imu_aperture_margin_m=args.native_imu_aperture_margin,
        native_imu_fix_min_streak=args.native_imu_fix_min_streak,
        native_imu_motion_gate_m=args.native_imu_motion_gate,
        disjoint_holdout_consensus=args.disjoint_holdout_consensus,
        disjoint_holdout_margin=args.disjoint_holdout_margin,
        disjoint_holdout_min_satellites=args.disjoint_holdout_min_satellites,
        disjoint_holdout_residual_clip=args.disjoint_holdout_residual_clip,
        disjoint_holdout_min_carrier_fraction=(
            args.disjoint_holdout_min_carrier_fraction
        ),
        causal_imu_motion_consensus=args.causal_imu_motion_consensus,
        causal_imu_motion_window_s=args.causal_imu_motion_window,
        causal_imu_motion_anchor_spacing_s=args.causal_imu_motion_anchor_spacing,
        causal_imu_motion_min_anchors_per_partition=(
            args.causal_imu_motion_min_anchors_per_partition
        ),
        causal_imu_motion_gate_m=args.causal_imu_motion_gate,
        causal_imu_motion_margin_m=args.causal_imu_motion_margin,
        causal_imu_motion_min_carrier_fraction=(
            args.causal_imu_motion_min_carrier_fraction
        ),
        ffbsi_lag_epochs=args.ffbsi_lag,
        ffbsi_backward_samples=args.ffbsi_samples,
        ffbsi_seed=args.ffbsi_seed,
    )
    tracking_ms = (time.perf_counter() - tracking_started) * 1000.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        list(rows[0]) if rows else ["epoch_index", "tow", "shadow_fixed", "x", "y", "z"]
    )
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    feedback_rows = (
        write_validated_pf_feedback(
            args.pf_feedback_csv,
            rows,
            basin_rows,
            group_index=args.group_index,
        )
        if args.pf_feedback_csv is not None
        else 0
    )
    total_ms = (time.perf_counter() - total_started) * 1000.0
    summary = {
        "schema": "gnss_gpu_ppc_basin_fgo_tracker_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "estimator_input_kinds": ["native_basin_jsonl"]
        + (["ppc_imu"] if args.imu_csv is not None else [])
        + (["embedded_native_imu_fgo"] if args.native_imu_fgo else []),
        "input_sha256": _sha256(args.basin_jsonl),
        "config": {
            "group_index": args.group_index,
            "likelihood_temperature": args.likelihood_temperature,
            "max_basins": args.max_basins,
            "parents_per_candidate": args.parents_per_candidate,
            "fix_gamma": args.fix_gamma,
            "fix_min_streak": args.fix_min_streak,
            "validation_conditioning": args.validation_conditioning,
            "validation_gap_tolerance_epochs": args.validation_gap_tolerance,
            "native_imu_fgo": args.native_imu_fgo,
            "native_imu_aperture_m": args.native_imu_aperture,
            "native_imu_aperture_margin_m": args.native_imu_aperture_margin,
            "native_imu_fix_min_streak": args.native_imu_fix_min_streak,
            "native_imu_motion_gate_m": args.native_imu_motion_gate,
            "disjoint_holdout_consensus": args.disjoint_holdout_consensus,
            "disjoint_holdout_margin": args.disjoint_holdout_margin,
            "disjoint_holdout_min_satellites": (args.disjoint_holdout_min_satellites),
            "disjoint_holdout_residual_clip": (args.disjoint_holdout_residual_clip),
            "disjoint_holdout_min_carrier_fraction": (
                args.disjoint_holdout_min_carrier_fraction
            ),
            "causal_imu_motion_consensus": args.causal_imu_motion_consensus,
            "causal_imu_motion_window_s": args.causal_imu_motion_window,
            "causal_imu_motion_anchor_spacing_s": (
                args.causal_imu_motion_anchor_spacing
            ),
            "causal_imu_motion_min_anchors_per_partition": (
                args.causal_imu_motion_min_anchors_per_partition
            ),
            "causal_imu_motion_gate_m": args.causal_imu_motion_gate,
            "causal_imu_motion_margin_m": args.causal_imu_motion_margin,
            "causal_imu_motion_min_carrier_fraction": (
                args.causal_imu_motion_min_carrier_fraction
            ),
            "ffbsi_lag_epochs": args.ffbsi_lag,
            "ffbsi_backward_samples": args.ffbsi_samples,
            "ffbsi_seed": args.ffbsi_seed,
        },
        "epochs": len(rows),
        "fixed_epochs": sum(int(row["shadow_fixed"]) for row in rows),
        "imu_prediction_epochs": sum(int(row.get("imu_used", 0)) for row in rows),
        "native_imu_fgo_available_epochs": sum(
            int(row.get("native_imu_fgo_available", 0)) for row in rows
        ),
        "native_imu_motion_epochs": sum(
            int(row.get("native_imu_motion_used", 0)) for row in rows
        ),
        "native_imu_aperture_epochs": sum(
            int(row.get("imu_aperture_selected", 0)) for row in rows
        ),
        "native_imu_accelerated_fix_epochs": sum(
            int(row.get("imu_accelerated_fix", 0)) for row in rows
        ),
        "disjoint_holdout_selected_epochs": sum(
            int(row.get("disjoint_holdout_selected", 0)) for row in rows
        ),
        "causal_imu_motion_selected_epochs": sum(
            int(row.get("causal_imu_motion_selected", 0)) for row in rows
        ),
        "ffbsi_output_epochs": sum(int(row.get("ffbsi_valid", 0)) for row in rows),
        "setup_runtime_ms": setup_ms,
        "tracking_runtime_ms": tracking_ms,
        "tracking_runtime_per_epoch_ms": (tracking_ms / len(rows) if rows else 0.0),
        "tracking_runtime_p95_ms": (
            float(np.quantile([row["epoch_runtime_ms"] for row in rows], 0.95))
            if rows
            else 0.0
        ),
        "tracking_runtime_maximum_ms": (
            max(float(row["epoch_runtime_ms"]) for row in rows) if rows else 0.0
        ),
        "runtime_ms": total_ms,
        "runtime_per_epoch_ms": tracking_ms / len(rows) if rows else 0.0,
        "output_sha256": _sha256(args.output),
        "pf_feedback_rows": feedback_rows,
        "pf_feedback_sha256": (
            _sha256(args.pf_feedback_csv) if args.pf_feedback_csv is not None else None
        ),
        "imu_input_sha256": _sha256(args.imu_csv) if args.imu_csv else None,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
