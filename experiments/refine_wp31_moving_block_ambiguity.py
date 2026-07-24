#!/usr/bin/env python3
"""Generate a moving-route anchor by resolving carrier integers over a block.

Truth is deliberately excluded from fitting and gating.  It is read only after
the selected hypothesis has been frozen, to produce an audit field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from scipy.optimize import least_squares
from shapely.geometry import LineString
from shapely.strtree import STRtree

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from build_wp31_osm_particle_route_bridge import _road_distances  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.nmea_writer import _ecef_to_lla_py  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.lambda_ambiguity import integer_search  # noqa: E402
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402
from run_wp29_tdcp_anchor_smoother import (  # noqa: E402
    _load_fusion_static_override,
    _load_static_position_override,
)


@dataclass(frozen=True)
class CarrierRow:
    epoch: int
    key: tuple[str, str, int]
    measured_cycles: float
    wavelength_m: float
    sat_ecef_k: np.ndarray
    sat_ecef_ref: np.ndarray
    base_range_k: float
    base_range_ref: float


def bias_correct_ddpr_epoch(
    obs: DDPseudorangeEpoch,
    pair_biases: dict[tuple[str, str], tuple[float, int]],
    *,
    epoch: int,
    max_age_epochs: int,
) -> DDPseudorangeEpoch | None:
    """Subtract only fresh, static-anchor-derived exact DD-pair biases."""

    if obs.sat_ids is None or obs.ref_sat_ids is None:
        return None
    retained: list[int] = []
    corrections: list[float] = []
    for index, (ref_sat, sat_id) in enumerate(zip(obs.ref_sat_ids, obs.sat_ids)):
        pair = pair_biases.get((ref_sat, sat_id))
        if pair is None:
            continue
        if epoch - pair[1] > max_age_epochs:
            continue
        retained.append(index)
        corrections.append(pair[0])
    if not retained:
        return None
    indices = np.asarray(retained, dtype=np.int64)
    return DDPseudorangeEpoch(
        dd_pseudorange_m=np.asarray(obs.dd_pseudorange_m)[indices]
        - np.asarray(corrections),
        sat_ecef_k=np.asarray(obs.sat_ecef_k)[indices],
        sat_ecef_ref=np.asarray(obs.sat_ecef_ref)[indices],
        base_range_k=np.asarray(obs.base_range_k)[indices],
        base_range_ref=np.asarray(obs.base_range_ref)[indices],
        weights=None if obs.weights is None else np.asarray(obs.weights)[indices],
        sat_ids=tuple(obs.sat_ids[index] for index in retained),
        ref_sat_ids=tuple(obs.ref_sat_ids[index] for index in retained),
    )


def filter_ddpr_excluded_satellites(
    obs: DDPseudorangeEpoch,
    excluded: frozenset[str],
) -> DDPseudorangeEpoch | None:
    """Drop DD pairs involving any satellite in ``excluded`` before gates/fitting."""

    if not excluded or obs.sat_ids is None or obs.ref_sat_ids is None:
        return obs
    retained = [
        index
        for index, (ref_sat, sat_id) in enumerate(zip(obs.ref_sat_ids, obs.sat_ids))
        if ref_sat not in excluded and sat_id not in excluded
    ]
    if len(retained) == len(obs.sat_ids):
        return obs
    if not retained:
        return None
    indices = np.asarray(retained, dtype=np.int64)
    return DDPseudorangeEpoch(
        dd_pseudorange_m=np.asarray(obs.dd_pseudorange_m)[indices],
        sat_ecef_k=np.asarray(obs.sat_ecef_k)[indices],
        sat_ecef_ref=np.asarray(obs.sat_ecef_ref)[indices],
        base_range_k=np.asarray(obs.base_range_k)[indices],
        base_range_ref=np.asarray(obs.base_range_ref)[indices],
        weights=None if obs.weights is None else np.asarray(obs.weights)[indices],
        sat_ids=tuple(obs.sat_ids[index] for index in retained),
        ref_sat_ids=tuple(obs.ref_sat_ids[index] for index in retained),
    )


def phase_epochs(start: int, end: int, stride: int, phase: int) -> range:
    if stride <= 0 or not 0 <= phase < stride:
        raise ValueError("evidence stride and phase are invalid")
    first = start + ((phase - start) % stride)
    return range(first, end, stride)


def choose_evidence_phase(diagnostics: list[dict[str, int]]) -> int:
    if not diagnostics:
        raise ValueError("evidence phase diagnostics are empty")
    winner = max(
        diagnostics,
        key=lambda row: (
            row["evidence_epochs"],
            row["raw_carrier_rows"],
            row["ddpr_epochs"],
            -row["phase"],
        ),
    )
    return int(winner["phase"])


def _read_trajectory(path: Path, start: int, end: int) -> dict[int, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return {
            int(row["epoch"]): np.asarray(
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                dtype=np.float64,
            )
            for row in csv.DictReader(fh)
            if start <= int(row["epoch"]) < end
        }


def fixed_boundary_affine_route(
    route: dict[int, np.ndarray],
    *,
    start: int,
    boundary_epoch: int,
    boundary_offset_ecef_m: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Blend a fitted start correction into a fixed truth-free boundary offset."""

    if not start < boundary_epoch or not route:
        raise ValueError("fixed affine boundary inputs are invalid")
    boundary_offset = np.asarray(boundary_offset_ecef_m, dtype=np.float64)
    if boundary_offset.shape != (3,) or not np.all(np.isfinite(boundary_offset)):
        raise ValueError("fixed affine boundary offset is invalid")
    denominator = float(boundary_epoch - start)
    scales = {epoch: float(boundary_epoch - epoch) / denominator for epoch in route}
    if any(not 0.0 < scale <= 1.0 for scale in scales.values()):
        raise ValueError("fixed affine boundary does not follow the moving block")
    adjusted = {
        epoch: position + (1.0 - scales[epoch]) * boundary_offset
        for epoch, position in route.items()
    }
    return adjusted, scales


def _load_right_boundary_profile(path: Path, expected_epoch: int) -> dict[str, Any]:
    source_bytes = path.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    if bool(payload.get("production_input_truth", True)):
        raise ValueError("right boundary profile is not production-safe")
    if not bool(payload.get("production_promoted", False)):
        raise ValueError("right boundary profile is not promoted")
    if payload.get("profile_mode") not in {
        "right_boundary_affine_zero",
        "right_boundary_affine_fixed",
    }:
        raise ValueError("right boundary profile mode is unsupported")
    segment = payload.get("segment", [])
    if len(segment) != 2 or int(segment[0]) != expected_epoch:
        raise ValueError("right boundary profile does not begin at block end")
    offset = np.asarray(payload.get("offset_ecef_m"), dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("right boundary profile offset is invalid")
    return {
        "epoch": int(segment[0]),
        "offset_ecef_m": offset,
        "reason": str(payload.get("reason", "")),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _carrier_rows(epoch: int, obs: DDCarrierEpoch) -> list[CarrierRow]:
    if obs.sat_ids is None or obs.ref_sat_ids is None:
        return []
    rows = []
    for index, (ref_sat, sat_id) in enumerate(zip(obs.ref_sat_ids, obs.sat_ids)):
        wavelength = float(obs.wavelengths_m[index])
        rows.append(
            CarrierRow(
                epoch=epoch,
                key=(str(ref_sat), str(sat_id), int(round(wavelength * 1.0e9))),
                measured_cycles=float(obs.dd_carrier_cycles[index]),
                wavelength_m=wavelength,
                sat_ecef_k=np.asarray(obs.sat_ecef_k[index]),
                sat_ecef_ref=np.asarray(obs.sat_ecef_ref[index]),
                base_range_k=float(obs.base_range_k[index]),
                base_range_ref=float(obs.base_range_ref[index]),
            )
        )
    return rows


def segment_carrier_arcs(
    trajectory: dict[int, np.ndarray],
    rows: list[CarrierRow],
    *,
    max_epoch_gap: int,
    max_float_jump_cycles: float = 0.75,
) -> tuple[list[CarrierRow], int]:
    """Split exact DD keys at observation gaps or route-referenced phase jumps."""
    grouped: dict[tuple[str, str, int], list[CarrierRow]] = {}
    for row in rows:
        grouped.setdefault(row.key, []).append(row)
    output: list[CarrierRow] = []
    split_count = 0
    for key, values in grouped.items():
        arc = 0
        previous_epoch: int | None = None
        previous_float: float | None = None
        for row in sorted(values, key=lambda item: item.epoch):
            expected, _ = _dd_expected_and_jacobian_m(
                trajectory[row.epoch],
                row.sat_ecef_k,
                row.sat_ecef_ref,
                row.base_range_k,
                row.base_range_ref,
            )
            floating = row.measured_cycles - expected / row.wavelength_m
            if previous_epoch is not None and (
                row.epoch - previous_epoch > max_epoch_gap
                or abs(floating - float(previous_float)) > max_float_jump_cycles
            ):
                arc += 1
                split_count += 1
            output.append(
                CarrierRow(
                    row.epoch,
                    (key[0], f"{key[1]}@arc{arc}", key[2]),
                    row.measured_cycles,
                    row.wavelength_m,
                    row.sat_ecef_k,
                    row.sat_ecef_ref,
                    row.base_range_k,
                    row.base_range_ref,
                )
            )
            previous_epoch = row.epoch
            previous_float = floating
    return output, split_count


def estimate_arc_integers(
    offset: np.ndarray,
    trajectory: dict[int, np.ndarray],
    rows: list[CarrierRow],
    *,
    min_arc_epochs: int = 3,
    max_center_residual_cycles: float = 0.45,
    offset_scales: dict[int, float] | None = None,
) -> tuple[dict[tuple[str, str, int], int], list[CarrierRow]]:
    """Median-round one integer per DD signal arc, then remove incoherent rows."""
    grouped: dict[tuple[str, str, int], list[tuple[CarrierRow, float]]] = {}
    for row in rows:
        scale = 1.0 if offset_scales is None else float(offset_scales[row.epoch])
        position = trajectory[row.epoch] + scale * np.asarray(offset)
        expected, _ = _dd_expected_and_jacobian_m(
            position,
            row.sat_ecef_k,
            row.sat_ecef_ref,
            row.base_range_k,
            row.base_range_ref,
        )
        floating = row.measured_cycles - expected / row.wavelength_m
        grouped.setdefault(row.key, []).append((row, float(floating)))
    integers: dict[tuple[str, str, int], int] = {}
    retained: list[CarrierRow] = []
    for key, values in grouped.items():
        epochs = {item.epoch for item, _ in values}
        if len(epochs) < int(min_arc_epochs):
            continue
        center = float(np.median([value for _, value in values]))
        integer = int(np.rint(center))
        coherent = [
            item
            for item, value in values
            if abs(value - integer) <= max_center_residual_cycles
        ]
        if len({item.epoch for item in coherent}) >= int(min_arc_epochs):
            integers[key] = integer
            retained.extend(coherent)
    return integers, retained


def _raw_residuals(
    offset: np.ndarray,
    trajectory: dict[int, np.ndarray],
    carrier_rows: list[CarrierRow],
    integers: dict[tuple[str, str, int], int],
    ddpr: dict[int, DDPseudorangeEpoch],
    epochs: set[int] | None = None,
    offset_scales: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cp_values, pr_values = [], []
    for row in carrier_rows:
        if epochs is not None and row.epoch not in epochs:
            continue
        scale = 1.0 if offset_scales is None else float(offset_scales[row.epoch])
        expected, _ = _dd_expected_and_jacobian_m(
            trajectory[row.epoch] + scale * offset,
            row.sat_ecef_k,
            row.sat_ecef_ref,
            row.base_range_k,
            row.base_range_ref,
        )
        cp_values.append(
            row.measured_cycles - expected / row.wavelength_m - integers[row.key]
        )
    for epoch, obs in ddpr.items():
        if epochs is not None and epoch not in epochs:
            continue
        scale = 1.0 if offset_scales is None else float(offset_scales[epoch])
        position = trajectory[epoch] + scale * offset
        for index in range(obs.n):
            expected, _ = _dd_expected_and_jacobian_m(
                position,
                obs.sat_ecef_k[index],
                obs.sat_ecef_ref[index],
                obs.base_range_k[index],
                obs.base_range_ref[index],
            )
            pr_values.append(float(obs.dd_pseudorange_m[index]) - expected)
    return np.asarray(cp_values), np.asarray(pr_values)


def cp_pr_consistency(
    carrier_rows: list[CarrierRow],
    integers: dict[tuple[str, str, int], int],
    ddpr: dict[int, DDPseudorangeEpoch],
    *,
    bad_pair_threshold_m: float = 5.0,
) -> dict[str, float | int]:
    """Compare fixed DD carrier range with DDPR after reference rebasing."""

    ddpr_groups: dict[tuple[int, str], dict[str, float]] = {}
    for epoch, obs in ddpr.items():
        if obs.sat_ids is None or obs.ref_sat_ids is None:
            continue
        for index, (ref_sat, sat_id) in enumerate(zip(obs.ref_sat_ids, obs.sat_ids)):
            ref = str(ref_sat).split("@", 1)[0]
            sat = str(sat_id).split("@", 1)[0]
            group = ddpr_groups.setdefault((epoch, ref), {ref: 0.0})
            group[sat] = float(obs.dd_pseudorange_m[index])
    innovations = []
    for row in carrier_rows:
        ref = str(row.key[0]).split("@", 1)[0]
        sat = re.sub(r"@arc\d+$", "", str(row.key[1])).split("@", 1)[0]
        group = next(
            (
                values
                for (epoch, _root_ref), values in ddpr_groups.items()
                if epoch == row.epoch and ref in values and sat in values
            ),
            None,
        )
        if group is None:
            continue
        fixed_carrier_m = (float(row.measured_cycles) - int(integers[row.key])) * float(
            row.wavelength_m
        )
        innovations.append(float(group[sat] - group[ref]) - fixed_carrier_m)
    values = np.asarray(innovations, dtype=np.float64)
    if values.size == 0:
        return {
            "checked_pairs": 0,
            "bad_pairs": 0,
            "rms_innovation_m": float("inf"),
            "median_abs_innovation_m": float("inf"),
            "p95_abs_innovation_m": float("inf"),
        }
    absolute = np.abs(values)
    return {
        "checked_pairs": int(values.size),
        "bad_pairs": int(np.count_nonzero(absolute > float(bad_pair_threshold_m))),
        "rms_innovation_m": float(np.sqrt(np.mean(np.square(values)))),
        "median_abs_innovation_m": float(np.median(absolute)),
        "p95_abs_innovation_m": float(np.percentile(absolute, 95.0)),
    }


def optimize_fixed_integers(
    seed: np.ndarray,
    trajectory: dict[int, np.ndarray],
    carrier_rows: list[CarrierRow],
    integers: dict[tuple[str, str, int], int],
    ddpr: dict[int, DDPseudorangeEpoch],
    *,
    carrier_sigma_cycles: float = 0.5,
    ddpr_sigma_m: float = 4.0,
    prior_sigma_m: float = 40.0,
    epochs: set[int] | None = None,
    offset_scales: dict[int, float] | None = None,
    up_prior: tuple[np.ndarray, float, float] | None = None,
) -> np.ndarray:
    seed = np.asarray(seed, dtype=np.float64)

    def residual(value: np.ndarray) -> np.ndarray:
        cp, pr = _raw_residuals(
            value,
            trajectory,
            carrier_rows,
            integers,
            ddpr,
            epochs,
            offset_scales,
        )
        parts = [
            cp / carrier_sigma_cycles,
            pr / ddpr_sigma_m,
            (value - seed) / prior_sigma_m,
        ]
        if up_prior is not None:
            local_up, center_m, sigma_m = up_prior
            parts.append(
                np.asarray([(float(np.dot(value, local_up)) - center_m) / sigma_m])
            )
        return np.concatenate(parts)

    return np.asarray(
        least_squares(residual, seed, loss="huber", f_scale=1.5, max_nfev=120).x
    )


def solve_hypothesis(
    seed: np.ndarray,
    trajectory: dict[int, np.ndarray],
    all_rows: list[CarrierRow],
    ddpr: dict[int, DDPseudorangeEpoch],
    *,
    iterations: int = 4,
    min_arc_epochs: int = 3,
    offset_scales: dict[int, float] | None = None,
    up_prior: tuple[np.ndarray, float, float] | None = None,
) -> dict[str, Any] | None:
    offset = np.asarray(seed, dtype=np.float64)
    integers: dict[tuple[str, str, int], int] = {}
    rows: list[CarrierRow] = []
    for _ in range(iterations):
        integers, rows = estimate_arc_integers(
            offset,
            trajectory,
            all_rows,
            min_arc_epochs=min_arc_epochs,
            offset_scales=offset_scales,
        )
        if not rows:
            return None
        offset = optimize_fixed_integers(
            offset,
            trajectory,
            rows,
            integers,
            ddpr,
            offset_scales=offset_scales,
            up_prior=up_prior,
        )
    cp, pr = _raw_residuals(
        offset,
        trajectory,
        rows,
        integers,
        ddpr,
        offset_scales=offset_scales,
    )
    return {
        "offset_ecef_m": offset,
        "integer_arcs": len(integers),
        "carrier_rows": len(cp),
        "ddpr_rows": len(pr),
        "carrier_rms_cycles": float(np.sqrt(np.mean(np.square(cp))))
        if len(cp)
        else float("inf"),
        "ddpr_rms_m": float(np.sqrt(np.mean(np.square(pr))))
        if len(pr)
        else float("inf"),
        "cp_pr_consistency": cp_pr_consistency(rows, integers, ddpr),
        "integers": integers,
        "retained_rows": rows,
    }


def float_ambiguity_seeds(
    trajectory: dict[int, np.ndarray],
    all_rows: list[CarrierRow],
    ddpr: dict[int, DDPseudorangeEpoch],
    *,
    min_arc_epochs: int = 3,
    carrier_sigma_cycles: float = 0.5,
    ddpr_sigma_m: float = 20.0,
    position_prior_sigma_m: float = 40.0,
    up_prior_sigma_m: float = 2.0,
    up_prior_center_m: float = 0.0,
    n_candidates: int = 12,
    partial_ar_max_drop_steps: int = 0,
    partial_ar_worst_axes: int = 3,
    partial_ar_candidates_per_subset: int = 2,
    offset_scales: dict[int, float] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Jointly float position/ambiguities, then map ILS candidates to positions."""
    grouped: dict[tuple[str, str, int], list[CarrierRow]] = {}
    for row in all_rows:
        grouped.setdefault(row.key, []).append(row)
    keys = sorted(
        key
        for key, rows in grouped.items()
        if len({row.epoch for row in rows}) >= int(min_arc_epochs)
    )
    if not keys:
        return [], {"float_integer_arcs": 0, "lambda_candidates": 0}
    key_index = {key: index for index, key in enumerate(keys)}
    design, target = [], []
    for row in all_rows:
        if row.key not in key_index:
            continue
        expected, jacobian = _dd_expected_and_jacobian_m(
            trajectory[row.epoch],
            row.sat_ecef_k,
            row.sat_ecef_ref,
            row.base_range_k,
            row.base_range_ref,
        )
        vector = np.zeros(3 + len(keys))
        scale = 1.0 if offset_scales is None else float(offset_scales[row.epoch])
        vector[:3] = (
            scale * np.asarray(jacobian) / row.wavelength_m / carrier_sigma_cycles
        )
        vector[3 + key_index[row.key]] = 1.0 / carrier_sigma_cycles
        design.append(vector)
        target.append(
            (row.measured_cycles - expected / row.wavelength_m) / carrier_sigma_cycles
        )
    for epoch, obs in ddpr.items():
        for index in range(obs.n):
            expected, jacobian = _dd_expected_and_jacobian_m(
                trajectory[epoch],
                obs.sat_ecef_k[index],
                obs.sat_ecef_ref[index],
                obs.base_range_k[index],
                obs.base_range_ref[index],
            )
            vector = np.zeros(3 + len(keys))
            scale = 1.0 if offset_scales is None else float(offset_scales[epoch])
            vector[:3] = scale * np.asarray(jacobian) / ddpr_sigma_m
            design.append(vector)
            target.append(
                (float(obs.dd_pseudorange_m[index]) - expected) / ddpr_sigma_m
            )
    for axis in range(3):
        vector = np.zeros(3 + len(keys))
        vector[axis] = 1.0 / position_prior_sigma_m
        design.append(vector)
        target.append(0.0)
    # The input is an OSM road trajectory.  Preserve its ellipsoidal height
    # weakly while allowing a broad horizontal correction.
    representative = np.median(np.asarray(list(trajectory.values())), axis=0)
    norm = np.linalg.norm(representative)
    local_up = representative / norm if norm > 1.0 else np.asarray([0.0, 0.0, 1.0])
    vector = np.zeros(3 + len(keys))
    vector[:3] = local_up / up_prior_sigma_m
    design.append(vector)
    target.append(float(up_prior_center_m) / up_prior_sigma_m)
    matrix = np.asarray(design)
    values = np.asarray(target)
    normal = matrix.T @ matrix
    covariance = np.linalg.inv(normal + np.eye(normal.shape[0]) * 1.0e-10)
    floating = covariance @ matrix.T @ values
    ahat = floating[3:]
    qahat = covariance[3:, 3:]
    candidates, residuals = integer_search(
        ahat, qahat, n_candidates=n_candidates, max_nodes=500_000
    )
    gain = covariance[:3, 3:] @ np.linalg.inv(qahat)
    offsets = [floating[:3] + gain @ (candidate - ahat) for candidate in candidates]
    ratio = (
        float(residuals[1] / residuals[0])
        if len(residuals) >= 2 and residuals[0] > 0
        else float("inf")
    )
    partial_diagnostics = []
    for subset in covariance_guided_partial_ar_subsets(
        qahat,
        minimum_ambiguities=4,
        max_drop_steps=partial_ar_max_drop_steps,
        worst_axes=partial_ar_worst_axes,
    ):
        sub_covariance = qahat[np.ix_(subset, subset)]
        sub_float = ahat[subset]
        sub_candidates, sub_residuals = integer_search(
            sub_float,
            sub_covariance,
            n_candidates=partial_ar_candidates_per_subset,
            max_nodes=500_000,
        )
        sub_gain = covariance[:3, 3:][:, subset] @ np.linalg.inv(sub_covariance)
        supplied = 0
        for candidate in sub_candidates:
            value = floating[:3] + sub_gain @ (candidate - sub_float)
            if all(np.linalg.norm(value - prior) > 0.05 for prior in offsets):
                offsets.append(value)
                supplied += 1
        sub_ratio = (
            float(sub_residuals[1] / sub_residuals[0])
            if len(sub_residuals) >= 2 and sub_residuals[0] > 0
            else float("inf")
        )
        dropped = sorted(set(range(len(keys))) - set(subset.tolist()))
        partial_diagnostics.append(
            {
                "drop_count": len(dropped),
                "retained_ambiguities": len(subset),
                "dropped_indices": dropped,
                "dropped_keys": [list(keys[index]) for index in dropped],
                "lambda_ratio": sub_ratio,
                "candidate_count": len(sub_candidates),
                "unique_position_seeds_supplied": supplied,
            }
        )
    return offsets, {
        "float_integer_arcs": len(keys),
        "float_offset_ecef_m": floating[:3].tolist(),
        "up_prior_center_m": float(up_prior_center_m),
        "up_prior_sigma_m": float(up_prior_sigma_m),
        "lambda_candidates": len(offsets),
        "lambda_ratio": ratio,
        "lambda_squared_residuals": residuals.tolist(),
        "partial_ar": {
            "method": "gnssplusplus_bsr_guided_covariance_axis_loading",
            "max_drop_steps": int(partial_ar_max_drop_steps),
            "worst_axes": int(partial_ar_worst_axes),
            "candidates_per_subset": int(partial_ar_candidates_per_subset),
            "subsets": partial_diagnostics,
        },
    }


def covariance_guided_partial_ar_subsets(
    covariance: np.ndarray,
    *,
    minimum_ambiguities: int = 4,
    max_drop_steps: int = 0,
    worst_axes: int = 3,
) -> list[np.ndarray]:
    """Mirror GNSS++ BSR-guided progressive ambiguity decimation."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return []
    if not np.all(np.isfinite(matrix)) or int(worst_axes) < 1:
        return []
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.size == 0 or float(np.min(eigenvalues)) < -1.0e-10:
        return []
    count = matrix.shape[0]
    axes = np.argsort(eigenvalues)[::-1][: min(int(worst_axes), count)]
    loading = np.sum(
        np.abs(eigenvectors[:, axes]) * np.maximum(eigenvalues[axes], 0.0), axis=1
    )
    current = list(range(count))
    output = []
    steps = min(max(0, count - int(minimum_ambiguities)), max(0, int(max_drop_steps)))
    for _step in range(steps):
        worst = max(current, key=lambda index: (float(loading[index]), -index))
        current = [index for index in current if index != worst]
        if len(current) < int(minimum_ambiguities):
            break
        output.append(np.asarray(current, dtype=np.int64))
    return output


def gsi_moving_up_prior(
    cache: dict[str, Any],
    trajectory: dict[int, np.ndarray],
    *,
    segment: tuple[int, int],
    max_antenna_height_spread_m: float = 0.5,
    max_target_up_spread_m: float = 0.5,
) -> dict[str, Any]:
    """Derive an Up-offset center from a frozen GSI cache and accepted anchors."""

    if cache.get("schema") != "wp50_gsi_moving_height_cache_v1":
        raise ValueError("unsupported moving GSI height cache")
    if bool(cache.get("production_input_truth", True)) or bool(
        cache.get("runtime_network_required", True)
    ):
        raise ValueError("moving GSI height cache is not production-safe")
    if [int(value) for value in cache.get("segment", [])] != list(segment):
        raise ValueError("moving GSI height cache segment mismatch")
    points = list(cache.get("calibration_points", []))
    target = dict(cache.get("target_point", {}))
    if len(points) < 2:
        raise ValueError("moving GSI height calibration has fewer than two anchors")
    sources = {str(point.get("dem_source")) for point in points}
    models = {str(point.get("geoid_model")) for point in points}
    if (
        len(sources) != 1
        or not next(iter(sources)).endswith("（レーザ）")
        or len(models) != 1
    ):
        raise ValueError("moving GSI height cache mixes unsupported height sources")
    offsets = []
    for point in points:
        ellipsoid_height = float(_ecef_to_lla_py(*point["antenna_position_ecef"])[2])
        ground_height = float(point["elevation_m"]) + float(point["geoid_height_m"])
        offsets.append(ellipsoid_height - ground_height)
    spread = float(np.ptp(np.asarray(offsets)))
    if spread > float(max_antenna_height_spread_m):
        raise ValueError("moving GSI antenna-height calibration spread is too large")
    calibrated_antenna_height = float(np.median(np.asarray(offsets)))
    target_points = list(cache.get("target_points", []))
    consensus: dict[str, Any] | None = None
    if target_points:
        source = next(iter(sources))
        model = next(iter(models))
        compatible = [
            point
            for point in target_points
            if str(point.get("dem_source")) == source
            and str(point.get("geoid_model")) == model
            and int(point.get("epoch", -1)) in trajectory
        ]
        if len(compatible) < 2:
            raise ValueError("moving GSI height cache has fewer than two compatible samples")
        sample_rows = []
        for point in compatible:
            epoch = int(point["epoch"])
            trajectory_height = float(_ecef_to_lla_py(*trajectory[epoch])[2])
            predicted_height = (
                float(point["elevation_m"])
                + float(point["geoid_height_m"])
                + calibrated_antenna_height
            )
            sample_rows.append(
                {
                    "epoch": epoch,
                    "predicted_height_m": predicted_height,
                    "trajectory_height_m": trajectory_height,
                    "up_center_m": predicted_height - trajectory_height,
                }
            )
        ordered = sorted(sample_rows, key=lambda row: float(row["up_center_m"]))
        clusters = []
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                cluster = ordered[left : right + 1]
                cluster_spread = float(
                    cluster[-1]["up_center_m"] - cluster[0]["up_center_m"]
                )
                if cluster_spread <= float(max_target_up_spread_m):
                    clusters.append(cluster)
                else:
                    break
        inliers = min(
            clusters,
            key=lambda cluster: (
                -len(cluster),
                float(cluster[-1]["up_center_m"] - cluster[0]["up_center_m"]),
                min(int(row["epoch"]) for row in cluster),
            ),
            default=[],
        )
        if len(inliers) < 2:
            raise ValueError("moving GSI height samples have no two-point consensus")
        inlier_centers = np.asarray([row["up_center_m"] for row in inliers])
        inlier_spread = float(np.ptp(inlier_centers))
        if inlier_spread > float(max_target_up_spread_m):
            raise ValueError("moving GSI height sample consensus spread is too large")
        up_prior_center = float(np.median(inlier_centers))
        predicted_height = float(np.median([row["predicted_height_m"] for row in inliers]))
        trajectory_height = float(
            np.median([row["trajectory_height_m"] for row in inliers])
        )
        consensus = {
            "method": "fixed_epoch_source_matched_densest_bounded_cluster",
            "total_samples": len(target_points),
            "compatible_samples": len(compatible),
            "inlier_samples": len(inliers),
            "inlier_spread_m": inlier_spread,
            "max_inlier_spread_m": float(max_target_up_spread_m),
            "inlier_epochs": [row["epoch"] for row in inliers],
        }
    else:
        target_sources = sources | {str(target.get("dem_source"))}
        target_models = models | {str(target.get("geoid_model"))}
        if len(target_sources) != 1 or len(target_models) != 1:
            raise ValueError("moving GSI height cache mixes unsupported height sources")
        predicted_height = (
            float(target["elevation_m"])
            + float(target["geoid_height_m"])
            + calibrated_antenna_height
        )
        representative = np.median(np.asarray(list(trajectory.values())), axis=0)
        trajectory_height = float(_ecef_to_lla_py(*representative)[2])
        up_prior_center = predicted_height - trajectory_height
    return {
        "source": "cached_gsi_dem_geoid_accepted_anchor_calibration",
        "dem_source": next(iter(sources)),
        "geoid_model": next(iter(models)),
        "antenna_height_offsets_m": offsets,
        "antenna_height_spread_m": spread,
        "max_antenna_height_spread_m": float(max_antenna_height_spread_m),
        "calibrated_antenna_height_m": calibrated_antenna_height,
        "predicted_antenna_ellipsoid_height_m": predicted_height,
        "trajectory_representative_ellipsoid_height_m": trajectory_height,
        "up_prior_center_m": up_prior_center,
        "target_sample_consensus": consensus,
    }


def block_stability(
    solution: dict[str, Any],
    trajectory: dict[int, np.ndarray],
    ddpr: dict[int, DDPseudorangeEpoch],
    start: int,
    end: int,
    blocks: int,
    offset_scales: dict[int, float] | None = None,
    up_prior: tuple[np.ndarray, float, float] | None = None,
) -> tuple[list[list[float]], float]:
    boundaries = np.linspace(start, end, blocks + 1, dtype=int)
    offsets = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        epochs = set(range(int(left), int(right)))
        offsets.append(
            optimize_fixed_integers(
                np.asarray(solution["offset_ecef_m"]),
                trajectory,
                solution["retained_rows"],
                solution["integers"],
                ddpr,
                epochs=epochs,
                offset_scales=offset_scales,
                up_prior=up_prior,
            )
        )
    center = np.asarray(solution["offset_ecef_m"])
    spread = max(float(np.linalg.norm(value - center)) for value in offsets)
    return [value.tolist() for value in offsets], spread


def _seed_offsets(
    route: dict[int, np.ndarray],
    comparison: dict[int, np.ndarray] | None,
    basin_path: Path | None,
    start: int,
    end: int,
    max_seeds: int,
) -> list[np.ndarray]:
    seeds = [np.zeros(3)]
    if comparison:
        shared = [epoch for epoch in route if epoch in comparison]
        if shared:
            seeds.append(np.median([comparison[e] - route[e] for e in shared], axis=0))
    if basin_path:
        offsets = []
        with basin_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                epoch = int(row["epoch"])
                if start <= epoch < end and epoch in route and epoch % 5 == 0:
                    position = np.asarray(
                        [
                            float(row["ecef_x"]),
                            float(row["ecef_y"]),
                            float(row["ecef_z"]),
                        ]
                    )
                    offsets.append(position - route[epoch])
        # Diverse seeds are more useful here than a dense cloud around one basin.
        for value in offsets:
            if all(np.linalg.norm(value - prior) > 1.0 for prior in seeds):
                seeds.append(value)
                if len(seeds) >= max_seeds:
                    break
    return seeds[:max_seeds]


def _external_seed_offsets(path: Path | None) -> list[np.ndarray]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        payload.get("seeds", payload.get("candidates", payload))
        if isinstance(payload, dict)
        else payload
    )
    output = []
    for row in rows:
        value = (
            row.get("offset_ecef_m", row.get("seed_offset_ecef_m"))
            if isinstance(row, dict)
            else row
        )
        if value is None:
            continue
        offset = np.asarray(value, dtype=np.float64)
        if offset.shape == (3,) and np.all(np.isfinite(offset)):
            output.append(offset)
    return output


def rank_road_translation_seeds(
    route_xy: np.ndarray,
    road: STRtree,
    *,
    radius_m: float = 40.0,
    coarse_step_m: float = 1.0,
    fine_step_m: float = 0.1,
    max_seeds: int = 24,
    dedup_m: float = 1.0,
    spatial_cell_m: float = 0.0,
) -> list[dict[str, float | list[float]]]:
    """Find diverse common XY translations that align a route shape to roads.

    ``spatial_cell_m`` enables corridor enumeration.  The best road match in
    every translation-space cell is retained and farthest-point ordering is
    used before fine refinement.  This prevents repeated parallel-road modes
    from consuming the whole local-refinement budget.  Zero preserves the
    original score-ordered proposal behavior.
    """
    route_xy = np.asarray(route_xy, dtype=np.float64)

    def metric(dx: float, dy: float) -> tuple[float, float, float]:
        distances = _road_distances(road, route_xy[:, 0] + dx, route_xy[:, 1] + dy)
        p95 = float(np.percentile(distances, 95.0))
        median = float(np.median(distances))
        return p95 + 0.05 * median, p95, median

    coarse = []
    values = np.arange(-radius_m, radius_m + 0.5 * coarse_step_m, coarse_step_m)
    for dx in values:
        for dy in values:
            score, p95, median = metric(float(dx), float(dy))
            coarse.append((score, float(dx), float(dy), p95, median))
    coarse.sort()
    centers = []
    if spatial_cell_m > 0.0:
        cell_best: dict[tuple[int, int], tuple[float, float, float, float, float]] = {}
        for row in coarse:
            cell = (
                int(np.floor((row[1] + radius_m) / spatial_cell_m)),
                int(np.floor((row[2] + radius_m) / spatial_cell_m)),
            )
            cell_best.setdefault(cell, row)
        # Enumerate plausible road corridors before spending parents on cells
        # whose best alignment is already farther than the frozen road gate.
        # This is a measurement-only ordering; no reference trajectory enters.
        preferred = [row for row in cell_best.values() if row[3] <= 1.0]
        deferred = [row for row in cell_best.values() if row[3] > 1.0]
        preferred.sort()
        deferred.sort()
        if not preferred:
            preferred, deferred = deferred, []
        remaining = preferred
        if remaining:
            centers.append(min(remaining))
            remaining.remove(centers[0])
        for pool in (remaining, deferred):
            while pool and len(centers) < max(8, max_seeds):
                chosen = min(
                    pool,
                    key=lambda row: (
                        -min(
                            np.hypot(row[1] - prior[1], row[2] - prior[2])
                            for prior in centers
                        ),
                        row[0],
                    ),
                )
                centers.append(chosen)
                pool.remove(chosen)
    else:
        for row in coarse:
            if all(
                np.hypot(row[1] - prior[1], row[2] - prior[2]) >= 2.0
                for prior in centers
            ):
                centers.append(row)
                if len(centers) >= max(8, max_seeds):
                    break
    fine = []
    refinements = np.arange(
        -coarse_step_m, coarse_step_m + 0.5 * fine_step_m, fine_step_m
    )
    for _score, center_x, center_y, _p95, _median in centers:
        best = None
        for delta_x in refinements:
            for delta_y in refinements:
                dx, dy = center_x + float(delta_x), center_y + float(delta_y)
                score, p95, median = metric(dx, dy)
                row = (score, dx, dy, p95, median)
                if best is None or row < best:
                    best = row
        fine.append(best)
    if spatial_cell_m <= 0.0:
        fine.sort()
    output = []
    for score, dx, dy, p95, median in fine:
        if all(
            np.hypot(dx - item["translation_xy_m"][0], dy - item["translation_xy_m"][1])
            >= dedup_m
            for item in output
        ):
            output.append(
                {
                    "translation_xy_m": [dx, dy],
                    "road_score": score,
                    "road_p95_m": p95,
                    "road_median_m": median,
                }
            )
            if len(output) >= max_seeds:
                break
    return output


def osm_road_seed_offsets(
    trajectory: dict[int, np.ndarray],
    cache_path: Path,
    *,
    radius_m: float,
    max_seeds: int,
    spatial_cell_m: float = 0.0,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    epsg = int(cache["epsg"])
    road = STRtree([LineString(row) for row in cache["projected_road_lines"]])
    to_map = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    to_ecef = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)
    positions = np.asarray([trajectory[epoch] for epoch in sorted(trajectory)])
    x, y, _z = to_map.transform(positions[:, 0], positions[:, 1], positions[:, 2])
    ranked = rank_road_translation_seeds(
        np.column_stack([x, y]),
        road,
        radius_m=radius_m,
        max_seeds=max_seeds,
        spatial_cell_m=spatial_cell_m,
    )
    if all(np.hypot(*row["translation_xy_m"]) > 0.5 for row in ranked):
        identity_distances = _road_distances(road, np.asarray(x), np.asarray(y))
        identity_p95 = float(np.percentile(identity_distances, 95.0))
        identity_median = float(np.median(identity_distances))
        ranked.append(
            {
                "translation_xy_m": [0.0, 0.0],
                "road_score": identity_p95 + 0.05 * identity_median,
                "road_p95_m": identity_p95,
                "road_median_m": identity_median,
                "mandatory_identity": True,
            }
        )
    representative = np.median(positions, axis=0)
    rx, ry, rh = to_map.transform(*representative)
    offsets = []
    for row in ranked:
        dx, dy = row["translation_xy_m"]
        shifted = np.asarray(to_ecef.transform(rx + dx, ry + dy, rh))
        offset = shifted - representative
        offsets.append(offset)
        row["offset_ecef_m"] = offset.tolist()
    return offsets, ranked


def shared_road_seed_offsets(
    trajectory: dict[int, np.ndarray],
    cache_path: Path,
    road_rows: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Reproject a truth-free parent corridor list onto another block."""
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    epsg = int(cache["epsg"])
    to_map = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    to_ecef = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)
    representative = np.median(np.asarray(list(trajectory.values())), axis=0)
    rx, ry, rh = to_map.transform(*representative)
    diagnostics = []
    offsets = []
    for source in road_rows:
        row = {key: value for key, value in source.items() if key != "offset_ecef_m"}
        dx, dy = row["translation_xy_m"]
        offset = np.asarray(to_ecef.transform(rx + dx, ry + dy, rh)) - representative
        row["offset_ecef_m"] = offset.tolist()
        diagnostics.append(row)
        offsets.append(offset)
    return offsets, diagnostics


def locally_refine_road_seeds(
    trajectory: dict[int, np.ndarray],
    road_rows: list[dict[str, Any]],
    cache_path: Path,
    carrier_rows: list[CarrierRow],
    ddpr: dict[int, DDPseudorangeEpoch],
    *,
    parent_count: int = 8,
    radius_m: float = 1.5,
    step_m: float = 0.25,
    height_offsets_m: tuple[float, ...] = (-0.5, 0.0, 0.5),
    seeds_per_parent: int = 4,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Search integer-coherent modes near each OSM route-alignment valley."""
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    epsg = int(cache["epsg"])
    to_map = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    to_ecef = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)
    representative = np.median(np.asarray(list(trajectory.values())), axis=0)
    rx, ry, rh = to_map.transform(*representative)
    output_offsets: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    audit_pool: list[dict[str, Any]] = []
    parent_rows = list(road_rows[:parent_count])
    identity = next((row for row in road_rows if row.get("mandatory_identity")), None)
    if identity is not None and identity not in parent_rows:
        parent_rows[-1:] = [identity]
    for parent_id, parent in enumerate(parent_rows):
        base_x, base_y = parent["translation_xy_m"]
        parent_radius = (
            max(radius_m, 3.0) if parent.get("mandatory_identity") else radius_m
        )
        parent_deltas = np.arange(-parent_radius, parent_radius + 0.5 * step_m, step_m)
        candidates = []
        for dx in parent_deltas:
            for dy in parent_deltas:
                for dh in height_offsets_m:
                    shifted = np.asarray(
                        to_ecef.transform(rx + base_x + dx, ry + base_y + dy, rh + dh)
                    )
                    offset = shifted - representative
                    integers, retained = estimate_arc_integers(
                        offset, trajectory, carrier_rows
                    )
                    if not retained:
                        continue
                    cp, pr = _raw_residuals(
                        offset, trajectory, retained, integers, ddpr
                    )
                    cp_rms = float(np.sqrt(np.mean(np.square(cp))))
                    pr_rms = (
                        float(np.sqrt(np.mean(np.square(pr))))
                        if len(pr)
                        else float("inf")
                    )
                    retained_fraction = len(retained) / max(len(carrier_rows), 1)
                    score = cp_rms + 0.25 * (1.0 - retained_fraction) + 0.002 * pr_rms
                    candidates.append(
                        (
                            score,
                            offset,
                            dx,
                            dy,
                            dh,
                            cp_rms,
                            pr_rms,
                            len(integers),
                            len(retained),
                        )
                    )
                    integer_signature = {
                        "|".join(
                            (
                                re.sub(r"@arc\d+$", "", str(key[0])),
                                re.sub(r"@arc\d+$", "", str(key[1])),
                                str(key[2]),
                            )
                        ): int(value)
                        for key, value in integers.items()
                    }
                    audit_pool.append(
                        {
                            "parent_road_seed": parent_id,
                            "proposal_score": score,
                            "parent_translation_xy_m": [base_x, base_y],
                            "local_delta_xyh_m": [float(dx), float(dy), float(dh)],
                            "map_translation_xyh_m": [
                                float(base_x + dx),
                                float(base_y + dy),
                                float(dh),
                            ],
                            "offset_ecef_m": offset.tolist(),
                            "carrier_rms_cycles": cp_rms,
                            "ddpr_rms_m": pr_rms,
                            "integer_arcs": len(integers),
                            "retained_carrier_rows": len(retained),
                            "integer_signature": integer_signature,
                        }
                    )
        candidates.sort(key=lambda row: row[0])
        chosen = []
        if seeds_per_parent > 0:
            for row in candidates:
                if all(np.linalg.norm(row[1] - prior[1]) >= 0.2 for prior in chosen):
                    chosen.append(row)
                    if len(chosen) >= seeds_per_parent:
                        break
        for rank, row in enumerate(chosen):
            output_offsets.append(row[1])
            diagnostics.append(
                {
                    "parent_road_seed": parent_id,
                    "local_rank": rank,
                    "local_delta_xyh_m": [row[2], row[3], row[4]],
                    "offset_ecef_m": row[1].tolist(),
                    "proposal_score": row[0],
                    "carrier_rms_cycles": row[5],
                    "ddpr_rms_m": row[6],
                    "integer_arcs": row[7],
                    "retained_carrier_rows": row[8],
                }
            )
    return output_offsets, diagnostics, audit_pool


def run(args: argparse.Namespace) -> dict[str, Any]:
    excluded_ddpr_satellites = frozenset(
        value.strip()
        for value in args.exclude_ddpr_satellites.split(",")
        if value.strip()
    )
    route = _read_trajectory(args.trajectory, args.start, args.end)
    offset_scales: dict[int, float] | None = None
    right_boundary_anchor = None
    right_boundary_profile = None
    if (
        args.right_boundary_anchor is not None
        and args.right_boundary_profile is not None
    ):
        raise ValueError("right boundary anchor modes are mutually exclusive")
    if args.right_boundary_anchor is not None:
        right_boundary_anchor = _load_static_position_override(
            args.right_boundary_anchor
        )
        boundary_epoch = int(right_boundary_anchor[0])
        if not args.start < args.end <= boundary_epoch:
            raise ValueError("right boundary anchor does not follow the moving block")
        denominator = float(boundary_epoch - args.start)
        offset_scales = {
            epoch: float(boundary_epoch - epoch) / denominator for epoch in route
        }
        if args.osm_cache is not None or args.road_parent_artifact is not None:
            raise ValueError(
                "road seed modes are unsupported for affine boundary state"
            )
    elif args.right_boundary_profile is not None:
        right_boundary_profile = _load_right_boundary_profile(
            args.right_boundary_profile, args.end
        )
        route, offset_scales = fixed_boundary_affine_route(
            route,
            start=args.start,
            boundary_epoch=right_boundary_profile["epoch"],
            boundary_offset_ecef_m=right_boundary_profile["offset_ecef_m"],
        )
        if args.osm_cache is not None or args.road_parent_artifact is not None:
            raise ValueError(
                "road seed modes are unsupported for affine boundary state"
            )
    comparison = (
        _read_trajectory(args.comparison_trajectory, args.start, args.end)
        if args.comparison_trajectory
        else None
    )
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=args.end,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    cache = RinexObservationCache()
    systems = ("G", "E", "J", "C")
    cp_engine = DDCarrierComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"]),
        allowed_systems=systems,
        observation_cache=cache,
    )
    pr_engine = DDPseudorangeComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"]),
        allowed_systems=systems,
        observation_cache=cache,
    )
    pair_bias_samples: dict[tuple[str, str], list[tuple[float, int]]] = {}
    pair_biases: dict[tuple[str, str], tuple[float, int]] = {}
    bias_anchor = None
    if args.ddpr_bias_anchor is not None:
        bias_anchor = _load_static_position_override(args.ddpr_bias_anchor)
    elif args.ddpr_bias_static_anchor is not None:
        bias_anchor = _load_fusion_static_override(*args.ddpr_bias_static_anchor)
    bias_updates = 0
    if bias_anchor is not None:
        anchor_start, anchor_end, anchor_position, _candidate_id, _reason = bias_anchor
        if anchor_end > args.start:
            raise RuntimeError(
                "DDPR bias anchor must end no later than the moving block"
            )
        calibration_start = max(anchor_start, anchor_end - args.ddpr_bias_tail_epochs)
        for epoch in range(calibration_start, anchor_end, args.ddpr_bias_stride):
            measurements = _build_dd_measurements(
                np.asarray(data["sat_ecef"][epoch]),
                np.asarray(data["system_ids"][epoch]),
                list(data["used_prns"][epoch]),
                np.asarray(data["weights"][epoch]),
                anchor_position,
                systems,
                min_elevation_deg=-90.0,
                min_snr=0.0,
                keep_best=0,
            )
            result = pr_engine.compute_dd(
                float(data["times"][epoch]),
                measurements,
                rover_position_approx=anchor_position,
                min_common_sats=4,
            )
            if result is None:
                continue
            obs = DDPseudorangeEpoch.from_result(result)
            residuals = []
            for index in range(obs.n):
                expected, _ = _dd_expected_and_jacobian_m(
                    anchor_position,
                    obs.sat_ecef_k[index],
                    obs.sat_ecef_ref[index],
                    obs.base_range_k[index],
                    obs.base_range_ref[index],
                )
                residuals.append(float(obs.dd_pseudorange_m[index]) - expected)
            for residual, ref_sat, sat_id in zip(
                residuals, obs.ref_sat_ids or (), obs.sat_ids or ()
            ):
                pair_bias_samples.setdefault((ref_sat, sat_id), []).append(
                    (float(residual), epoch)
                )
            bias_updates += 1
        pair_biases = {
            key: (
                float(np.median([value for value, _epoch in samples])),
                max(epoch for _value, epoch in samples),
            )
            for key, samples in pair_bias_samples.items()
        }
    families = tuple(value for value in args.carrier_families.split(",") if value)
    if args.stride_phase == "auto":
        phases = list(range(args.stride))
    elif args.stride_phase == "start":
        phases = [args.start % args.stride]
    else:
        phase = int(args.stride_phase)
        if not 0 <= phase < args.stride:
            raise ValueError("explicit evidence stride phase is outside the stride")
        phases = [phase]
    phase_payloads: dict[
        int, tuple[list[CarrierRow], dict[int, DDPseudorangeEpoch], list[int]]
    ] = {}
    phase_diagnostics: list[dict[str, int]] = []
    for phase in phases:
        phase_rows: list[CarrierRow] = []
        phase_ddpr: dict[int, DDPseudorangeEpoch] = {}
        phase_evidence: list[int] = []
        for epoch in phase_epochs(args.start, args.end, args.stride, phase):
            approximate = route.get(epoch)
            if approximate is None:
                continue
            measurements = _build_dd_measurements(
                np.asarray(data["sat_ecef"][epoch]),
                np.asarray(data["system_ids"][epoch]),
                list(data["used_prns"][epoch]),
                np.asarray(data["weights"][epoch]),
                approximate,
                systems,
                min_elevation_deg=-90.0,
                min_snr=0.0,
                keep_best=0,
            )
            cp = cp_engine.compute_dd_families(
                float(data["times"][epoch]),
                measurements,
                rover_position_approx=approximate,
                min_common_sats=2,
                carrier_families=families,
                reference_rank=args.carrier_reference_rank,
            )
            pr = pr_engine.compute_dd(
                float(data["times"][epoch]),
                measurements,
                rover_position_approx=approximate,
                min_common_sats=4,
            )
            if cp is not None:
                phase_rows.extend(_carrier_rows(epoch, DDCarrierEpoch.from_result(cp)))
            if pr is not None:
                obs = filter_ddpr_excluded_satellites(
                    DDPseudorangeEpoch.from_result(pr), excluded_ddpr_satellites
                )
            else:
                obs = None
            if obs is not None:
                if bias_anchor is not None:
                    corrected = bias_correct_ddpr_epoch(
                        obs,
                        pair_biases,
                        epoch=epoch,
                        max_age_epochs=args.ddpr_bias_max_age_epochs,
                    )
                    if corrected is not None:
                        phase_ddpr[epoch] = corrected
                else:
                    phase_ddpr[epoch] = obs
            if cp is not None or pr is not None:
                phase_evidence.append(epoch)
        phase_payloads[phase] = (phase_rows, phase_ddpr, phase_evidence)
        phase_diagnostics.append(
            {
                "phase": phase,
                "evidence_epochs": len(set(phase_evidence)),
                "raw_carrier_rows": len(phase_rows),
                "ddpr_epochs": len(phase_ddpr),
            }
        )
    selected_phase = choose_evidence_phase(phase_diagnostics)
    all_rows, ddpr, evidence_epochs = phase_payloads[selected_phase]
    raw_carrier_row_count = len(all_rows)
    all_rows, carrier_arc_splits = segment_carrier_arcs(
        route,
        all_rows,
        max_epoch_gap=args.stride * 2,
        max_float_jump_cycles=args.max_float_jump_cycles,
    )
    seeds = _seed_offsets(
        route, comparison, args.basin_trace, args.start, args.end, args.max_seeds
    )
    for value in _external_seed_offsets(args.external_seeds):
        if all(np.linalg.norm(value - prior) > 0.05 for prior in seeds):
            seeds.append(value)
    gsi_height_prior = None
    up_prior_center_m = 0.0
    final_up_prior: tuple[np.ndarray, float, float] | None = None
    if args.gsi_height_cache is not None:
        cache_bytes = args.gsi_height_cache.read_bytes()
        cache_payload = json.loads(cache_bytes.decode("utf-8"))
        gsi_height_prior = gsi_moving_up_prior(
            cache_payload,
            route,
            segment=(args.start, args.end),
            max_antenna_height_spread_m=args.max_gsi_antenna_height_spread_m,
        )
        gsi_height_prior["cache_sha256"] = hashlib.sha256(cache_bytes).hexdigest()
        up_prior_center_m = float(gsi_height_prior["up_prior_center_m"])
        if offset_scales is not None:
            representative_scale = float(
                np.median(np.asarray(list(offset_scales.values())))
            )
            if representative_scale <= 0.0:
                raise ValueError(
                    "affine boundary scale is not positive in moving block"
                )
            up_prior_center_m /= representative_scale
            gsi_height_prior["affine_reference_up_prior_center_m"] = up_prior_center_m
        if args.enforce_final_up_prior:
            representative = np.median(np.asarray(list(route.values())), axis=0)
            local_up = representative / np.linalg.norm(representative)
            final_up_prior = (
                local_up,
                up_prior_center_m,
                float(args.up_prior_sigma_m),
            )
    lambda_seeds, lambda_diagnostics = float_ambiguity_seeds(
        route,
        all_rows,
        ddpr,
        min_arc_epochs=args.min_arc_epochs,
        n_candidates=args.lambda_candidates,
        up_prior_sigma_m=args.up_prior_sigma_m,
        up_prior_center_m=up_prior_center_m,
        partial_ar_max_drop_steps=args.partial_ar_max_drop_steps,
        partial_ar_worst_axes=args.partial_ar_worst_axes,
        partial_ar_candidates_per_subset=args.partial_ar_candidates_per_subset,
        offset_scales=offset_scales,
    )
    for value in lambda_seeds:
        if all(np.linalg.norm(value - prior) > 0.05 for prior in seeds):
            seeds.append(value)
    road_diagnostics: list[dict[str, Any]] = []
    road_local_diagnostics: list[dict[str, Any]] = []
    road_local_audit_pool: list[dict[str, Any]] = []
    if args.osm_cache:
        if args.road_parent_artifact:
            parent_payload = json.loads(
                args.road_parent_artifact.read_text(encoding="utf-8")
            )
            road_seeds, road_diagnostics = shared_road_seed_offsets(
                route, args.osm_cache, parent_payload["osm_road_seed_diagnostics"]
            )
        else:
            road_seeds, road_diagnostics = osm_road_seed_offsets(
                route,
                args.osm_cache,
                radius_m=args.road_search_radius_m,
                max_seeds=args.road_seed_count,
                spatial_cell_m=args.road_spatial_cell_m,
            )
        for value in road_seeds:
            if all(np.linalg.norm(value - prior) > 0.05 for prior in seeds):
                seeds.append(value)
        if args.road_local_refine:
            local_seeds, road_local_diagnostics, road_local_audit_pool = (
                locally_refine_road_seeds(
                    route,
                    road_diagnostics,
                    args.osm_cache,
                    all_rows,
                    ddpr,
                    parent_count=args.road_local_parent_count,
                    radius_m=args.road_local_radius_m,
                    step_m=args.road_local_step_m,
                    height_offsets_m=tuple(
                        float(value)
                        for value in args.road_local_height_offsets_m.split(",")
                    ),
                    seeds_per_parent=args.road_local_seeds_per_parent,
                )
            )
            for value in local_seeds:
                if all(np.linalg.norm(value - prior) > 0.05 for prior in seeds):
                    seeds.append(value)
    hypotheses = []
    truth = np.asarray(data["ground_truth"])
    for seed_id, seed in enumerate(seeds):
        solution = solve_hypothesis(
            seed,
            route,
            all_rows,
            ddpr,
            min_arc_epochs=args.min_arc_epochs,
            offset_scales=offset_scales,
            up_prior=final_up_prior,
        )
        if solution is None:
            continue
        block_offsets, spread = block_stability(
            solution,
            route,
            ddpr,
            args.start,
            args.end,
            args.bootstrap_blocks,
            offset_scales=offset_scales,
            up_prior=final_up_prior,
        )
        errors = [
            np.linalg.norm(
                route[e]
                + (1.0 if offset_scales is None else offset_scales[e])
                * solution["offset_ecef_m"]
                - truth[e]
            )
            for e in route
        ]
        hypotheses.append(
            {
                "seed_id": seed_id,
                "seed_offset_ecef_m": seed.tolist(),
                "offset_ecef_m": solution["offset_ecef_m"].tolist(),
                "offset_at_segment_end_ecef_m": (
                    (1.0 if offset_scales is None else offset_scales[args.end - 1])
                    * solution["offset_ecef_m"]
                ).tolist(),
                "integer_arcs": solution["integer_arcs"],
                "carrier_rows": solution["carrier_rows"],
                "ddpr_rows": solution["ddpr_rows"],
                "carrier_rms_cycles": solution["carrier_rms_cycles"],
                "ddpr_rms_m": solution["ddpr_rms_m"],
                "cp_pr_consistency": solution["cp_pr_consistency"],
                "block_offsets_ecef_m": block_offsets,
                "block_spread_m": spread,
                "audit_median_error_m": float(np.median(errors)),
                "audit_sub50cm_epochs": int(np.count_nonzero(np.asarray(errors) < 0.5)),
            }
        )
    eligible = [
        row
        for row in hypotheses
        if row["integer_arcs"] >= args.min_integer_arcs
        and row["carrier_rows"] >= args.min_carrier_rows
        and row["ddpr_rows"] >= args.min_ddpr_rows
        and row["carrier_rms_cycles"] <= args.max_carrier_rms_cycles
        and row["ddpr_rms_m"] <= args.max_ddpr_rms_m
        and row["block_spread_m"] <= args.max_block_spread_m
    ]
    eligible.sort(key=lambda row: (row["carrier_rms_cycles"], row["ddpr_rms_m"]))
    selected = eligible[0] if eligible else None
    road_local_supply_audit = None
    if road_local_audit_pool:
        for row in road_local_audit_pool:
            offset = np.asarray(row["offset_ecef_m"])
            errors = [
                np.linalg.norm(route[epoch] + offset - truth[epoch]) for epoch in route
            ]
            row["audit_median_error_m"] = float(np.median(errors))
            row["audit_sub50cm_epochs"] = int(
                np.count_nonzero(np.asarray(errors) < 0.5)
            )
        ranked_pool = sorted(
            road_local_audit_pool, key=lambda row: row["proposal_score"]
        )
        best_audit = min(
            road_local_audit_pool, key=lambda row: row["audit_median_error_m"]
        )
        best_rank = next(
            index for index, row in enumerate(ranked_pool, start=1) if row is best_audit
        )
        parent_pool = sorted(
            (
                row
                for row in road_local_audit_pool
                if row["parent_road_seed"] == best_audit["parent_road_seed"]
            ),
            key=lambda row: row["proposal_score"],
        )
        best_parent_rank = next(
            index for index, row in enumerate(parent_pool, start=1) if row is best_audit
        )
        road_local_supply_audit = {
            "selection_eligible": False,
            "pool_candidates": len(road_local_audit_pool),
            "best_audit_median_error_m": best_audit["audit_median_error_m"],
            "best_audit_sub50cm_epochs": best_audit["audit_sub50cm_epochs"],
            "best_audit_proposal_rank": best_rank,
            "best_audit_parent_proposal_rank": best_parent_rank,
            "best_audit_parent_road_seed": best_audit["parent_road_seed"],
            "best_audit_offset_ecef_m": best_audit["offset_ecef_m"],
            "best_audit_proposal_score": best_audit["proposal_score"],
            "selected_local_hypotheses_supplied": int(
                any(row["audit_median_error_m"] < 0.5 for row in hypotheses)
            ),
        }
    # Diagnostic ceiling only: this branch is evaluated after production
    # hypotheses are frozen and can never enter ``eligible`` or selection.
    if offset_scales is None:
        oracle_offset = np.median(
            [truth[epoch] - route[epoch] for epoch in sorted(route)], axis=0
        )
    else:
        oracle_offset = np.median(
            [
                (truth[epoch] - route[epoch]) / offset_scales[epoch]
                for epoch in sorted(route)
                if offset_scales[epoch] >= 0.1
            ],
            axis=0,
        )
    oracle_solution = solve_hypothesis(
        oracle_offset,
        route,
        all_rows,
        ddpr,
        min_arc_epochs=args.min_arc_epochs,
        offset_scales=offset_scales,
        up_prior=final_up_prior,
    )
    oracle_audit: dict[str, Any] | None = None
    if oracle_solution is not None:
        oracle_blocks, oracle_spread = block_stability(
            oracle_solution,
            route,
            ddpr,
            args.start,
            args.end,
            args.bootstrap_blocks,
            offset_scales=offset_scales,
            up_prior=final_up_prior,
        )
        oracle_errors = [
            np.linalg.norm(
                route[epoch]
                + (1.0 if offset_scales is None else offset_scales[epoch])
                * oracle_solution["offset_ecef_m"]
                - truth[epoch]
            )
            for epoch in route
        ]
        oracle_audit = {
            "selection_eligible": False,
            "truth_derived_seed_ecef_m": oracle_offset.tolist(),
            "fitted_offset_ecef_m": oracle_solution["offset_ecef_m"].tolist(),
            "integer_arcs": oracle_solution["integer_arcs"],
            "carrier_rows": oracle_solution["carrier_rows"],
            "ddpr_rows": oracle_solution["ddpr_rows"],
            "carrier_rms_cycles": oracle_solution["carrier_rms_cycles"],
            "ddpr_rms_m": oracle_solution["ddpr_rms_m"],
            "block_offsets_ecef_m": oracle_blocks,
            "block_spread_m": oracle_spread,
            "audit_median_error_m": float(np.median(oracle_errors)),
            "audit_sub50cm_epochs": int(
                np.count_nonzero(np.asarray(oracle_errors) < 0.5)
            ),
        }
    result = {
        "schema": "wp31_moving_block_ambiguity_v1",
        "production_input_truth": False,
        "truth_usage": "post_selection_audit_only",
        "segment": [args.start, args.end],
        "offset_model": (
            {"mode": "constant"}
            if offset_scales is None
            else (
                {
                    "mode": "right_boundary_affine_fixed",
                    "reference_epoch": args.start,
                    "boundary_epoch": right_boundary_profile["epoch"],
                    "boundary_offset_ecef_m": right_boundary_profile[
                        "offset_ecef_m"
                    ].tolist(),
                    "boundary_profile_reason": right_boundary_profile["reason"],
                    "boundary_profile_source": str(args.right_boundary_profile),
                    "boundary_profile_sha256": right_boundary_profile["sha256"],
                    "segment_start_scale": offset_scales[args.start],
                    "segment_end_scale": offset_scales[args.end - 1],
                }
                if right_boundary_profile is not None
                else {
                    "mode": "right_boundary_affine_zero",
                    "reference_epoch": args.start,
                    "boundary_epoch": int(right_boundary_anchor[0]),
                    "boundary_anchor_reason": right_boundary_anchor[4],
                    "boundary_anchor_source": str(args.right_boundary_anchor),
                    "boundary_anchor_sha256": hashlib.sha256(
                        args.right_boundary_anchor.read_bytes()
                    ).hexdigest(),
                    "segment_start_scale": offset_scales[args.start],
                    "segment_end_scale": offset_scales[args.end - 1],
                }
            )
        ),
        "stride": args.stride,
        "stride_phase_mode": args.stride_phase,
        "selected_stride_phase": selected_phase,
        "stride_phase_diagnostics": phase_diagnostics,
        "evidence_epochs": len(set(evidence_epochs)),
        "ddpr_bias_calibration": {
            "enabled": bias_anchor is not None,
            "truth_free_static_anchor": bias_anchor[4]
            if bias_anchor is not None
            else None,
            "anchor_segment": list(bias_anchor[:2])
            if bias_anchor is not None
            else None,
            "calibration_updates": bias_updates,
            "exact_pair_biases_learned": len(pair_biases),
            "max_age_epochs": args.ddpr_bias_max_age_epochs,
        },
        "raw_carrier_rows": raw_carrier_row_count,
        "ddpr_excluded_satellites": sorted(excluded_ddpr_satellites),
        "carrier_families": list(families),
        "carrier_reference_rank": int(args.carrier_reference_rank),
        "carrier_arc_splits": carrier_arc_splits,
        "segmented_carrier_rows": len(all_rows),
        "ddpr_epochs": len(ddpr),
        "float_ambiguity_diagnostics": lambda_diagnostics,
        "gsi_height_prior": gsi_height_prior,
        "final_up_prior_enforced": final_up_prior is not None,
        "osm_road_seed_diagnostics": road_diagnostics,
        "osm_road_local_diagnostics": road_local_diagnostics,
        "gate": {
            "min_integer_arcs": args.min_integer_arcs,
            "min_carrier_rows": args.min_carrier_rows,
            "min_ddpr_rows": args.min_ddpr_rows,
            "max_carrier_rms_cycles": args.max_carrier_rms_cycles,
            "max_ddpr_rms_m": args.max_ddpr_rms_m,
            "max_block_spread_m": args.max_block_spread_m,
        },
        "selected_seed_id": selected["seed_id"] if selected else None,
        "selection_reason": "all_truth_free_gates_pass"
        if selected
        else "no_hypothesis_passed_truth_free_gates",
        "truth_seeded_oracle_diagnostic": oracle_audit,
        "osm_road_local_supply_audit": road_local_supply_audit,
        "hypotheses": hypotheses,
    }
    if args.road_local_pool_output:
        result["_truth_free_road_local_pool"] = [
            {key: value for key, value in row.items() if not key.startswith("audit_")}
            for row in road_local_audit_pool
        ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--comparison-trajectory", type=Path)
    parser.add_argument("--basin-trace", type=Path)
    parser.add_argument("--external-seeds", type=Path)
    parser.add_argument(
        "--right-boundary-anchor",
        type=Path,
        help="accepted static anchor whose start epoch fixes affine correction to zero",
    )
    parser.add_argument(
        "--right-boundary-profile",
        type=Path,
        help="promoted affine profile whose start fixes correction at this block end",
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--stride-phase", default="start")
    parser.add_argument("--max-seeds", type=int, default=24)
    parser.add_argument("--lambda-candidates", type=int, default=12)
    parser.add_argument("--partial-ar-max-drop-steps", type=int, default=0)
    parser.add_argument("--partial-ar-worst-axes", type=int, default=3)
    parser.add_argument("--partial-ar-candidates-per-subset", type=int, default=2)
    parser.add_argument("--up-prior-sigma-m", type=float, default=2.0)
    parser.add_argument("--gsi-height-cache", type=Path)
    parser.add_argument("--enforce-final-up-prior", action="store_true")
    parser.add_argument("--max-gsi-antenna-height-spread-m", type=float, default=0.5)
    parser.add_argument("--osm-cache", type=Path)
    parser.add_argument("--road-search-radius-m", type=float, default=40.0)
    parser.add_argument("--road-seed-count", type=int, default=24)
    parser.add_argument("--road-spatial-cell-m", type=float, default=0.0)
    parser.add_argument("--road-parent-artifact", type=Path)
    parser.add_argument("--road-local-refine", action="store_true")
    parser.add_argument("--road-local-parent-count", type=int, default=8)
    parser.add_argument("--road-local-radius-m", type=float, default=1.5)
    parser.add_argument("--road-local-step-m", type=float, default=0.25)
    parser.add_argument("--road-local-height-offsets-m", default="-0.5,0,0.5")
    parser.add_argument("--road-local-pool-output", type=Path)
    parser.add_argument("--road-local-seeds-per-parent", type=int, default=4)
    parser.add_argument("--carrier-families", default="L1_E1_B1,L5_E5A_B2A")
    parser.add_argument("--carrier-reference-rank", type=int, default=0)
    parser.add_argument(
        "--exclude-ddpr-satellites",
        default="",
        help="comma-separated satellite ids (e.g. G07,C39) to drop from DDPR "
        "pairs before gates/fitting; default empty keeps current behavior",
    )
    parser.add_argument("--ddpr-bias-anchor", type=Path)
    parser.add_argument("--ddpr-bias-static-anchor", type=Path, nargs=2)
    parser.add_argument("--ddpr-bias-tail-epochs", type=int, default=100)
    parser.add_argument("--ddpr-bias-stride", type=int, default=5)
    parser.add_argument("--ddpr-bias-max-age-epochs", type=int, default=100)
    parser.add_argument("--max-float-jump-cycles", type=float, default=0.75)
    parser.add_argument("--min-arc-epochs", type=int, default=3)
    parser.add_argument("--bootstrap-blocks", type=int, default=4)
    parser.add_argument("--min-integer-arcs", type=int, default=4)
    parser.add_argument("--min-carrier-rows", type=int, default=24)
    parser.add_argument("--min-ddpr-rows", type=int, default=40)
    parser.add_argument("--max-carrier-rms-cycles", type=float, default=0.5)
    parser.add_argument("--max-ddpr-rms-m", type=float, default=4.0)
    parser.add_argument("--max-block-spread-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ddpr_bias_anchor is not None and args.ddpr_bias_static_anchor is not None:
        parser.error("provide at most one DDPR bias anchor form")
    if args.enforce_final_up_prior and args.gsi_height_cache is None:
        parser.error("final Up prior enforcement requires --gsi-height-cache")
    result = run(args)
    pool = result.pop("_truth_free_road_local_pool", None)
    if args.road_local_pool_output:
        args.road_local_pool_output.parent.mkdir(parents=True, exist_ok=True)
        args.road_local_pool_output.write_text(
            json.dumps(
                {
                    "schema": "wp31_moving_block_truth_free_local_pool_v1",
                    "production_input_truth": False,
                    "segment": [args.start, args.end],
                    "candidates": pool or [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
