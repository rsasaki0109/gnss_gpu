"""Pivot-invariant robust DD pseudorange integrity scores."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from gnss_gpu.dd_quality import dd_pseudorange_residuals_m


@dataclass(frozen=True)
class MultiPivotDDPRResult:
    scores: np.ndarray
    probabilities: np.ndarray
    n_constellations: int
    n_satellites: int
    best_index: int


@dataclass(frozen=True)
class SatellitePairCostResult:
    satellite_ids: tuple[str, ...]
    mean_pair_costs: np.ndarray
    pair_counts: np.ndarray


def _system_id(satellite_id: str) -> str:
    value = str(satellite_id)
    return value[0] if value else "?"


def multipivot_ddpr_scores(
    dd_result,
    candidate_positions_ecef: np.ndarray,
    *,
    scale_m: float = 5.0,
    trim_largest_pairs: int = 1,
    temperature: float = 1.0,
    excluded_satellites: Iterable[str] = (),
) -> MultiPivotDDPRResult:
    """Score candidates using all constellation-local satellite pair residuals.

    A conventional DD result provides innovations relative to one reference.
    Adding a common constant to those reconstructed single-difference
    innovations changes no pairwise difference, so the score is invariant to
    the reference originally selected by the DD computer.
    """

    positions = np.asarray(candidate_positions_ecef, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("candidate_positions_ecef must be finite with shape (n, 3)")
    if positions.shape[0] == 0:
        raise ValueError("at least one candidate is required")
    scale = float(scale_m)
    temp = float(temperature)
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(temp) or temp <= 0.0:
        raise ValueError("scale_m and temperature must be finite and positive")
    trim = int(trim_largest_pairs)
    if trim < 0:
        raise ValueError("trim_largest_pairs must be non-negative")
    excluded = {str(value) for value in excluded_satellites}

    residuals = np.asarray(
        [dd_pseudorange_residuals_m(dd_result, position) for position in positions],
        dtype=np.float64,
    )
    ref_ids = tuple(str(value) for value in dd_result.ref_sat_ids)
    sat_ids = tuple(str(value) for value in dd_result.sat_ids)
    if residuals.shape[1] != len(ref_ids) or len(ref_ids) != len(sat_ids):
        raise ValueError("DD result satellite IDs do not match residual rows")

    score_sum = np.zeros(len(positions), dtype=np.float64)
    total_pairs = 0
    constellation_count = 0
    unique_satellites: set[str] = set()
    for system in sorted({_system_id(value) for value in ref_ids + sat_ids}):
        row_indices = [
            index for index, (ref, sat) in enumerate(zip(ref_ids, sat_ids))
            if _system_id(ref) == system and _system_id(sat) == system
        ]
        if not row_indices:
            continue
        reference_values = sorted({ref_ids[index] for index in row_indices})
        if len(reference_values) != 1:
            raise ValueError("each constellation must use one reference in a DD result")
        reference = reference_values[0]
        satellites = [reference] + [sat_ids[index] for index in row_indices]
        if len(set(satellites)) < 2:
            continue
        innovations = np.zeros((len(positions), len(satellites)), dtype=np.float64)
        innovations[:, 1:] = residuals[:, row_indices]
        retained = [
            index for index, satellite in enumerate(satellites)
            if satellite not in excluded
        ]
        satellites = [satellites[index] for index in retained]
        innovations = innovations[:, retained]
        if len(set(satellites)) < 2:
            continue
        pair_costs: list[np.ndarray] = []
        for left in range(len(satellites)):
            for right in range(left + 1, len(satellites)):
                difference = innovations[:, left] - innovations[:, right]
                pair_costs.append(np.log1p((difference / scale) ** 2))
        costs = np.stack(pair_costs, axis=1)
        keep = max(1, costs.shape[1] - min(trim, costs.shape[1] - 1))
        robust_cost = np.mean(np.sort(costs, axis=1)[:, :keep], axis=1)
        score_sum -= robust_cost * keep
        total_pairs += keep
        constellation_count += 1
        unique_satellites.update(satellites)

    if total_pairs == 0:
        raise ValueError("DD result has insufficient same-constellation support")
    scores = score_sum / float(total_pairs)
    shifted = (scores - float(np.max(scores))) / temp
    probabilities = np.exp(np.clip(shifted, -745.0, 0.0))
    probabilities /= float(np.sum(probabilities))
    return MultiPivotDDPRResult(
        scores=scores,
        probabilities=probabilities,
        n_constellations=constellation_count,
        n_satellites=len(unique_satellites),
        best_index=int(np.argmax(scores)),
    )


def satellite_pair_costs(
    dd_result,
    receiver_position_ecef: np.ndarray,
    *,
    scale_m: float = 5.0,
) -> SatellitePairCostResult:
    """Return pivot-invariant mean pair cost incident to each satellite."""

    position = np.asarray(receiver_position_ecef, dtype=np.float64).reshape(3)
    scale = float(scale_m)
    if not np.all(np.isfinite(position)):
        raise ValueError("receiver_position_ecef must be finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale_m must be finite and positive")
    residuals = np.asarray(
        dd_pseudorange_residuals_m(dd_result, position), dtype=np.float64
    )
    ref_ids = tuple(str(value) for value in dd_result.ref_sat_ids)
    sat_ids = tuple(str(value) for value in dd_result.sat_ids)
    cost_sum: dict[str, float] = {}
    pair_count: dict[str, int] = {}
    for system in sorted({_system_id(value) for value in ref_ids + sat_ids}):
        row_indices = [
            index
            for index, (ref, sat) in enumerate(zip(ref_ids, sat_ids))
            if _system_id(ref) == system and _system_id(sat) == system
        ]
        if not row_indices:
            continue
        references = {ref_ids[index] for index in row_indices}
        if len(references) != 1:
            raise ValueError("each constellation must use one reference in a DD result")
        satellites = [next(iter(references))] + [sat_ids[index] for index in row_indices]
        innovations = np.zeros(len(satellites), dtype=np.float64)
        innovations[1:] = residuals[row_indices]
        for left in range(len(satellites)):
            for right in range(left + 1, len(satellites)):
                difference = innovations[left] - innovations[right]
                cost = float(np.log1p((difference / scale) ** 2))
                for satellite in (satellites[left], satellites[right]):
                    cost_sum[satellite] = cost_sum.get(satellite, 0.0) + cost
                    pair_count[satellite] = pair_count.get(satellite, 0) + 1
    if not pair_count:
        raise ValueError("DD result has insufficient same-constellation support")
    ordered = tuple(sorted(pair_count))
    return SatellitePairCostResult(
        satellite_ids=ordered,
        mean_pair_costs=np.asarray(
            [cost_sum[satellite] / pair_count[satellite] for satellite in ordered],
            dtype=np.float64,
        ),
        pair_counts=np.asarray([pair_count[satellite] for satellite in ordered], dtype=np.int64),
    )
