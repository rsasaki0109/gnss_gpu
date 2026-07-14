"""Candidate-centred 3D-mapping-aided GNSS likelihoods.

The functions in this module score a small cloud of receiver-position
hypotheses.  They deliberately keep geometry generation (for example a
PLATEAU BVH LOS query) separate from measurement scoring so that the numerical
model can be tested without CUDA or a city model.

The pseudorange score is clock-free: a receiver clock term is estimated for
each candidate from its predicted-LOS measurements and removed before the
per-satellite likelihood is evaluated.  Predicted LOS residuals use a narrow
zero-mean Gaussian.  Predicted NLOS residuals use an asymmetric Gaussian whose
positive tail is wider, reflecting the excess path length of reflected or
diffracted signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


_LOG_2PI = float(np.log(2.0 * np.pi))
_C_LIGHT = 299_792_458.0
_OMEGA_E = 7.2921151467e-5


@dataclass(frozen=True)
class Candidate3DMAResult:
    """Detailed output of :func:`score_candidate_positions`."""

    scores: np.ndarray
    probabilities: np.ndarray
    pseudorange_scores: np.ndarray
    visibility_scores: np.ndarray
    road_scores: np.ndarray
    clock_bias_m: np.ndarray
    innovations_m: np.ndarray
    best_index: int


@dataclass(frozen=True)
class RecurrenceVectorResult:
    """Four-satellite position solutions and candidate classification scores."""

    scores: np.ndarray
    probabilities: np.ndarray
    subset_positions_ecef: np.ndarray
    subset_indices: np.ndarray
    ranging_errors_m: np.ndarray
    best_index: int


def solve_four_satellite_position(
    satellite_ecef,
    pseudoranges_m,
    initial_position_ecef,
    *,
    max_iterations: int = 12,
    tolerance_m: float = 1.0e-4,
) -> tuple[np.ndarray, float]:
    """Solve the exact four-observation position/clock system."""

    satellites = _as_finite_array(satellite_ecef, name="satellite_ecef", ndim=2)
    pseudoranges = _as_finite_array(pseudoranges_m, name="pseudoranges_m", ndim=1)
    position = _as_finite_array(
        initial_position_ecef, name="initial_position_ecef", ndim=1
    ).copy()
    if satellites.shape != (4, 3) or pseudoranges.shape != (4,) or position.shape != (3,):
        raise ValueError("four-satellite solve requires shapes (4,3), (4,), and (3,)")
    clock = float(np.median(pseudoranges - np.linalg.norm(satellites - position, axis=1)))
    for _ in range(max(1, int(max_iterations))):
        delta = satellites - position[None, :]
        ranges = np.linalg.norm(delta, axis=1)
        if np.any(ranges <= 1.0):
            raise ValueError("satellite and receiver positions are degenerate")
        los = delta / ranges[:, None]
        residual = pseudoranges - (ranges + clock)
        design = np.column_stack((-los, np.ones(4, dtype=np.float64)))
        try:
            step = np.linalg.solve(design, residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError("four-satellite geometry is singular") from exc
        position += step[:3]
        clock += float(step[3])
        if float(np.linalg.norm(step)) <= float(tolerance_m):
            break
    if not (np.isfinite(position).all() and np.isfinite(clock)):
        raise ValueError("four-satellite solution is non-finite")
    return position, clock


def recurrence_vector_scores(
    candidate_positions_ecef,
    satellite_ecef,
    pseudoranges_m,
    predicted_los,
    initial_position_ecef,
    *,
    observed_los_probability=None,
    satellite_weights=None,
    clock_group_ids=None,
    max_satellites_per_group: int = 9,
    sigma_los_m: float = 3.0,
    nlos_bias_m: float = 15.0,
    sigma_nlos_m: float = 20.0,
) -> RecurrenceVectorResult:
    """Faithful four-satellite recurrence-vector visibility scoring.

    A standalone position is first solved for every usable four-satellite
    subset. Candidate-minus-subset recurrence vectors are then projected onto
    that subset's LOS directions to estimate per-satellite ranging errors.
    Error-derived LOS/NLOS probabilities are compared with the candidate's 3D
    visibility classification and accumulated across subsets.
    """

    candidates = _as_finite_array(
        candidate_positions_ecef, name="candidate_positions_ecef", ndim=2
    )
    satellites = _as_finite_array(satellite_ecef, name="satellite_ecef", ndim=2)
    pseudoranges = _as_finite_array(pseudoranges_m, name="pseudoranges_m", ndim=1)
    visibility = np.asarray(predicted_los, dtype=bool)
    source = _as_finite_array(
        initial_position_ecef, name="initial_position_ecef", ndim=1
    )
    if candidates.shape[1] != 3 or satellites.shape[1] != 3:
        raise ValueError("candidate and satellite positions must have three columns")
    n_sat = satellites.shape[0]
    if pseudoranges.shape != (n_sat,) or visibility.shape != (len(candidates), n_sat):
        raise ValueError("pseudorange/visibility shapes must match satellites and candidates")
    if source.shape != (3,) or n_sat < 4:
        raise ValueError("initial position must have shape (3,) and at least four satellites are required")
    observed = (
        np.ones(n_sat, dtype=np.float64)
        if observed_los_probability is None
        else _as_finite_array(observed_los_probability, name="observed_los_probability", ndim=1)
    )
    weights = (
        np.ones(n_sat, dtype=np.float64)
        if satellite_weights is None
        else _as_finite_array(satellite_weights, name="satellite_weights", ndim=1)
    )
    if observed.shape != (n_sat,) or weights.shape != (n_sat,):
        raise ValueError("observed probabilities and weights must match satellites")
    groups = np.zeros(n_sat, dtype=np.int64) if clock_group_ids is None else np.asarray(clock_group_ids)
    if groups.shape != (n_sat,):
        raise ValueError("clock_group_ids must match satellites")

    subset_list: list[tuple[int, ...]] = []
    quality = observed * np.maximum(weights, 0.0)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        members = members[np.argsort(quality[members])[::-1]][: max(4, int(max_satellites_per_group))]
        if members.size >= 4:
            subset_list.extend(combinations(members.tolist(), 4))
    if not subset_list:
        selected = np.argsort(quality)[::-1][: max(4, int(max_satellites_per_group))]
        subset_list.extend(combinations(selected.tolist(), 4))

    subset_positions: list[np.ndarray] = []
    valid_subsets: list[tuple[int, ...]] = []
    errors_by_subset: list[np.ndarray] = []
    scores = np.zeros(len(candidates), dtype=np.float64)
    score_weight = np.zeros(len(candidates), dtype=np.float64)
    sigma_los = _positive_scalar(sigma_los_m, name="sigma_los_m")
    sigma_nlos = _positive_scalar(sigma_nlos_m, name="sigma_nlos_m")
    nlos_bias = float(nlos_bias_m)
    for subset_tuple in subset_list:
        subset = np.asarray(subset_tuple, dtype=np.int64)
        try:
            subset_position, _clock = solve_four_satellite_position(
                satellites[subset], pseudoranges[subset], source
            )
        except ValueError:
            continue
        los_delta = satellites[subset] - subset_position[None, :]
        los = los_delta / np.linalg.norm(los_delta, axis=1)[:, None]
        recurrence = candidates - subset_position[None, :]
        ranging_errors = recurrence @ los.T
        los_log = -0.5 * (ranging_errors / sigma_los) ** 2 - np.log(sigma_los)
        nlos_log = -0.5 * ((ranging_errors - nlos_bias) / sigma_nlos) ** 2 - np.log(sigma_nlos)
        peak = np.maximum(los_log, nlos_log)
        p_los = np.exp(los_log - peak) / (
            np.exp(los_log - peak) + np.exp(nlos_log - peak)
        )
        classification = np.where(visibility[:, subset], p_los, 1.0 - p_los)
        subset_weights = np.maximum(weights[subset], 0.0)[None, :]
        scores += np.sum(subset_weights * np.log(np.clip(classification, 1.0e-12, 1.0)), axis=1)
        score_weight += float(np.sum(subset_weights))
        subset_positions.append(subset_position)
        valid_subsets.append(subset_tuple)
        errors_by_subset.append(ranging_errors)
    if not valid_subsets:
        raise ValueError("no nonsingular four-satellite subset solution")
    scores /= np.maximum(score_weight, 1.0e-12)
    shifted = scores - float(np.max(scores))
    probability = np.exp(np.clip(shifted, -745.0, 0.0))
    probability /= float(np.sum(probability))
    return RecurrenceVectorResult(
        scores=scores,
        probabilities=probability,
        subset_positions_ecef=np.asarray(subset_positions),
        subset_indices=np.asarray(valid_subsets, dtype=np.int64),
        ranging_errors_m=np.asarray(errors_by_subset).transpose(1, 0, 2),
        best_index=int(np.argmax(scores)),
    )


def road_mode_trigger(
    source_road_distances_m,
    *,
    closest_candidate_road_distances_m=None,
    min_distance_m: float = 2.5,
    max_candidate_distance_m: float = 0.5,
    min_contiguous_epochs: int = 10,
) -> bool:
    """Return whether an OSM road-mode correction is eligible to run.

    The gate uses only the source trajectory and map geometry.  Non-finite
    distances break a run, as do epochs closer than ``min_distance_m`` to the
    road centreline.  When closest candidate distances are supplied, the
    search grid must also reach within ``max_candidate_distance_m`` of a road.
    This makes a road prior abstain when the source already agrees with the
    mapped road or when it would merely pull the estimate to a grid boundary.
    """

    distances = np.asarray(source_road_distances_m, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("source_road_distances_m must be one-dimensional")
    threshold = float(min_distance_m)
    reach_threshold = float(max_candidate_distance_m)
    contiguous = int(min_contiguous_epochs)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("min_distance_m must be finite and >= 0")
    if not np.isfinite(reach_threshold) or reach_threshold < 0.0:
        raise ValueError("max_candidate_distance_m must be finite and >= 0")
    if contiguous <= 0:
        raise ValueError("min_contiguous_epochs must be positive")

    active_mask = np.isfinite(distances) & (distances >= threshold)
    if closest_candidate_road_distances_m is not None:
        closest = np.asarray(
            closest_candidate_road_distances_m, dtype=np.float64
        )
        if closest.shape != distances.shape:
            raise ValueError(
                "closest_candidate_road_distances_m must match source distances"
            )
        active_mask &= np.isfinite(closest) & (closest <= reach_threshold)

    run = 0
    for active in active_mask:
        run = run + 1 if active else 0
        if run >= contiguous:
            return True
    return False


def visibility_mode_cluster_scores(
    scores,
    predicted_los,
    grid_shape: tuple[int, int],
    *,
    score_margin: float = 4.0,
    max_hamming: int = 1,
    outside_penalty: float = 5.0,
) -> np.ndarray:
    """Prefer a spatially supported PLATEAU-visibility likelihood mode.

    High-scoring grid cells are joined by 4-neighbour region growing when
    their predicted visibility masks differ by at most ``max_hamming`` bits.
    The component with the largest log probability mass is retained as the
    preferred mode.  Other components receive a finite penalty so multi-epoch
    accumulation can still recover if the preferred mode changes later.
    """

    values = _as_finite_array(scores, name="scores", ndim=1)
    los = np.asarray(predicted_los, dtype=bool)
    rows, columns = (int(grid_shape[0]), int(grid_shape[1]))
    if rows <= 0 or columns <= 0 or rows * columns != values.size:
        raise ValueError("grid_shape must be positive and match scores")
    if los.ndim != 2 or los.shape[0] != values.size:
        raise ValueError("predicted_los must have one row per score")
    margin = float(score_margin)
    penalty = float(outside_penalty)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("score_margin must be finite and >= 0")
    if int(max_hamming) < 0:
        raise ValueError("max_hamming must be >= 0")
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("outside_penalty must be finite and >= 0")

    active = values >= float(np.max(values)) - margin
    visited = np.zeros(values.size, dtype=bool)
    components: list[list[int]] = []
    for seed in np.flatnonzero(active):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [int(seed)]
        component: list[int] = []
        while stack:
            index = stack.pop()
            component.append(index)
            row, column = divmod(index, columns)
            neighbours = []
            if row > 0:
                neighbours.append(index - columns)
            if row + 1 < rows:
                neighbours.append(index + columns)
            if column > 0:
                neighbours.append(index - 1)
            if column + 1 < columns:
                neighbours.append(index + 1)
            for neighbour in neighbours:
                if visited[neighbour] or not active[neighbour]:
                    continue
                hamming = int(np.count_nonzero(los[index] != los[neighbour]))
                if hamming <= int(max_hamming):
                    visited[neighbour] = True
                    stack.append(int(neighbour))
        components.append(component)

    def log_mass(component: list[int]) -> float:
        component_scores = values[np.asarray(component, dtype=np.int64)]
        peak = float(np.max(component_scores))
        return peak + float(np.log(np.sum(np.exp(component_scores - peak))))

    preferred = max(components, key=log_mass)
    preferred_mask = np.zeros(values.size, dtype=bool)
    preferred_mask[preferred] = True
    adjusted = values.copy()
    adjusted[~preferred_mask] -= penalty
    return adjusted


def robust_subset_consensus_scores(
    innovations_m,
    predicted_los,
    *,
    observed_los_probability=None,
    satellite_weights=None,
    scale_m: float = 3.0,
    subset_size: int = 4,
    max_satellites: int = 10,
    subset_quantile: float = 0.2,
) -> np.ndarray:
    """Score candidates from the best-supported small satellite subsets.

    The highest-quality satellites form many small subsets.  Each subset gets
    a clock-free robust dispersion cost, and the requested lower quantile is
    used as the consensus score.  Thus several mutually consistent LOS
    measurements can support a candidate without requiring every observation
    to be clean.
    """

    innovations = _as_finite_array(
        innovations_m, name="innovations_m", ndim=2
    )
    los = np.asarray(predicted_los, dtype=bool)
    if los.shape != innovations.shape:
        raise ValueError("predicted_los must match innovations_m")
    n_satellites = innovations.shape[1]
    subset_n = int(subset_size)
    max_n = int(max_satellites)
    if subset_n < 2 or subset_n > n_satellites:
        raise ValueError("subset_size must lie in [2, n_satellite]")
    if max_n < subset_n:
        raise ValueError("max_satellites must be >= subset_size")
    quantile = float(subset_quantile)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("subset_quantile must lie in [0, 1]")
    scale = _positive_scalar(scale_m, name="scale_m")

    if satellite_weights is None:
        weights = np.ones(n_satellites, dtype=np.float64)
    else:
        weights = _as_finite_array(
            satellite_weights, name="satellite_weights", ndim=1
        )
        if weights.shape != (n_satellites,) or np.any(weights < 0.0):
            raise ValueError(
                "satellite_weights must have shape (n_satellite,) and be >= 0"
            )
    if observed_los_probability is None:
        observed = np.ones(n_satellites, dtype=np.float64)
    else:
        observed = _as_finite_array(
            observed_los_probability, name="observed_los_probability", ndim=1
        )
        if observed.shape != (n_satellites,) or np.any((observed < 0.0) | (observed > 1.0)):
            raise ValueError(
                "observed_los_probability must have shape (n_satellite,) and lie in [0, 1]"
            )

    selected = np.argsort(observed * weights)[::-1][: min(max_n, n_satellites)]
    subset_indices = np.asarray(
        list(combinations(selected.tolist(), subset_n)), dtype=np.int64
    )
    if subset_indices.size == 0:
        raise ValueError("no satellite subsets could be generated")
    values = innovations[:, subset_indices]
    centred = values - np.median(values, axis=2)[:, :, None]
    costs = np.mean(np.log1p((centred / scale) ** 2), axis=2)
    eligible = np.all(los[:, subset_indices], axis=2)
    eligible_costs = np.where(eligible, costs, np.nan)

    scores = np.empty(innovations.shape[0], dtype=np.float64)
    for candidate_index in range(innovations.shape[0]):
        candidate_costs = eligible_costs[candidate_index]
        candidate_costs = candidate_costs[np.isfinite(candidate_costs)]
        if candidate_costs.size == 0:
            candidate_costs = costs[candidate_index]
        scores[candidate_index] = -float(np.quantile(candidate_costs, quantile))
    return scores


def multipivot_consensus_scores(
    innovations_m,
    predicted_los,
    *,
    observed_los_probability=None,
    satellite_weights=None,
    scale_m: float = 5.0,
    max_pivots: int = 6,
) -> np.ndarray:
    """Score candidates from robust single-difference residual consensus.

    Several likely-LOS satellites are used as alternative pivots.  A bad pivot
    therefore cannot dominate the result as it does in a conventional
    single-reference construction.  The returned values are log-like scores;
    larger is better.
    """

    innovations = _as_finite_array(
        innovations_m, name="innovations_m", ndim=2
    )
    los = np.asarray(predicted_los, dtype=bool)
    if los.shape != innovations.shape:
        raise ValueError("predicted_los must match innovations_m")
    n_satellites = innovations.shape[1]
    if n_satellites < 4:
        raise ValueError("innovations_m must contain at least 4 satellites")
    scale = _positive_scalar(scale_m, name="scale_m")
    if int(max_pivots) <= 0:
        raise ValueError("max_pivots must be positive")

    if satellite_weights is None:
        weights = np.ones(n_satellites, dtype=np.float64)
    else:
        weights = _as_finite_array(
            satellite_weights, name="satellite_weights", ndim=1
        )
        if weights.shape != (n_satellites,) or np.any(weights < 0.0):
            raise ValueError(
                "satellite_weights must have shape (n_satellite,) and be >= 0"
            )
    if observed_los_probability is None:
        observed = np.ones(n_satellites, dtype=np.float64)
    else:
        observed = _as_finite_array(
            observed_los_probability, name="observed_los_probability", ndim=1
        )
        if observed.shape != (n_satellites,) or np.any((observed < 0.0) | (observed > 1.0)):
            raise ValueError(
                "observed_los_probability must have shape (n_satellite,) and lie in [0, 1]"
            )

    pivot_priority = observed * weights
    pivot_indices = np.argsort(pivot_priority)[::-1][: min(int(max_pivots), n_satellites)]
    pivot_scores = np.empty((innovations.shape[0], len(pivot_indices)), dtype=np.float64)
    for column, pivot in enumerate(pivot_indices):
        differences = innovations - innovations[:, pivot, None]
        robust_cost = np.log1p((differences / scale) ** 2)
        usable = los & los[:, pivot, None]
        pair_weights = weights[None, :] * observed[None, :] * usable
        pair_weights[:, pivot] = 0.0
        denom = np.sum(pair_weights, axis=1)
        fallback = denom <= 0.0
        denom[fallback] = 1.0
        pivot_scores[:, column] = -np.sum(pair_weights * robust_cost, axis=1) / denom
        pivot_scores[fallback, column] = -np.inf
    finite_count = np.sum(np.isfinite(pivot_scores), axis=1)
    valid = finite_count > 0
    if not np.any(valid):
        raise ValueError("no candidate has a usable LOS pivot pair")
    scores = np.empty(innovations.shape[0], dtype=np.float64)
    for candidate_index in np.flatnonzero(valid):
        finite_scores = pivot_scores[candidate_index]
        scores[candidate_index] = float(
            np.median(finite_scores[np.isfinite(finite_scores)])
        )
    scores[~valid] = float(np.min(scores[valid])) - 1.0
    return scores


def temporal_bias_consistency_scores(
    innovations_by_epoch,
    satellite_ids_by_epoch,
    *,
    scale_m: float = 2.0,
    min_epochs_per_satellite: int = 8,
) -> np.ndarray:
    """Score constant-offset candidates after removing per-satellite intercepts.

    Each satellite's temporal median innovation is treated as an unknown code
    bias.  Candidates are compared using only the remaining time variation,
    with a Cauchy cost to limit changing multipath.  Inputs may contain a
    different satellite set at every epoch, but must share the candidate axis.
    """

    if not innovations_by_epoch:
        raise ValueError("innovations_by_epoch must not be empty")
    if len(innovations_by_epoch) != len(satellite_ids_by_epoch):
        raise ValueError("innovation and satellite-id epoch counts must match")
    scale = _positive_scalar(scale_m, name="scale_m")
    if int(min_epochs_per_satellite) < 2:
        raise ValueError("min_epochs_per_satellite must be >= 2")

    arrays: list[np.ndarray] = []
    candidate_count: int | None = None
    histories: dict[str, list[np.ndarray]] = {}
    for epoch_index, (values, sat_ids) in enumerate(
        zip(innovations_by_epoch, satellite_ids_by_epoch)
    ):
        arr = _as_finite_array(
            values, name=f"innovations_by_epoch[{epoch_index}]", ndim=2
        )
        ids = [str(sat_id) for sat_id in sat_ids]
        if arr.shape[1] != len(ids):
            raise ValueError("each satellite-id row must match its innovation columns")
        if candidate_count is None:
            candidate_count = arr.shape[0]
        elif arr.shape[0] != candidate_count:
            raise ValueError("all epochs must share the candidate count")
        arrays.append(arr)
        for sat_index, sat_id in enumerate(ids):
            histories.setdefault(sat_id, []).append(arr[:, sat_index])

    assert candidate_count is not None
    total_cost = np.zeros(candidate_count, dtype=np.float64)
    total_rows = 0
    for history in histories.values():
        if len(history) < int(min_epochs_per_satellite):
            continue
        values = np.stack(history, axis=1)
        centred = values - np.median(values, axis=1)[:, None]
        total_cost += np.sum(np.log1p((centred / scale) ** 2), axis=1)
        total_rows += values.shape[1]
    if total_rows == 0:
        raise ValueError("no satellite meets min_epochs_per_satellite")
    return -total_cost / float(total_rows)


def _as_finite_array(value, *, name: str, ndim: int) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def _positive_scalar(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return out


def horizontal_candidates_ecef(
    center_ecef,
    east_offsets_m,
    north_offsets_m,
    *,
    grid: bool = True,
) -> np.ndarray:
    """Generate horizontal position hypotheses in the local ENU tangent plane.

    Parameters
    ----------
    center_ecef : array-like, shape (3,)
        Centre of the candidate cloud in ECEF metres.
    east_offsets_m, north_offsets_m : array-like
        Local offsets in metres.  With ``grid=True`` their Cartesian product is
        returned; otherwise the arrays are paired element by element.
    grid : bool
        Generate a full east/north grid instead of paired offsets.
    """

    center = _as_finite_array(center_ecef, name="center_ecef", ndim=1)
    if center.shape != (3,) or np.linalg.norm(center) < 1.0e6:
        raise ValueError("center_ecef must have shape (3,) and be a valid ECEF position")
    east = _as_finite_array(east_offsets_m, name="east_offsets_m", ndim=1)
    north = _as_finite_array(north_offsets_m, name="north_offsets_m", ndim=1)
    if east.size == 0 or north.size == 0:
        raise ValueError("offset arrays must not be empty")

    if grid:
        east_grid, north_grid = np.meshgrid(east, north, indexing="xy")
        east_values = east_grid.ravel()
        north_values = north_grid.ravel()
    else:
        if east.shape != north.shape:
            raise ValueError("paired east/north offsets must have the same shape")
        east_values = east
        north_values = north

    x, y, z = center
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.hypot(x, y))
    east_hat = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north_hat = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    )
    return (
        center[None, :]
        + east_values[:, None] * east_hat[None, :]
        + north_values[:, None] * north_hat[None, :]
    )


def cn0_to_los_probability(
    cn0_dbhz,
    *,
    midpoint_dbhz: float = 32.0,
    scale_db: float = 4.0,
) -> np.ndarray:
    """Convert C/N0 into a soft observed-LOS probability using a logistic model."""

    cn0 = _as_finite_array(cn0_dbhz, name="cn0_dbhz", ndim=1)
    scale = _positive_scalar(scale_db, name="scale_db")
    midpoint = float(midpoint_dbhz)
    if not np.isfinite(midpoint):
        raise ValueError("midpoint_dbhz must be finite")
    logits = np.clip((cn0 - midpoint) / scale, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _weighted_clock_bias(
    residuals: np.ndarray,
    weights: np.ndarray,
    los: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_count = residuals.shape[0]
    clocks_by_satellite = np.empty_like(residuals)
    diagnostic_clocks = np.empty(candidate_count, dtype=np.float64)
    groups = np.unique(group_ids)
    for idx in range(candidate_count):
        group_clock_values: list[float] = []
        group_clock_weights: list[float] = []
        for group in groups:
            member = (group_ids == group) & (weights > 0.0)
            use = member & los[idx]
            if not np.any(use):
                use = member
            denom = float(np.sum(weights[use]))
            if denom <= 0.0:
                continue
            clock = float(np.sum(weights[use] * residuals[idx, use]) / denom)
            clocks_by_satellite[idx, group_ids == group] = clock
            group_clock_values.append(clock)
            group_clock_weights.append(denom)
        if not group_clock_values:
            raise ValueError("satellite_weights must contain a positive value")
        diagnostic_clocks[idx] = float(
            np.average(group_clock_values, weights=group_clock_weights)
        )
    return clocks_by_satellite, diagnostic_clocks


def score_candidate_positions(
    candidate_ecef,
    satellite_ecef,
    pseudoranges_m,
    predicted_los,
    *,
    satellite_weights=None,
    clock_group_ids=None,
    observed_los_probability=None,
    road_outside_distance_m=None,
    sigma_los_m: float = 3.0,
    nlos_bias_m: float = 15.0,
    sigma_nlos_negative_m: float = 8.0,
    sigma_nlos_positive_m: float = 25.0,
    visibility_weight: float = 1.0,
    road_sigma_m: float = 1.5,
    road_weight: float = 1.0,
    apply_sagnac: bool = True,
) -> Candidate3DMAResult:
    """Score receiver-position candidates using 3DMA and pseudoranges.

    ``predicted_los`` is normally obtained by calling
    ``BVHAccelerator.check_los_batch(candidate_ecef, repeated_satellite_ecef)``.
    ``clock_group_ids`` may identify separate constellation or signal clock
    offsets. ``road_outside_distance_m`` should be zero for candidates inside the
    allowed road corridor and their positive distance outside it.  Omitting
    either the observed visibility probability or road distance disables that
    component cleanly.
    """

    candidates = _as_finite_array(candidate_ecef, name="candidate_ecef", ndim=2)
    satellites = _as_finite_array(satellite_ecef, name="satellite_ecef", ndim=2)
    pseudoranges = _as_finite_array(pseudoranges_m, name="pseudoranges_m", ndim=1)
    los = np.asarray(predicted_los, dtype=bool)
    if candidates.shape[1:] != (3,) or candidates.shape[0] == 0:
        raise ValueError("candidate_ecef must have shape (n_candidate, 3)")
    if satellites.shape[1:] != (3,) or satellites.shape[0] < 4:
        raise ValueError("satellite_ecef must have shape (n_satellite, 3), with at least 4 satellites")
    if pseudoranges.shape != (satellites.shape[0],):
        raise ValueError("pseudoranges_m must have shape (n_satellite,)")
    if los.shape != (candidates.shape[0], satellites.shape[0]):
        raise ValueError("predicted_los must have shape (n_candidate, n_satellite)")

    if satellite_weights is None:
        weights = np.ones(satellites.shape[0], dtype=np.float64)
    else:
        weights = _as_finite_array(satellite_weights, name="satellite_weights", ndim=1)
        if weights.shape != (satellites.shape[0],) or np.any(weights < 0.0):
            raise ValueError("satellite_weights must have shape (n_satellite,) and be >= 0")
        if not np.any(weights > 0.0):
            raise ValueError("satellite_weights must contain a positive value")

    if clock_group_ids is None:
        group_ids = np.zeros(satellites.shape[0], dtype=np.int64)
    else:
        group_ids_raw = np.asarray(clock_group_ids)
        if group_ids_raw.ndim != 1 or group_ids_raw.shape != (satellites.shape[0],):
            raise ValueError("clock_group_ids must have shape (n_satellite,)")
        if not np.issubdtype(group_ids_raw.dtype, np.number):
            raise ValueError("clock_group_ids must be numeric")
        group_ids_float = np.asarray(group_ids_raw, dtype=np.float64)
        if not np.all(np.isfinite(group_ids_float)):
            raise ValueError("clock_group_ids must be finite")
        _, group_ids = np.unique(group_ids_raw, return_inverse=True)

    sigma_los = _positive_scalar(sigma_los_m, name="sigma_los_m")
    nlos_bias = float(nlos_bias_m)
    if not np.isfinite(nlos_bias) or nlos_bias < 0.0:
        raise ValueError("nlos_bias_m must be finite and >= 0")
    sigma_nlos_neg = _positive_scalar(sigma_nlos_negative_m, name="sigma_nlos_negative_m")
    sigma_nlos_pos = _positive_scalar(sigma_nlos_positive_m, name="sigma_nlos_positive_m")
    vis_weight = float(visibility_weight)
    r_weight = float(road_weight)
    if not np.isfinite(vis_weight) or vis_weight < 0.0:
        raise ValueError("visibility_weight must be finite and >= 0")
    if not np.isfinite(r_weight) or r_weight < 0.0:
        raise ValueError("road_weight must be finite and >= 0")

    satellite_rows = np.broadcast_to(
        satellites[None, :, :], (candidates.shape[0], satellites.shape[0], 3)
    )
    if apply_sagnac:
        approximate_ranges = np.linalg.norm(
            satellite_rows - candidates[:, None, :], axis=2
        )
        theta = _OMEGA_E * approximate_ranges / _C_LIGHT
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        satellite_rows = satellite_rows.copy()
        satellite_rows[:, :, 0] = (
            satellites[None, :, 0] * cos_theta
            + satellites[None, :, 1] * sin_theta
        )
        satellite_rows[:, :, 1] = (
            -satellites[None, :, 0] * sin_theta
            + satellites[None, :, 1] * cos_theta
        )
    geometric_ranges = np.linalg.norm(
        satellite_rows - candidates[:, None, :], axis=2
    )
    raw_residuals = pseudoranges[None, :] - geometric_ranges
    clock_by_satellite, clock_bias = _weighted_clock_bias(
        raw_residuals, weights, los, group_ids
    )
    innovations = raw_residuals - clock_by_satellite

    los_logpdf = -0.5 * (innovations / sigma_los) ** 2 - np.log(sigma_los) - 0.5 * _LOG_2PI
    nlos_centred = innovations - nlos_bias
    nlos_sigma = np.where(nlos_centred < 0.0, sigma_nlos_neg, sigma_nlos_pos)
    nlos_logpdf = (
        -0.5 * (nlos_centred / nlos_sigma) ** 2
        - np.log(nlos_sigma)
        - 0.5 * _LOG_2PI
    )
    logpdf = np.where(los, los_logpdf, nlos_logpdf)
    pseudorange_scores = np.sum(weights[None, :] * logpdf, axis=1)

    visibility_scores = np.zeros(candidates.shape[0], dtype=np.float64)
    if observed_los_probability is not None:
        observed = _as_finite_array(
            observed_los_probability, name="observed_los_probability", ndim=1
        )
        if observed.shape != (satellites.shape[0],) or np.any((observed < 0.0) | (observed > 1.0)):
            raise ValueError(
                "observed_los_probability must have shape (n_satellite,) and lie in [0, 1]"
            )
        eps = 1.0e-9
        agreement = np.where(los, observed[None, :], 1.0 - observed[None, :])
        visibility_scores = vis_weight * np.sum(
            weights[None, :] * np.log(np.clip(agreement, eps, 1.0)), axis=1
        )

    road_scores = np.zeros(candidates.shape[0], dtype=np.float64)
    if road_outside_distance_m is not None:
        road_distance = _as_finite_array(
            road_outside_distance_m, name="road_outside_distance_m", ndim=1
        )
        if road_distance.shape != (candidates.shape[0],) or np.any(road_distance < 0.0):
            raise ValueError(
                "road_outside_distance_m must have shape (n_candidate,) and be >= 0"
            )
        road_sigma = _positive_scalar(road_sigma_m, name="road_sigma_m")
        road_scores = -0.5 * r_weight * (road_distance / road_sigma) ** 2

    scores = pseudorange_scores + visibility_scores + road_scores
    shifted = scores - float(np.max(scores))
    probabilities = np.exp(np.clip(shifted, -745.0, 0.0))
    probabilities /= float(np.sum(probabilities))
    best_index = int(np.argmax(scores))
    return Candidate3DMAResult(
        scores=scores,
        probabilities=probabilities,
        pseudorange_scores=pseudorange_scores,
        visibility_scores=visibility_scores,
        road_scores=road_scores,
        clock_bias_m=clock_bias,
        innovations_m=innovations,
        best_index=best_index,
    )


__all__ = [
    "Candidate3DMAResult",
    "RecurrenceVectorResult",
    "cn0_to_los_probability",
    "horizontal_candidates_ecef",
    "multipivot_consensus_scores",
    "recurrence_vector_scores",
    "road_mode_trigger",
    "robust_subset_consensus_scores",
    "score_candidate_positions",
    "solve_four_satellite_position",
    "temporal_bias_consistency_scores",
    "visibility_mode_cluster_scores",
]
