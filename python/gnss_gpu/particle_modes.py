"""Scalable weighted-mode extraction for particle-position posteriors.

The GPU particle filter deliberately permits multimodal posteriors.  This
module keeps those modes separate at emission time instead of reducing the
cloud to one global weighted mean.  It uses weighted voxel density and
connected components, avoiding an O(N**2) neighbourhood matrix for the
50,000-particle PPC configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class ParticleMode:
    """One connected high-density region and its assigned posterior mass."""

    position: np.ndarray
    covariance: np.ndarray
    mass: float
    core_mass: float
    particle_count: int
    core_cell_count: int
    peak_cell_mass: float


@dataclass(frozen=True)
class ParticleModeResult:
    """Mode decomposition of a normalized particle posterior."""

    modes: tuple[ParticleMode, ...]
    weighted_mean: np.ndarray
    assigned_mass: float
    noise_mass: float
    effective_sample_size: float
    input_particle_count: int
    analyzed_particle_count: int


@dataclass(frozen=True)
class ParticleModeSelection:
    """An accepted mode or an explicit abstention with diagnostics."""

    accepted: bool
    mode_index: int | None
    position: np.ndarray | None
    reason: str
    selected_mass: float
    runner_up_mass: float
    score_ratio: float
    prediction_distance_m: float
    weighted_mean_distance_m: float


_NEIGHBOUR_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def _normalized_weights(log_weights: np.ndarray, finite_rows: np.ndarray) -> np.ndarray:
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)[finite_rows]
    finite_lw = np.isfinite(lw)
    if not np.any(finite_lw):
        return np.empty(0, dtype=np.float64)
    floor = float(np.min(lw[finite_lw])) - 100.0
    lw = np.where(finite_lw, lw, floor)
    shifted = lw - float(np.max(lw))
    weights = np.exp(np.maximum(shifted, -745.0))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.empty(0, dtype=np.float64)
    return weights / total


def _weighted_moments(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(np.sum(weights))
    mean = np.sum(points * weights[:, None], axis=0) / total
    delta = points - mean
    covariance = (delta * weights[:, None]).T @ delta / total
    return mean, covariance


def extract_particle_modes(
    particles: np.ndarray,
    log_weights: np.ndarray,
    *,
    voxel_size_m: float = 1.0,
    min_core_cell_mass: float = 1.0e-4,
    min_core_cell_particles: int = 3,
    min_mode_mass: float = 0.01,
    assignment_radius_m: float = 3.0,
    max_modes: int = 8,
    max_particles: int = 8192,
) -> ParticleModeResult:
    """Extract weighted position modes using dense-voxel region growing.

    Low-mass voxels cannot connect two dense peaks.  Their particles are later
    assigned to the nearest core component only when its core centroid is
    within ``assignment_radius_m``; remote posterior dust remains noise.
    Returned modes are sorted by decreasing assigned posterior mass.
    """

    xyz_all = np.asarray(particles, dtype=np.float64)
    lw_all = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if xyz_all.ndim != 2 or xyz_all.shape[1] < 3:
        raise ValueError("particles must have shape (N, >=3)")
    if xyz_all.shape[0] != lw_all.size:
        raise ValueError("particles and log_weights must have the same length")
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    if not 0.0 <= min_core_cell_mass <= 1.0:
        raise ValueError("min_core_cell_mass must be in [0, 1]")
    if min_core_cell_particles < 1:
        raise ValueError("min_core_cell_particles must be at least one")
    if not 0.0 <= min_mode_mass <= 1.0:
        raise ValueError("min_mode_mass must be in [0, 1]")
    if assignment_radius_m < 0.0:
        raise ValueError("assignment_radius_m must be non-negative")
    if max_modes < 1:
        raise ValueError("max_modes must be at least one")
    if max_particles < 1:
        raise ValueError("max_particles must be at least one")

    finite_rows = np.all(np.isfinite(xyz_all[:, :3]), axis=1)
    xyz = xyz_all[finite_rows, :3]
    weights = _normalized_weights(lw_all, finite_rows)
    if xyz.shape[0] == 0 or weights.size == 0:
        return ParticleModeResult((), np.full(3, np.nan), 0.0, 1.0, 0.0, 0, 0)

    weighted_mean, _ = _weighted_moments(xyz, weights)
    ess = 1.0 / float(np.sum(weights * weights))
    input_particle_count = int(xyz.shape[0])
    if xyz.shape[0] > int(max_particles):
        # Deterministic systematic resampling bounds host clustering cost while
        # retaining the normalized weighted posterior represented by the GPU.
        cdf = np.cumsum(weights)
        targets = (np.arange(int(max_particles), dtype=np.float64) + 0.5) / float(
            max_particles
        )
        indices = np.searchsorted(cdf, targets, side="left")
        xyz = xyz[np.minimum(indices, xyz.shape[0] - 1)]
        weights = np.full(xyz.shape[0], 1.0 / xyz.shape[0], dtype=np.float64)
    analyzed_particle_count = int(xyz.shape[0])
    origin = np.min(xyz, axis=0)
    cells = np.floor((xyz - origin) / float(voxel_size_m)).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    cell_mass = np.bincount(inverse, weights=weights, minlength=len(unique_cells))
    cell_count = np.bincount(inverse, minlength=len(unique_cells))

    core_ids = np.flatnonzero(
        (cell_mass >= float(min_core_cell_mass))
        & (cell_count >= int(min_core_cell_particles))
    )
    if core_ids.size == 0:
        return ParticleModeResult(
            (), weighted_mean, 0.0, 1.0, ess, input_particle_count, analyzed_particle_count
        )
    core_lookup = {tuple(unique_cells[idx]): int(idx) for idx in core_ids}

    parent = {int(idx): int(idx) for idx in core_ids}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            nxt = parent[value]
            parent[value] = root
            value = nxt
        return root

    def union(lhs: int, rhs: int) -> None:
        a = find(lhs)
        b = find(rhs)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for cell_id in core_ids:
        cell = unique_cells[cell_id]
        for offset in _NEIGHBOUR_OFFSETS:
            other = core_lookup.get(tuple(cell + offset))
            if other is not None:
                union(int(cell_id), other)

    component_cells: dict[int, list[int]] = {}
    for cell_id in core_ids:
        component_cells.setdefault(find(int(cell_id)), []).append(int(cell_id))

    components: list[dict[str, object]] = []
    for ids in component_cells.values():
        ids_arr = np.asarray(ids, dtype=np.int64)
        point_mask = np.isin(inverse, ids_arr)
        core_weight = weights[point_mask]
        core_points = xyz[point_mask]
        core_mass = float(np.sum(core_weight))
        core_mean, _ = _weighted_moments(core_points, core_weight)
        components.append(
            {
                "cell_ids": ids_arr,
                "core_mean": core_mean,
                "core_mass": core_mass,
                "peak_cell_mass": float(np.max(cell_mass[ids_arr])),
            }
        )

    components.sort(key=lambda value: float(value["core_mass"]), reverse=True)
    components = components[: int(max_modes)]
    core_means = np.stack([np.asarray(value["core_mean"]) for value in components])

    point_labels = np.full(xyz.shape[0], -1, dtype=np.int64)
    for label, component in enumerate(components):
        point_labels[np.isin(inverse, component["cell_ids"])] = label

    unassigned = np.flatnonzero(point_labels < 0)
    if unassigned.size and assignment_radius_m > 0.0:
        # K is capped by max_modes, so this is O(N*K), not O(N**2).
        distances = np.linalg.norm(
            xyz[unassigned, None, :] - core_means[None, :, :], axis=2
        )
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(unassigned.size), nearest]
        accepted = nearest_distance <= float(assignment_radius_m)
        point_labels[unassigned[accepted]] = nearest[accepted]

    modes: list[ParticleMode] = []
    for label, component in enumerate(components):
        point_mask = point_labels == label
        mode_weights = weights[point_mask]
        mass = float(np.sum(mode_weights))
        if mass < float(min_mode_mass):
            point_labels[point_mask] = -1
            continue
        mean, covariance = _weighted_moments(xyz[point_mask], mode_weights)
        modes.append(
            ParticleMode(
                position=mean,
                covariance=covariance,
                mass=mass,
                core_mass=float(component["core_mass"]),
                particle_count=int(np.count_nonzero(point_mask)),
                core_cell_count=int(len(component["cell_ids"])),
                peak_cell_mass=float(component["peak_cell_mass"]),
            )
        )

    modes.sort(key=lambda mode: mode.mass, reverse=True)
    assigned_mass = float(sum(mode.mass for mode in modes))
    return ParticleModeResult(
        tuple(modes),
        weighted_mean,
        assigned_mass,
        max(0.0, 1.0 - assigned_mass),
        ess,
        input_particle_count,
        analyzed_particle_count,
    )


def select_particle_mode(
    result: ParticleModeResult,
    *,
    predicted_position: np.ndarray | None = None,
    prediction_sigma_m: float = 5.0,
    min_selected_mass: float = 0.20,
    min_score_ratio: float = 1.5,
    require_multiple_modes: bool = False,
    max_prediction_distance_m: float = 20.0,
    min_weighted_mean_distance_m: float = 0.0,
    max_weighted_mean_distance_m: float = 20.0,
) -> ParticleModeSelection:
    """Select a reachable posterior mode, abstaining when evidence is weak."""

    if not result.modes:
        return ParticleModeSelection(
            False, None, None, "no_modes", 0.0, 0.0, 0.0, np.nan, np.nan
        )

    masses = np.asarray([mode.mass for mode in result.modes], dtype=np.float64)
    scores = masses.copy()
    prediction_distances = np.full(masses.size, np.nan, dtype=np.float64)
    if predicted_position is not None:
        predicted = np.asarray(predicted_position, dtype=np.float64).reshape(-1)
        if predicted.size < 3 or not np.all(np.isfinite(predicted[:3])):
            return ParticleModeSelection(
                False, None, None, "invalid_prediction", 0.0, 0.0, 0.0, np.nan, np.nan
            )
        prediction_distances = np.asarray(
            [np.linalg.norm(mode.position - predicted[:3]) for mode in result.modes]
        )
        sigma = max(float(prediction_sigma_m), 1.0e-6)
        scores *= np.exp(-0.5 * np.square(prediction_distances / sigma))

    order = np.argsort(scores)[::-1]
    selected_idx = int(order[0])
    runner_idx = int(order[1]) if order.size > 1 else None
    selected_score = float(scores[selected_idx])
    runner_score = float(scores[runner_idx]) if runner_idx is not None else 0.0
    score_ratio = (
        selected_score / runner_score
        if runner_score > 0.0
        else (float("inf") if selected_score > 0.0 else 0.0)
    )
    selected = result.modes[selected_idx]
    prediction_distance = float(prediction_distances[selected_idx])
    mean_distance = float(np.linalg.norm(selected.position - result.weighted_mean))
    runner_mass = float(masses[runner_idx]) if runner_idx is not None else 0.0

    reason = "accepted"
    if require_multiple_modes and len(result.modes) < 2:
        reason = "single_mode"
    elif selected.mass < float(min_selected_mass):
        reason = "selected_mass"
    elif score_ratio < float(min_score_ratio):
        reason = "score_ratio"
    elif (
        np.isfinite(prediction_distance)
        and max_prediction_distance_m > 0.0
        and prediction_distance > float(max_prediction_distance_m)
    ):
        reason = "prediction_distance"
    elif (
        min_weighted_mean_distance_m > 0.0
        and mean_distance < float(min_weighted_mean_distance_m)
    ):
        reason = "weighted_mean_proximity"
    elif (
        max_weighted_mean_distance_m > 0.0
        and mean_distance > float(max_weighted_mean_distance_m)
    ):
        reason = "weighted_mean_distance"

    accepted = reason == "accepted"
    return ParticleModeSelection(
        accepted=accepted,
        mode_index=selected_idx,
        position=selected.position.copy() if accepted else None,
        reason=reason,
        selected_mass=float(selected.mass),
        runner_up_mass=runner_mass,
        score_ratio=float(score_ratio),
        prediction_distance_m=prediction_distance,
        weighted_mean_distance_m=mean_distance,
    )
