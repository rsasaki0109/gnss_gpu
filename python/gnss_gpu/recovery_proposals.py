"""Deterministic position proposals for ambiguity-basin recovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from gnss_gpu.lambda_ambiguity import integer_search


RawAmbiguityKey: TypeAlias = tuple[str, str, int]
VersionedAmbiguityKey: TypeAlias = tuple[RawAmbiguityKey, int]
AssignmentItem: TypeAlias = tuple[VersionedAmbiguityKey, int]


@dataclass(frozen=True)
class _BankEntry:
    epoch: int
    position_ecef: np.ndarray
    velocity_ecef: np.ndarray
    log_weight: float


@dataclass(frozen=True)
class _AssignmentBankEntry:
    epoch: int
    assignment: tuple[AssignmentItem, ...]
    log_weight: float


class RecoveryAssignmentBank:
    """Bounded causal bank of generation-versioned integer assignments."""

    def __init__(
        self,
        max_assignments: int,
        max_age_epochs: int,
        min_assignment_size: int,
    ) -> None:
        self.max_assignments = int(max_assignments)
        self.max_age_epochs = int(max_age_epochs)
        self.min_assignment_size = int(min_assignment_size)
        if min(self.max_assignments, self.max_age_epochs, self.min_assignment_size) < 1:
            raise ValueError("assignment bank size, age, and dimension must be positive")
        self._entries: list[_AssignmentBankEntry] = []

    def clear(self) -> None:
        """Discard every assignment at a detected ambiguity reset boundary."""

        self._entries.clear()

    def update(
        self,
        epoch: int,
        assignments: Iterable[dict[VersionedAmbiguityKey, int]],
        log_weights: Iterable[float],
    ) -> None:
        assignment_list = [
            tuple(sorted((key, int(value)) for key, value in assignment.items()))
            for assignment in assignments
        ]
        weights = np.asarray(list(log_weights), dtype=np.float64).reshape(-1)
        if len(assignment_list) != len(weights):
            raise ValueError("assignments and log_weights must have equal length")
        if not np.all(np.isfinite(weights)):
            raise ValueError("assignment bank weights must be finite")
        candidates = [
            _AssignmentBankEntry(int(epoch), assignment, float(weight))
            for assignment, weight in zip(assignment_list, weights)
            if len(assignment) >= self.min_assignment_size
        ]
        candidates.extend(
            entry
            for entry in self._entries
            if int(epoch) - entry.epoch <= self.max_age_epochs
        )
        candidates.sort(key=lambda entry: (entry.log_weight, entry.epoch), reverse=True)
        selected: dict[tuple[AssignmentItem, ...], _AssignmentBankEntry] = {}
        for entry in candidates:
            selected.setdefault(entry.assignment, entry)
            if len(selected) >= self.max_assignments:
                break
        self._entries = list(selected.values())

    def compatible_assignments(
        self,
        active_versioned_keys: Iterable[VersionedAmbiguityKey],
        observed_raw_keys: Iterable[RawAmbiguityKey],
        *,
        min_size: int | None = None,
    ) -> tuple[dict[VersionedAmbiguityKey, int], ...]:
        active = set(active_versioned_keys)
        observed = set(observed_raw_keys)
        minimum = self.min_assignment_size if min_size is None else int(min_size)
        if minimum < 1:
            raise ValueError("compatible assignment minimum must be positive")
        compatible: dict[tuple[AssignmentItem, ...], dict[VersionedAmbiguityKey, int]] = {}
        for entry in self._entries:
            projected = tuple(
                item
                for item in entry.assignment
                if item[0] in active and item[0][0] in observed
            )
            if len(projected) >= minimum:
                compatible.setdefault(projected, dict(projected))
        return tuple(compatible.values())

    def rebased_assignments(
        self,
        active_versioned_keys: Iterable[VersionedAmbiguityKey],
        observed_raw_keys: Iterable[RawAmbiguityKey],
        *,
        min_size: int | None = None,
    ) -> tuple[dict[VersionedAmbiguityKey, int], ...]:
        """Re-express same-segment DD integers under the currently observed pivot."""

        active_by_raw = {raw: generation for raw, generation in active_versioned_keys}
        observed = tuple(
            raw for raw in observed_raw_keys if raw in active_by_raw
        )
        minimum = self.min_assignment_size if min_size is None else int(min_size)
        if minimum < 1:
            raise ValueError("rebased assignment minimum must be positive")
        outputs: dict[tuple[AssignmentItem, ...], dict[VersionedAmbiguityKey, int]] = {}
        for entry in self._entries:
            graphs: dict[
                tuple[str, int], dict[str, list[tuple[str, int]]]
            ] = {}
            for (raw_key, _generation), value in entry.assignment:
                ref, sat, wavelength = raw_key
                group = (ref[:1], int(wavelength))
                adjacency = graphs.setdefault(group, {})
                adjacency.setdefault(ref, []).append((sat, int(value)))
                adjacency.setdefault(sat, []).append((ref, -int(value)))
            potentials_by_group: dict[
                tuple[str, int], tuple[dict[str, int], dict[str, int]]
            ] = {}
            for group, adjacency in graphs.items():
                potentials: dict[str, int] = {}
                components: dict[str, int] = {}
                component = 0
                consistent = True
                for start in adjacency:
                    if start in potentials:
                        continue
                    potentials[start] = 0
                    components[start] = component
                    stack = [start]
                    while stack and consistent:
                        node = stack.pop()
                        for neighbor, delta in adjacency[node]:
                            expected = potentials[node] + delta
                            if neighbor in potentials:
                                if potentials[neighbor] != expected:
                                    consistent = False
                                    break
                            else:
                                potentials[neighbor] = expected
                                components[neighbor] = component
                                stack.append(neighbor)
                    component += 1
                if consistent:
                    potentials_by_group[group] = (potentials, components)
            projected: dict[VersionedAmbiguityKey, int] = {}
            for raw_key in observed:
                ref, sat, wavelength = raw_key
                values = potentials_by_group.get((ref[:1], int(wavelength)))
                if values is None:
                    continue
                potentials, components = values
                if (
                    ref in potentials
                    and sat in potentials
                    and components[ref] == components[sat]
                ):
                    projected[(raw_key, int(active_by_raw[raw_key]))] = int(
                        potentials[sat] - potentials[ref]
                    )
            if len(projected) >= minimum:
                canonical = tuple(sorted(projected.items()))
                outputs.setdefault(canonical, projected)
        return tuple(outputs.values())


def complete_versioned_assignment(
    raw_keys: tuple[RawAmbiguityKey, ...],
    generations: dict[RawAmbiguityKey, int],
    ahat_cycles: np.ndarray,
    qahat_cycles2: np.ndarray,
    stable_assignment: dict[VersionedAmbiguityKey, int],
    *,
    target_size: int,
    n_candidates: int,
) -> tuple[tuple[dict[VersionedAmbiguityKey, int], float], ...]:
    """Complete unchanged historical integers with current-generation search."""

    keys = tuple(raw_keys)
    ahat = np.asarray(ahat_cycles, dtype=np.float64).reshape(-1)
    covariance = np.asarray(qahat_cycles2, dtype=np.float64)
    if ahat.size != len(keys) or covariance.shape != (len(keys), len(keys)):
        raise ValueError("ambiguity seed dimensions do not match keys")
    target = int(target_size)
    if target < 1 or target > len(keys) or int(n_candidates) < 1:
        raise ValueError("completion target and candidate count must be valid")
    index = {key: position for position, key in enumerate(keys)}
    stable = [
        (versioned, int(value))
        for versioned, value in stable_assignment.items()
        if versioned[0] in index
        and generations.get(versioned[0]) == versioned[1]
    ]
    stable.sort(key=lambda item: covariance[index[item[0][0]], index[item[0][0]]])
    stable = stable[:target]
    if not stable:
        return ()
    fixed_indices = np.asarray([index[item[0][0]] for item in stable], dtype=np.int64)
    fixed_values = np.asarray([item[1] for item in stable], dtype=np.float64)
    missing_count = target - len(stable)
    available = [position for position in range(len(keys)) if position not in fixed_indices]
    available.sort(key=lambda position: covariance[position, position])
    missing_indices = np.asarray(available[:missing_count], dtype=np.int64)
    if missing_indices.size != missing_count:
        return ()
    fixed_innovation = fixed_values - ahat[fixed_indices]
    qff = covariance[np.ix_(fixed_indices, fixed_indices)]
    try:
        fixed_solved = np.linalg.solve(qff, fixed_innovation)
    except np.linalg.LinAlgError:
        return ()
    fixed_distance = float(fixed_innovation @ fixed_solved)
    if missing_count == 0:
        return ((dict(stable), fixed_distance),)
    qmf = covariance[np.ix_(missing_indices, fixed_indices)]
    qfm = covariance[np.ix_(fixed_indices, missing_indices)]
    qmm = covariance[np.ix_(missing_indices, missing_indices)]
    conditional_mean = ahat[missing_indices] + qmf @ fixed_solved
    try:
        conditional_covariance = qmm - qmf @ np.linalg.solve(qff, qfm)
        candidates, residuals = integer_search(
            conditional_mean,
            0.5 * (conditional_covariance + conditional_covariance.T),
            n_candidates=int(n_candidates),
        )
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return ()
    results = []
    for candidate, residual in zip(candidates, residuals):
        assignment = dict(stable)
        for position, value in zip(missing_indices, candidate):
            raw_key = keys[int(position)]
            assignment[(raw_key, int(generations[raw_key]))] = int(value)
        results.append((assignment, fixed_distance + float(residual)))
    return tuple(results)


class RecoveryPositionBank:
    """Short causal bank of spatially distinct motion-propagated basin positions."""

    def __init__(
        self,
        max_seeds: int,
        separation_m: float,
        max_age_epochs: int,
        selection_mode: str = "weight",
    ) -> None:
        self.max_seeds = int(max_seeds)
        self.separation_m = float(separation_m)
        self.max_age_epochs = int(max_age_epochs)
        self.selection_mode = str(selection_mode)
        if self.max_seeds < 1 or self.max_age_epochs < 1:
            raise ValueError("bank size and age must be positive")
        if not np.isfinite(self.separation_m) or self.separation_m <= 0.0:
            raise ValueError("bank separation must be finite and positive")
        if self.selection_mode not in {"weight", "farthest"}:
            raise ValueError("bank selection mode must be weight or farthest")
        self._entries: list[_BankEntry] = []

    @property
    def positions(self) -> tuple[np.ndarray, ...]:
        return tuple(entry.position_ecef.copy() for entry in self._entries)

    def update(
        self,
        epoch: int,
        positions_ecef: np.ndarray,
        log_weights: np.ndarray,
        *,
        velocities_ecef: np.ndarray | None = None,
        dt_seconds: float = 0.0,
        displacement_ecef_m: np.ndarray | None = None,
        reference_position_ecef: np.ndarray | None = None,
        max_reference_distance_m: float = np.inf,
    ) -> None:
        positions = np.asarray(positions_ecef, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(log_weights, dtype=np.float64).reshape(-1)
        if len(positions) != len(weights):
            raise ValueError("positions and log_weights must have equal length")
        velocities = (
            np.zeros_like(positions)
            if velocities_ecef is None
            else np.asarray(velocities_ecef, dtype=np.float64).reshape(-1, 3)
        )
        if len(velocities) != len(positions):
            raise ValueError("positions and velocities must have equal length")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(weights)):
            raise ValueError("bank inputs must be finite")
        if not np.all(np.isfinite(velocities)) or not np.isfinite(dt_seconds):
            raise ValueError("bank velocities and time step must be finite")
        displacement = (
            None
            if displacement_ecef_m is None
            else np.asarray(displacement_ecef_m, dtype=np.float64).reshape(3)
        )
        if displacement is not None and not np.all(np.isfinite(displacement)):
            raise ValueError("bank displacement must be finite")
        reference = (
            None
            if reference_position_ecef is None
            else np.asarray(reference_position_ecef, dtype=np.float64).reshape(3)
        )
        if reference is not None and not np.all(np.isfinite(reference)):
            raise ValueError("bank reference position must be finite")
        if np.isnan(max_reference_distance_m) or max_reference_distance_m <= 0.0:
            raise ValueError("bank reference distance must be positive")
        current = [
            _BankEntry(
                int(epoch), position.copy(), velocity.copy(), float(weight)
            )
            for position, velocity, weight in zip(positions, velocities, weights)
        ]
        retained = [
            _BankEntry(
                entry.epoch,
                entry.position_ecef
                + (
                    displacement
                    if displacement is not None
                    else entry.velocity_ecef * float(dt_seconds)
                ),
                entry.velocity_ecef,
                entry.log_weight,
            )
            for entry in self._entries
            if int(epoch) - entry.epoch <= self.max_age_epochs
        ]
        candidates = current + retained
        if reference is not None and np.isfinite(max_reference_distance_m):
            candidates = [
                entry
                for entry in candidates
                if np.linalg.norm(entry.position_ecef - reference)
                <= float(max_reference_distance_m)
            ]
        candidates.sort(key=lambda entry: (entry.log_weight, entry.epoch), reverse=True)
        selected: list[_BankEntry] = []
        remaining = list(candidates)
        while remaining and len(selected) < self.max_seeds:
            if not selected or self.selection_mode == "weight":
                entry = remaining.pop(0)
            else:
                distances = [
                    min(
                        np.linalg.norm(candidate.position_ecef - other.position_ecef)
                        for other in selected
                    )
                    for candidate in remaining
                ]
                best_index = max(
                    range(len(remaining)),
                    key=lambda index: (distances[index], remaining[index].log_weight),
                )
                entry = remaining.pop(best_index)
            if all(
                np.linalg.norm(entry.position_ecef - other.position_ecef)
                >= self.separation_m
                for other in selected
            ):
                selected.append(entry)
        self._entries = selected


def covariance_axis_position_seeds(
    center_ecef: np.ndarray,
    covariance_m2: np.ndarray,
    radii_m: Iterable[float],
    *,
    direction_mode: str = "axes",
) -> tuple[np.ndarray, ...]:
    """Return center and covariance-frame direction seeds at fixed radii."""

    center = np.asarray(center_ecef, dtype=np.float64).reshape(3)
    covariance = np.asarray(covariance_m2, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(covariance)):
        raise ValueError("center and covariance must be finite")
    covariance = 0.5 * (covariance + covariance.T)
    _values, vectors = np.linalg.eigh(covariance)
    radii = tuple(float(value) for value in radii_m)
    if any(not np.isfinite(value) or value <= 0.0 for value in radii):
        raise ValueError("proposal radii must be finite and positive")
    mode = str(direction_mode).strip().lower()
    if mode == "axes":
        directions = [
            sign * vectors[:, axis]
            for axis in range(3)
            for sign in (1.0, -1.0)
        ]
    elif mode == "cube26":
        directions = []
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, 0.0, 1.0):
                for z in (-1.0, 0.0, 1.0):
                    coefficients = np.asarray([x, y, z], dtype=np.float64)
                    norm = float(np.linalg.norm(coefficients))
                    if norm > 0.0:
                        directions.append(vectors @ (coefficients / norm))
    else:
        raise ValueError("direction_mode must be axes or cube26")
    seeds = [center.copy()]
    for radius in radii:
        for direction in directions:
            seeds.append(center + radius * direction)
    return tuple(seeds)
