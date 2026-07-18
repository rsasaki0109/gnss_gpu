"""Deterministic position proposals for ambiguity-basin recovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from gnss_gpu.lambda_ambiguity import (
    IntegerSearchWorkspace,
    integer_search,
    integer_search_prepared,
    prepare_integer_search,
)


RawAmbiguityKey: TypeAlias = tuple[str, str, int]
VersionedAmbiguityKey: TypeAlias = tuple[RawAmbiguityKey, int]
AssignmentItem: TypeAlias = tuple[VersionedAmbiguityKey, int]
SatelliteArcKey: TypeAlias = tuple[str, int]
VersionedSatelliteArcKey: TypeAlias = tuple[SatelliteArcKey, int]
ArcPotentialItem: TypeAlias = tuple[VersionedSatelliteArcKey, int]


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


@dataclass(frozen=True)
class _ArcAssignmentBankEntry:
    epoch: int
    potentials: tuple[ArcPotentialItem, ...]
    log_weight: float


class SatelliteArcTracker:
    """Track pivot-invariant per-satellite carrier continuity generations."""

    def __init__(self, slip_threshold_cycles: float, max_gap_epochs: int = 1) -> None:
        self.slip_threshold_cycles = float(slip_threshold_cycles)
        self.max_gap_epochs = int(max_gap_epochs)
        if not np.isfinite(self.slip_threshold_cycles) or self.slip_threshold_cycles <= 0:
            raise ValueError("arc slip threshold must be finite and positive")
        if self.max_gap_epochs < 1:
            raise ValueError("arc maximum gap must be positive")
        self._generations: dict[SatelliteArcKey, int] = {}
        self._potentials: dict[SatelliteArcKey, float] = {}
        self._last_seen: dict[SatelliteArcKey, int] = {}

    @property
    def generations(self) -> dict[SatelliteArcKey, int]:
        return dict(self._generations)

    def update(
        self,
        epoch: int,
        raw_keys: Iterable[RawAmbiguityKey],
        ambiguity_cycles: Iterable[float],
    ) -> tuple[SatelliteArcKey, ...]:
        """Update arcs from DD float values and return selectively reset satellites."""

        keys = tuple(raw_keys)
        values = np.asarray(tuple(ambiguity_cycles), dtype=np.float64).reshape(-1)
        if len(keys) != values.size:
            raise ValueError("arc keys and ambiguity values must have equal length")
        if not np.all(np.isfinite(values)):
            raise ValueError("arc ambiguity values must be finite")
        current: dict[SatelliteArcKey, float] = {}
        by_group: dict[tuple[str, int], list[tuple[str, str, float]]] = {}
        for (ref, sat, wavelength), value in zip(keys, values):
            group = (ref[:1], int(wavelength))
            by_group.setdefault(group, []).append((ref, sat, float(value)))
        for (constellation, wavelength), edges in by_group.items():
            adjacency: dict[str, list[tuple[str, float]]] = {}
            for ref, sat, value in edges:
                adjacency.setdefault(ref, []).append((sat, value))
                adjacency.setdefault(sat, []).append((ref, -value))
            potentials: dict[str, float] = {}
            for start in adjacency:
                if start in potentials:
                    continue
                potentials[start] = 0.0
                stack = [start]
                while stack:
                    node = stack.pop()
                    for neighbor, delta in adjacency[node]:
                        expected = potentials[node] + delta
                        if neighbor not in potentials:
                            potentials[neighbor] = expected
                            stack.append(neighbor)
                        elif abs(potentials[neighbor] - expected) > 0.25:
                            raise ValueError("inconsistent DD ambiguity graph")
            common_deltas = [
                potential - self._potentials[(satellite, wavelength)]
                for satellite, potential in potentials.items()
                if (satellite, wavelength) in self._potentials
                and int(epoch) - self._last_seen[(satellite, wavelength)]
                <= self.max_gap_epochs
            ]
            gauge_delta = float(np.median(common_deltas)) if common_deltas else 0.0
            for satellite, potential in potentials.items():
                current[(satellite, wavelength)] = float(potential - gauge_delta)

        slipped: list[SatelliteArcKey] = []
        for key, potential in current.items():
            previous = self._potentials.get(key)
            last_seen = self._last_seen.get(key)
            generation = self._generations.get(key, 0)
            gap_reset = (
                last_seen is not None
                and int(epoch) - int(last_seen) > self.max_gap_epochs
            )
            slip_reset = (
                previous is not None
                and not gap_reset
                and abs(float(potential) - float(previous))
                > self.slip_threshold_cycles
            )
            if gap_reset or slip_reset:
                generation += 1
                slipped.append(key)
            self._generations[key] = generation
            self._potentials[key] = float(potential)
            self._last_seen[key] = int(epoch)
        return tuple(sorted(slipped))


class RecoveryArcAssignmentBank:
    """Bounded assignment bank keyed by pivot-free per-satellite arc IDs."""

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
            raise ValueError("arc assignment bank size, age, and dimension must be positive")
        self._entries: list[_ArcAssignmentBankEntry] = []

    @staticmethod
    def _satellite_potentials(
        assignment: dict[VersionedAmbiguityKey, int],
        arc_generations: dict[SatelliteArcKey, int],
    ) -> tuple[ArcPotentialItem, ...] | None:
        graphs: dict[tuple[str, int], dict[str, list[tuple[str, int]]]] = {}
        for (raw_key, _generation), value in assignment.items():
            ref, sat, wavelength = raw_key
            group = (ref[:1], int(wavelength))
            adjacency = graphs.setdefault(group, {})
            adjacency.setdefault(ref, []).append((sat, int(value)))
            adjacency.setdefault(sat, []).append((ref, -int(value)))
        output: dict[VersionedSatelliteArcKey, int] = {}
        for (_constellation, wavelength), adjacency in graphs.items():
            potentials: dict[str, int] = {}
            for start in adjacency:
                if start in potentials:
                    continue
                potentials[start] = 0
                stack = [start]
                while stack:
                    node = stack.pop()
                    for neighbor, delta in adjacency[node]:
                        expected = potentials[node] + delta
                        if neighbor in potentials:
                            if potentials[neighbor] != expected:
                                return None
                        else:
                            potentials[neighbor] = expected
                            stack.append(neighbor)
            anchor = min(potentials)
            offset = potentials[anchor]
            for satellite, potential in potentials.items():
                arc_key = (satellite, wavelength)
                if arc_key not in arc_generations:
                    return None
                output[(arc_key, int(arc_generations[arc_key]))] = int(
                    potential - offset
                )
        return tuple(sorted(output.items()))

    def update(
        self,
        epoch: int,
        assignments: Iterable[dict[VersionedAmbiguityKey, int]],
        log_weights: Iterable[float],
        arc_generations: dict[SatelliteArcKey, int],
    ) -> None:
        assignment_list = list(assignments)
        weights = np.asarray(list(log_weights), dtype=np.float64).reshape(-1)
        if len(assignment_list) != len(weights):
            raise ValueError("arc assignments and log_weights must have equal length")
        if not np.all(np.isfinite(weights)):
            raise ValueError("arc assignment bank weights must be finite")
        candidates: list[_ArcAssignmentBankEntry] = []
        for assignment, weight in zip(assignment_list, weights):
            if len(assignment) < self.min_assignment_size:
                continue
            potentials = self._satellite_potentials(assignment, arc_generations)
            if potentials is not None:
                candidates.append(
                    _ArcAssignmentBankEntry(int(epoch), potentials, float(weight))
                )
        candidates.extend(
            entry
            for entry in self._entries
            if int(epoch) - entry.epoch <= self.max_age_epochs
        )
        candidates.sort(key=lambda entry: (entry.log_weight, entry.epoch), reverse=True)
        selected: dict[tuple[ArcPotentialItem, ...], _ArcAssignmentBankEntry] = {}
        for entry in candidates:
            selected.setdefault(entry.potentials, entry)
            if len(selected) >= self.max_assignments:
                break
        self._entries = list(selected.values())

    def compatible_assignments(
        self,
        active_versioned_keys: Iterable[VersionedAmbiguityKey],
        observed_raw_keys: Iterable[RawAmbiguityKey],
        arc_generations: dict[SatelliteArcKey, int],
        *,
        min_size: int | None = None,
    ) -> tuple[dict[VersionedAmbiguityKey, int], ...]:
        active_by_raw = {raw: generation for raw, generation in active_versioned_keys}
        observed = tuple(raw for raw in observed_raw_keys if raw in active_by_raw)
        minimum = self.min_assignment_size if min_size is None else int(min_size)
        if minimum < 1:
            raise ValueError("arc compatible assignment minimum must be positive")
        outputs: dict[tuple[AssignmentItem, ...], dict[VersionedAmbiguityKey, int]] = {}
        for entry in self._entries:
            potentials = dict(entry.potentials)
            projected: dict[VersionedAmbiguityKey, int] = {}
            for raw_key in observed:
                ref, sat, wavelength = raw_key
                ref_arc = (ref, int(wavelength))
                sat_arc = (sat, int(wavelength))
                if ref_arc not in arc_generations or sat_arc not in arc_generations:
                    continue
                ref_key = (ref_arc, int(arc_generations[ref_arc]))
                sat_key = (sat_arc, int(arc_generations[sat_arc]))
                if ref_key in potentials and sat_key in potentials:
                    projected[(raw_key, int(active_by_raw[raw_key]))] = int(
                        potentials[sat_key] - potentials[ref_key]
                    )
            if len(projected) >= minimum:
                canonical = tuple(sorted(projected.items()))
                outputs.setdefault(canonical, projected)
        return tuple(outputs.values())


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
    search_cache: dict[
        tuple[tuple[int, ...], tuple[int, ...]], IntegerSearchWorkspace
    ]
    | None = None,
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
    cache_key = (
        tuple(int(value) for value in fixed_indices),
        tuple(int(value) for value in missing_indices),
    )
    try:
        workspace = None if search_cache is None else search_cache.get(cache_key)
        if workspace is None:
            conditional_covariance = qmm - qmf @ np.linalg.solve(qff, qfm)
            workspace = prepare_integer_search(
                0.5 * (conditional_covariance + conditional_covariance.T)
            )
            if search_cache is not None:
                search_cache[cache_key] = workspace
        candidates, residuals = integer_search_prepared(
            conditional_mean,
            workspace,
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
