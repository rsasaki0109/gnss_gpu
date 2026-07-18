"""Deterministic position proposals for ambiguity-basin recovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


RawAmbiguityKey: TypeAlias = tuple[str, str, int]
VersionedAmbiguityKey: TypeAlias = tuple[RawAmbiguityKey, int]
AssignmentItem: TypeAlias = tuple[VersionedAmbiguityKey, int]


@dataclass(frozen=True)
class _BankEntry:
    epoch: int
    position_ecef: np.ndarray
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
    ) -> tuple[dict[VersionedAmbiguityKey, int], ...]:
        active = set(active_versioned_keys)
        observed = set(observed_raw_keys)
        compatible: dict[tuple[AssignmentItem, ...], dict[VersionedAmbiguityKey, int]] = {}
        for entry in self._entries:
            projected = tuple(
                item
                for item in entry.assignment
                if item[0] in active and item[0][0] in observed
            )
            if len(projected) >= self.min_assignment_size:
                compatible.setdefault(projected, dict(projected))
        return tuple(compatible.values())


class RecoveryPositionBank:
    """Short causal bank of spatially distinct motion-propagated basin positions."""

    def __init__(self, max_seeds: int, separation_m: float, max_age_epochs: int) -> None:
        self.max_seeds = int(max_seeds)
        self.separation_m = float(separation_m)
        self.max_age_epochs = int(max_age_epochs)
        if self.max_seeds < 1 or self.max_age_epochs < 1:
            raise ValueError("bank size and age must be positive")
        if not np.isfinite(self.separation_m) or self.separation_m <= 0.0:
            raise ValueError("bank separation must be finite and positive")
        self._entries: list[_BankEntry] = []

    @property
    def positions(self) -> tuple[np.ndarray, ...]:
        return tuple(entry.position_ecef.copy() for entry in self._entries)

    def update(
        self,
        epoch: int,
        positions_ecef: np.ndarray,
        log_weights: np.ndarray,
    ) -> None:
        positions = np.asarray(positions_ecef, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(log_weights, dtype=np.float64).reshape(-1)
        if len(positions) != len(weights):
            raise ValueError("positions and log_weights must have equal length")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(weights)):
            raise ValueError("bank inputs must be finite")
        current = [
            _BankEntry(int(epoch), position.copy(), float(weight))
            for position, weight in zip(positions, weights)
        ]
        retained = [
            entry
            for entry in self._entries
            if int(epoch) - entry.epoch <= self.max_age_epochs
        ]
        candidates = current + retained
        candidates.sort(key=lambda entry: (entry.log_weight, entry.epoch), reverse=True)
        selected: list[_BankEntry] = []
        for entry in candidates:
            if all(
                np.linalg.norm(entry.position_ecef - other.position_ecef)
                >= self.separation_m
                for other in selected
            ):
                selected.append(entry)
                if len(selected) >= self.max_seeds:
                    break
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
