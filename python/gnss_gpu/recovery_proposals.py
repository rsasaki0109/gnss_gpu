"""Deterministic position proposals for ambiguity-basin recovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _BankEntry:
    epoch: int
    position_ecef: np.ndarray
    log_weight: float


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
