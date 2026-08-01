"""Fixed-lag genealogical FFBSi for ambiguity-basin particle histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gnss_gpu.ambiguity_basin_pf import AssignmentItem

if TYPE_CHECKING:
    from gnss_gpu.ambiguity_basin_pf import AmbiguityBasinParticleFilter


@dataclass(frozen=True)
class BasinHistoryNode:
    epoch: int
    tow_s: float
    basin_id: str
    parent_basin_id: str | None
    assignment: tuple[AssignmentItem, ...]
    mean: np.ndarray
    covariance: np.ndarray
    log_weight: float


@dataclass(frozen=True)
class BasinFFBSiEstimate:
    target_epoch: int
    target_tow_s: float
    position_ecef_m: np.ndarray
    position_covariance_m2: np.ndarray
    map_assignment: tuple[AssignmentItem, ...]
    assignment_probability: float
    effective_samples: int
    requested_samples: int


class FixedLagBasinFFBSi:
    """Retain bounded PF snapshots and sample backward through parent ids."""

    def __init__(self, lag_epochs: int, backward_samples: int = 128) -> None:
        self.lag_epochs = int(lag_epochs)
        self.backward_samples = int(backward_samples)
        if self.lag_epochs < 1 or self.backward_samples < 1:
            raise ValueError("lag_epochs and backward_samples must be positive")
        self._history: list[tuple[int, float, tuple[BasinHistoryNode, ...]]] = []

    def capture(self, particle_filter: AmbiguityBasinParticleFilter, tow_s: float) -> None:
        previous_ids = (
            {node.basin_id for node in self._history[-1][2]}
            if self._history
            else set()
        )
        nodes = tuple(
            BasinHistoryNode(
                epoch=int(particle_filter.epoch),
                tow_s=float(tow_s),
                basin_id=basin.basin_id,
                parent_basin_id=(
                    basin.basin_id
                    if basin.basin_id in previous_ids
                    else basin.parent_basin_id
                ),
                assignment=basin.assignment,
                mean=basin.conditional.mean.copy(),
                covariance=basin.conditional.covariance.copy(),
                log_weight=float(basin.log_weight),
            )
            for basin in particle_filter.basins
        )
        self._history.append((int(particle_filter.epoch), float(tow_s), nodes))
        keep = self.lag_epochs + 2
        if len(self._history) > keep:
            self._history = self._history[-keep:]

    @staticmethod
    def _weights(nodes: tuple[BasinHistoryNode, ...]) -> np.ndarray:
        log_weights = np.asarray([node.log_weight for node in nodes], dtype=np.float64)
        maximum = float(np.max(log_weights))
        weights = np.exp(log_weights - maximum)
        return weights / np.sum(weights)

    def estimate(self, *, seed: int = 0) -> BasinFFBSiEstimate | None:
        if len(self._history) <= self.lag_epochs:
            return None
        target_index = len(self._history) - 1 - self.lag_epochs
        target_epoch, target_tow, target_nodes = self._history[target_index]
        latest_nodes = self._history[-1][2]
        if not target_nodes or not latest_nodes:
            return None
        rng = np.random.default_rng(int(seed))
        latest_indices = rng.choice(
            len(latest_nodes),
            size=self.backward_samples,
            replace=True,
            p=self._weights(latest_nodes),
        )
        samples: list[BasinHistoryNode] = []
        snapshots = [entry[2] for entry in self._history]
        for latest_index in latest_indices:
            node = latest_nodes[int(latest_index)]
            valid = True
            for history_index in range(len(snapshots) - 2, target_index - 1, -1):
                parent_id = node.parent_basin_id
                parent = next(
                    (
                        candidate
                        for candidate in snapshots[history_index]
                        if candidate.basin_id == parent_id
                    ),
                    None,
                )
                if parent is None:
                    valid = False
                    break
                node = parent
            if valid and node.epoch == target_epoch:
                samples.append(node)
        if not samples:
            return None

        positions = np.asarray([sample.mean[:3] for sample in samples])
        mean = positions.mean(axis=0)
        covariance = np.zeros((3, 3), dtype=np.float64)
        assignment_count: dict[tuple[AssignmentItem, ...], int] = {}
        for sample, position in zip(samples, positions):
            delta = position - mean
            covariance += sample.covariance[:3, :3] + np.outer(delta, delta)
            assignment_count[sample.assignment] = (
                assignment_count.get(sample.assignment, 0) + 1
            )
        covariance /= len(samples)
        map_assignment, count = max(
            assignment_count.items(), key=lambda item: item[1]
        )
        return BasinFFBSiEstimate(
            target_epoch=target_epoch,
            target_tow_s=target_tow,
            position_ecef_m=mean,
            position_covariance_m2=0.5 * (covariance + covariance.T),
            map_assignment=map_assignment,
            assignment_probability=count / len(samples),
            effective_samples=len(samples),
            requested_samples=self.backward_samples,
        )


__all__ = [
    "BasinFFBSiEstimate",
    "BasinHistoryNode",
    "FixedLagBasinFFBSi",
]
