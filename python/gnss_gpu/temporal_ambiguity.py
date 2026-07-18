"""Multi-epoch posterior over ambiguity assignments without runtime FGO."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _logsumexp_axis0(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=0)
    return maximum + np.log(np.exp(values - maximum[None, :]).sum(axis=0))


@dataclass(frozen=True)
class TemporalAmbiguityConfig:
    birth_mass: float = 0.05
    assignment_change_cost: float = 2.0
    incompatible_cost: float = 12.0
    death_cost: float = 6.0
    motion_sigma_m: float = 3.0
    max_history_epochs: int = 100

    def __post_init__(self) -> None:
        if not 0.0 < self.birth_mass < 1.0:
            raise ValueError("birth_mass must be in (0, 1)")
        if (
            self.assignment_change_cost < 0.0
            or self.incompatible_cost < 0.0
            or self.death_cost < 0.0
        ):
            raise ValueError("assignment transition costs must be non-negative")
        if not math.isfinite(self.motion_sigma_m) or self.motion_sigma_m <= 0.0:
            raise ValueError("motion_sigma_m must be finite and positive")
        if self.max_history_epochs < 2:
            raise ValueError("max_history_epochs must be at least two")


@dataclass(frozen=True)
class TemporalAmbiguityCandidate:
    candidate_id: str
    assignment: tuple[object, ...]
    epoch_log_likelihood: float
    position_ecef: np.ndarray
    velocity_ecef: np.ndarray

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not math.isfinite(self.epoch_log_likelihood):
            raise ValueError("epoch_log_likelihood must be finite")
        position = np.asarray(self.position_ecef, dtype=np.float64).reshape(3).copy()
        velocity = np.asarray(self.velocity_ecef, dtype=np.float64).reshape(3).copy()
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("candidate position and velocity must be finite")
        object.__setattr__(self, "position_ecef", position)
        object.__setattr__(self, "velocity_ecef", velocity)


@dataclass(frozen=True)
class TemporalAmbiguityHypothesis:
    candidate_id: str
    log_probability: float
    ancestor_id: str | None
    dwell_epochs: int


@dataclass(frozen=True)
class TemporalAmbiguityPosterior:
    epoch: int
    map_candidate_id: str
    gamma: float
    ess: float
    dwell_epochs: int
    n_candidates: int


@dataclass(frozen=True)
class _Frame:
    epoch: int
    hypotheses: tuple[TemporalAmbiguityHypothesis, ...]


def _assignment_relation(previous: tuple[object, ...], current: tuple[object, ...]) -> tuple[bool, int]:
    prev = dict(previous)  # type: ignore[arg-type]
    curr = dict(current)  # type: ignore[arg-type]
    overlap = set(prev).intersection(curr)
    compatible = bool(overlap) and all(prev[key] == curr[key] for key in overlap)
    changes = len(set(prev).symmetric_difference(curr))
    return compatible, changes


class TemporalAmbiguityFilter:
    """Normalized forward filter with immediate ancestry and Viterbi backtrace."""

    def __init__(self, config: TemporalAmbiguityConfig | None = None) -> None:
        self.config = config or TemporalAmbiguityConfig()
        self._last_epoch: int | None = None
        self._candidates: dict[str, TemporalAmbiguityCandidate] = {}
        self._hypotheses: tuple[TemporalAmbiguityHypothesis, ...] = ()
        self._history: list[_Frame] = []
        self._relation_cache: dict[tuple[str, str], tuple[bool, int]] = {}

    @property
    def hypotheses(self) -> tuple[TemporalAmbiguityHypothesis, ...]:
        return self._hypotheses

    def step(
        self,
        epoch: int,
        dt: float,
        candidates: Iterable[TemporalAmbiguityCandidate],
    ) -> TemporalAmbiguityPosterior:
        current = list(candidates)
        if not current:
            self._last_epoch = int(epoch)
            self._candidates = {}
            self._hypotheses = ()
            self._history.append(_Frame(int(epoch), ()))
            self._trim_history()
            return TemporalAmbiguityPosterior(int(epoch), "", 0.0, 0.0, 0, 0)
        if len({candidate.candidate_id for candidate in current}) != len(current):
            raise ValueError("candidate IDs must be unique within an epoch")
        if self._last_epoch is not None and int(epoch) <= self._last_epoch:
            raise ValueError("temporal ambiguity epochs must be strictly increasing")
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")

        observation = np.asarray(
            [candidate.epoch_log_likelihood for candidate in current], dtype=np.float64
        )
        observation -= float(np.max(observation))
        ancestors: list[str | None] = [None] * len(current)
        dwell = np.ones(len(current), dtype=np.int64)

        if not self._hypotheses:
            scores = observation - math.log(len(current))
        else:
            previous = list(self._hypotheses)
            assignment_score = np.empty(
                (len(previous), len(current)), dtype=np.float64
            )
            motion_variance = self.config.motion_sigma_m**2
            for p_index, hypothesis in enumerate(previous):
                old = self._candidates[hypothesis.candidate_id]
                for c_index, candidate in enumerate(current):
                    pair = (old.candidate_id, candidate.candidate_id)
                    relation = self._relation_cache.get(pair)
                    if relation is None:
                        relation = _assignment_relation(
                            old.assignment, candidate.assignment
                        )
                        self._relation_cache[pair] = relation
                    compatible, changes = relation
                    assignment_score[p_index, c_index] = (
                        -self.config.assignment_change_cost * changes
                        if compatible else -self.config.incompatible_cost
                    )
            previous_position = np.asarray(
                [
                    self._candidates[item.candidate_id].position_ecef
                    + self._candidates[item.candidate_id].velocity_ecef * float(dt)
                    for item in previous
                ]
            )
            current_position = np.asarray(
                [candidate.position_ecef for candidate in current]
            )
            residual = current_position[None, :, :] - previous_position[:, None, :]
            motion_score = -0.5 * np.sum(residual * residual, axis=2) / motion_variance
            transition = assignment_score + motion_score
            for p_index in range(len(previous)):
                # Normalize with an explicit death/release state.  Without it,
                # a lineage that has no compatible successor is forced to
                # transfer all of its mass to an unrelated current candidate.
                row_normalizer = _logsumexp(
                    np.concatenate(
                        [transition[p_index], np.asarray([-self.config.death_cost])]
                    )
                )
                transition[p_index] -= row_normalizer

            previous_log = np.asarray(
                [hypothesis.log_probability for hypothesis in previous], dtype=np.float64
            )
            propagated = (
                previous_log[:, None]
                + math.log1p(-self.config.birth_mass)
                + transition
            )
            birth = math.log(self.config.birth_mass) - math.log(len(current))
            predicted_log = _logsumexp_axis0(
                np.vstack(
                    [propagated, np.full((1, len(current)), birth, dtype=np.float64)]
                )
            )
            for c_index, candidate in enumerate(current):
                winner = int(np.argmax(propagated[:, c_index]))
                if propagated[winner, c_index] < birth:
                    winner = len(previous)
                if winner < len(previous):
                    ancestor = previous[winner]
                    ancestors[c_index] = ancestor.candidate_id
                    if ancestor.candidate_id == candidate.candidate_id:
                        dwell[c_index] = ancestor.dwell_epochs + 1
            scores = predicted_log + observation

        scores -= _logsumexp(scores)
        self._hypotheses = tuple(
            TemporalAmbiguityHypothesis(
                candidate_id=candidate.candidate_id,
                log_probability=float(score),
                ancestor_id=ancestor,
                dwell_epochs=int(candidate_dwell),
            )
            for candidate, score, ancestor, candidate_dwell in zip(
                current, scores, ancestors, dwell
            )
        )
        self._candidates = {candidate.candidate_id: candidate for candidate in current}
        self._last_epoch = int(epoch)
        self._history.append(_Frame(int(epoch), self._hypotheses))
        self._trim_history()

        probabilities = np.exp(scores)
        map_index = int(np.argmax(probabilities))
        map_hypothesis = self._hypotheses[map_index]
        return TemporalAmbiguityPosterior(
            epoch=int(epoch),
            map_candidate_id=map_hypothesis.candidate_id,
            gamma=float(probabilities[map_index]),
            ess=float(1.0 / np.sum(probabilities**2)),
            dwell_epochs=map_hypothesis.dwell_epochs,
            n_candidates=len(current),
        )

    def viterbi_path(self, max_epochs: int | None = None) -> tuple[str, ...]:
        if not self._history or not self._history[-1].hypotheses:
            return ()
        limit = len(self._history) if max_epochs is None else max(1, int(max_epochs))
        frames = self._history[-limit:]
        current = max(frames[-1].hypotheses, key=lambda item: item.log_probability)
        path = [current.candidate_id]
        for frame in reversed(frames[:-1]):
            if current.ancestor_id is None:
                break
            matches = [
                hypothesis
                for hypothesis in frame.hypotheses
                if hypothesis.candidate_id == current.ancestor_id
            ]
            if not matches:
                break
            current = matches[0]
            path.append(current.candidate_id)
        return tuple(reversed(path))

    def _trim_history(self) -> None:
        excess = len(self._history) - self.config.max_history_epochs
        if excess > 0:
            del self._history[:excess]
