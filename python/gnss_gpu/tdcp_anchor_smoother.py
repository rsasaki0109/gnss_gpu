"""Trusted-anchor Viterbi smoothing over RBPF position candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AnchorCandidateEpoch:
    epoch: int
    positions_ecef: np.ndarray
    log_weights: np.ndarray


@dataclass(frozen=True)
class ConstrainedViterbiAudit:
    """Per-candidate max-product evidence, normalized independently by epoch."""

    forward_relative: Mapping[int, np.ndarray]
    backward_relative: Mapping[int, np.ndarray]
    max_marginal_relative: Mapping[int, np.ndarray]


def constrained_viterbi_audit(
    candidate_epochs: Sequence[AnchorCandidateEpoch],
    interval_displacements_ecef: Mapping[tuple[int, int], np.ndarray],
    *,
    constrained_indices: Mapping[int, int],
    transition_sigma_m: float,
    emission_weight: float,
    transition_loss: str = "gaussian",
) -> ConstrainedViterbiAudit:
    """Compute forward/backward max-marginals for a constrained path graph."""

    epochs = tuple(candidate_epochs)
    if not epochs or not constrained_indices:
        raise ValueError("candidate epochs and constrained_indices must be non-empty")
    sigma = float(transition_sigma_m)
    weight = float(emission_weight)
    constraints = {int(epoch): int(index) for epoch, index in constrained_indices.items()}

    def constrain(scores: np.ndarray, item: AnchorCandidateEpoch) -> np.ndarray:
        selected = constraints.get(item.epoch)
        if selected is not None:
            scores = scores.copy()
            scores[np.arange(len(scores)) != selected] = -np.inf
        return scores

    def transition(previous: AnchorCandidateEpoch, current: AnchorCandidateEpoch) -> np.ndarray:
        displacement = np.asarray(
            interval_displacements_ecef[(previous.epoch, current.epoch)], dtype=np.float64
        ).reshape(3)
        predicted = np.asarray(previous.positions_ecef) + displacement
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef)[:, None, :] - predicted[None, :, :],
            axis=2,
        )
        return _transition_scores(distances, sigma, transition_loss)

    forward: list[np.ndarray] = [
        constrain(weight * np.asarray(epochs[0].log_weights, dtype=np.float64), epochs[0])
    ]
    for previous, current in zip(epochs[:-1], epochs[1:]):
        combined = transition(previous, current) + forward[-1][None, :]
        scores = weight * np.asarray(current.log_weights) + np.max(combined, axis=1)
        forward.append(constrain(scores, current))

    backward: list[np.ndarray] = [np.empty(0)] * len(epochs)
    backward[-1] = constrain(np.zeros(len(epochs[-1].positions_ecef)), epochs[-1])
    for index in range(len(epochs) - 2, -1, -1):
        previous, current = epochs[index], epochs[index + 1]
        next_emission = weight * np.asarray(current.log_weights, dtype=np.float64)
        combined = transition(previous, current) + (next_emission + backward[index + 1])[:, None]
        backward[index] = constrain(np.max(combined, axis=0), previous)

    def relative(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not np.any(finite):
            return values.copy()
        return values - float(np.max(values[finite]))

    return ConstrainedViterbiAudit(
        forward_relative={item.epoch: relative(values) for item, values in zip(epochs, forward)},
        backward_relative={item.epoch: relative(values) for item, values in zip(epochs, backward)},
        max_marginal_relative={
            item.epoch: relative(left + right)
            for item, left, right in zip(epochs, forward, backward)
        },
    )


def _transition_scores(
    distances_m: np.ndarray, sigma_m: float, loss: str
) -> np.ndarray:
    normalized = np.asarray(distances_m, dtype=np.float64) / sigma_m
    normalized_sq = np.square(normalized)
    if loss == "gaussian":
        return -0.5 * normalized_sq
    if loss == "huber":
        absolute = np.abs(normalized)
        return -np.where(absolute <= 1.0, 0.5 * normalized_sq, absolute - 0.5)
    if loss == "cauchy":
        return -np.log1p(normalized_sq)
    raise ValueError("transition_loss must be 'gaussian', 'huber', or 'cauchy'")


def anchored_viterbi_path(
    candidate_epochs: Sequence[AnchorCandidateEpoch],
    interval_displacements_ecef: Mapping[tuple[int, int], np.ndarray],
    *,
    anchor_epoch: int,
    anchor_index: int,
    transition_sigma_m: float,
    emission_weight: float,
    transition_loss: str = "gaussian",
) -> dict[int, int]:
    """Find the best candidate path forward and backward from a trusted anchor."""

    epochs = tuple(candidate_epochs)
    if not epochs:
        return {}
    epoch_values = [item.epoch for item in epochs]
    if epoch_values != sorted(epoch_values) or len(set(epoch_values)) != len(epoch_values):
        raise ValueError("candidate epochs must be unique and sorted")
    try:
        anchor_position = epoch_values.index(int(anchor_epoch))
    except ValueError as exc:
        raise ValueError("anchor_epoch is not present") from exc
    sigma = float(transition_sigma_m)
    weight = float(emission_weight)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("transition_sigma_m must be finite and positive")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("emission_weight must be finite and non-negative")
    for item in epochs:
        positions = np.asarray(item.positions_ecef, dtype=np.float64)
        log_weights = np.asarray(item.log_weights, dtype=np.float64)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] == 0
            or log_weights.shape != (positions.shape[0],)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(log_weights))
        ):
            raise ValueError("each candidate epoch must contain finite aligned candidates")
    if not 0 <= int(anchor_index) < len(epochs[anchor_position].positions_ecef):
        raise ValueError("anchor_index is out of range")

    path = {int(anchor_epoch): int(anchor_index)}

    def transition(previous: AnchorCandidateEpoch, current: AnchorCandidateEpoch) -> np.ndarray:
        key = (previous.epoch, current.epoch)
        displacement = np.asarray(interval_displacements_ecef[key], dtype=np.float64).reshape(3)
        predicted = np.asarray(previous.positions_ecef) + displacement
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef)[:, None, :] - predicted[None, :, :],
            axis=2,
        )
        return _transition_scores(distances, sigma, transition_loss)

    scores = np.full(len(epochs[anchor_position].positions_ecef), -np.inf)
    scores[int(anchor_index)] = 0.0
    backpointers: list[tuple[int, np.ndarray]] = []
    for index in range(anchor_position + 1, len(epochs)):
        previous, current = epochs[index - 1], epochs[index]
        combined = transition(previous, current) + scores[None, :]
        backpointer = np.argmax(combined, axis=1)
        scores = (
            combined[np.arange(len(current.positions_ecef)), backpointer]
            + weight * np.asarray(current.log_weights)
        )
        backpointers.append((current.epoch, backpointer))
    selected = int(np.argmax(scores))
    for epoch, backpointer in reversed(backpointers):
        path[epoch] = selected
        selected = int(backpointer[selected])

    scores = np.full(len(epochs[anchor_position].positions_ecef), -np.inf)
    scores[int(anchor_index)] = 0.0
    backpointers = []
    for index in range(anchor_position - 1, -1, -1):
        current, following = epochs[index], epochs[index + 1]
        reverse_displacement = -np.asarray(
            interval_displacements_ecef[(current.epoch, following.epoch)],
            dtype=np.float64,
        ).reshape(3)
        predicted = np.asarray(following.positions_ecef) + reverse_displacement
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef)[:, None, :] - predicted[None, :, :],
            axis=2,
        )
        combined = _transition_scores(distances, sigma, transition_loss) + scores[None, :]
        backpointer = np.argmax(combined, axis=1)
        scores = (
            combined[np.arange(len(current.positions_ecef)), backpointer]
            + weight * np.asarray(current.log_weights)
        )
        backpointers.append((current.epoch, backpointer))
    selected = int(np.argmax(scores))
    for epoch, backpointer in reversed(backpointers):
        path[epoch] = selected
        selected = int(backpointer[selected])
    return path


def constrained_viterbi_path(
    candidate_epochs: Sequence[AnchorCandidateEpoch],
    interval_displacements_ecef: Mapping[tuple[int, int], np.ndarray],
    *,
    constrained_indices: Mapping[int, int],
    transition_sigma_m: float,
    emission_weight: float,
    transition_loss: str = "gaussian",
) -> dict[int, int]:
    """Find one global Viterbi path subject to any number of trusted anchors."""

    epochs = tuple(candidate_epochs)
    if not epochs or not constrained_indices:
        raise ValueError("candidate epochs and constrained_indices must be non-empty")
    epoch_values = [item.epoch for item in epochs]
    if epoch_values != sorted(epoch_values) or len(set(epoch_values)) != len(epoch_values):
        raise ValueError("candidate epochs must be unique and sorted")
    sigma = float(transition_sigma_m)
    weight = float(emission_weight)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("transition_sigma_m must be finite and positive")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("emission_weight must be finite and non-negative")
    constraints = {int(epoch): int(index) for epoch, index in constrained_indices.items()}
    if not set(constraints).issubset(epoch_values):
        raise ValueError("every constrained epoch must be present")
    for item in epochs:
        positions = np.asarray(item.positions_ecef, dtype=np.float64)
        log_weights = np.asarray(item.log_weights, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or log_weights.shape != (len(positions),):
            raise ValueError("candidate positions and log weights must be aligned")
        constrained = constraints.get(item.epoch)
        if constrained is not None and not 0 <= constrained < len(positions):
            raise ValueError("constrained candidate index is out of range")

    scores = weight * np.asarray(epochs[0].log_weights, dtype=np.float64)
    if epochs[0].epoch in constraints:
        selected = constraints[epochs[0].epoch]
        scores[np.arange(len(scores)) != selected] = -np.inf
    backpointers: list[np.ndarray] = []
    for previous, current in zip(epochs[:-1], epochs[1:]):
        displacement = np.asarray(
            interval_displacements_ecef[(previous.epoch, current.epoch)],
            dtype=np.float64,
        ).reshape(3)
        predicted = np.asarray(previous.positions_ecef) + displacement
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef)[:, None, :] - predicted[None, :, :],
            axis=2,
        )
        combined = _transition_scores(distances, sigma, transition_loss) + scores[None, :]
        backpointer = np.argmax(combined, axis=1)
        scores = (
            combined[np.arange(len(current.positions_ecef)), backpointer]
            + weight * np.asarray(current.log_weights)
        )
        if current.epoch in constraints:
            selected = constraints[current.epoch]
            scores[np.arange(len(scores)) != selected] = -np.inf
        backpointers.append(backpointer)
    selected = int(np.argmax(scores))
    path = {epochs[-1].epoch: selected}
    for item, backpointer in zip(reversed(epochs[:-1]), reversed(backpointers)):
        selected = int(backpointer[selected])
        path[item.epoch] = selected
    return path


def constrained_greedy_path(
    candidate_epochs: Sequence[AnchorCandidateEpoch],
    interval_displacements_ecef: Mapping[tuple[int, int], np.ndarray],
    *,
    constrained_indices: Mapping[int, int],
    transition_sigma_m: float,
    emission_weight: float,
    transition_loss: str = "gaussian",
) -> dict[int, int]:
    """Select causally from the earliest anchor without future-path backpressure."""

    epochs = tuple(candidate_epochs)
    if not epochs or not constrained_indices:
        raise ValueError("candidate epochs and constrained_indices must be non-empty")
    epoch_values = [item.epoch for item in epochs]
    constraints = {int(epoch): int(index) for epoch, index in constrained_indices.items()}
    if not set(constraints).issubset(epoch_values):
        raise ValueError("every constrained epoch must be present")
    anchor_position = min(
        range(len(epochs)), key=lambda index: epoch_values[index] if epoch_values[index] in constraints else np.inf
    )
    if epoch_values[anchor_position] not in constraints:
        raise ValueError("no constrained epoch is present")
    sigma = float(transition_sigma_m)
    weight = float(emission_weight)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("transition_sigma_m must be finite and positive")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("emission_weight must be finite and non-negative")

    path = {epoch_values[anchor_position]: constraints[epoch_values[anchor_position]]}
    for index in range(anchor_position + 1, len(epochs)):
        previous, current = epochs[index - 1], epochs[index]
        constrained = constraints.get(current.epoch)
        if constrained is not None:
            path[current.epoch] = constrained
            continue
        displacement = np.asarray(
            interval_displacements_ecef[(previous.epoch, current.epoch)], dtype=np.float64
        ).reshape(3)
        predicted = np.asarray(previous.positions_ecef[path[previous.epoch]]) + displacement
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef) - predicted.reshape(1, 3), axis=1
        )
        scores = _transition_scores(distances, sigma, transition_loss) + weight * np.asarray(
            current.log_weights
        )
        path[current.epoch] = int(np.argmax(scores))

    for index in range(anchor_position - 1, -1, -1):
        current, following = epochs[index], epochs[index + 1]
        constrained = constraints.get(current.epoch)
        if constrained is not None:
            path[current.epoch] = constrained
            continue
        displacement = np.asarray(
            interval_displacements_ecef[(current.epoch, following.epoch)], dtype=np.float64
        ).reshape(3)
        predicted = np.asarray(following.positions_ecef[path[following.epoch]]) - displacement
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef) - predicted.reshape(1, 3), axis=1
        )
        scores = _transition_scores(distances, sigma, transition_loss) + weight * np.asarray(
            current.log_weights
        )
        path[current.epoch] = int(np.argmax(scores))
    return path


def _assignment_continuity_score(
    previous: Mapping[tuple[str, str, int], int],
    current: Mapping[tuple[str, str, int], int],
    *,
    match_bonus: float,
    conflict_penalty: float,
) -> float:
    shared = set(previous) & set(current)
    if not shared:
        return 0.0
    matches = sum(previous[key] == current[key] for key in shared)
    conflicts = len(shared) - matches
    return float(match_bonus) * matches - float(conflict_penalty) * conflicts


def constrained_assignment_viterbi_path(
    candidate_epochs: Sequence[AnchorCandidateEpoch],
    interval_displacements_ecef: Mapping[tuple[int, int], np.ndarray],
    candidate_assignments: Mapping[
        int, Sequence[Mapping[tuple[str, str, int], int]]
    ],
    *,
    constrained_indices: Mapping[int, int],
    transition_sigma_m: float,
    emission_weight: float,
    transition_loss: str = "gaussian",
    assignment_match_bonus: float = 2.0,
    assignment_conflict_penalty: float = 4.0,
) -> dict[int, int]:
    """Global motion/weight Viterbi path with integer-assignment continuity."""

    epochs = tuple(candidate_epochs)
    if not epochs or not constrained_indices:
        raise ValueError("candidate epochs and constrained_indices must be non-empty")
    epoch_values = [item.epoch for item in epochs]
    if epoch_values != sorted(epoch_values) or len(set(epoch_values)) != len(epoch_values):
        raise ValueError("candidate epochs must be unique and sorted")
    sigma = float(transition_sigma_m)
    weight = float(emission_weight)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("transition_sigma_m must be finite and positive")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("emission_weight must be finite and non-negative")
    constraints = {int(epoch): int(index) for epoch, index in constrained_indices.items()}
    if not set(constraints).issubset(epoch_values):
        raise ValueError("every constrained epoch must be present")
    for item in epochs:
        assignments = candidate_assignments.get(item.epoch)
        if assignments is None or len(assignments) != len(item.positions_ecef):
            raise ValueError("candidate assignments must align with every epoch")
        constrained = constraints.get(item.epoch)
        if constrained is not None and not 0 <= constrained < len(item.positions_ecef):
            raise ValueError("constrained candidate index is out of range")

    scores = weight * np.asarray(epochs[0].log_weights, dtype=np.float64)
    if epochs[0].epoch in constraints:
        selected = constraints[epochs[0].epoch]
        scores[np.arange(len(scores)) != selected] = -np.inf
    backpointers: list[np.ndarray] = []
    for previous, current in zip(epochs[:-1], epochs[1:]):
        displacement = np.asarray(
            interval_displacements_ecef[(previous.epoch, current.epoch)],
            dtype=np.float64,
        ).reshape(3)
        predicted = np.asarray(previous.positions_ecef) + displacement
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef)[:, None, :] - predicted[None, :, :],
            axis=2,
        )
        continuity = np.asarray(
            [
                [
                    _assignment_continuity_score(
                        previous_assignment,
                        current_assignment,
                        match_bonus=assignment_match_bonus,
                        conflict_penalty=assignment_conflict_penalty,
                    )
                    for previous_assignment in candidate_assignments[previous.epoch]
                ]
                for current_assignment in candidate_assignments[current.epoch]
            ],
            dtype=np.float64,
        )
        combined = (
            _transition_scores(distances, sigma, transition_loss)
            + continuity
            + scores[None, :]
        )
        backpointer = np.argmax(combined, axis=1)
        scores = (
            combined[np.arange(len(current.positions_ecef)), backpointer]
            + weight * np.asarray(current.log_weights)
        )
        if current.epoch in constraints:
            selected = constraints[current.epoch]
            scores[np.arange(len(scores)) != selected] = -np.inf
        backpointers.append(backpointer)
    selected = int(np.argmax(scores))
    path = {epochs[-1].epoch: selected}
    for item, backpointer in zip(reversed(epochs[:-1]), reversed(backpointers)):
        selected = int(backpointer[selected])
        path[item.epoch] = selected
    return path


def constrained_assignment_viterbi_audit(
    candidate_epochs: Sequence[AnchorCandidateEpoch],
    interval_displacements_ecef: Mapping[tuple[int, int], np.ndarray],
    candidate_assignments: Mapping[
        int, Sequence[Mapping[tuple[str, str, int], int]]
    ],
    *,
    constrained_indices: Mapping[int, int],
    transition_sigma_m: float,
    emission_weight: float,
    transition_loss: str = "gaussian",
    assignment_match_bonus: float = 2.0,
    assignment_conflict_penalty: float = 4.0,
) -> ConstrainedViterbiAudit:
    """Compute assignment-aware forward/backward max-marginals."""

    epochs = tuple(candidate_epochs)
    if not epochs or not constrained_indices:
        raise ValueError("candidate epochs and constrained_indices must be non-empty")
    sigma = float(transition_sigma_m)
    weight = float(emission_weight)
    constraints = {int(epoch): int(index) for epoch, index in constrained_indices.items()}

    def constrain(scores: np.ndarray, item: AnchorCandidateEpoch) -> np.ndarray:
        selected = constraints.get(item.epoch)
        if selected is not None:
            scores = scores.copy()
            scores[np.arange(len(scores)) != selected] = -np.inf
        return scores

    def transition(previous: AnchorCandidateEpoch, current: AnchorCandidateEpoch) -> np.ndarray:
        displacement = np.asarray(
            interval_displacements_ecef[(previous.epoch, current.epoch)],
            dtype=np.float64,
        ).reshape(3)
        predicted = np.asarray(previous.positions_ecef) + displacement
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef)[:, None, :] - predicted[None, :, :],
            axis=2,
        )
        continuity = np.asarray(
            [
                [
                    _assignment_continuity_score(
                        previous_assignment,
                        current_assignment,
                        match_bonus=assignment_match_bonus,
                        conflict_penalty=assignment_conflict_penalty,
                    )
                    for previous_assignment in candidate_assignments[previous.epoch]
                ]
                for current_assignment in candidate_assignments[current.epoch]
            ],
            dtype=np.float64,
        )
        return _transition_scores(distances, sigma, transition_loss) + continuity

    forward: list[np.ndarray] = [
        constrain(weight * np.asarray(epochs[0].log_weights), epochs[0])
    ]
    for previous, current in zip(epochs[:-1], epochs[1:]):
        combined = transition(previous, current) + forward[-1][None, :]
        forward.append(
            constrain(
                weight * np.asarray(current.log_weights) + np.max(combined, axis=1),
                current,
            )
        )
    backward: list[np.ndarray] = [np.empty(0)] * len(epochs)
    backward[-1] = constrain(np.zeros(len(epochs[-1].positions_ecef)), epochs[-1])
    for index in range(len(epochs) - 2, -1, -1):
        previous, current = epochs[index], epochs[index + 1]
        combined = transition(previous, current) + (
            weight * np.asarray(current.log_weights) + backward[index + 1]
        )[:, None]
        backward[index] = constrain(np.max(combined, axis=0), previous)

    def relative(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        return values.copy() if not np.any(finite) else values - np.max(values[finite])

    return ConstrainedViterbiAudit(
        forward_relative={item.epoch: relative(value) for item, value in zip(epochs, forward)},
        backward_relative={item.epoch: relative(value) for item, value in zip(epochs, backward)},
        max_marginal_relative={
            item.epoch: relative(left + right)
            for item, left, right in zip(epochs, forward, backward)
        },
    )


def constrained_assignment_greedy_path(
    candidate_epochs: Sequence[AnchorCandidateEpoch],
    interval_displacements_ecef: Mapping[tuple[int, int], np.ndarray],
    candidate_assignments: Mapping[
        int, Sequence[Mapping[tuple[str, str, int], int]]
    ],
    *,
    constrained_indices: Mapping[int, int],
    transition_sigma_m: float,
    emission_weight: float,
    transition_loss: str = "gaussian",
    assignment_match_bonus: float = 2.0,
    assignment_conflict_penalty: float = 4.0,
    candidate_reacquisition_flags: Mapping[int, Sequence[bool]] | None = None,
    reacquisition_min_exact_pairs: int = 4,
    reacquisition_min_stable_anchors: int = 10,
    reacquisition_window_anchors: int = 0,
    reacquisition_ignore_assignment: bool = False,
    reacquisition_dead_reckon: bool = False,
) -> dict[int, int]:
    """Causal motion path with persistent integer-assignment identity evidence."""

    epochs = tuple(candidate_epochs)
    if not epochs or not constrained_indices:
        raise ValueError("candidate epochs and constrained_indices must be non-empty")
    if not np.isfinite(transition_sigma_m) or transition_sigma_m <= 0.0:
        raise ValueError("transition_sigma_m must be finite and positive")
    if not np.isfinite(emission_weight) or emission_weight < 0.0:
        raise ValueError("emission_weight must be finite and non-negative")
    if not np.isfinite(assignment_match_bonus) or assignment_match_bonus < 0.0:
        raise ValueError("assignment_match_bonus must be finite and non-negative")
    if not np.isfinite(assignment_conflict_penalty) or assignment_conflict_penalty < 0.0:
        raise ValueError(
            "assignment_conflict_penalty must be finite and non-negative"
        )
    if int(reacquisition_min_exact_pairs) < 1:
        raise ValueError("reacquisition_min_exact_pairs must be positive")
    if int(reacquisition_min_stable_anchors) < 1:
        raise ValueError("reacquisition_min_stable_anchors must be positive")
    if int(reacquisition_window_anchors) < 0:
        raise ValueError("reacquisition_window_anchors must be non-negative")
    epoch_values = [item.epoch for item in epochs]
    constraints = {int(epoch): int(index) for epoch, index in constrained_indices.items()}
    if not set(constraints).issubset(epoch_values):
        raise ValueError("every constrained epoch must be present")
    anchor_position = min(
        range(len(epochs)),
        key=lambda index: epoch_values[index]
        if epoch_values[index] in constraints
        else np.inf,
    )
    if epoch_values[anchor_position] not in constraints:
        raise ValueError("no constrained epoch is present")
    for item in epochs:
        assignments = candidate_assignments.get(item.epoch)
        if assignments is None or len(assignments) != len(item.positions_ecef):
            raise ValueError("candidate assignments must align with every epoch")
        if candidate_reacquisition_flags is not None:
            flags = candidate_reacquisition_flags.get(item.epoch)
            if flags is None or len(flags) != len(item.positions_ecef):
                raise ValueError("reacquisition flags must align with every epoch")
        constrained = constraints.get(item.epoch)
        if constrained is not None and not 0 <= constrained < len(item.positions_ecef):
            raise ValueError("constrained candidate index is out of range")

    path = {epoch_values[anchor_position]: constraints[epoch_values[anchor_position]]}

    def choose(
        previous_index: int,
        current_index: int,
        reverse: bool,
        stable_anchors: int,
        force_reacquisition: bool = False,
        predicted_override: np.ndarray | None = None,
    ) -> tuple[int, bool, np.ndarray]:
        previous = epochs[previous_index]
        current = epochs[current_index]
        if reverse:
            displacement = -np.asarray(
                interval_displacements_ecef[(current.epoch, previous.epoch)],
                dtype=np.float64,
            ).reshape(3)
        else:
            displacement = np.asarray(
                interval_displacements_ecef[(previous.epoch, current.epoch)],
                dtype=np.float64,
            ).reshape(3)
        selected_previous = path[previous.epoch]
        predicted = (
            np.asarray(previous.positions_ecef[selected_previous]) + displacement
            if predicted_override is None
            else np.asarray(predicted_override, dtype=np.float64).reshape(3)
        )
        distances = np.linalg.norm(
            np.asarray(current.positions_ecef) - predicted.reshape(1, 3), axis=1
        )
        previous_assignment = candidate_assignments[previous.epoch][selected_previous]
        overlap_counts: list[tuple[int, int]] = []
        for assignment in candidate_assignments[current.epoch]:
            shared = set(previous_assignment) & set(assignment)
            matches = sum(
                previous_assignment[key] == assignment[key] for key in shared
            )
            overlap_counts.append((matches, len(shared) - matches))
        continuity = np.asarray(
            [
                _assignment_continuity_score(
                    previous_assignment,
                    assignment,
                    match_bonus=assignment_match_bonus,
                    conflict_penalty=assignment_conflict_penalty,
                )
                for assignment in candidate_assignments[current.epoch]
            ],
            dtype=np.float64,
        )
        motion_scores = (
            _transition_scores(distances, transition_sigma_m, transition_loss)
            + float(emission_weight) * np.asarray(current.log_weights)
        )
        scores = motion_scores + continuity
        triggered_reacquisition = False
        if candidate_reacquisition_flags is not None:
            has_exact_continuation = any(
                matches >= int(reacquisition_min_exact_pairs) and conflicts == 0
                for matches, conflicts in overlap_counts
            )
            flags = np.asarray(
                candidate_reacquisition_flags[current.epoch], dtype=bool
            )
            triggered_reacquisition = bool(
                stable_anchors >= int(reacquisition_min_stable_anchors)
                and not has_exact_continuation
                and np.any(flags)
            )
            in_reacquisition = force_reacquisition or triggered_reacquisition
            if in_reacquisition and reacquisition_ignore_assignment:
                scores = motion_scores
            if in_reacquisition and np.any(flags):
                scores = np.where(flags, scores, -np.inf)
        return int(np.argmax(scores)), triggered_reacquisition, predicted

    stable_anchors = 1
    reacquisition_remaining = 0
    recovery_prediction: np.ndarray | None = None
    for index in range(anchor_position + 1, len(epochs)):
        current = epochs[index]
        constrained = constraints.get(current.epoch)
        triggered_reacquisition = False
        if constrained is not None:
            path[current.epoch] = constrained
        else:
            if (
                reacquisition_dead_reckon
                and reacquisition_remaining > 0
                and recovery_prediction is not None
            ):
                previous = epochs[index - 1]
                recovery_prediction = recovery_prediction + np.asarray(
                    interval_displacements_ecef[(previous.epoch, current.epoch)],
                    dtype=np.float64,
                ).reshape(3)
            selected, triggered_reacquisition, predicted = choose(
                index - 1,
                index,
                False,
                stable_anchors,
                reacquisition_remaining > 0,
                recovery_prediction,
            )
            path[current.epoch] = selected
        if triggered_reacquisition:
            reacquisition_remaining = int(reacquisition_window_anchors)
            recovery_prediction = predicted.copy() if reacquisition_dead_reckon else None
        elif reacquisition_remaining > 0:
            reacquisition_remaining -= 1
            if reacquisition_remaining == 0:
                recovery_prediction = None
        previous = epochs[index - 1]
        previous_assignment = candidate_assignments[previous.epoch][path[previous.epoch]]
        current_assignment = candidate_assignments[current.epoch][path[current.epoch]]
        stable_anchors = (
            stable_anchors + 1
            if previous_assignment == current_assignment
            else 1
        )
    for index in range(anchor_position - 1, -1, -1):
        current = epochs[index]
        constrained = constraints.get(current.epoch)
        path[current.epoch] = (
            constrained
            if constrained is not None
            else choose(index + 1, index, True, 1)[0]
        )
    return path


def interpolate_path_position(
    epoch: int,
    anchor_positions: Mapping[int, np.ndarray],
) -> np.ndarray | None:
    """Linearly interpolate between the surrounding selected anchor positions."""

    if not anchor_positions:
        return None
    epochs = sorted(anchor_positions)
    index = int(np.searchsorted(epochs, int(epoch)))
    if index < len(epochs) and epochs[index] == int(epoch):
        return np.asarray(anchor_positions[epochs[index]], dtype=np.float64).copy()
    if index == 0 or index == len(epochs):
        return None
    left, right = epochs[index - 1], epochs[index]
    fraction = (int(epoch) - left) / float(right - left)
    return (
        (1.0 - fraction) * np.asarray(anchor_positions[left], dtype=np.float64)
        + fraction * np.asarray(anchor_positions[right], dtype=np.float64)
    )


__all__ = [
    "AnchorCandidateEpoch",
    "anchored_viterbi_path",
    "constrained_viterbi_path",
    "constrained_greedy_path",
    "constrained_assignment_greedy_path",
    "interpolate_path_position",
]
