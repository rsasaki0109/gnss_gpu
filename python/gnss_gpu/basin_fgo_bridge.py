"""Common-normal conditional FGO evaluation for ambiguity-basin particles.

This module is the CPU reference contract for the PF/FGO bridge.  A fixed-lag
FGO is linearized once, then discrete hypotheses which constrain the same
state columns are evaluated by one Cholesky factorization and a multi-column
solve.  The native CUDA implementation can follow this contract without
changing estimator semantics.

For a base quadratic

    0.5 * dx.T @ H @ dx - g.T @ dx

and a hypothesis residual ``A @ dx - b``, the conditioned system is

    Hc = H + A.T @ W @ A
    gc = g + A.T @ W @ b.

Hypotheses with an identical ``A`` and ``W`` share ``Hc``.  Their relative
Laplace log evidence includes both the minimized quadratic and ``log|Hc|``.
The additive base-graph constant is common to every hypothesis and omitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from gnss_gpu.ambiguity_basin_pf import AmbiguityBasinParticleFilter


@dataclass(frozen=True)
class ConditionedFGOHypothesis:
    """One discrete basin expressed as linear constraints at a shared state."""

    basin_id: str
    constraint_matrix: np.ndarray
    target_residual: np.ndarray
    sigma: float | np.ndarray


@dataclass(frozen=True)
class ConditionedFGOResult:
    """Conditional continuous estimate and evidence for one basin."""

    basin_id: str
    success: bool
    state: np.ndarray | None
    delta: np.ndarray | None
    covariance: np.ndarray | None
    relative_log_evidence: float
    minimized_quadratic: float
    log_determinant: float
    failure_reason: str
    factorization_group: int


@dataclass(frozen=True)
class ConditionedFGOBatchResult:
    """Results and execution diagnostics for a common-normal batch."""

    hypotheses: tuple[ConditionedFGOResult, ...]
    factorization_count: int
    rhs_columns: int
    state_size: int

    @property
    def all_succeeded(self) -> bool:
        return bool(self.hypotheses) and all(item.success for item in self.hypotheses)


@dataclass(frozen=True)
class NativeFGOHypothesisCandidate:
    """Truth-free native MultiSD hypothesis ready to become a basin birth."""

    group_index: int
    rank: int
    assignment: Mapping[tuple[tuple[str, str, int], int], int]
    position_ecef_m: np.ndarray
    position_covariance_m2: np.ndarray
    velocity_ecef_mps: np.ndarray | None
    relative_log_evidence: float
    incremental_likelihood_rows: int
    source_id: str
    validation_pass: bool


@dataclass(frozen=True)
class NativeFGOTransitionResult:
    parent_child_branches: int
    resulting_basins: int
    minimum_conflicts: int
    maximum_conflicts: int


def _as_finite_symmetric_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return 0.5 * (matrix + matrix.T)


def _prepare_hypothesis(
    hypothesis: ConditionedFGOHypothesis,
    state_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hypothesis.basin_id:
        raise ValueError("basin_id must not be empty")
    design = np.asarray(hypothesis.constraint_matrix, dtype=np.float64)
    target = np.asarray(hypothesis.target_residual, dtype=np.float64).reshape(-1)
    if design.ndim != 2 or design.shape != (target.size, state_size):
        raise ValueError(
            "constraint_matrix must have shape (len(target_residual), state_size)"
        )
    sigma = np.broadcast_to(
        np.asarray(hypothesis.sigma, dtype=np.float64), target.shape
    ).copy()
    if (
        target.size == 0
        or not np.all(np.isfinite(design))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(sigma))
        or np.any(sigma <= 0.0)
    ):
        raise ValueError("hypothesis constraints and positive sigmas must be finite")
    precision = 1.0 / np.square(sigma)
    return design, target, precision


def _pattern_key(design: np.ndarray, precision: np.ndarray) -> tuple[object, ...]:
    # C-contiguous byte keys make grouping deterministic and require exact
    # equality.  Near-equal patterns must not accidentally share a normal
    # matrix; the native path uses the same fail-closed rule on column indices.
    design = np.ascontiguousarray(design)
    precision = np.ascontiguousarray(precision)
    return (design.shape, design.tobytes(), precision.shape, precision.tobytes())


def evaluate_conditioned_fgo_batch(
    normal_matrix: np.ndarray,
    normal_rhs: np.ndarray,
    linearization_state: np.ndarray,
    hypotheses: Iterable[ConditionedFGOHypothesis],
    *,
    damping: float = 0.0,
) -> ConditionedFGOBatchResult:
    """Evaluate discrete hypotheses with one factorization per constraint pattern.

    A failed pattern produces explicit failed results for every member of that
    group.  Other patterns may still be diagnostic successes, but callers that
    mutate a live PF should use :func:`apply_conditioned_batch` which requires
    complete success and therefore fails closed.
    """

    normal = _as_finite_symmetric_matrix(normal_matrix, "normal_matrix")
    state_size = int(normal.shape[0])
    rhs = np.asarray(normal_rhs, dtype=np.float64).reshape(-1)
    state0 = np.asarray(linearization_state, dtype=np.float64).reshape(-1)
    if rhs.size != state_size or state0.size != state_size:
        raise ValueError("normal_rhs and linearization_state must match normal_matrix")
    if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(state0)):
        raise ValueError("normal_rhs and linearization_state must be finite")
    if not np.isfinite(damping) or float(damping) < 0.0:
        raise ValueError("damping must be finite and non-negative")

    items = list(hypotheses)
    if len({item.basin_id for item in items}) != len(items):
        raise ValueError("basin_id values must be unique within a batch")
    prepared = [_prepare_hypothesis(item, state_size) for item in items]
    grouped: dict[tuple[object, ...], list[int]] = {}
    for index, (design, _target, precision) in enumerate(prepared):
        grouped.setdefault(_pattern_key(design, precision), []).append(index)

    results: list[ConditionedFGOResult | None] = [None] * len(items)
    factorization_count = 0
    rhs_columns = 0
    for group_index, indices in enumerate(grouped.values()):
        design, _target, precision = prepared[indices[0]]
        conditioned_normal = normal + (design.T * precision) @ design
        if damping:
            conditioned_normal = conditioned_normal.copy()
            conditioned_normal.flat[:: state_size + 1] += float(damping)
        factorization_count += 1
        try:
            chol = np.linalg.cholesky(conditioned_normal)
            log_determinant = 2.0 * float(np.sum(np.log(np.diag(chol))))
            identity = np.eye(state_size, dtype=np.float64)
            covariance = np.linalg.solve(chol.T, np.linalg.solve(chol, identity))
        except np.linalg.LinAlgError:
            for index in indices:
                results[index] = ConditionedFGOResult(
                    basin_id=items[index].basin_id,
                    success=False,
                    state=None,
                    delta=None,
                    covariance=None,
                    relative_log_evidence=-np.inf,
                    minimized_quadratic=np.inf,
                    log_determinant=np.inf,
                    failure_reason="non_positive_definite_conditioned_normal",
                    factorization_group=group_index,
                )
            continue

        columns = np.column_stack(
            [
                rhs + design.T @ (precision * prepared[index][1])
                for index in indices
            ]
        )
        rhs_columns += int(columns.shape[1])
        deltas = np.linalg.solve(chol.T, np.linalg.solve(chol, columns))
        for column, index in enumerate(indices):
            target = prepared[index][1]
            delta = deltas[:, column]
            # This is the candidate-dependent part of the optimized quadratic.
            # The base residual constant is shared and deliberately omitted.
            constant = 0.5 * float(np.sum(precision * np.square(target)))
            minimized = constant - 0.5 * float(columns[:, column] @ delta)
            log_evidence = -minimized - 0.5 * log_determinant
            finite = bool(
                np.all(np.isfinite(delta))
                and np.all(np.isfinite(covariance))
                and np.isfinite(log_evidence)
            )
            results[index] = ConditionedFGOResult(
                basin_id=items[index].basin_id,
                success=finite,
                state=(state0 + delta) if finite else None,
                delta=delta.copy() if finite else None,
                covariance=covariance.copy() if finite else None,
                relative_log_evidence=float(log_evidence) if finite else -np.inf,
                minimized_quadratic=float(minimized) if finite else np.inf,
                log_determinant=float(log_determinant) if finite else np.inf,
                failure_reason="" if finite else "non_finite_conditioned_solution",
                factorization_group=group_index,
            )

    return ConditionedFGOBatchResult(
        hypotheses=tuple(item for item in results if item is not None),
        factorization_count=factorization_count,
        rhs_columns=rhs_columns,
        state_size=state_size,
    )


def apply_conditioned_batch(
    particle_filter: AmbiguityBasinParticleFilter,
    batch: ConditionedFGOBatchResult,
    *,
    navigation_indices: Sequence[int],
    likelihood_temperature: float = 1.0,
) -> bool:
    """Atomically feed conditional FGO states and evidence back to a basin PF.

    The six ``navigation_indices`` select ECEF position and velocity from the
    FGO state.  No mutation occurs unless every live basin has exactly one
    successful result.  This makes a partial CUDA/CPU failure equivalent to a
    default-off bridge for that epoch.
    """

    indices = np.asarray(tuple(navigation_indices), dtype=np.int64)
    if indices.shape != (6,) or np.unique(indices).size != 6:
        raise ValueError("navigation_indices must contain six distinct indices")
    if not np.isfinite(likelihood_temperature) or not 0.0 < likelihood_temperature <= 1.0:
        raise ValueError("likelihood_temperature must be in (0, 1]")

    by_id = {item.basin_id: item for item in batch.hypotheses}
    live_ids = {basin.basin_id for basin in particle_filter.basins}
    if set(by_id) != live_ids or not batch.all_succeeded:
        return False
    if np.any(indices < 0) or np.any(indices >= batch.state_size):
        raise ValueError("navigation_indices are outside the FGO state")

    updates: list[tuple[object, np.ndarray, np.ndarray, float]] = []
    for basin in particle_filter.basins:
        result = by_id[basin.basin_id]
        assert result.state is not None and result.covariance is not None
        mean = result.state[indices]
        covariance = result.covariance[np.ix_(indices, indices)]
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            return False
        updates.append(
            (
                basin,
                mean.copy(),
                covariance.copy(),
                float(likelihood_temperature) * result.relative_log_evidence,
            )
        )

    # Commit only after the complete batch has passed validation.
    log_likelihoods: dict[str, float] = {}
    for basin, mean, covariance, log_likelihood in updates:
        basin.conditional.mean = mean  # type: ignore[attr-defined]
        basin.conditional.covariance = covariance  # type: ignore[attr-defined]
        basin.conditional._repair_covariance()  # type: ignore[attr-defined]
        log_likelihoods[basin.basin_id] = log_likelihood  # type: ignore[attr-defined]
    particle_filter.update_log_likelihoods(log_likelihoods)
    return True


def parse_native_fgo_hypotheses(
    payload: Mapping[str, Any],
    *,
    group_index: int,
    require_evaluated: bool = True,
) -> tuple[NativeFGOHypothesisCandidate, ...]:
    """Parse one comparable PAR group from ``gnss_fgo`` diagnostic JSON."""

    rows = payload.get("multisd_validation_hypothesis_details")
    if not isinstance(rows, list):
        raise ValueError("native FGO payload lacks hypothesis details")
    candidates: list[NativeFGOHypothesisCandidate] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("group_index") != int(group_index):
            continue
        if require_evaluated and row.get("evaluated") is not True:
            continue
        position = np.asarray(row.get("position_ecef"), dtype=np.float64).reshape(-1)
        covariance_flat = np.asarray(
            row.get("position_covariance_m2"), dtype=np.float64
        ).reshape(-1)
        evidence = float(
            row.get(
                "incremental_log_likelihood",
                row.get("relative_log_evidence", np.nan),
            )
        )
        evidence_rows = int(row.get("incremental_likelihood_rows", 0))
        if (
            position.shape != (3,)
            or covariance_flat.shape != (9,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(covariance_flat))
            or row.get("position_covariance_valid") is not True
            or not np.isfinite(evidence)
            or ("incremental_log_likelihood" in row and evidence_rows <= 0)
        ):
            raise ValueError("native FGO hypothesis has invalid state or evidence")
        covariance = covariance_flat.reshape(3, 3)
        covariance = 0.5 * (covariance + covariance.T)
        if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
            raise ValueError("native FGO position covariance must be positive definite")

        fixed_rows = row.get("fixed_integers")
        if not isinstance(fixed_rows, list) or not fixed_rows:
            raise ValueError("native FGO hypothesis has no fixed integer identity")
        assignment: dict[tuple[tuple[str, str, int], int], int] = {}
        for fixed in fixed_rows:
            if not isinstance(fixed, Mapping):
                raise ValueError("fixed integer entry must be an object")
            satellite = str(fixed.get("satellite", ""))
            reference = str(fixed.get("reference_satellite", ""))
            wavelength = float(fixed.get("wavelength_m", np.nan))
            segment = int(fixed.get("segment_index", -1))
            reference_segment = int(fixed.get("reference_segment_index", -1))
            if (
                not satellite
                or not reference
                or not np.isfinite(wavelength)
                or wavelength <= 0.0
                or segment < 0
                or reference_segment < 0
            ):
                raise ValueError("fixed integer identity is incomplete")
            key = (
                (reference, satellite, int(round(wavelength * 1.0e9))),
                (segment << 32) | reference_segment,
            )
            if key in assignment:
                raise ValueError("duplicate fixed integer identity")
            assignment[key] = int(fixed["fixed_cycles"])

        velocity: np.ndarray | None = None
        if row.get("velocity_valid") is True:
            velocity_value = np.asarray(
                row.get("velocity_ecef_mps"), dtype=np.float64
            ).reshape(-1)
            if velocity_value.shape != (3,) or not np.all(np.isfinite(velocity_value)):
                raise ValueError("native FGO velocity is marked valid but malformed")
            velocity = velocity_value
        rank = int(row.get("rank", -1))
        candidates.append(
            NativeFGOHypothesisCandidate(
                group_index=int(group_index),
                rank=rank,
                assignment=assignment,
                position_ecef_m=position,
                position_covariance_m2=covariance,
                velocity_ecef_mps=velocity,
                relative_log_evidence=evidence,
                incremental_likelihood_rows=evidence_rows,
                source_id=f"native_fgo:g{int(group_index)}:r{rank}",
                validation_pass=row.get("pass") is True,
            )
        )
    if not candidates:
        raise ValueError(f"native FGO payload has no usable group {group_index}")
    candidates.sort(key=lambda candidate: candidate.rank)
    return tuple(candidates)


def parse_native_fgo_jsonl(
    path: Path,
    *,
    epoch_index: int,
    group_index: int,
) -> tuple[NativeFGOHypothesisCandidate, ...]:
    """Load one epoch/group from the truth-free ``gnss_solve`` basin stream."""

    rows: list[Mapping[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid basin JSONL line {line_number}") from exc
            if not isinstance(row, Mapping) or row.get("schema") != "gnsspp_multisd_basin_v1":
                raise ValueError(f"invalid basin JSONL schema on line {line_number}")
            if row.get("epoch_index") == int(epoch_index):
                rows.append(row)
    return parse_native_fgo_hypotheses(
        {"multisd_validation_hypothesis_details": rows},
        group_index=group_index,
    )


def spawn_native_fgo_candidates(
    particle_filter: AmbiguityBasinParticleFilter,
    candidates: Sequence[NativeFGOHypothesisCandidate],
    *,
    prior_mass: float,
    fallback_velocity_ecef_mps: np.ndarray,
    velocity_sigma_mps: float = 1.0,
    likelihood_temperature: float = 1.0,
) -> int:
    """Birth one native conditional-FGO group into the basin PF."""

    if not candidates:
        return 0
    groups = {candidate.group_index for candidate in candidates}
    if len(groups) != 1:
        raise ValueError("relative FGO evidence cannot be mixed across PAR groups")
    fallback_velocity = np.asarray(
        fallback_velocity_ecef_mps, dtype=np.float64
    ).reshape(3)
    if not np.all(np.isfinite(fallback_velocity)):
        raise ValueError("fallback velocity must be finite")
    if not np.isfinite(velocity_sigma_mps) or velocity_sigma_mps <= 0.0:
        raise ValueError("velocity_sigma_mps must be finite and positive")
    if not np.isfinite(likelihood_temperature) or not 0.0 < likelihood_temperature <= 1.0:
        raise ValueError("likelihood_temperature must be in (0, 1]")

    # Imported lazily to keep this module's linear algebra contract usable
    # without creating a runtime import cycle.
    from gnss_gpu.ambiguity_basin_pf import BasinKalmanState

    conditionals = []
    for candidate in candidates:
        velocity = (
            candidate.velocity_ecef_mps
            if candidate.velocity_ecef_mps is not None
            else fallback_velocity
        )
        conditionals.append(
            BasinKalmanState.from_position(
                candidate.position_ecef_m,
                candidate.position_covariance_m2,
                velocity_ecef=velocity,
                velocity_sigma_mps=float(velocity_sigma_mps),
            )
        )
    particle_filter.spawn(
        [candidate.assignment for candidate in candidates],
        conditionals,
        prior_mass=float(prior_mass),
        candidate_log_weights=[
            float(likelihood_temperature) * candidate.relative_log_evidence
            for candidate in candidates
        ],
        candidate_source_ids=[candidate.source_id for candidate in candidates],
    )
    return len(candidates)


def transition_native_fgo_candidates(
    particle_filter: AmbiguityBasinParticleFilter,
    candidates: Sequence[NativeFGOHypothesisCandidate],
    *,
    fallback_velocity_ecef_mps: np.ndarray,
    velocity_sigma_mps: float = 1.0,
    parents_per_candidate: int = 2,
    integer_conflict_log_penalty: float = 8.0,
    arc_churn_log_penalty: float = 0.05,
    likelihood_temperature: float = 1.0,
    position_transition_temperature: float = 1.0,
    position_variance_floor_m2: float = 0.0025,
) -> NativeFGOTransitionResult:
    """Propagate cumulative basin lineages through one native FGO top-K group."""

    if not candidates:
        raise ValueError("at least one native FGO candidate is required")
    if len({candidate.group_index for candidate in candidates}) != 1:
        raise ValueError("relative FGO evidence cannot be mixed across PAR groups")
    if parents_per_candidate < 1:
        raise ValueError("parents_per_candidate must be positive")
    if integer_conflict_log_penalty < 0.0 or arc_churn_log_penalty < 0.0:
        raise ValueError("transition penalties must be non-negative")
    if not np.isfinite(likelihood_temperature) or not 0.0 < likelihood_temperature <= 1.0:
        raise ValueError("likelihood_temperature must be in (0, 1]")
    if (
        not np.isfinite(position_transition_temperature)
        or position_transition_temperature < 0.0
    ):
        raise ValueError("position_transition_temperature must be non-negative")
    if not np.isfinite(position_variance_floor_m2) or position_variance_floor_m2 <= 0.0:
        raise ValueError("position_variance_floor_m2 must be finite and positive")
    fallback_velocity = np.asarray(
        fallback_velocity_ecef_mps, dtype=np.float64
    ).reshape(3)
    if not np.all(np.isfinite(fallback_velocity)):
        raise ValueError("fallback velocity must be finite")

    if not particle_filter.basins:
        count = spawn_native_fgo_candidates(
            particle_filter,
            candidates,
            prior_mass=1.0,
            fallback_velocity_ecef_mps=fallback_velocity,
            velocity_sigma_mps=velocity_sigma_mps,
            likelihood_temperature=likelihood_temperature,
        )
        return NativeFGOTransitionResult(count, len(particle_filter.basins), 0, 0)

    from gnss_gpu.ambiguity_basin_pf import BasinKalmanState

    assignments = []
    conditionals = []
    log_weights = []
    parent_ids = []
    source_ids = []
    conflict_counts: list[int] = []
    for candidate in candidates:
        candidate_assignment = dict(candidate.assignment)
        ranked_parents = []
        for parent in particle_filter.basins:
            parent_assignment = parent.assignment_dict
            common = set(parent_assignment) & set(candidate_assignment)
            conflicts = sum(
                parent_assignment[key] != candidate_assignment[key] for key in common
            )
            churn = len(set(parent_assignment) ^ set(candidate_assignment))
            innovation = (
                candidate.position_ecef_m - parent.conditional.mean[:3]
            )
            innovation_covariance = (
                parent.conditional.covariance[:3, :3]
                + candidate.position_covariance_m2
                + np.eye(3) * float(position_variance_floor_m2)
            )
            try:
                chol = np.linalg.cholesky(innovation_covariance)
                whitened = np.linalg.solve(chol, innovation)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "FGO-to-PF position transition covariance is singular"
                ) from exc
            position_log_likelihood = -0.5 * (
                float(whitened @ whitened)
                + 2.0 * float(np.log(np.diag(chol)).sum())
                + 3.0 * float(np.log(2.0 * np.pi))
            )
            transition_score = (
                parent.log_weight
                - float(integer_conflict_log_penalty) * conflicts
                - float(arc_churn_log_penalty) * churn
                + float(position_transition_temperature)
                * position_log_likelihood
            )
            ranked_parents.append((transition_score, conflicts, parent))
        ranked_parents.sort(key=lambda item: item[0], reverse=True)
        for transition_score, conflicts, parent in ranked_parents[
            : int(parents_per_candidate)
        ]:
            velocity = (
                candidate.velocity_ecef_mps
                if candidate.velocity_ecef_mps is not None
                else parent.conditional.mean[3:6]
            )
            if not np.all(np.isfinite(velocity)):
                velocity = fallback_velocity
            conditionals.append(
                BasinKalmanState.from_position(
                    candidate.position_ecef_m,
                    candidate.position_covariance_m2,
                    velocity_ecef=velocity,
                    velocity_sigma_mps=float(velocity_sigma_mps),
                )
            )
            assignments.append(candidate_assignment)
            log_weights.append(
                float(
                    transition_score
                    + float(likelihood_temperature)
                    * candidate.relative_log_evidence
                )
            )
            parent_ids.append(parent.basin_id)
            source_ids.append(candidate.source_id)
            conflict_counts.append(int(conflicts))

    particle_filter.replace_with_transitions(
        assignments,
        conditionals,
        log_weights,
        parent_ids,
        candidate_source_ids=source_ids,
    )
    return NativeFGOTransitionResult(
        parent_child_branches=len(assignments),
        resulting_basins=len(particle_filter.basins),
        minimum_conflicts=min(conflict_counts, default=0),
        maximum_conflicts=max(conflict_counts, default=0),
    )
