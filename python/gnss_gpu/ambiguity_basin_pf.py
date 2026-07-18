"""Rao-Blackwellized PF over integer-ambiguity basins (no FGO).

The sampled state is a versioned integer assignment.  Each discrete basin
owns a six-state ECEF position/velocity Kalman conditional and accumulates the
measurement marginal likelihood produced by that conditional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Iterable, Mapping

import numpy as np

from gnss_gpu.dd_float_kf import AmbiguityKey, _dd_geometry_and_design, _pair_keys


VersionedAmbiguityKey = tuple[AmbiguityKey, int]
AssignmentItem = tuple[VersionedAmbiguityKey, int]


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    vmax = float(np.max(values))
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


def _canonical_assignment(
    assignment: Mapping[VersionedAmbiguityKey, int] | Iterable[AssignmentItem],
) -> tuple[AssignmentItem, ...]:
    items = assignment.items() if isinstance(assignment, Mapping) else assignment
    return tuple(sorted(((key, int(value)) for key, value in items), key=lambda x: x[0]))


@dataclass
class BasinKalmanState:
    mean: np.ndarray
    covariance: np.ndarray
    accel_process_sigma_mps2: float = 2.0

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(6).copy()
        self.covariance = np.asarray(self.covariance, dtype=np.float64).reshape(6, 6).copy()
        self._repair_covariance()

    @classmethod
    def from_position(
        cls,
        position_ecef: np.ndarray,
        position_covariance: np.ndarray,
        *,
        velocity_ecef: np.ndarray | None = None,
        velocity_sigma_mps: float = 3.0,
        accel_process_sigma_mps2: float = 2.0,
    ) -> "BasinKalmanState":
        mean = np.zeros(6, dtype=np.float64)
        mean[:3] = np.asarray(position_ecef, dtype=np.float64).reshape(3)
        if velocity_ecef is not None:
            mean[3:6] = np.asarray(velocity_ecef, dtype=np.float64).reshape(3)
        covariance = np.zeros((6, 6), dtype=np.float64)
        covariance[:3, :3] = np.asarray(position_covariance, dtype=np.float64).reshape(3, 3)
        covariance[3:6, 3:6] = np.eye(3) * float(velocity_sigma_mps) ** 2
        return cls(mean, covariance, accel_process_sigma_mps2)

    def clone(self) -> "BasinKalmanState":
        return BasinKalmanState(
            self.mean.copy(), self.covariance.copy(), self.accel_process_sigma_mps2
        )

    def predict(self, dt: float) -> None:
        dt = float(dt)
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:6] = np.eye(3) * dt
        accel_var = float(self.accel_process_sigma_mps2) ** 2
        process = np.zeros((6, 6), dtype=np.float64)
        process[:3, :3] = np.eye(3) * 0.25 * dt**4 * accel_var
        process[:3, 3:6] = np.eye(3) * 0.5 * dt**3 * accel_var
        process[3:6, :3] = process[:3, 3:6]
        process[3:6, 3:6] = np.eye(3) * dt**2 * accel_var
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + process
        self._repair_covariance()

    def update_velocity(self, velocity_ecef: np.ndarray, sigma_mps: float) -> float:
        design = np.zeros((3, 6), dtype=np.float64)
        design[:, 3:6] = np.eye(3)
        residual = np.asarray(velocity_ecef, dtype=np.float64).reshape(3) - self.mean[3:6]
        return self.update_linear(design, residual, np.full(3, float(sigma_mps) ** 2))

    def update_pseudorange(self, dd_result, sigma_pr_m: float = 5.0) -> float:
        expected, position_design = _dd_geometry_and_design(dd_result, self.mean[:3])
        residual = np.asarray(dd_result.dd_pseudorange_m, dtype=np.float64) - expected
        design = np.zeros((residual.size, 6), dtype=np.float64)
        design[:, :3] = position_design
        weights = np.clip(np.asarray(dd_result.dd_weights, dtype=np.float64), 1e-6, None)
        return self.update_linear(design, residual, float(sigma_pr_m) ** 2 / weights)

    def update_fixed_carrier(
        self,
        dd_result,
        assignment: Mapping[VersionedAmbiguityKey, int],
        generations: Mapping[AmbiguityKey, int],
        *,
        sigma_cp_cycles: float = 0.05,
    ) -> tuple[float, int]:
        pair_keys = _pair_keys(dd_result)
        expected, position_design = _dd_geometry_and_design(dd_result, self.mean[:3])
        wavelengths = np.asarray(dd_result.wavelengths_m, dtype=np.float64)
        carrier = np.asarray(dd_result.dd_carrier_cycles, dtype=np.float64)
        rows: list[int] = []
        integers: list[int] = []
        for i, pair_key in enumerate(pair_keys):
            versioned = (pair_key, int(generations.get(pair_key, -1)))
            if versioned in assignment:
                rows.append(i)
                integers.append(int(assignment[versioned]))
        if not rows:
            return 0.0, 0
        idx = np.asarray(rows, dtype=np.int64)
        fixed = np.asarray(integers, dtype=np.float64)
        predicted = expected[idx] + wavelengths[idx] * fixed
        residual = carrier[idx] * wavelengths[idx] - predicted
        design = np.zeros((idx.size, 6), dtype=np.float64)
        design[:, :3] = position_design[idx]
        weights = np.clip(np.asarray(dd_result.dd_weights, dtype=np.float64)[idx], 1e-6, None)
        variance = (float(sigma_cp_cycles) * wavelengths[idx]) ** 2 / weights
        return self.update_linear(design, residual, variance), int(idx.size)

    def update_linear(
        self,
        design: np.ndarray,
        residual: np.ndarray,
        variance: np.ndarray,
    ) -> float:
        h = np.asarray(design, dtype=np.float64).reshape(-1, 6)
        innovation = np.asarray(residual, dtype=np.float64).reshape(-1)
        variances = np.broadcast_to(np.asarray(variance, dtype=np.float64), innovation.shape)
        valid = (
            np.all(np.isfinite(h), axis=1)
            & np.isfinite(innovation)
            & np.isfinite(variances)
            & (variances > 0.0)
        )
        h = h[valid]
        innovation = innovation[valid]
        variances = variances[valid]
        if innovation.size == 0:
            return 0.0
        if innovation.size > self.mean.size:
            return self._update_information(h, innovation, variances)
        rmat = np.diag(variances)
        innovation_cov = h @ self.covariance @ h.T + rmat
        innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
        try:
            chol = np.linalg.cholesky(innovation_cov)
            solved = np.linalg.solve(chol.T, np.linalg.solve(chol, innovation))
            gain = np.linalg.solve(innovation_cov, h @ self.covariance).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("basin KF innovation covariance is singular") from exc
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        log_likelihood = -0.5 * (
            float(innovation @ solved) + log_det + innovation.size * np.log(2.0 * np.pi)
        )
        prior_cov = self.covariance
        self.mean = self.mean + gain @ innovation
        ikh = np.eye(6) - gain @ h
        self.covariance = ikh @ prior_cov @ ikh.T + gain @ rmat @ gain.T
        self._repair_covariance()
        return float(log_likelihood)

    def _update_information(
        self,
        design: np.ndarray,
        innovation: np.ndarray,
        variances: np.ndarray,
    ) -> float:
        """Exact KF update in six-state information space.

        DD batches commonly have 20-30 rows.  The determinant lemma and
        Woodbury identity avoid factoring that row-sized innovation matrix;
        both the posterior and Gaussian marginal likelihood require only
        six-dimensional Cholesky factors.
        """

        h = np.asarray(design, dtype=np.float64)
        residual = np.asarray(innovation, dtype=np.float64)
        variance = np.asarray(variances, dtype=np.float64)
        try:
            prior_chol = np.linalg.cholesky(self.covariance)
            prior_precision = np.linalg.solve(
                prior_chol.T,
                np.linalg.solve(prior_chol, np.eye(self.mean.size)),
            )
            weighted_h = h / variance[:, None]
            posterior_precision = prior_precision + h.T @ weighted_h
            posterior_precision = 0.5 * (
                posterior_precision + posterior_precision.T
            )
            posterior_chol = np.linalg.cholesky(posterior_precision)
            information_residual = h.T @ (residual / variance)
            correction = np.linalg.solve(
                posterior_chol.T,
                np.linalg.solve(posterior_chol, information_residual),
            )
            posterior_covariance = np.linalg.solve(
                posterior_chol.T,
                np.linalg.solve(posterior_chol, np.eye(self.mean.size)),
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("basin KF information update is singular") from exc

        log_det = (
            float(np.sum(np.log(variance)))
            + 2.0 * float(np.sum(np.log(np.diag(prior_chol))))
            + 2.0 * float(np.sum(np.log(np.diag(posterior_chol))))
        )
        mahalanobis = float(
            residual @ (residual / variance)
            - information_residual @ correction
        )
        # Roundoff can make a theoretically non-negative quadratic form tiny
        # and negative when the measurement batch is highly redundant.
        mahalanobis = max(mahalanobis, 0.0)
        log_likelihood = -0.5 * (
            mahalanobis + log_det + residual.size * np.log(2.0 * np.pi)
        )
        self.mean = self.mean + correction
        self.covariance = posterior_covariance
        self._repair_covariance()
        return float(log_likelihood)

    def _repair_covariance(self) -> None:
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        min_eig = float(np.min(np.linalg.eigvalsh(self.covariance)))
        if min_eig < 1.0e-12:
            self.covariance += np.eye(6) * (1.0e-12 - min_eig)


@dataclass
class IntegerBasin:
    basin_id: str
    assignment: tuple[AssignmentItem, ...]
    conditional: BasinKalmanState
    log_weight: float
    cumulative_log_marginal: float = 0.0
    lineage: tuple[str, ...] = field(default_factory=tuple)
    birth_epoch: int = 0

    @property
    def assignment_dict(self) -> dict[VersionedAmbiguityKey, int]:
        return dict(self.assignment)


@dataclass(frozen=True)
class BasinPosterior:
    map_assignment: tuple[AssignmentItem, ...]
    gamma: float
    ess: float
    n_basins: int
    fixed: bool
    fix_streak: int


@dataclass(frozen=True)
class PositionClusterPosterior:
    mean_position_ecef: np.ndarray
    gamma: float
    rms_spread_m: float
    n_members: int
    representative_basin_id: str | None


@dataclass(frozen=True)
class BasinUpdateEvidence:
    log_marginal: float
    n_rows: int
    n_basins: int


class AmbiguityBasinParticleFilter:
    """Discrete basin PF with Rao-Blackwellized navigation conditionals."""

    def __init__(
        self,
        *,
        max_basins: int = 128,
        fix_gamma_threshold: float = 0.99,
        fix_min_streak: int = 3,
        min_fixed_ambiguities: int = 6,
    ) -> None:
        self.max_basins = int(max_basins)
        self.fix_gamma_threshold = float(fix_gamma_threshold)
        self.fix_min_streak = int(fix_min_streak)
        self.min_fixed_ambiguities = int(min_fixed_ambiguities)
        self.basins: list[IntegerBasin] = []
        self.epoch = -1
        self._ids = count()
        self._last_map: tuple[AssignmentItem, ...] | None = None
        self._fix_streak = 0

    def spawn(
        self,
        assignments: Iterable[Mapping[VersionedAmbiguityKey, int]],
        conditionals: Iterable[BasinKalmanState],
        *,
        prior_mass: float = 1.0,
        parent_id: str | None = None,
        candidate_log_weights: Iterable[float] | None = None,
    ) -> None:
        assignments_list = list(assignments)
        conditionals_list = list(conditionals)
        if len(assignments_list) != len(conditionals_list):
            raise ValueError("assignments and conditionals must have equal length")
        if not assignments_list:
            return
        if float(prior_mass) <= 0.0 or float(prior_mass) > 1.0:
            raise ValueError("prior_mass must be positive")
        if candidate_log_weights is None:
            candidate_log_mass = np.full(
                len(assignments_list), -np.log(len(assignments_list)), dtype=np.float64
            )
        else:
            candidate_log_mass = np.asarray(
                list(candidate_log_weights), dtype=np.float64
            ).reshape(-1)
            if candidate_log_mass.size != len(assignments_list):
                raise ValueError("candidate_log_weights must match assignments")
            candidate_log_mass -= _logsumexp(candidate_log_mass)
        if self.basins:
            old_scale = max(1.0 - float(prior_mass), 1.0e-12)
            for basin in self.basins:
                basin.log_weight += np.log(old_scale)
        for assignment, conditional, log_candidate_mass in zip(
            assignments_list, conditionals_list, candidate_log_mass
        ):
            basin_id = f"b{next(self._ids)}"
            lineage = (parent_id, basin_id) if parent_id else (basin_id,)
            self.basins.append(
                IntegerBasin(
                    basin_id=basin_id,
                    assignment=_canonical_assignment(assignment),
                    conditional=conditional.clone(),
                    log_weight=float(np.log(float(prior_mass)) + log_candidate_mass),
                    lineage=lineage,
                    birth_epoch=max(self.epoch, 0),
                )
            )
        self._deduplicate()
        self._normalize_and_cap()

    def predict(self, dt: float) -> None:
        self.epoch += 1
        for basin in self.basins:
            basin.conditional.predict(dt)

    def update_log_likelihoods(self, log_likelihood_by_id: Mapping[str, float]) -> None:
        for basin in self.basins:
            increment = float(log_likelihood_by_id.get(basin.basin_id, 0.0))
            basin.log_weight += increment
            basin.cumulative_log_marginal += increment
        self._normalize_and_cap()

    def update_pseudorange(
        self, dd_result, sigma_pr_m: float = 5.0
    ) -> BasinUpdateEvidence:
        if not self.basins:
            return BasinUpdateEvidence(0.0, 0, 0)
        prior = np.asarray([basin.log_weight for basin in self.basins], dtype=np.float64)
        increments: list[float] = []
        for basin in self.basins:
            increment = basin.conditional.update_pseudorange(dd_result, sigma_pr_m)
            increments.append(float(increment))
            basin.log_weight += increment
            basin.cumulative_log_marginal += increment
        log_marginal = _logsumexp(prior + np.asarray(increments)) - _logsumexp(prior)
        n_basins = len(self.basins)
        self._normalize_and_cap()
        return BasinUpdateEvidence(log_marginal, int(dd_result.n_dd), n_basins)

    def update_velocity(
        self, velocity_ecef: np.ndarray, sigma_mps: float
    ) -> BasinUpdateEvidence:
        if not self.basins:
            return BasinUpdateEvidence(0.0, 0, 0)
        prior = np.asarray([basin.log_weight for basin in self.basins], dtype=np.float64)
        increments: list[float] = []
        for basin in self.basins:
            increment = basin.conditional.update_velocity(velocity_ecef, sigma_mps)
            increments.append(float(increment))
            basin.log_weight += increment
            basin.cumulative_log_marginal += increment
        log_marginal = _logsumexp(prior + np.asarray(increments)) - _logsumexp(prior)
        n_basins = len(self.basins)
        self._normalize_and_cap()
        return BasinUpdateEvidence(log_marginal, 3, n_basins)

    def update_fixed_carrier(
        self,
        dd_result,
        generations: Mapping[AmbiguityKey, int],
        *,
        sigma_cp_cycles: float = 0.05,
    ) -> BasinUpdateEvidence:
        if not self.basins:
            return BasinUpdateEvidence(0.0, 0, 0)
        prior = np.asarray([basin.log_weight for basin in self.basins], dtype=np.float64)
        increments: list[float] = []
        max_rows = 0
        for basin in self.basins:
            increment, n_rows = basin.conditional.update_fixed_carrier(
                dd_result,
                basin.assignment_dict,
                generations,
                sigma_cp_cycles=sigma_cp_cycles,
            )
            increments.append(float(increment))
            basin.log_weight += increment
            basin.cumulative_log_marginal += increment
            max_rows = max(max_rows, int(n_rows))
        log_marginal = _logsumexp(prior + np.asarray(increments)) - _logsumexp(prior)
        n_basins = len(self.basins)
        self._normalize_and_cap()
        return BasinUpdateEvidence(log_marginal, max_rows, n_basins)

    def release(self, keys: Iterable[VersionedAmbiguityKey]) -> None:
        remove = set(keys)
        for basin in self.basins:
            basin.assignment = tuple(item for item in basin.assignment if item[0] not in remove)
        self._deduplicate()
        self._normalize_and_cap()

    def retain_compatible(
        self,
        active_keys: Iterable[VersionedAmbiguityKey],
        *,
        min_assignment_size: int | None = None,
    ) -> None:
        """Drop fixed basins invalidated by a slip generation or outage.

        The later reset branch may retain a float conditional explicitly.  The
        G3/G4 MVP uses conservative release-by-abandonment so stale fixed
        basins cannot win merely because they evaluate fewer carrier rows.
        """

        active = set(active_keys)
        minimum = (
            self.min_fixed_ambiguities
            if min_assignment_size is None
            else int(min_assignment_size)
        )
        self.basins = [
            basin
            for basin in self.basins
            if len(basin.assignment) >= minimum
            and all(item[0] in active for item in basin.assignment)
        ]
        self._normalize_and_cap()

    def posterior(self) -> BasinPosterior:
        if not self.basins:
            return BasinPosterior((), 0.0, 0.0, 0, False, 0)
        self._normalize_and_cap()
        mass: dict[tuple[AssignmentItem, ...], float] = {}
        weights = np.exp(np.asarray([basin.log_weight for basin in self.basins]))
        for basin, weight in zip(self.basins, weights):
            mass[basin.assignment] = mass.get(basin.assignment, 0.0) + float(weight)
        map_assignment, gamma = max(mass.items(), key=lambda item: item[1])
        eligible = len(map_assignment) >= self.min_fixed_ambiguities
        if (
            eligible
            and map_assignment == self._last_map
            and gamma > self.fix_gamma_threshold
        ):
            self._fix_streak += 1
        elif eligible and gamma > self.fix_gamma_threshold:
            self._fix_streak = 1
        else:
            self._fix_streak = 0
        self._last_map = map_assignment
        ess = float(1.0 / np.sum(weights * weights))
        return BasinPosterior(
            map_assignment=map_assignment,
            gamma=float(gamma),
            ess=ess,
            n_basins=len(self.basins),
            fixed=bool(
                gamma > self.fix_gamma_threshold and self._fix_streak >= self.fix_min_streak
                and eligible
            ),
            fix_streak=int(self._fix_streak),
        )

    def map_basin(self) -> IntegerBasin | None:
        if not self.basins:
            return None
        posterior = self.posterior()
        candidates = [b for b in self.basins if b.assignment == posterior.map_assignment]
        return max(candidates, key=lambda basin: basin.log_weight)

    def position_cluster_posterior(
        self, radius_m: float = 0.5
    ) -> PositionClusterPosterior:
        """Aggregate posterior mass over connected position-basin clusters."""

        if not self.basins:
            return PositionClusterPosterior(np.full(3, np.nan), 0.0, 0.0, 0, None)
        if not np.isfinite(radius_m) or float(radius_m) <= 0.0:
            raise ValueError("position cluster radius must be finite and positive")
        self._normalize_and_cap()
        positions = np.asarray([basin.conditional.mean[:3] for basin in self.basins])
        weights = np.exp(np.asarray([basin.log_weight for basin in self.basins]))
        # Use posterior balls, not single-link connected components.  Chained
        # basins can otherwise aggregate mass across a diameter far larger
        # than the requested RTK position radius.
        neighborhoods = [
            np.flatnonzero(
                np.linalg.norm(positions - positions[center], axis=1)
                <= float(radius_m)
            ).tolist()
            for center in range(len(self.basins))
        ]
        best_members = max(
            neighborhoods, key=lambda members: float(weights[members].sum())
        )
        member_weights = weights[best_members]
        gamma = float(member_weights.sum())
        normalized = member_weights / gamma
        member_positions = positions[best_members]
        mean_position = np.sum(normalized[:, None] * member_positions, axis=0)
        spread = float(
            np.sqrt(
                np.sum(
                    normalized
                    * np.sum((member_positions - mean_position) ** 2, axis=1)
                )
            )
        )
        representative_index = max(
            best_members, key=lambda index: self.basins[index].log_weight
        )
        return PositionClusterPosterior(
            mean_position_ecef=mean_position,
            gamma=gamma,
            rms_spread_m=spread,
            n_members=len(best_members),
            representative_basin_id=self.basins[representative_index].basin_id,
        )

    def _deduplicate(self) -> None:
        grouped: dict[tuple[AssignmentItem, ...], list[IntegerBasin]] = {}
        for basin in self.basins:
            grouped.setdefault(basin.assignment, []).append(basin)
        merged: list[IntegerBasin] = []
        for assignment, group in grouped.items():
            if len(group) == 1:
                merged.append(group[0])
                continue
            log_weights = np.asarray([basin.log_weight for basin in group])
            total_log_weight = _logsumexp(log_weights)
            weights = np.exp(log_weights - total_log_weight)
            mean = sum(w * basin.conditional.mean for w, basin in zip(weights, group))
            covariance = np.zeros((6, 6), dtype=np.float64)
            for weight, basin in zip(weights, group):
                delta = basin.conditional.mean - mean
                covariance += weight * (basin.conditional.covariance + np.outer(delta, delta))
            representative = max(group, key=lambda basin: basin.log_weight)
            representative.assignment = assignment
            representative.log_weight = float(total_log_weight)
            representative.conditional = BasinKalmanState(
                mean, covariance, representative.conditional.accel_process_sigma_mps2
            )
            representative.cumulative_log_marginal = float(
                sum(w * b.cumulative_log_marginal for w, b in zip(weights, group))
            )
            merged.append(representative)
        self.basins = merged

    def _normalize_and_cap(self) -> None:
        if not self.basins:
            return
        self.basins.sort(key=lambda basin: basin.log_weight, reverse=True)
        if len(self.basins) > self.max_basins:
            self.basins = self.basins[: self.max_basins]
        values = np.asarray([basin.log_weight for basin in self.basins], dtype=np.float64)
        normalizer = _logsumexp(values)
        for basin in self.basins:
            basin.log_weight -= normalizer
