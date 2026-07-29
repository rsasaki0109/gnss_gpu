"""IMU/map-aware multi-hypothesis control and safe outage recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from gnss_gpu.ambiguity_basin_pf import (
    AmbiguityBasinParticleFilter,
    IntegerBasin,
)
from gnss_gpu.evidence import AcceptanceDecision


@dataclass
class InertialBiasState:
    accel_bias_mps2: np.ndarray
    gyro_bias_radps: np.ndarray
    covariance: np.ndarray
    accel_random_walk_mps2_sqrthz: float = 0.01
    gyro_random_walk_radps_sqrthz: float = 0.001

    def __post_init__(self) -> None:
        self.accel_bias_mps2 = np.asarray(self.accel_bias_mps2, dtype=np.float64).reshape(3)
        self.gyro_bias_radps = np.asarray(self.gyro_bias_radps, dtype=np.float64).reshape(3)
        self.covariance = np.asarray(self.covariance, dtype=np.float64).reshape(6, 6)
        if not (
            np.all(np.isfinite(self.accel_bias_mps2))
            and np.all(np.isfinite(self.gyro_bias_radps))
            and np.all(np.isfinite(self.covariance))
        ):
            raise ValueError("inertial bias state must be finite")
        if float(np.min(np.linalg.eigvalsh(0.5 * (self.covariance + self.covariance.T)))) < 0:
            raise ValueError("inertial bias covariance must be positive semidefinite")
        if self.accel_random_walk_mps2_sqrthz < 0 or self.gyro_random_walk_radps_sqrthz < 0:
            raise ValueError("bias random walks must be non-negative")

    @classmethod
    def zero(cls, sigma_accel: float = 0.1, sigma_gyro: float = 0.01) -> "InertialBiasState":
        covariance = np.diag([sigma_accel**2] * 3 + [sigma_gyro**2] * 3)
        return cls(np.zeros(3), np.zeros(3), covariance)

    @property
    def mean(self) -> np.ndarray:
        return np.r_[self.accel_bias_mps2, self.gyro_bias_radps]

    def propagate(self, dt: float) -> None:
        dt = float(dt)
        noise = np.diag(
            [self.accel_random_walk_mps2_sqrthz**2 * dt] * 3
            + [self.gyro_random_walk_radps_sqrthz**2 * dt] * 3
        )
        self.covariance = self.covariance + noise


@dataclass(frozen=True)
class PreintegratedNavigationDelta:
    dt: float
    cv_position_correction_ecef_m: np.ndarray
    delta_velocity_ecef_mps: np.ndarray
    covariance: np.ndarray
    bias_jacobian: np.ndarray
    sample_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("preintegrated delta dt must be finite and positive")
        if self.sample_count < 1:
            raise ValueError("preintegrated delta requires at least one IMU sample")
        correction = np.asarray(self.cv_position_correction_ecef_m, dtype=np.float64).reshape(3)
        velocity = np.asarray(self.delta_velocity_ecef_mps, dtype=np.float64).reshape(3)
        covariance = np.asarray(self.covariance, dtype=np.float64).reshape(6, 6)
        jacobian = np.asarray(self.bias_jacobian, dtype=np.float64).reshape(6, 6)
        if not all(np.all(np.isfinite(value)) for value in (correction, velocity, covariance, jacobian)):
            raise ValueError("preintegrated delta fields must be finite")
        object.__setattr__(self, "cv_position_correction_ecef_m", correction)
        object.__setattr__(self, "delta_velocity_ecef_mps", velocity)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "bias_jacobian", jacobian)

    def corrected(self, bias: InertialBiasState) -> tuple[np.ndarray, np.ndarray]:
        raw = np.r_[self.cv_position_correction_ecef_m, self.delta_velocity_ecef_mps]
        corrected = raw - self.bias_jacobian @ bias.mean
        covariance = self.covariance + self.bias_jacobian @ bias.covariance @ self.bias_jacobian.T
        return corrected, covariance


@dataclass(frozen=True)
class MapProposal:
    proposal_id: str
    position_ecef_m: np.ndarray
    covariance_m2: np.ndarray
    log_prior: float = 0.0

    def __post_init__(self) -> None:
        position = np.asarray(self.position_ecef_m, dtype=np.float64).reshape(3)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64).reshape(3, 3)
        if (
            not self.proposal_id
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(covariance))
            or not math.isfinite(self.log_prior)
        ):
            raise ValueError("map proposal fields must be finite and identified")
        if float(np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T)))) <= 0:
            raise ValueError("map proposal covariance must be positive definite")
        object.__setattr__(self, "position_ecef_m", position)
        object.__setattr__(self, "covariance_m2", covariance)


class NavigationPhase(str, Enum):
    TRACKING = "tracking"
    COASTING = "coasting"
    DEGRADED = "degraded"
    REACQUIRING = "reacquiring"


@dataclass(frozen=True)
class RecoveryPolicy:
    maximum_coast_epochs: int = 10
    required_reacquisition_streak: int = 3
    collapse_gamma_threshold: float = 0.98
    minimum_diverse_basins: int = 2
    map_proposal_prior_mass: float = 0.25
    maximum_map_proposal_distance_m: float = 30.0
    covariance_inflation_per_outage_epoch: float = 1.25

    def __post_init__(self) -> None:
        if self.maximum_coast_epochs < 1 or self.required_reacquisition_streak < 1:
            raise ValueError("recovery epoch limits must be positive")
        if not 0 < self.collapse_gamma_threshold <= 1:
            raise ValueError("collapse gamma threshold must be in (0, 1]")
        if self.minimum_diverse_basins < 2:
            raise ValueError("minimum_diverse_basins must be at least two")
        if not 0 < self.map_proposal_prior_mass < 1:
            raise ValueError("map proposal prior mass must be in (0, 1)")
        if self.maximum_map_proposal_distance_m <= 0:
            raise ValueError("maximum map proposal distance must be positive")
        if self.covariance_inflation_per_outage_epoch < 1:
            raise ValueError("covariance inflation must be at least one")


@dataclass(frozen=True)
class NavigationSafetyStatus:
    phase: NavigationPhase
    safe_to_emit_fix: bool
    outage_epochs: int
    reacquisition_streak: int
    recovery_epochs: int | None
    basin_count: int
    posterior_gamma: float
    posterior_entropy: float
    premature_collapse: bool


def _posterior_entropy(basins: Iterable[IntegerBasin]) -> float:
    weights = np.exp(np.asarray([basin.log_weight for basin in basins], dtype=np.float64))
    if weights.size == 0:
        return 0.0
    weights /= np.sum(weights)
    positive = weights > 0
    return float(-np.sum(weights[positive] * np.log(weights[positive])))


class MultiHypothesisNavigationController:
    """Coordinate existing basin PF, IMU covariance, map proposals, and outages."""

    def __init__(
        self,
        pf: AmbiguityBasinParticleFilter,
        *,
        bias: InertialBiasState | None = None,
        policy: RecoveryPolicy | None = None,
    ) -> None:
        self.pf = pf
        self.bias = bias or InertialBiasState.zero()
        self.policy = policy or RecoveryPolicy()
        self.phase = NavigationPhase.TRACKING
        self.outage_epochs = 0
        self.reacquisition_streak = 0
        self.recovery_epochs: int | None = None
        self._reacquisition_epochs = 0

    def predict_imu(self, delta: PreintegratedNavigationDelta | None, *, fallback_dt: float) -> None:
        if delta is None:
            self.pf.predict(fallback_dt)
            return
        corrected, covariance = delta.corrected(self.bias)
        self.pf.predict_inertial(
            delta.dt,
            cv_position_correction_ecef_m=corrected[:3],
            delta_velocity_ecef_mps=corrected[3:],
            process_covariance=covariance,
        )
        self.bias.propagate(delta.dt)

    def propose_from_map(self, proposals: Iterable[MapProposal]) -> int:
        proposals = list(proposals)
        if not proposals or not self.pf.basins:
            return 0
        parents = sorted(self.pf.basins, key=lambda basin: basin.log_weight, reverse=True)
        assignments = []
        conditionals = []
        log_weights = []
        source_ids = []
        for parent in parents[: self.policy.minimum_diverse_basins]:
            for proposal in proposals:
                distance = float(
                    np.linalg.norm(parent.conditional.mean[:3] - proposal.position_ecef_m)
                )
                if distance > self.policy.maximum_map_proposal_distance_m:
                    continue
                conditional = parent.conditional.clone()
                design = np.zeros((3, 6), dtype=np.float64)
                design[:, :3] = np.eye(3)
                residual = proposal.position_ecef_m - conditional.mean[:3]
                conditional.update_linear(design, residual, np.diag(proposal.covariance_m2))
                assignments.append(parent.assignment_dict)
                conditionals.append(conditional)
                log_weights.append(
                    parent.log_weight
                    + proposal.log_prior
                    - 0.5 * distance**2 / max(float(np.trace(proposal.covariance_m2)), 1.0e-9)
                )
                source_ids.append(f"map:{proposal.proposal_id}")
        if not assignments:
            return 0
        self.pf.spawn(
            assignments,
            conditionals,
            prior_mass=self.policy.map_proposal_prior_mass,
            candidate_log_weights=log_weights,
            candidate_source_ids=source_ids,
        )
        return len(assignments)

    def observe_gnss(
        self,
        *,
        available: bool,
        evidence_decision: AcceptanceDecision | None = None,
    ) -> NavigationSafetyStatus:
        if not available:
            self.outage_epochs += 1
            self.reacquisition_streak = 0
            self._reacquisition_epochs = 0
            self.recovery_epochs = None
            self.pf.invalidate_fix()
            inflation = self.policy.covariance_inflation_per_outage_epoch
            for basin in self.pf.basins:
                basin.conditional.covariance *= inflation
            self.phase = (
                NavigationPhase.COASTING
                if self.outage_epochs <= self.policy.maximum_coast_epochs
                else NavigationPhase.DEGRADED
            )
            return self.status()

        # Advance the PF's temporal FIX state exactly once per GNSS epoch.
        self.pf.posterior()
        if self.outage_epochs > 0 or self.phase in {
            NavigationPhase.COASTING,
            NavigationPhase.DEGRADED,
            NavigationPhase.REACQUIRING,
        }:
            self.phase = NavigationPhase.REACQUIRING
            self._reacquisition_epochs += 1
            safe_evidence = evidence_decision is not None and evidence_decision.accepted
            if safe_evidence:
                self.reacquisition_streak += 1
            else:
                self.reacquisition_streak = 0
                self.pf.invalidate_fix()
            if self.reacquisition_streak >= self.policy.required_reacquisition_streak:
                self.phase = NavigationPhase.TRACKING
                self.recovery_epochs = self._reacquisition_epochs
                self.outage_epochs = 0
                self._reacquisition_epochs = 0
        return self.status()

    def status(self) -> NavigationSafetyStatus:
        posterior = self.pf.posterior_snapshot()
        premature = (
            self.phase != NavigationPhase.TRACKING
            and posterior.gamma >= self.policy.collapse_gamma_threshold
            and posterior.n_basins < self.policy.minimum_diverse_basins
        )
        safe = (
            self.phase == NavigationPhase.TRACKING
            and not premature
            and posterior.fixed
        )
        return NavigationSafetyStatus(
            phase=self.phase,
            safe_to_emit_fix=safe,
            outage_epochs=self.outage_epochs,
            reacquisition_streak=self.reacquisition_streak,
            recovery_epochs=self.recovery_epochs,
            basin_count=posterior.n_basins,
            posterior_gamma=posterior.gamma,
            posterior_entropy=_posterior_entropy(self.pf.basins),
            premature_collapse=premature,
        )
