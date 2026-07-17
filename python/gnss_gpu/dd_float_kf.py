"""FGO-free double-difference float RTK Kalman filter.

The filter is deliberately independent of the diffuse position-particle cloud.
It maintains ECEF position/velocity and a dynamic set of float DD carrier
ambiguities, including their joint covariance.  That joint marginal is the
seed required by LAMBDA and by the WP23b integer-basin particle filter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


AmbiguityKey = tuple[str, str, int]


@dataclass(frozen=True)
class FloatUpdateDiagnostics:
    n_input: int
    n_used: int
    innovation_rms: float
    normalized_innovation_sq: float
    covariance_min_eig: float
    ambiguities_reset: int = 0


@dataclass(frozen=True)
class FloatAmbiguitySeed:
    keys: tuple[AmbiguityKey, ...]
    ahat_cycles: np.ndarray
    qahat_cycles2: np.ndarray
    position_ambiguity_cov: np.ndarray


@dataclass
class _Track:
    index: int
    generation: int
    last_seen_epoch: int


def _symmetrize_psd(covariance: np.ndarray, floor: float = 1.0e-12) -> np.ndarray:
    cov = 0.5 * (np.asarray(covariance, dtype=np.float64) + np.asarray(covariance).T)
    eig_min = float(np.min(np.linalg.eigvalsh(cov)))
    if eig_min < floor:
        cov = cov + np.eye(cov.shape[0], dtype=np.float64) * (floor - eig_min)
    return cov


def _pair_keys(result) -> tuple[AmbiguityKey, ...]:
    n = int(getattr(result, "n_dd", 0))
    refs = tuple(getattr(result, "ref_sat_ids", ()))
    sats = tuple(getattr(result, "sat_ids", ()))
    wavelengths = np.asarray(getattr(result, "wavelengths_m", ()), dtype=np.float64)
    if len(refs) != n or len(sats) != n or wavelengths.size != n:
        raise ValueError("DD carrier result lacks stable pair IDs or wavelengths")
    return tuple(
        (str(refs[i]), str(sats[i]), int(round(float(wavelengths[i]) * 1.0e9)))
        for i in range(n)
    )


def _pr_by_pair(result) -> dict[tuple[str, str], float]:
    if result is None:
        return {}
    n = int(getattr(result, "n_dd", 0))
    refs = tuple(getattr(result, "ref_sat_ids", ()))
    sats = tuple(getattr(result, "sat_ids", ()))
    values = np.asarray(getattr(result, "dd_pseudorange_m", ()), dtype=np.float64)
    if len(refs) != n or len(sats) != n or values.size != n:
        return {}
    return {(str(refs[i]), str(sats[i])): float(values[i]) for i in range(n)}


def _dd_geometry_and_design(result, position_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position_ecef, dtype=np.float64).reshape(3)
    sat_k = np.asarray(result.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(result.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    range_k = np.linalg.norm(sat_k - pos, axis=1)
    range_ref = np.linalg.norm(sat_ref - pos, axis=1)
    expected = (
        range_k
        - range_ref
        - np.asarray(result.base_range_k, dtype=np.float64)
        + np.asarray(result.base_range_ref, dtype=np.float64)
    )
    unit_k = (sat_k - pos) / np.maximum(range_k[:, None], 1.0)
    unit_ref = (sat_ref - pos) / np.maximum(range_ref[:, None], 1.0)
    design = -unit_k + unit_ref
    return expected, design


class DDFloatKalmanFilter:
    """Dynamic-state float RTK KF suitable for independent LAMBDA seeding."""

    NAV_DIM = 6

    def __init__(
        self,
        position_ecef: np.ndarray,
        *,
        velocity_ecef: np.ndarray | None = None,
        position_sigma_m: float = 30.0,
        velocity_sigma_mps: float = 5.0,
        accel_process_sigma_mps2: float = 2.0,
        ambiguity_init_sigma_cycles: float = 30.0,
        ambiguity_process_sigma_cycles: float = 0.002,
        max_track_age_epochs: int = 10,
    ) -> None:
        self.mean = np.zeros(self.NAV_DIM, dtype=np.float64)
        self.mean[:3] = np.asarray(position_ecef, dtype=np.float64).reshape(3)
        if velocity_ecef is not None:
            self.mean[3:6] = np.asarray(velocity_ecef, dtype=np.float64).reshape(3)
        self.covariance = np.diag(
            [float(position_sigma_m) ** 2] * 3 + [float(velocity_sigma_mps) ** 2] * 3
        )
        self.accel_process_sigma_mps2 = float(accel_process_sigma_mps2)
        self.ambiguity_init_sigma_cycles = float(ambiguity_init_sigma_cycles)
        self.ambiguity_process_sigma_cycles = float(ambiguity_process_sigma_cycles)
        self.max_track_age_epochs = int(max_track_age_epochs)
        self._tracks: dict[AmbiguityKey, _Track] = {}
        self._generation_by_pair: dict[AmbiguityKey, int] = {}
        self.epoch = -1

    @property
    def position_ecef(self) -> np.ndarray:
        return self.mean[:3].copy()

    @property
    def velocity_ecef(self) -> np.ndarray:
        return self.mean[3:6].copy()

    def predict(self, dt: float) -> None:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.epoch += 1
        n = self.mean.size
        transition = np.eye(n, dtype=np.float64)
        transition[:3, 3:6] = np.eye(3) * dt
        self.mean = transition @ self.mean

        q = np.zeros((n, n), dtype=np.float64)
        accel_var = self.accel_process_sigma_mps2**2
        q[:3, :3] = np.eye(3) * (0.25 * dt**4 * accel_var)
        q[:3, 3:6] = np.eye(3) * (0.5 * dt**3 * accel_var)
        q[3:6, :3] = q[:3, 3:6]
        q[3:6, 3:6] = np.eye(3) * (dt**2 * accel_var)
        if n > self.NAV_DIM:
            q[self.NAV_DIM :, self.NAV_DIM :] = np.eye(n - self.NAV_DIM) * (
                self.ambiguity_process_sigma_cycles**2 * dt
            )
        self.covariance = _symmetrize_psd(
            transition @ self.covariance @ transition.T + q
        )
        self._drop_stale_tracks()

    def update_velocity(
        self,
        velocity_ecef: np.ndarray,
        *,
        sigma_mps: float | np.ndarray,
    ) -> FloatUpdateDiagnostics:
        observation = np.asarray(velocity_ecef, dtype=np.float64).reshape(3)
        design = np.zeros((3, self.mean.size), dtype=np.float64)
        design[:, 3:6] = np.eye(3)
        sigma = np.broadcast_to(np.asarray(sigma_mps, dtype=np.float64), (3,))
        return self._update(design, observation - self.mean[3:6], sigma**2)

    def update_pseudorange(
        self,
        dd_result,
        *,
        sigma_pr_m: float = 5.0,
        huber_k_sigma: float = 2.5,
    ) -> FloatUpdateDiagnostics:
        n = int(getattr(dd_result, "n_dd", 0))
        if n < 1:
            return self._empty_diagnostics(n)
        expected, pos_design = _dd_geometry_and_design(dd_result, self.mean[:3])
        residual = np.asarray(dd_result.dd_pseudorange_m, dtype=np.float64) - expected
        design = np.zeros((n, self.mean.size), dtype=np.float64)
        design[:, :3] = pos_design
        weights = np.clip(np.asarray(dd_result.dd_weights, dtype=np.float64), 1.0e-6, None)
        variance = float(sigma_pr_m) ** 2 / weights
        return self._update(design, residual, variance, huber_k_sigma=huber_k_sigma)

    def update_carrier(
        self,
        dd_result,
        *,
        dd_pseudorange_result=None,
        sigma_cp_cycles: float = 0.10,
        huber_k_sigma: float = 3.0,
        slip_threshold_cycles: float = 2.0,
    ) -> FloatUpdateDiagnostics:
        n = int(getattr(dd_result, "n_dd", 0))
        if n < 1:
            return self._empty_diagnostics(n)
        keys = _pair_keys(dd_result)
        pr_map = _pr_by_pair(dd_pseudorange_result)
        expected, _ = _dd_geometry_and_design(dd_result, self.mean[:3])
        wavelengths = np.asarray(dd_result.wavelengths_m, dtype=np.float64)
        carrier_cycles = np.asarray(dd_result.dd_carrier_cycles, dtype=np.float64)

        new_keys: set[AmbiguityKey] = set()
        for i, key in enumerate(keys):
            if key not in self._tracks:
                pair_pr = pr_map.get((key[0], key[1]))
                if pair_pr is not None and np.isfinite(pair_pr):
                    # CP is already in cycles.  CP - PR/lambda cancels the
                    # same-epoch geometry/clock term and gives a robust root.
                    seed_cycles = carrier_cycles[i] - pair_pr / wavelengths[i]
                else:
                    seed_cycles = carrier_cycles[i] - expected[i] / wavelengths[i]
                self._add_track(key, float(seed_cycles))
                new_keys.add(key)
            self._tracks[key].last_seen_epoch = self.epoch

        # Adding states changes the state dimension, so recompute geometry and
        # the carrier design against the current mean.
        expected, pos_design = _dd_geometry_and_design(dd_result, self.mean[:3])
        n_reset = 0
        if float(slip_threshold_cycles) > 0.0:
            for i, key in enumerate(keys):
                if key in new_keys:
                    continue
                idx = self._tracks[key].index
                prefit_cycles = (
                    carrier_cycles[i]
                    - expected[i] / wavelengths[i]
                    - self.mean[idx]
                )
                if abs(float(prefit_cycles)) > float(slip_threshold_cycles):
                    pair_pr = pr_map.get((key[0], key[1]))
                    seed_cycles = (
                        carrier_cycles[i] - pair_pr / wavelengths[i]
                        if pair_pr is not None and np.isfinite(pair_pr)
                        else carrier_cycles[i] - expected[i] / wavelengths[i]
                    )
                    self._reset_track(key, float(seed_cycles))
                    n_reset += 1

        design = np.zeros((n, self.mean.size), dtype=np.float64)
        design[:, :3] = pos_design
        predicted = expected.copy()
        for i, key in enumerate(keys):
            idx = self._tracks[key].index
            design[i, idx] = wavelengths[i]
            predicted[i] += wavelengths[i] * self.mean[idx]
        observation_m = carrier_cycles * wavelengths
        residual = observation_m - predicted
        weights = np.clip(np.asarray(dd_result.dd_weights, dtype=np.float64), 1.0e-6, None)
        variance = (float(sigma_cp_cycles) * wavelengths) ** 2 / weights
        diagnostics = self._update(
            design, residual, variance, huber_k_sigma=huber_k_sigma
        )
        return replace(diagnostics, ambiguities_reset=n_reset)

    def ambiguity_seed(
        self,
        keys: Iterable[AmbiguityKey] | None = None,
    ) -> FloatAmbiguitySeed:
        selected = tuple(self._tracks if keys is None else keys)
        if not selected:
            return FloatAmbiguitySeed(
                keys=(),
                ahat_cycles=np.zeros(0),
                qahat_cycles2=np.zeros((0, 0)),
                position_ambiguity_cov=np.zeros((3, 0)),
            )
        missing = [key for key in selected if key not in self._tracks]
        if missing:
            raise KeyError(f"unknown ambiguity keys: {missing}")
        indices = np.array([self._tracks[key].index for key in selected], dtype=np.int64)
        qahat = self.covariance[np.ix_(indices, indices)]
        return FloatAmbiguitySeed(
            keys=selected,
            ahat_cycles=self.mean[indices].copy(),
            qahat_cycles2=_symmetrize_psd(qahat),
            position_ambiguity_cov=self.covariance[np.ix_(np.arange(3), indices)].copy(),
        )

    def condition_position_on_integers(
        self,
        keys: Iterable[AmbiguityKey],
        integers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Condition the navigation Gaussian on one integer basin."""

        seed = self.ambiguity_seed(keys)
        fixed = np.asarray(integers, dtype=np.float64).reshape(-1)
        if fixed.size != len(seed.keys):
            raise ValueError("integer vector size does not match ambiguity keys")
        if fixed.size == 0:
            return self.position_ecef, self.covariance[:3, :3].copy(), 0.0
        innovation = fixed - seed.ahat_cycles
        try:
            solved = np.linalg.solve(seed.qahat_cycles2, innovation)
            gain = np.linalg.solve(
                seed.qahat_cycles2,
                seed.position_ambiguity_cov.T,
            ).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("ambiguity covariance is singular") from exc
        position = self.mean[:3] + seed.position_ambiguity_cov @ solved
        covariance = self.covariance[:3, :3] - gain @ seed.position_ambiguity_cov.T
        mahalanobis = float(innovation @ solved)
        return position, _symmetrize_psd(covariance), mahalanobis

    def release(self, keys: Iterable[AmbiguityKey]) -> None:
        remove = {key for key in keys if key in self._tracks}
        if remove:
            self._remove_tracks(remove)

    def _add_track(self, key: AmbiguityKey, seed_cycles: float) -> None:
        old_n = self.mean.size
        self.mean = np.append(self.mean, float(seed_cycles))
        expanded = np.zeros((old_n + 1, old_n + 1), dtype=np.float64)
        expanded[:old_n, :old_n] = self.covariance
        expanded[old_n, old_n] = self.ambiguity_init_sigma_cycles**2
        self.covariance = expanded
        generation = self._generation_by_pair.get(key, -1) + 1
        self._generation_by_pair[key] = generation
        self._tracks[key] = _Track(old_n, generation, self.epoch)

    def _reset_track(self, key: AmbiguityKey, seed_cycles: float) -> None:
        track = self._tracks[key]
        idx = int(track.index)
        self.mean[idx] = float(seed_cycles)
        self.covariance[idx, :] = 0.0
        self.covariance[:, idx] = 0.0
        self.covariance[idx, idx] = self.ambiguity_init_sigma_cycles**2
        generation = self._generation_by_pair.get(key, track.generation) + 1
        self._generation_by_pair[key] = generation
        track.generation = generation
        track.last_seen_epoch = self.epoch

    def _drop_stale_tracks(self) -> None:
        if self.max_track_age_epochs < 0:
            return
        stale = {
            key
            for key, track in self._tracks.items()
            if self.epoch - track.last_seen_epoch > self.max_track_age_epochs
        }
        if stale:
            self._remove_tracks(stale)

    def _remove_tracks(self, remove: set[AmbiguityKey]) -> None:
        keep_keys = [key for key in self._tracks if key not in remove]
        keep_indices = list(range(self.NAV_DIM)) + [self._tracks[key].index for key in keep_keys]
        self.mean = self.mean[np.asarray(keep_indices, dtype=np.int64)]
        self.covariance = self.covariance[np.ix_(keep_indices, keep_indices)]
        self._tracks = {
            key: _Track(self.NAV_DIM + i, self._tracks[key].generation, self._tracks[key].last_seen_epoch)
            for i, key in enumerate(keep_keys)
        }

    def _update(
        self,
        design: np.ndarray,
        residual: np.ndarray,
        variance: np.ndarray,
        *,
        huber_k_sigma: float = 0.0,
    ) -> FloatUpdateDiagnostics:
        h = np.asarray(design, dtype=np.float64)
        innovation = np.asarray(residual, dtype=np.float64).reshape(-1)
        measurement_var = np.asarray(variance, dtype=np.float64).reshape(-1)
        valid = (
            np.all(np.isfinite(h), axis=1)
            & np.isfinite(innovation)
            & np.isfinite(measurement_var)
            & (measurement_var > 0.0)
        )
        h = h[valid]
        innovation = innovation[valid]
        measurement_var = measurement_var[valid]
        if innovation.size == 0:
            return self._empty_diagnostics(int(np.asarray(residual).size))

        predicted_var = np.einsum("ij,jk,ik->i", h, self.covariance, h)
        standardized = np.abs(innovation) / np.sqrt(np.maximum(predicted_var + measurement_var, 1e-12))
        if float(huber_k_sigma) > 0.0:
            robust_weight = np.minimum(
                1.0,
                float(huber_k_sigma) / np.maximum(standardized, 1.0e-12),
            )
            measurement_var = measurement_var / np.maximum(robust_weight, 1.0e-6) ** 2

        rmat = np.diag(measurement_var)
        innovation_cov = _symmetrize_psd(h @ self.covariance @ h.T + rmat)
        try:
            solved_residual = np.linalg.solve(innovation_cov, innovation)
            gain = np.linalg.solve(innovation_cov, h @ self.covariance).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("float KF innovation covariance is singular") from exc
        prior_cov = self.covariance
        self.mean = self.mean + gain @ innovation
        identity_minus_kh = np.eye(self.mean.size) - gain @ h
        self.covariance = _symmetrize_psd(
            identity_minus_kh @ prior_cov @ identity_minus_kh.T + gain @ rmat @ gain.T
        )
        return FloatUpdateDiagnostics(
            n_input=int(np.asarray(residual).size),
            n_used=int(innovation.size),
            innovation_rms=float(np.sqrt(np.mean(innovation**2))),
            normalized_innovation_sq=float(innovation @ solved_residual / innovation.size),
            covariance_min_eig=float(np.min(np.linalg.eigvalsh(self.covariance))),
        )

    def _empty_diagnostics(self, n_input: int) -> FloatUpdateDiagnostics:
        return FloatUpdateDiagnostics(
            n_input=int(n_input),
            n_used=0,
            innovation_rms=float("nan"),
            normalized_innovation_sq=float("nan"),
            covariance_min_eig=float(np.min(np.linalg.eigvalsh(self.covariance))),
        )
