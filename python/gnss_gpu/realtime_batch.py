"""Persistent CPU/CUDA batches for screen, candidate score, and affine refit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import cuda
except ImportError:  # pragma: no cover - exercised on minimal installs
    cuda = None


def cpu_candidate_rms(residuals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if residuals.ndim != 2 or weights.shape != residuals.shape:
        raise ValueError("candidate residual and weight batches must share shape (B, N)")
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("candidate batch values must be finite with non-negative weights")
    total = np.sum(weights, axis=1)
    result = np.full(residuals.shape[0], np.inf)
    valid = total > 0
    result[valid] = np.sqrt(
        np.sum(weights[valid] * np.square(residuals[valid]), axis=1) / total[valid]
    )
    return result


def cpu_affine_refit(
    epochs: np.ndarray,
    offsets_ecef_m: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray(epochs, dtype=np.float64)
    offsets = np.asarray(offsets_ecef_m, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        epochs.ndim != 1
        or offsets.ndim != 3
        or offsets.shape[1:] != (epochs.size, 3)
        or weights.shape != offsets.shape[:2]
    ):
        raise ValueError("affine batch inputs must have shapes (N), (B,N,3), (B,N)")
    endpoints = np.empty((offsets.shape[0], 2, 3), dtype=np.float64)
    rms = np.empty(offsets.shape[0], dtype=np.float64)
    for index in range(offsets.shape[0]):
        w = weights[index]
        design = np.column_stack((np.ones(epochs.size), epochs))
        coeff, *_ = np.linalg.lstsq(design * np.sqrt(w)[:, None], offsets[index] * np.sqrt(w)[:, None], rcond=None)
        endpoints[index, 0] = coeff[0] + coeff[1] * epochs[0]
        endpoints[index, 1] = coeff[0] + coeff[1] * epochs[-1]
        residual = offsets[index] - design @ coeff
        rms[index] = np.sqrt(np.sum(w * np.sum(residual * residual, axis=1)) / np.sum(w))
    return endpoints, rms


def cpu_arc_outlier_fraction(
    residuals: np.ndarray,
    valid: np.ndarray,
    *,
    edge_m: float,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    if residuals.ndim != 2 or valid.shape != residuals.shape or edge_m <= 0:
        raise ValueError("arc screen inputs must have shapes (E,S) and positive edge")
    epochs, satellites = residuals.shape
    outlier = np.zeros(satellites)
    present = np.zeros(satellites)
    for epoch in range(epochs):
        active = np.flatnonzero(valid[epoch])
        for sat in active:
            peers = active[active != sat]
            if peers.size == 0:
                continue
            present[sat] += 1
            inconsistent = np.count_nonzero(
                np.abs(residuals[epoch, sat] - residuals[epoch, peers]) >= edge_m
            )
            if inconsistent * 2 > peers.size:
                outlier[sat] += 1
    return np.divide(outlier, present, out=np.zeros_like(outlier), where=present > 0)


if cuda is not None:

    @cuda.jit
    def _candidate_rms_kernel(residuals, weights, candidates, observations, output):
        candidate = cuda.grid(1)
        if candidate >= candidates:
            return
        weighted_square = 0.0
        total = 0.0
        for obs in range(observations):
            weight = weights[candidate, obs]
            value = residuals[candidate, obs]
            weighted_square += weight * value * value
            total += weight
        output[candidate] = (weighted_square / total) ** 0.5 if total > 0 else 1.0e300

    @cuda.jit
    def _affine_refit_kernel(epochs, offsets, weights, candidates, observations, endpoints, rms):
        candidate = cuda.grid(1)
        if candidate >= candidates:
            return
        sw = 0.0
        st = 0.0
        stt = 0.0
        sy0 = 0.0
        sy1 = 0.0
        sy2 = 0.0
        sty0 = 0.0
        sty1 = 0.0
        sty2 = 0.0
        for obs in range(observations):
            w = weights[candidate, obs]
            t = epochs[obs]
            sw += w
            st += w * t
            stt += w * t * t
            sy0 += w * offsets[candidate, obs, 0]
            sy1 += w * offsets[candidate, obs, 1]
            sy2 += w * offsets[candidate, obs, 2]
            sty0 += w * t * offsets[candidate, obs, 0]
            sty1 += w * t * offsets[candidate, obs, 1]
            sty2 += w * t * offsets[candidate, obs, 2]
        determinant = sw * stt - st * st
        if determinant <= 1.0e-15:
            rms[candidate] = 1.0e300
            return
        intercept0 = (stt * sy0 - st * sty0) / determinant
        intercept1 = (stt * sy1 - st * sty1) / determinant
        intercept2 = (stt * sy2 - st * sty2) / determinant
        slope0 = (sw * sty0 - st * sy0) / determinant
        slope1 = (sw * sty1 - st * sy1) / determinant
        slope2 = (sw * sty2 - st * sy2) / determinant
        first = epochs[0]
        last = epochs[observations - 1]
        endpoints[candidate, 0, 0] = intercept0 + slope0 * first
        endpoints[candidate, 0, 1] = intercept1 + slope1 * first
        endpoints[candidate, 0, 2] = intercept2 + slope2 * first
        endpoints[candidate, 1, 0] = intercept0 + slope0 * last
        endpoints[candidate, 1, 1] = intercept1 + slope1 * last
        endpoints[candidate, 1, 2] = intercept2 + slope2 * last
        square = 0.0
        for obs in range(observations):
            t = epochs[obs]
            d0 = offsets[candidate, obs, 0] - intercept0 - slope0 * t
            d1 = offsets[candidate, obs, 1] - intercept1 - slope1 * t
            d2 = offsets[candidate, obs, 2] - intercept2 - slope2 * t
            square += weights[candidate, obs] * (d0 * d0 + d1 * d1 + d2 * d2)
        rms[candidate] = (square / sw) ** 0.5 if sw > 0 else 1.0e300

    @cuda.jit
    def _arc_screen_kernel(residuals, valid, epochs, satellites, edge_m, output):
        sat = cuda.grid(1)
        if sat >= satellites:
            return
        present = 0
        outlier = 0
        for epoch in range(epochs):
            if not valid[epoch, sat]:
                continue
            peers = 0
            inconsistent = 0
            for other in range(satellites):
                if other == sat or not valid[epoch, other]:
                    continue
                peers += 1
                if abs(residuals[epoch, sat] - residuals[epoch, other]) >= edge_m:
                    inconsistent += 1
            if peers > 0:
                present += 1
                if inconsistent * 2 > peers:
                    outlier += 1
        output[sat] = outlier / present if present > 0 else 0.0


@dataclass(frozen=True)
class BatchWorkspaceCapacity:
    candidates: int
    observations: int
    epochs: int
    satellites: int


class CudaBatchWorkspace:
    """Persistent device buffers; calls transfer inputs but never allocate."""

    def __init__(self, capacity: BatchWorkspaceCapacity) -> None:
        if cuda is None or not cuda.is_available():
            raise RuntimeError("Numba CUDA is unavailable")
        if min(
            capacity.candidates,
            capacity.observations,
            capacity.epochs,
            capacity.satellites,
        ) < 1:
            raise ValueError("batch workspace capacities must be positive")
        self.capacity = capacity
        self.stream = cuda.stream()
        self.d_candidate_residuals = cuda.device_array(
            (capacity.candidates, capacity.observations), dtype=np.float64, stream=self.stream
        )
        self.d_candidate_weights = cuda.device_array_like(self.d_candidate_residuals)
        self.d_candidate_rms = cuda.device_array(capacity.candidates, dtype=np.float64)
        self.d_epochs = cuda.device_array(capacity.observations, dtype=np.float64)
        self.d_offsets = cuda.device_array(
            (capacity.candidates, capacity.observations, 3), dtype=np.float64
        )
        self.d_affine_endpoints = cuda.device_array(
            (capacity.candidates, 2, 3), dtype=np.float64
        )
        self.d_affine_rms = cuda.device_array(capacity.candidates, dtype=np.float64)
        self.d_screen_residuals = cuda.device_array(
            (capacity.epochs, capacity.satellites), dtype=np.float64
        )
        self.d_screen_valid = cuda.device_array(
            (capacity.epochs, capacity.satellites), dtype=np.bool_
        )
        self.d_outlier_fraction = cuda.device_array(capacity.satellites, dtype=np.float64)
        self.h_candidate_residuals = cuda.pinned_array(
            (capacity.candidates, capacity.observations), dtype=np.float64
        )
        self.h_candidate_weights = cuda.pinned_array(
            (capacity.candidates, capacity.observations), dtype=np.float64
        )
        self.h_candidate_rms = cuda.pinned_array(capacity.candidates, dtype=np.float64)
        self.h_epochs = cuda.pinned_array(capacity.observations, dtype=np.float64)
        self.h_offsets = cuda.pinned_array(
            (capacity.candidates, capacity.observations, 3), dtype=np.float64
        )
        self.h_affine_endpoints = cuda.pinned_array(
            (capacity.candidates, 2, 3), dtype=np.float64
        )
        self.h_affine_rms = cuda.pinned_array(capacity.candidates, dtype=np.float64)
        self.h_screen_residuals = cuda.pinned_array(
            (capacity.epochs, capacity.satellites), dtype=np.float64
        )
        self.h_screen_valid = cuda.pinned_array(
            (capacity.epochs, capacity.satellites), dtype=np.bool_
        )
        self.h_outlier_fraction = cuda.pinned_array(capacity.satellites, dtype=np.float64)

    def candidate_rms(self, residuals: np.ndarray, weights: np.ndarray) -> np.ndarray:
        residuals = np.ascontiguousarray(residuals, dtype=np.float64)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        candidates, observations = residuals.shape
        if weights.shape != residuals.shape or candidates > self.capacity.candidates or observations > self.capacity.observations:
            raise ValueError("candidate batch exceeds workspace")
        self.h_candidate_residuals.fill(0.0)
        self.h_candidate_weights.fill(0.0)
        self.h_candidate_residuals[:candidates, :observations] = residuals
        self.h_candidate_weights[:candidates, :observations] = weights
        self.d_candidate_residuals.copy_to_device(
            self.h_candidate_residuals, stream=self.stream
        )
        self.d_candidate_weights.copy_to_device(
            self.h_candidate_weights, stream=self.stream
        )
        _candidate_rms_kernel[(candidates + 127) // 128, 128, self.stream](
            self.d_candidate_residuals,
            self.d_candidate_weights,
            candidates,
            observations,
            self.d_candidate_rms,
        )
        self.d_candidate_rms.copy_to_host(self.h_candidate_rms, stream=self.stream)
        self.stream.synchronize()
        return self.h_candidate_rms[:candidates].copy()

    def affine_refit(
        self,
        epochs: np.ndarray,
        offsets: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        epochs = np.ascontiguousarray(epochs, dtype=np.float64)
        offsets = np.ascontiguousarray(offsets, dtype=np.float64)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        candidates, observations, axes = offsets.shape
        if (
            axes != 3
            or epochs.shape != (observations,)
            or weights.shape != (candidates, observations)
            or candidates > self.capacity.candidates
            or observations > self.capacity.observations
        ):
            raise ValueError("affine batch exceeds workspace")
        self.h_epochs.fill(0.0)
        self.h_offsets.fill(0.0)
        self.h_candidate_weights.fill(0.0)
        self.h_epochs[:observations] = epochs
        self.h_offsets[:candidates, :observations] = offsets
        self.h_candidate_weights[:candidates, :observations] = weights
        self.d_epochs.copy_to_device(self.h_epochs, stream=self.stream)
        self.d_offsets.copy_to_device(self.h_offsets, stream=self.stream)
        self.d_candidate_weights.copy_to_device(
            self.h_candidate_weights, stream=self.stream
        )
        _affine_refit_kernel[(candidates + 127) // 128, 128, self.stream](
            self.d_epochs,
            self.d_offsets,
            self.d_candidate_weights,
            candidates,
            observations,
            self.d_affine_endpoints,
            self.d_affine_rms,
        )
        self.d_affine_endpoints.copy_to_host(
            self.h_affine_endpoints, stream=self.stream
        )
        self.d_affine_rms.copy_to_host(self.h_affine_rms, stream=self.stream)
        self.stream.synchronize()
        return (
            self.h_affine_endpoints[:candidates].copy(),
            self.h_affine_rms[:candidates].copy(),
        )

    def arc_outlier_fraction(
        self,
        residuals: np.ndarray,
        valid: np.ndarray,
        *,
        edge_m: float,
    ) -> np.ndarray:
        residuals = np.ascontiguousarray(residuals, dtype=np.float64)
        valid = np.ascontiguousarray(valid, dtype=np.bool_)
        epochs, satellites = residuals.shape
        if (
            valid.shape != residuals.shape
            or epochs > self.capacity.epochs
            or satellites > self.capacity.satellites
            or edge_m <= 0
        ):
            raise ValueError("arc batch exceeds workspace")
        self.h_screen_residuals.fill(0.0)
        self.h_screen_valid.fill(False)
        self.h_screen_residuals[:epochs, :satellites] = residuals
        self.h_screen_valid[:epochs, :satellites] = valid
        self.d_screen_residuals.copy_to_device(
            self.h_screen_residuals, stream=self.stream
        )
        self.d_screen_valid.copy_to_device(self.h_screen_valid, stream=self.stream)
        _arc_screen_kernel[(satellites + 127) // 128, 128, self.stream](
            self.d_screen_residuals,
            self.d_screen_valid,
            epochs,
            satellites,
            edge_m,
            self.d_outlier_fraction,
        )
        self.d_outlier_fraction.copy_to_host(
            self.h_outlier_fraction, stream=self.stream
        )
        self.stream.synchronize()
        return self.h_outlier_fraction[:satellites].copy()

    def synchronize(self) -> None:
        self.stream.synchronize()
