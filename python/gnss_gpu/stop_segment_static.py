"""Static GNSS refinement for IMU stop-detected trajectory segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch, UndiffPseudorangeEpoch


@dataclass(frozen=True)
class StaticStopSegmentConfig:
    min_epochs: int = 5
    min_observations: int = 40
    prior_sigma_m: float = 20.0
    undiff_pr_sigma_m: float = 8.0
    dd_pr_sigma_m: float = 4.0
    dd_cp_sigma_cycles: float = 0.50
    huber_k: float = 1.5
    max_iterations: int = 15
    max_update_m: float | None = 25.0
    blend: float = 1.0
    min_weight: float = 1e-3


@dataclass(frozen=True)
class StaticStopSegmentSolve:
    position_ecef: np.ndarray
    applied: bool
    reason: str
    iterations: int
    n_observations: int
    n_undiff_pr: int
    n_dd_pr: int
    n_dd_cp: int
    initial_cost: float
    final_cost: float
    final_norm_rms: float
    update_norm_m: float


def apply_static_stop_segment_gnss(
    smoothed_aligned: np.ndarray,
    stop_flags: Sequence[bool] | np.ndarray,
    dd_carrier: Sequence[DDCarrierEpoch | None],
    dd_pseudorange: Sequence[DDPseudorangeEpoch | None],
    undiff_pseudorange: Sequence[UndiffPseudorangeEpoch | None],
    epoch_diagnostics: list[dict[str, object]] | None = None,
    *,
    config: StaticStopSegmentConfig | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    cfg = StaticStopSegmentConfig() if config is None else config
    corrected = np.asarray(smoothed_aligned, dtype=np.float64).copy()
    ranges = _stop_segment_ranges(stop_flags, min_epochs=cfg.min_epochs)
    info: dict[str, object] = {
        "segments": int(len(ranges)),
        "segments_applied": 0,
        "epochs_applied": 0,
        "min_observations": int(cfg.min_observations),
    }
    if epoch_diagnostics is not None:
        for row in epoch_diagnostics:
            row["stop_segment_static_applied"] = False
            row["stop_segment_static_id"] = None
            row["stop_segment_static_shift_m"] = None
            row["stop_segment_static_norm_rms"] = None
            row["stop_segment_static_reason"] = None

    alpha = float(np.clip(float(cfg.blend), 0.0, 1.0))
    for seg_id, (start, end) in enumerate(ranges):
        if start < 0 or end > len(corrected) or end <= start:
            continue
        samples = corrected[start:end]
        finite = np.isfinite(samples).all(axis=1)
        if np.count_nonzero(finite) < max(1, int(cfg.min_epochs)):
            solve = _empty_solve(samples, "not_enough_finite_positions")
        else:
            initial = np.median(samples[finite], axis=0)
            solve = solve_static_stop_segment(
                initial,
                dd_carrier[start:end],
                dd_pseudorange[start:end],
                undiff_pseudorange[start:end],
                cfg,
            )
        accepted = bool(solve.applied)
        if accepted:
            target = np.asarray(solve.position_ecef, dtype=np.float64)
            corrected[start:end] = (1.0 - alpha) * corrected[start:end] + alpha * target
            info["segments_applied"] = int(info["segments_applied"]) + 1
            info["epochs_applied"] = int(info["epochs_applied"]) + int(end - start)
        if epoch_diagnostics is not None:
            for i in range(start, end):
                epoch_diagnostics[i]["stop_segment_static_applied"] = accepted
                epoch_diagnostics[i]["stop_segment_static_id"] = int(seg_id)
                epoch_diagnostics[i]["stop_segment_static_shift_m"] = float(solve.update_norm_m)
                epoch_diagnostics[i]["stop_segment_static_norm_rms"] = float(solve.final_norm_rms)
                epoch_diagnostics[i]["stop_segment_static_reason"] = str(solve.reason)
    return corrected, info


def solve_static_stop_segment(
    initial_position_ecef: np.ndarray,
    dd_carrier: Sequence[DDCarrierEpoch | None],
    dd_pseudorange: Sequence[DDPseudorangeEpoch | None],
    undiff_pseudorange: Sequence[UndiffPseudorangeEpoch | None],
    config: StaticStopSegmentConfig | None = None,
) -> StaticStopSegmentSolve:
    cfg = StaticStopSegmentConfig() if config is None else config
    initial = np.asarray(initial_position_ecef, dtype=np.float64).ravel()[:3]
    if initial.shape != (3,) or not np.isfinite(initial).all():
        return _empty_solve(initial, "invalid_initial")

    x = initial.copy()
    initial_cost, n_obs, counts = _evaluate_static_cost(
        x,
        initial,
        dd_carrier,
        dd_pseudorange,
        undiff_pseudorange,
        cfg,
    )
    if n_obs < int(cfg.min_observations):
        return _solve_result(
            x,
            False,
            "not_enough_observations",
            0,
            n_obs,
            counts,
            initial_cost,
            initial_cost,
            initial,
        )

    cost = initial_cost
    iterations = 0
    reason = "max_iterations"
    for iteration in range(max(1, int(cfg.max_iterations))):
        hessian, gradient, lin_cost, n_lin, counts = _linearize_static_system(
            x,
            initial,
            dd_carrier,
            dd_pseudorange,
            undiff_pseudorange,
            cfg,
        )
        if n_lin < int(cfg.min_observations):
            reason = "not_enough_observations"
            break
        if not np.isfinite(hessian).all() or not np.isfinite(gradient).all():
            reason = "nonfinite_normal_equations"
            break
        damped = hessian + np.eye(3, dtype=np.float64) * 1e-6
        try:
            delta = -np.linalg.solve(damped, gradient)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(damped, gradient, rcond=None)[0]
        if not np.isfinite(delta).all():
            reason = "nonfinite_step"
            break
        iterations = iteration + 1
        if float(np.linalg.norm(delta)) < 1e-4:
            cost = lin_cost
            reason = "converged"
            break

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            candidate = x + float(scale) * delta
            candidate_cost, candidate_n, candidate_counts = _evaluate_static_cost(
                candidate,
                initial,
                dd_carrier,
                dd_pseudorange,
                undiff_pseudorange,
                cfg,
            )
            if candidate_n >= int(cfg.min_observations) and (
                candidate_cost <= cost or not np.isfinite(cost)
            ):
                x = candidate
                cost = candidate_cost
                counts = candidate_counts
                accepted = True
                break
        if not accepted:
            reason = "no_descent"
            break
    else:
        cost, n_obs, counts = _evaluate_static_cost(
            x,
            initial,
            dd_carrier,
            dd_pseudorange,
            undiff_pseudorange,
            cfg,
        )

    final_cost, n_obs, counts = _evaluate_static_cost(
        x,
        initial,
        dd_carrier,
        dd_pseudorange,
        undiff_pseudorange,
        cfg,
    )
    update_norm = float(np.linalg.norm(x - initial))
    applied = bool(np.isfinite(x).all() and n_obs >= int(cfg.min_observations))
    if cfg.max_update_m is not None and update_norm > float(cfg.max_update_m):
        applied = False
        reason = "max_update"
    if final_cost > initial_cost and np.isfinite(initial_cost):
        applied = False
        reason = "cost_increased"
    if reason == "max_iterations" and applied:
        reason = "ok"
    return _solve_result(
        x,
        applied,
        reason,
        iterations,
        n_obs,
        counts,
        initial_cost,
        final_cost,
        initial,
    )


def _linearize_static_system(
    x: np.ndarray,
    prior: np.ndarray,
    dd_carrier: Sequence[DDCarrierEpoch | None],
    dd_pseudorange: Sequence[DDPseudorangeEpoch | None],
    undiff_pseudorange: Sequence[UndiffPseudorangeEpoch | None],
    cfg: StaticStopSegmentConfig,
) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, int]]:
    hessian = np.zeros((3, 3), dtype=np.float64)
    gradient = np.zeros(3, dtype=np.float64)
    state = _Accumulator()
    _accumulate_prior(hessian, gradient, state, x, prior, cfg)
    _accumulate_undiff_pr(hessian, gradient, state, x, undiff_pseudorange, cfg)
    _accumulate_dd_pr(hessian, gradient, state, x, dd_pseudorange, cfg)
    _accumulate_dd_cp(hessian, gradient, state, x, dd_carrier, cfg)
    return hessian, gradient, state.cost, state.n, state.counts


def _evaluate_static_cost(
    x: np.ndarray,
    prior: np.ndarray,
    dd_carrier: Sequence[DDCarrierEpoch | None],
    dd_pseudorange: Sequence[DDPseudorangeEpoch | None],
    undiff_pseudorange: Sequence[UndiffPseudorangeEpoch | None],
    cfg: StaticStopSegmentConfig,
) -> tuple[float, int, dict[str, int]]:
    _hessian, _gradient, cost, n_obs, counts = _linearize_static_system(
        x,
        prior,
        dd_carrier,
        dd_pseudorange,
        undiff_pseudorange,
        cfg,
    )
    return cost, n_obs, counts


@dataclass
class _Accumulator:
    cost: float = 0.0
    n: int = 0
    counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = {"undiff_pr": 0, "dd_pr": 0, "dd_cp": 0}


def _accumulate_prior(
    hessian: np.ndarray,
    gradient: np.ndarray,
    state: _Accumulator,
    x: np.ndarray,
    prior: np.ndarray,
    cfg: StaticStopSegmentConfig,
) -> None:
    sigma = float(cfg.prior_sigma_m)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return
    for axis in range(3):
        jac = np.zeros(3, dtype=np.float64)
        jac[axis] = 1.0
        _add_residual(
            hessian,
            gradient,
            state,
            float(x[axis] - prior[axis]),
            jac,
            sigma,
            cfg.huber_k,
            None,
        )


def _accumulate_undiff_pr(
    hessian: np.ndarray,
    gradient: np.ndarray,
    state: _Accumulator,
    x: np.ndarray,
    epochs: Sequence[UndiffPseudorangeEpoch | None],
    cfg: StaticStopSegmentConfig,
) -> None:
    for obs in epochs:
        if obs is None:
            continue
        sat = np.asarray(obs.sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(obs.pseudoranges_m, dtype=np.float64).ravel()
        weights = _weights(obs.weights, len(pr), cfg.min_weight)
        n = min(len(sat), len(pr), len(weights))
        if n < 4:
            continue
        sat = sat[:n]
        pr = pr[:n]
        weights = weights[:n]
        valid = np.isfinite(sat).all(axis=1) & np.isfinite(pr) & np.isfinite(weights)
        if np.count_nonzero(valid) < 4:
            continue
        sat = sat[valid]
        pr = pr[valid]
        weights = weights[valid]
        vec = x.reshape(1, 3) - sat
        ranges = np.linalg.norm(vec, axis=1)
        good = np.isfinite(ranges) & (ranges > 1e3)
        if np.count_nonzero(good) < 4:
            continue
        sat = sat[good]
        pr = pr[good]
        weights = np.maximum(weights[good], float(cfg.min_weight))
        ranges = ranges[good]
        jac = vec[good] / ranges.reshape(-1, 1)
        w_sum = float(np.sum(weights))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            continue
        clock_bias = float(np.sum(weights * (pr - ranges)) / w_sum)
        mean_jac = np.sum(weights.reshape(-1, 1) * jac, axis=0) / w_sum
        for rng, pseudorange, weight, jac_row in zip(ranges, pr, weights, jac):
            sigma = float(cfg.undiff_pr_sigma_m) / float(np.sqrt(max(weight, cfg.min_weight)))
            _add_residual(
                hessian,
                gradient,
                state,
                float(rng + clock_bias - pseudorange),
                np.asarray(jac_row - mean_jac, dtype=np.float64),
                sigma,
                cfg.huber_k,
                "undiff_pr",
            )


def _accumulate_dd_pr(
    hessian: np.ndarray,
    gradient: np.ndarray,
    state: _Accumulator,
    x: np.ndarray,
    epochs: Sequence[DDPseudorangeEpoch | None],
    cfg: StaticStopSegmentConfig,
) -> None:
    for obs in epochs:
        if obs is None:
            continue
        dd = np.asarray(obs.dd_pseudorange_m, dtype=np.float64).ravel()
        sat_k = np.asarray(obs.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
        sat_ref = np.asarray(obs.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
        base_k = np.asarray(obs.base_range_k, dtype=np.float64).ravel()
        base_ref = np.asarray(obs.base_range_ref, dtype=np.float64).ravel()
        weights = _weights(obs.weights, len(dd), cfg.min_weight)
        n = min(len(dd), len(sat_k), len(sat_ref), len(base_k), len(base_ref), len(weights))
        for i in range(n):
            if not _valid_dd_row(dd[i], sat_k[i], sat_ref[i], base_k[i], base_ref[i], 1.0):
                continue
            expected, jac = _dd_expected_and_jacobian_m(x, sat_k[i], sat_ref[i], base_k[i], base_ref[i])
            sigma = float(cfg.dd_pr_sigma_m) / float(np.sqrt(max(weights[i], cfg.min_weight)))
            _add_residual(
                hessian,
                gradient,
                state,
                float(expected - dd[i]),
                jac,
                sigma,
                cfg.huber_k,
                "dd_pr",
            )


def _accumulate_dd_cp(
    hessian: np.ndarray,
    gradient: np.ndarray,
    state: _Accumulator,
    x: np.ndarray,
    epochs: Sequence[DDCarrierEpoch | None],
    cfg: StaticStopSegmentConfig,
) -> None:
    sigma_base = float(cfg.dd_cp_sigma_cycles)
    if not np.isfinite(sigma_base) or sigma_base <= 0.0:
        return
    for obs in epochs:
        if obs is None:
            continue
        dd = np.asarray(obs.dd_carrier_cycles, dtype=np.float64).ravel()
        sat_k = np.asarray(obs.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
        sat_ref = np.asarray(obs.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
        base_k = np.asarray(obs.base_range_k, dtype=np.float64).ravel()
        base_ref = np.asarray(obs.base_range_ref, dtype=np.float64).ravel()
        wavelengths = np.asarray(obs.wavelengths_m, dtype=np.float64).ravel()
        weights = _weights(obs.weights, len(dd), cfg.min_weight)
        n = min(len(dd), len(sat_k), len(sat_ref), len(base_k), len(base_ref), len(wavelengths), len(weights))
        for i in range(n):
            if not _valid_dd_row(dd[i], sat_k[i], sat_ref[i], base_k[i], base_ref[i], wavelengths[i]):
                continue
            expected, jac_m = _dd_expected_and_jacobian_m(
                x,
                sat_k[i],
                sat_ref[i],
                base_k[i],
                base_ref[i],
            )
            raw = float(dd[i] - expected / wavelengths[i])
            residual = raw - float(np.round(raw))
            jac = -jac_m / float(wavelengths[i])
            sigma = sigma_base / float(np.sqrt(max(weights[i], cfg.min_weight)))
            _add_residual(
                hessian,
                gradient,
                state,
                residual,
                jac,
                sigma,
                cfg.huber_k,
                "dd_cp",
            )


def _add_residual(
    hessian: np.ndarray,
    gradient: np.ndarray,
    state: _Accumulator,
    residual: float,
    jacobian: np.ndarray,
    sigma: float,
    huber_k: float,
    count_key: str | None,
) -> None:
    if not (np.isfinite(residual) and np.isfinite(jacobian).all()):
        return
    if not np.isfinite(sigma) or sigma <= 0.0:
        return
    normalized = float(residual) / float(sigma)
    jac = np.asarray(jacobian, dtype=np.float64).ravel()[:3] / float(sigma)
    weight, cost = _huber_weight_cost(normalized, huber_k)
    hessian += weight * np.outer(jac, jac)
    gradient += weight * jac * normalized
    state.cost += cost
    state.n += 1
    if count_key is not None and state.counts is not None:
        state.counts[count_key] = int(state.counts.get(count_key, 0)) + 1


def _huber_weight_cost(value: float, huber_k: float) -> tuple[float, float]:
    abs_value = abs(float(value))
    k = float(huber_k)
    if not np.isfinite(k) or k <= 0.0 or abs_value <= k:
        return 1.0, 0.5 * abs_value * abs_value
    return k / max(abs_value, 1e-12), k * (abs_value - 0.5 * k)


def _dd_expected_and_jacobian_m(
    x: np.ndarray,
    sat_k: np.ndarray,
    sat_ref: np.ndarray,
    base_k: float,
    base_ref: float,
) -> tuple[float, np.ndarray]:
    rng_k, jac_k = _range_and_jacobian(x, sat_k)
    rng_ref, jac_ref = _range_and_jacobian(x, sat_ref)
    return float(rng_k - rng_ref - base_k + base_ref), jac_k - jac_ref


def _range_and_jacobian(x: np.ndarray, sat: np.ndarray) -> tuple[float, np.ndarray]:
    vec = np.asarray(x, dtype=np.float64) - np.asarray(sat, dtype=np.float64)
    rng = float(np.linalg.norm(vec))
    if not np.isfinite(rng) or rng <= 1e-9:
        return rng, np.zeros(3, dtype=np.float64)
    return rng, vec / rng


def _weights(values: np.ndarray | None, n: int, min_weight: float) -> np.ndarray:
    if values is None:
        return np.ones(int(n), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).ravel()
    if len(arr) != int(n):
        return np.ones(int(n), dtype=np.float64)
    out = arr.astype(np.float64, copy=True)
    out[~np.isfinite(out)] = float(min_weight)
    return np.maximum(out, float(min_weight))


def _valid_dd_row(
    observed: float,
    sat_k: np.ndarray,
    sat_ref: np.ndarray,
    base_k: float,
    base_ref: float,
    wavelength: float,
) -> bool:
    return bool(
        np.isfinite(observed)
        and np.isfinite(sat_k).all()
        and np.isfinite(sat_ref).all()
        and np.isfinite(base_k)
        and np.isfinite(base_ref)
        and np.isfinite(wavelength)
        and float(wavelength) > 0.0
    )


def _stop_segment_ranges(stop_flags: Sequence[bool] | np.ndarray, *, min_epochs: int) -> list[tuple[int, int]]:
    flags = np.asarray(stop_flags, dtype=bool).ravel()
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_stop in enumerate(flags):
        if is_stop:
            if start is None:
                start = i
            continue
        if start is not None and i - start >= int(min_epochs):
            ranges.append((start, i))
        start = None
    if start is not None and len(flags) - start >= int(min_epochs):
        ranges.append((start, len(flags)))
    return ranges


def _solve_result(
    position: np.ndarray,
    applied: bool,
    reason: str,
    iterations: int,
    n_observations: int,
    counts: dict[str, int],
    initial_cost: float,
    final_cost: float,
    initial: np.ndarray,
) -> StaticStopSegmentSolve:
    final_rms = float(np.sqrt(max(0.0, 2.0 * float(final_cost) / max(1, int(n_observations)))))
    return StaticStopSegmentSolve(
        position_ecef=np.asarray(position, dtype=np.float64).ravel()[:3],
        applied=bool(applied),
        reason=str(reason),
        iterations=int(iterations),
        n_observations=int(n_observations),
        n_undiff_pr=int(counts.get("undiff_pr", 0)),
        n_dd_pr=int(counts.get("dd_pr", 0)),
        n_dd_cp=int(counts.get("dd_cp", 0)),
        initial_cost=float(initial_cost),
        final_cost=float(final_cost),
        final_norm_rms=final_rms,
        update_norm_m=float(np.linalg.norm(np.asarray(position, dtype=np.float64).ravel()[:3] - initial)),
    )


def _empty_solve(position: np.ndarray, reason: str) -> StaticStopSegmentSolve:
    pos = np.asarray(position, dtype=np.float64).ravel()
    if pos.size < 3:
        pos = np.zeros(3, dtype=np.float64)
    return StaticStopSegmentSolve(
        position_ecef=pos[:3],
        applied=False,
        reason=str(reason),
        iterations=0,
        n_observations=0,
        n_undiff_pr=0,
        n_dd_pr=0,
        n_dd_cp=0,
        initial_cost=float("nan"),
        final_cost=float("nan"),
        final_norm_rms=float("nan"),
        update_norm_m=float("nan"),
    )
