"""Truth-free static-position solver with one carrier integer per DD arc.

The legacy static solver wraps every carrier row independently.  That is useful
for local refinement, but it removes the temporal constraint that a carrier
ambiguity is constant while a satellite pair remains continuous.  This module
keeps that constraint explicit and is intentionally separate from the runtime
PF until its shadow gates have been validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m


@dataclass(frozen=True)
class SharedIntegerConfig:
    ambiguity_model: str = "exact_pair"
    carrier_sigma_cycles: float = 0.08
    dd_pr_sigma_m: float = 4.0
    prior_sigma_m: float = 20.0
    huber_k: float = 2.5
    min_arc_samples: int = 3
    max_epoch_gap: int = 10
    slip_threshold_cycles: float = 0.75
    min_carrier_rows: int = 30
    max_iterations: int = 30
    max_update_m: float = 8.0


@dataclass(frozen=True)
class SharedIntegerSolve:
    position_ecef: np.ndarray
    applied: bool
    reason: str
    iterations: int
    carrier_rows: int
    carrier_arcs: int
    dd_pr_rows: int
    initial_cost: float
    final_cost: float
    carrier_rms_cycles: float
    update_norm_m: float


@dataclass(frozen=True)
class _CarrierRow:
    epoch: int
    key: tuple[str, str, int]
    observed_cycles: float
    sat_k: np.ndarray
    sat_ref: np.ndarray
    base_k: float
    base_ref: float
    wavelength_m: float
    weight: float


def _carrier_rows(epochs: Sequence[DDCarrierEpoch | None]) -> list[_CarrierRow]:
    rows: list[_CarrierRow] = []
    for epoch, obs in enumerate(epochs):
        if obs is None or obs.sat_ids is None or obs.ref_sat_ids is None:
            continue
        n = min(
            obs.n,
            len(obs.sat_ids),
            len(obs.ref_sat_ids),
            len(np.asarray(obs.wavelengths_m).ravel()),
        )
        weights = (
            np.ones(obs.n, dtype=np.float64)
            if obs.weights is None
            else np.asarray(obs.weights, dtype=np.float64).ravel()
        )
        for i in range(min(n, len(weights))):
            wavelength = float(obs.wavelengths_m[i])
            values = (
                float(obs.dd_carrier_cycles[i]),
                float(obs.base_range_k[i]),
                float(obs.base_range_ref[i]),
                wavelength,
                float(weights[i]),
            )
            if not all(np.isfinite(value) for value in values) or wavelength <= 0.0:
                continue
            sat_k = np.asarray(obs.sat_ecef_k[i], dtype=np.float64)
            sat_ref = np.asarray(obs.sat_ecef_ref[i], dtype=np.float64)
            if not np.isfinite(sat_k).all() or not np.isfinite(sat_ref).all():
                continue
            rows.append(
                _CarrierRow(
                    epoch=epoch,
                    key=(
                        str(obs.ref_sat_ids[i]),
                        str(obs.sat_ids[i]),
                        int(round(wavelength * 1e9)),
                    ),
                    observed_cycles=values[0],
                    sat_k=sat_k,
                    sat_ref=sat_ref,
                    base_k=values[1],
                    base_ref=values[2],
                    wavelength_m=wavelength,
                    weight=max(values[4], 1e-6),
                )
            )
    return rows


def _raw_ambiguity(position: np.ndarray, row: _CarrierRow) -> tuple[float, np.ndarray]:
    expected, jac_m = _dd_expected_and_jacobian_m(
        position, row.sat_k, row.sat_ref, row.base_k, row.base_ref
    )
    return (
        float(row.observed_cycles - expected / row.wavelength_m),
        -np.asarray(jac_m, dtype=np.float64) / row.wavelength_m,
    )


def _build_arcs(
    rows: Sequence[_CarrierRow], position: np.ndarray, cfg: SharedIntegerConfig
) -> list[list[_CarrierRow]]:
    grouped: dict[tuple[str, str, int], list[_CarrierRow]] = {}
    for row in rows:
        grouped.setdefault(row.key, []).append(row)
    arcs: list[list[_CarrierRow]] = []
    for pair_rows in grouped.values():
        pair_rows.sort(key=lambda row: row.epoch)
        current: list[_CarrierRow] = []
        previous_epoch: int | None = None
        previous_raw: float | None = None
        for row in pair_rows:
            raw, _jac = _raw_ambiguity(position, row)
            split = (
                previous_epoch is not None
                and (
                    row.epoch - previous_epoch > int(cfg.max_epoch_gap)
                    or abs(raw - float(previous_raw)) > float(cfg.slip_threshold_cycles)
                )
            )
            if split:
                if len(current) >= int(cfg.min_arc_samples):
                    arcs.append(current)
                current = []
            current.append(row)
            previous_epoch = row.epoch
            previous_raw = raw
        if len(current) >= int(cfg.min_arc_samples):
            arcs.append(current)
    return arcs


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = 0.5 * float(np.sum(sorted_weights))
    return float(sorted_values[min(int(np.searchsorted(np.cumsum(sorted_weights), threshold)), len(values) - 1)])


def _arc_integers(
    position: np.ndarray, arcs: Sequence[Sequence[_CarrierRow]]
) -> list[int]:
    integers: list[int] = []
    for arc in arcs:
        raw = np.asarray([_raw_ambiguity(position, row)[0] for row in arc])
        weights = np.asarray([row.weight for row in arc])
        integers.append(int(np.round(_weighted_median(raw, weights))))
    return integers


def _satellite_potential_constraints(
    position: np.ndarray, rows: Sequence[_CarrierRow]
) -> tuple[list[list[_CarrierRow]], list[int], int]:
    """Express DD integers as differences of pivot-invariant satellite integers."""
    by_wavelength: dict[tuple[int, str], list[_CarrierRow]] = {}
    for row in rows:
        constellation = row.key[0].split("@", 1)[0][:1]
        by_wavelength.setdefault((row.key[2], constellation), []).append(row)
    constraints: list[list[_CarrierRow]] = []
    integers: list[int] = []
    n_potentials = 0
    for family_rows in by_wavelength.values():
        satellite_ids = sorted(
            {row.key[0] for row in family_rows} | {row.key[1] for row in family_rows}
        )
        if len(satellite_ids) < 2:
            continue
        index = {satellite_id: i for i, satellite_id in enumerate(satellite_ids)}
        design = np.zeros((len(family_rows), len(satellite_ids) - 1), dtype=np.float64)
        observed = np.empty(len(family_rows), dtype=np.float64)
        weights = np.empty(len(family_rows), dtype=np.float64)
        for i, row in enumerate(family_rows):
            ref_index = index[row.key[0]]
            sat_index = index[row.key[1]]
            if ref_index > 0:
                design[i, ref_index - 1] = -1.0
            if sat_index > 0:
                design[i, sat_index - 1] = 1.0
            observed[i] = _raw_ambiguity(position, row)[0]
            weights[i] = np.sqrt(max(row.weight, 1e-6))
        weighted_design = design * weights.reshape(-1, 1)
        weighted_observed = observed * weights
        potential_float = np.zeros(len(satellite_ids), dtype=np.float64)
        potential_float[1:] = np.linalg.lstsq(
            weighted_design, weighted_observed, rcond=None
        )[0]
        potential_integer = np.rint(potential_float).astype(np.int64)
        # Rounding a correlated float graph node-by-node is not, in general,
        # the best integer graph.  Integer Gauss-Seidel cheaply repairs those
        # rounding conflicts while preserving the fixed gauge at node zero.
        for _iteration in range(20):
            changed = False
            for node in range(1, len(satellite_ids)):
                implied: list[float] = []
                implied_weights: list[float] = []
                for row, raw, weight in zip(family_rows, observed, weights):
                    ref_index = index[row.key[0]]
                    sat_index = index[row.key[1]]
                    if sat_index == node:
                        implied.append(float(raw + potential_integer[ref_index]))
                    elif ref_index == node:
                        implied.append(float(potential_integer[sat_index] - raw))
                    else:
                        continue
                    implied_weights.append(float(weight * weight))
                if not implied:
                    continue
                target = int(
                    np.rint(
                        np.average(
                            np.asarray(implied), weights=np.asarray(implied_weights)
                        )
                    )
                )
                if target != int(potential_integer[node]):
                    potential_integer[node] = target
                    changed = True
            if not changed:
                break
        for row in family_rows:
            constraints.append([row])
            integers.append(
                int(
                    potential_integer[index[row.key[1]]]
                    - potential_integer[index[row.key[0]]]
                )
            )
        n_potentials += len(satellite_ids) - 1
    return constraints, integers, n_potentials


def _huber(value: float, k: float) -> tuple[float, float]:
    absolute = abs(float(value))
    if absolute <= k:
        return 1.0, 0.5 * absolute * absolute
    return k / max(absolute, 1e-12), k * (absolute - 0.5 * k)


def _linearize(
    position: np.ndarray,
    prior: np.ndarray,
    arcs: Sequence[Sequence[_CarrierRow]],
    integers: Sequence[int],
    dd_pseudorange: Sequence[DDPseudorangeEpoch | None],
    cfg: SharedIntegerConfig,
) -> tuple[np.ndarray, np.ndarray, float, int, int, np.ndarray]:
    hessian = np.zeros((3, 3), dtype=np.float64)
    gradient = np.zeros(3, dtype=np.float64)
    cost = 0.0
    carrier_residuals: list[float] = []

    def add(residual: float, jacobian: np.ndarray, sigma: float, scale: float = 1.0) -> None:
        nonlocal cost
        normalized = residual / sigma
        weight, row_cost = _huber(normalized, float(cfg.huber_k))
        jac = np.asarray(jacobian, dtype=np.float64) / sigma
        combined = weight * max(float(scale), 1e-6)
        hessian[:] += combined * np.outer(jac, jac)
        gradient[:] += combined * jac * normalized
        cost += max(float(scale), 1e-6) * row_cost

    if np.isfinite(cfg.prior_sigma_m) and cfg.prior_sigma_m > 0.0:
        for axis in range(3):
            jac = np.zeros(3, dtype=np.float64)
            jac[axis] = 1.0
            add(float(position[axis] - prior[axis]), jac, float(cfg.prior_sigma_m))

    for arc, integer in zip(arcs, integers):
        for row in arc:
            raw, jac = _raw_ambiguity(position, row)
            residual = raw - int(integer)
            carrier_residuals.append(residual)
            add(residual, jac, float(cfg.carrier_sigma_cycles), row.weight)

    dd_pr_rows = 0
    for obs in dd_pseudorange:
        if obs is None:
            continue
        weights = np.ones(obs.n) if obs.weights is None else np.asarray(obs.weights).ravel()
        n = min(obs.n, len(weights))
        for i in range(n):
            expected, jac = _dd_expected_and_jacobian_m(
                position,
                obs.sat_ecef_k[i],
                obs.sat_ecef_ref[i],
                obs.base_range_k[i],
                obs.base_range_ref[i],
            )
            observed = float(obs.dd_pseudorange_m[i])
            if not np.isfinite(observed) or not np.isfinite(expected):
                continue
            add(
                float(expected - observed),
                jac,
                float(cfg.dd_pr_sigma_m),
                max(float(weights[i]), 1e-6),
            )
            dd_pr_rows += 1
    return (
        hessian,
        gradient,
        float(cost),
        sum(len(arc) for arc in arcs),
        dd_pr_rows,
        np.asarray(carrier_residuals, dtype=np.float64),
    )


def solve_static_shared_integers(
    initial_position_ecef: np.ndarray,
    dd_carrier: Sequence[DDCarrierEpoch | None],
    dd_pseudorange: Sequence[DDPseudorangeEpoch | None],
    config: SharedIntegerConfig | None = None,
) -> SharedIntegerSolve:
    cfg = SharedIntegerConfig() if config is None else config
    initial = np.asarray(initial_position_ecef, dtype=np.float64).reshape(3)
    rows = _carrier_rows(dd_carrier)
    if cfg.ambiguity_model not in {"exact_pair", "satellite_potential"}:
        raise ValueError("ambiguity_model must be exact_pair or satellite_potential")
    arcs = _build_arcs(rows, initial, cfg) if cfg.ambiguity_model == "exact_pair" else []
    if cfg.ambiguity_model == "satellite_potential":
        arcs, _initial_integers, n_constraints = _satellite_potential_constraints(
            initial, rows
        )
    else:
        n_constraints = len(arcs)
    carrier_rows = sum(len(arc) for arc in arcs)
    if carrier_rows < int(cfg.min_carrier_rows):
        return SharedIntegerSolve(initial, False, "not_enough_carrier_rows", 0, carrier_rows, len(arcs), 0, float("inf"), float("inf"), float("inf"), 0.0)

    x = initial.copy()
    integers = (
        _arc_integers(x, arcs)
        if cfg.ambiguity_model == "exact_pair"
        else _satellite_potential_constraints(x, rows)[1]
    )
    _h, _g, initial_cost, _nc, dd_pr_rows, residuals = _linearize(
        x, initial, arcs, integers, dd_pseudorange, cfg
    )
    cost = initial_cost
    reason = "max_iterations"
    iterations = 0
    for iteration in range(max(1, int(cfg.max_iterations))):
        integers = (
            _arc_integers(x, arcs)
            if cfg.ambiguity_model == "exact_pair"
            else _satellite_potential_constraints(x, rows)[1]
        )
        hessian, gradient, linear_cost, _nc, dd_pr_rows, residuals = _linearize(
            x, initial, arcs, integers, dd_pseudorange, cfg
        )
        try:
            delta = -np.linalg.solve(hessian + np.eye(3) * 1e-6, gradient)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(hessian + np.eye(3) * 1e-6, gradient, rcond=None)[0]
        iterations = iteration + 1
        if not np.isfinite(delta).all():
            reason = "nonfinite_step"
            break
        if np.linalg.norm(delta) < 1e-4:
            cost = linear_cost
            reason = "converged"
            break
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
            candidate = x + float(scale) * delta
            _ch, _cg, candidate_cost, _cn, _dp, _cr = _linearize(
                candidate, initial, arcs, integers, dd_pseudorange, cfg
            )
            if candidate_cost <= cost:
                x = candidate
                cost = candidate_cost
                accepted = True
                break
        if not accepted:
            reason = "no_descent"
            break

    integers = (
        _arc_integers(x, arcs)
        if cfg.ambiguity_model == "exact_pair"
        else _satellite_potential_constraints(x, rows)[1]
    )
    _h, _g, final_cost, _nc, dd_pr_rows, residuals = _linearize(
        x, initial, arcs, integers, dd_pseudorange, cfg
    )
    update = float(np.linalg.norm(x - initial))
    applied = bool(
        np.isfinite(final_cost)
        and final_cost <= initial_cost
        and update <= float(cfg.max_update_m)
    )
    if update > float(cfg.max_update_m):
        reason = "max_update"
    return SharedIntegerSolve(
        position_ecef=x,
        applied=applied,
        reason=reason,
        iterations=iterations,
        carrier_rows=carrier_rows,
        carrier_arcs=n_constraints,
        dd_pr_rows=dd_pr_rows,
        initial_cost=initial_cost,
        final_cost=final_cost,
        carrier_rms_cycles=float(np.sqrt(np.mean(np.square(residuals)))) if len(residuals) else float("inf"),
        update_norm_m=update,
    )
