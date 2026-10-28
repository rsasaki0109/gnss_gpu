"""Dominant-first-path constrained fit; no calibrated gate or uncertainty claim.

Spatial objective adapted from frozen cached_drift_carrier_paths_cpu._problem.
The amplitude constraint is independent of geometric parameters, so the same
variable-projection envelope gradient applies away from active-set ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, overload
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.cached_drift_carrier_paths_cpu import temporal_chirp
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.residual_dynamic_paths_cpu import _inverse_std
from gnss_gpu.noise_aware_multipath_cpu import _select_equivalent_converged
from gnss_gpu.dominant_path_amplitudes_cpu import dominant_amplitudes

BOUNDS = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (0.01, 1.2), (-12.0, 12.0)]


def fit_parameters(fit):
    return [
        fit.first_delay_chips,
        fit.common_carrier_hz,
        fit.common_drift_hz_s,
        fit.excess_chips,
        fit.relative_doppler_hz,
    ]


def geometric_interior(parameters):
    return bool(
        min(min(x - lo, hi - x) for x, (lo, hi) in zip(parameters, BOUNDS)) >= 1e-4
    )


def dominant_problem(blocks, weights, two):
    f, s = ca_passband_spectrum(3.6 * CHIP_RATE)
    y = np.asarray(blocks, dtype=complex)
    if (
        y.ndim != 2
        or y.shape[0] not in (10, 20, 40)
        or y.shape[1] != len(s)
        or not np.all(np.isfinite(y))
        or np.max(np.abs(y)) == 0
    ):
        raise ValueError(
            "finite nonzero 10/20/40-block full-passband observation required"
        )
    inv = _inverse_std(weights, len(s))
    y = y * inv
    y = y / np.max(np.abs(y))
    s = s * inv
    slope = -2j * np.pi * f / CHIP_RATE
    energy = float(np.vdot(y, y).real)

    @lru_cache(maxsize=64)
    def spatial(delay):
        # Owned by this observation/weight problem; never shared across signals.
        # Cache only identical scalar delays, without rounding or approximation.
        v = s * np.exp(slope * delay)
        dv = slope * v
        return v, dv, y @ v.conj(), y @ dv.conj()

    @overload
    def solve(p, details: Literal[False] = False) -> tuple[float, np.ndarray]: ...

    @overload
    def solve(p, details: Literal[True]) -> tuple[float, np.ndarray, np.ndarray]: ...

    def solve(p, details=False):
        first, common, drift = p[:3]
        c, dc, ac = temporal_chirp(common, drift, y.shape[0])
        v, dv, q, dq = spatial(first)
        projections = {id(v): q, id(dv): dq}
        templates = [(c, v)]
        derivatives = [[(c, dv)], [(dc, v)], [(ac, v)]]
        if two:
            gap, relative = p[3:]
            e, de, ae = temporal_chirp(common + relative, drift, y.shape[0])
            w, dw, qw, dqw = spatial(first + gap)
            projections.update({id(w): qw, id(dw): dqw})
            templates.append((e, w))
            derivatives[0].append((e, dw))
            derivatives[1].append((de, w))
            derivatives[2].append((ae, w))
            derivatives += [[None, (e, dw)], [None, (de, w)]]

        def inner(a, b):
            return np.vdot(a[0], b[0]) * np.vdot(a[1], b[1])

        def corr(a):
            return np.vdot(a[0], projections[id(a[1])])

        gram = np.array([[inner(a, b) for b in templates] for a in templates])
        rhs = np.array([corr(a) for a in templates])
        amp = dominant_amplitudes(gram, rhs) if two else np.linalg.solve(gram, rhs)
        loss = float(
            energy - 2 * np.vdot(rhs, amp).real + np.vdot(amp, gram @ amp).real
        )
        if loss < 1e-8 * energy:
            residual = y - sum(
                u[:, None] * v * amp[i] for i, (u, v) in enumerate(templates)
            )
            loss = float(np.vdot(residual, residual).real)
        grad = []
        for terms in derivatives:
            value = sum(
                amp[i].conjugate()
                * (corr(a) - sum(inner(a, b) * amp[j] for j, b in enumerate(templates)))
                for i, a in enumerate(terms)
                if a is not None
            )
            grad.append(-2 * value.real)
        if not details:
            return loss, np.array(grad)
        a = np.column_stack([(u[:, None] * v).ravel() for u, v in templates])
        jac = np.column_stack(
            [
                sum(
                    (term[0][:, None] * term[1] * amp[i]).ravel()
                    for i, term in enumerate(terms)
                    if term is not None
                )
                for terms in derivatives
            ]
        )
        residual = y.ravel() - a @ amp
        return float(np.vdot(residual, residual).real), a, jac

    return solve


def fit_dominant_paths(blocks, weights, single_seed, two_seed):
    solve = dominant_problem(blocks, weights, True)
    drifts = sorted(set((0.0, single_seed.common_drift_hz_s)))
    seeds = [
        (d, c, a, g, r)
        for d in np.linspace(-1, 1, 9)
        for c in (-1.5, 0.0, 1.5)
        for a in drifts
        for g in (0.05, 0.2, 0.5, 1.0)
        for r in (-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0)
    ]
    d = float(np.clip(single_seed.first_delay_chips, -1, 1))
    seeds += [
        (d, single_seed.common_carrier_hz, single_seed.common_drift_hz_s, g, r)
        for g in (0.05, 0.2, 0.5, 1.0)
        for r in (-5.0, -1.0, 0.0, 1.0, 5.0)
    ]
    seeds = sorted(seeds, key=lambda p: solve(p)[0])[:12] + [fit_parameters(two_seed)]
    fits = [
        minimize(
            solve,
            p,
            jac=True,
            method="L-BFGS-B",
            bounds=BOUNDS,
            options=dict(maxiter=300, ftol=1e-13, gtol=1e-9),
        )
        for p in seeds
    ]
    best = _select_equivalent_converged(fits)
    loss, a, _ = solve(best.x, True)
    y = np.asarray(blocks) * _inverse_std(weights, np.asarray(blocks).shape[1])
    y = y / np.max(np.abs(y))
    y = y.ravel()
    amp = dominant_amplitudes(a.conj().T @ a, a.conj().T @ y)
    ratio = float(abs(amp[1]) / abs(amp[0])) if abs(amp[0]) else None
    return dict(
        parameters=best.x.tolist(),
        loss=loss,
        converged=bool(best.success),
        geometric_interior=geometric_interior(best.x),
        echo_to_first_voltage_ratio=ratio,
        amplitude_constraint_active=ratio is not None and abs(ratio - 1) < 1e-7,
    )
