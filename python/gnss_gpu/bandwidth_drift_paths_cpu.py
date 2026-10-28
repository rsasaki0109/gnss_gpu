"""Variable-bandwidth common-drift fit; default preserves the frozen 3.6-chip-rate arithmetic.

Single path has 5 real parameters, two paths 9 including complex amplitudes.
The direct frequency is common; echo frequency is common + relative Doppler.
No model order is selected here. Local usability does not imply detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, overload
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.residual_dynamic_paths_cpu import _inverse_std
from gnss_gpu.noise_aware_multipath_cpu import _select_equivalent_converged


@dataclass(frozen=True)
class DriftFit:
    first_delay_chips: float
    common_carrier_hz: float
    common_drift_hz_s: float
    excess_chips: float | None
    relative_doppler_hz: float | None
    first_std_chips: float | None
    converged: bool
    locally_usable: bool
    reason: str
    loss: float


@lru_cache(maxsize=3)
def _quadrature(count):
    nodes, weights = np.polynomial.legendre.leggauss(24)
    t = np.arange(count)[:, None] * 0.020 + (nodes + 1) * 0.010
    return t, weights / 2


def temporal_chirp(frequency, drift, count):
    """Block means and analytic derivatives in initial frequency and drift."""
    t, weights = _quadrature(count)
    phase = np.exp(2j * np.pi * (frequency * t + 0.5 * drift * t * t))
    return (
        phase @ weights,
        (phase * (2j * np.pi * t)) @ weights,
        (phase * (1j * np.pi * t * t)) @ weights,
    )


def _problem(blocks, weights, two, bandwidth_hz=3.6 * CHIP_RATE):
    f, s = ca_passband_spectrum(bandwidth_hz)
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
        amp = np.linalg.solve(gram, rhs)
        loss = float(energy - np.vdot(rhs, amp).real)
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


def fit_drift_paths(
    blocks, weights, *, two, single_seed=None, bandwidth_hz=3.6 * CHIP_RATE
):
    """Fit common initial carrier +/-2 Hz, drift +/-3 Hz/s, relative +/-12 Hz.

    Deterministic coarse search plus 12 local starts. Optional single-path fit
    is an observation-derived initializer, never true delay/carrier information.
    """
    solve = _problem(blocks, weights, two, bandwidth_hz)
    bounds = (
        [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)]
        if two
        else [(-1.0, 2.2), (-2.0, 2.0), (-3.0, 3.0)]
    )
    if two:
        bounds += [(0.01, 1.2), (-12.0, 12.0)]
        drifts = (
            (-2.0, 0.0, 2.0)
            if single_seed is None
            else sorted(set((0.0, single_seed.common_drift_hz_s)))
        )
        seeds = [
            (d, c, a, g, r)
            for d in np.linspace(-1, 1, 9)
            for c in (-1.5, 0.0, 1.5)
            for a in drifts
            for g in (0.05, 0.2, 0.5, 1.0)
            for r in (-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0)
        ]
        if single_seed is not None:
            d = float(np.clip(single_seed.first_delay_chips, -1, 1))
            seeds += [
                (d, single_seed.common_carrier_hz, single_seed.common_drift_hz_s, g, r)
                for g in (0.05, 0.2, 0.5, 1.0)
                for r in (-5.0, -1.0, 0.0, 1.0, 5.0)
            ]
    else:
        seeds = [
            (d, c, a)
            for d in np.linspace(-1, 2.2, 49)
            for c in (-1.5, 0.0, 1.5)
            for a in (-2.0, 0.0, 2.0)
        ]
    seeds = sorted(seeds, key=lambda p: solve(p)[0])[:12]
    fits = [
        minimize(
            solve,
            seed,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9},
        )
        for seed in seeds
    ]
    best = _select_equivalent_converged(fits)
    loss, a, jac = solve(best.x, True)
    projected = jac - a @ np.linalg.lstsq(a, jac, rcond=None)[0]
    info = np.real(projected.conj().T @ projected)
    eig = np.linalg.eigvalsh(info)
    std = None
    if eig[0] > max(1e-20, 1e-12 * eig[-1]):
        std = float(
            np.sqrt(
                max(
                    0,
                    loss / (2 * len(a) - (9 if two else 5)) * np.linalg.inv(info)[0, 0],
                )
            )
        )
    if not best.success:
        reason = "optimizer did not converge"
    elif min(min(x - lo, hi - x) for x, (lo, hi) in zip(best.x, bounds)) < 1e-4:
        reason = "search boundary"
    elif std is None or std > 0.1:
        reason = "weak local delay information"
    else:
        reason = "interior locally conditioned fit"
    d, c, a = map(float, best.x[:3])
    g, r = map(float, best.x[3:]) if two else (None, None)
    return DriftFit(
        d,
        c,
        a,
        g,
        r,
        std,
        bool(best.success),
        reason == "interior locally conditioned fit",
        reason,
        loss,
    )


def estimate_weights(blocks, preliminary, bandwidth_hz=3.6 * CHIP_RATE):
    """Use both fitted carriers in residual prediction; one refit only."""
    p = preliminary
    _, a, _ = _problem(
        blocks, np.ones(np.asarray(blocks).shape[1]), True, bandwidth_hz
    )(
        (
            p.first_delay_chips,
            p.common_carrier_hz,
            p.common_drift_hz_s,
            p.excess_chips,
            p.relative_doppler_hz,
        ),
        True,
    )
    y = np.asarray(blocks).ravel()
    residual = (y - a @ np.linalg.lstsq(a, y, rcond=None)[0]).reshape(
        np.asarray(blocks).shape
    )
    variance = np.mean(np.abs(residual) ** 2, axis=0)
    kernel = np.ones(101)
    smooth = np.convolve(variance, kernel, mode="same") / np.convolve(
        np.ones_like(variance), kernel, mode="same"
    )
    mean = float(smooth.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("positive residual noise energy required")
    weights = np.maximum(smooth, mean * 1e-6)
    return weights / weights.mean()


def score_drift_paths(blocks, bandwidth_hz=3.6 * CHIP_RATE):
    """Uncalibrated ranking score, not a detector or posterior probability."""
    ones = np.ones(np.asarray(blocks).shape[-1])
    initial_single = fit_drift_paths(blocks, ones, two=False, bandwidth_hz=bandwidth_hz)
    preliminary = fit_drift_paths(
        blocks, ones, two=True, single_seed=initial_single, bandwidth_hz=bandwidth_hz
    )
    weights = estimate_weights(blocks, preliminary, bandwidth_hz)
    single = fit_drift_paths(blocks, weights, two=False, bandwidth_hz=bandwidth_hz)
    two = fit_drift_paths(
        blocks, weights, two=True, single_seed=single, bandwidth_hz=bandwidth_hz
    )
    n = 2 * np.asarray(blocks).size
    tiny = np.finfo(float).tiny
    score = float(
        n * np.log(max(single.loss, tiny) / max(two.loss, tiny)) - 4 * np.log(n)
    )
    return dict(single=single, two=two, preliminary=preliminary, score=score), weights
