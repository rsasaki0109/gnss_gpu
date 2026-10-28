"""One residual-based noise estimate and weighted dynamic refit; recalibrate separately."""

from __future__ import annotations

from typing import Literal, overload
from typing import cast
from scipy.optimize import OptimizeResult
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.noise_aware_multipath_cpu import _select_equivalent_converged
from gnss_gpu.dynamic_multipath_cpu import DynamicFit, temporal_echo, fit_dynamic_paths
from gnss_gpu.calibrated_dynamic_paths_cpu import DynamicScore


def _inverse_std(weights, count):
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (count,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("finite positive spectral variance weights required")
    return 1 / np.sqrt(weights / weights.mean())


def _problem(blocks, noise_weights):
    f, s = ca_passband_spectrum(3.6 * CHIP_RATE)
    y = np.asarray(blocks, dtype=complex)
    if y.shape != (10, len(s)) or not np.all(np.isfinite(y)) or np.max(np.abs(y)) == 0:
        raise ValueError("finite nonzero ten-block full-passband observation required")
    inv = _inverse_std(noise_weights, len(s))
    y = y * inv
    s = s * inv
    y = y / np.max(np.abs(y))
    slope = -2j * np.pi * f / CHIP_RATE
    ones = np.ones(10)
    energy = float(np.vdot(y, y).real)

    @overload
    def solve(p, details: Literal[False] = False) -> tuple[float, np.ndarray]: ...

    @overload
    def solve(p, details: Literal[True]) -> tuple[float, np.ndarray, np.ndarray]: ...

    def solve(p, details=False):
        first, gap, fd = p
        c, dc = temporal_echo(fd)
        u = [ones, c]
        v = [s * np.exp(slope * first), s * np.exp(slope * (first + gap))]

        def inner(a, b):
            return np.vdot(a[0], b[0]) * np.vdot(a[1], b[1])

        def corr(a):
            return np.vdot(a[0], y @ a[1].conj())

        templates = list(zip(u, v))
        gram = np.array([[inner(a, b) for b in templates] for a in templates])
        rhs = np.array([corr(a) for a in templates])
        amp = np.linalg.solve(gram, rhs)
        loss = float(energy - np.vdot(rhs, amp).real)
        if loss < 1e-8 * energy:
            # Avoid subtracting nearly equal energies in noiseless validation.
            residual = y - sum(
                uu[:, None] * vv * amp[i] for i, (uu, vv) in enumerate(templates)
            )
            loss = float(np.vdot(residual, residual).real)
        deriv = [
            [(ones, slope * v[0]), (c, slope * v[1])],
            [None, (c, slope * v[1])],
            [None, (dc, v[1])],
        ]
        grad = []
        for terms in deriv:
            value = 0j
            for i, a in enumerate(terms):
                if a is not None:
                    value += amp[i].conjugate() * (
                        corr(a)
                        - sum(inner(a, b) * amp[j] for j, b in enumerate(templates))
                    )
            grad.append(-2 * value.real)
        if not details:
            return loss, np.array(grad)
        a = np.column_stack([(uu[:, None] * vv).ravel() for uu, vv in templates])
        jac = np.column_stack(
            [
                sum(
                    (term[0][:, None] * term[1] * amp[i]).ravel()
                    for i, term in enumerate(terms)
                    if term is not None
                )
                for terms in deriv
            ]
        )
        residual = y.ravel() - a @ amp
        return float(np.vdot(residual, residual).real), a, jac

    return solve


def fit_weighted_dynamic_paths(blocks, noise_weights):
    """Unknown echo Doppler [-12,12] Hz; first delay [-1,1], gap [.01,1.2].

    Direct carrier/nav already wiped; complex amplitudes unknown. Given spectral variance weights are used for both data and templates.
    Always fits two paths: locally_usable is not a calibrated detection claim.
    """
    solve = _problem(blocks, noise_weights)
    seeds = [
        (d, g, fd)
        for d in np.linspace(-1, 1, 9)
        for g in (0.05, 0.2, 0.5, 1.0)
        for fd in (-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0)
    ]
    seeds = sorted(seeds, key=lambda p: solve(p)[0])[:12]
    fits = [
        minimize(
            solve,
            seed,
            jac=True,
            method="L-BFGS-B",
            bounds=((-1, 1), (0.01, 1.2), (-12, 12)),
            options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9},
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
            np.sqrt(max(0, loss / (2 * len(a) - 7) * np.linalg.inv(info)[0, 0]))
        )
    first, gap, fd = map(float, best.x)
    if not best.success:
        reason = "optimizer did not converge"
    elif min(first + 1, 1 - first, gap - 0.01, 1.2 - gap, fd + 12, 12 - fd) < 1e-4:
        reason = "search boundary"
    elif std is None or std > 0.1:
        reason = "weak local delay information"
    else:
        reason = "interior locally conditioned fit"
    return DynamicFit(
        first,
        gap,
        fd,
        std,
        bool(best.success),
        reason == "interior locally conditioned fit",
        reason,
        loss,
    )


def fit_weighted_single_path(blocks, noise_weights):
    """Constant complex amplitude/direct carrier, same full-block normalization."""
    _, s = ca_passband_spectrum(3.6 * CHIP_RATE)
    y = np.asarray(blocks, dtype=complex)
    if y.shape != (10, len(s)) or not np.all(np.isfinite(y)) or np.max(np.abs(y)) == 0:
        raise ValueError("finite nonzero ten-block full-passband observation required")
    inv = _inverse_std(noise_weights, len(s))
    y = y * inv
    s = s * inv
    y = y / np.max(np.abs(y))
    f, _ = ca_passband_spectrum(3.6 * CHIP_RATE)
    total = y.sum(axis=0)
    energy = float(np.vdot(y, y).real)
    norm = 10 * np.vdot(s, s).real

    def loss(delay, explicit=False):
        template = s * np.exp(-2j * np.pi * f * delay / CHIP_RATE)
        amplitude = np.vdot(template, total) / norm
        value = energy - abs(np.vdot(template, total)) ** 2 / norm
        if explicit or value < 1e-8 * energy:
            residual = y - amplitude * template
            return float(np.vdot(residual, residual).real)
        return float(value)

    grid = np.linspace(-1, 2.2, 49)
    index = int(np.argmin([loss(d) for d in grid]))
    fit = cast(OptimizeResult, minimize_scalar(
        loss,
        bounds=(grid[max(0, index - 1)], grid[min(48, index + 1)]),
        method="bounded",
        options={"xatol": 1e-10},
    ))
    return float(fit.x), loss(fit.x, True), bool(fit.success)


def estimate_residual_weights(blocks, preliminary):
    """Subtract fitted time-varying paths; smooth residual energy over 101 bins.

    Preliminary fit may be locally invalid: it is used only as a deterministic
    nuisance prediction, not accepted as a ranging result. Data reuse and model
    error are part of the pipeline that must be calibrated independently.
    """
    f, s = ca_passband_spectrum(3.6 * CHIP_RATE)
    y = np.asarray(blocks, dtype=complex)
    if y.shape != (10, len(s)) or not np.all(np.isfinite(y)):
        raise ValueError("finite ten-block full-passband observation required")
    first = preliminary.first_delay_chips
    gap = preliminary.excess_chips
    direct = np.broadcast_to(s * np.exp(-2j * np.pi * f * first / CHIP_RATE), y.shape)
    echo = (
        temporal_echo(preliminary.echo_doppler_hz)[0][:, None]
        * s
        * np.exp(-2j * np.pi * f * (first + gap) / CHIP_RATE)
    )
    a = np.column_stack([direct.ravel(), echo.ravel()])
    amplitudes = np.linalg.lstsq(a, y.ravel(), rcond=None)[0]
    residual = (y.ravel() - a @ amplitudes).reshape(y.shape)
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


def score_residual_dynamic_paths(blocks):
    """No truth/profile/CN0 supplied; one preliminary fit and one weighted refit."""
    preliminary = fit_dynamic_paths(blocks)
    weights = estimate_residual_weights(blocks, preliminary)
    first, loss, converged = fit_weighted_single_path(blocks, weights)
    two = fit_weighted_dynamic_paths(blocks, weights)
    n = 2 * np.asarray(blocks).size
    tiny = np.finfo(float).tiny
    score = float(n * np.log(max(loss, tiny) / max(two.loss, tiny)) - 4 * np.log(n))
    return DynamicScore(first, loss, converged, score, two), preliminary, weights
