"""Experimental white-noise two-path block fit; no calibrated path selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload
import numpy as np
from scipy.optimize import minimize
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.noise_aware_multipath_cpu import _select_equivalent_converged


@dataclass(frozen=True)
class DynamicFit:
    first_delay_chips: float
    excess_chips: float
    echo_doppler_hz: float
    first_std_chips: float | None
    converged: bool
    locally_usable: bool
    reason: str
    loss: float


def temporal_echo(fd):
    t = (np.arange(10) + 0.5) * 0.020
    x = fd * 0.020
    sinc = np.sinc(x)
    ds = (
        (-(np.pi**2) * x / 3 + np.pi**4 * x**3 / 30)
        if abs(x) < 1e-4
        else (np.pi * x * np.cos(np.pi * x) - np.sin(np.pi * x)) / (np.pi * x * x)
    )
    phase = np.exp(2j * np.pi * fd * t)
    return sinc * phase, phase * (0.020 * ds + 2j * np.pi * t * sinc)


def _problem(blocks):
    f, s = ca_passband_spectrum(3.6 * CHIP_RATE)
    y = np.asarray(blocks, dtype=complex)
    if y.shape != (10, len(s)) or not np.all(np.isfinite(y)) or np.max(np.abs(y)) == 0:
        raise ValueError("finite nonzero ten-block full-passband observation required")
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


def fit_dynamic_paths(blocks):
    """Unknown echo Doppler [-12,12] Hz; first delay [-1,1], gap [.01,1.2].

    Direct carrier/nav already wiped; complex amplitudes unknown. White noise
    assumed; temporal signal deviations are NOT treated as a noise estimate.
    Always fits two paths: locally_usable is not a calibrated detection claim.
    """
    solve = _problem(blocks)
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
