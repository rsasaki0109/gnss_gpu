"""Single-path-null score and selector for the frozen dynamic block estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from scipy.optimize import OptimizeResult
import numpy as np
from scipy.optimize import minimize_scalar
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.dynamic_multipath_cpu import DynamicFit, fit_dynamic_paths
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate


@dataclass(frozen=True)
class DynamicScore:
    single_delay_chips: float
    single_loss: float
    single_converged: bool
    bic_improvement: float
    two_path: DynamicFit


def fit_single_path(blocks):
    """Constant complex amplitude/direct carrier, same full-block normalization."""
    _, s = ca_passband_spectrum(3.6 * CHIP_RATE)
    y = np.asarray(blocks, dtype=complex)
    if y.shape != (10, len(s)) or not np.all(np.isfinite(y)) or np.max(np.abs(y)) == 0:
        raise ValueError("finite nonzero ten-block full-passband observation required")
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


def score_dynamic_paths(blocks):
    single, loss, converged = fit_single_path(blocks)
    two = fit_dynamic_paths(blocks)
    n = 2 * np.asarray(blocks).size
    # 7 real parameters vs 3; unknown common noise variance cancels.
    # This is only a ranking score: null boundary/nonidentifiability need calibration.
    tiny = np.finfo(float).tiny
    score = float(n * np.log(max(loss, tiny) / max(two.loss, tiny)) - 4 * np.log(n))
    return DynamicScore(single, loss, converged, score, two)


def select_dynamic_paths(score, gate):
    """Strict calibrated score gate followed by frozen local validity guards."""
    if not np.isfinite(gate.score) or not np.isfinite(score.bic_improvement):
        raise ValueError("finite score and gate required")
    if score.bic_improvement <= gate.score:
        if (
            not score.single_converged
            or min(score.single_delay_chips + 1, 2.2 - score.single_delay_chips) < 1e-4
        ):
            return dict(
                model="ambiguous", delay_chips=None, reason="single-path fit invalid"
            )
        return dict(
            model="single_path",
            delay_chips=score.single_delay_chips,
            reason="below calibrated gate",
        )
    if not score.two_path.locally_usable:
        return dict(model="ambiguous", delay_chips=None, reason=score.two_path.reason)
    return dict(
        model="two_path",
        delay_chips=score.two_path.first_delay_chips,
        reason="calibrated gate and local guards passed",
    )
