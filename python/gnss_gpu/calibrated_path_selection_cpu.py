"""Conditional single-path-null calibration for the frozen default v2 fit.

Calibration is only for the generating distribution: PRN1, 3.683 MHz one-sided
bandwidth, 37 dB-Hz, 200 ms ideal coherence, fixed default delay assistance.
It does not establish a receiver acquisition false-alarm rate or cover mismatch.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
from gnss_gpu.bandlimited_multipath_v2_cpu import fit_bandlimited_paths
from gnss_gpu.multipath_information_cpu import CHIP_RATE


@dataclass(frozen=True)
class PathSelectionGate:
    score: float
    calibration_count: int
    tail_budget: float
    confidence_failure: float


def calibrate_path_gate(scores, *, tail_budget=.01, confidence_failure=.05):
    """Maximum order statistic with strict exceedance.

    For iid continuous-score null draws, P(tail > alpha) <=
    (1-alpha)**n. Ties with strict exceedance are conservative. The guarantee
    is marginal over the calibration's nuisance distribution, not uniform over
    delay, phase, power, bandwidth or integration time. No echo data is used.
    """
    alpha, delta = float(tail_budget), float(confidence_failure)
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError('tail budget and confidence failure must be in (0,1)')
    values = np.asarray(scores,dtype=float)
    required = math.ceil(math.log(delta)/math.log1p(-alpha))
    if values.ndim != 1 or len(values) < required or not np.all(np.isfinite(values)):
        raise ValueError('finite one-dimensional calibration scores and enough iid null records required')
    return PathSelectionGate(float(values.max()),len(values),alpha,delta)


def _select_default_fit(fit, gate):
    """Apply only to frozen v2 DEFAULT search settings, retaining all guards."""
    if not np.isfinite(gate.score) or not np.isfinite(fit.bic_improvement):
        raise ValueError('finite fitted score and gate required')
    if fit.bic_improvement <= gate.score:
        return replace(fit,selected_model='single_path',selected_delay_chips=fit.single_delay_chips,
                       reason='below calibrated single-path-null gate')
    if not fit.converged:
        reason='optimizer did not converge'
    elif min(fit.first_delay_chips+1,1-fit.first_delay_chips,
             fit.excess_chips-.01,1.2-fit.excess_chips) < 1e-4:
        reason='two-path search boundary'
    elif fit.first_std_chips is None or not np.isfinite(fit.first_std_chips) or fit.first_std_chips > .1:
        reason='weak local delay information'
    else:
        return replace(fit,selected_model='two_path',selected_delay_chips=fit.first_delay_chips,
                       reason='calibrated gate and unchanged local guards passed')
    return replace(fit,selected_model='ambiguous',selected_delay_chips=None,reason=reason)


def fit_calibrated_paths(observation, gate):
    """Fixed-context entry point; runtime signal power/coherence is not verified."""
    fit = fit_bandlimited_paths(observation,3.6*CHIP_RATE)
    return _select_default_fit(fit,gate)
