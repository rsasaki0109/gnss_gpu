"""Version 2: prefer converged, numerically equivalent multistart solutions.

The original implementation is frozen for experiment reproducibility.
Only multistart winner selection changes here.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from gnss_gpu.input_validation import positive_float, finite_float
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE


@dataclass(frozen=True)
class BandlimitedFit:
    single_delay_chips: float
    first_delay_chips: float
    excess_chips: float
    selected_delay_chips: Optional[float]
    selected_model: str
    reason: str
    bic_improvement: float
    first_std_chips: Optional[float]
    amplitude_magnitudes: tuple
    converged: bool


def fit_bandlimited_paths(observation, bandwidth_hz, *, prn=1,
                          center_chips=0., radius_chips=1., max_excess_chips=1.2,
                          max_std_chips=.1):
    """Fit complex passband Fourier coefficients, ordered as ca_passband_spectrum.

    A fixed external coarse-delay window is required. Carrier/nav polarity are
    already removed. Independent equal-variance complex spectral noise is
    assumed, not independent filtered time samples. Unknown complex amplitudes
    are solved by least squares for each delay pair. Earlier path need not be
    stronger. Twelve deterministic coarse-grid seeds initialize bounded local
    optimization. BIC, local uncertainty and boundary checks are diagnostics,
    not calibrated detection/coverage. No truth or C/N0 is supplied.
    """
    frequencies, spectrum = ca_passband_spectrum(bandwidth_hz, prn=prn)
    y = np.asarray(observation, dtype=complex)
    if y.shape != spectrum.shape or not np.all(np.isfinite(y)):
        raise ValueError('observation must be finite and match passband spectrum')
    if np.linalg.norm(y) == 0:
        raise ValueError('zero observation has no delay information')
    center = finite_float('center_chips', center_chips)
    radius = positive_float('radius_chips', radius_chips)
    maximum = positive_float('max_excess_chips', max_excess_chips)
    std_limit = positive_float('max_std_chips', max_std_chips)
    minimum = .01
    if maximum <= minimum or radius > 10 or maximum > 10:
        raise ValueError('require .01 < max_excess_chips <= 10 and radius <= 10')
    low, high = center-radius, center+radius
    slope = -2j*np.pi*frequencies/CHIP_RATE
    # Scale normalization leaves fitted delays/BIC invariant and avoids overflow.
    y = y/np.max(np.abs(y))

    def template(delay):
        return spectrum*np.exp(slope*delay)

    def single(delay):
        a = template(delay)
        residual = y-a*(np.vdot(a, y)/np.vdot(a, a))
        return float(np.vdot(residual, residual).real)

    grid = np.linspace(low, high+maximum, 49)
    index = int(np.argmin([single(d) for d in grid]))
    fitted_single = minimize_scalar(single, bounds=(grid[max(0,index-1)],
                                    grid[min(len(grid)-1,index+1)]), method='bounded',
                                    options={'xatol': 1e-10})
    single_delay = float(fitted_single.x)

    def solve(parameters):
        first, gap = parameters
        a = np.column_stack([template(first), template(first+gap)])
        amplitudes = np.linalg.lstsq(a, y, rcond=None)[0]
        residual = y-a@amplitudes
        derivatives = slope[:, None]*a*amplitudes
        # Parameters are first and gap, rather than independent path delays.
        jacobian = np.column_stack([derivatives.sum(axis=1), derivatives[:, 1]])
        gradient = -2*np.real(jacobian.conj().T@residual)
        return float(np.vdot(residual, residual).real), gradient, a, amplitudes, derivatives

    seeds = [(d, g) for d in np.linspace(low, high, 9)
             for g in (.05, .2, .5, min(1., maximum)) if g <= maximum]
    seeds = sorted(seeds, key=lambda p: solve(p)[0])[:12]
    fits = [minimize(lambda p: solve(p)[:2], seed, jac=True, method='L-BFGS-B',
                    bounds=((low, high), (minimum, maximum)),
                    options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-9})
            for seed in seeds]
    best = _select_equivalent_converged(fits)
    error, _, a, amplitudes, derivative = solve(best.x)
    projected = derivative-a@np.linalg.lstsq(a, derivative, rcond=None)[0]
    information = np.real(projected.conj().T@projected)
    eig = np.linalg.eigvalsh(information)
    std = None
    if eig[0] > max(1e-20, 1e-12*eig[-1]):
        variance = error/(2*len(y)-6)
        std = float(np.sqrt(max(0., variance*np.linalg.inv(information)[0, 0])))
    n = 2*len(y)
    tiny = np.finfo(float).tiny
    bic = float(n*np.log(max(fitted_single.fun, tiny)/max(error, tiny))-3*np.log(n))
    first, gap = map(float, best.x)
    if bic <= 0:
        model, reason, selected = 'single_path', 'BIC prefers one path', single_delay
    elif not best.success:
        model, reason, selected = 'ambiguous', 'optimizer did not converge', None
    elif min(first-low, high-first, gap-minimum, maximum-gap) < 1e-4:
        model, reason, selected = 'ambiguous', 'two-path search boundary', None
    elif std is None or std > std_limit:
        model, reason, selected = 'ambiguous', 'weak local delay information', None
    else:
        model, reason, selected = 'two_path', 'interior locally conditioned fit', first
    return BandlimitedFit(single_delay, first, gap, selected, model, reason, bic,
                          std, tuple(float(abs(v)) for v in amplitudes), bool(best.success))


def _select_equivalent_converged(fits):
    """Resolve floating-point objective ties without accepting failed solutions.

    A successful alternative must agree in objective within 64 float64 eps
    times max(1, |best objective|), and in both delays within 1e-7 chip
    (about 0.03 mm). A worse basin or distinct delay solution is not a tie.
    Existing boundary, BIC and information checks still run after selection.
    """
    best = min(fits, key=lambda fit: fit.fun)
    if best.success:
        return best
    tolerance = 64*np.finfo(float).eps*max(1., abs(best.fun))
    equivalent = [fit for fit in fits if fit.success and np.isfinite(fit.fun)
                  and abs(fit.fun-best.fun) <= tolerance
                  and np.all(np.isfinite(fit.x))
                  and np.max(np.abs(fit.x-best.x)) <= 1e-7]
    return min(equivalent, key=lambda fit: fit.fun) if equivalent else best
