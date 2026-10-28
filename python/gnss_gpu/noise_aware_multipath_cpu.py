"""Estimated spectral-noise weighting of the frozen v2 delay fit.

Only template/data weighting changes; prior search and numerical guards remain.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import OptimizeResult
from typing import cast
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


def _fit_weighted_spectrum(observation, bandwidth_hz, noise_weights, *, prn=1,
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
    weights = np.asarray(noise_weights, dtype=float)
    if weights.shape != spectrum.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('finite positive spectral variance weights required')
    inverse_std = 1/np.sqrt(weights/weights.mean())
    y = y*inverse_std
    spectrum = spectrum*inverse_std
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
    fitted_single = cast(OptimizeResult, minimize_scalar(single, bounds=(grid[max(0,index-1)],
                                    grid[min(len(grid)-1,index+1)]), method='bounded',
                                    options={'xatol': 1e-10}))
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


def _select_equivalent_converged(fits: list[OptimizeResult]) -> OptimizeResult:
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


def estimate_noise_weights(blocks):
    """Estimate relative spectral variance from >=3 aligned static blocks.

    Subtract each harmonic's across-block mean, use sample variance, then
    average 101 adjacent harmonics with edge-count normalization. No simulated
    noise profile, true delay or true C/N0 is used. Constant signal cancels;
    moving/changing signals can contaminate the estimate. Gaussian block-mean
    noise is independent of sample deviations under the stationary model.
    """
    data = np.asarray(blocks, dtype=complex)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 101 or not np.all(np.isfinite(data)):
        raise ValueError('finite block matrix with >=3 rows and >=101 harmonics required')
    residual = data-data.mean(axis=0)
    variance = np.sum(np.abs(residual)**2,axis=0)/(len(data)-1)
    kernel = np.ones(101)
    smoothed = np.convolve(variance,kernel,mode='same')/np.convolve(np.ones_like(variance),kernel,mode='same')
    mean = float(smoothed.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError('positive across-block noise variance required')
    weights = np.maximum(smoothed,mean*1e-6)
    return weights/weights.mean()


def fit_noise_aware_paths(blocks):
    """Fixed PRN1/B=3.6 chip rates, ten aligned 20 ms blocks (200 ms).

    Returns the raw BIC diagnostics; calibrated selection needs a separately
    calibrated full-pipeline gate. The old white-noise gate must not be reused.
    """
    data = np.asarray(blocks,dtype=complex)
    count = len(ca_passband_spectrum(3.6*CHIP_RATE)[0])
    if data.shape != (10,count):
        raise ValueError('ten full passband 20 ms spectral blocks required')
    weights = estimate_noise_weights(data)
    return _fit_weighted_spectrum(data.mean(axis=0),3.6*CHIP_RATE,weights)
