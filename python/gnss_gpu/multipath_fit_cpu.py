"""Local two-path C/A aperture fitting with explicit ambiguity diagnostics.

No geometry or true path delays are accepted. This is a parameter estimator
after acquisition, not a detector or a guaranteed LOS/multipath separator.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gnss_gpu.code_delay_cpu import CodeDelayEstimate, _averaged_template, estimate_ca_delay
from gnss_gpu.input_validation import positive_float, positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.validation.tracking_residuals import generate_ca_code


@dataclass(frozen=True)
class TwoPathDelayFit:
    single: CodeDelayEstimate
    first_delay_samples: float
    second_delay_samples: float
    first_amplitude: complex
    second_amplitude: complex
    fit_residual_fraction: float
    bic_improvement: float
    delay_condition_number: Optional[float]
    first_delay_std_samples: Optional[float]
    selected_model: str
    reason: str

    @property
    def delay_samples(self):
        """Selected delay; ambiguous preferred two-path fits return None."""
        if self.selected_model == "single_path":
            return self.single.delay_samples
        if self.selected_model == "two_path":
            return self.first_delay_samples
        return None


def fit_ca_two_paths(iq, sampling_freq, *, prn=1, carrier_hz=0.,
                     search_radius_chips=.5, max_excess_chips=1.,
                     max_delay_std_samples=.25, starts=8, max_iterations=20):
    """Fit two constant-complex-amplitude paths at a supplied common carrier.

    One 1 ms centered-aperture period and integer >=2 samples/chip are required,
    as in estimate_ca_delay. Carrier must come from acquisition, not truth.
    Search a neighborhood of the strongest single-path code peak; restrict the
    earliest path to be stronger than the later path. This LOS-dominant prior
    can fail in NLOS. Delays are unwrapped locally for ordering and returned
    modulo one code period, so first can numerically exceed second at a wrap.

    A coarse integer-sample pair search seeds variable-projection Gauss-Newton
    fits. Unknown complex amplitudes are solved linearly at each delay pair.
    BIC compares 3 versus 6 real parameters with 2*N real white-noise samples.
    It is a model-selection diagnostic, not a calibrated multipath alarm. Local
    projected-Jacobian rank, knot/boundary checks and an approximate noise-based
    delay standard deviation reject poorly conditioned preferred two-path fits.
    They do not establish global identifiability or cover model mismatch.

    Short-delay templates in the same affine aperture interval can be exactly
    nonidentifiable. A small residual alone must not be interpreted as separation.
    Each path's code and carrier drift within 1 ms are neglected; common carrier
    and constant amplitudes are approximations in moving scenes. No tracking
    state or original acquisition detection threshold is modified.
    """
    radius = positive_float("search_radius_chips", search_radius_chips)
    excess = positive_float("max_excess_chips", max_excess_chips)
    max_std = positive_float("max_delay_std_samples", max_delay_std_samples)
    starts = positive_int("starts", starts)
    iterations = positive_int("max_iterations", max_iterations)
    single = estimate_ca_delay(iq, sampling_freq, prn=prn, carrier_hz=carrier_hz)
    if single is None:
        return None
    fs = single.sampling_freq
    spc = round(fs/CHIP_RATE)
    count = 1023*spc
    if radius*spc >= count/4 or excess*spc >= count/4:
        raise ValueError("search radius/excess must be less than one quarter code period")
    data = np.asarray(iq, dtype=np.float64).reshape(-1, 2)
    signal = data[:, 0]+1j*data[:, 1]
    scale = float(np.max(np.abs(signal)))
    signal = signal/scale*np.exp(-2j*np.pi*float(carrier_hz)*np.arange(count)/fs)
    energy = float(np.vdot(signal, signal).real)
    code = generate_ca_code(prn)
    low, high = single.delay_samples-radius*spc, single.delay_samples+radius*spc
    max_gap, min_gap = excess*spc, .05
    if max_gap <= min_gap:
        raise ValueError("max_excess_chips is too small for the minimum .05 sample gap")

    def template(delay):
        return _averaged_template(code, spc, delay)

    def solve(delays):
        design = np.column_stack([template(delays[0]), template(delays[1])])
        gram = design.T@design
        if np.linalg.det(gram) <= 1e-12*np.prod(np.diag(gram)):
            return None
        amplitudes = np.linalg.solve(gram, design.T@signal)
        residual = signal-design@amplitudes
        error = float(np.vdot(residual, residual).real)
        return error, design, gram, amplitudes, residual

    def information(delays, fitted):
        _, design, gram, amplitudes, _ = fitted
        derivatives = []
        for delay, amplitude in zip(delays, amplitudes):
            left = np.floor(delay+.5)-.5
            derivatives.append(amplitude*(template(left+1)-template(left)))
        derivative = np.column_stack(derivatives)
        projected = derivative-design@np.linalg.solve(gram, design.T@derivative)
        return np.real(projected.conj().T@projected), projected

    # Gram/projection tables make the coarse pair search independent of IQ
    # length once templates have been computed. Integer centers avoid knots.
    grid = np.arange(np.floor(low), np.ceil(high+max_gap)+1)
    templates = np.array([template(delay) for delay in grid])
    gram = templates@templates.T
    projections = templates@signal
    pairs = []
    for i, first in enumerate(grid):
        if not low <= first <= high:
            continue
        for j in range(i+1, len(grid)):
            if grid[j]-first > max_gap:
                break
            aa, bb, ab = gram[i, i], gram[j, j], gram[i, j]
            determinant = aa*bb-ab*ab
            if determinant <= 1e-12*aa*bb:
                continue
            captured = float((bb*abs(projections[i])**2+aa*abs(projections[j])**2
                              -2*ab*np.real(np.conj(projections[i])*projections[j]))/determinant)
            pairs.append((energy-captured, first, grid[j]))
    # Tiny neighborhoods still need a valid seed when no integer pair fits.
    pairs.extend([(energy, float(np.clip(single.delay_samples, low, high)),
                   float(np.clip(single.delay_samples, low, high)+max_gap/2))])
    best = None
    for _, first, second in sorted(pairs)[:starts]:
        delays = np.array([first, second], dtype=np.float64)
        fitted = solve(delays)
        if fitted is None:
            continue
        for _ in range(iterations):
            hessian, projected = information(delays, fitted)
            eigenvalues = np.linalg.eigvalsh(hessian)
            if eigenvalues[0] <= max(1e-12, 1e-10*eigenvalues[-1]):
                break
            step = np.linalg.solve(hessian, np.real(projected.conj().T@fitted[4]))
            if np.max(np.abs(step)) < 1e-6:
                break
            improved = False
            for fraction in (1., .5, .25, .125, .0625, .03125):
                trial = delays+fraction*np.clip(step, -1., 1.)
                trial[0] = np.clip(trial[0], low, high)
                trial[1] = np.clip(trial[1], trial[0]+min_gap, trial[0]+max_gap)
                candidate = solve(trial)
                if candidate is not None and candidate[0] < fitted[0]:
                    delays, fitted, improved = trial, candidate, True
                    break
            if not improved:
                break
        if abs(fitted[3][0]) < abs(fitted[3][1]):
            continue
        if best is None or fitted[0] < best[1][0]:
            best = delays, fitted
    if best is None:
        # An admissible minimum-gap pair is retained as a diagnostic, even when
        # the LOS-dominant assumption has no candidate. It is never selected.
        delays = np.array([single.delay_samples, single.delay_samples+max_gap/2])
        fitted = solve(delays)
        if fitted is None:
            return None
        best = delays, fitted
        admissible = False
    else:
        admissible = True
    delays, fitted = best
    error, _, _, amplitudes, _ = fitted
    hessian, _ = information(delays, fitted)
    eigenvalues = np.linalg.eigvalsh(hessian)
    rank_ok = eigenvalues[0] > max(1e-12, 1e-8*eigenvalues[-1])
    condition = float(eigenvalues[-1]/eigenvalues[0]) if rank_ok else None
    std = float(np.sqrt(error/(2*count-6)*np.linalg.inv(hessian)[0, 0])) if rank_ok else None
    # A shared floating-point floor prevents BIC interpreting numerical residual
    # differences of perfect fits as physical evidence for an extra path.
    floor = 64*np.finfo(float).eps*energy
    one_error = max(single.fit_residual_fraction*energy, floor)
    two_error = max(error, floor)
    bic = float(2*count*np.log(one_error/two_error)-3*np.log(2*count))
    near_knot = bool(np.any(np.abs((delays+.5)-np.round(delays+.5)) < 1e-4))
    at_bound = (min(abs(delays[0]-low), abs(delays[0]-high),
                    abs(delays[1]-delays[0]-min_gap), abs(delays[1]-delays[0]-max_gap)) < 1e-4)
    model, reason = "single_path", "BIC prefers the single-path model"
    if bic > 0:
        model, reason = "ambiguous", "two-path candidate is not reliable"
        if not admissible:
            reason = "no LOS-dominant two-path candidate"
        elif not rank_ok:
            reason = "rank-deficient delay information"
        elif near_knot or at_bound:
            reason = "candidate at template knot or search boundary"
        elif std is None or std > max_std:
            reason = "local delay standard deviation exceeds configured limit"
        else:
            model, reason = "two_path", "BIC and local conditioning checks passed; not an integrity guarantee"
    def wrap(value):
        return float((value+count/2) % count-count/2)
    return TwoPathDelayFit(single, wrap(delays[0]), wrap(delays[1]),
                           complex(amplitudes[0]*scale), complex(amplitudes[1]*scale),
                           float(np.clip(error/energy, 0., 1.)), bic, condition, std, model, reason)
