"""Ideal bandlimited C/A two-path local information, not a receiver estimator.

Unknown complex path amplitudes are nuisance parameters. Carrier, navigation
polarity and coherent integration are ideal/known. C/N0 refers to direct-path
power BEFORE the unity-gain brick-wall filter; filtered code is not renormalized.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gnss_gpu.input_validation import finite_float, positive_float, nonnegative_float
from gnss_gpu.validation.tracking_residuals import generate_ca_code

CHIP_RATE = 1023000.
LIGHT_SPEED = 299792458.


def ca_passband_spectrum(bandwidth_hz, *, prn=1):
    """Fourier-series coefficients of code[j] on [j,j+1) chips.

    bandwidth_hz is the ONE-SIDED baseband cutoff: retain |f| <= B.
    The periodic, steady-state brick-wall model has no filter startup transient.
    """
    bandwidth = positive_float('bandwidth_hz', bandwidth_hz)
    k = np.arange(-int(np.floor(bandwidth/1000)), int(np.floor(bandwidth/1000))+1)
    code = np.asarray(generate_ca_code(prn), dtype=float)
    spectrum = np.fft.fft(code)[k % 1023]/1023
    spectrum *= np.sinc(k/1023)*np.exp(-1j*np.pi*k/1023)
    return k*1000., spectrum


def two_path_spectrum(bandwidth_hz, delay_chips, excess_chips, *, prn=1):
    """Return two template columns and derivatives w.r.t. their own delays."""
    delay = finite_float('delay_chips', delay_chips)
    excess = nonnegative_float('excess_chips', excess_chips)
    frequencies, spectrum = ca_passband_spectrum(bandwidth_hz, prn=prn)
    slope = -2j*np.pi*frequencies/CHIP_RATE
    templates = spectrum[:, None]*np.exp(slope[:, None]*np.array([delay, delay+excess]))
    return templates, slope[:, None]*templates


@dataclass(frozen=True)
class TwoPathInformation:
    status: str
    direct_std_bound_m: Optional[float]
    echo_std_bound_m: Optional[float]
    delay_information_eigenvalues: tuple
    retained_direct_power: float


def two_path_information(*, sampling_freq, bandwidth_hz, excess_chips,
                         echo_voltage=.8, phase_rad=0., cn0_dbhz=37.,
                         integration_s=.020, prn=1):
    """Local CRLB with both path delays and both complex amplitudes unknown.

    Proper complex white noise PSD N0 gives Fisher information
    2*(C/N0)*T*Re(D^H P_perp D). Ideal filtering limits signal AND noise to
    the passband. Above Nyquist, oversampling adds no independent information:
    filtered time samples have correlated noise. The spectral representation
    avoids treating these extra samples as independent observations.

    This is a local unbiased-estimator lower bound, not achievable RMSE or a
    detection/availability prediction. Singular cases return None, not a
    pseudoinverse falsely implying finite uncertainty. Relative eigenvalue
    cutoff 1e-12 is a numerical convention, not a resolution criterion.
    """
    fs = positive_float('sampling_freq', sampling_freq)
    bandwidth = positive_float('bandwidth_hz', bandwidth_hz)
    if fs <= 2*bandwidth:
        raise ValueError('sampling_freq must exceed twice the one-sided bandwidth')
    duration = positive_float('integration_s', integration_s)
    periods = duration/.001
    if periods < 1 or not np.isclose(periods, round(periods), rtol=0, atol=1e-9):
        raise ValueError('integration_s must be an integer number of code periods')
    amplitude = nonnegative_float('echo_voltage', echo_voltage)
    phase = finite_float('phase_rad', phase_rad)
    cn0 = finite_float('cn0_dbhz', cn0_dbhz)
    if not -100 <= cn0 <= 100:
        raise ValueError('cn0_dbhz must lie in [-100, 100]')
    design, derivative = two_path_spectrum(bandwidth, 0., excess_chips, prn=prn)
    power = float(np.vdot(design[:, 0], design[:, 0]).real)
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    if len(s) < 2 or s[-1] <= 1e-12*s[0]:
        return TwoPathInformation('singular_amplitudes', None, None, (), power)
    derivative = derivative*np.array([1., amplitude*np.exp(1j*phase)])
    projected = derivative-u@(u.conj().T@derivative)
    information = 2*10**(cn0/10)*duration*np.real(projected.conj().T@projected)
    eigenvalues = np.linalg.eigvalsh(information)
    eigen_tuple = tuple(float(v) for v in eigenvalues)
    if eigenvalues[0] <= max(1e-20, 1e-12*eigenvalues[-1]):
        return TwoPathInformation('singular_delays', None, None, eigen_tuple, power)
    covariance = np.linalg.inv(information)
    bounds = np.sqrt(np.diag(covariance))*LIGHT_SPEED/CHIP_RATE
    return TwoPathInformation('local_bound', float(bounds[0]), float(bounds[1]),
                              eigen_tuple, power)
