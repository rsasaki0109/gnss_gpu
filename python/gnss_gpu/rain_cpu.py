"""Rain-path attenuation and an independent, illustrative wet-surface model.

P.838-3 circular-polarization coefficients were checked against ITU-Rpy 0.4.0,
commit 094abb618c2a7a8cc145e93940f4571650bc2cf5. See docs/rain_simulation.md
and docs/itu_rpy_notice.txt. No rainfall-to-film-thickness law is assumed.
"""

from dataclasses import dataclass

import numpy as np

from gnss_gpu.fresnel import GPS_L1_FREQ, complex_permittivity
from gnss_gpu.input_validation import finite_float, nonnegative_float, positive_float

C_LIGHT = 299792458.


def circular_rain_coefficients(freq_hz=GPS_L1_FREQ):
    """Return k, alpha in gamma=k*R**alpha [dB/km], R in mm/h.

    ITU-R P.838-3 frequency fits, 1..1000 GHz. Circular polarization sets the
    cos(2*tau) term to zero, so elevation dependence of k/alpha cancels.
    Elevation still determines the actual path length through the rain layer.
    """
    f = positive_float("freq_hz", freq_hz)/1e9
    if not 1 <= f <= 1000:
        raise ValueError("P.838-3 frequency must lie in 1..1000 GHz")
    x = np.log10(f)

    def curve(a, b, c, slope, intercept):
        return float(np.sum(np.asarray(a)*np.exp(-((x-np.asarray(b))/np.asarray(c))**2))+slope*x+intercept)

    kh = 10**curve([-5.33980, -.35351, -.23789, -.94158], [-.10008, 1.2697, .86036, .64552],
                   [1.13098, .454, .15354, .16817], -.18961, .71147)
    kv = 10**curve([-3.80595, -3.44965, -.39902, .50167], [.56934, -.22911, .73042, 1.07319],
                   [.81061, .51059, .11899, .27195], -.16398, .63297)
    ah = curve([-.14318, .29591, .32177, -5.37610, 16.1721], [1.82442, .77564, .63773, -.96230, -3.29980],
               [-.55187, .19822, .13164, 1.47828, 3.4399], .67849, -1.95537)
    av = curve([-.07771, .56727, -.20238, -48.2991, 48.5833], [2.3384, .95545, 1.1452, .791669, .791459],
               [-.76284, .54039, .26809, .116226, .116479], -.053739, .83433)
    return float((kh+kv)/2), float((kh*ah+kv*av)/(kh+kv))


def rain_specific_attenuation(rain_rate_mm_h, freq_hz=GPS_L1_FREQ):
    rate = nonnegative_float("rain_rate_mm_h", rain_rate_mm_h)
    k, alpha = circular_rain_coefficients(freq_hz)
    result = k*rate**alpha
    if not np.isfinite(result):
        raise ValueError("rain attenuation exceeds supported numeric range")
    return float(result)


def segment_layer_length(start, end, layer_top_m, *, layer_bottom_m=0.):
    """Segment length inside an infinite horizontal slab, in Cartesian metres.

    Inputs broadcast over (...,3). Boundary points/segments are included;
    a zero-height layer contains no volume and has zero path length.
    """
    bottom = finite_float("layer_bottom_m", layer_bottom_m)
    top = finite_float("layer_top_m", layer_top_m)
    if top < bottom:
        raise ValueError("layer_top_m must be >= layer_bottom_m")
    a, b = np.broadcast_arrays(np.asarray(start, dtype=float), np.asarray(end, dtype=float))
    if a.ndim < 1 or a.shape[-1] != 3 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("segment endpoints must be finite (...,3) arrays")
    delta = b-a
    length = np.linalg.norm(delta, axis=-1)
    if top == bottom:
        return np.zeros_like(length)
    dz = delta[..., 2]
    t0, t1 = np.zeros_like(dz), np.zeros_like(dz)
    np.divide(bottom-a[..., 2], dz, out=t0, where=dz != 0)
    np.divide(top-a[..., 2], dz, out=t1, where=dz != 0)
    fraction = np.maximum(0., np.minimum(1., np.maximum(t0, t1))-np.maximum(0., np.minimum(t0, t1)))
    fraction = np.where(dz == 0, ((a[..., 2] >= bottom) & (a[..., 2] <= top)).astype(float), fraction)
    return fraction*length


@dataclass(frozen=True)
class RainLayer:
    rate_mm_h: float = 0.
    top_m: float = 3000.
    bottom_m: float = 0.

    def __post_init__(self):
        for name in ("rate_mm_h", "top_m", "bottom_m"):
            value = nonnegative_float(name, getattr(self, name)) if name == "rate_mm_h" else finite_float(name, getattr(self, name))
            object.__setattr__(self, name, value)
        if self.top_m < self.bottom_m:
            raise ValueError("rain layer top must be >= bottom")

    def loss_db(self, start, end, freq_hz=GPS_L1_FREQ):
        return rain_specific_attenuation(self.rate_mm_h, freq_hz)*segment_layer_length(
            start, end, self.top_m, layer_bottom_m=self.bottom_m)/1000.


def water_film_fresnel(theta_rad, substrate="concrete", *, thickness_m=0.,
                        water_permittivity=80-8j, freq_hz=GPS_L1_FREQ):
    """Air / homogeneous water film / semi-infinite substrate field coefficients.

    Return parallel, perpendicular coefficients in the existing Fresnel basis.
    exp(+j*omega*t) convention uses passive eps.imag<=0 and exp(-j*k*z).
    water_permittivity is an EXPLICIT assumed parameter, not a temperature-
    dependent water model or a measured wet-wall calibration. No roughness,
    infiltration, droplets or rainfall-rate/thickness mapping is included.
    """
    theta = np.asarray(theta_rad, dtype=float)
    if not np.all(np.isfinite(theta)) or np.any(theta < 0) or np.any(theta > np.pi/2):
        raise ValueError("theta_rad must lie in [0, pi/2]")
    thickness = nonnegative_float("thickness_m", thickness_m)
    frequency = positive_float("freq_hz", freq_hz)
    water, base = complex(water_permittivity), complex_permittivity(substrate, frequency)
    if any(not np.isfinite(eps) or eps.real <= 0 or eps.imag > 0 for eps in (water, base)):
        raise ValueError("film/substrate must have positive real and nonpositive imaginary permittivity")
    transverse = np.sin(theta)**2
    q0, q1, q2 = np.cos(theta), np.sqrt(water-transverse+0j), np.sqrt(base-transverse+0j)
    phase = np.exp(-4j*np.pi*frequency/C_LIGHT*q1*thickness)
    s01, s12 = (q0-q1)/(q0+q1), (q1-q2)/(q1+q2)
    p01 = (water*q0-q1)/(water*q0+q1)
    p12 = (base*q1-water*q2)/(base*q1+water*q2)
    parallel = (p01+p12*phase)/(1+p01*p12*phase)
    perpendicular = (s01+s12*phase)/(1+s01*s12*phase)
    return parallel, perpendicular
