"""Sample-resolved CPU IQ for a moving receiver and one finite vertical wall.

Coordinates are Cartesian metres in a shared local frame. The wall is stationary;
transmitter/receiver motion can include constant acceleration. This is an ideal specular model,
with no diffraction, wall transmission or full front-end filtering. An optional
prescribed antenna response supports independently controlled sensitivity tests.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from gnss_gpu.fresnel import (
    _canonical_polarization, circular_reflection_voltage, reflection_coefficient_complex,
)
from gnss_gpu.input_validation import finite_float, nonnegative_float, positive_int
from gnss_gpu.rain_cpu import RainLayer, water_film_fresnel
from gnss_gpu.wet_delay_cpu import WetDelayResidualLayer
from gnss_gpu.wet_antenna_cpu import WetAntennaResponse
from gnss_gpu.signal_sim_cpu import CHIP_RATE, L1_HZ, CpuSignalSimulator, _add_noise, _sample_index
from gnss_gpu.validation.tracking_residuals import generate_ca_code

C_LIGHT = 299792458.0


@dataclass(frozen=True)
class VerticalWall:
    """Rectangle in the plane x=x_m, inclusive y/z bounds, metres."""

    x_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float
    material: str = "concrete"
    polarization: str = "rhcp"
    antenna_lhcp_response: complex = 0j
    water_film_thickness_m: float = 0.
    water_permittivity: complex = 80-8j

    def __post_init__(self):
        for name in ("x_m", "y_min_m", "y_max_m", "z_min_m", "z_max_m"):
            object.__setattr__(self, name, finite_float(name, getattr(self, name)))
        if self.y_min_m >= self.y_max_m or self.z_min_m >= self.z_max_m:
            raise ValueError("wall bounds must have positive width and height")
        reflection_coefficient_complex(0., self.material, polarization=self.polarization)
        try:
            response = complex(self.antenna_lhcp_response)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("antenna_lhcp_response must be a finite complex voltage ratio") from exc
        if not np.isfinite(response):
            raise ValueError("antenna_lhcp_response must be a finite complex voltage ratio")
        if response != 0j and _canonical_polarization(self.polarization) != "rhcp":
            raise ValueError("antenna_lhcp_response requires the RHCP incident/receive basis")
        object.__setattr__(self, "antenna_lhcp_response", response)
        thickness = nonnegative_float("water_film_thickness_m", self.water_film_thickness_m)
        object.__setattr__(self, "water_film_thickness_m", thickness)
        water = complex(self.water_permittivity)
        if not np.isfinite(water) or water.real <= 0 or water.imag > 0:
            raise ValueError("water_permittivity must be a passive positive-real dielectric")
        object.__setattr__(self, "water_permittivity", water)

    def contains(self, points):
        return ((points[:, 1] >= self.y_min_m) & (points[:, 1] <= self.y_max_m)
                & (points[:, 2] >= self.z_min_m) & (points[:, 2] <= self.z_max_m))


@dataclass
class WallSignalBlock:
    """IQ plus per-sample geometry; ranges are geometric path lengths in metres.

    Reflected ranges describe the image path; only reflection_visible samples
    carry that path. Invalid reflection points are NaN and coefficients zero.
    Wet residuals are separate from geometric ranges/rates. Propagation-range
    properties and Doppler include them; spreading still uses geometry.
    Antenna code bias and carrier phase intercept are separate receiver terms;
    Doppler also includes the antenna phase derivative.
    """

    iq: np.ndarray
    direct_range_m: np.ndarray
    reflected_range_m: np.ndarray
    direct_visible: np.ndarray
    reflection_visible: np.ndarray
    reflection_points_m: np.ndarray
    reflection_coefficient: np.ndarray
    direct_range_rate_mps: np.ndarray
    reflected_range_rate_mps: np.ndarray
    direct_rain_loss_db: np.ndarray
    reflected_rain_loss_db: np.ndarray
    direct_wet_delay_m: np.ndarray
    reflected_wet_delay_m: np.ndarray
    direct_wet_delay_rate_mps: np.ndarray
    reflected_wet_delay_rate_mps: np.ndarray
    direct_antenna_loss_db: np.ndarray
    reflected_antenna_loss_db: np.ndarray
    direct_antenna_code_bias_m: np.ndarray
    reflected_antenna_code_bias_m: np.ndarray
    direct_antenna_phase_rad: np.ndarray
    reflected_antenna_phase_rad: np.ndarray
    direct_antenna_code_rate_mps: np.ndarray
    reflected_antenna_code_rate_mps: np.ndarray
    direct_antenna_phase_rate_rad_s: np.ndarray
    reflected_antenna_phase_rate_rad_s: np.ndarray

    @property
    def direct_propagation_range_m(self):
        return self.direct_range_m+self.direct_wet_delay_m

    @property
    def reflected_propagation_range_m(self):
        return self.reflected_range_m+self.reflected_wet_delay_m

    @property
    def direct_doppler_hz(self):
        return (-L1_HZ*(self.direct_range_rate_mps+self.direct_wet_delay_rate_mps)/C_LIGHT
                +self.direct_antenna_phase_rate_rad_s/(2*np.pi))

    @property
    def reflected_doppler_hz(self):
        return (-L1_HZ*(self.reflected_range_rate_mps+self.reflected_wet_delay_rate_mps)/C_LIGHT
                +self.reflected_antenna_phase_rate_rad_s/(2*np.pi))


def _vector(name, value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of shape (3,)")
    return result


@lru_cache(maxsize=1)
def _moving_kernel():
    from numba import njit, prange

    @njit(cache=True)
    def integral(left, right, frequency):
        duration = right-left
        arg = math.pi*frequency*duration
        sinc = 1. if arg == 0. else math.sin(arg)/arg
        angle = math.pi*frequency*(left+right)
        return duration*sinc*complex(math.cos(angle), math.sin(angle))

    @njit(parallel=True, cache=True)
    def synthesize(code, ranges, rates, gains, code_biases, code_rates, phase_biases, phase_rates,
                   fs, intermediate, start, reference, nav, aperture):
        count = ranges.shape[1]
        out = np.empty((count, 2), dtype=np.float32)
        for i in prange(count):
            t = (start + i) / fs
            real, imag = 0., 0.
            for path in range(2):
                gain = gains[path, i]
                if gain == 0.:
                    continue
                delay = (ranges[path, i] - reference) / C_LIGHT
                # Positive path delay retards both spreading code and RF phase.
                chip_phase = (t-delay-code_biases[path, i]/C_LIGHT)*CHIP_RATE
                chip = int(math.floor(chip_phase)) % 1023
                phase = 2*math.pi*(intermediate*t - L1_HZ*delay)+phase_biases[path, i]
                cosine, sine = math.cos(phase), math.sin(phase)
                field = complex(code[chip], 0.)
                if aperture > 0:
                    # Locally affine delay during the sampling aperture; at
                    # >=2 samples/chip and subluminal motion at most one edge.
                    chip_rate = CHIP_RATE*(1-(rates[path, i]+code_rates[path, i])/C_LIGHT)
                    frequency = intermediate-L1_HZ*rates[path, i]/C_LIGHT+phase_rates[path, i]/(2*math.pi)
                    half = aperture/2
                    first = int(math.floor(chip_phase-chip_rate*half))
                    boundary = min(half, max(-half, (first+1-chip_phase)/chip_rate))
                    field = (code[first % 1023]*integral(-half, boundary, frequency)
                             + code[(first+1) % 1023]*integral(boundary, half, frequency))/aperture
                value = nav*gain*field*complex(cosine, sine)
                real += value.real
                imag += value.imag
            out[i, 0], out[i, 1] = real, imag
        return out

    return synthesize


def simulate_wall_signal(simulator: CpuSignalSimulator, wall: VerticalWall,
                         rx_position_m, rx_velocity_mps, transmitter_m,
                         n_samples, *, start_sample=0, prn=1, nav_bit=1,
                         reference_range_m=0., include_direct=True,
                         include_reflection=True, integrate_samples=False,
                         transmitter_velocity_mps=(0., 0., 0.),
                         rx_acceleration_mps2=(0., 0., 0.),
                         transmitter_acceleration_mps2=(0., 0., 0.), rain_layer=None,
                         wet_delay_layer=None, antenna_response=None):
    """Synthesize a block at absolute sample indices with sample-wise geometry.

    Positions and velocity refer to sample zero, regardless of block start.
    Both bodies use p(t)=p0+v0*t+0.5*a*t². A single instantaneous geometric
    path drives code delay, carrier phase and range rate. The direct Doppler is
    -f_L1/c * dot(v_sat-v_rx, p_sat-p_rx)/range, matching Sanwale et al. (2026),
    arXiv:2605.01553v1 Eq. (8). Motion is supplied, not orbital propagation.
    No transmit-time, Earth-rotation, clock or relativistic corrections are added.
    reference_range_m removes a fixed common delay from code and RF carrier
    (a receiver clock/time alignment); it does not change physical path lengths.
    Amplitudes follow 1/path_length, normalized to unit direct amplitude at time
    zero. Reflection includes complex Fresnel phase and image-path spreading.
    nav_bit is constant. Noise follows the supplied CpuSignalSimulator settings;
    its fixed seeds repeat per call, as in generate_epoch. No history is stored.
    include_* are diagnostic path-isolation switches, not visibility overrides.
    integrate_samples averages RF over a centered aperture of 1/fs, using
    affine path delay and constant field gain within each aperture. This models
    integrate-and-dump sampling, not a brick-wall anti-alias filter. It requires
    >=2 samples/chip. Visibility is evaluated at the aperture center.
    For RHCP illumination/receive normalization, wall.antenna_lhcp_response
    adds a complex LHCP receive voltage relative to the unit RHCP response.
    Direct RHCP amplitude remains unchanged; the reflected voltage includes
    both co- and cross-polarized fields. Zero preserves the ideal RHCP model.
    rain_layer optionally applies P.838-3 circular-polarization power attenuation
    on only the segment portions inside its horizontal slab, separately on the
    direct ray and both reflected legs. Noise is added AFTER attenuation. This
    does not model rain-induced phase/group delay.
    A wall water film independently changes its complex reflection coefficient;
    there is no implicit rainfall-to-film-depth mapping.
    wet_delay_layer independently adds a signed neutral-atmosphere residual to
    both code and carrier paths, with its analytic derivative in Doppler/code
    rate. Geometric ranges/rates and spreading remain geometric; separate wet
    delay/rate arrays expose the atmospheric component. There is no rain-to-ZWD
    law, ray bending or humidity/temperature model.
    antenna_response optionally adds prescribed time/direction-dependent loss,
    envelope code bias and independent carrier phase intercept. Code-bias and
    phase derivatives affect their respective rates; no RF phase is inferred
    from the independent code-bias parameter. Antenna loss precedes fixed
    receiver noise. This is a calibration-ready sensitivity response, not an
    antenna/radome EM or rainfall-to-wetting model.
    """
    if not isinstance(simulator, CpuSignalSimulator):
        raise TypeError("simulator must be a CpuSignalSimulator")
    if not isinstance(wall, VerticalWall):
        raise TypeError("wall must be a VerticalWall")
    if rain_layer is not None and not isinstance(rain_layer, RainLayer):
        raise TypeError("rain_layer must be a RainLayer or None")
    if wet_delay_layer is not None and not isinstance(wet_delay_layer, WetDelayResidualLayer):
        raise TypeError("wet_delay_layer must be a WetDelayResidualLayer or None")
    if antenna_response is not None and not isinstance(antenna_response, WetAntennaResponse):
        raise TypeError("antenna_response must be a WetAntennaResponse or None")
    count = positive_int("n_samples", n_samples)
    start = _sample_index("start_sample", start_sample)
    if start + count > 2**53 - 1:
        raise ValueError("sample interval must end below 2**53")
    prn = positive_int("prn", prn)
    if prn > 32:
        raise ValueError("prn must be in 1..32")
    if nav_bit not in (-1, 1):
        raise ValueError("nav_bit must be -1 or +1")
    reference = finite_float("reference_range_m", reference_range_m)
    rx0 = _vector("rx_position_m", rx_position_m)
    velocity = _vector("rx_velocity_mps", rx_velocity_mps)
    sat_velocity = _vector("transmitter_velocity_mps", transmitter_velocity_mps)
    acceleration = _vector("rx_acceleration_mps2", rx_acceleration_mps2)
    sat_acceleration = _vector("transmitter_acceleration_mps2", transmitter_acceleration_mps2)
    if integrate_samples and simulator.sampling_freq < 2*CHIP_RATE:
        raise ValueError("integrated sampling requires at least two samples/chip")
    sat = _vector("transmitter_m", transmitter_m)
    times = (start + np.arange(count, dtype=np.float64)) / simulator.sampling_freq
    with np.errstate(over="ignore", invalid="ignore"):
        rx = rx0 + times[:, None]*velocity
        if np.any(acceleration):
            rx += .5*times[:, None]**2*acceleration
        satellite = sat + times[:, None]*sat_velocity
        if np.any(sat_acceleration):
            satellite += .5*times[:, None]**2*sat_acceleration
        rx_velocities = velocity+times[:, None]*acceleration
        sat_velocities = sat_velocity+times[:, None]*sat_acceleration
        image_velocities = sat_velocities.copy()
        image_velocities[:, 0] *= -1
        relative_velocity = sat_velocities-rx_velocities
        image_relative_velocity = image_velocities-rx_velocities
        direct_vector = satellite-rx
        direct = np.linalg.norm(direct_vector, axis=1)
        amplitude_range = np.linalg.norm(sat-rx0)
        image = satellite.copy()
        image[:, 0] = 2*wall.x_m-satellite[:, 0]
        image_vector = image-rx
        reflected = np.linalg.norm(image_vector, axis=1)
    velocities = (rx_velocities, sat_velocities, relative_velocity, image_relative_velocity)
    if any(not np.all(np.isfinite(value)) or np.any(np.linalg.norm(value, axis=1) >= C_LIGHT)
           for value in velocities):
        raise ValueError("body and relative path velocities must be below the speed of light")
    if (not np.all(np.isfinite(rx)) or not np.all(np.isfinite(satellite)) or not np.all(np.isfinite(direct))
            or not np.all(np.isfinite(reflected)) or not np.isfinite(amplitude_range)
            or amplitude_range <= 0 or np.any(direct <= 0)):
        raise ValueError("geometry must have finite, positive path lengths")
    if np.any(rx[:, 0] == wall.x_m) or np.any(satellite[:, 0] == wall.x_m):
        raise ValueError("receiver/transmitter must not lie on the wall plane")
    # A two-sided opaque rectangle blocks only segment intersections within it.
    u = np.zeros(count)
    np.divide(wall.x_m-rx[:, 0], direct_vector[:, 0], out=u,
              where=direct_vector[:, 0] != 0)
    direct_points = rx+u[:, None]*direct_vector
    blocked = ((direct_vector[:, 0] != 0) & (u > 0) & (u < 1)
               & wall.contains(direct_points))
    direct_visible = ~blocked
    u = np.zeros(count)
    np.divide(wall.x_m-rx[:, 0], image_vector[:, 0], out=u,
              where=image_vector[:, 0] != 0)
    points = rx+u[:, None]*image_vector
    reflection_visible = (((rx[:, 0] > wall.x_m) == (satellite[:, 0] > wall.x_m))
                          & (u > 0) & (u < 1) & wall.contains(points))
    safe_reflected = np.where(reflected > 0, reflected, 1.)
    angle = np.arccos(np.clip(np.abs(image_vector[:, 0])/safe_reflected, 0, 1))
    coefficients = np.asarray(reflection_coefficient_complex(
        angle, wall.material, polarization=wall.polarization), dtype=np.complex128)
    if wall.antenna_lhcp_response != 0j:
        coefficients = np.asarray(circular_reflection_voltage(
            angle, wall.material, receiver_lhcp=wall.antenna_lhcp_response), dtype=np.complex128)
    if wall.water_film_thickness_m > 0:
        parallel, perpendicular = water_film_fresnel(
            angle, wall.material, thickness_m=wall.water_film_thickness_m,
            water_permittivity=wall.water_permittivity)
        # The transfer formula references the outer air/film face. Geometric
        # ranges still reference the substrate wall plane; a film extending
        # into air replaces that air segment. Compensate its round-trip phase.
        reference_phase = np.exp(4j*np.pi*L1_HZ/C_LIGHT*np.cos(angle)*wall.water_film_thickness_m)
        parallel, perpendicular = parallel*reference_phase, perpendicular*reference_phase
        same, cross = .5*(parallel+perpendicular), .5*(parallel-perpendicular)
        pol = _canonical_polarization(wall.polarization)
        if pol == "rhcp":
            coefficients = same+wall.antenna_lhcp_response*cross
        elif pol == "rhcp_cross":
            coefficients = cross
        elif pol == "parallel":
            coefficients = parallel
        elif pol == "perpendicular":
            coefficients = perpendicular
        else:
            coefficients = .5*(np.abs(parallel)+np.abs(perpendicular))
    coefficients = np.where(reflection_visible, coefficients, 0j)
    gains = np.stack([
        np.where(direct_visible & bool(include_direct), amplitude_range/direct, 0.),
        coefficients * amplitude_range/safe_reflected * bool(include_reflection),
    ]).astype(np.complex128)
    rain_losses = np.zeros((2, count))
    if rain_layer is not None:
        rain_losses[0] = rain_layer.loss_db(rx, satellite)
        rain_losses[1] = rain_layer.loss_db(rx, points)+rain_layer.loss_db(points, satellite)
        gains *= 10**(-rain_losses/20)
    ranges = np.stack([direct, reflected])
    rates = np.stack([np.sum(direct_vector*relative_velocity, axis=1)/direct,
                      np.sum(image_vector*image_relative_velocity, axis=1)/safe_reflected])
    wet_delays, wet_rates = np.zeros((2, count)), np.zeros((2, count))
    if wet_delay_layer is not None:
        wet_delays[0], wet_rates[0] = wet_delay_layer.excess_range_and_rate(
            rx, satellite, rx_velocities, sat_velocities)
        # Differentiate the image-ray intersection with the fixed wall plane.
        u_rate = np.zeros(count)
        valid_u = np.where(reflection_visible, u, 0.)
        np.divide(-rx_velocities[:, 0]-valid_u*image_relative_velocity[:, 0], image_vector[:, 0],
                  out=u_rate, where=reflection_visible & (image_vector[:, 0] != 0))
        point_velocities = rx_velocities+u_rate[:, None]*image_vector+valid_u[:, None]*image_relative_velocity
        wet_points = np.where(reflection_visible[:, None], points, rx)
        leg1, rate1 = wet_delay_layer.excess_range_and_rate(rx, wet_points, rx_velocities, point_velocities)
        leg2, rate2 = wet_delay_layer.excess_range_and_rate(wet_points, satellite, point_velocities, sat_velocities)
        wet_delays[1] = np.where(reflection_visible, leg1+leg2, 0.)
        wet_rates[1] = np.where(reflection_visible, rate1+rate2, 0.)
    propagation_ranges, propagation_rates = ranges+wet_delays, rates+wet_rates
    antenna_loss, antenna_code, antenna_phase, antenna_code_rate, antenna_phase_rate = (
        np.zeros((2, count)) for _ in range(5))
    if antenna_response is not None:
        responses = [antenna_response.sample(times, direct_vector, relative_velocity),
                     antenna_response.sample(times,
                         np.where(reflection_visible[:, None], image_vector, direct_vector),
                         np.where(reflection_visible[:, None], image_relative_velocity, relative_velocity))]
        for path, response in enumerate(responses):
            mask = np.ones(count, dtype=bool) if path == 0 else reflection_visible
            for destination, values in ((antenna_loss, response.loss_db), (antenna_code, response.code_bias_m),
                                        (antenna_phase, response.phase_bias_rad), (antenna_code_rate, response.code_bias_rate_mps),
                                        (antenna_phase_rate, response.phase_rate_rad_s)):
                destination[path] = np.where(mask, values, 0.)
        gains *= 10**(-antenna_loss/20)
    if (not np.all(np.isfinite(propagation_ranges)) or np.any(propagation_ranges[0] <= 0)
            or not np.all(np.isfinite(propagation_rates)) or np.any(np.abs(propagation_rates) >= C_LIGHT)):
        raise ValueError("atmosphere-adjusted paths/rates exceed supported range")
    if np.any(np.abs(propagation_rates+antenna_code_rate) >= C_LIGHT):
        raise ValueError("antenna-adjusted code rates exceed supported range")
    # Guard integer chip indexing and phase arithmetic before entering Numba.
    with np.errstate(over="ignore", invalid="ignore"):
        delays = (propagation_ranges-reference)/C_LIGHT
        chips = (times[None, :]-delays-antenna_code/C_LIGHT)*CHIP_RATE
        phases = 2*np.pi*(simulator.intermediate_freq*times[None, :]-L1_HZ*delays)+antenna_phase
    if (not np.all(np.isfinite(chips)) or np.any(np.abs(chips) > 2**53-1)
            or not np.all(np.isfinite(phases))):
        raise ValueError("path phases exceed supported numeric range")
    out = _moving_kernel()(generate_ca_code(prn), propagation_ranges, propagation_rates, gains,
                           antenna_code, antenna_code_rate, antenna_phase, antenna_phase_rate, simulator.sampling_freq,
                           simulator.intermediate_freq, start, reference, nav_bit,
                           1/simulator.sampling_freq if integrate_samples else 0.)
    _add_noise(out, simulator.noise_floor_db, simulator.noise_seed)
    return WallSignalBlock(out.ravel(), direct, reflected, direct_visible,
                           reflection_visible, np.where(reflection_visible[:, None], points, np.nan),
                           coefficients, rates[0], rates[1], rain_losses[0], rain_losses[1],
                           wet_delays[0], wet_delays[1], wet_rates[0], wet_rates[1],
                           antenna_loss[0], antenna_loss[1], antenna_code[0], antenna_code[1],
                           antenna_phase[0], antenna_phase[1], antenna_code_rate[0], antenna_code_rate[1],
                           antenna_phase_rate[0], antenna_phase_rate[1])
