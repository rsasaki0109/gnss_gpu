"""Prescribed wet antenna/radome response, awaiting empirical calibration.

Loss, code-epoch bias and carrier phase intercept are independent controls.
This is not a water-film EM solution, a rainfall law or a receiver noise model.
"""

from dataclasses import dataclass

import numpy as np

from gnss_gpu.input_validation import finite_float, nonnegative_float, positive_float


@dataclass(frozen=True)
class AntennaResponseSamples:
    wetness: np.ndarray
    loss_db: np.ndarray
    code_bias_m: np.ndarray
    phase_bias_rad: np.ndarray
    code_bias_rate_mps: np.ndarray
    phase_rate_rad_s: np.ndarray


@dataclass(frozen=True)
class WetAntennaResponse:
    """Smooth prescribed wetting/plateau/drying and simple elevation dependence.

    Wetness w rises 0->1 with cubic smoothstep, stays at one, then falls to
    zero with smoothstep. Each response is w*(constant + elevation_term*q),
    q=1-clip(ray_up/|ray|,0,1). Lower-hemisphere rays share the horizon value;
    this is an illustrative pattern basis, not a measured antenna pattern.
    Code bias [m] is an envelope/group-delay control. Phase [rad] is a separate
    intercept at the received carrier; it is NOT -2*pi*f_L1*code_bias/c.
    """

    wet_start_s: float = .05
    wetting_s: float = .05
    drying_start_s: float = .2
    drying_s: float = .2
    loss_db: float = 0.
    elevation_loss_db: float = 0.
    code_bias_m: float = 0.
    elevation_code_bias_m: float = 0.
    phase_bias_rad: float = 0.
    elevation_phase_bias_rad: float = 0.

    def __post_init__(self):
        for name in ("wet_start_s", "loss_db", "elevation_loss_db"):
            object.__setattr__(self, name, nonnegative_float(name, getattr(self, name)))
        for name in ("wetting_s", "drying_s"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        for name in ("drying_start_s", "code_bias_m", "elevation_code_bias_m",
                     "phase_bias_rad", "elevation_phase_bias_rad"):
            object.__setattr__(self, name, finite_float(name, getattr(self, name)))
        if self.drying_start_s < self.wet_start_s+self.wetting_s:
            raise ValueError("drying_start_s must be after wetting completes")

    def wetness(self, time_s):
        """Return w and dw/dt; schedule times are prescribed, not inferred."""
        t = np.asarray(time_s, dtype=float)
        if not np.all(np.isfinite(t)):
            raise ValueError("time_s must be finite")
        rise = np.clip((t-self.wet_start_s)/self.wetting_s, 0., 1.)
        fall = np.clip((t-self.drying_start_s)/self.drying_s, 0., 1.)
        up, down = rise*rise*(3-2*rise), fall*fall*(3-2*fall)
        return up*(1-down), 6*rise*(1-rise)/self.wetting_s*(1-down)-up*6*fall*(1-fall)/self.drying_s

    def sample(self, time_s, ray_vector_m, ray_velocity_mps=(0., 0., 0.)):
        """Evaluate per-ray response and analytic rates at sample centers.

        Direction points from receiver towards the arriving ray's source/image.
        Inputs broadcast over (...,3); time must broadcast with the leading
        dimensions. Amplitude is held locally constant over a sampling aperture.
        """
        ray, velocity = np.broadcast_arrays(np.asarray(ray_vector_m, dtype=float), np.asarray(ray_velocity_mps, dtype=float))
        if ray.ndim < 1 or ray.shape[-1] != 3 or not np.all(np.isfinite(ray)) or not np.all(np.isfinite(velocity)):
            raise ValueError("ray vectors and velocities must be finite (...,3) arrays")
        length = np.linalg.norm(ray, axis=-1)
        if np.any(length <= 0) or not np.all(np.isfinite(length)):
            raise ValueError("ray vectors must have finite positive length")
        length_rate = np.sum(ray*velocity, axis=-1)/length
        up = ray[..., 2]/length
        up_rate = (velocity[..., 2]-up*length_rate)/length
        q = 1-np.clip(up, 0., 1.)
        q_rate = np.where((up > 0) & (up < 1), -up_rate, 0.)
        w, w_rate = self.wetness(time_s)

        def response(constant, slope):
            value = constant+slope*q
            return w*value, w_rate*value+w*slope*q_rate

        loss, _ = response(self.loss_db, self.elevation_loss_db)
        code, code_rate = response(self.code_bias_m, self.elevation_code_bias_m)
        phase, phase_rate = response(self.phase_bias_rad, self.elevation_phase_bias_rad)
        values = (loss, code, phase, code_rate, phase_rate)
        if any(not np.all(np.isfinite(value)) for value in values):
            raise ValueError("antenna response exceeds supported numeric range")
        return AntennaResponseSamples(np.broadcast_to(w, np.shape(loss)), loss, code, phase, code_rate, phase_rate)
