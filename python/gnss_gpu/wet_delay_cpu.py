"""Controlled neutral wet-delay residual in a flat, uniform refractivity slab.

This is a sensitivity model, not a water-vapour retrieval or a rainfall law.
Straight rays through a full slab give the elementary csc(elevation) mapping.
"""

from dataclasses import dataclass

import numpy as np

from gnss_gpu.input_validation import finite_float
from gnss_gpu.rain_cpu import segment_layer_length


@dataclass(frozen=True)
class WetDelayResidualLayer:
    """Actual-minus-applied wet delay for the entire vertical slab, in metres.

    A signed residual models under/over-correction; negative values do not mean
    negative physical water vapour. The implied residual refractivity is
    constant: delta(n-1)=zenith_residual_m/(top_m-bottom_m). No bending,
    horizontal gradient, time evolution, dispersion or rain-rate relation.
    """

    zenith_residual_m: float = 0.
    top_m: float = 3000.
    bottom_m: float = 0.

    def __post_init__(self):
        for name in ("zenith_residual_m", "top_m", "bottom_m"):
            object.__setattr__(self, name, finite_float(name, getattr(self, name)))
        if self.top_m <= self.bottom_m:
            raise ValueError("wet-delay layer must have positive height")

    def excess_range_and_rate(self, start, end, start_velocity=(0., 0., 0.),
                               end_velocity=(0., 0., 0.)):
        """Signed nondispersive excess path [m] and its derivative [m/s].

        Endpoints/velocities broadcast over (...,3). Derivatives are piecewise
        analytic for slab clipping. At exact boundary/tangency cusps a unique
        classical derivative need not exist; hard boundary masks are not smooth
        weather transitions. Tests of rates avoid those cusps.
        """
        a, b, va, vb = np.broadcast_arrays(np.asarray(start, dtype=float), np.asarray(end, dtype=float),
                                           np.asarray(start_velocity, dtype=float), np.asarray(end_velocity, dtype=float))
        if a.ndim < 1 or a.shape[-1] != 3 or any(not np.all(np.isfinite(value)) for value in (a, b, va, vb)):
            raise ValueError("endpoints and velocities must be finite (...,3) arrays")
        inside = segment_layer_length(a, b, self.top_m, layer_bottom_m=self.bottom_m)
        delta, velocity = b-a, vb-va
        length = np.linalg.norm(delta, axis=-1)
        length_rate = np.zeros_like(length)
        np.divide(np.sum(delta*velocity, axis=-1), length, out=length_rate, where=length > 0)
        dz, vz = delta[..., 2], velocity[..., 2]
        t0, t1, dt0, dt1 = (np.zeros_like(dz) for _ in range(4))
        np.divide(self.bottom_m-a[..., 2], dz, out=t0, where=dz != 0)
        np.divide(self.top_m-a[..., 2], dz, out=t1, where=dz != 0)
        np.divide(-va[..., 2]-t0*vz, dz, out=dt0, where=dz != 0)
        np.divide(-va[..., 2]-t1*vz, dz, out=dt1, where=dz != 0)
        lower, upper = np.minimum(t0, t1), np.maximum(t0, t1)
        d_lower, d_upper = np.where(t0 <= t1, dt0, dt1), np.where(t0 <= t1, dt1, dt0)
        fraction = np.maximum(0., np.minimum(1., upper)-np.maximum(0., lower))
        fraction_rate = np.where(fraction > 0,
                                 np.where(upper < 1, d_upper, 0.)-np.where(lower > 0, d_lower, 0.), 0.)
        fraction = np.where(dz == 0, ((a[..., 2] >= self.bottom_m) & (a[..., 2] <= self.top_m)).astype(float), fraction)
        fraction_rate = np.where(dz == 0, 0., fraction_rate)
        scale = self.zenith_residual_m/(self.top_m-self.bottom_m)
        return scale*inside, scale*(fraction*length_rate+fraction_rate*length)
