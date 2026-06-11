"""True-NLOS pseudorange bias model (direct path blocked).

When a satellite's direct line of sight is blocked by a building, the receiver
cannot track the (absent) direct signal. Instead it acquires and tracks the
strongest arriving replica -- a reflected or diffracted path -- whose code phase
is delayed by that path's *full* geometric excess delay. The measured
pseudorange is therefore biased long by the excess delay of the tracked replica.
This is the dominant urban positioning error and, unlike the bounded LOS
multipath bias, scales to tens of metres.

The diffraction amplitude model (knife-edge vs UTD) matters here because it sets
each diffracted replica's carrier-to-noise ratio, hence whether it clears the
receiver's tracking threshold and which path ends up being tracked. A model that
over-attenuates diffraction (the knife edge) drops trackable NLOS paths that a
more accurate model (UTD) -- and the real receiver -- would keep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def path_cn0_dbhz(amplitude_ratio: float, cn0_los_dbhz: float = 45.0) -> float:
    """Carrier-to-noise of a replica relative to the open-sky LOS C/N0.

    ``amplitude_ratio`` is the replica field amplitude divided by the free-space
    direct amplitude; C/N0 drops by 20*log10(amplitude_ratio).
    """
    a = max(float(amplitude_ratio), 1e-12)
    return float(cn0_los_dbhz) + 20.0 * math.log10(a)


@dataclass
class TrackedPath:
    excess_delay_m: float
    amplitude_ratio: float
    cn0_dbhz: float


def select_tracked_path(
    paths,
    *,
    cn0_los_dbhz: float = 45.0,
    cn0_threshold_dbhz: float = 28.0,
) -> TrackedPath | None:
    """Pick the strongest replica a receiver could track for an NLOS satellite.

    ``paths`` is any iterable of objects exposing ``amplitude`` (field ratio to
    the free-space direct) and ``excess_delay`` (metres). Returns the strongest
    path whose received C/N0 clears ``cn0_threshold_dbhz``, or None if no replica
    is trackable (the satellite would then drop out / not be measured).
    """
    best: TrackedPath | None = None
    for p in paths:
        amp = float(p.amplitude)
        cn0 = path_cn0_dbhz(amp, cn0_los_dbhz)
        if cn0 < cn0_threshold_dbhz:
            continue
        if best is None or amp > best.amplitude_ratio:
            best = TrackedPath(float(p.excess_delay), amp, cn0)
    return best


def nlos_bias_m(
    paths,
    *,
    cn0_los_dbhz: float = 45.0,
    cn0_threshold_dbhz: float = 28.0,
) -> float | None:
    """NLOS pseudorange bias (m) = excess delay of the tracked replica, or None."""
    tracked = select_tracked_path(
        paths, cn0_los_dbhz=cn0_los_dbhz, cn0_threshold_dbhz=cn0_threshold_dbhz)
    return None if tracked is None else tracked.excess_delay_m


@dataclass
class ReflectionReplica:
    """A specular reflection turned into an NLOS tracking candidate.

    ``amplitude`` is the field ratio to the free-space direct, i.e. the Fresnel
    reflection magnitude |Gamma| at the path's incidence angle (spreading over
    the small excess path is neglected, consistent with how the diffracted
    replicas are normalised). This lets reflection and diffraction replicas be
    pooled and compared on the same C/N0 footing in :func:`select_tracked_path`.
    """

    excess_delay: float
    amplitude: float
    incidence_angle: float
    triangle_id: int


def reflection_replicas(
    refl_paths,
    *,
    material: str = "concrete",
    polarization: str = "rhcp",
    freq_hz=None,
    ground_material: str | None = None,
):
    """Attach a Fresnel amplitude to image-method reflection paths.

    ``refl_paths`` is an iterable of objects with ``excess_delay``,
    ``incidence_angle`` (radians from the surface normal) and ``triangle_id``
    (-1 for the ground plane). Returns a list of :class:`ReflectionReplica`,
    each carrying the field ratio |Gamma| as ``amplitude`` so it can be pooled
    with diffracted replicas for NLOS tracking. ``ground_material`` overrides the
    material for ground reflections (triangle_id == -1) when given.
    """
    from gnss_gpu.fresnel import GPS_L1_FREQ, reflection_coefficient

    freq = GPS_L1_FREQ if freq_hz is None else freq_hz
    out = []
    for p in refl_paths:
        tri_id = int(getattr(p, "triangle_id", -1))
        mat = material
        if ground_material is not None and tri_id == -1:
            mat = ground_material
        angle = float(p.incidence_angle)
        amp = float(reflection_coefficient(
            angle, material=mat, freq_hz=freq, polarization=polarization))
        out.append(ReflectionReplica(
            excess_delay=float(p.excess_delay),
            amplitude=amp,
            incidence_angle=angle,
            triangle_id=tri_id,
        ))
    return out


def pooled_nlos_bias_m(
    *path_groups,
    cn0_los_dbhz: float = 45.0,
    cn0_threshold_dbhz: float = 28.0,
) -> float | None:
    """NLOS bias from the strongest trackable replica across several path lists.

    Each positional argument is an iterable of replica objects (diffraction
    paths, :class:`ReflectionReplica`, ...) exposing ``amplitude`` and
    ``excess_delay``. They are pooled and the single strongest trackable replica
    sets the pseudorange bias -- so a strong reflection (large |Gamma|, tens of
    metres of excess delay) outvotes a weak grazing-diffraction replica, which is
    what lets the predicted NLOS bias reach the real urban magnitude.
    """
    pooled = []
    for group in path_groups:
        if group:
            pooled.extend(group)
    return nlos_bias_m(
        pooled,
        cn0_los_dbhz=cn0_los_dbhz,
        cn0_threshold_dbhz=cn0_threshold_dbhz,
    )


def predict_nlos_bias_samples_m(
    per_sat_paths,
    *,
    cn0_los_dbhz: float = 45.0,
    cn0_threshold_dbhz: float = 28.0,
):
    """Map a list of per-NLOS-satellite path lists to tracked-bias samples (m).

    Satellites with no trackable replica are dropped (they would not produce a
    measurement), so the returned list length is the count of *measured* NLOS
    satellites under this amplitude model.
    """
    out = []
    for paths in per_sat_paths:
        bias = nlos_bias_m(
            paths, cn0_los_dbhz=cn0_los_dbhz, cn0_threshold_dbhz=cn0_threshold_dbhz)
        if bias is not None:
            out.append(bias)
    return out


__all__ = [
    "path_cn0_dbhz",
    "TrackedPath",
    "select_tracked_path",
    "nlos_bias_m",
    "ReflectionReplica",
    "reflection_replicas",
    "pooled_nlos_bias_m",
    "predict_nlos_bias_samples_m",
]
