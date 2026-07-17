"""Receiver antenna gain patterns for the scenario engine.

:mod:`gnss_gpu.scenario` simulates C/N0 from an elevation-shaped carrier
power model (``cn0_zenith_dbhz * sin(elevation)``) minus any NLOS/diffraction
attenuation. That is equivalent to assuming an ideal, lossless, perfectly
isotropic-over-the-upper-hemisphere receive antenna, which no real GNSS
antenna is: survey-grade choke-ring/patch antennas have several dB of gain
at zenith and roll off toward the horizon, while a phone's internal linear
antenna is lossy (negative gain) everywhere and has some azimuth ripple from
the surrounding chassis.

This module adds that as an optional, composable gain term:
``cn0_dbhz += pattern.gain_db(elevation_rad, azimuth_rad)``.

:class:`AntennaPattern` represents gain (dBic, referenced to an ideal
circularly-polarized isotropic radiator) as a function of elevation and
azimuth, built from a regular ``(elevation, azimuth)`` grid with bilinear
interpolation. Elevation outside the grid is clamped to the nearest edge
sample (flat extrapolation); azimuth is periodic (wraps at 360 degrees).

Azimuth-symmetric patterns -- the common case for a simulation-grade survey
antenna model, since real choke-ring/patch antennas are close to
rotationally symmetric -- can be built from a single 1-D elevation cut via
:meth:`AntennaPattern.from_elevation_cut`.

Four presets are provided via :meth:`AntennaPattern.preset`: ``"isotropic"``
(0 dB everywhere, i.e. today's implicit model), ``"patch"`` (choke-ring-ish
survey patch), ``"helix"`` (a volute/quadrifilar helix antenna), and
``"smartphone"`` (a lossy, mildly azimuth-rippled internal linear antenna).
See each builder's docstring below for the exact numbers used and the
reasoning behind them; they are simulation-grade approximations of typical
published antenna patterns, not a specific antenna's measured pattern.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["AntennaPattern"]

_TWO_PI = 2.0 * math.pi


class AntennaPattern:
    """Elevation/azimuth-dependent receiver antenna gain, in dBic.

    Internally stores a regular ``(n_el, n_az)`` grid of gain samples and
    bilinearly interpolates :meth:`gain_db` queries against it. Prefer the
    constructors below over calling ``AntennaPattern(...)`` directly:

    - :meth:`from_table` -- a user-defined grid, in degrees.
    - :meth:`from_elevation_cut` -- an azimuth-symmetric pattern from a
      single 1-D elevation cut.
    - :meth:`preset` -- one of the built-in presets described in the module
      docstring.
    """

    def __init__(
        self,
        el_grid_rad: np.ndarray,
        az_grid_rad: np.ndarray,
        gain_db_grid: np.ndarray,
    ) -> None:
        el = np.asarray(el_grid_rad, dtype=np.float64)
        az = np.asarray(az_grid_rad, dtype=np.float64)
        gain = np.asarray(gain_db_grid, dtype=np.float64)

        if el.ndim != 1 or el.size < 2:
            raise ValueError("AntennaPattern: el_grid_rad must be 1-D with >= 2 points")
        if az.ndim != 1 or az.size < 2:
            raise ValueError("AntennaPattern: az_grid_rad must be 1-D with >= 2 points")
        if not np.all(np.diff(el) > 0.0):
            raise ValueError("AntennaPattern: el_grid_rad must be strictly increasing")
        if not np.all(np.diff(az) > 0.0):
            raise ValueError("AntennaPattern: az_grid_rad must be strictly increasing")
        if gain.shape != (el.size, az.size):
            raise ValueError(
                "AntennaPattern: gain_db_grid must have shape "
                f"(n_el, n_az) = {(el.size, az.size)}, got {gain.shape}"
            )
        if not np.isfinite(gain).all():
            raise ValueError("AntennaPattern: gain_db_grid must be finite")

        # Azimuth is periodic. If the supplied grid does not already span a
        # full 360 degrees (e.g. samples at 0, 10, ..., 350), append a
        # wrap-around column equal to the first column at az[0] + 2*pi so
        # that interpolation across the 350->360(=0) seam is bilinear too,
        # instead of needing a special case in gain_db().
        if az[-1] - az[0] < _TWO_PI - 1.0e-9:
            az = np.concatenate([az, [az[0] + _TWO_PI]])
            gain = np.concatenate([gain, gain[:, :1]], axis=1)

        self._el = el
        self._az = az
        self._gain = gain

    # -- construction ---------------------------------------------------

    @classmethod
    def from_table(cls, el_deg, az_deg, gain_db) -> "AntennaPattern":
        """Build a pattern from a user-defined ``(elevation, azimuth)`` grid.

        ``el_deg`` and ``az_deg`` are 1-D, strictly increasing, in degrees
        (elevation typically ``[0, 90]``, azimuth typically ``[0, 360)``).
        ``gain_db`` has shape ``(len(el_deg), len(az_deg))``.
        """
        el_deg = np.asarray(el_deg, dtype=np.float64)
        az_deg = np.asarray(az_deg, dtype=np.float64)
        gain_db = np.asarray(gain_db, dtype=np.float64)
        return cls(np.radians(el_deg), np.radians(az_deg), gain_db)

    @classmethod
    def from_elevation_cut(cls, el_deg, gain_db) -> "AntennaPattern":
        """Build an azimuth-symmetric pattern from a 1-D elevation cut.

        ``el_deg`` and ``gain_db`` are 1-D arrays of equal length, strictly
        increasing in elevation (degrees). The resulting pattern's
        :meth:`gain_db` ignores azimuth entirely.
        """
        el_deg = np.asarray(el_deg, dtype=np.float64)
        gain_db = np.asarray(gain_db, dtype=np.float64)
        if el_deg.ndim != 1 or gain_db.ndim != 1 or el_deg.shape != gain_db.shape:
            raise ValueError(
                "AntennaPattern.from_elevation_cut: el_deg and gain_db must be "
                "1-D arrays of equal length"
            )
        az_deg = np.array([0.0, 360.0], dtype=np.float64)
        gain_grid = np.column_stack([gain_db, gain_db])
        return cls.from_table(el_deg, az_deg, gain_grid)

    @classmethod
    def preset(cls, name: str) -> "AntennaPattern":
        """Build one of the built-in presets by name (case-insensitive).

        See the module docstring and each ``_preset_*`` builder below for
        the gain numbers and the reasoning behind them.
        """
        key = str(name).strip().lower()
        builder = _PRESET_BUILDERS.get(key)
        if builder is None:
            raise ValueError(
                f"AntennaPattern.preset: unknown preset {name!r}; expected one "
                f"of {sorted(_PRESET_BUILDERS)}"
            )
        return builder()

    # -- evaluation -------------------------------------------------------

    def gain_db(self, elevation_rad, azimuth_rad):
        """Bilinearly-interpolated gain (dBic) at ``(elevation_rad, azimuth_rad)``.

        Both arguments accept scalars or numpy arrays of any (broadcastable)
        shape; the return type mirrors the input (``float`` for scalar
        input, an ``ndarray`` of the broadcast shape otherwise). Elevation
        outside the stored grid is clamped to the nearest edge sample;
        azimuth wraps modulo 360 degrees.
        """
        el_in = np.asarray(elevation_rad, dtype=np.float64)
        az_in = np.asarray(azimuth_rad, dtype=np.float64)
        el_b, az_b = np.broadcast_arrays(el_in, az_in)
        scalar_input = el_b.ndim == 0

        el_flat = np.atleast_1d(el_b).ravel()
        az_flat = np.atleast_1d(az_b).ravel()

        el_c = np.clip(el_flat, self._el[0], self._el[-1])
        az_c = np.mod(az_flat - self._az[0], _TWO_PI) + self._az[0]

        i_el = np.clip(np.searchsorted(self._el, el_c, side="right") - 1, 0, self._el.size - 2)
        i_az = np.clip(np.searchsorted(self._az, az_c, side="right") - 1, 0, self._az.size - 2)

        el0 = self._el[i_el]
        el1 = self._el[i_el + 1]
        t_el = np.where(el1 > el0, (el_c - el0) / np.where(el1 > el0, el1 - el0, 1.0), 0.0)

        az0 = self._az[i_az]
        az1 = self._az[i_az + 1]
        t_az = np.where(az1 > az0, (az_c - az0) / np.where(az1 > az0, az1 - az0, 1.0), 0.0)

        g00 = self._gain[i_el, i_az]
        g01 = self._gain[i_el, i_az + 1]
        g10 = self._gain[i_el + 1, i_az]
        g11 = self._gain[i_el + 1, i_az + 1]

        g0 = g00 * (1.0 - t_az) + g01 * t_az
        g1 = g10 * (1.0 - t_az) + g11 * t_az
        gain = g0 * (1.0 - t_el) + g1 * t_el

        result = gain.reshape(el_b.shape)
        if scalar_input:
            return float(result)
        return result


# ---------------------------------------------------------------------------
# Built-in presets
#
# All presets are simulation-grade approximations (smooth analytic shapes
# fit to sensible zenith/horizon numbers), not a measured pattern of any
# specific antenna model. Elevation is sampled every 5 degrees from 0 to 90;
# azimuth (where not symmetric) every 22.5 degrees over the full circle.
# ---------------------------------------------------------------------------

_EL_DEG = np.arange(0.0, 90.0 + 1.0e-9, 5.0)


def _preset_isotropic() -> AntennaPattern:
    """0 dBic at every elevation and azimuth -- today's implicit model.

    Selecting this preset is bit-for-bit equivalent to leaving
    ``ScenarioConfig.antenna = None`` (the C/N0 gain term added is exactly
    0.0 everywhere), it just makes the "no antenna shaping" choice explicit
    in a config.
    """
    return AntennaPattern.from_elevation_cut(np.array([0.0, 90.0]), np.array([0.0, 0.0]))


def _preset_patch() -> AntennaPattern:
    """Choke-ring-ish survey patch antenna: azimuth-symmetric.

    +3.0 dBic at zenith, rolling off smoothly to -5.0 dB at the horizon.
    A choke ring (or choke-ring-like ground plane) is specifically designed
    to suppress low-elevation/multipath response, so most of the gain drop
    happens in the bottom ~30 degrees -- modeled here as
    ``gain(el) = 3.0 - 8.0 * (1 - sin(el)) ** 1.5``, which gives roughly:
    0 deg -> -5.0 dB, 10 deg -> -3.0 dB, 30 deg -> +0.2 dB, 45 deg -> +1.7 dB,
    60 deg -> +2.6 dB, 90 deg -> +3.0 dB.
    """
    sin_el = np.sin(np.radians(_EL_DEG))
    zenith, horizon = 3.0, -5.0
    gain = zenith + (horizon - zenith) * np.power(1.0 - sin_el, 1.5)
    return AntennaPattern.from_elevation_cut(_EL_DEG, gain)


def _preset_helix() -> AntennaPattern:
    """Volute/quadrifilar helix antenna: azimuth-symmetric, gentler rolloff.

    +2.0 dBic at zenith, -2.0 dB at the horizon, linear in ``sin(elevation)``
    (no choke ring, so no aggressive low-elevation suppression):
    ``gain(el) = 2.0 - 4.0 * (1 - sin(el))``, giving roughly: 0 deg -> -2.0 dB,
    30 deg -> 0.0 dB, 45 deg -> +0.8 dB, 60 deg -> +1.5 dB, 90 deg -> +2.0 dB.
    """
    sin_el = np.sin(np.radians(_EL_DEG))
    zenith, horizon = 2.0, -2.0
    gain = zenith + (horizon - zenith) * (1.0 - sin_el)
    return AntennaPattern.from_elevation_cut(_EL_DEG, gain)


def _preset_smartphone() -> AntennaPattern:
    """Internal phone antenna: lossy, linearly polarized, mild azimuth ripple.

    A phone's GNSS antenna is small, often linearly polarized (a few dB
    polarization mismatch loss against a circularly-polarized satellite
    signal) and sits close to a lossy chassis/hand/body, so gain is negative
    everywhere: -3.0 dBic at zenith down to -8.0 dB near the horizon, via
    ``base(el) = -3.0 - 5.0 * (1 - sin(el)) ** 1.2``, plus a +/-1.0 dB
    two-lobe azimuth ripple (``cos(2*azimuth)``) representing the chassis'
    non-uniform loading. Roughly: 0 deg -> -7..-9 dB, 30 deg -> -4.2..-6.2 dB,
    90 deg -> -2.0..-4.0 dB (range is the azimuth ripple).
    """
    az_deg = np.arange(0.0, 360.0, 22.5)
    sin_el = np.sin(np.radians(_EL_DEG))
    zenith, horizon = -3.0, -8.0
    base = zenith + (horizon - zenith) * np.power(1.0 - sin_el, 1.2)
    ripple_amp_db = 1.0
    ripple = ripple_amp_db * np.cos(2.0 * np.radians(az_deg))
    gain_grid = base[:, None] + ripple[None, :]
    return AntennaPattern.from_table(_EL_DEG, az_deg, gain_grid)


_PRESET_BUILDERS = {
    "isotropic": _preset_isotropic,
    "patch": _preset_patch,
    "helix": _preset_helix,
    "smartphone": _preset_smartphone,
}
