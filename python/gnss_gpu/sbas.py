"""SBAS and QZSS augmentation correction models for GNSS positioning.

Provides SBAS (WAAS/MSAS/EGNOS) fast/long-term/ionospheric corrections
and QZSS CLAS centimeter-level augmentation.  All pure Python -- these
corrections involve small data volumes that do not benefit from GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_LIGHT = 299792458.0

# SBAS system identifiers by PRN range
SBAS_SYSTEMS = {
    "WAAS": range(131, 139),   # Wide Area Augmentation System (Americas)
    "EGNOS": range(120, 129),  # European Geostationary Navigation Overlay Service
    "MSAS": range(129, 138),   # Multi-functional Satellite Augmentation System (Japan)
    "GAGAN": range(127, 129),  # GPS Aided GEO Augmented Navigation (India)
}

# QZSS PRN range (per IS-QZSS)
QZSS_PRN_MIN = 193
QZSS_PRN_MAX = 202

# Earth parameters for pierce-point calculation
EARTH_RADIUS = 6371000.0          # mean Earth radius [m]
IONO_HEIGHT = 350000.0            # single-layer ionosphere height [m]

# UDRE (User Differential Range Error) index -> sigma [m]
# Table from RTCA DO-229D
UDRE_SIGMA = {
    0: 1.0,
    1: 1.5,
    2: 2.0,
    3: 3.0,
    4: 4.5,
    5: 6.0,
    6: 8.0,
    7: 10.0,
    8: 15.0,
    9: 20.0,
    10: 50.0,
    11: 100.0,
    12: 150.0,
    13: float("inf"),   # not monitored
    14: float("inf"),
    15: float("inf"),
}

# GIVE (Grid Ionospheric Vertical Error) index -> sigma [m]
GIVE_SIGMA = {
    0: 0.3,
    1: 0.6,
    2: 0.9,
    3: 1.2,
    4: 1.5,
    5: 1.8,
    6: 2.1,
    7: 2.4,
    8: 2.7,
    9: 3.0,
    10: 3.6,
    11: 4.5,
    12: 6.0,
    13: 9.0,
    14: 45.0,
    15: float("inf"),
}


# ---------------------------------------------------------------------------
# Data classes for correction data
# ---------------------------------------------------------------------------

@dataclass
class FastCorrection:
    """SBAS fast clock correction for a single satellite."""
    prc: float = 0.0      # pseudorange correction [m]
    rrc: float = 0.0      # range rate correction [m/s]
    udre_index: int = 13   # UDRE indicator (0-15)
    t_apply: float = 0.0  # time of applicability [GPS seconds]


@dataclass
class LongTermCorrection:
    """SBAS long-term satellite correction (position + clock)."""
    dx: float = 0.0   # ECEF X correction [m]
    dy: float = 0.0   # ECEF Y correction [m]
    dz: float = 0.0   # ECEF Z correction [m]
    da_f0: float = 0.0  # clock offset correction [m]


@dataclass
class IonoGridPoint:
    """Single ionospheric grid point (IGP)."""
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    vertical_delay: float = 0.0   # vertical ionospheric delay [m]
    give_index: int = 15          # GIVE indicator (0-15)


# ---------------------------------------------------------------------------
# SBAS Correction
# ---------------------------------------------------------------------------

class SBASCorrection:
    """SBAS (WAAS/MSAS/EGNOS) differential correction handler.

    Manages fast clock corrections, long-term orbital/clock corrections,
    and ionospheric grid corrections broadcast by SBAS GEO satellites.
    """

    def __init__(self):
        self.fast_corrections: dict[int, FastCorrection] = {}
        self.long_term: dict[int, LongTermCorrection] = {}
        self.iono_grid: list[IonoGridPoint] = []

    # -- Fast correction --------------------------------------------------

    def set_fast_correction(self, prn: int, prc: float, rrc: float = 0.0,
                            udre_index: int = 0, t_apply: float = 0.0):
        """Store a fast clock correction for a satellite.

        Parameters
        ----------
        prn : int
            Satellite PRN number.
        prc : float
            Pseudorange correction [m].
        rrc : float
            Range-rate correction [m/s].
        udre_index : int
            UDRE indicator (0-15, lower is better).
        t_apply : float
            Time of applicability [GPS seconds of week].
        """
        self.fast_corrections[prn] = FastCorrection(
            prc=prc, rrc=rrc, udre_index=udre_index, t_apply=t_apply)

    def apply_fast_correction(self, prn: int, pseudorange: float,
                              t_current: float = 0.0) -> float:
        """Apply SBAS fast clock correction to a pseudorange.

        Parameters
        ----------
        prn : int
            Satellite PRN number.
        pseudorange : float
            Raw pseudorange measurement [m].
        t_current : float
            Current GPS time [s] for range-rate extrapolation.

        Returns
        -------
        float
            Corrected pseudorange [m].  If no correction is available
            for the given PRN, the original pseudorange is returned.
        """
        if prn not in self.fast_corrections:
            return pseudorange
        fc = self.fast_corrections[prn]
        dt = t_current - fc.t_apply
        correction = fc.prc + fc.rrc * dt
        return pseudorange + correction

    # -- Long-term correction ---------------------------------------------

    def set_long_term_correction(self, prn: int, dx: float, dy: float,
                                 dz: float, da_f0: float = 0.0):
        """Store a long-term correction for a satellite.

        Parameters
        ----------
        prn : int
            Satellite PRN.
        dx, dy, dz : float
            ECEF position corrections [m].
        da_f0 : float
            Clock offset correction [m].
        """
        self.long_term[prn] = LongTermCorrection(
            dx=dx, dy=dy, dz=dz, da_f0=da_f0)

    def apply_long_term_correction(self, prn: int, sat_pos: np.ndarray,
                                   clock_offset: float = 0.0):
        """Apply long-term orbital and clock corrections.

        Parameters
        ----------
        prn : int
            Satellite PRN.
        sat_pos : array_like, shape (3,)
            Satellite ECEF position [m].
        clock_offset : float
            Satellite clock offset [m].

        Returns
        -------
        corrected_pos : numpy.ndarray, shape (3,)
        corrected_clock : float
        """
        sat_pos = np.asarray(sat_pos, dtype=np.float64)
        if prn not in self.long_term:
            return sat_pos.copy(), clock_offset
        lt = self.long_term[prn]
        corrected_pos = sat_pos + np.array([lt.dx, lt.dy, lt.dz])
        corrected_clock = clock_offset + lt.da_f0
        return corrected_pos, corrected_clock

    # -- Ionospheric grid correction --------------------------------------

    def set_iono_grid(self, grid_points: list[IonoGridPoint]):
        """Load ionospheric grid points.

        Parameters
        ----------
        grid_points : list of IonoGridPoint
            SBAS ionospheric grid points with vertical delays and GIVE.
        """
        self.iono_grid = list(grid_points)

    def apply_iono_correction(self, lat: float, lon: float,
                              el: float, az: float) -> float:
        """Apply SBAS ionospheric grid correction using pierce-point interpolation.

        Computes the ionospheric pierce point (IPP), locates the four
        surrounding grid points, and performs bilinear interpolation of
        the vertical delay.  The vertical delay is then mapped to slant
        using an obliquity factor.

        Parameters
        ----------
        lat : float
            Receiver geodetic latitude [rad].
        lon : float
            Receiver geodetic longitude [rad].
        el : float
            Satellite elevation angle [rad].
        az : float
            Satellite azimuth angle [rad].

        Returns
        -------
        float
            Slant ionospheric delay [m].  Returns 0.0 if fewer than 4
            grid points are loaded.
        """
        if len(self.iono_grid) < 4:
            return 0.0

        # Compute ionospheric pierce point (IPP)
        ipp_lat, ipp_lon = _iono_pierce_point(lat, lon, el, az)

        # Obliquity (slant) factor
        obliquity = _obliquity_factor(el)

        # Find surrounding grid points and interpolate
        vertical_delay = _bilinear_interpolate(
            self.iono_grid, math.degrees(ipp_lat), math.degrees(ipp_lon))

        return vertical_delay * obliquity

    # -- Integrity --------------------------------------------------------

    def integrity_check(self, prn: int) -> dict:
        """Check SBAS integrity information for a satellite.

        Parameters
        ----------
        prn : int
            Satellite PRN.

        Returns
        -------
        dict
            ``usable`` : bool
                Whether the satellite is usable.
            ``udre_index`` : int
                UDRE indicator (0-15).
            ``udre_sigma`` : float
                1-sigma UDRE [m].
        """
        if prn not in self.fast_corrections:
            return {"usable": False, "udre_index": 15,
                    "udre_sigma": float("inf")}

        fc = self.fast_corrections[prn]
        sigma = UDRE_SIGMA.get(fc.udre_index, float("inf"))
        usable = fc.udre_index <= 12  # indices 13-15 => not monitored
        return {"usable": usable, "udre_index": fc.udre_index,
                "udre_sigma": sigma}

    def iono_integrity(self, lat_deg: float, lon_deg: float) -> dict:
        """Check ionospheric grid integrity near a location.

        Parameters
        ----------
        lat_deg, lon_deg : float
            Location in degrees.

        Returns
        -------
        dict
            ``give_index`` : int
                Minimum GIVE index among surrounding grid points.
            ``give_sigma`` : float
                Corresponding 1-sigma GIVE [m].
        """
        if not self.iono_grid:
            return {"give_index": 15, "give_sigma": float("inf")}

        # Find closest grid point
        best = min(self.iono_grid,
                   key=lambda g: (g.lat_deg - lat_deg) ** 2 +
                                 (g.lon_deg - lon_deg) ** 2)
        sigma = GIVE_SIGMA.get(best.give_index, float("inf"))
        return {"give_index": best.give_index, "give_sigma": sigma}


# ---------------------------------------------------------------------------
# QZSS Augmentation
# ---------------------------------------------------------------------------

class QZSSAugmentation:
    """QZSS augmentation service handler.

    Supports CLAS (Centimeter Level Augmentation Service) corrections
    and QZSS-specific satellite handling.
    """

    def __init__(self):
        self.clas_corrections: dict[int, dict] = {}

    @staticmethod
    def is_qzss_prn(prn: int) -> bool:
        """Check whether a PRN belongs to QZSS (193-202).

        Parameters
        ----------
        prn : int
            Satellite PRN number.

        Returns
        -------
        bool
        """
        return QZSS_PRN_MIN <= prn <= QZSS_PRN_MAX

    def set_clas_correction(self, prn: int, *,
                            code_bias: float = 0.0,
                            phase_bias: float = 0.0,
                            orbit_correction: np.ndarray | None = None,
                            clock_correction: float = 0.0):
        """Store CLAS correction data for a satellite.

        Parameters
        ----------
        prn : int
            Satellite PRN.
        code_bias : float
            Code bias correction [m].
        phase_bias : float
            Phase bias correction [m].
        orbit_correction : array_like, shape (3,), optional
            Orbit correction in along/cross/radial [m].
        clock_correction : float
            Clock correction [m].
        """
        self.clas_corrections[prn] = {
            "code_bias": code_bias,
            "phase_bias": phase_bias,
            "orbit_correction": (np.asarray(orbit_correction, dtype=np.float64)
                                 if orbit_correction is not None
                                 else np.zeros(3)),
            "clock_correction": clock_correction,
        }

    def apply_clas(self, pseudoranges: np.ndarray,
                   carrier_phases: np.ndarray,
                   prn_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Apply QZSS CLAS corrections for centimeter-level positioning.

        Parameters
        ----------
        pseudoranges : array_like, shape (n_sat,)
            Pseudorange measurements [m].
        carrier_phases : array_like, shape (n_sat,)
            Carrier-phase measurements [m].
        prn_list : list of int
            PRN numbers corresponding to each measurement.

        Returns
        -------
        corrected_pr : numpy.ndarray, shape (n_sat,)
            Corrected pseudoranges [m].
        corrected_cp : numpy.ndarray, shape (n_sat,)
            Corrected carrier phases [m].
        """
        pr = np.array(pseudoranges, dtype=np.float64)
        cp = np.array(carrier_phases, dtype=np.float64)

        for i, prn in enumerate(prn_list):
            if prn in self.clas_corrections:
                corr = self.clas_corrections[prn]
                pr[i] += corr["code_bias"] + corr["clock_correction"]
                cp[i] += corr["phase_bias"] + corr["clock_correction"]

        return pr, cp

    def available_corrections(self) -> list[int]:
        """Return list of PRNs with available CLAS corrections."""
        return sorted(self.clas_corrections.keys())


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _iono_pierce_point(lat: float, lon: float,
                       el: float, az: float) -> tuple[float, float]:
    """Compute ionospheric pierce point (IPP) at single-layer height.

    Parameters
    ----------
    lat, lon : float
        Receiver position [rad].
    el, az : float
        Satellite elevation and azimuth [rad].

    Returns
    -------
    ipp_lat, ipp_lon : float
        Pierce-point latitude and longitude [rad].
    """
    psi = math.pi / 2.0 - el - math.asin(
        EARTH_RADIUS / (EARTH_RADIUS + IONO_HEIGHT) * math.cos(el))

    ipp_lat = math.asin(
        math.sin(lat) * math.cos(psi) +
        math.cos(lat) * math.sin(psi) * math.cos(az))

    ipp_lon = lon + math.asin(
        math.sin(psi) * math.sin(az) / math.cos(ipp_lat))

    return ipp_lat, ipp_lon


def _obliquity_factor(el: float) -> float:
    """Ionospheric obliquity (slant) factor.

    Parameters
    ----------
    el : float
        Satellite elevation angle [rad].

    Returns
    -------
    float
        Obliquity mapping factor (>= 1.0).
    """
    sin_arg = EARTH_RADIUS / (EARTH_RADIUS + IONO_HEIGHT) * math.cos(el)
    return 1.0 / math.sqrt(1.0 - sin_arg ** 2)


def _bilinear_interpolate(grid: list[IonoGridPoint],
                          lat_deg: float, lon_deg: float) -> float:
    """Bilinear interpolation of vertical ionospheric delay on the SBAS grid.

    Finds the four grid points forming a cell around (lat_deg, lon_deg)
    and performs standard bilinear interpolation.  If no enclosing cell
    is found, falls back to inverse-distance weighting of the four
    nearest points.

    Parameters
    ----------
    grid : list of IonoGridPoint
        Available grid points.
    lat_deg, lon_deg : float
        Query point in degrees.

    Returns
    -------
    float
        Interpolated vertical ionospheric delay [m].
    """
    # Unique sorted latitudes and longitudes in the grid
    lats = sorted(set(g.lat_deg for g in grid))
    lons = sorted(set(g.lon_deg for g in grid))

    # Find bounding lat/lon
    lat_lo = lat_hi = None
    for i in range(len(lats) - 1):
        if lats[i] <= lat_deg <= lats[i + 1]:
            lat_lo, lat_hi = lats[i], lats[i + 1]
            break

    lon_lo = lon_hi = None
    for i in range(len(lons) - 1):
        if lons[i] <= lon_deg <= lons[i + 1]:
            lon_lo, lon_hi = lons[i], lons[i + 1]
            break

    # Build lookup
    grid_map = {(g.lat_deg, g.lon_deg): g.vertical_delay for g in grid}

    if (lat_lo is not None and lon_lo is not None and
            (lat_lo, lon_lo) in grid_map and (lat_lo, lon_hi) in grid_map and
            (lat_hi, lon_lo) in grid_map and (lat_hi, lon_hi) in grid_map):
        # Standard bilinear interpolation
        dlat = lat_hi - lat_lo
        dlon = lon_hi - lon_lo
        t = (lat_deg - lat_lo) / dlat if dlat > 0 else 0.0
        u = (lon_deg - lon_lo) / dlon if dlon > 0 else 0.0

        v00 = grid_map[(lat_lo, lon_lo)]
        v01 = grid_map[(lat_lo, lon_hi)]
        v10 = grid_map[(lat_hi, lon_lo)]
        v11 = grid_map[(lat_hi, lon_hi)]

        return (v00 * (1 - t) * (1 - u) +
                v01 * (1 - t) * u +
                v10 * t * (1 - u) +
                v11 * t * u)

    # Fallback: inverse-distance weighting of four nearest points
    dists = []
    for g in grid:
        d = math.sqrt((g.lat_deg - lat_deg) ** 2 +
                      (g.lon_deg - lon_deg) ** 2)
        dists.append((d, g.vertical_delay))
    dists.sort()
    nearest = dists[:4]

    if nearest[0][0] < 1e-12:
        return nearest[0][1]

    weights = [1.0 / d for d, _ in nearest]
    total_w = sum(weights)
    return sum(w * v / total_w for w, (_, v) in zip(weights, nearest))
