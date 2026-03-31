"""UrbanNav dataset loader.

Supports the UrbanNavDataset format:
  https://github.com/weisongwen/UrbanNavDataset

Each run directory typically contains:
- *_GNSS.csv         : raw GNSS observations (pseudorange, carrier phase, CN0, …)
- *_groundtruth.csv  : RTK ground-truth trajectory
- *.obs / *.rnx      : RINEX 3 observation file (optional)
- *.nav / *.rnx      : RINEX navigation file (optional)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from gnss_gpu.io.rinex import read_rinex_obs, RinexObs
from gnss_gpu.io.nav_rinex import read_nav_rinex, NavMessage


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GnssObs:
    """One epoch of raw GNSS observations from the CSV file."""

    time: float                        # GPS time [s]
    prn: list[str]                     # satellite identifiers, e.g. ["G01", "G05", …]
    pseudorange: np.ndarray            # [m], shape (n_sat,)
    carrier: np.ndarray                # carrier phase [cycles], shape (n_sat,), NaN if absent
    cn0: np.ndarray                    # signal C/N0 [dB-Hz], shape (n_sat,), NaN if absent
    doppler: np.ndarray                # Doppler [Hz], shape (n_sat,), NaN if absent
    extra: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Column-name aliases used across different UrbanNav CSV variants
# ---------------------------------------------------------------------------

_TIME_ALIASES = ("GPS_time", "gps_time", "time", "timestamp", "Time", "GPSTime")
_PRN_ALIASES = ("PRN", "prn", "SatID", "sat_id", "sv", "SV")
_PR_ALIASES = ("pseudorange", "Pseudorange", "pr", "P1", "P2", "C1", "C2")
_CARRIER_ALIASES = ("carrier", "CarrierPhase", "L1", "L2", "carrier_phase")
_CN0_ALIASES = ("CN0", "cn0", "CNR", "snr", "SNR", "C/N0")
_DOPPLER_ALIASES = ("doppler", "Doppler", "D1", "D2")


def _pick_col(header: list[str], aliases: tuple[str, ...]) -> str | None:
    """Return the first matching column name from *header*, or None."""
    for alias in aliases:
        if alias in header:
            return alias
    return None


def _safe_float(val: str) -> float:
    """Convert *val* to float; return NaN on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

_WGS84_A = 6_378_137.0          # semi-major axis [m]
_WGS84_E2 = 6.694379990141316e-3  # first eccentricity squared


def _llh_to_ecef(lat_deg: float, lon_deg: float, alt: float) -> np.ndarray:
    """Convert geodetic (lat, lon [deg], alt [m]) to ECEF [m]."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + alt) * np.sin(lat)
    return np.array([x, y, z])


_GT_TIME_ALIASES = ("GPS_time", "gps_time", "time", "timestamp", "Time")
_GT_LAT_ALIASES = ("latitude", "lat", "Latitude", "Lat")
_GT_LON_ALIASES = ("longitude", "lon", "Longitude", "Lon")
_GT_ALT_ALIASES = ("altitude", "alt", "Altitude", "Alt", "height", "Height")
_GT_X_ALIASES = ("x", "X", "ECEF_X", "ecef_x", "pos_x")
_GT_Y_ALIASES = ("y", "Y", "ECEF_Y", "ecef_y", "pos_y")
_GT_Z_ALIASES = ("z", "Z", "ECEF_Z", "ecef_z", "pos_z")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class UrbanNavLoader:
    """Load a single UrbanNav run directory.

    Parameters
    ----------
    data_dir:
        Path to the directory that contains the run's CSV and/or RINEX files.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

    # ------------------------------------------------------------------
    # File discovery helpers
    # ------------------------------------------------------------------

    def _find_gnss_csv(self) -> Path | None:
        """Return the first *_GNSS.csv* file found, or any *.csv matching heuristics."""
        for pattern in ("*_GNSS.csv", "*_gnss.csv", "*GNSS*.csv"):
            matches = sorted(self.data_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _find_ground_truth_csv(self) -> Path | None:
        for pattern in ("*groundtruth*.csv", "*ground_truth*.csv", "*gt*.csv", "*GTruth*.csv"):
            matches = sorted(self.data_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _find_rinex_obs(self) -> Path | None:
        for pattern in ("*.obs", "*.OBS", "*_obs.rnx", "*.rnx"):
            matches = sorted(self.data_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _find_rinex_nav(self) -> Path | None:
        for pattern in ("*.nav", "*.NAV", "*_nav.rnx", "*.n", "*.N"):
            matches = sorted(self.data_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_gnss_csv(self, filepath: str | Path | None = None) -> dict[str, np.ndarray]:
        """Load the GNSS CSV file.

        Columns accepted (first matching alias is used):
          GPS_time, PRN, pseudorange, carrier, CN0, doppler

        Returns
        -------
        dict with keys:
          ``time``        : (N,)     GPS time per row [s]
          ``prn``         : (N,)     satellite PRN strings
          ``pseudorange`` : (N,)     pseudorange [m]
          ``carrier``     : (N,)     carrier phase [cycles] (NaN if absent)
          ``cn0``         : (N,)     C/N0 [dB-Hz] (NaN if absent)
          ``doppler``     : (N,)     Doppler [Hz] (NaN if absent)

        Any additional numeric columns are also included in the dict.
        """
        path = Path(filepath) if filepath else self._find_gnss_csv()
        if path is None:
            raise FileNotFoundError("No GNSS CSV file found in " + str(self.data_dir))

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            return {
                "time": np.array([]),
                "prn": np.array([], dtype=object),
                "pseudorange": np.array([]),
                "carrier": np.array([]),
                "cn0": np.array([]),
                "doppler": np.array([]),
            }

        header = list(rows[0].keys())

        time_col = _pick_col(header, _TIME_ALIASES)
        prn_col = _pick_col(header, _PRN_ALIASES)
        pr_col = _pick_col(header, _PR_ALIASES)
        carrier_col = _pick_col(header, _CARRIER_ALIASES)
        cn0_col = _pick_col(header, _CN0_ALIASES)
        doppler_col = _pick_col(header, _DOPPLER_ALIASES)

        known_cols = {time_col, prn_col, pr_col, carrier_col, cn0_col, doppler_col} - {None}

        n = len(rows)
        times = np.empty(n)
        prns: list[str] = []
        pseudoranges = np.full(n, np.nan)
        carriers = np.full(n, np.nan)
        cn0s = np.full(n, np.nan)
        dopplers = np.full(n, np.nan)

        for i, row in enumerate(rows):
            times[i] = _safe_float(row[time_col]) if time_col else float("nan")
            prns.append(str(row[prn_col]).strip() if prn_col else "")
            if pr_col:
                pseudoranges[i] = _safe_float(row[pr_col])
            if carrier_col:
                carriers[i] = _safe_float(row[carrier_col])
            if cn0_col:
                cn0s[i] = _safe_float(row[cn0_col])
            if doppler_col:
                dopplers[i] = _safe_float(row[doppler_col])

        result: dict[str, np.ndarray] = {
            "time": times,
            "prn": np.array(prns, dtype=object),
            "pseudorange": pseudoranges,
            "carrier": carriers,
            "cn0": cn0s,
            "doppler": dopplers,
        }

        # Include any remaining numeric columns
        for col in header:
            if col in known_cols:
                continue
            vals = np.array([_safe_float(row[col]) for row in rows])
            result[col] = vals

        return result

    def load_ground_truth(
        self,
        filepath: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load ground-truth trajectory.

        Accepts:
        - ECEF columns  (x/y/z or ECEF_X/Y/Z)
        - Geodetic columns (latitude, longitude, altitude)

        Returns
        -------
        times : ndarray, shape (N,)
            GPS time [s] per row.
        ecef : ndarray, shape (N, 3)
            ECEF positions [m].
        """
        path = Path(filepath) if filepath else self._find_ground_truth_csv()
        if path is None:
            raise FileNotFoundError("No ground-truth CSV file found in " + str(self.data_dir))

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            return np.array([]), np.empty((0, 3))

        header = list(rows[0].keys())

        time_col = _pick_col(header, _GT_TIME_ALIASES)
        x_col = _pick_col(header, _GT_X_ALIASES)
        y_col = _pick_col(header, _GT_Y_ALIASES)
        z_col = _pick_col(header, _GT_Z_ALIASES)
        lat_col = _pick_col(header, _GT_LAT_ALIASES)
        lon_col = _pick_col(header, _GT_LON_ALIASES)
        alt_col = _pick_col(header, _GT_ALT_ALIASES)

        n = len(rows)
        times = np.empty(n)
        ecef = np.empty((n, 3))

        use_ecef = (x_col is not None) and (y_col is not None) and (z_col is not None)
        use_llh = (lat_col is not None) and (lon_col is not None)

        if not use_ecef and not use_llh:
            raise ValueError(
                "Ground-truth CSV has neither ECEF (x/y/z) nor geodetic "
                "(latitude/longitude) columns."
            )

        for i, row in enumerate(rows):
            times[i] = _safe_float(row[time_col]) if time_col else float("nan")
            if use_ecef:
                ecef[i, 0] = _safe_float(row[x_col])
                ecef[i, 1] = _safe_float(row[y_col])
                ecef[i, 2] = _safe_float(row[z_col])
            else:
                lat = _safe_float(row[lat_col])
                lon = _safe_float(row[lon_col])
                alt = _safe_float(row[alt_col]) if alt_col else 0.0
                ecef[i] = _llh_to_ecef(lat, lon, alt)

        return times, ecef

    def load_rinex_obs(self, filepath: str | Path | None = None) -> RinexObs | None:
        """Load RINEX observation file if available.

        Uses :func:`gnss_gpu.io.rinex.read_rinex_obs` internally.

        Returns
        -------
        :class:`~gnss_gpu.io.rinex.RinexObs` or ``None`` if no file is found.
        """
        path = Path(filepath) if filepath else self._find_rinex_obs()
        if path is None:
            return None
        return read_rinex_obs(path)

    def load_rinex_nav(
        self, filepath: str | Path | None = None
    ) -> dict[int, list[NavMessage]] | None:
        """Load RINEX navigation file if available.

        Uses :func:`gnss_gpu.io.nav_rinex.read_nav_rinex` internally.

        Returns
        -------
        dict mapping PRN -> list of :class:`~gnss_gpu.io.nav_rinex.NavMessage`,
        or ``None`` if no file is found.
        """
        path = Path(filepath) if filepath else self._find_rinex_nav()
        if path is None:
            return None
        return read_nav_rinex(path)

    def epochs(
        self,
        gnss_csv: str | Path | None = None,
        ground_truth_csv: str | Path | None = None,
        time_tolerance: float = 0.5,
    ) -> Iterator[tuple[GnssObs, np.ndarray]]:
        """Yield synchronised (gnss_obs, truth_ecef) pairs per epoch.

        GNSS rows that share the same GPS time (within floating-point equality)
        are grouped into one :class:`GnssObs`.  Each epoch is then matched to
        the nearest ground-truth sample whose time difference is within
        *time_tolerance* seconds.

        Parameters
        ----------
        gnss_csv:
            Path override for the GNSS CSV.  Auto-discovered when *None*.
        ground_truth_csv:
            Path override for the ground-truth CSV.  Auto-discovered when *None*.
        time_tolerance:
            Maximum allowed time gap [s] between a GNSS epoch and its matched
            ground-truth sample.

        Yields
        ------
        gnss_obs : GnssObs
            All satellite observations for this epoch.
        truth_ecef : ndarray, shape (3,)
            ECEF ground-truth position [m] at the nearest matching time.
        """
        gnss_data = self.load_gnss_csv(gnss_csv)
        gt_times, gt_ecef = self.load_ground_truth(ground_truth_csv)

        times = gnss_data["time"]
        prns = gnss_data["prn"]
        pseudoranges = gnss_data["pseudorange"]
        carriers = gnss_data["carrier"]
        cn0s = gnss_data["cn0"]
        dopplers = gnss_data["doppler"]

        extra_keys = [
            k for k in gnss_data if k not in {"time", "prn", "pseudorange", "carrier", "cn0", "doppler"}
        ]

        # Group rows by epoch time
        unique_times = np.unique(times[~np.isnan(times)])

        for t in unique_times:
            mask = times == t
            idx = np.where(mask)[0]

            gnss_obs = GnssObs(
                time=float(t),
                prn=list(prns[idx]),
                pseudorange=pseudoranges[idx].copy(),
                carrier=carriers[idx].copy(),
                cn0=cn0s[idx].copy(),
                doppler=dopplers[idx].copy(),
                extra={k: gnss_data[k][idx].copy() for k in extra_keys},
            )

            # Match ground truth
            if len(gt_times) == 0:
                continue
            dt = np.abs(gt_times - t)
            best = int(np.argmin(dt))
            if dt[best] > time_tolerance:
                continue

            yield gnss_obs, gt_ecef[best]
