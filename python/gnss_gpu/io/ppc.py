"""PPC-Dataset loader.

Supports the taroz/PPC-Dataset layout:
  https://github.com/taroz/PPC-Dataset

Each run directory contains:
- rover.obs      : rover RINEX observation file
- base.obs       : reference-station RINEX observation file
- base.nav       : broadcast navigation file
- reference.csv  : ground-truth trajectory and attitude
- imu.csv        : synchronized IMU samples
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gnss_gpu.io.nav_rinex import _datetime_to_gps_seconds_of_week, read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs


C_LIGHT = 299_792_458.0

_TIME_ALIASES = ("GPS TOW (s)", "GPS TOW", "gps_tow", "GPS_time", "time")
_X_ALIASES = ("ECEF X (m)", "ECEF X", "ECEF_X", "ecef_x", "x", "X")
_Y_ALIASES = ("ECEF Y (m)", "ECEF Y", "ECEF_Y", "ecef_y", "y", "Y")
_Z_ALIASES = ("ECEF Z (m)", "ECEF Z", "ECEF_Z", "ecef_z", "z", "Z")
_LAT_ALIASES = ("Latitude (deg)", "Latitude", "latitude", "lat", "Lat")
_LON_ALIASES = ("Longitude (deg)", "Longitude", "longitude", "lon", "Lon")
_ALT_ALIASES = (
    "Ellipsoid Height (m)",
    "Ellipsoid Height",
    "altitude",
    "Altitude",
    "height",
    "Height",
    "alt",
)

_IMU_ALIASES = {
    "time": _TIME_ALIASES,
    "acc_x": ("Acc X (m/s^2)", "Acc X", "acc_x"),
    "acc_y": ("Acc Y (m/s^2)", "Acc Y", "acc_y"),
    "acc_z": ("Acc Z (m/s^2)", "Acc Z", "acc_z"),
    "gyro_x": ("Ang Rate X (deg/s)", "Ang Rate X", "gyro_x"),
    "gyro_y": ("Ang Rate Y (deg/s)", "Ang Rate Y", "gyro_y"),
    "gyro_z": ("Ang Rate Z (deg/s)", "Ang Rate Z", "gyro_z"),
}

_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.694379990141316e-3
_SYSTEM_ID_MAP = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4}


def _pick_col(header: list[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in header:
            return alias
    return None


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _llh_to_ecef(lat_deg: float, lon_deg: float, alt: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + alt) * np.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def _nearest_index(sorted_times: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(sorted_times, t))
    if idx <= 0:
        return 0
    if idx >= len(sorted_times):
        return len(sorted_times) - 1
    prev_idx = idx - 1
    return idx if abs(sorted_times[idx] - t) < abs(sorted_times[prev_idx] - t) else prev_idx


class PPCDatasetLoader:
    """Load a single PPC-Dataset run directory."""

    REQUIRED_FILES = ("rover.obs", "base.obs", "base.nav", "reference.csv")

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

    @classmethod
    def is_run_directory(cls, path: str | Path) -> bool:
        p = Path(path)
        return p.is_dir() and all((p / name).exists() for name in cls.REQUIRED_FILES)

    def load_ground_truth(
        self,
        filepath: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load PPC reference trajectory as (times, ecef)."""
        path = Path(filepath) if filepath else self.data_dir / "reference.csv"
        if not path.exists():
            raise FileNotFoundError(f"reference.csv not found: {path}")

        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, skipinitialspace=True))

        if not rows:
            return np.array([]), np.empty((0, 3))

        header = list(rows[0].keys())
        time_col = _pick_col(header, _TIME_ALIASES)
        x_col = _pick_col(header, _X_ALIASES)
        y_col = _pick_col(header, _Y_ALIASES)
        z_col = _pick_col(header, _Z_ALIASES)
        lat_col = _pick_col(header, _LAT_ALIASES)
        lon_col = _pick_col(header, _LON_ALIASES)
        alt_col = _pick_col(header, _ALT_ALIASES)

        use_ecef = x_col is not None and y_col is not None and z_col is not None
        use_llh = lat_col is not None and lon_col is not None
        if not use_ecef and not use_llh:
            raise ValueError("reference.csv has neither ECEF nor geodetic columns")

        times = np.empty(len(rows), dtype=np.float64)
        ecef = np.empty((len(rows), 3), dtype=np.float64)

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

    def load_imu(self, filepath: str | Path | None = None) -> dict[str, np.ndarray]:
        """Load synchronized IMU samples from imu.csv."""
        path = Path(filepath) if filepath else self.data_dir / "imu.csv"
        if not path.exists():
            raise FileNotFoundError(f"imu.csv not found: {path}")

        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, skipinitialspace=True))

        if not rows:
            return {key: np.array([]) for key in _IMU_ALIASES}

        header = list(rows[0].keys())
        cols = {key: _pick_col(header, aliases) for key, aliases in _IMU_ALIASES.items()}

        data: dict[str, np.ndarray] = {}
        for key, col in cols.items():
            if col is None:
                data[key] = np.full(len(rows), np.nan)
            else:
                data[key] = np.array([_safe_float(row[col]) for row in rows], dtype=np.float64)
        return data

    def load_experiment_data(
        self,
        max_epochs: int | None = None,
        start_epoch: int = 0,
        obs_code: str = "C1C",
        snr_code: str = "S1C",
        systems: tuple[str, ...] = ("G",),
        time_tolerance: float = 0.15,
    ) -> dict:
        """Build experiment-ready arrays from a PPC run."""
        from gnss_gpu.ephemeris import Ephemeris

        rover_obs = read_rinex_obs(self.data_dir / "rover.obs")
        base_obs = read_rinex_obs(self.data_dir / "base.obs")
        nav_messages = read_nav_rinex_multi(self.data_dir / "base.nav", systems=systems)
        eph = Ephemeris(nav_messages)

        gt_times, gt_ecef = self.load_ground_truth()
        if len(gt_times) == 0:
            raise ValueError("reference.csv is empty")

        sat_ecef_list: list[np.ndarray] = []
        pseudorange_list: list[np.ndarray] = []
        weight_list: list[np.ndarray] = []
        truth_list: list[np.ndarray] = []
        time_list: list[float] = []
        sat_id_list_per_epoch: list[list[str]] = []
        system_id_list: list[np.ndarray] = []
        usable_epoch_index = 0

        for epoch in rover_obs.epochs:
            tow = _datetime_to_gps_seconds_of_week(epoch.time)

            sat_id_list: list[str] = []
            pseudoranges: list[float] = []
            snr_vals: list[float] = []

            for sat_id in epoch.satellites:
                if not sat_id or sat_id[0] not in systems:
                    continue
                obs = epoch.observations.get(sat_id, {})
                pr = obs.get(obs_code, 0.0)
                if not pr or pr < 1e6:
                    continue
                sat_id_list.append(sat_id)
                pseudoranges.append(float(pr))
                snr = float(obs.get(snr_code, 1.0))
                snr_vals.append(snr if np.isfinite(snr) and snr > 0.0 else 1.0)

            if len(sat_id_list) < 4:
                continue

            sat_ecef, sat_clk, used_sat_ids = eph.compute(
                tow,
                sat_id_list,
                obs_codes=[obs_code] * len(sat_id_list),
            )
            if len(used_sat_ids) < 4:
                continue

            pr_map = {sat_id: pr for sat_id, pr in zip(sat_id_list, pseudoranges)}
            snr_map = {sat_id: snr for sat_id, snr in zip(sat_id_list, snr_vals)}

            pr_corr = np.array(
                [pr_map[sat_id] + sat_clk[i] * C_LIGHT for i, sat_id in enumerate(used_sat_ids)],
                dtype=np.float64,
            )
            weights = np.array([max(snr_map[sat_id], 1.0) for sat_id in used_sat_ids], dtype=np.float64)
            system_ids = np.array([_SYSTEM_ID_MAP[sat_id[0]] for sat_id in used_sat_ids], dtype=np.int32)

            gt_idx = _nearest_index(gt_times, tow)
            if abs(gt_times[gt_idx] - tow) > time_tolerance:
                continue

            if usable_epoch_index < start_epoch:
                usable_epoch_index += 1
                continue

            sat_ecef_list.append(np.asarray(sat_ecef, dtype=np.float64))
            pseudorange_list.append(pr_corr)
            weight_list.append(weights)
            truth_list.append(gt_ecef[gt_idx].astype(np.float64))
            time_list.append(float(tow))
            sat_id_list_per_epoch.append(list(used_sat_ids))
            system_id_list.append(system_ids)
            usable_epoch_index += 1

            if max_epochs is not None and len(time_list) >= max_epochs:
                break

        if not time_list:
            raise ValueError("No usable PPC epochs found")

        times = np.array(time_list, dtype=np.float64)
        ground_truth = np.vstack(truth_list)
        sat_counts = np.array([len(sats) for sats in sat_id_list_per_epoch], dtype=np.int32)
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.2

        return {
            "dataset_name": f"PPC {self.data_dir.parent.name}/{self.data_dir.name}",
            "sat_ecef": sat_ecef_list,
            "pseudoranges": pseudorange_list,
            "weights": weight_list,
            "system_ids": system_id_list,
            "ground_truth": ground_truth,
            "times": times,
            "origin_ecef": ground_truth[0].copy(),
            "base_ecef": np.asarray(base_obs.header.approx_position, dtype=np.float64),
            "n_epochs": len(times),
            "n_satellites": int(np.median(sat_counts)),
            "satellite_counts": sat_counts,
            "dt": dt,
            "used_prns": sat_id_list_per_epoch,
            "constellations": tuple(sorted({sat_id[0] for sats in sat_id_list_per_epoch for sat_id in sats})),
        }
