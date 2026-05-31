"""Fast PPC RINEX window geometry loader for narrow diagnostics."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu.ephemeris import Ephemeris  # noqa: E402
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi  # noqa: E402
from gnss_gpu.io.ppc import (  # noqa: E402
    C_LIGHT,
    _PSEUDORANGE_CODE_PREFERENCES,
    _SNR_CODE_PREFERENCES,
    _SYSTEM_ID_MAP,
    _compute_at_transmit_time,
    _nearest_index,
    _pick_col,
    _pick_obs_value,
    _safe_float,
    _valid_nav_obs_mask,
)


_TIME_ALIASES = ("GPS TOW (s)", "GPS TOW", "gps_tow", "GPS_time", "time")
_X_ALIASES = ("ECEF X (m)", "ECEF X", "ECEF_X", "ecef_x", "x", "X")
_Y_ALIASES = ("ECEF Y (m)", "ECEF Y", "ECEF_Y", "ecef_y", "y", "Y")
_Z_ALIASES = ("ECEF Z (m)", "ECEF Z", "ECEF_Z", "ecef_z", "z", "Z")


def _datetime_to_tow(epoch_time: datetime) -> float:
    dow = epoch_time.weekday()
    gps_dow = (dow + 1) % 7
    sod = (
        epoch_time.hour * 3600
        + epoch_time.minute * 60
        + epoch_time.second
        + epoch_time.microsecond * 1e-6
    )
    return gps_dow * 86400.0 + sod


def _normalize_sat_id(sat_id: str) -> str:
    sat_id = sat_id.strip()
    if not sat_id:
        return sat_id
    sys_char = sat_id[0]
    prn_str = sat_id[1:].strip()
    if not prn_str:
        return sat_id
    try:
        return f"{sys_char}{int(prn_str):02d}"
    except ValueError:
        return sat_id


def _looks_like_sat_id(text: str) -> bool:
    text = text.strip()
    return len(text) >= 2 and text[0].isalpha() and text[1:].strip().isdigit()


def _read_obs_header(lines: list[str]) -> tuple[dict[str, list[str]], np.ndarray, int]:
    obs_types: dict[str, list[str]] = {}
    approx_position = np.zeros(3, dtype=np.float64)
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        label = line[60:].strip() if len(line) > 60 else ""
        if label.startswith("SYS / # / OBS TYPES"):
            sys_char = line[0].strip()
            n_types = int(line[3:6])
            obs_list = line[7:60].split()
            while len(obs_list) < n_types:
                idx += 1
                obs_list.extend(lines[idx][7:60].split())
            obs_types[sys_char] = obs_list[:n_types]
        elif label == "APPROX POSITION XYZ":
            vals = line[:60].split()
            if len(vals) >= 3:
                approx_position = np.asarray([float(v) for v in vals[:3]], dtype=np.float64)
        elif label == "END OF HEADER":
            return obs_types, approx_position, idx + 1
        idx += 1
    raise ValueError("RINEX header END OF HEADER not found")


def _parse_rover_window_epochs(
    rover_obs_path: Path,
    *,
    systems: tuple[str, ...],
    start_tow: float,
    end_tow: float,
) -> tuple[np.ndarray, list[dict[str, dict[str, float]]], dict[str, list[str]], np.ndarray]:
    with rover_obs_path.open() as f:
        lines = f.readlines()
    obs_types, approx_position, idx = _read_obs_header(lines)
    times: list[float] = []
    observations: list[dict[str, dict[str, float]]] = []
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith(">"):
            idx += 1
            continue
        parts = line[2:].split()
        if len(parts) < 8:
            idx += 1
            continue
        try:
            sec = float(parts[5])
            sec_int = int(sec)
            usec = int(round((sec - sec_int) * 1e6))
            epoch_time = datetime(
                int(parts[0]),
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                sec_int,
                usec,
            )
            epoch_flag = int(parts[6])
            n_sat = int(parts[7])
        except (ValueError, IndexError):
            idx += 1
            continue
        tow = round(_datetime_to_tow(epoch_time), 1)
        if tow > end_tow:
            break
        idx += 1
        if epoch_flag > 1:
            idx += n_sat
            continue
        if tow < start_tow:
            idx += n_sat
            continue

        epoch_obs: dict[str, dict[str, float]] = {}
        for _ in range(n_sat):
            if idx >= len(lines):
                break
            obs_line = lines[idx]
            sat_id = _normalize_sat_id(obs_line[:3])
            sys_char = sat_id[0] if sat_id else ""
            obs_codes = obs_types.get(sys_char, [])
            obs_record = obs_line.rstrip("\n")
            target_len = 3 + 16 * len(obs_codes)
            while len(obs_record) < target_len and idx + 1 < len(lines):
                next_line = lines[idx + 1]
                next_id = next_line[:3].strip()
                if next_line.startswith(">") or _looks_like_sat_id(next_id):
                    break
                idx += 1
                obs_record += lines[idx][3:].rstrip("\n")
            idx += 1

            if not sat_id or sys_char not in systems:
                continue
            sat_obs: dict[str, float] = {}
            pos = 3
            for obs_code in obs_codes:
                val_str = obs_record[pos : pos + 14].strip() if pos + 14 <= len(obs_record) else ""
                try:
                    sat_obs[obs_code] = float(val_str) if val_str else 0.0
                except ValueError:
                    sat_obs[obs_code] = 0.0
                pos += 16
            epoch_obs[sat_id] = sat_obs
        times.append(tow)
        observations.append(epoch_obs)
    return np.asarray(times, dtype=np.float64), observations, obs_types, approx_position


def _load_ground_truth_ecef(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    if not rows:
        return np.array([], dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    header = list(rows[0].keys())
    time_col = _pick_col(header, _TIME_ALIASES)
    x_col = _pick_col(header, _X_ALIASES)
    y_col = _pick_col(header, _Y_ALIASES)
    z_col = _pick_col(header, _Z_ALIASES)
    if time_col is None or x_col is None or y_col is None or z_col is None:
        raise ValueError("reference.csv must contain TOW and ECEF columns")
    times = np.empty(len(rows), dtype=np.float64)
    ecef = np.empty((len(rows), 3), dtype=np.float64)
    for i, row in enumerate(rows):
        times[i] = _safe_float(row[time_col])
        ecef[i, 0] = _safe_float(row[x_col])
        ecef[i, 1] = _safe_float(row[y_col])
        ecef[i, 2] = _safe_float(row[z_col])
    return times, ecef


def _read_base_position(base_obs_path: Path) -> np.ndarray:
    with base_obs_path.open() as f:
        lines = []
        for line in f:
            lines.append(line)
            if (line[60:].strip() if len(line) > 60 else "") == "END OF HEADER":
                break
    _obs_types, approx_position, _idx = _read_obs_header(lines)
    return approx_position


def load_ppc_window_geometry(
    run_dir: Path,
    *,
    start_tow: float,
    end_tow: float,
    systems: tuple[str, ...] = ("G",),
    transmit_time_iterations: int = 2,
    time_tolerance: float = 0.15,
) -> dict:
    """Load PPC geometry for a TOW window without parsing all rover epochs."""

    run_dir = Path(run_dir)
    rover_times, rover_epochs, _obs_types, _rover_approx = _parse_rover_window_epochs(
        run_dir / "rover.obs",
        systems=systems,
        start_tow=float(start_tow),
        end_tow=float(end_tow),
    )
    nav_messages = read_nav_rinex_multi(run_dir / "base.nav", systems=systems)
    eph = Ephemeris(nav_messages)
    gt_times, gt_ecef = _load_ground_truth_ecef(run_dir / "reference.csv")
    base_ecef = _read_base_position(run_dir / "base.obs")

    sat_ecef_list: list[np.ndarray] = []
    pseudorange_list: list[np.ndarray] = []
    weight_list: list[np.ndarray] = []
    truth_list: list[np.ndarray] = []
    time_list: list[float] = []
    sat_id_list_per_epoch: list[list[str]] = []
    system_id_list: list[np.ndarray] = []

    for tow, epoch_obs in zip(rover_times, rover_epochs):
        sat_ids: list[str] = []
        pseudoranges: list[float] = []
        snr_vals: list[float] = []
        pseudorange_codes: list[str] = []
        for sat_id, obs in epoch_obs.items():
            sys_char = sat_id[0]
            pr, pr_code = _pick_obs_value(
                sys_char,
                obs,
                "C1C",
                _PSEUDORANGE_CODE_PREFERENCES,
                "C1",
                min_abs=1e6,
            )
            if not np.isfinite(pr) or pr < 1e6:
                continue
            sat_ids.append(sat_id)
            pseudoranges.append(float(pr))
            pseudorange_codes.append(pr_code)
            snr, _snr_code = _pick_obs_value(
                sys_char,
                obs,
                "S1C",
                _SNR_CODE_PREFERENCES,
                "S1",
                min_abs=0.0,
            )
            snr_vals.append(snr if np.isfinite(snr) and snr > 0.0 else 1.0)
        if len(sat_ids) < 4:
            continue

        if int(transmit_time_iterations) > 0:
            sat_ecef, sat_clk, used_sat_ids = _compute_at_transmit_time(
                eph,
                float(tow),
                sat_ids,
                pseudorange_codes,
                pseudoranges,
                int(transmit_time_iterations),
            )
        else:
            sat_ecef, sat_clk, used_sat_ids = eph.compute(
                float(tow),
                sat_ids,
                obs_codes=pseudorange_codes,
            )
        if len(used_sat_ids) < 4:
            continue

        pr_map = {sat_id: pr for sat_id, pr in zip(sat_ids, pseudoranges)}
        snr_map = {sat_id: snr for sat_id, snr in zip(sat_ids, snr_vals)}
        pr_corr = np.array(
            [pr_map[sat_id] + sat_clk[i] * C_LIGHT for i, sat_id in enumerate(used_sat_ids)],
            dtype=np.float64,
        )
        weights = np.array([max(snr_map[sat_id], 1.0) for sat_id in used_sat_ids], dtype=np.float64)
        system_ids = np.array([_SYSTEM_ID_MAP[sat_id[0]] for sat_id in used_sat_ids], dtype=np.int32)
        sat_ecef = np.asarray(sat_ecef, dtype=np.float64)
        valid = _valid_nav_obs_mask(sat_ecef, pr_corr, weights)
        if int(valid.sum()) < 4:
            continue
        if not bool(np.all(valid)):
            sat_ecef = sat_ecef[valid]
            pr_corr = pr_corr[valid]
            weights = weights[valid]
            system_ids = system_ids[valid]
            used_sat_ids = [sat_id for sat_id, keep in zip(used_sat_ids, valid) if bool(keep)]

        gt_idx = _nearest_index(gt_times, float(tow))
        if abs(float(gt_times[gt_idx]) - float(tow)) > float(time_tolerance):
            continue

        sat_ecef_list.append(sat_ecef)
        pseudorange_list.append(pr_corr)
        weight_list.append(weights)
        truth_list.append(gt_ecef[gt_idx].astype(np.float64))
        time_list.append(float(tow))
        sat_id_list_per_epoch.append(list(used_sat_ids))
        system_id_list.append(system_ids)

    if not time_list:
        raise ValueError("No usable PPC window epochs found")

    times = np.asarray(time_list, dtype=np.float64)
    ground_truth = np.vstack(truth_list)
    sat_counts = np.array([len(sats) for sats in sat_id_list_per_epoch], dtype=np.int32)
    return {
        "dataset_name": f"PPC window {run_dir.parent.name}/{run_dir.name}",
        "sat_ecef": sat_ecef_list,
        "pseudoranges": pseudorange_list,
        "weights": weight_list,
        "system_ids": system_id_list,
        "ground_truth": ground_truth,
        "times": times,
        "origin_ecef": ground_truth[0].copy(),
        "base_ecef": base_ecef,
        "n_epochs": len(times),
        "n_satellites": int(np.median(sat_counts)),
        "satellite_counts": sat_counts,
        "dt": float(np.median(np.diff(times))) if len(times) > 1 else 0.2,
        "used_prns": sat_id_list_per_epoch,
        "constellations": tuple(sorted({sat_id[0] for sats in sat_id_list_per_epoch for sat_id in sats})),
    }
