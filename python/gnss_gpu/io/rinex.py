"""RINEX observation file parser.

The parser covers the small observation subset used by the project: station
metadata, observation-code headers, epoch timestamps, satellite IDs, and numeric
observation values.  It supports RINEX 3.x and the RINEX 2.x files commonly
published by NOAA CORS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class RinexHeader:
    version: float = 0.0
    sat_system: str = ""
    marker_name: str = ""
    approx_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obs_types: dict[str, list[str]] = field(default_factory=dict)
    interval: float = 0.0


@dataclass
class RinexEpoch:
    time: datetime
    satellites: list[str]
    observations: dict[str, dict[str, float]]  # {sat_id: {obs_type: value}}


@dataclass
class RinexObs:
    header: RinexHeader
    epochs: list[RinexEpoch]

    def pseudoranges(self, obs_code: str = "C1C") -> tuple[list[datetime], np.ndarray, list[list[str]]]:
        """Extract pseudoranges as numpy array.

        Returns:
            times: list of epoch datetimes
            pr: array of shape [n_epoch, max_n_sat] (NaN for missing)
            sat_ids: list of satellite ID lists per epoch
        """
        times = []
        all_pr = []
        all_sats = []
        for ep in self.epochs:
            pr_vals = {}
            for sat, obs in ep.observations.items():
                if obs_code in obs and obs[obs_code] != 0.0:
                    pr_vals[sat] = obs[obs_code]
            if pr_vals:
                times.append(ep.time)
                all_sats.append(list(pr_vals.keys()))
                all_pr.append(list(pr_vals.values()))

        if not times:
            return [], np.array([]), []

        max_sat = max(len(p) for p in all_pr)
        pr_array = np.full((len(times), max_sat), np.nan)
        for i, vals in enumerate(all_pr):
            pr_array[i, : len(vals)] = vals

        return times, pr_array, all_sats


def _parse_two_digit_year(year: int) -> int:
    return 2000 + year if year < 80 else 1900 + year


def _rinex3_observation_codes(header: RinexHeader, lines: list[str], idx: int) -> int:
    line = lines[idx]
    sys_char = line[0].strip()
    if not sys_char:
        return idx
    n_types = int(line[3:6])
    obs_list = line[7:60].split()
    while len(obs_list) < n_types:
        idx += 1
        obs_list.extend(lines[idx][7:60].split())
    header.obs_types[sys_char] = obs_list[:n_types]
    return idx


def _rinex2_observation_codes(header: RinexHeader, lines: list[str], idx: int) -> int:
    line = lines[idx]
    n_types = int(line[:6])
    obs_list = line[6:60].split()
    while len(obs_list) < n_types:
        idx += 1
        obs_list.extend(lines[idx][6:60].split())
    sys_char = header.sat_system or "G"
    header.obs_types[sys_char] = obs_list[:n_types]
    header.obs_types[""] = obs_list[:n_types]
    return idx


def _parse_rinex2_satellite_ids(
    lines: list[str],
    idx: int,
    n_sat: int,
    default_system: str,
) -> tuple[list[str], int]:
    sats: list[str] = []
    line_idx = idx
    while len(sats) < n_sat and line_idx < len(lines):
        sat_field = lines[line_idx][32:68]
        for pos in range(0, len(sat_field), 3):
            sat = sat_field[pos : pos + 3].strip()
            if not sat:
                continue
            if sat[0].isdigit() or sat[0] == " ":
                if default_system:
                    sat = f"{default_system}{int(sat):02d}"
            elif len(sat) >= 2:
                try:
                    sat = f"{sat[0]}{int(sat[1:]):02d}"
                except ValueError:
                    pass
            sats.append(sat)
            if len(sats) >= n_sat:
                break
        if len(sats) < n_sat:
            line_idx += 1
    return sats, line_idx


def _parse_rinex_obs_values(
    lines: list[str],
    idx: int,
    obs_codes: list[str],
) -> tuple[dict[str, float], int]:
    n_lines = max(1, (len(obs_codes) + 4) // 5)
    obs_text = ""
    for offset in range(n_lines):
        if idx + offset >= len(lines):
            break
        obs_text += lines[idx + offset].rstrip("\n")
    sat_obs: dict[str, float] = {}
    for obs_idx, obs_code in enumerate(obs_codes):
        seg = obs_text[obs_idx * 16 : obs_idx * 16 + 16]
        val_str = seg[:14].strip()
        try:
            sat_obs[obs_code] = float(val_str) if val_str else 0.0
        except ValueError:
            sat_obs[obs_code] = 0.0
    return sat_obs, idx + n_lines


def read_rinex_obs(filepath: str | Path) -> RinexObs:
    """Parse a RINEX observation file."""
    filepath = Path(filepath)
    header = RinexHeader()
    epochs: list[RinexEpoch] = []

    with open(filepath) as f:
        lines = f.readlines()

    idx = 0
    # Parse header
    while idx < len(lines):
        line = lines[idx]
        label = line[60:].strip() if len(line) > 60 else ""

        if label == "RINEX VERSION / TYPE":
            header.version = float(line[:9])
            header.sat_system = line[40:41].strip()
        elif label == "MARKER NAME":
            header.marker_name = line[:60].strip()
        elif label == "APPROX POSITION XYZ":
            vals = line[:60].split()
            header.approx_position = np.array([float(v) for v in vals[:3]])
        elif label.startswith("SYS / # / OBS TYPES"):
            idx = _rinex3_observation_codes(header, lines, idx)
        elif label == "# / TYPES OF OBSERV":
            idx = _rinex2_observation_codes(header, lines, idx)
        elif label == "INTERVAL":
            header.interval = float(line[:10])
        elif label == "END OF HEADER":
            idx += 1
            break
        idx += 1

    is_rinex2 = header.version < 3.0

    # Parse epochs
    while idx < len(lines):
        line = lines[idx]
        if is_rinex2:
            parts = line[:32].split()
            if len(parts) < 8:
                idx += 1
                continue
            try:
                year = _parse_two_digit_year(int(parts[0]))
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                sec = float(parts[5])
                sec_int = int(sec)
                usec = int(round((sec - sec_int) * 1e6))
                if usec >= 1_000_000:
                    sec_int += 1
                    usec -= 1_000_000
                epoch_time = datetime(year, month, day, hour, minute, sec_int, usec)
                epoch_flag = int(parts[6])
                n_sat = int(parts[7])
            except (ValueError, IndexError):
                idx += 1
                continue

            sat_ids, sat_list_end_idx = _parse_rinex2_satellite_ids(
                lines,
                idx,
                n_sat,
                header.sat_system or "G",
            )
            idx = sat_list_end_idx + 1
            if epoch_flag > 1:
                obs_codes = header.obs_types.get(header.sat_system or "G", header.obs_types.get("", []))
                idx += n_sat * max(1, (len(obs_codes) + 4) // 5)
                continue

            observations: dict[str, dict[str, float]] = {}
            for sat_id in sat_ids:
                sys_char = sat_id[0] if sat_id and not sat_id[0].isdigit() else (header.sat_system or "G")
                obs_codes = header.obs_types.get(sys_char, header.obs_types.get("", []))
                sat_obs, idx = _parse_rinex_obs_values(lines, idx, obs_codes)
                observations[sat_id] = sat_obs

            epochs.append(
                RinexEpoch(time=epoch_time, satellites=sat_ids, observations=observations)
            )
            continue

        if not line.startswith(">"):
            idx += 1
            continue

        # Epoch header: > YYYY MM DD HH MM SS.SSSSSSS  flag  n_sat
        parts = line[2:].split()
        if len(parts) < 7:
            idx += 1
            continue

        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            sec = float(parts[5])
            sec_int = int(sec)
            usec = int((sec - sec_int) * 1e6)
            epoch_time = datetime(year, month, day, hour, minute, sec_int, usec)
            epoch_flag = int(parts[6])
            n_sat = int(parts[7]) if len(parts) > 7 else 0
        except (ValueError, IndexError):
            idx += 1
            continue

        if epoch_flag > 1:
            idx += 1 + n_sat
            continue

        satellites = []
        observations: dict[str, dict[str, float]] = {}

        for _ in range(n_sat):
            idx += 1
            if idx >= len(lines):
                break
            obs_line = lines[idx]
            sat_id = obs_line[:3].strip()
            satellites.append(sat_id)

            sys_char = sat_id[0] if sat_id else ""
            obs_codes = header.obs_types.get(sys_char, [])
            sat_obs: dict[str, float] = {}
            pos = 3
            for oc in obs_codes:
                val_str = obs_line[pos : pos + 14].strip() if pos + 14 <= len(obs_line) else ""
                try:
                    sat_obs[oc] = float(val_str) if val_str else 0.0
                except ValueError:
                    sat_obs[oc] = 0.0
                pos += 16  # 14 chars value + 1 LLI + 1 signal strength

            observations[sat_id] = sat_obs

        epochs.append(RinexEpoch(time=epoch_time, satellites=satellites, observations=observations))
        idx += 1

    return RinexObs(header=header, epochs=epochs)
