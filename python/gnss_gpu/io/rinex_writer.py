"""RINEX 3.04 observation file writer.

This module produces observation files that mirror what
``gnss_gpu.io.rinex.read_rinex_obs`` (RINEX 3.x branch) can parse: a
"SYS / # / OBS TYPES" header block (with the standard 13-codes-per-line
continuation), ">"-prefixed epoch records, and satellite lines with
14.3f observation values padded with two blank characters for the
(unused) LLI / signal-strength slots.

Two entry points are provided:

``write_rinex_obs``
    Takes a :class:`RinexObsHeader` and an iterable of
    :class:`EpochRecord` (satellite-aligned per-epoch observation
    arrays) and writes a complete file.

``write_rinex_obs_from_arrays``
    Convenience wrapper for flat, row-per-observation arrays (as a
    scenario simulator would naturally produce): it groups rows by
    epoch time and code-maps them to C1C/L1C/D1C/S1C before delegating
    to ``write_rinex_obs``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

_CODES_PER_LINE = 13  # RINEX 3.04: max observation-type codes per header line
_DEFAULT_CODE_MAP = {
    "pseudorange_m": "C1C",
    "carrier_cycles": "L1C",
    "doppler_hz": "D1C",
    "cn0_dbhz": "S1C",
}


@dataclass
class RinexObsHeader:
    """Metadata written to the RINEX observation file header."""

    marker_name: str = ""
    receiver_type: str = ""
    antenna_type: str = ""
    approx_position_ecef: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obs_types: dict[str, list[str]] = field(default_factory=dict)
    interval_s: float = 1.0
    time_first_obs: datetime | None = None
    program: str = "gnss_gpu"
    run_by: str = ""
    comment: list[str] = field(default_factory=list)


@dataclass
class EpochRecord:
    """One observation epoch.

    ``obs`` maps an observation code (e.g. ``"C1C"``) to a numpy array
    (or plain sequence) aligned with ``sat_ids``.  A missing code for a
    given satellite, or a NaN/None value, is written as a blank field.
    """

    time: datetime
    sat_ids: list[str]
    obs: dict[str, np.ndarray]


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _fmt_value(value: object) -> str:
    """Format one observation value as RINEX3 14.3f, or 14 blanks."""
    if _is_blank(value):
        return " " * 14
    return f"{float(value):14.3f}"


def _hline(body: str, label: str) -> str:
    """Left-justify body to 60 cols and append the 20-col label."""
    if len(body) > 60:
        body = body[:60]
    return f"{body:<60}{label:<20}\n"


def _ordered_systems(obs_types: Mapping[str, list[str]]) -> list[str]:
    priority = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4, "S": 5, "I": 6}
    return sorted(obs_types.keys(), key=lambda s: (priority.get(s, 99), s))


def _obs_type_lines(sys_char: str, codes: Sequence[str]) -> list[str]:
    codes = list(codes)
    chunks = [codes[i : i + _CODES_PER_LINE] for i in range(0, len(codes), _CODES_PER_LINE)] or [[]]
    lines: list[str] = []
    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx == 0:
            prefix = f"{sys_char:<1}{'':2}{len(codes):3d} "
        else:
            prefix = " " * 7
        codes_str = "".join(f"{c:<3} " for c in chunk)
        lines.append(_hline(prefix + codes_str, "SYS / # / OBS TYPES"))
    return lines


def _write_header(f, header: RinexObsHeader, first_epoch_time: datetime | None) -> None:
    systems = _ordered_systems(header.obs_types)
    sys_char = systems[0] if len(systems) == 1 else "M"

    version_body = f"{3.04:9.2f}{'':11}{'OBSERVATION DATA':<20}{sys_char:<1}"
    f.write(_hline(version_body, "RINEX VERSION / TYPE"))

    run_by = header.run_by or header.program
    now = datetime.now(timezone.utc)
    pgm_body = f"{header.program:<20}{run_by:<20}{now.strftime('%Y%m%d %H%M%S'):<15}UTC"
    f.write(_hline(pgm_body, "PGM / RUN BY / DATE"))

    for line in header.comment:
        f.write(_hline(line, "COMMENT"))

    f.write(_hline(header.marker_name, "MARKER NAME"))

    if header.receiver_type:
        rec_body = f"{'':20}{'':20}{header.receiver_type:<20}"
        f.write(_hline(rec_body, "REC # / TYPE / VERS"))

    if header.antenna_type:
        ant_body = f"{'':20}{header.antenna_type:<20}"
        f.write(_hline(ant_body, "ANT # / TYPE"))

    pos = np.asarray(header.approx_position_ecef, dtype=float).reshape(-1)
    pos_body = "".join(f"{v:14.4f}" for v in pos[:3])
    f.write(_hline(pos_body, "APPROX POSITION XYZ"))

    for sys in systems:
        for line in _obs_type_lines(sys, header.obs_types[sys]):
            f.write(line)

    f.write(_hline(f"{header.interval_s:10.3f}", "INTERVAL"))

    t0 = header.time_first_obs or first_epoch_time
    if t0 is not None:
        sec = t0.second + t0.microsecond / 1e6
        time_body = (
            f"{t0.year:6d}{t0.month:6d}{t0.day:6d}{t0.hour:6d}{t0.minute:6d}"
            f"{sec:13.7f}{'':5}{'GPS':<3}"
        )
        f.write(_hline(time_body, "TIME OF FIRST OBS"))

    f.write(_hline("", "END OF HEADER"))


def _write_epoch(f, epoch: EpochRecord, obs_types: Mapping[str, list[str]]) -> None:
    t = epoch.time
    sec = t.second + t.microsecond / 1e6
    n_sat = len(epoch.sat_ids)
    epoch_line = (
        f">{t.year:5d}{t.month:3d}{t.day:3d}{t.hour:3d}{t.minute:3d}"
        f"{sec:11.7f}{0:3d}{n_sat:3d}\n"
    )
    f.write(epoch_line)

    for i, sat_id in enumerate(epoch.sat_ids):
        sys_char = sat_id[0] if sat_id else ""
        codes = obs_types.get(sys_char, [])
        parts = [f"{sat_id:<3}"]
        for code in codes:
            arr = epoch.obs.get(code)
            value = arr[i] if arr is not None and i < len(arr) else None
            parts.append(_fmt_value(value))
            parts.append("  ")
        f.write("".join(parts).rstrip() + "\n")


def write_rinex_obs(
    path: str | Path,
    header: RinexObsHeader,
    epochs: Iterable[EpochRecord],
) -> None:
    """Write a RINEX 3.04 observation file.

    ``header.obs_types`` must map each satellite-system character
    ("G", "R", "E", "C", "J", ...) present in ``epochs`` to the list of
    observation codes to emit for that system, e.g.
    ``{"G": ["C1C", "L1C", "D1C", "S1C"]}``.
    """
    epochs = list(epochs)
    first_time = epochs[0].time if epochs else None

    path = Path(path)
    with open(path, "w", newline="\n") as f:
        _write_header(f, header, first_time)
        for epoch in epochs:
            _write_epoch(f, epoch, header.obs_types)


def write_rinex_obs_from_arrays(
    path: str | Path,
    epoch_times: Sequence[datetime],
    sat_ids: Sequence[str],
    pseudorange_m: np.ndarray | None = None,
    carrier_cycles: np.ndarray | None = None,
    doppler_hz: np.ndarray | None = None,
    cn0_dbhz: np.ndarray | None = None,
    header: RinexObsHeader | None = None,
) -> None:
    """Write a RINEX obs file from flat, row-per-observation arrays.

    ``epoch_times[i]``, ``sat_ids[i]`` and the i-th entry of each value
    array describe one (epoch, satellite) observation.  Rows are
    grouped by epoch time (equal ``datetime`` values) preserving first
    appearance order.  Values map to observation codes C1C/L1C/D1C/S1C
    respectively; a ``None`` array simply omits that code.
    """
    epoch_times = list(epoch_times)
    sat_ids = list(sat_ids)
    n = len(sat_ids)
    if len(epoch_times) != n:
        raise ValueError("epoch_times and sat_ids must have the same length")

    raw_arrays = {
        "pseudorange_m": pseudorange_m,
        "carrier_cycles": carrier_cycles,
        "doppler_hz": doppler_hz,
        "cn0_dbhz": cn0_dbhz,
    }
    arrays: dict[str, np.ndarray] = {}
    for arg_name, code in _DEFAULT_CODE_MAP.items():
        arr = raw_arrays[arg_name]
        if arr is not None:
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if len(arr) != n:
                raise ValueError(f"{arg_name} must have length {n}, got {len(arr)}")
            arrays[code] = arr

    groups: dict[datetime, list[int]] = {}
    order: list[datetime] = []
    for i, t in enumerate(epoch_times):
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(i)

    systems = sorted({sid[0] for sid in sat_ids if sid})
    codes = list(arrays.keys())
    if header is None:
        header = RinexObsHeader(obs_types={sys: list(codes) for sys in systems})
    elif not header.obs_types:
        header.obs_types = {sys: list(codes) for sys in systems}

    epochs: list[EpochRecord] = []
    for t in order:
        idxs = groups[t]
        ep_sat_ids = [sat_ids[i] for i in idxs]
        ep_obs = {code: arr[idxs] for code, arr in arrays.items()}
        epochs.append(EpochRecord(time=t, sat_ids=ep_sat_ids, obs=ep_obs))

    write_rinex_obs(path, header, epochs)
