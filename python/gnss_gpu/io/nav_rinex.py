"""RINEX 2/3 navigation file parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class NavMessage:
    """Single satellite navigation message (broadcast ephemeris)."""

    prn: int
    toc: datetime  # time of clock
    system: str = "G"
    # Clock correction coefficients
    af0: float = 0.0
    af1: float = 0.0
    af2: float = 0.0
    # Keplerian orbital parameters
    sqrt_a: float = 0.0  # sqrt of semi-major axis [m^0.5]
    e: float = 0.0  # eccentricity
    i0: float = 0.0  # inclination at reference time [rad]
    omega0: float = 0.0  # longitude of ascending node at reference [rad]
    omega: float = 0.0  # argument of perigee [rad]
    M0: float = 0.0  # mean anomaly at reference [rad]
    delta_n: float = 0.0  # mean motion correction [rad/s]
    omega_dot: float = 0.0  # rate of right ascension [rad/s]
    idot: float = 0.0  # rate of inclination [rad/s]
    # Harmonic correction terms
    cuc: float = 0.0  # argument of latitude cosine correction [rad]
    cus: float = 0.0  # argument of latitude sine correction [rad]
    crc: float = 0.0  # orbit radius cosine correction [m]
    crs: float = 0.0  # orbit radius sine correction [m]
    cic: float = 0.0  # inclination cosine correction [rad]
    cis: float = 0.0  # inclination sine correction [rad]
    # Reference times
    toe: float = 0.0  # time of ephemeris (GPS seconds of week)
    week: int = 0  # GPS week number
    # Group delay
    tgd: float = 0.0  # group delay [s]
    data_sources: float = 0.0  # Galileo I/NAV/F/NAV data-source bit mask
    bgd_e5a_e1: float = 0.0  # Galileo BGD(E5a/E1) [s]
    bgd_e5b_e1: float = 0.0  # Galileo BGD(E5b/E1) [s]
    # Additional fields
    iode: float = 0.0  # issue of data (ephemeris)
    iodc: float = 0.0  # issue of data (clock)
    codes_on_l2: float = 0.0
    l2_p_flag: float = 0.0
    sv_accuracy: float = 0.0
    sv_health: float = 0.0
    fit_interval: float = 0.0
    toc_seconds: float = 0.0  # toc as GPS seconds of week

    @property
    def sat_id(self) -> str:
        """RINEX-style satellite identifier, e.g. ``G05``."""
        return f"{self.system}{self.prn:02d}"


def _parse_nav_float(s: str) -> float:
    """Parse a RINEX navigation float value (handles D exponent notation)."""
    s = s.strip()
    if not s:
        return 0.0
    s = s.replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _datetime_to_gps_seconds_of_week(dt: datetime) -> float:
    """Convert datetime to GPS seconds of week."""
    # GPS epoch: January 6, 1980
    gps_epoch = datetime(1980, 1, 6)
    delta = dt - gps_epoch
    total_seconds = delta.total_seconds()
    seconds_of_week = total_seconds % 604800.0
    return seconds_of_week


def _datetime_to_gps_week(dt: datetime) -> int:
    """Convert datetime to GPS week number."""
    gps_epoch = datetime(1980, 1, 6)
    delta = dt - gps_epoch
    total_seconds = delta.total_seconds()
    return int(total_seconds // 604800)


def read_gps_klobuchar_from_nav_header(filepath: str | Path) -> tuple[list[float] | None, list[float] | None]:
    """Read GPS Klobuchar α/β from a mixed RINEX 3 navigation header (``GPSA`` / ``GPSB``).

    Returns ``(None, None)`` if those lines are missing; callers should fall back to defaults.
    """
    filepath = Path(filepath)
    alpha: list[float] | None = None
    beta: list[float] | None = None
    with open(filepath) as f:
        for raw in f:
            if "END OF HEADER" in raw:
                break
            label = raw[60:].strip() if len(raw) > 60 else ""
            parts = raw.split()
            if len(parts) < 5:
                continue
            if parts[0] == "GPSA" and "IONOSPHERIC" in label:
                alpha = [_parse_nav_float(parts[i]) for i in range(1, 5)]
            elif parts[0] == "GPSB" and "IONOSPHERIC" in label:
                beta = [_parse_nav_float(parts[i]) for i in range(1, 5)]
    return alpha, beta


def read_nav_rinex(
    filepath: str | Path,
    systems: tuple[str, ...] = ("G",),
    key_by_sat_id: bool = False,
) -> dict[int | str, list[NavMessage]]:
    """Parse RINEX 2/3 navigation file.

    Args:
        filepath: path to the RINEX navigation file (.nav, .n, etc.)
        systems: constellation identifiers to keep in RINEX 3 mixed-nav files.
        key_by_sat_id: when True, key the result by strings such as ``G05``.

    Returns:
        dict mapping PRN number or sat-id string to list of NavMessage objects,
        sorted by toc.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    idx = 0
    version = 0.0

    # Parse header
    while idx < len(lines):
        line = lines[idx]
        # Check for header labels - use flexible matching since column alignment
        # may vary across RINEX generators
        line_stripped = line.rstrip()
        if "RINEX VERSION / TYPE" in line_stripped:
            # Version is in the first 20 columns (standard) or wherever the number is
            version_str = line[:20].strip().split()[0] if line[:20].strip() else ""
            if version_str:
                try:
                    version = float(version_str)
                except ValueError:
                    pass
        elif "END OF HEADER" in line_stripped:
            idx += 1
            break
        idx += 1

    is_v3 = version >= 3.0
    nav_messages: dict[int | str, list[NavMessage]] = {}
    systems_set = {system.upper() for system in systems}

    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue

        try:
            if is_v3:
                nav, consumed = _parse_v3_record(lines, idx, systems_set)
            else:
                nav, consumed = _parse_v2_record(lines, idx)
        except (ValueError, IndexError):
            idx += 1
            continue

        if nav is not None:
            key = nav.sat_id if key_by_sat_id else nav.prn
            if key not in nav_messages:
                nav_messages[key] = []
            nav_messages[key].append(nav)

        idx += consumed

    for key in nav_messages:
        nav_messages[key].sort(key=lambda m: m.toc)

    return nav_messages


def read_nav_rinex_multi(
    filepath: str | Path,
    systems: tuple[str, ...] = ("G", "E", "J"),
) -> dict[str, list[NavMessage]]:
    """Parse a mixed RINEX 3 navigation file keyed by sat-id strings."""
    return read_nav_rinex(filepath, systems=systems, key_by_sat_id=True)


def _v3_record_length(sys_char: str) -> int:
    if sys_char in {"R", "S"}:
        return 4
    return 8


def _parse_v3_record(
    lines: list[str], idx: int, systems_set: set[str]
) -> tuple[NavMessage | None, int]:
    """Parse a RINEX 3 navigation record (8 lines)."""
    if idx >= len(lines):
        return None, 1

    line0 = lines[idx]

    # System identifier
    sys_char = line0[0] if line0 and line0[0] != " " else "G"
    record_length = _v3_record_length(sys_char)
    if idx + record_length - 1 >= len(lines):
        return None, 1
    if sys_char not in systems_set:
        return None, record_length

    # PRN
    prn_str = line0[1:3].strip()
    if not prn_str:
        return None, record_length
    prn = int(prn_str)

    # Epoch: YYYY MM DD HH MM SS
    year = int(line0[4:8])
    month = int(line0[9:11])
    day = int(line0[12:14])
    hour = int(line0[15:17])
    minute = int(line0[18:20])
    sec = int(float(line0[21:23]))
    toc = datetime(year, month, day, hour, minute, sec)

    af0 = _parse_nav_float(line0[23:42])
    af1 = _parse_nav_float(line0[42:61])
    af2 = _parse_nav_float(line0[61:80])

    # Broadcast orbit lines (lines 1-7 from the record)
    vals = []
    for i in range(1, 8):
        ln = lines[idx + i] if idx + i < len(lines) else ""
        # Each line has 4 values at columns 4-23, 23-42, 42-61, 61-80
        for j in range(4):
            start = 4 + j * 19
            end = start + 19
            vals.append(_parse_nav_float(ln[start:end] if end <= len(ln) else ""))

    data_sources = 0.0
    bgd_e5a_e1 = 0.0
    bgd_e5b_e1 = 0.0
    tgd = vals[22]
    if sys_char == "E":
        data_sources = vals[17]
        bgd_e5a_e1 = vals[22]
        bgd_e5b_e1 = vals[23]
        # Keep the legacy field populated for callers that only know about tgd.
        tgd = bgd_e5a_e1

    nav = NavMessage(
        prn=prn,
        toc=toc,
        system=sys_char,
        af0=af0,
        af1=af1,
        af2=af2,
        # Line 1: IODE, Crs, Delta_n, M0
        iode=vals[0],
        crs=vals[1],
        delta_n=vals[2],
        M0=vals[3],
        # Line 2: Cuc, e, Cus, sqrt(A)
        cuc=vals[4],
        e=vals[5],
        cus=vals[6],
        sqrt_a=vals[7],
        # Line 3: toe, Cic, OMEGA0, Cis
        toe=vals[8],
        cic=vals[9],
        omega0=vals[10],
        cis=vals[11],
        # Line 4: i0, Crc, omega, OMEGA_DOT
        i0=vals[12],
        crc=vals[13],
        omega=vals[14],
        omega_dot=vals[15],
        # Line 5: IDOT, codes_on_l2, week, l2_p_flag
        idot=vals[16],
        codes_on_l2=vals[17],
        data_sources=data_sources,
        week=int(vals[18]),
        l2_p_flag=vals[19],
        # Line 6: SV accuracy, SV health, TGD, IODC
        sv_accuracy=vals[20],
        sv_health=vals[21],
        tgd=tgd,
        bgd_e5a_e1=bgd_e5a_e1,
        bgd_e5b_e1=bgd_e5b_e1,
        iodc=vals[23],
        # Line 7: transmission time, fit interval
        fit_interval=vals[25] if len(vals) > 25 else 0.0,
        toc_seconds=_datetime_to_gps_seconds_of_week(toc),
    )

    return nav, record_length


def _parse_v2_record(lines: list[str], idx: int) -> tuple[NavMessage | None, int]:
    """Parse a RINEX 2 navigation record (8 lines)."""
    if idx + 7 >= len(lines):
        return None, 1

    line0 = lines[idx]

    # PRN (columns 0-1)
    prn_str = line0[0:2].strip()
    if not prn_str:
        return None, 8
    try:
        prn = int(prn_str)
    except ValueError:
        return None, 8

    # Epoch: YY MM DD HH MM SS.S
    year = int(line0[3:5])
    year += 2000 if year < 80 else 1900
    month = int(line0[6:8])
    day = int(line0[9:11])
    hour = int(line0[12:14])
    minute = int(line0[15:17])
    sec = int(float(line0[17:22]))
    toc = datetime(year, month, day, hour, minute, sec)

    af0 = _parse_nav_float(line0[22:41])
    af1 = _parse_nav_float(line0[41:60])
    af2 = _parse_nav_float(line0[60:79])

    # Broadcast orbit lines (lines 1-7)
    vals = []
    for i in range(1, 8):
        ln = lines[idx + i] if idx + i < len(lines) else ""
        # RINEX 2: 3 spaces indent, then 4 values at 19-char width
        for j in range(4):
            start = 3 + j * 19
            end = start + 19
            vals.append(_parse_nav_float(ln[start:end] if end <= len(ln) else ""))

    nav = NavMessage(
        prn=prn,
        toc=toc,
        af0=af0,
        af1=af1,
        af2=af2,
        iode=vals[0],
        crs=vals[1],
        delta_n=vals[2],
        M0=vals[3],
        cuc=vals[4],
        e=vals[5],
        cus=vals[6],
        sqrt_a=vals[7],
        toe=vals[8],
        cic=vals[9],
        omega0=vals[10],
        cis=vals[11],
        i0=vals[12],
        crc=vals[13],
        omega=vals[14],
        omega_dot=vals[15],
        idot=vals[16],
        codes_on_l2=vals[17],
        week=int(vals[18]),
        l2_p_flag=vals[19],
        sv_accuracy=vals[20],
        sv_health=vals[21],
        tgd=vals[22],
        iodc=vals[23],
        fit_interval=vals[25] if len(vals) > 25 else 0.0,
        toc_seconds=_datetime_to_gps_seconds_of_week(toc),
    )

    return nav, 8
