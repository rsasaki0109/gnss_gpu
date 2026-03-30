"""NMEA sentence parser."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path


@dataclass
class GGAMessage:
    time: datetime
    latitude: float   # degrees (positive = N)
    longitude: float  # degrees (positive = E)
    fix_quality: int
    n_satellites: int
    hdop: float
    altitude: float   # meters above MSL
    geoid_sep: float  # meters


@dataclass
class RMCMessage:
    time: datetime
    status: str        # A=active, V=void
    latitude: float
    longitude: float
    speed_knots: float
    course: float


def _parse_lat(val: str, ns: str) -> float:
    if not val:
        return 0.0
    deg = float(val[:2])
    minutes = float(val[2:])
    lat = deg + minutes / 60.0
    return -lat if ns == "S" else lat


def _parse_lon(val: str, ew: str) -> float:
    if not val:
        return 0.0
    deg = float(val[:3])
    minutes = float(val[3:])
    lon = deg + minutes / 60.0
    return -lon if ew == "W" else lon


def _parse_time(time_str: str, date_str: str = "") -> datetime:
    h = int(time_str[0:2])
    m = int(time_str[2:4])
    s = float(time_str[4:])
    s_int = int(s)
    us = int((s - s_int) * 1e6)
    if date_str:
        day = int(date_str[0:2])
        mon = int(date_str[2:4])
        year = int(date_str[4:6]) + 2000
        return datetime(year, mon, day, h, m, s_int, us)
    return datetime(2000, 1, 1, h, m, s_int, us)


def _verify_checksum(sentence: str) -> bool:
    if "*" not in sentence:
        return True
    body, chk = sentence.split("*", 1)
    body = body.lstrip("$")
    calc = 0
    for c in body:
        calc ^= ord(c)
    try:
        return calc == int(chk[:2], 16)
    except ValueError:
        return False


def parse_nmea(filepath: str | Path) -> list[GGAMessage | RMCMessage]:
    """Parse NMEA file and return list of parsed messages."""
    filepath = Path(filepath)
    messages: list[GGAMessage | RMCMessage] = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("$"):
                continue
            if not _verify_checksum(line):
                continue

            body = line.split("*")[0].lstrip("$")
            fields = body.split(",")
            sentence_type = fields[0][-3:]  # GGA, RMC, etc.

            try:
                if sentence_type == "GGA" and len(fields) >= 15:
                    messages.append(GGAMessage(
                        time=_parse_time(fields[1]),
                        latitude=_parse_lat(fields[2], fields[3]),
                        longitude=_parse_lon(fields[4], fields[5]),
                        fix_quality=int(fields[6]) if fields[6] else 0,
                        n_satellites=int(fields[7]) if fields[7] else 0,
                        hdop=float(fields[8]) if fields[8] else 0.0,
                        altitude=float(fields[9]) if fields[9] else 0.0,
                        geoid_sep=float(fields[11]) if fields[11] else 0.0,
                    ))
                elif sentence_type == "RMC" and len(fields) >= 12:
                    messages.append(RMCMessage(
                        time=_parse_time(fields[1], fields[9]),
                        status=fields[2],
                        latitude=_parse_lat(fields[3], fields[4]),
                        longitude=_parse_lon(fields[5], fields[6]),
                        speed_knots=float(fields[7]) if fields[7] else 0.0,
                        course=float(fields[8]) if fields[8] else 0.0,
                    ))
            except (ValueError, IndexError):
                continue

    return messages
