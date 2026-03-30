"""NMEA sentence writer."""

from __future__ import annotations

import math
from datetime import datetime, date
from pathlib import Path

import numpy as np


class NMEAWriter:
    """Generate NMEA sentences from positioning results."""

    def __init__(self, talker_id: str = "GP"):
        self.talker_id = talker_id

    def gga(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        time_utc: datetime | None = None,
        fix_quality: int = 1,
        n_sats: int = 0,
        hdop: float = 1.0,
        geoid_sep: float = 0.0,
    ) -> str:
        """Generate $GPGGA sentence.

        fix_quality: 0=invalid, 1=GPS, 2=DGPS, 4=RTK fixed, 5=RTK float
        """
        if time_utc is None:
            time_utc = datetime.utcnow()
        time_str = time_utc.strftime("%H%M%S.%f")[:10]

        lat_nmea, lat_dir = self._deg_to_nmea(lat_deg, is_lat=True)
        lon_nmea, lon_dir = self._deg_to_nmea(lon_deg, is_lat=False)

        body = (
            f"{self.talker_id}GGA,"
            f"{time_str},"
            f"{lat_nmea},{lat_dir},"
            f"{lon_nmea},{lon_dir},"
            f"{fix_quality},"
            f"{n_sats:02d},"
            f"{hdop:.1f},"
            f"{alt_m:.3f},M,"
            f"{geoid_sep:.3f},M,,"
        )
        chk = self._checksum(body)
        return f"${body}*{chk}"

    def rmc(
        self,
        lat_deg: float,
        lon_deg: float,
        time_utc: datetime | None = None,
        date_val: date | None = None,
        speed_knots: float = 0.0,
        course_deg: float = 0.0,
        status: str = "A",
    ) -> str:
        """Generate $GPRMC sentence."""
        if time_utc is None:
            time_utc = datetime.utcnow()
        if date_val is None:
            date_val = time_utc.date() if isinstance(time_utc, datetime) else date.today()

        time_str = time_utc.strftime("%H%M%S.%f")[:10]
        date_str = f"{date_val.day:02d}{date_val.month:02d}{date_val.year % 100:02d}"

        lat_nmea, lat_dir = self._deg_to_nmea(lat_deg, is_lat=True)
        lon_nmea, lon_dir = self._deg_to_nmea(lon_deg, is_lat=False)

        body = (
            f"{self.talker_id}RMC,"
            f"{time_str},"
            f"{status},"
            f"{lat_nmea},{lat_dir},"
            f"{lon_nmea},{lon_dir},"
            f"{speed_knots:.1f},"
            f"{course_deg:.1f},"
            f"{date_str},,,"
        )
        chk = self._checksum(body)
        return f"${body}*{chk}"

    def gsa(
        self,
        prn_list: list[int],
        pdop: float = 0.0,
        hdop: float = 0.0,
        vdop: float = 0.0,
        fix_type: int = 3,
        mode: str = "A",
    ) -> str:
        """Generate $GPGSA sentence."""
        prns = list(prn_list)[:12]
        prn_fields = [f"{p:02d}" for p in prns]
        # Pad to 12 fields
        while len(prn_fields) < 12:
            prn_fields.append("")

        prn_str = ",".join(prn_fields)
        body = (
            f"{self.talker_id}GSA,"
            f"{mode},{fix_type},"
            f"{prn_str},"
            f"{pdop:.1f},{hdop:.1f},{vdop:.1f}"
        )
        chk = self._checksum(body)
        return f"${body}*{chk}"

    def gsv(
        self,
        satellites: list[tuple[int, int, int, int]],
    ) -> list[str]:
        """Generate $GPGSV sentences.

        satellites: list of (prn, elevation_deg, azimuth_deg, snr_db)
        """
        n_sats = len(satellites)
        n_msgs = math.ceil(n_sats / 4) if n_sats > 0 else 1
        sentences: list[str] = []

        for msg_idx in range(n_msgs):
            start = msg_idx * 4
            chunk = satellites[start : start + 4]
            sat_fields: list[str] = []
            for prn, elev, az, snr in chunk:
                sat_fields.append(f"{prn:02d},{elev:02d},{az:03d},{snr:02d}")
            # Pad remaining slots with empty fields
            while len(sat_fields) < 4:
                sat_fields.append(",,,")

            body = (
                f"{self.talker_id}GSV,"
                f"{n_msgs},{msg_idx + 1},{n_sats:02d},"
                + ",".join(sat_fields)
            )
            chk = self._checksum(body)
            sentences.append(f"${body}*{chk}")

        return sentences

    def vtg(
        self,
        course_deg: float = 0.0,
        speed_knots: float = 0.0,
        speed_kmh: float = 0.0,
    ) -> str:
        """Generate $GPVTG sentence."""
        body = (
            f"{self.talker_id}VTG,"
            f"{course_deg:.1f},T,,M,"
            f"{speed_knots:.1f},N,"
            f"{speed_kmh:.1f},K"
        )
        chk = self._checksum(body)
        return f"${body}*{chk}"

    @staticmethod
    def _checksum(sentence: str) -> str:
        """Compute NMEA checksum (XOR of all chars between $ and *)."""
        calc = 0
        for c in sentence:
            calc ^= ord(c)
        return f"{calc:02X}"

    @staticmethod
    def _deg_to_nmea(deg: float, is_lat: bool = True) -> tuple[str, str]:
        """Convert decimal degrees to NMEA format (DDMM.MMMM or DDDMM.MMMM).

        Returns (value_string, direction_char).
        """
        if is_lat:
            direction = "N" if deg >= 0 else "S"
        else:
            direction = "E" if deg >= 0 else "W"

        deg = abs(deg)
        d = int(deg)
        minutes = (deg - d) * 60.0

        if is_lat:
            value = f"{d:02d}{minutes:07.4f}"
        else:
            value = f"{d:03d}{minutes:07.4f}"

        return value, direction

    def write_epoch(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        time_utc: datetime | None = None,
        n_sats: int = 0,
        hdop: float = 1.0,
        speed_knots: float = 0.0,
        course_deg: float = 0.0,
        prn_list: list[int] | None = None,
        pdop: float = 0.0,
        vdop: float = 0.0,
        fix_quality: int = 1,
    ) -> list[str]:
        """Write a complete epoch (GGA + RMC + GSA + VTG sentences).

        Returns list of NMEA sentence strings.
        """
        sentences: list[str] = []
        sentences.append(
            self.gga(lat_deg, lon_deg, alt_m, time_utc=time_utc,
                     fix_quality=fix_quality, n_sats=n_sats, hdop=hdop)
        )
        sentences.append(
            self.rmc(lat_deg, lon_deg, time_utc=time_utc,
                     speed_knots=speed_knots, course_deg=course_deg)
        )
        if prn_list is not None:
            sentences.append(
                self.gsa(prn_list, pdop=pdop, hdop=hdop, vdop=vdop)
            )
        speed_kmh = speed_knots * 1.852
        sentences.append(
            self.vtg(course_deg=course_deg, speed_knots=speed_knots,
                     speed_kmh=speed_kmh)
        )
        return sentences


def _ecef_to_lla_py(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Pure-Python ECEF to LLA conversion (WGS84).

    Fallback when the C++/CUDA extension is not available.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = 2.0 * f - f * f
    ep2 = (a * a - b * b) / (b * b)

    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - e2 * a * math.cos(theta) ** 3,
    )
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N

    return math.degrees(lat), math.degrees(lon), alt


def positions_to_nmea(
    lat_deg_array: np.ndarray,
    lon_deg_array: np.ndarray,
    alt_m_array: np.ndarray,
    times: list[datetime] | None = None,
    filepath: str | Path | None = None,
) -> str | None:
    """Convert arrays of positions to NMEA file.

    If filepath is given, write to file and return None.
    Otherwise return the NMEA string.
    """
    writer = NMEAWriter()
    lines: list[str] = []

    n = len(lat_deg_array)
    for i in range(n):
        t = times[i] if times is not None else None
        sentences = writer.write_epoch(
            lat_deg_array[i], lon_deg_array[i], alt_m_array[i], time_utc=t,
        )
        lines.extend(sentences)

    result = "\n".join(lines) + "\n"

    if filepath is not None:
        Path(filepath).write_text(result)
        return None
    return result


def ecef_to_nmea(
    ecef_positions: np.ndarray,
    times: list[datetime] | None = None,
    filepath: str | Path | None = None,
) -> str | None:
    """Convert ECEF positions [N, 3] to NMEA output.

    Handles ECEF to LLA conversion internally.  Uses the C++/CUDA extension
    if available, otherwise falls back to a pure-Python implementation.
    """
    ecef_positions = np.asarray(ecef_positions, dtype=np.float64)
    n = ecef_positions.shape[0]

    lat = np.empty(n)
    lon = np.empty(n)
    alt = np.empty(n)

    try:
        from gnss_gpu._gnss_gpu import ecef_to_lla as _ecef_to_lla

        _ecef_to_lla(
            ecef_positions[:, 0].copy(),
            ecef_positions[:, 1].copy(),
            ecef_positions[:, 2].copy(),
            lat, lon, alt, n,
        )
        lat = np.degrees(lat)
        lon = np.degrees(lon)
    except (ImportError, TypeError):
        for i in range(n):
            lat[i], lon[i], alt[i] = _ecef_to_lla_py(
                ecef_positions[i, 0],
                ecef_positions[i, 1],
                ecef_positions[i, 2],
            )

    return positions_to_nmea(lat, lon, alt, times=times, filepath=filepath)
