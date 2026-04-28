"""Base-station correction and MatRTKLIB-style nav helpers for GSDC2023."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate import ecef_to_lla
from experiments.gsdc2023_imu import enu_to_ecef_relative
from experiments.gsdc2023_residual_model import geometric_range_with_sagnac
from experiments.gsdc2023_signal_model import is_l5_signal
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_gps_klobuchar_from_nav_header, read_nav_rinex_multi
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.spp import C_LIGHT, _elevation_azimuth, _iono_klobuchar


GPS_L1_FREQUENCY_HZ = 1575.42e6
GPS_L5_FREQUENCY_HZ = 1176.45e6
GPS_L5_TGD_SCALE = (GPS_L1_FREQUENCY_HZ / GPS_L5_FREQUENCY_HZ) ** 2
LIGHT_SPEED_MPS = 299792458.0
GPS_WEEK_SECONDS = 604800.0
GPS_LEAP_SECONDS = 18.0
GPS_EPOCH_UNIX_SECONDS = datetime(1980, 1, 6, tzinfo=timezone.utc).timestamp()
BASE_MOVMEAN_N_1S = 151
BASE_MOVMEAN_N_15S = 11
BASE_OBS_TRIM_MARGIN_S = 180.0
GPS_NAV_PRODUCT_ADJUSTMENT_THRESHOLD_M = 0.005

BaseSettingFn = Callable[[Path, str, str, str], tuple[str, str | None]]
BaseResidualLoader = Callable[
    [str, str, str, str, str | None, str, float, float, tuple[str, ...]],
    tuple[np.ndarray, tuple[str, ...], np.ndarray],
]
PhoneTimeSpanFn = Callable[[Path, str, str, str, np.ndarray], tuple[float, float]]


def read_raw_gnss_csv(raw_path: Path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    last_error: Exception | None = None
    for compression in ("zip", None):
        try:
            return pd.read_csv(raw_path, compression=compression, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to read raw GNSS file: {raw_path}")


def trip_course_phone(trip: str | Path) -> tuple[str | None, str | None, str | None]:
    parts = Path(trip).parts
    if len(parts) >= 3 and parts[0] in {"train", "test"}:
        return parts[0], parts[1], parts[2]
    if len(parts) >= 2:
        return None, parts[-2], parts[-1]
    return None, None, None


@lru_cache(maxsize=8)
def load_settings_frame_cached(data_root_str: str, split: str) -> pd.DataFrame | None:
    settings_path = Path(data_root_str) / f"settings_{split}.csv"
    if not settings_path.is_file():
        return None
    return pd.read_csv(settings_path)


def base_metadata_dir(data_root: Path) -> Path:
    """Directory containing ``base_position.csv`` / ``base_offset.csv``."""

    sibling = data_root.parent / "base"
    if sibling.is_dir():
        return sibling
    if data_root.name == "sdc2023":
        gsdc_base = data_root.parent.parent / "base"
        if gsdc_base.is_dir():
            return gsdc_base
    return sibling


def gps_abs_seconds_from_datetime(dt: datetime) -> float:
    gps_epoch = datetime(1980, 1, 6)
    return float((dt.replace(tzinfo=None) - gps_epoch).total_seconds())


def unix_ms_to_gps_abs_seconds(times_ms: np.ndarray) -> np.ndarray:
    unix_s = np.asarray(times_ms, dtype=np.float64).reshape(-1) * 1e-3
    return unix_s + GPS_LEAP_SECONDS - GPS_EPOCH_UNIX_SECONDS


def signal_type_base_obs_codes(signal_type: str) -> list[str]:
    sig = signal_type.upper()
    if "L5" in sig or "E5" in sig:
        return ["C5Q", "C5X", "C5I", "C5", "P5"]
    if sig.startswith("GAL") or "E1" in sig:
        return ["C1C", "C1X", "C1A", "C1B", "C1Z", "C1", "P1"]
    if sig.startswith("QZS") or "QZS" in sig:
        return ["C1C", "C1X", "C1Z", "C1L", "C1S", "C1", "P1"]
    return ["C1C", "C1W", "C1P", "C1X", "C1S", "C1L", "C1", "P1"]


def signal_type_iono_scale(signal_type: str) -> float:
    freq_hz = GPS_L5_FREQUENCY_HZ if is_l5_signal(signal_type) else GPS_L1_FREQUENCY_HZ
    return float((GPS_L1_FREQUENCY_HZ / freq_hz) ** 2)


def select_base_pseudorange_observation(
    observations: dict[str, float],
    signal_type: str,
) -> tuple[str, float] | None:
    """Select a base RINEX pseudorange code matching the Android signal band."""

    def valid(value: float) -> bool:
        return np.isfinite(value) and value > 1.0e7

    candidates = signal_type_base_obs_codes(signal_type)
    for code in candidates:
        value = float(observations.get(code, 0.0) or 0.0)
        if valid(value):
            return code, value

    prefixes = []
    for code in candidates:
        if len(code) >= 2:
            prefix = code[:2]
            if prefix not in prefixes:
                prefixes.append(prefix)
    for prefix in prefixes:
        for code in sorted(observations):
            if not code.startswith(prefix):
                continue
            value = float(observations.get(code, 0.0) or 0.0)
            if valid(value):
                return code, value
    return None


def slot_sat_id(constellation_type: int, svid: int) -> str | None:
    if int(constellation_type) == 1:
        return f"G{int(svid):02d}"
    if int(constellation_type) == 4:
        prn = int(svid)
        if prn > 192:
            prn -= 192
        return f"J{prn:02d}"
    if int(constellation_type) == 6:
        return f"E{int(svid):02d}"
    return None


def course_nav_path(data_root: Path, split: str, course: str) -> Path:
    course_dir = data_root / split / course
    nav_paths = sorted(course_dir.glob("brdc.*"))
    if not nav_paths:
        raise FileNotFoundError(f"broadcast navigation file not found under {course_dir}")
    return nav_paths[0]


def trip_nav_path(trip_dir: Path) -> Path | None:
    nav_paths = sorted(Path(trip_dir).parent.glob("brdc.*"))
    return nav_paths[0] if nav_paths else None


@lru_cache(maxsize=64)
def gps_tgd_m_by_svid_cached(nav_path: str) -> dict[int, float]:
    try:
        nav = read_nav_rinex_multi(Path(nav_path), systems=("G",))
    except Exception:  # noqa: BLE001
        return {}
    out: dict[int, float] = {}
    for key, messages in nav.items():
        try:
            svid = int(str(key)[1:] if str(key).upper().startswith("G") else str(key))
        except ValueError:
            continue
        for message in messages:
            tgd = float(getattr(message, "tgd", np.nan))
            if np.isfinite(tgd):
                out[svid] = tgd * LIGHT_SPEED_MPS
                break
    return out


def gps_tgd_m_by_svid_for_trip(trip_dir: Path) -> dict[int, float]:
    nav_path = trip_nav_path(trip_dir)
    if nav_path is None:
        return {}
    return gps_tgd_m_by_svid_cached(str(nav_path.resolve()))


def filter_matrtklib_duplicate_gps_nav_messages(messages: list[object]) -> list[object]:
    """Mirror MatRTKLIB's effective handling of near-duplicate GPS eph records."""
    out: list[object] = []
    for message in sorted(messages, key=lambda item: (float(getattr(item, "toe", 0.0)), getattr(item, "toc", 0))):
        toe = float(getattr(message, "toe", np.nan))
        if out:
            prev_toe = float(getattr(out[-1], "toe", np.nan))
            if np.isfinite(toe) and np.isfinite(prev_toe) and 0.0 <= toe - prev_toe <= 60.0:
                continue
        out.append(message)
    return out


@lru_cache(maxsize=64)
def gps_matrtklib_nav_messages_cached(nav_path: str) -> dict[int, tuple[tuple[object, ...], tuple[object, ...]]]:
    try:
        nav = read_nav_rinex_multi(Path(nav_path), systems=("G",))
    except Exception:  # noqa: BLE001
        return {}
    out: dict[int, tuple[tuple[object, ...], tuple[object, ...]]] = {}
    for key, messages in nav.items():
        try:
            svid = int(str(key)[1:] if str(key).upper().startswith("G") else str(key))
        except ValueError:
            continue
        full = tuple(sorted(messages, key=lambda item: (getattr(item, "toc", None), float(getattr(item, "toe", 0.0)))))
        filtered = tuple(filter_matrtklib_duplicate_gps_nav_messages(list(messages)))
        if full:
            out[svid] = (full, filtered if filtered else full)
    return out


def gps_matrtklib_nav_messages_for_trip(trip_dir: Path) -> dict[int, tuple[tuple[object, ...], tuple[object, ...]]]:
    nav_path = trip_nav_path(trip_dir)
    if nav_path is None:
        return {}
    return gps_matrtklib_nav_messages_cached(str(nav_path.resolve()))


def select_gps_nav_message(messages: tuple[object, ...], gps_tow_s: float) -> object | None:
    best = None
    best_dt = np.inf
    for message in messages:
        toe = float(getattr(message, "toe", np.nan))
        if not np.isfinite(toe):
            continue
        dt = abs(float(gps_tow_s) - toe)
        if dt > GPS_WEEK_SECONDS / 2.0:
            dt = GPS_WEEK_SECONDS - dt
        if dt <= best_dt:
            best = message
            best_dt = dt
    return best


def gps_arrival_tow_s_from_row(row: object) -> float:
    arrival_ns = getattr(row, "ArrivalTimeNanosSinceGpsEpoch", np.nan)
    arrival_ns = float(arrival_ns)
    if not np.isfinite(arrival_ns):
        return np.nan
    return (arrival_ns * 1.0e-9) % GPS_WEEK_SECONDS


def gps_broadcast_clock_bias_s(message: object, transmit_tow_s: float) -> float:
    """Mirror RTKLIB eph2clk: broadcast clock polynomial without relativity/TGD."""

    toc_s = float(getattr(message, "toc_seconds", getattr(message, "toe", 0.0)))
    ts = float(transmit_tow_s) - toc_s
    if ts > GPS_WEEK_SECONDS / 2.0:
        ts -= GPS_WEEK_SECONDS
    if ts < -GPS_WEEK_SECONDS / 2.0:
        ts += GPS_WEEK_SECONDS

    af0 = float(getattr(message, "af0", 0.0))
    af1 = float(getattr(message, "af1", 0.0))
    af2 = float(getattr(message, "af2", 0.0))
    t = ts
    for _ in range(2):
        t = ts - (af0 + af1 * t + af2 * t * t)
    return af0 + af1 * t + af2 * t * t


def gps_matrtklib_sat_product_adjustment(
    *,
    svid: int,
    arrival_tow_s: float,
    l1_raw_pseudorange_m: float,
    derived_common_clock_m: float,
    nav_messages_by_svid: dict[int, tuple[tuple[object, ...], tuple[object, ...]]],
    receiver_clock_bias_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    if (
        not np.isfinite(arrival_tow_s)
        or not np.isfinite(l1_raw_pseudorange_m)
        or not np.isfinite(derived_common_clock_m)
    ):
        return None
    nav_pair = nav_messages_by_svid.get(int(svid))
    if nav_pair is None:
        return None
    full_messages, filtered_messages = nav_pair
    full_message = select_gps_nav_message(full_messages, arrival_tow_s)
    filtered_message = select_gps_nav_message(filtered_messages, arrival_tow_s)
    if full_message is None or filtered_message is None:
        return None

    rx_clock_bias_m = float(receiver_clock_bias_m) if np.isfinite(receiver_clock_bias_m) else 0.0
    transmit_tow_s = float(arrival_tow_s) - float(l1_raw_pseudorange_m) / LIGHT_SPEED_MPS + rx_clock_bias_m / LIGHT_SPEED_MPS
    _, full_clock_s = Ephemeris._compute_single_cpu(full_message, transmit_tow_s, "C1")
    _, filtered_clock_s = Ephemeris._compute_single_cpu(filtered_message, transmit_tow_s, "C1")
    full_common_clock_m = float(full_clock_s) * LIGHT_SPEED_MPS + float(getattr(full_message, "tgd", 0.0)) * LIGHT_SPEED_MPS
    filtered_common_clock_m = (
        float(filtered_clock_s) * LIGHT_SPEED_MPS + float(getattr(filtered_message, "tgd", 0.0)) * LIGHT_SPEED_MPS
    )

    same_selected_message = (
        int(round(float(getattr(full_message, "iode", -1.0)))) == int(round(float(getattr(filtered_message, "iode", -2.0))))
        and abs(float(getattr(full_message, "toe", np.nan)) - float(getattr(filtered_message, "toe", np.nan))) < 1.0e-6
    )
    if same_selected_message:
        if abs(float(derived_common_clock_m) - filtered_common_clock_m) <= GPS_NAV_PRODUCT_ADJUSTMENT_THRESHOLD_M:
            return None
    elif abs(float(derived_common_clock_m) - full_common_clock_m) >= abs(
        float(derived_common_clock_m) - filtered_common_clock_m
    ):
        return None

    corrected_tow_s = transmit_tow_s - gps_broadcast_clock_bias_s(filtered_message, transmit_tow_s)

    sat_pos, clock_s = Ephemeris._compute_single_cpu(filtered_message, corrected_tow_s, "C1")
    sat_pos_dt, clock_dt_s = Ephemeris._compute_single_cpu(filtered_message, corrected_tow_s + 1.0e-3, "C1")
    sat_vel = (np.asarray(sat_pos_dt, dtype=np.float64) - np.asarray(sat_pos, dtype=np.float64)) / 1.0e-3
    sat_clock_bias_m = float(clock_s) * LIGHT_SPEED_MPS + float(getattr(filtered_message, "tgd", 0.0)) * LIGHT_SPEED_MPS
    sat_clock_drift_mps = (float(clock_dt_s) - float(clock_s)) * LIGHT_SPEED_MPS / 1.0e-3
    return np.asarray(sat_pos, dtype=np.float64), sat_vel, sat_clock_bias_m, sat_clock_drift_mps


def gps_sat_clock_bias_adjustment_m(
    constellation_type: int,
    svid: int,
    signal_type: str,
    tgd_m_by_svid: dict[int, float],
) -> float:
    if int(constellation_type) != 1:
        return 0.0
    tgd_m = tgd_m_by_svid.get(int(svid))
    if tgd_m is None or not np.isfinite(tgd_m):
        return 0.0
    scale = GPS_L5_TGD_SCALE if is_l5_signal(signal_type) else 1.0
    return float(scale * tgd_m)


def course_base_obs_path(data_root: Path, split: str, course: str, base_name: str, rinex_type: str | None) -> Path:
    suffix = "rnx3" if rinex_type == "V3" else "rnx2" if rinex_type == "V2" else "rnx3"
    path = data_root / split / course / f"{base_name}_{suffix}.obs"
    if not path.is_file() and suffix != "rnx3":
        alt = data_root / split / course / f"{base_name}_rnx3.obs"
        if alt.is_file():
            return alt
    if not path.is_file() and suffix != "rnx2":
        alt = data_root / split / course / f"{base_name}_rnx2.obs"
        if alt.is_file():
            return alt
    if not path.is_file():
        raise FileNotFoundError(f"base observation file not found: {path}")
    return path


def base_setting(data_root: Path, split: str, course: str, phone: str) -> tuple[str, str | None]:
    settings = load_settings_frame_cached(str(data_root), split)
    if settings is None:
        raise FileNotFoundError(f"settings_{split}.csv not found under {data_root}")
    rows = settings[(settings["Course"].astype(str) == course) & (settings["Phone"].astype(str) == phone)]
    if rows.empty:
        raise RuntimeError(f"settings row not found for {split}/{course}/{phone}")
    row = rows.iloc[0]
    base_raw = row.get("Base1", np.nan)
    if pd.isna(base_raw) or not str(base_raw).strip():
        raise RuntimeError(f"Base1 is empty for {split}/{course}/{phone}")
    rinex_raw = row.get("RINEX", np.nan)
    rinex_type = str(rinex_raw).strip() if pd.notna(rinex_raw) and str(rinex_raw).strip() else None
    return str(base_raw).strip(), rinex_type


def read_base_station_xyz(data_root: Path, course: str, base_name: str, *, apply_offset: bool = True) -> np.ndarray:
    base_dir = base_metadata_dir(data_root)
    pos_path = base_dir / "base_position.csv"
    off_path = base_dir / "base_offset.csv"
    if not pos_path.is_file():
        raise FileNotFoundError(f"base_position.csv not found: {pos_path}")
    if not off_path.is_file():
        raise FileNotFoundError(f"base_offset.csv not found: {off_path}")

    year = int(str(course).split("-", maxsplit=1)[0])
    base_l = str(base_name).lower()
    pos_df = pd.read_csv(pos_path)
    rows = pos_df[
        pos_df["Base"].astype(str).str.lower().str.contains(base_l, regex=False, na=False)
        & (pd.to_numeric(pos_df["Year"], errors="coerce") == year)
    ]
    if rows.empty:
        rows = pos_df[pos_df["Base"].astype(str).str.lower().str.contains(base_l, regex=False, na=False)]
    if rows.empty:
        raise RuntimeError(f"base position not found for {base_name} in {pos_path}")
    xyz = rows[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").mean(axis=0).to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise RuntimeError(f"invalid base position for {base_name} in {pos_path}")
    if not apply_offset:
        return xyz

    off_df = pd.read_csv(off_path)
    off_rows = off_df[off_df["Base"].astype(str).str.lower().str.contains(base_l, regex=False, na=False)]
    if off_rows.empty:
        return xyz
    enu_offset = off_rows[["E", "N", "U"]].apply(pd.to_numeric, errors="coerce").mean(axis=0).to_numpy(dtype=np.float64)
    if not np.isfinite(enu_offset).all() or np.linalg.norm(enu_offset) == 0.0:
        return xyz
    return enu_to_ecef_relative(enu_offset, xyz)


def moving_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return (
        pd.DataFrame(arr)
        .rolling(window=max(int(window), 1), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def round_seconds_to_interval_like_matlab(values: np.ndarray | float, interval_s: float) -> np.ndarray | float:
    interval = float(interval_s)
    if not np.isfinite(interval) or interval <= 0.0:
        return values
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.floor(arr / interval + 0.5) * interval
    if np.isscalar(values):
        return float(rounded)
    return rounded


def matlab_base_time_span_mask(
    base_times_gps_s: np.ndarray,
    phone_start_gps_s: float,
    phone_end_gps_s: float,
    base_dt_s: float,
) -> np.ndarray:
    """Match preprocessing.m base selectTimeSpan(ts-180, te+180)."""

    base_times = np.asarray(base_times_gps_s, dtype=np.float64)
    if (
        base_times.size == 0
        or not np.isfinite(phone_start_gps_s)
        or not np.isfinite(phone_end_gps_s)
        or not np.isfinite(base_dt_s)
        or base_dt_s <= 0.0
    ):
        return np.ones(base_times.shape, dtype=bool)

    start = round_seconds_to_interval_like_matlab(phone_start_gps_s - BASE_OBS_TRIM_MARGIN_S, base_dt_s)
    end = round_seconds_to_interval_like_matlab(phone_end_gps_s + BASE_OBS_TRIM_MARGIN_S, base_dt_s)
    rounded_base = round_seconds_to_interval_like_matlab(base_times, base_dt_s)
    return (rounded_base >= start) & (rounded_base <= end)


@lru_cache(maxsize=32)
def trip_full_phone_time_span_gps_abs_cached(
    data_root_str: str,
    split: str,
    course: str,
    phone: str,
) -> tuple[float, float]:
    raw_path = Path(data_root_str) / split / course / phone / "device_gnss.csv"
    frame = read_raw_gnss_csv(raw_path, usecols=["utcTimeMillis"])
    times_ms = pd.to_numeric(frame["utcTimeMillis"], errors="coerce").to_numpy(dtype=np.float64)
    times_ms = times_ms[np.isfinite(times_ms)]
    if times_ms.size == 0:
        raise RuntimeError(f"device_gnss.csv has no finite utcTimeMillis values: {raw_path}")
    phone_times = unix_ms_to_gps_abs_seconds(np.array([np.min(times_ms), np.max(times_ms)], dtype=np.float64))
    return float(phone_times[0]), float(phone_times[1])


def trip_phone_time_span_for_base_trim(
    data_root: Path,
    split: str,
    course: str,
    phone: str,
    selected_phone_times_gps_s: np.ndarray,
) -> tuple[float, float]:
    try:
        return trip_full_phone_time_span_gps_abs_cached(str(data_root), split, course, phone)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError):
        selected = np.asarray(selected_phone_times_gps_s, dtype=np.float64)
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            return float("nan"), float("nan")
        return float(np.min(selected)), float(np.max(selected))


def rtklib_tropo_saastamoinen(lat_rad: float, alt_m: float, el_rad: float, humidity: float = 0.7) -> float:
    if (
        not np.isfinite(lat_rad)
        or not np.isfinite(alt_m)
        or not np.isfinite(el_rad)
        or alt_m < -100.0
        or alt_m > 10000.0
        or el_rad <= 0.0
    ):
        return 0.0
    hgt = max(float(alt_m), 0.0)
    pres = 1013.25 * ((1.0 - 2.2557e-5 * hgt) ** 5.2568)
    temp = 15.0 - 6.5e-3 * hgt + 273.16
    water_vapor = 6.108 * float(humidity) * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    denom = 1.0 - 0.00266 * np.cos(2.0 * float(lat_rad)) - 0.00028 * hgt / 1000.0
    sin_el = np.sin(float(el_rad))
    if denom == 0.0 or sin_el <= 0.0:
        return 0.0
    dry = 0.0022768 * pres / denom / sin_el
    wet = 0.002277 * (1255.0 / temp + 0.05) * water_vapor / sin_el
    return float(dry + wet)


@lru_cache(maxsize=8)
def load_base_residual_series_cached(
    data_root_str: str,
    split: str,
    course: str,
    base_name: str,
    rinex_type: str | None,
    signal_type: str,
    phone_start_gps_s: float,
    phone_end_gps_s: float,
    sat_ids_key: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    data_root = Path(data_root_str)
    base_obs_path = course_base_obs_path(data_root, split, course, base_name, rinex_type)
    nav_path = course_nav_path(data_root, split, course)
    base_xyz = read_base_station_xyz(data_root, course, base_name, apply_offset=False)
    base_obs = read_rinex_obs(base_obs_path)
    nav_messages = read_nav_rinex_multi(nav_path, systems=("G", "E", "J"))
    for sat_id, messages in list(nav_messages.items()):
        system = str(getattr(messages[0], "system", "")).upper() if messages else str(sat_id).upper()[:1]
        if system == "G":
            nav_messages[sat_id] = filter_matrtklib_duplicate_gps_nav_messages(list(messages))
    nav = Ephemeris(nav_messages)
    alpha, beta = read_gps_klobuchar_from_nav_header(nav_path)
    lat, lon, alt = ecef_to_lla(float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2]))
    if alpha is None:
        alpha = [0.1118e-07, -0.7451e-08, -0.5961e-07, 0.1192e-06]
    if beta is None:
        beta = [0.1167e06, -0.2294e06, -0.1311e06, 0.1049e07]
    iono_scale = signal_type_iono_scale(signal_type)

    sat_ids = tuple(sat_ids_key)
    base_times_all = np.asarray([gps_abs_seconds_from_datetime(ep.time) for ep in base_obs.epochs], dtype=np.float64)
    if base_times_all.size > 1:
        base_dt = float(np.nanmedian(np.diff(base_times_all)))
        keep = matlab_base_time_span_mask(base_times_all, phone_start_gps_s, phone_end_gps_s, base_dt)
        if np.any(keep):
            base_times = base_times_all[keep]
            base_epochs = [epoch for epoch, keep_epoch in zip(base_obs.epochs, keep) if bool(keep_epoch)]
        else:
            base_times = base_times_all
            base_epochs = base_obs.epochs
    else:
        base_times = base_times_all
        base_epochs = base_obs.epochs
    residuals = np.full((base_times.size, len(sat_ids)), np.nan, dtype=np.float64)
    sat_to_col = {sat_id: idx for idx, sat_id in enumerate(sat_ids)}

    for epoch_idx, epoch in enumerate(base_epochs):
        gps_abs_s = base_times[epoch_idx]
        gps_tow = gps_abs_s % 604800.0
        epoch_observations: list[tuple[str, str, float]] = []
        for sat_id in sat_ids:
            obs = epoch.observations.get(sat_id)
            if not obs:
                continue
            selected = select_base_pseudorange_observation(obs, signal_type)
            if selected is None:
                continue
            selected_code, selected_pr = selected
            epoch_observations.append((sat_id, selected_code, selected_pr))
        if not epoch_observations:
            continue

        for sat_id, selected_code, pr in epoch_observations:
            col = sat_to_col.get(str(sat_id))
            if col is None:
                continue
            nav_msg = nav.select_ephemeris(sat_id, gps_tow, selected_code)
            if nav_msg is None:
                continue
            transmit_tow = gps_tow - float(pr) / C_LIGHT
            sat_pos = None
            sat_clk_s = np.nan
            for _ in range(3):
                sat_pos, sat_clk_s = Ephemeris._compute_single_cpu(
                    nav_msg,
                    transmit_tow,
                    selected_code,
                    apply_group_delay=False,
                )
                transmit_tow = gps_tow - float(pr) / C_LIGHT - float(sat_clk_s)
            if sat_pos is None:
                continue
            sat_pos = np.asarray(sat_pos, dtype=np.float64)
            rho = float(geometric_range_with_sagnac(sat_pos, base_xyz))
            if not np.isfinite(rho) or rho <= 1.0e6:
                continue
            el, az = _elevation_azimuth(base_xyz, sat_pos)
            trop = rtklib_tropo_saastamoinen(float(lat), float(alt), float(el))
            iono = iono_scale * _iono_klobuchar(alpha, beta, float(lat), float(lon), float(az), float(el), gps_tow)
            residuals[epoch_idx, col] = float(pr) + float(sat_clk_s) * C_LIGHT - rho - trop - iono

    if base_times.size > 1:
        dt = float(np.nanmedian(np.diff(base_times)))
    else:
        dt = 0.0
    window = BASE_MOVMEAN_N_1S if dt > 0.0 and dt <= 1.5 else BASE_MOVMEAN_N_15S
    residuals = moving_nanmean(residuals, window)
    return base_times, sat_ids, residuals


def compute_base_pseudorange_correction_matrix(
    data_root: Path,
    trip: str,
    times_ms: np.ndarray,
    slot_keys: list[tuple[int, int] | tuple[int, int, str]],
    signal_type: str,
    *,
    base_setting_fn: BaseSettingFn = base_setting,
    base_residual_loader: BaseResidualLoader = load_base_residual_series_cached,
    phone_span_fn: PhoneTimeSpanFn = trip_phone_time_span_for_base_trim,
) -> np.ndarray:
    split, course, phone = trip_course_phone(trip)
    if split is None or course is None or phone is None:
        raise RuntimeError(f"trip must include split/course/phone for base correction: {trip}")
    base_name, rinex_type = base_setting_fn(data_root, split, course, phone)
    normalized_slots = [
        (int(key[0]), int(key[1]), str(key[2]) if len(key) >= 3 else signal_type)
        for key in slot_keys
    ]
    sat_ids = [slot_sat_id(constellation_type, svid) for constellation_type, svid, _ in normalized_slots]
    supported_pairs = [
        (idx, sat_id, normalized_slots[idx][2])
        for idx, sat_id in enumerate(sat_ids)
        if sat_id is not None and sat_id.startswith(("G", "E", "J"))
    ]
    correction = np.full((len(times_ms), len(slot_keys)), np.nan, dtype=np.float64)
    if not supported_pairs:
        return correction

    phone_times = unix_ms_to_gps_abs_seconds(times_ms)
    phone_start_gps_s, phone_end_gps_s = phone_span_fn(
        data_root,
        split,
        course,
        phone,
        phone_times,
    )
    for group_signal_type in sorted({sig for _, _, sig in supported_pairs}):
        group_pairs = [(slot_idx, sat_id) for slot_idx, sat_id, sig in supported_pairs if sig == group_signal_type]
        supported_sat_ids = tuple(sorted({sat_id for _, sat_id in group_pairs if sat_id is not None}))
        base_times, corr_sat_ids, residuals = base_residual_loader(
            str(data_root),
            split,
            course,
            base_name,
            rinex_type,
            group_signal_type,
            phone_start_gps_s,
            phone_end_gps_s,
            supported_sat_ids,
        )
        if base_times.size == 0:
            continue
        sat_to_base_col = {sat_id: idx for idx, sat_id in enumerate(corr_sat_ids)}
        for slot_idx, sat_id in group_pairs:
            base_col = sat_to_base_col.get(str(sat_id))
            if base_col is None:
                continue
            series = residuals[:, base_col]
            valid = np.isfinite(series)
            if valid.sum() < 2:
                continue
            in_range = (phone_times >= base_times[valid][0]) & (phone_times <= base_times[valid][-1])
            correction[in_range, slot_idx] = np.interp(phone_times[in_range], base_times[valid], series[valid])
    return correction
