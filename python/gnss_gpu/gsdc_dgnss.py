"""GSDC 2023 DGNSS helpers built around NOAA CORS RINEX files.

The module keeps the CORS-specific work outside the PF experiments:

* choose candidate CORS stations from a GSDC trajectory/run name;
* download and normalize NOAA CORS RINEX 2 observation files;
* adapt GSDC ``device_gnss.csv`` rows to ``DDPseudorangeComputer`` inputs;
* solve a conservative DD-pseudorange position update around the Android WLS
  seed for train-set diagnostics.
"""

from __future__ import annotations

import gzip
import math
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np

from gnss_gpu.dd_pseudorange import DDPseudorangeComputer, DDPseudorangeResult

GPS_WEEK_SECONDS = 604800.0
DEFAULT_CORS_CACHE = Path("/tmp/gsdc_cors")
GSDC_L1_E1_SIGNALS = frozenset({"GPS_L1_CA", "GAL_E1_C_P"})
SIGNAL_SYSTEM_IDS = {"GPS_L1_CA": 0, "GAL_E1_C_P": 2}


@dataclass(frozen=True)
class CorsDownload:
    station: str
    rinex_obs_path: Path
    source_url: str


@dataclass(frozen=True)
class DDWLSConfig:
    min_dd_pairs: int = 3
    dd_sigma_m: float = 8.0
    prior_sigma_m: float = 3.0
    max_shift_m: float = 20.0
    max_iter: int = 4
    tol_m: float = 1.0e-3


AREA_CANDIDATES: dict[str, tuple[str, ...]] = {
    "mtv": ("SLAC", "P222", "ZOA2", "ZOA1"),
    "pao": ("SLAC", "P222", "ZOA2", "ZOA1"),
    "sjc": ("MHC2", "MHCB", "P222", "P217", "SLAC"),
    "lax": ("TORP", "CRHS", "PVEP", "VDCY", "JPLM"),
    "la": ("VDCY", "JPLM", "CRHS", "TORP"),
}

AREA_CENTERS_LLA: dict[str, tuple[float, float]] = {
    "mtv": (37.3861, -122.0839),
    "sjc": (37.3382, -121.8863),
    "lax": (33.9425, -118.4081),
    "la": (34.0522, -118.2437),
}


def _station_id(station: str) -> str:
    station = station.strip().lower()
    if len(station) != 4 or not station.isalnum():
        raise ValueError(f"CORS station must be a 4-character ID, got {station!r}")
    return station


def _run_date(run_name: str) -> date:
    parts = run_name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse GSDC run date from {run_name!r}")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _gps_tow_from_arrival_ns(arrival_ns: float | int) -> float:
    return (float(arrival_ns) * 1.0e-9) % GPS_WEEK_SECONDS


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + alt_m) * cos_lat * math.cos(lon),
            (n + alt_m) * cos_lat * math.sin(lon),
            (n * (1.0 - e2) + alt_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def _infer_area_from_run_name(run_name: str) -> str | None:
    lowered = run_name.lower()
    for token in ("sjc", "lax", "pao", "mtv"):
        if f"-{token}" in lowered or lowered.endswith(f"-{token}"):
            return token
    if "los-angeles" in lowered:
        return "la"
    return None


def _trajectory_center_ecef(trajectory_ecef: np.ndarray | None) -> np.ndarray | None:
    if trajectory_ecef is None:
        return None
    arr = np.asarray(trajectory_ecef, dtype=np.float64).reshape(-1, 3)
    finite = arr[np.all(np.isfinite(arr), axis=1) & (np.linalg.norm(arr, axis=1) > 1.0e6)]
    if finite.size == 0:
        return None
    return np.median(finite, axis=0)


def cors_station_candidates(
    *,
    run_name: str | None = None,
    trajectory_ecef: np.ndarray | None = None,
    extra_stations: Sequence[str] = (),
) -> list[str]:
    """Return ordered CORS station candidates for a GSDC route.

    The route token is used when available because the GSDC run names encode the
    coarse city/area.  If no token is present, the trajectory centroid is matched
    to the nearest known GSDC area center.
    """

    area = _infer_area_from_run_name(run_name) if run_name else None
    if area is None:
        center = _trajectory_center_ecef(trajectory_ecef)
        if center is not None:
            best_area = None
            best_dist = float("inf")
            for name, (lat, lon) in AREA_CENTERS_LLA.items():
                dist = float(np.linalg.norm(center - _lla_to_ecef(lat, lon)))
                if dist < best_dist:
                    best_area = name
                    best_dist = dist
            area = best_area
    if area is None:
        area = "mtv"

    ordered: list[str] = []
    for station in [*extra_stations, *AREA_CANDIDATES.get(area, AREA_CANDIDATES["mtv"])]:
        station_up = station.strip().upper()
        if station_up and station_up not in ordered:
            ordered.append(station_up)
    return ordered


def cors_rinex_urls(station: str, run_date: date, *, prefer_hatanaka: bool = True) -> list[str]:
    station_l = _station_id(station)
    doy = run_date.timetuple().tm_yday
    yy = run_date.year % 100
    exts = ("d", "o") if prefer_hatanaka else ("o", "d")
    roots = (
        "https://noaa-cors-pds.s3.amazonaws.com/rinex",
        "https://geodesy.noaa.gov/corsdata/rinex",
    )
    urls: list[str] = []
    for ext in exts:
        filename = f"{station_l}{doy:03d}0.{yy:02d}{ext}.gz"
        for root in roots:
            urls.append(f"{root}/{run_date.year}/{doy:03d}/{station_l}/{filename}")
    return urls


def _download(url: str, dest: Path, *, timeout_s: float = 30.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "gnss-gpu-gsdc-dgnss/0.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)


def _gunzip(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as inp, open(dest, "wb") as out:
        shutil.copyfileobj(inp, out)
    return dest


def _convert_hatanaka(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    crx2rnx = shutil.which("crx2rnx")
    if crx2rnx is not None:
        subprocess.run(
            [crx2rnx, "-f", str(src)],
            cwd=str(src.parent),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        produced = src.with_suffix(".o")
        if not produced.exists():
            produced = src.with_name(src.name[:-1] + "o")
        if produced.exists():
            if produced != dest:
                shutil.move(str(produced), dest)
            return dest

    try:
        import hatanaka  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Hatanaka CORS file requires crx2rnx or the optional hatanaka package"
        ) from exc
    decompressed = hatanaka.decompress(str(src))
    if isinstance(decompressed, bytes):
        dest.write_bytes(decompressed)
    else:
        dest.write_text(str(decompressed))
    return dest


def normalize_cors_rinex(downloaded_gz: Path) -> Path:
    """Return a plain RINEX observation path for a NOAA ``.d.gz``/``.o.gz`` file."""

    if downloaded_gz.suffix != ".gz":
        plain = downloaded_gz
    else:
        plain = downloaded_gz.with_suffix("")
        if not plain.exists() or plain.stat().st_mtime < downloaded_gz.stat().st_mtime:
            _gunzip(downloaded_gz, plain)
    if plain.name.endswith("d"):
        obs_path = plain.with_name(plain.name[:-1] + "o")
        if not obs_path.exists() or obs_path.stat().st_mtime < plain.stat().st_mtime:
            _convert_hatanaka(plain, obs_path)
        return obs_path
    return plain


def fetch_cors_rinex(
    station: str,
    run_date: date,
    *,
    cache_dir: Path = DEFAULT_CORS_CACHE,
    timeout_s: float = 30.0,
) -> CorsDownload:
    """Download the first available NOAA CORS daily RINEX observation file."""

    station_l = _station_id(station)
    doy = run_date.timetuple().tm_yday
    station_dir = cache_dir / f"{run_date.year}" / f"{doy:03d}" / station_l
    last_error: Exception | None = None
    for url in cors_rinex_urls(station_l, run_date):
        filename = url.rsplit("/", 1)[-1]
        gz_path = station_dir / filename
        if not gz_path.exists():
            try:
                _download(url, gz_path, timeout_s=timeout_s)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if gz_path.exists() and gz_path.stat().st_size == 0:
                    gz_path.unlink()
                continue
        obs_path = normalize_cors_rinex(gz_path)
        return CorsDownload(station=station_l.upper(), rinex_obs_path=obs_path, source_url=url)
    raise FileNotFoundError(
        f"No NOAA CORS RINEX observation file found for {station_l.upper()} on {run_date}: "
        f"{last_error}"
    )


def fetch_first_available_cors_rinex(
    stations: Sequence[str],
    run_date: date,
    *,
    cache_dir: Path = DEFAULT_CORS_CACHE,
    timeout_s: float = 30.0,
) -> CorsDownload:
    errors: list[str] = []
    for station in stations:
        try:
            return fetch_cors_rinex(
                station,
                run_date,
                cache_dir=cache_dir,
                timeout_s=timeout_s,
            )
        except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(f"{station}: {exc}")
    raise FileNotFoundError("; ".join(errors))


def gsdc_corrected_pseudorange(row) -> float:
    iono = 0.0 if not np.isfinite(row.IonosphericDelayMeters) else float(row.IonosphericDelayMeters)
    tropo = 0.0 if not np.isfinite(row.TroposphericDelayMeters) else float(row.TroposphericDelayMeters)
    isrb = 0.0 if not np.isfinite(row.IsrbMeters) else float(row.IsrbMeters)
    return float(row.RawPseudorangeMeters + row.SvClockBiasMeters - iono - tropo - isrb)


def gsdc_epoch_measurements(
    group,
    *,
    apply_gsdc_corrections: bool = True,
) -> list[SimpleNamespace]:
    """Adapt one GSDC dataframe group to DD pseudorange measurement rows."""

    measurements: list[SimpleNamespace] = []
    for row in group.itertuples(index=False):
        system_id = SIGNAL_SYSTEM_IDS.get(str(row.SignalType))
        if system_id is None:
            continue
        sat_ecef = np.array(
            [
                row.SvPositionXEcefMeters,
                row.SvPositionYEcefMeters,
                row.SvPositionZEcefMeters,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(sat_ecef)):
            continue
        pseudorange = (
            gsdc_corrected_pseudorange(row)
            if apply_gsdc_corrections
            else float(row.RawPseudorangeMeters)
        )
        if not np.isfinite(pseudorange) or abs(pseudorange) < 1.0e6:
            continue
        measurements.append(
            SimpleNamespace(
                system_id=system_id,
                prn=int(row.Svid),
                satellite_ecef=sat_ecef,
                corrected_pseudorange=pseudorange,
                elevation=math.radians(max(float(row.SvElevationDegrees), 0.0)),
                snr=float(row.Cn0DbHz) if np.isfinite(row.Cn0DbHz) else 0.0,
                weight=max((float(row.Cn0DbHz) - 20.0) / 25.0, 0.1)
                if np.isfinite(row.Cn0DbHz)
                else 0.1,
            )
        )
    return measurements


def iter_gsdc_rover_epochs(
    gnss_csv: Path,
    *,
    signal_types: Iterable[str] = GSDC_L1_E1_SIGNALS,
    apply_gsdc_corrections: bool = False,
):
    """Yield per-epoch GSDC rover measurements for DD pseudorange."""

    import pandas as pd

    usecols = [
        "ArrivalTimeNanosSinceGpsEpoch",
        "Svid",
        "RawPseudorangeMeters",
        "SvClockBiasMeters",
        "IonosphericDelayMeters",
        "TroposphericDelayMeters",
        "IsrbMeters",
        "SignalType",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvElevationDegrees",
        "Cn0DbHz",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    df = pd.read_csv(gnss_csv, usecols=usecols, low_memory=False)
    df = df[df["SignalType"].isin(set(signal_types))].copy()
    critical = [
        "RawPseudorangeMeters",
        "SvClockBiasMeters",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    df = df.dropna(subset=critical)
    for arrival_ns, group in df.groupby("ArrivalTimeNanosSinceGpsEpoch", sort=True):
        measurements = gsdc_epoch_measurements(
            group,
            apply_gsdc_corrections=apply_gsdc_corrections,
        )
        if len(measurements) < 4:
            continue
        wls = np.array(
            [
                group["WlsPositionXEcefMeters"].iloc[0],
                group["WlsPositionYEcefMeters"].iloc[0],
                group["WlsPositionZEcefMeters"].iloc[0],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(wls)) or np.linalg.norm(wls) < 1.0e6:
            continue
        yield {
            "arrival_ns": int(arrival_ns),
            "tow": _gps_tow_from_arrival_ns(arrival_ns),
            "measurements": measurements,
            "wls_ecef": wls,
        }


def dd_pseudorange_position_update(
    seed_ecef: np.ndarray,
    dd: DDPseudorangeResult,
    config: DDWLSConfig = DDWLSConfig(),
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Conservative DD-pseudorange WLS update around a seed ECEF position."""

    seed = np.asarray(seed_ecef, dtype=np.float64).reshape(3)
    stats: dict[str, float | int | bool] = {
        "accepted": False,
        "n_dd": int(dd.n_dd),
        "shift_m": 0.0,
        "initial_rms_m": float("inf"),
        "final_rms_m": float("inf"),
    }
    if dd.n_dd < int(config.min_dd_pairs) or not np.all(np.isfinite(seed)):
        return seed.copy(), stats

    pos = seed.copy()
    for _iteration in range(int(config.max_iter)):
        range_k = np.linalg.norm(dd.sat_ecef_k - pos, axis=1)
        range_ref = np.linalg.norm(dd.sat_ecef_ref - pos, axis=1)
        expected = range_k - range_ref - dd.base_range_k + dd.base_range_ref
        residual = dd.dd_pseudorange_m - expected
        if not np.all(np.isfinite(residual)):
            return seed.copy(), stats
        if math.isinf(float(stats["initial_rms_m"])):
            stats["initial_rms_m"] = float(np.sqrt(np.mean(residual * residual)))

        unit_k = (dd.sat_ecef_k - pos) / np.maximum(range_k[:, None], 1.0)
        unit_ref = (dd.sat_ecef_ref - pos) / np.maximum(range_ref[:, None], 1.0)
        design = -unit_k + unit_ref
        weights = np.clip(dd.dd_weights, 1.0e-6, None) / max(config.dd_sigma_m, 1.0e-6) ** 2
        lhs = design * np.sqrt(weights)[:, None]
        rhs = residual * np.sqrt(weights)
        if config.prior_sigma_m > 0.0:
            prior_w = 1.0 / (config.prior_sigma_m * config.prior_sigma_m)
            lhs = np.vstack([lhs, np.eye(3) * math.sqrt(prior_w)])
            rhs = np.concatenate([rhs, (seed - pos) * math.sqrt(prior_w)])
        try:
            delta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return seed.copy(), stats
        if not np.all(np.isfinite(delta)):
            return seed.copy(), stats
        pos += delta
        if float(np.linalg.norm(delta)) < float(config.tol_m):
            break

    shift = float(np.linalg.norm(pos - seed))
    range_k = np.linalg.norm(dd.sat_ecef_k - pos, axis=1)
    range_ref = np.linalg.norm(dd.sat_ecef_ref - pos, axis=1)
    residual = dd.dd_pseudorange_m - (range_k - range_ref - dd.base_range_k + dd.base_range_ref)
    stats["shift_m"] = shift
    stats["final_rms_m"] = float(np.sqrt(np.mean(residual * residual)))
    if shift <= float(config.max_shift_m):
        stats["accepted"] = True
        return pos, stats
    return seed.copy(), stats


def build_dd_computer(download: CorsDownload) -> DDPseudorangeComputer:
    return DDPseudorangeComputer(
        download.rinex_obs_path,
        pseudorange_obs_code=None,
        allowed_systems=("G", "E"),
        base_epoch_tolerance_s=0.6,
    )


def run_date_from_gsdc_run(run_name: str) -> date:
    return _run_date(run_name)
