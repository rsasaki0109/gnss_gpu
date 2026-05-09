"""Reset-safe post smoothing for GSDC2023 latitude/longitude submissions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
_EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class SmoothConfig:
    median_window: int = 7
    smooth_window: int = 5
    blend: float = 0.5
    max_correction_m: float = 3.0
    smooth_kernel: str = "triangular"
    gaussian_sigma: float = 1.0
    hampel_sigma: float = 6.0
    hampel_min_m: float = 8.0
    segment_gap_ms: float = 3000.0
    segment_step_m: float = 100.0
    min_segment_points: int = 5


@dataclass(frozen=True)
class SmoothStats:
    groups: int
    rows: int
    segments: int
    corrected_rows: int
    hampel_rows: int
    max_correction_m: float
    mean_correction_m: float
    p95_correction_m: float


def _positive_odd(value: int) -> int:
    value = int(value)
    if value < 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _local_radii(lat0_rad: float) -> tuple[float, float]:
    sin_lat = math.sin(lat0_rad)
    denom = math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    prime_vertical = _WGS84_A / denom
    meridian = _WGS84_A * (1.0 - _WGS84_E2) / (denom * denom * denom)
    return meridian, prime_vertical


def latlon_to_local_m(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    origin_lat_deg: float | None = None,
    origin_lon_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if origin_lat_deg is None:
        origin_lat_deg = float(lat[valid][0]) if np.any(valid) else 0.0
    if origin_lon_deg is None:
        origin_lon_deg = float(lon[valid][0]) if np.any(valid) else 0.0

    lat0 = math.radians(origin_lat_deg)
    lon0 = math.radians(origin_lon_deg)
    meridian, prime_vertical = _local_radii(lat0)
    north = (np.deg2rad(lat) - lat0) * meridian
    east = (np.deg2rad(lon) - lon0) * prime_vertical * math.cos(lat0)
    return east, north, float(origin_lat_deg), float(origin_lon_deg)


def local_m_to_latlon(
    east_m: np.ndarray,
    north_m: np.ndarray,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    lat0 = math.radians(origin_lat_deg)
    lon0 = math.radians(origin_lon_deg)
    meridian, prime_vertical = _local_radii(lat0)
    lat = lat0 + north / meridian
    lon = lon0 + east / (prime_vertical * math.cos(lat0))
    return np.rad2deg(lat), np.rad2deg(lon)


def haversine_m(lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(np.asarray(lat_a, dtype=np.float64))
    lon1 = np.deg2rad(np.asarray(lon_a, dtype=np.float64))
    lat2 = np.deg2rad(np.asarray(lat_b, dtype=np.float64))
    lon2 = np.deg2rad(np.asarray(lon_b, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def gsdc_score_m(errors_m: np.ndarray) -> dict[str, float | int]:
    errors = np.asarray(errors_m, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {
            "n": 0,
            "score_m": float("nan"),
            "p50_m": float("nan"),
            "p95_m": float("nan"),
            "mean_m": float("nan"),
            "max_m": float("nan"),
        }
    p50 = float(np.percentile(errors, 50))
    p95 = float(np.percentile(errors, 95))
    return {
        "n": int(errors.size),
        "score_m": 0.5 * (p50 + p95),
        "p50_m": p50,
        "p95_m": p95,
        "mean_m": float(np.mean(errors)),
        "max_m": float(np.max(errors)),
    }


def _window_bounds(i: int, n: int, half: int) -> tuple[int, int]:
    return max(0, i - half), min(n, i + half + 1)


def _rolling_median_xy(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    window = _positive_odd(window)
    half = window // 2
    med_x = np.array(x, copy=True)
    med_y = np.array(y, copy=True)
    for i in range(n):
        lo, hi = _window_bounds(i, n, half)
        med_x[i] = float(np.nanmedian(x[lo:hi]))
        med_y[i] = float(np.nanmedian(y[lo:hi]))
    return med_x, med_y


def _triangular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    n = int(values.size)
    window = _positive_odd(window)
    if window <= 1 or n == 0:
        return np.array(values, copy=True)
    half = window // 2
    smoothed = np.array(values, copy=True)
    for i in range(half, n - half):
        lo, hi = i - half, i + half + 1
        offsets = np.arange(lo, hi, dtype=np.float64) - float(i)
        weights = (half + 1.0) - np.abs(offsets)
        smoothed[i] = float(np.sum(values[lo:hi] * weights) / np.sum(weights))
    return smoothed


def _gaussian_smooth(values: np.ndarray, window: int, sigma: float) -> np.ndarray:
    n = int(values.size)
    window = _positive_odd(window)
    if window <= 1 or n == 0:
        return np.array(values, copy=True)
    half = window // 2
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = max(float(half) / 1.5, 1.0)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= np.sum(weights)
    smoothed = np.array(values, copy=True)
    for i in range(half, n - half):
        smoothed[i] = float(np.sum(values[i - half : i + half + 1] * weights))
    return smoothed


def _boxcar_smooth(values: np.ndarray, window: int) -> np.ndarray:
    n = int(values.size)
    window = _positive_odd(window)
    if window <= 1 or n == 0:
        return np.array(values, copy=True)
    half = window // 2
    smoothed = np.array(values, copy=True)
    for i in range(half, n - half):
        smoothed[i] = float(np.mean(values[i - half : i + half + 1]))
    return smoothed


def _smooth_series(values: np.ndarray, config: SmoothConfig) -> np.ndarray:
    kernel = str(config.smooth_kernel).lower()
    if kernel == "triangular":
        return _triangular_smooth(values, config.smooth_window)
    if kernel == "gaussian":
        return _gaussian_smooth(values, config.smooth_window, config.gaussian_sigma)
    if kernel == "boxcar":
        return _boxcar_smooth(values, config.smooth_window)
    raise ValueError(f"unsupported smooth kernel: {config.smooth_kernel}")


def _split_segments(times_ms: np.ndarray, east_m: np.ndarray, north_m: np.ndarray, config: SmoothConfig) -> list[slice]:
    n = int(east_m.size)
    if n == 0:
        return []
    breaks = [0]
    times = np.asarray(times_ms, dtype=np.float64)
    step = np.hypot(np.diff(east_m), np.diff(north_m))
    isolated_spike = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        neighbor_step = math.hypot(float(east_m[i + 1] - east_m[i - 1]), float(north_m[i + 1] - north_m[i - 1]))
        if step[i - 1] > config.segment_step_m and step[i] > config.segment_step_m and neighbor_step <= config.segment_step_m:
            isolated_spike[i] = True
    dt = np.diff(times) if times.size == n else np.zeros(max(0, n - 1), dtype=np.float64)
    for i in range(1, n):
        gap_break = bool(dt.size and np.isfinite(dt[i - 1]) and dt[i - 1] > config.segment_gap_ms)
        step_break = bool(
            np.isfinite(step[i - 1])
            and step[i - 1] > config.segment_step_m
            and not isolated_spike[i - 1]
            and not isolated_spike[i]
        )
        if gap_break or step_break:
            breaks.append(i)
    breaks.append(n)
    return [slice(start, end) for start, end in zip(breaks[:-1], breaks[1:]) if end > start]


def _apply_cap(dx: np.ndarray, dy: np.ndarray, max_norm: float) -> tuple[np.ndarray, np.ndarray]:
    if max_norm <= 0.0:
        return np.zeros_like(dx), np.zeros_like(dy)
    norm = np.hypot(dx, dy)
    scale = np.ones_like(norm)
    mask = norm > max_norm
    scale[mask] = max_norm / np.maximum(norm[mask], 1e-12)
    return dx * scale, dy * scale


def smooth_local_xy(
    times_ms: np.ndarray,
    east_m: np.ndarray,
    north_m: np.ndarray,
    config: SmoothConfig,
) -> tuple[np.ndarray, np.ndarray, SmoothStats]:
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    out_east = np.array(east, copy=True)
    out_north = np.array(north, copy=True)
    n = int(east.size)
    segments = _split_segments(np.asarray(times_ms, dtype=np.float64), east, north, config)
    hampel_mask_total = np.zeros(n, dtype=bool)

    for segment in segments:
        idx = np.arange(segment.start, segment.stop)
        if idx.size < config.min_segment_points:
            continue
        seg_e = east[idx]
        seg_n = north[idx]
        med_e, med_n = _rolling_median_xy(seg_e, seg_n, config.median_window)
        dist = np.hypot(seg_e - med_e, seg_n - med_n)
        local_scale = np.zeros_like(dist)
        half = _positive_odd(config.median_window) // 2
        for j in range(idx.size):
            lo, hi = _window_bounds(j, idx.size, half)
            local_dist = np.hypot(seg_e[lo:hi] - med_e[j], seg_n[lo:hi] - med_n[j])
            local_scale[j] = 1.4826 * float(np.nanmedian(local_dist))
        threshold = np.maximum(config.hampel_min_m, config.hampel_sigma * np.maximum(local_scale, 1e-6))
        hampel = dist > threshold
        repaired_e = np.array(seg_e, copy=True)
        repaired_n = np.array(seg_n, copy=True)
        repaired_e[hampel] = med_e[hampel]
        repaired_n[hampel] = med_n[hampel]

        if config.smooth_window > 1 and config.blend != 0.0:
            smooth_e = _smooth_series(repaired_e, config)
            smooth_n = _smooth_series(repaired_n, config)
            target_e = repaired_e + config.blend * (smooth_e - repaired_e)
            target_n = repaired_n + config.blend * (smooth_n - repaired_n)
        else:
            target_e = repaired_e
            target_n = repaired_n

        dx, dy = _apply_cap(target_e - seg_e, target_n - seg_n, config.max_correction_m)
        out_east[idx] = seg_e + dx
        out_north[idx] = seg_n + dy
        hampel_mask_total[idx[hampel]] = True

    correction = np.hypot(out_east - east, out_north - north)
    corrected = correction > 1e-9
    stats = SmoothStats(
        groups=1,
        rows=n,
        segments=len(segments),
        corrected_rows=int(np.count_nonzero(corrected)),
        hampel_rows=int(np.count_nonzero(hampel_mask_total)),
        max_correction_m=float(np.max(correction)) if correction.size else 0.0,
        mean_correction_m=float(np.mean(correction)) if correction.size else 0.0,
        p95_correction_m=float(np.percentile(correction, 95)) if correction.size else 0.0,
    )
    return out_east, out_north, stats


def _group_indices(df: pd.DataFrame, group_column: str | None) -> Iterable[tuple[object, np.ndarray]]:
    if group_column and group_column in df.columns:
        for key, idx in df.groupby(group_column, sort=False).indices.items():
            yield key, np.asarray(idx, dtype=np.int64)
    else:
        yield "all", np.arange(len(df), dtype=np.int64)


def smooth_dataframe(
    df: pd.DataFrame,
    config: SmoothConfig,
    *,
    lat_column: str = "LatitudeDegrees",
    lon_column: str = "LongitudeDegrees",
    time_column: str = "UnixTimeMillis",
    group_column: str | None = "tripId",
) -> tuple[pd.DataFrame, SmoothStats]:
    missing = [name for name in (lat_column, lon_column) if name not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    out = df.copy()
    group_count = 0
    total_segments = 0
    total_corrected = 0
    total_hampel = 0
    corrections: list[np.ndarray] = []

    for _key, idx in _group_indices(df, group_column):
        group_count += 1
        lat = df.iloc[idx][lat_column].to_numpy(dtype=np.float64)
        lon = df.iloc[idx][lon_column].to_numpy(dtype=np.float64)
        if time_column in df.columns:
            times = df.iloc[idx][time_column].to_numpy(dtype=np.float64)
        else:
            times = np.arange(idx.size, dtype=np.float64) * 1000.0
        east, north, lat0, lon0 = latlon_to_local_m(lat, lon)
        smooth_e, smooth_n, stats = smooth_local_xy(times, east, north, config)
        smooth_lat, smooth_lon = local_m_to_latlon(smooth_e, smooth_n, lat0, lon0)
        out.loc[out.index[idx], lat_column] = smooth_lat
        out.loc[out.index[idx], lon_column] = smooth_lon
        total_segments += stats.segments
        total_corrected += stats.corrected_rows
        total_hampel += stats.hampel_rows
        corrections.append(np.hypot(smooth_e - east, smooth_n - north))

    correction_all = np.concatenate(corrections) if corrections else np.zeros(0, dtype=np.float64)
    aggregate = SmoothStats(
        groups=group_count,
        rows=len(df),
        segments=total_segments,
        corrected_rows=total_corrected,
        hampel_rows=total_hampel,
        max_correction_m=float(np.max(correction_all)) if correction_all.size else 0.0,
        mean_correction_m=float(np.mean(correction_all)) if correction_all.size else 0.0,
        p95_correction_m=float(np.percentile(correction_all, 95)) if correction_all.size else 0.0,
    )
    return out, aggregate


def score_dataframe(
    df: pd.DataFrame,
    *,
    lat_column: str = "LatitudeDegrees",
    lon_column: str = "LongitudeDegrees",
    truth_lat_column: str = "GroundTruthLatitudeDegrees",
    truth_lon_column: str = "GroundTruthLongitudeDegrees",
) -> dict[str, float | int] | None:
    required = [lat_column, lon_column, truth_lat_column, truth_lon_column]
    if any(name not in df.columns for name in required):
        return None
    lat = df[lat_column].to_numpy(dtype=np.float64)
    lon = df[lon_column].to_numpy(dtype=np.float64)
    truth_lat = df[truth_lat_column].to_numpy(dtype=np.float64)
    truth_lon = df[truth_lon_column].to_numpy(dtype=np.float64)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(truth_lat) & np.isfinite(truth_lon)
    if not np.any(valid):
        return None
    return gsdc_score_m(haversine_m(lat[valid], lon[valid], truth_lat[valid], truth_lon[valid]))


def _parse_csv_values(raw: str, cast: type) -> list:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def _scan_configs(args: argparse.Namespace) -> list[SmoothConfig]:
    median_windows = _parse_csv_values(args.median_window, int)
    smooth_windows = _parse_csv_values(args.smooth_window, int)
    blends = _parse_csv_values(args.blend, float)
    caps = _parse_csv_values(args.max_correction_m, float)
    kernels = _parse_csv_values(args.smooth_kernel, str)
    gaussian_sigmas = _parse_csv_values(args.gaussian_sigma, float)
    configs: list[SmoothConfig] = []
    for median_window in median_windows:
        for smooth_window in smooth_windows:
            for blend in blends:
                for max_correction in caps:
                    for kernel in kernels:
                        for gaussian_sigma in gaussian_sigmas:
                            configs.append(
                                SmoothConfig(
                                    median_window=median_window,
                                    smooth_window=smooth_window,
                                    blend=blend,
                                    max_correction_m=max_correction,
                                    smooth_kernel=kernel,
                                    gaussian_sigma=gaussian_sigma,
                                    hampel_sigma=args.hampel_sigma,
                                    hampel_min_m=args.hampel_min_m,
                                    segment_gap_ms=args.segment_gap_ms,
                                    segment_step_m=args.segment_step_m,
                                    min_segment_points=args.min_segment_points,
                                ),
                            )
    return configs


def _run_config(df: pd.DataFrame, config: SmoothConfig, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    before_score = score_dataframe(
        df,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
        truth_lat_column=args.truth_lat_column,
        truth_lon_column=args.truth_lon_column,
    )
    out, stats = smooth_dataframe(
        df,
        config,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
        time_column=args.time_column,
        group_column=args.group_column,
    )
    after_score = score_dataframe(
        out,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
        truth_lat_column=args.truth_lat_column,
        truth_lon_column=args.truth_lon_column,
    )
    before_value = None if before_score is None else before_score["score_m"]
    after_value = None if after_score is None else after_score["score_m"]
    summary = {
        "config": asdict(config),
        "stats": asdict(stats),
        "score_before": before_score,
        "score_after": after_score,
        "score_delta_m": (
            None
            if before_value is None or after_value is None
            else float(after_value) - float(before_value)
        ),
    }
    return out, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="submission.csv or bridge_positions.csv")
    parser.add_argument("--output", type=Path, help="optional smoothed CSV output")
    parser.add_argument("--summary", type=Path, help="optional JSON summary output")
    parser.add_argument("--scan", action="store_true", help="scan comma-separated config values and keep best score")
    parser.add_argument("--lat-column", default="LatitudeDegrees")
    parser.add_argument("--lon-column", default="LongitudeDegrees")
    parser.add_argument("--truth-lat-column", default="GroundTruthLatitudeDegrees")
    parser.add_argument("--truth-lon-column", default="GroundTruthLongitudeDegrees")
    parser.add_argument("--time-column", default="UnixTimeMillis")
    parser.add_argument("--group-column", default="tripId")
    parser.add_argument("--median-window", default="7")
    parser.add_argument("--smooth-window", default="5")
    parser.add_argument("--blend", default="0.5")
    parser.add_argument("--max-correction-m", default="3.0")
    parser.add_argument("--smooth-kernel", default="triangular", help="triangular, gaussian, or boxcar")
    parser.add_argument("--gaussian-sigma", default="1.0")
    parser.add_argument("--hampel-sigma", type=float, default=6.0)
    parser.add_argument("--hampel-min-m", type=float, default=8.0)
    parser.add_argument("--segment-gap-ms", type=float, default=3000.0)
    parser.add_argument("--segment-step-m", type=float, default=100.0)
    parser.add_argument("--min-segment-points", type=int, default=5)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    configs = _scan_configs(args)
    if not args.scan and len(configs) != 1:
        raise ValueError("comma-separated config grids require --scan")

    summaries = []
    outputs: list[pd.DataFrame] = []
    for config in configs:
        out, summary = _run_config(df, config, args)
        outputs.append(out)
        summaries.append(summary)

    if args.scan and summaries[0]["score_after"] is not None:
        best_index = min(range(len(summaries)), key=lambda i: float(summaries[i]["score_after"]["score_m"]))
    else:
        best_index = 0

    payload = {
        "input": str(args.input),
        "output": None if args.output is None else str(args.output),
        "best_index": int(best_index),
        "best": summaries[best_index],
        "runs": summaries,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        outputs[best_index].to_csv(args.output, index=False)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload["best"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
