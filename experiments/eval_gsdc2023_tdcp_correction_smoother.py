#!/usr/bin/env python3
"""Evaluate a TDCP error-state correction smoother on GSDC2023 train trips.

The bridge applies TDCP geometry correction against ``kaggle_wls``.  The TDCP
interval solve therefore estimates the *difference* between consecutive WLS
position errors, not the vehicle's absolute displacement.  This script turns
those interval deltas into a conservative horizontal correction trajectory:

    min_c  sum_i ||c_i||^2 / sigma_anchor^2
         + sum_i ||(c_{i+1} - c_i) - d_i||^2 / sigma_tdcp^2

where ``c_i`` is the 2D correction added to WLS in local EN metres.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.analyze_gsdc2023_tdcp_displacement import _estimate_interval, _fallback_data_root  # noqa: E402
from experiments.eval_gsdc2023_ct_rbpf_fgo import discover_train_trips  # noqa: E402
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    build_trip_arrays,
)
from experiments.postprocess_gsdc2023_submission_accel_smooth import smooth_axis_by_accel  # noqa: E402
from experiments.postprocess_gsdc2023_submission_hampel import hampel_filter_1d  # noqa: E402
from experiments.postprocess_gsdc2023_submission_heading import smooth_heading_outliers  # noqa: E402
from experiments.postprocess_gsdc2023_submission_kalman import rts_smooth_1d  # noqa: E402
from experiments.postprocess_gsdc2023_submission_stop_snap import detect_stationary_runs  # noqa: E402
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m, latlon_to_local_m, local_m_to_latlon  # noqa: E402


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


@dataclass(frozen=True)
class TdcpSmootherConfig:
    label: str
    sigma_anchor_m: float
    sigma_tdcp_m: float
    max_condition: float
    max_postfit_rms_m: float
    min_pairs: int
    max_delta_m: float


def _row_with_score(
    *,
    trip: str,
    config: str,
    valid_tdcp_intervals: int,
    score: dict[str, float | int],
    smoother_config: TdcpSmootherConfig | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        **score,
        "trip": trip,
        "config": config,
        "valid_tdcp_intervals": int(valid_tdcp_intervals),
    }
    if smoother_config is not None:
        row.update(
            {
                "sigma_anchor_m": float(smoother_config.sigma_anchor_m),
                "sigma_tdcp_m": float(smoother_config.sigma_tdcp_m),
                "max_condition": float(smoother_config.max_condition),
                "max_postfit_rms_m": float(smoother_config.max_postfit_rms_m),
                "min_pairs": int(smoother_config.min_pairs),
                "max_delta_m": float(smoother_config.max_delta_m),
            },
        )
    else:
        row.update(
            {
                "sigma_anchor_m": float("nan"),
                "sigma_tdcp_m": float("nan"),
                "max_condition": float("nan"),
                "max_postfit_rms_m": float("nan"),
                "min_pairs": 0,
                "max_delta_m": float("nan"),
            },
        )
    return row


def _ecef_to_lla(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    a = _WGS84_A
    f = _WGS84_F
    b = a * (1.0 - f)
    ep2 = (a * a - b * b) / (b * b)
    lat = np.full(arr.shape[0], np.nan, dtype=np.float64)
    lon = np.full(arr.shape[0], np.nan, dtype=np.float64)
    h = np.full(arr.shape[0], np.nan, dtype=np.float64)
    for i, row in enumerate(arr):
        if not np.isfinite(row).all() or np.linalg.norm(row) <= 1.0:
            continue
        x, y, z = map(float, row)
        p = float(np.hypot(x, y))
        th = np.arctan2(z * a, p * b)
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        lat_i = np.arctan2(z + ep2 * b * sin_th**3, p - _WGS84_E2 * a * cos_th**3)
        lon_i = np.arctan2(y, x)
        n = a / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat_i) ** 2)
        lat[i] = np.degrees(lat_i)
        lon[i] = np.degrees(lon_i)
        h[i] = p / np.cos(lat_i) - n
    return lat, lon, h


def _lla_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(height_m, dtype=np.float64)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    out = np.empty((lat.size, 3), dtype=np.float64)
    out[:, 0] = (n + h) * np.cos(lat) * np.cos(lon)
    out[:, 1] = (n + h) * np.cos(lat) * np.sin(lon)
    out[:, 2] = (n * (1.0 - _WGS84_E2) + h) * np.sin(lat)
    invalid = ~(np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h))
    out[invalid] = np.nan
    return out


def _ecef_to_enu_delta(delta_ecef: np.ndarray, origin_lat_deg: float, origin_lon_deg: float) -> np.ndarray:
    lat = np.radians(float(origin_lat_deg))
    lon = np.radians(float(origin_lon_deg))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return np.asarray(delta_ecef, dtype=np.float64) @ rot.T


def _solve_tridiagonal(diag: np.ndarray, off: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = int(diag.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    a = np.asarray(off, dtype=np.float64).copy()
    b = np.asarray(diag, dtype=np.float64).copy()
    c = np.asarray(off, dtype=np.float64).copy()
    d = np.asarray(rhs, dtype=np.float64).copy()
    for i in range(1, n):
        denom = max(abs(b[i - 1]), 1e-12)
        m = a[i - 1] / denom
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    out = np.zeros(n, dtype=np.float64)
    out[-1] = d[-1] / max(abs(b[-1]), 1e-12)
    for i in range(n - 2, -1, -1):
        out[i] = (d[i] - c[i] * out[i + 1]) / max(abs(b[i]), 1e-12)
    return out


def _solve_correction_axis(delta_meas: np.ndarray, valid: np.ndarray, cfg: TdcpSmootherConfig) -> np.ndarray:
    n_interval = int(delta_meas.size)
    n = n_interval + 1
    prior_w = 1.0 / max(float(cfg.sigma_anchor_m) ** 2, 1e-12)
    tdcp_w = 1.0 / max(float(cfg.sigma_tdcp_m) ** 2, 1e-12)
    diag = np.full(n, prior_w, dtype=np.float64)
    off = np.zeros(max(0, n - 1), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    for i in range(n_interval):
        if not bool(valid[i]) or not np.isfinite(delta_meas[i]):
            continue
        meas = float(np.clip(delta_meas[i], -cfg.max_delta_m, cfg.max_delta_m))
        w = tdcp_w
        diag[i] += w
        diag[i + 1] += w
        off[i] -= w
        rhs[i] -= w * meas
        rhs[i + 1] += w * meas
    return _solve_tridiagonal(diag, off, rhs)


def _apply_v8_chain(lat: np.ndarray, lon: np.ndarray, times_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_s = np.asarray(lat, dtype=np.float64).copy()
    lon_s = np.asarray(lon, dtype=np.float64).copy()
    for _ in range(3):
        lat_s = hampel_filter_1d(lat_s, 21, 2.5, 5e-7)
        lon_s = hampel_filter_1d(lon_s, 21, 2.5, 5e-7)

    east, north, lat0, lon0 = latlon_to_local_m(lat_s, lon_s)
    dt = np.diff(np.asarray(times_ms, dtype=np.float64)) / 1000.0
    for _ in range(2):
        east, _ = smooth_axis_by_accel(east, dt, accel_max=3.0)
        north, _ = smooth_axis_by_accel(north, dt, accel_max=3.0)
    lat_s, lon_s = local_m_to_latlon(east, north, lat0, lon0)

    steps = haversine_m(lat_s[:-1], lon_s[:-1], lat_s[1:], lon_s[1:]) if lat_s.size > 1 else np.empty(0)
    for lo, hi in detect_stationary_runs(steps, move_threshold_m=2.0, min_run_length=10):
        lat_s[lo : hi + 1] = float(np.median(lat_s[lo : hi + 1]))
        lon_s[lo : hi + 1] = float(np.median(lon_s[lo : hi + 1]))

    dt = np.diff(np.asarray(times_ms, dtype=np.float64)) / 1000.0
    lat_s, lon_s, _ = smooth_heading_outliers(lat_s, lon_s, dt, heading_max_dps=45.0)

    east, north, lat0, lon0 = latlon_to_local_m(lat_s, lon_s)
    east = rts_smooth_1d(east, dt, sigma_a=1.0, sigma_z=1.0)
    north = rts_smooth_1d(north, dt, sigma_a=1.0, sigma_z=1.0)
    return local_m_to_latlon(east, north, lat0, lon0)


def _score_latlon(lat: np.ndarray, lon: np.ndarray, truth_lat: np.ndarray, truth_lon: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(truth_lat) & np.isfinite(truth_lon)
    return gsdc_score_m(haversine_m(lat[valid], lon[valid], truth_lat[valid], truth_lon[valid]))


def _tdcp_interval_deltas(batch, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    n_interval = min(batch.tdcp_meas.shape[0], batch.sat_ecef.shape[0] - 1, batch.kaggle_wls.shape[0] - 1)
    dpos = np.full((n_interval, 3), np.nan, dtype=np.float64)
    quality: list[dict[str, float]] = []
    for t in range(n_interval):
        est = _estimate_interval(
            sat_ecef_next=batch.sat_ecef[t + 1],
            tdcp_meas=batch.tdcp_meas[t],
            tdcp_weights=batch.tdcp_weights[t],
            sys_kind_next=(batch.sys_kind[t + 1] if batch.sys_kind is not None else None),
            n_clock=batch.n_clock,
            clock_mode=args.clock_mode,
            reference_ecef_next=batch.kaggle_wls[t + 1],
            min_pairs=args.min_pairs,
            huber_k=args.huber_k,
            max_iter=args.huber_iters,
        )
        if est is None:
            quality.append(
                {
                    "pair_count": 0.0,
                    "postfit_rms_m": float("inf"),
                    "condition_number": float("inf"),
                },
            )
            continue
        dpos[t] = np.asarray(est["dpos"], dtype=np.float64)
        quality.append(
            {
                "pair_count": float(est["pair_count"]),
                "postfit_rms_m": float(est["postfit_rms_m"]),
                "condition_number": float(est["condition_number"]),
            },
        )
    return dpos, np.isfinite(dpos).all(axis=1), quality


def _apply_tdcp_smoother(
    raw_lat: np.ndarray,
    raw_lon: np.ndarray,
    raw_xyz: np.ndarray,
    dpos_ecef: np.ndarray,
    interval_valid: np.ndarray,
    quality: list[dict[str, float]],
    cfg: TdcpSmootherConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    east, north, lat0, lon0 = latlon_to_local_m(raw_lat, raw_lon)
    dpos_enu = _ecef_to_enu_delta(dpos_ecef, lat0, lon0)
    valid = np.asarray(interval_valid, dtype=bool).copy()
    for i, q in enumerate(quality):
        valid[i] = (
            bool(valid[i])
            and float(q["pair_count"]) >= float(cfg.min_pairs)
            and float(q["postfit_rms_m"]) <= float(cfg.max_postfit_rms_m)
            and float(q["condition_number"]) <= float(cfg.max_condition)
        )
    corr_e = _solve_correction_axis(dpos_enu[:, 0], valid, cfg)
    corr_n = _solve_correction_axis(dpos_enu[:, 1], valid, cfg)
    lat_corr, lon_corr = local_m_to_latlon(east + corr_e, north + corr_n, lat0, lon0)
    return lat_corr, lon_corr, int(np.count_nonzero(valid))


def analyze_trip(trip: str, args: argparse.Namespace, configs: list[TdcpSmootherConfig]) -> list[dict[str, object]]:
    max_epochs = int(args.max_epochs) if int(args.max_epochs) > 0 else 1_000_000_000
    batch = build_trip_arrays(
        args.data_root / trip,
        max_epochs=max_epochs,
        start_epoch=args.start_epoch,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        use_tdcp=True,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        data_root=args.data_root,
        trip=trip,
        dual_frequency=args.dual_frequency,
        raw_frame_epoch_window=args.raw_frame_epoch_window,
    )
    n = min(batch.kaggle_wls.shape[0], batch.truth.shape[0])
    times = batch.times_ms[:n]
    raw_xyz = batch.kaggle_wls[:n]
    truth_xyz = batch.truth[:n]
    raw_lat, raw_lon, raw_h = _ecef_to_lla(raw_xyz)
    truth_lat, truth_lon, _ = _ecef_to_lla(truth_xyz)
    v8_lat, v8_lon = _apply_v8_chain(raw_lat, raw_lon, times)

    rows: list[dict[str, object]] = []
    for label, lat, lon, valid_count in [
        ("raw_wls", raw_lat, raw_lon, 0),
        ("v8_chain", v8_lat, v8_lon, 0),
    ]:
        score = _score_latlon(lat, lon, truth_lat, truth_lon)
        rows.append(_row_with_score(trip=trip, config=label, valid_tdcp_intervals=valid_count, score=score))

    if batch.tdcp_meas is None or batch.tdcp_weights is None:
        raw_score = rows[0]
        v8_score = rows[1]
        for cfg in configs:
            rows.append(
                _row_with_score(
                    trip=trip,
                    config=cfg.label,
                    valid_tdcp_intervals=0,
                    score=raw_score,
                    smoother_config=cfg,
                ),
            )
            rows.append(
                _row_with_score(
                    trip=trip,
                    config=cfg.label + "_v8",
                    valid_tdcp_intervals=0,
                    score=v8_score,
                    smoother_config=cfg,
                ),
            )
            rows.append(
                _row_with_score(
                    trip=trip,
                    config=cfg.label + "_on_v8",
                    valid_tdcp_intervals=0,
                    score=v8_score,
                    smoother_config=cfg,
                ),
            )
        return rows

    dpos, interval_valid, quality = _tdcp_interval_deltas(batch, args)
    v8_xyz = _lla_to_ecef(v8_lat, v8_lon, raw_h)
    adjusted_to_v8 = dpos[: n - 1].copy()
    adjusted_to_v8 += raw_xyz[1:n] - raw_xyz[: n - 1]
    adjusted_to_v8 -= v8_xyz[1:n] - v8_xyz[: n - 1]
    for cfg in configs:
        lat_corr, lon_corr, valid_count = _apply_tdcp_smoother(
            raw_lat,
            raw_lon,
            raw_xyz,
            dpos[: n - 1],
            interval_valid[: n - 1],
            quality[: n - 1],
            cfg,
        )
        score = _score_latlon(lat_corr, lon_corr, truth_lat, truth_lon)
        if args.eval_mode == "all":
            rows.append(
                _row_with_score(
                    trip=trip,
                    config=cfg.label,
                    valid_tdcp_intervals=valid_count,
                    score=score,
                    smoother_config=cfg,
                ),
            )
        lat_on_v8, lon_on_v8, valid_count = _apply_tdcp_smoother(
            v8_lat,
            v8_lon,
            v8_xyz,
            adjusted_to_v8,
            interval_valid[: n - 1],
            quality[: n - 1],
            cfg,
        )
        score = _score_latlon(lat_on_v8, lon_on_v8, truth_lat, truth_lon)
        rows.append(
            _row_with_score(
                trip=trip,
                config=cfg.label + "_on_v8",
                valid_tdcp_intervals=valid_count,
                score=score,
                smoother_config=cfg,
            ),
        )
        if args.eval_mode == "all":
            lat_v8, lon_v8 = _apply_v8_chain(lat_corr, lon_corr, times)
            score = _score_latlon(lat_v8, lon_v8, truth_lat, truth_lon)
            rows.append(
                _row_with_score(
                    trip=trip,
                    config=cfg.label + "_v8",
                    valid_tdcp_intervals=valid_count,
                    score=score,
                    smoother_config=cfg,
                ),
            )
    return rows


def _fmt_param(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return text


def _config_label(anchor: float, tdcp: float, cond: float, rms: float, pairs: int, max_delta: float) -> str:
    return (
        f"tdcp_a{_fmt_param(anchor)}_t{_fmt_param(tdcp)}_c{_fmt_param(cond)}_"
        f"r{_fmt_param(rms)}_p{int(pairs)}_d{_fmt_param(max_delta)}"
    )


def _default_configs() -> list[TdcpSmootherConfig]:
    return [
        TdcpSmootherConfig("tdcp_a0p5_t0p05_c10_r0p02_p8", 0.5, 0.05, 10.0, 0.02, 8, 5.0),
        TdcpSmootherConfig("tdcp_a1_t0p05_c10_r0p02_p8", 1.0, 0.05, 10.0, 0.02, 8, 5.0),
        TdcpSmootherConfig("tdcp_a2_t0p05_c10_r0p02_p8", 2.0, 0.05, 10.0, 0.02, 8, 5.0),
        TdcpSmootherConfig("tdcp_a1_t0p10_c10_r0p02_p8", 1.0, 0.10, 10.0, 0.02, 8, 5.0),
        TdcpSmootherConfig("tdcp_a2_t0p10_c15_r0p05_p8", 2.0, 0.10, 15.0, 0.05, 8, 5.0),
    ]


def _best_configs() -> list[TdcpSmootherConfig]:
    return [TdcpSmootherConfig("tdcp_a2_t0p10_c15_r0p05_p8", 2.0, 0.10, 15.0, 0.05, 8, 5.0)]


def _grid_configs() -> list[TdcpSmootherConfig]:
    configs: list[TdcpSmootherConfig] = []
    for anchor in (0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        for tdcp in (0.05, 0.075, 0.10, 0.15, 0.20):
            for cond in (8.0, 10.0, 15.0, 20.0, 30.0):
                for rms in (0.02, 0.035, 0.05, 0.075, 0.10):
                    for pairs in (6, 8, 10):
                        for max_delta in (3.0, 5.0, 8.0):
                            configs.append(
                                TdcpSmootherConfig(
                                    _config_label(anchor, tdcp, cond, rms, pairs, max_delta),
                                    anchor,
                                    tdcp,
                                    cond,
                                    rms,
                                    pairs,
                                    max_delta,
                                ),
                            )
    return configs


def _configs_from_csv(path: Path, limit: int) -> list[TdcpSmootherConfig]:
    frame = pd.read_csv(path)
    configs: list[TdcpSmootherConfig] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        label = str(row["config"])
        if label.endswith("_on_v8"):
            label = label[: -len("_on_v8")]
        if label in seen:
            continue
        seen.add(label)
        configs.append(
            TdcpSmootherConfig(
                label,
                float(row["sigma_anchor_m"]),
                float(row["sigma_tdcp_m"]),
                float(row["max_condition"]),
                float(row["max_postfit_rms_m"]),
                int(row["min_pairs"]),
                float(row["max_delta_m"]),
            ),
        )
        if limit > 0 and len(configs) >= int(limit):
            break
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_fallback_data_root())
    parser.add_argument("--trip", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--min-pairs", type=int, default=6)
    parser.add_argument("--huber-k", type=float, default=2.5)
    parser.add_argument("--huber-iters", type=int, default=5)
    parser.add_argument("--clock-mode", choices=("common", "signal"), default="common")
    parser.add_argument("--preset", choices=("sweep", "best", "grid"), default="sweep")
    parser.add_argument("--eval-mode", choices=("all", "on_v8_only"), default="all")
    parser.add_argument("--config-csv", type=Path, default=None)
    parser.add_argument("--config-limit", type=int, default=0)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-frame-epoch-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument(
        "--tdcp-geometry-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TDCP_GEOMETRY_CORRECTION,
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("experiments/results/gsdc2023_tdcp_correction_smoother"),
    )
    args = parser.parse_args()

    trips = discover_train_trips(args.data_root)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no train trips found")

    if args.config_csv is not None:
        configs = _configs_from_csv(args.config_csv, args.config_limit)
        args.eval_mode = "on_v8_only"
    elif args.preset == "best":
        configs = _best_configs()
    elif args.preset == "grid":
        configs = _grid_configs()
        args.eval_mode = "on_v8_only"
    else:
        configs = _default_configs()
    all_rows: list[dict[str, object]] = []
    for idx, trip in enumerate(trips, start=1):
        rows = analyze_trip(trip, args, configs)
        all_rows.extend(rows)
        best = min(rows, key=lambda r: float(r["score_m"]) if pd.notna(r["score_m"]) else float("inf"))
        base = next((row for row in rows if row["config"] == "v8_chain"), rows[0])
        print(
            f"[{idx}/{len(trips)}] {trip} v8={float(base['score_m']):.3f} "
            f"best={best['config']}:{float(best['score_m']):.3f}",
            flush=True,
        )

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_trip_path = out_prefix.with_name(out_prefix.name + "_per_trip.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    with per_trip_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    frame = pd.DataFrame(all_rows)
    summary = (
        frame.groupby("config", as_index=False)
        .agg(
            trips=("trip", "count"),
            mean_score_m=("score_m", "mean"),
            median_score_m=("score_m", "median"),
            mean_p50_m=("p50_m", "mean"),
            mean_p95_m=("p95_m", "mean"),
            mean_valid_tdcp_intervals=("valid_tdcp_intervals", "mean"),
            sigma_anchor_m=("sigma_anchor_m", "first"),
            sigma_tdcp_m=("sigma_tdcp_m", "first"),
            max_condition=("max_condition", "first"),
            max_postfit_rms_m=("max_postfit_rms_m", "first"),
            min_pairs=("min_pairs", "first"),
            max_delta_m=("max_delta_m", "first"),
        )
        .sort_values("mean_score_m")
    )
    summary.to_csv(summary_path, index=False)
    print("\n" + summary.to_string(index=False), flush=True)
    print(f"wrote: {per_trip_path}")
    print(f"wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
