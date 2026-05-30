#!/usr/bin/env python3
"""Apply the GSDC2023 TDCP error-state smoother to an existing submission CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.analyze_gsdc2023_tdcp_displacement import _fallback_data_root  # noqa: E402
from experiments.eval_gsdc2023_tdcp_correction_smoother import (  # noqa: E402
    TdcpSmootherConfig,
    _apply_tdcp_smoother,
    _config_label,
    _ecef_to_lla,
    _lla_to_ecef,
    _tdcp_interval_deltas,
)
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    build_trip_arrays,
)


def _config_from_args(args: argparse.Namespace) -> TdcpSmootherConfig:
    label = args.smoother_config_label
    if label is None:
        label = (
            _config_label(
                args.sigma_anchor_m,
                args.sigma_tdcp_m,
                args.max_condition,
                args.max_postfit_rms_m,
                args.smoother_min_pairs,
                args.max_delta_m,
            )
            + "_on_v8"
        )
    return TdcpSmootherConfig(
        label,
        args.sigma_anchor_m,
        args.sigma_tdcp_m,
        args.max_condition,
        args.max_postfit_rms_m,
        args.smoother_min_pairs,
        args.max_delta_m,
    )


def _build_batch(trip_id: str, args: argparse.Namespace):
    max_epochs = int(args.max_epochs) if int(args.max_epochs) > 0 else 1_000_000_000
    trip = f"test/{trip_id}"
    return build_trip_arrays(
        args.data_root / trip,
        max_epochs=max_epochs,
        start_epoch=0,
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


def _apply_trip(group: pd.DataFrame, args: argparse.Namespace, cfg: TdcpSmootherConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    trip_id = str(group["tripId"].iloc[0])
    out = group.sort_values("UnixTimeMillis").copy()
    stats: dict[str, object] = {
        "tripId": trip_id,
        "config": cfg.label,
        "rows": int(len(out)),
        "rows_changed": 0,
        "valid_tdcp_intervals": 0,
        "status": "pass",
    }
    try:
        batch = _build_batch(trip_id, args)
    except Exception as exc:  # pragma: no cover - diagnostic path
        stats["status"] = f"build_failed:{type(exc).__name__}"
        return out, stats

    times = out["UnixTimeMillis"].to_numpy(dtype=np.float64)
    batch_time_to_idx = {int(round(float(t))): i for i, t in enumerate(batch.times_ms)}
    batch_idx = np.array([batch_time_to_idx.get(int(round(float(t))), -1) for t in times], dtype=np.int64)
    matched = batch_idx >= 0
    if not np.any(matched):
        stats["status"] = "time_mismatch"
        return out, stats
    if batch.tdcp_meas is None or batch.tdcp_weights is None:
        stats["status"] = "no_tdcp"
        return out, stats

    raw_xyz = batch.kaggle_wls
    raw_lat, raw_lon, raw_h = _ecef_to_lla(raw_xyz)
    base_lat_sub = out["LatitudeDegrees"].to_numpy(dtype=np.float64).copy()
    base_lon_sub = out["LongitudeDegrees"].to_numpy(dtype=np.float64).copy()
    base_lat = raw_lat.copy()
    base_lon = raw_lon.copy()
    base_lat[batch_idx[matched]] = base_lat_sub[matched]
    base_lon[batch_idx[matched]] = base_lon_sub[matched]
    base_xyz = _lla_to_ecef(base_lat, base_lon, raw_h)

    dpos, interval_valid, quality = _tdcp_interval_deltas(batch, args)
    n = min(base_xyz.shape[0], raw_xyz.shape[0], dpos.shape[0] + 1)
    adjusted = dpos[: n - 1].copy()
    adjusted += raw_xyz[1:n] - raw_xyz[: n - 1]
    adjusted -= base_xyz[1:n] - base_xyz[: n - 1]
    lat_corr, lon_corr, valid_count = _apply_tdcp_smoother(
        base_lat[:n],
        base_lon[:n],
        base_xyz[:n],
        adjusted,
        interval_valid[: n - 1],
        quality[: n - 1],
        cfg,
    )
    matched_batch_idx = batch_idx[matched]
    assignable = matched_batch_idx < n
    matched_out_idx = out.index.to_numpy()[matched][assignable]
    matched_batch_idx = matched_batch_idx[assignable]
    new_lat = lat_corr[matched_batch_idx]
    new_lon = lon_corr[matched_batch_idx]
    changed = (
        np.abs(new_lat - base_lat_sub[matched][assignable]) > 1e-12
    ) | (
        np.abs(new_lon - base_lon_sub[matched][assignable]) > 1e-12
    )
    out.loc[matched_out_idx, "LatitudeDegrees"] = new_lat
    out.loc[matched_out_idx, "LongitudeDegrees"] = new_lon
    stats["rows_changed"] = int(np.count_nonzero(changed))
    stats["valid_tdcp_intervals"] = int(valid_count)
    if int(np.count_nonzero(matched)) != int(len(out)):
        stats["status"] = "partial_time_match"
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=_fallback_data_root())
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--min-pairs", type=int, default=6)
    parser.add_argument("--huber-k", type=float, default=2.5)
    parser.add_argument("--huber-iters", type=int, default=5)
    parser.add_argument("--clock-mode", choices=("common", "signal"), default="common")
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-frame-epoch-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument(
        "--tdcp-geometry-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TDCP_GEOMETRY_CORRECTION,
    )
    parser.add_argument("--smoother-config-label", type=str, default=None)
    parser.add_argument("--sigma-anchor-m", type=float, default=2.0)
    parser.add_argument("--sigma-tdcp-m", type=float, default=0.10)
    parser.add_argument("--max-condition", type=float, default=15.0)
    parser.add_argument("--max-postfit-rms-m", type=float, default=0.05)
    parser.add_argument("--smoother-min-pairs", type=int, default=8)
    parser.add_argument("--max-delta-m", type=float, default=5.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    cfg = _config_from_args(args)
    print(
        f"config={cfg.label} sigma_anchor={cfg.sigma_anchor_m:g} sigma_tdcp={cfg.sigma_tdcp_m:g} "
        f"max_condition={cfg.max_condition:g} max_postfit_rms={cfg.max_postfit_rms_m:g} "
        f"smoother_min_pairs={cfg.min_pairs} max_delta={cfg.max_delta_m:g}",
        flush=True,
    )
    outputs: list[pd.DataFrame] = []
    stats_rows: list[dict[str, object]] = []
    for idx, (_, group) in enumerate(df.groupby("tripId", sort=False), start=1):
        trip_out, stats = _apply_trip(group, args, cfg)
        outputs.append(trip_out)
        stats_rows.append(stats)
        print(
            f"[{idx}/{df['tripId'].nunique()}] {stats['tripId']} "
            f"status={stats['status']} changed={stats['rows_changed']}/{stats['rows']} "
            f"valid_tdcp={stats['valid_tdcp_intervals']}",
            flush=True,
        )

    out = pd.concat(outputs, axis=0).sort_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    stats_frame = pd.DataFrame(stats_rows)
    stats_path = args.stats_output or args.output.with_name(args.output.stem + "_tdcp_stats.csv")
    stats_frame.to_csv(stats_path, index=False)
    print(
        f"trips={len(stats_rows)} rows={len(out)} changed={int(stats_frame['rows_changed'].sum())} "
        f"valid_tdcp={int(stats_frame['valid_tdcp_intervals'].sum())}",
        flush=True,
    )
    print(f"wrote: {args.output}")
    print(f"wrote: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
