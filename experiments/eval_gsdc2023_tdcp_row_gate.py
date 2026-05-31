#!/usr/bin/env python3
"""Evaluate row-level aggressive/conservative TDCP gates on GSDC2023 train."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.analyze_gsdc2023_tdcp_displacement import _fallback_data_root  # noqa: E402
from experiments.eval_gsdc2023_ct_rbpf_fgo import discover_train_trips  # noqa: E402
from experiments.eval_gsdc2023_tdcp_correction_smoother import (  # noqa: E402
    TdcpSmootherConfig,
    _apply_tdcp_smoother,
    _apply_v8_chain,
    _ecef_to_lla,
    _lla_to_ecef,
    _score_latlon,
    _tdcp_interval_deltas,
)
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    build_trip_arrays,
)
from experiments.smooth_gsdc2023_submission import haversine_m  # noqa: E402


_A32_PHONES = {"samsunga325g", "samsunga32", "sm-a325f"}
_AGGRESSIVE = TdcpSmootherConfig("aggressive", 4.0, 0.05, 30.0, 0.10, 6, 3.0)
_CONSERVATIVE = TdcpSmootherConfig("conservative", 4.0, 0.05, 15.0, 0.10, 10, 5.0)


def _phone(trip: str) -> str:
    return str(trip).split("/")[-1].lower()


def _is_lax_pixel5(trip: str) -> bool:
    text = str(trip).lower()
    return "us-ca-lax-" in text and _phone(text) == "pixel5"


def _is_february_lax_pixel5(trip: str) -> bool:
    text = str(trip).lower()
    return _is_lax_pixel5(text) and "/2022-02-" in text


def _use_v12_conservative(trip: str) -> bool:
    phone = _phone(trip)
    return phone in _A32_PHONES or _is_february_lax_pixel5(trip)


def _parse_floats(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _build_batch(trip: str, args: argparse.Namespace):
    max_epochs = int(args.max_epochs) if int(args.max_epochs) > 0 else 1_000_000_000
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


def _tdcp_on_v8_latlon(
    batch,
    args: argparse.Namespace,
    cfg: TdcpSmootherConfig,
    v8_lat: np.ndarray,
    v8_lon: np.ndarray,
    v8_xyz: np.ndarray,
    raw_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    n = min(v8_xyz.shape[0], raw_xyz.shape[0])
    if batch.tdcp_meas is None or batch.tdcp_weights is None:
        return v8_lat.copy(), v8_lon.copy(), 0
    dpos, interval_valid, quality = _tdcp_interval_deltas(batch, args)
    adjusted = dpos[: n - 1].copy()
    adjusted += raw_xyz[1:n] - raw_xyz[: n - 1]
    adjusted -= v8_xyz[1:n] - v8_xyz[: n - 1]
    return _apply_tdcp_smoother(
        v8_lat,
        v8_lon,
        v8_xyz,
        adjusted,
        interval_valid[: n - 1],
        quality[: n - 1],
        cfg,
    )


def _score_row(
    *,
    trip: str,
    config: str,
    lat: np.ndarray,
    lon: np.ndarray,
    truth_lat: np.ndarray,
    truth_lon: np.ndarray,
    aggressive_rows: int,
    conservative_rows: int,
    max_ag_cons_delta_m: float,
) -> dict[str, object]:
    score = _score_latlon(lat, lon, truth_lat, truth_lon)
    return {
        **score,
        "trip": trip,
        "config": config,
        "aggressive_rows": int(aggressive_rows),
        "conservative_rows": int(conservative_rows),
        "max_ag_cons_delta_m": float(max_ag_cons_delta_m),
    }


def analyze_trip(trip: str, args: argparse.Namespace, global_thresholds: list[float], adaptive_pairs: list[tuple[float, float]]) -> list[dict[str, object]]:
    batch = _build_batch(trip, args)
    n = min(batch.kaggle_wls.shape[0], batch.truth.shape[0])
    raw_xyz = batch.kaggle_wls[:n]
    truth_xyz = batch.truth[:n]
    times = batch.times_ms[:n]
    raw_lat, raw_lon, raw_h = _ecef_to_lla(raw_xyz)
    truth_lat, truth_lon, _ = _ecef_to_lla(truth_xyz)
    v8_lat, v8_lon = _apply_v8_chain(raw_lat, raw_lon, times)
    v8_xyz = _lla_to_ecef(v8_lat, v8_lon, raw_h)
    ag_lat, ag_lon, _ = _tdcp_on_v8_latlon(batch, args, _AGGRESSIVE, v8_lat, v8_lon, v8_xyz, raw_xyz)
    co_lat, co_lon, _ = _tdcp_on_v8_latlon(batch, args, _CONSERVATIVE, v8_lat, v8_lon, v8_xyz, raw_xyz)
    ag_co_delta = haversine_m(ag_lat, ag_lon, co_lat, co_lon)

    rows: list[dict[str, object]] = []
    rows.append(
        _score_row(
            trip=trip,
            config="v8",
            lat=v8_lat,
            lon=v8_lon,
            truth_lat=truth_lat,
            truth_lon=truth_lon,
            aggressive_rows=0,
            conservative_rows=0,
            max_ag_cons_delta_m=float(np.nanmax(ag_co_delta)),
        ),
    )
    rows.append(
        _score_row(
            trip=trip,
            config="conservative",
            lat=co_lat,
            lon=co_lon,
            truth_lat=truth_lat,
            truth_lon=truth_lon,
            aggressive_rows=0,
            conservative_rows=int(n),
            max_ag_cons_delta_m=float(np.nanmax(ag_co_delta)),
        ),
    )
    rows.append(
        _score_row(
            trip=trip,
            config="aggressive",
            lat=ag_lat,
            lon=ag_lon,
            truth_lat=truth_lat,
            truth_lon=truth_lon,
            aggressive_rows=int(n),
            conservative_rows=0,
            max_ag_cons_delta_m=float(np.nanmax(ag_co_delta)),
        ),
    )

    for threshold in global_thresholds:
        use_aggressive = ag_co_delta <= float(threshold)
        lat = np.where(use_aggressive, ag_lat, co_lat)
        lon = np.where(use_aggressive, ag_lon, co_lon)
        rows.append(
            _score_row(
                trip=trip,
                config=f"global_rowgate_t{threshold:g}",
                lat=lat,
                lon=lon,
                truth_lat=truth_lat,
                truth_lon=truth_lon,
                aggressive_rows=int(np.count_nonzero(use_aggressive)),
                conservative_rows=int(n - np.count_nonzero(use_aggressive)),
                max_ag_cons_delta_m=float(np.nanmax(ag_co_delta)),
            ),
        )

    use_v12_conservative = _use_v12_conservative(trip)
    for lax_threshold, a32_threshold in adaptive_pairs:
        if not use_v12_conservative:
            use_aggressive = np.ones(n, dtype=bool)
        elif _phone(trip) in _A32_PHONES:
            use_aggressive = ag_co_delta <= float(a32_threshold)
        elif _is_february_lax_pixel5(trip):
            use_aggressive = ag_co_delta <= float(lax_threshold)
        else:
            use_aggressive = np.zeros(n, dtype=bool)
        lat = np.where(use_aggressive, ag_lat, co_lat)
        lon = np.where(use_aggressive, ag_lon, co_lon)
        rows.append(
            _score_row(
                trip=trip,
                config=f"adaptive_rowgate_lax{lax_threshold:g}_a32{a32_threshold:g}",
                lat=lat,
                lon=lon,
                truth_lat=truth_lat,
                truth_lon=truth_lon,
                aggressive_rows=int(np.count_nonzero(use_aggressive)),
                conservative_rows=int(n - np.count_nonzero(use_aggressive)),
                max_ag_cons_delta_m=float(np.nanmax(ag_co_delta)),
            ),
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_fallback_data_root())
    parser.add_argument("--trip", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--global-thresholds", default="0.25,0.5,0.75,1,1.5,2,3,5,10")
    parser.add_argument("--adaptive-lax-thresholds", default="0.3,0.5,0.75,1")
    parser.add_argument("--adaptive-a32-thresholds", default="0.5,1,1.5,2,3")
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
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/gsdc2023_tdcp_row_gate"))
    args = parser.parse_args()

    global_thresholds = _parse_floats(args.global_thresholds)
    adaptive_pairs = [
        (lax, a32)
        for lax in _parse_floats(args.adaptive_lax_thresholds)
        for a32 in _parse_floats(args.adaptive_a32_thresholds)
    ]

    trips = discover_train_trips(args.data_root)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no train trips found")

    all_rows: list[dict[str, object]] = []
    for idx, trip in enumerate(trips, start=1):
        rows = analyze_trip(trip, args, global_thresholds, adaptive_pairs)
        all_rows.extend(rows)
        best = min(rows, key=lambda row: float(row["score_m"]))
        v8 = next(row for row in rows if row["config"] == "v8")
        print(
            f"[{idx}/{len(trips)}] {trip} v8={float(v8['score_m']):.3f} "
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

    import pandas as pd

    frame = pd.DataFrame(all_rows)
    base = frame.loc[frame["config"] == "v8", ["trip", "score_m"]].rename(columns={"score_m": "v8_score_m"})
    joined = frame.merge(base, on="trip", how="left")
    joined["delta_vs_v8"] = joined["score_m"] - joined["v8_score_m"]
    summary = (
        joined.groupby("config", as_index=False)
        .agg(
            trips=("trip", "count"),
            mean_score_m=("score_m", "mean"),
            median_score_m=("score_m", "median"),
            mean_delta_vs_v8_m=("delta_vs_v8", "mean"),
            wins=("delta_vs_v8", lambda s: int((s < -1e-9).sum())),
            wash=("delta_vs_v8", lambda s: int((s.abs() <= 1e-9).sum())),
            regressions=("delta_vs_v8", lambda s: int((s > 1e-9).sum())),
            worst_delta_m=("delta_vs_v8", "max"),
            mean_aggressive_rows=("aggressive_rows", "mean"),
            mean_conservative_rows=("conservative_rows", "mean"),
        )
        .sort_values(["mean_score_m", "regressions", "worst_delta_m"])
    )
    summary.to_csv(summary_path, index=False)
    print("\n" + summary.to_string(index=False), flush=True)
    print(f"wrote: {per_trip_path}")
    print(f"wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
