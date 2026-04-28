#!/usr/bin/env python3
"""Compare raw-bridge solver output with and without MATLAB diagnostics masks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    BridgeConfig,
    BridgeResult,
    DEFAULT_MOTION_SIGMA_M,
    DEFAULT_ROOT,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    POSITION_SOURCES,
    validate_raw_gsdc2023_trip,
)
from experiments.gsdc2023_audit_cli import (  # noqa: E402
    add_data_root_trip_args as _add_data_root_trip_args,
    add_max_epochs_arg as _add_max_epochs_arg,
    add_multi_gnss_arg as _add_multi_gnss_arg,
    add_output_dir_arg as _add_output_dir_arg,
    nonnegative_max_epochs as _nonnegative_max_epochs,
    resolve_trip_dir as _resolve_trip_dir,
    resolved_output_root as _resolved_output_root,
)


_TRAJECTORY_SOURCES = (
    ("selected", "selected_state"),
    ("baseline", "kaggle_wls"),
    ("raw_wls", "raw_wls"),
    ("fgo", "fgo_state"),
)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _metric_value(metrics: dict | None, key: str) -> float | None:
    if not metrics:
        return None
    return _finite_float(metrics.get(key))


def _metrics_row(case: str, result: BridgeResult) -> dict[str, object]:
    payload = result.metrics_payload()
    selected_metrics = payload.get("selected_metrics") or {}
    baseline_metrics = payload.get("kaggle_wls_metrics") or {}
    raw_metrics = payload.get("raw_wls_metrics") or {}
    fgo_metrics = payload.get("fgo_metrics") or {}
    return {
        "case": case,
        "trip": result.trip,
        "n_epochs": int(payload["n_epochs"]),
        "max_sats": int(payload["max_sats"]),
        "selected_source_mode": payload["selected_source_mode"],
        "selected_mse_pr": _finite_float(payload.get("selected_mse_pr")),
        "baseline_mse_pr": _finite_float(payload.get("baseline_mse_pr")),
        "raw_wls_mse_pr": _finite_float(payload.get("raw_wls_mse_pr")),
        "fgo_mse_pr": _finite_float(payload.get("fgo_mse_pr")),
        "selected_score_m": _finite_float(payload.get("selected_score_m")),
        "baseline_score_m": _finite_float(payload.get("kaggle_wls_score_m")),
        "raw_wls_score_m": _finite_float(payload.get("raw_wls_score_m")),
        "fgo_score_m": _finite_float(payload.get("fgo_score_m")),
        "selected_rms_2d_m": _metric_value(selected_metrics, "rms_2d_m"),
        "baseline_rms_2d_m": _metric_value(baseline_metrics, "rms_2d_m"),
        "raw_wls_rms_2d_m": _metric_value(raw_metrics, "rms_2d_m"),
        "fgo_rms_2d_m": _metric_value(fgo_metrics, "rms_2d_m"),
        "failed_chunks": int(payload["failed_chunks"]),
        "fgo_iters": int(payload["fgo_iters"]),
        "observation_mask_applied": bool(payload["observation_mask_applied"]),
        "observation_mask_count": int(payload["observation_mask_count"]),
        "residual_mask_count": int(payload["residual_mask_count"]),
        "doppler_residual_mask_count": int(payload["doppler_residual_mask_count"]),
        "pseudorange_doppler_mask_count": int(payload["pseudorange_doppler_mask_count"]),
        "tdcp_consistency_mask_count": int(payload["tdcp_consistency_mask_count"]),
        "tdcp_weight_scale": _finite_float(payload.get("tdcp_weight_scale")),
        "tdcp_geometry_correction_applied": bool(payload.get("tdcp_geometry_correction_applied", False)),
        "tdcp_geometry_correction_count": int(payload.get("tdcp_geometry_correction_count", 0)),
        "dual_frequency": bool(payload["dual_frequency"]),
        "graph_relative_height": bool(payload["graph_relative_height"]),
    }


def _state_by_time(result: BridgeResult, attr: str) -> dict[int, np.ndarray]:
    state = np.asarray(getattr(result, attr), dtype=np.float64)
    times = np.asarray(result.times_ms, dtype=np.float64)
    return {
        int(round(float(time_ms))): state[idx, :3].copy()
        for idx, time_ms in enumerate(times)
        if idx < state.shape[0] and np.all(np.isfinite(state[idx, :3]))
    }


def _delta_by_epoch(raw: BridgeResult, masked: BridgeResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, attr in _TRAJECTORY_SOURCES:
        raw_by_time = _state_by_time(raw, attr)
        masked_by_time = _state_by_time(masked, attr)
        common_times = sorted(set(raw_by_time) & set(masked_by_time))
        for time_ms in common_times:
            raw_xyz = raw_by_time[time_ms]
            masked_xyz = masked_by_time[time_ms]
            rows.append(
                {
                    "source": source,
                    "utcTimeMillis": int(time_ms),
                    "delta_m": float(np.linalg.norm(masked_xyz - raw_xyz)),
                    "raw_x_ecef_m": float(raw_xyz[0]),
                    "raw_y_ecef_m": float(raw_xyz[1]),
                    "raw_z_ecef_m": float(raw_xyz[2]),
                    "masked_x_ecef_m": float(masked_xyz[0]),
                    "masked_y_ecef_m": float(masked_xyz[1]),
                    "masked_z_ecef_m": float(masked_xyz[2]),
                },
            )
    return pd.DataFrame(rows)


def _delta_stats(epoch_deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, _attr in _TRAJECTORY_SOURCES:
        sub = epoch_deltas[epoch_deltas["source"] == source] if not epoch_deltas.empty else pd.DataFrame()
        if sub.empty:
            rows.append(
                {
                    "source": source,
                    "common_epochs": 0,
                    "mean_delta_m": None,
                    "median_delta_m": None,
                    "p95_delta_m": None,
                    "max_delta_m": None,
                    "max_delta_utcTimeMillis": None,
                },
            )
            continue
        deltas = sub["delta_m"].to_numpy(dtype=np.float64)
        max_idx = int(np.argmax(deltas))
        rows.append(
            {
                "source": source,
                "common_epochs": int(deltas.size),
                "mean_delta_m": float(np.mean(deltas)),
                "median_delta_m": float(np.median(deltas)),
                "p95_delta_m": float(np.percentile(deltas, 95)),
                "max_delta_m": float(np.max(deltas)),
                "max_delta_utcTimeMillis": int(sub.iloc[max_idx]["utcTimeMillis"]),
            },
        )
    return pd.DataFrame(rows)


def _payload_delta(masked_payload: dict[str, object], raw_payload: dict[str, object], key: str) -> float | None:
    masked_value = _finite_float(masked_payload.get(key))
    raw_value = _finite_float(raw_payload.get(key))
    if masked_value is None or raw_value is None:
        return None
    return float(masked_value - raw_value)


def compare_solver_with_diagnostics_mask(
    data_root: Path,
    trip: str,
    *,
    diagnostics_path: Path | None,
    max_epochs: int,
    start_epoch: int,
    config: BridgeConfig,
    raw_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_result = validate_raw_gsdc2023_trip(
        data_root,
        trip,
        max_epochs=max_epochs,
        start_epoch=start_epoch,
        config=config,
    )
    if raw_only:
        metrics = pd.DataFrame([_metrics_row("raw_bridge", raw_result)])
        empty_deltas = pd.DataFrame()
        raw_payload = raw_result.metrics_payload()
        summary = {
            "data_root": str(data_root),
            "trip": trip,
            "diagnostics_path": None,
            "max_epochs": int(max_epochs),
            "start_epoch": int(start_epoch),
            "raw_only": True,
            "config": {
                "position_source": config.position_source,
                "motion_sigma_m": float(config.motion_sigma_m),
                "clock_drift_sigma_m": float(config.clock_drift_sigma_m),
                "fgo_iters": int(config.fgo_iters),
                "chunk_epochs": int(config.chunk_epochs),
                "signal_type": config.signal_type,
                "constellation_type": int(config.constellation_type),
                "weight_mode": config.weight_mode,
                "use_vd": bool(config.use_vd),
                "multi_gnss": bool(config.multi_gnss),
                "tdcp_enabled": bool(config.tdcp_enabled),
                "tdcp_consistency_threshold_m": float(config.tdcp_consistency_threshold_m),
                "tdcp_weight_scale": float(config.tdcp_weight_scale),
                "tdcp_geometry_correction": bool(config.tdcp_geometry_correction),
                "apply_observation_mask": bool(config.apply_observation_mask),
                "pseudorange_residual_mask_m": float(config.pseudorange_residual_mask_m),
                "pseudorange_residual_mask_l5_m": (
                    float(config.pseudorange_residual_mask_l5_m)
                    if config.pseudorange_residual_mask_l5_m is not None
                    else None
                ),
                "doppler_residual_mask_mps": float(config.doppler_residual_mask_mps),
                "pseudorange_doppler_mask_m": float(config.pseudorange_doppler_mask_m),
                "dual_frequency": bool(config.dual_frequency),
            },
            "raw_bridge": raw_payload,
            "diagnostics_mask": None,
            "delta": {},
            "trajectory_delta": [],
            "largest_epoch_deltas": [],
        }
        return metrics, empty_deltas, empty_deltas, summary

    if diagnostics_path is None:
        raise ValueError("diagnostics_path is required unless raw_only=True")
    masked_result = validate_raw_gsdc2023_trip(
        data_root,
        trip,
        max_epochs=max_epochs,
        start_epoch=start_epoch,
        config=replace(config, matlab_residual_diagnostics_mask_path=diagnostics_path),
    )
    metrics = pd.DataFrame(
        [
            _metrics_row("raw_bridge", raw_result),
            _metrics_row("diagnostics_mask", masked_result),
        ],
    )
    epoch_deltas = _delta_by_epoch(raw_result, masked_result)
    deltas = _delta_stats(epoch_deltas)

    raw_payload = raw_result.metrics_payload()
    masked_payload = masked_result.metrics_payload()
    summary = {
        "data_root": str(data_root),
        "trip": trip,
        "diagnostics_path": str(diagnostics_path),
        "max_epochs": int(max_epochs),
        "start_epoch": int(start_epoch),
        "config": {
            "position_source": config.position_source,
            "motion_sigma_m": float(config.motion_sigma_m),
            "clock_drift_sigma_m": float(config.clock_drift_sigma_m),
            "fgo_iters": int(config.fgo_iters),
            "chunk_epochs": int(config.chunk_epochs),
            "signal_type": config.signal_type,
            "constellation_type": int(config.constellation_type),
            "weight_mode": config.weight_mode,
            "use_vd": bool(config.use_vd),
            "multi_gnss": bool(config.multi_gnss),
            "tdcp_enabled": bool(config.tdcp_enabled),
            "tdcp_consistency_threshold_m": float(config.tdcp_consistency_threshold_m),
            "tdcp_weight_scale": float(config.tdcp_weight_scale),
            "tdcp_geometry_correction": bool(config.tdcp_geometry_correction),
            "apply_observation_mask": bool(config.apply_observation_mask),
            "pseudorange_residual_mask_m": float(config.pseudorange_residual_mask_m),
            "pseudorange_residual_mask_l5_m": (
                float(config.pseudorange_residual_mask_l5_m)
                if config.pseudorange_residual_mask_l5_m is not None
                else None
            ),
            "doppler_residual_mask_mps": float(config.doppler_residual_mask_mps),
            "pseudorange_doppler_mask_m": float(config.pseudorange_doppler_mask_m),
            "dual_frequency": bool(config.dual_frequency),
        },
        "raw_only": False,
        "raw_bridge": raw_payload,
        "diagnostics_mask": masked_payload,
        "delta": {
            "selected_mse_pr": _payload_delta(masked_payload, raw_payload, "selected_mse_pr"),
            "baseline_mse_pr": _payload_delta(masked_payload, raw_payload, "baseline_mse_pr"),
            "raw_wls_mse_pr": _payload_delta(masked_payload, raw_payload, "raw_wls_mse_pr"),
            "fgo_mse_pr": _payload_delta(masked_payload, raw_payload, "fgo_mse_pr"),
            "selected_score_m": _payload_delta(masked_payload, raw_payload, "selected_score_m"),
            "baseline_score_m": _payload_delta(masked_payload, raw_payload, "kaggle_wls_score_m"),
            "raw_wls_score_m": _payload_delta(masked_payload, raw_payload, "raw_wls_score_m"),
            "fgo_score_m": _payload_delta(masked_payload, raw_payload, "fgo_score_m"),
        },
        "trajectory_delta": deltas.to_dict(orient="records"),
        "largest_epoch_deltas": (
            epoch_deltas.sort_values("delta_m", ascending=False).head(10).to_dict(orient="records")
            if not epoch_deltas.empty
            else []
        ),
    }
    return metrics, deltas, epoch_deltas, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_trip_args(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--diagnostics", type=Path, default=None, help="phone_data_residual_diagnostics.csv")
    _add_max_epochs_arg(parser, default=200, help_text="0 means all usable epochs")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--motion-sigma-m", type=float, default=DEFAULT_MOTION_SIGMA_M)
    parser.add_argument("--fgo-iters", type=int, default=8)
    parser.add_argument("--clock-drift-sigma-m", type=float, default=1.0)
    parser.add_argument("--position-source", choices=POSITION_SOURCES, default="fgo")
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--signal-type", type=str, default="GPS_L1_CA")
    parser.add_argument("--constellation-type", type=int, default=1)
    parser.add_argument("--weight-mode", choices=("sin2el", "cn0"), default="sin2el")
    parser.add_argument("--gated-threshold", type=float, default=GATED_BASELINE_THRESHOLD_DEFAULT)
    parser.add_argument("--vd", action=argparse.BooleanOptionalAction, default=True)
    _add_multi_gnss_arg(parser)
    parser.add_argument("--tdcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument(
        "--tdcp-geometry-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TDCP_GEOMETRY_CORRECTION,
    )
    parser.add_argument("--observation-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-min-cn0-dbhz", type=float, default=OBS_MASK_MIN_CN0_DBHZ)
    parser.add_argument("--observation-min-elevation-deg", type=float, default=OBS_MASK_MIN_ELEVATION_DEG)
    parser.add_argument("--pseudorange-residual-mask-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_M)
    parser.add_argument("--pseudorange-residual-mask-l5-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_L5_M)
    parser.add_argument("--doppler-residual-mask-mps", type=float, default=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS)
    parser.add_argument("--pseudorange-doppler-mask-m", type=float, default=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--raw-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run only the raw bridge case; useful when diagnostics CSV is unavailable",
    )
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    trip_dir = _resolve_trip_dir(args)
    diagnostics_path = args.diagnostics or (trip_dir / "phone_data_residual_diagnostics.csv")
    if not args.raw_only and not diagnostics_path.is_file():
        raise FileNotFoundError(f"diagnostics CSV not found: {diagnostics_path}")

    config = BridgeConfig(
        motion_sigma_m=args.motion_sigma_m,
        clock_drift_sigma_m=args.clock_drift_sigma_m,
        fgo_iters=args.fgo_iters,
        signal_type=args.signal_type,
        constellation_type=args.constellation_type,
        weight_mode=args.weight_mode,
        position_source=args.position_source,
        chunk_epochs=max(args.chunk_epochs, 0),
        gated_baseline_threshold=args.gated_threshold,
        use_vd=args.vd,
        multi_gnss=args.multi_gnss,
        tdcp_enabled=args.tdcp,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        apply_observation_mask=args.observation_mask,
        observation_min_cn0_dbhz=args.observation_min_cn0_dbhz,
        observation_min_elevation_deg=args.observation_min_elevation_deg,
        pseudorange_residual_mask_m=args.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=args.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=args.doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=args.pseudorange_doppler_mask_m,
        dual_frequency=args.dual_frequency,
    )
    metrics, deltas, epoch_deltas, summary = compare_solver_with_diagnostics_mask(
        data_root,
        args.trip,
        diagnostics_path=diagnostics_path,
        max_epochs=_nonnegative_max_epochs(args),
        start_epoch=max(args.start_epoch, 0),
        config=config,
        raw_only=args.raw_only,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _resolved_output_root(args) / f"gsdc2023_diagnostics_mask_solver_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics_by_case.csv", index=False)
    deltas.to_csv(out_dir / "trajectory_delta.csv", index=False)
    epoch_deltas.to_csv(out_dir / "trajectory_delta_by_epoch.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"comparison_dir={out_dir}")


if __name__ == "__main__":
    main()
