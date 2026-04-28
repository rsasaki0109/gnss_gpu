#!/usr/bin/env python3
"""Export epoch-level GSDC2023 bridge error and residual diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.evaluate import compute_metrics, ecef_errors_2d_3d
from experiments.gsdc2023_raw_bridge import (
    BridgeConfig,
    DEFAULT_MOTION_SIGMA_M,
    DEFAULT_ROOT,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    FACTOR_DT_MAX_S,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    POSITION_SOURCES,
    build_trip_arrays,
    fit_state_with_clock_bias,
    solve_trip,
)
from experiments.gsdc2023_output import score_from_metrics
from experiments.gsdc2023_validation_context import max_epochs_for_build


SOURCE_COLUMNS = ("baseline", "raw_wls", "fgo", "selected")


def _source_states(result) -> dict[str, np.ndarray]:
    return {
        "baseline": np.asarray(result.kaggle_wls, dtype=np.float64).reshape(-1, 3),
        "raw_wls": np.asarray(result.raw_wls, dtype=np.float64),
        "fgo": np.asarray(result.fgo_state, dtype=np.float64),
        "selected": np.asarray(result.selected_state, dtype=np.float64),
    }


def _source_pr_wmse(batch, states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for source, state in states.items():
        _, _, _, per_epoch = fit_state_with_clock_bias(
            batch.sat_ecef,
            batch.pseudorange,
            batch.weights,
            np.asarray(state, dtype=np.float64)[:, :3],
            sys_kind=batch.sys_kind,
            n_clock=batch.n_clock,
        )
        out[source] = np.asarray(per_epoch, dtype=np.float64)
    return out


def _source_errors(
    states: dict[str, np.ndarray],
    truth: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_epoch = next(iter(states.values())).shape[0]
    if truth is None:
        nan = np.full(n_epoch, np.nan, dtype=np.float64)
        return ({source: nan.copy() for source in states}, {source: nan.copy() for source in states})
    err_2d: dict[str, np.ndarray] = {}
    err_3d: dict[str, np.ndarray] = {}
    for source, state in states.items():
        e2d, e3d = ecef_errors_2d_3d(np.asarray(state, dtype=np.float64)[:, :3], truth)
        err_2d[source] = e2d
        err_3d[source] = e3d
    return err_2d, err_3d


def _source_baseline_gap(states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    baseline = states["baseline"][:, :3]
    return {
        source: np.linalg.norm(np.asarray(state, dtype=np.float64)[:, :3] - baseline, axis=1)
        for source, state in states.items()
    }


def epoch_diagnostics_frame(result, batch) -> pd.DataFrame:
    states = _source_states(result)
    pr_wmse = _source_pr_wmse(batch, states)
    err_2d, err_3d = _source_errors(states, result.truth)
    baseline_gap = _source_baseline_gap(states)

    rows: dict[str, object] = {
        "epoch": np.arange(result.n_epochs, dtype=np.int64),
        "UnixTimeMillis": result.times_ms.astype(np.int64),
        "selected_source": result.selected_sources.astype(str),
    }
    for source in SOURCE_COLUMNS:
        rows[f"{source}_pr_wmse_m2"] = pr_wmse[source]
        rows[f"{source}_error_2d_m"] = err_2d[source]
        rows[f"{source}_error_3d_m"] = err_3d[source]
        rows[f"{source}_baseline_gap_m"] = baseline_gap[source]
    rows["fgo_minus_baseline_error_2d_m"] = err_2d["fgo"] - err_2d["baseline"]
    rows["fgo_minus_baseline_pr_wmse_m2"] = pr_wmse["fgo"] - pr_wmse["baseline"]
    rows["raw_minus_baseline_pr_wmse_m2"] = pr_wmse["raw_wls"] - pr_wmse["baseline"]
    return pd.DataFrame(rows)


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return None
    if float(np.std(x[finite])) == 0.0 or float(np.std(y[finite])) == 0.0:
        return None
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _score_from_error(error_2d_m: np.ndarray) -> float | None:
    error_2d_m = np.asarray(error_2d_m, dtype=np.float64).reshape(-1)
    finite = error_2d_m[np.isfinite(error_2d_m)]
    if finite.size == 0:
        return None
    return float(0.5 * (np.percentile(finite, 50) + np.percentile(finite, 95)))


def chunk_diagnostics_frame(result, epoch_frame: pd.DataFrame) -> pd.DataFrame:
    records = result.chunk_selection_records or [
        {"start_epoch": 0, "end_epoch": int(result.n_epochs), "auto_source": "", "gated_source": ""}
    ]
    rows: list[dict[str, object]] = []
    for record in records:
        start = int(record.get("start_epoch", 0))
        end = int(record.get("end_epoch", result.n_epochs))
        chunk = epoch_frame.iloc[start:end]
        row: dict[str, object] = {
            "start_epoch": start,
            "end_epoch": end,
            "n_epochs": int(len(chunk)),
            "auto_source": str(record.get("auto_source", "")),
            "gated_source": str(record.get("gated_source", "")),
        }
        source_scores: dict[str, float | None] = {}
        for source in SOURCE_COLUMNS:
            score = _score_from_error(chunk[f"{source}_error_2d_m"].to_numpy())
            source_scores[source] = score
            row[f"{source}_score_m"] = score
            row[f"{source}_rms2d_m"] = float(np.sqrt(np.nanmean(chunk[f"{source}_error_2d_m"].to_numpy() ** 2)))
            row[f"{source}_mean_pr_wmse_m2"] = float(np.nanmean(chunk[f"{source}_pr_wmse_m2"].to_numpy()))
            row[f"{source}_p95_baseline_gap_m"] = float(np.nanpercentile(chunk[f"{source}_baseline_gap_m"].to_numpy(), 95))
        finite_scores = {name: score for name, score in source_scores.items() if score is not None}
        row["oracle_source"] = min(finite_scores, key=finite_scores.get) if finite_scores else ""
        row["fgo_minus_baseline_score_m"] = (
            float(source_scores["fgo"] - source_scores["baseline"])
            if source_scores["fgo"] is not None and source_scores["baseline"] is not None
            else None
        )
        candidates = record.get("candidates", {})
        if isinstance(candidates, dict):
            for source in ("baseline", "raw_wls", "fgo", "fgo_no_tdcp"):
                quality = candidates.get(source)
                if isinstance(quality, dict):
                    row[f"{source}_candidate_mse_pr"] = quality.get("mse_pr")
                    row[f"{source}_candidate_quality_score"] = quality.get("quality_score")
                    row[f"{source}_candidate_gap_p95_m"] = quality.get("baseline_gap_p95_m")
        rows.append(row)
    return pd.DataFrame(rows)


def summary_payload(result, epoch_frame: pd.DataFrame, chunk_frame: pd.DataFrame) -> dict[str, object]:
    metrics = result.metrics_payload()
    fgo_wmse = epoch_frame["fgo_pr_wmse_m2"].to_numpy()
    fgo_err = epoch_frame["fgo_error_2d_m"].to_numpy()
    fgo_err_delta = epoch_frame["fgo_minus_baseline_error_2d_m"].to_numpy()
    fgo_wmse_delta = epoch_frame["fgo_minus_baseline_pr_wmse_m2"].to_numpy()
    oracle_counts = (
        chunk_frame["oracle_source"].value_counts(dropna=False).astype(int).to_dict()
        if "oracle_source" in chunk_frame
        else {}
    )
    return {
        "trip": result.trip,
        "n_epochs": result.n_epochs,
        "selected_source_mode": result.selected_source_mode,
        "selected_source_counts": metrics.get("selected_source_counts"),
        "scores_m": {
            "selected": metrics.get("selected_score_m"),
            "baseline": metrics.get("kaggle_wls_score_m"),
            "raw_wls": metrics.get("raw_wls_score_m"),
            "fgo": metrics.get("fgo_score_m"),
        },
        "mse_pr": {
            "selected": metrics.get("selected_mse_pr"),
            "baseline": metrics.get("baseline_mse_pr"),
            "raw_wls": metrics.get("raw_wls_mse_pr"),
            "fgo": metrics.get("fgo_mse_pr"),
        },
        "correlation": {
            "fgo_error_2d_vs_fgo_pr_wmse": _finite_corr(fgo_err, fgo_wmse),
            "fgo_minus_baseline_error_2d_vs_fgo_minus_baseline_pr_wmse": _finite_corr(
                fgo_err_delta,
                fgo_wmse_delta,
            ),
        },
        "oracle_chunk_source_counts": oracle_counts,
        "worst_fgo_minus_baseline_chunks": (
            chunk_frame.sort_values("fgo_minus_baseline_score_m", ascending=False)
            .head(10)
            .to_dict(orient="records")
            if "fgo_minus_baseline_score_m" in chunk_frame
            else []
        ),
    }


def _build_config(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        motion_sigma_m=args.motion_sigma_m,
        factor_dt_max_s=args.factor_dt_max_s,
        clock_drift_sigma_m=args.clock_drift_sigma_m,
        fgo_iters=args.fgo_iters,
        signal_type=args.signal_type,
        constellation_type=args.constellation_type,
        weight_mode=args.weight_mode,
        position_source=args.position_source,
        chunk_epochs=args.chunk_epochs,
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


def _build_batch(data_root: Path, trip: str, config: BridgeConfig, *, max_epochs: int, start_epoch: int):
    return build_trip_arrays(
        data_root / trip,
        max_epochs=max_epochs_for_build(max_epochs),
        start_epoch=start_epoch,
        constellation_type=config.constellation_type,
        signal_type=config.signal_type,
        weight_mode=config.weight_mode,
        multi_gnss=config.multi_gnss,
        use_tdcp=config.tdcp_enabled,
        tdcp_consistency_threshold_m=config.tdcp_consistency_threshold_m,
        tdcp_weight_scale=config.tdcp_weight_scale,
        tdcp_geometry_correction=config.tdcp_geometry_correction,
        apply_base_correction=config.apply_base_correction,
        data_root=data_root,
        trip=trip,
        apply_observation_mask=config.apply_observation_mask,
        observation_min_cn0_dbhz=config.observation_min_cn0_dbhz,
        observation_min_elevation_deg=config.observation_min_elevation_deg,
        pseudorange_residual_mask_m=config.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=config.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=config.doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=config.pseudorange_doppler_mask_m,
        dual_frequency=config.dual_frequency,
        factor_dt_max_s=config.factor_dt_max_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", required=True, help="relative trip path under data root")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--motion-sigma-m", type=float, default=DEFAULT_MOTION_SIGMA_M)
    parser.add_argument("--factor-dt-max-s", type=float, default=FACTOR_DT_MAX_S)
    parser.add_argument("--fgo-iters", type=int, default=8)
    parser.add_argument("--clock-drift-sigma-m", type=float, default=1.0)
    parser.add_argument("--signal-type", type=str, default="GPS_L1_CA")
    parser.add_argument("--constellation-type", type=int, default=1)
    parser.add_argument("--weight-mode", choices=("sin2el", "cn0"), default="sin2el")
    parser.add_argument("--position-source", choices=POSITION_SOURCES, default="gated")
    parser.add_argument("--chunk-epochs", type=int, default=100)
    parser.add_argument("--gated-threshold", type=float, default=GATED_BASELINE_THRESHOLD_DEFAULT)
    parser.add_argument("--vd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument(
        "--tdcp-geometry-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TDCP_GEOMETRY_CORRECTION,
    )
    parser.add_argument("--observation-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--observation-min-cn0-dbhz", type=float, default=OBS_MASK_MIN_CN0_DBHZ)
    parser.add_argument("--observation-min-elevation-deg", type=float, default=OBS_MASK_MIN_ELEVATION_DEG)
    parser.add_argument("--pseudorange-residual-mask-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_M)
    parser.add_argument("--pseudorange-residual-mask-l5-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_L5_M)
    parser.add_argument("--doppler-residual-mask-mps", type=float, default=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS)
    parser.add_argument("--pseudorange-doppler-mask-m", type=float, default=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = _build_config(args)
    batch = _build_batch(
        args.data_root,
        args.trip,
        config,
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
    )
    result = solve_trip(args.trip, batch, config)

    epoch_frame = epoch_diagnostics_frame(result, batch)
    chunk_frame = chunk_diagnostics_frame(result, epoch_frame)
    payload = summary_payload(result, epoch_frame, chunk_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    epoch_frame.to_csv(args.output_dir / "epoch_diagnostics.csv", index=False)
    chunk_frame.to_csv(args.output_dir / "chunk_diagnostics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["scores_m"], indent=2))
    print(f"epoch_diagnostics={args.output_dir / 'epoch_diagnostics.csv'}")
    print(f"chunk_diagnostics={args.output_dir / 'chunk_diagnostics.csv'}")
    print(f"summary={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
