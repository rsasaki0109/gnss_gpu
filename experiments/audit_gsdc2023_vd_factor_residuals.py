#!/usr/bin/env python3
"""Run VD seed Doppler/TDCP residual diagnostics across selected GSDC2023 trips."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.diagnose_gsdc2023_vd_factor_residuals import (  # noqa: E402
    run_vd_factor_residual_diagnosis,
)
from experiments.gsdc2023_audit_cli import (  # noqa: E402
    add_data_root_arg as _add_data_root_arg,
    add_max_epochs_arg as _add_max_epochs_arg,
    add_output_dir_arg as _add_output_dir_arg,
    nonnegative_max_epochs as _nonnegative_max_epochs,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_audit_output import (  # noqa: E402
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    VD_SEED_FACTOR_GUARD_DOPPLER_RMS_MPS,
    VD_SEED_FACTOR_GUARD_MIN_COUNT,
    VD_SEED_FACTOR_GUARD_TDCP_RMS_M,
    _seed_vd_state,
    _vd_seed_factor_guard_enabled_for_phone,
    _vd_seed_doppler_rms,
    _vd_seed_tdcp_rms,
    build_trip_arrays,
    fit_state_with_clock_bias,
    run_wls,
)
from experiments.gsdc2023_clock_state import (  # noqa: E402
    factor_break_mask as _factor_break_mask,
    segment_ranges as _segment_ranges,
)
from experiments.gsdc2023_observation_matrix import repair_baseline_wls  # noqa: E402
from experiments.gsdc2023_solver_context import build_solver_execution_context as _build_solver_execution_context  # noqa: E402
from experiments.gsdc2023_validation_context import max_epochs_for_build  # noqa: E402


DEFAULT_VD_FACTOR_RESIDUAL_TRIPS: tuple[str, ...] = (
    "test/2021-11-05-18-28-us-ca-mtv-m/pixel6pro",
    "test/2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro",
    "test/2023-05-25-17-32-us-ca-pao-j/pixel6pro",
    "train/2021-12-08-20-28-us-ca-lax-c/pixel5",
)

DiagnoseFn = Callable[[object], dict[str, object]]


def _guard_thresholds() -> dict[str, float | int]:
    return {
        "min_count": int(VD_SEED_FACTOR_GUARD_MIN_COUNT),
        "doppler_weighted_rms_mps": float(VD_SEED_FACTOR_GUARD_DOPPLER_RMS_MPS),
        "tdcp_weighted_rms_m": float(VD_SEED_FACTOR_GUARD_TDCP_RMS_M),
    }


def _guard_reason(
    *,
    doppler_count: int,
    doppler_rms: float,
    tdcp_count: int,
    tdcp_rms: float,
) -> str:
    if doppler_count >= VD_SEED_FACTOR_GUARD_MIN_COUNT and doppler_rms > VD_SEED_FACTOR_GUARD_DOPPLER_RMS_MPS:
        return "doppler"
    if tdcp_count >= VD_SEED_FACTOR_GUARD_MIN_COUNT and tdcp_rms > VD_SEED_FACTOR_GUARD_TDCP_RMS_M:
        return "tdcp"
    return ""


def _metric(summary: dict[str, object], section: str, name: str) -> float:
    value = (summary.get(section) or {}).get(name) if isinstance(summary.get(section), dict) else None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _count(summary: dict[str, object], section: str, name: str) -> int:
    value = (summary.get(section) or {}).get(name) if isinstance(summary.get(section), dict) else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _guard_residual_failure_rows(trip_summary: pd.DataFrame) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for row in trip_summary.itertuples(index=False):
        reason = _guard_reason(
            doppler_count=int(row.doppler_count),
            doppler_rms=float(row.doppler_weighted_rms_mps),
            tdcp_count=int(row.tdcp_count),
            tdcp_rms=float(row.tdcp_weighted_rms_m),
        )
        if not reason:
            continue
        phone = Path(str(row.trip)).name
        failures.append(
            {
                "trip": str(row.trip),
                "phone": phone,
                "phone_guard_enabled": bool(_vd_seed_factor_guard_enabled_for_phone(phone)),
                "reject_reason": reason,
                "doppler_count": int(row.doppler_count),
                "doppler_weighted_rms_mps": float(row.doppler_weighted_rms_mps),
                "tdcp_count": int(row.tdcp_count),
                "tdcp_weighted_rms_m": float(row.tdcp_weighted_rms_m),
            },
        )
    return failures


def _reason_counts(segment_frame: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    if segment_frame.empty or not bool(mask.any()):
        return {}
    reasons = segment_frame.loc[mask, "reject_reason"].fillna("").astype(str)
    reasons = reasons[reasons.ne("")]
    return {str(reason): int(count) for reason, count in reasons.value_counts().sort_index().items()}


def guard_segment_summary_payload(segment_frame: pd.DataFrame) -> dict[str, object]:
    if segment_frame.empty:
        return {
            "guard_segment_count": 0,
            "guard_threshold_rejected_segment_count": 0,
            "guard_threshold_rejected_epoch_count": 0,
            "guard_rejected_segment_count": 0,
            "guard_rejected_epoch_count": 0,
            "guard_disabled_threshold_rejected_segment_count": 0,
            "guard_disabled_threshold_rejected_epoch_count": 0,
            "guard_threshold_reject_reason_counts": {},
            "guard_effective_reject_reason_counts": {},
        }

    threshold_mask = segment_frame["would_reject"].astype(bool)
    effective_mask = segment_frame["effective_reject"].astype(bool)
    disabled_threshold_mask = threshold_mask & ~effective_mask
    return {
        "guard_segment_count": int(len(segment_frame)),
        "guard_threshold_rejected_segment_count": int(threshold_mask.sum()),
        "guard_threshold_rejected_epoch_count": int(segment_frame.loc[threshold_mask, "segment_epochs"].sum()),
        "guard_rejected_segment_count": int(effective_mask.sum()),
        "guard_rejected_epoch_count": int(segment_frame.loc[effective_mask, "segment_epochs"].sum()),
        "guard_disabled_threshold_rejected_segment_count": int(disabled_threshold_mask.sum()),
        "guard_disabled_threshold_rejected_epoch_count": int(
            segment_frame.loc[disabled_threshold_mask, "segment_epochs"].sum(),
        ),
        "guard_threshold_reject_reason_counts": _reason_counts(segment_frame, threshold_mask),
        "guard_effective_reject_reason_counts": _reason_counts(segment_frame, effective_mask),
    }


def vd_factor_residual_audit(
    data_root: Path,
    trips: Sequence[str],
    *,
    start_epoch: int,
    max_epochs: int,
    output_dir: Path,
    multi_gnss: bool,
    dual_frequency: bool,
    observation_mask: bool,
    tdcp_use_drift: str,
    top: int,
    require_guard_clean: bool = False,
    verbose: bool = False,
    diagnose_fn: DiagnoseFn = run_vd_factor_residual_diagnosis,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for trip_idx, trip in enumerate(trips, start=1):
        if verbose:
            print(f"[{trip_idx}/{len(trips)}] {trip}", file=sys.stderr, flush=True)
        trip_output_dir = output_dir / trip.replace("/", "__")
        args = SimpleNamespace(
            data_root=Path(data_root),
            trip=trip,
            start_epoch=int(start_epoch),
            max_epochs=int(max_epochs),
            signal_type="GPS_L1_CA",
            constellation_type=1,
            weight_mode="sin2el",
            multi_gnss=bool(multi_gnss),
            dual_frequency=bool(dual_frequency),
            tdcp=True,
            tdcp_use_drift=str(tdcp_use_drift),
            tdcp_consistency_threshold_m=1.5,
            tdcp_weight_scale=DEFAULT_TDCP_WEIGHT_SCALE,
            tdcp_geometry_correction=DEFAULT_TDCP_GEOMETRY_CORRECTION,
            observation_mask=bool(observation_mask),
            observation_min_cn0_dbhz=OBS_MASK_MIN_CN0_DBHZ,
            observation_min_elevation_deg=OBS_MASK_MIN_ELEVATION_DEG,
            pseudorange_residual_mask_m=OBS_MASK_RESIDUAL_THRESHOLD_M,
            pseudorange_residual_mask_l5_m=OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
            doppler_residual_mask_mps=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
            pseudorange_doppler_mask_m=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
            output_dir=trip_output_dir,
            top=int(top),
        )
        try:
            summary = diagnose_fn(args)
        except Exception as exc:  # noqa: BLE001
            errors.append({"trip": trip, "error": f"{type(exc).__name__}: {exc}"})
            continue
        phone = Path(trip).name
        doppler_count = _count(summary, "doppler_seed_residual", "count")
        doppler_rms = _metric(summary, "doppler_seed_residual", "weighted_rms")
        tdcp_count = _count(summary, "tdcp_seed_residual", "count")
        tdcp_rms = _metric(summary, "tdcp_seed_residual", "weighted_rms")
        reason = _guard_reason(
            doppler_count=doppler_count,
            doppler_rms=doppler_rms,
            tdcp_count=tdcp_count,
            tdcp_rms=tdcp_rms,
        )
        phone_guard_enabled = _vd_seed_factor_guard_enabled_for_phone(phone)
        rows.append(
            {
                "trip": trip,
                "phone": phone,
                "phone_guard_enabled": bool(phone_guard_enabled),
                "n_epochs": int(summary.get("n_epochs", 0) or 0),
                "n_sat_slots": int(summary.get("n_sat_slots", 0) or 0),
                "tdcp_use_drift": bool(summary.get("tdcp_use_drift", False)),
                "doppler_count": doppler_count,
                "doppler_weighted_rms_mps": doppler_rms,
                "doppler_abs_p95_mps": _metric(summary, "doppler_seed_residual", "abs_p95"),
                "doppler_abs_max_mps": _metric(summary, "doppler_seed_residual", "abs_max"),
                "tdcp_count": tdcp_count,
                "tdcp_weighted_rms_m": tdcp_rms,
                "tdcp_abs_p95_m": _metric(summary, "tdcp_seed_residual", "abs_p95"),
                "tdcp_abs_max_m": _metric(summary, "tdcp_seed_residual", "abs_max"),
                "guard_threshold_reject_reason": reason,
                "guard_threshold_would_reject": bool(reason),
                "guard_effective_reject": bool(reason and phone_guard_enabled),
                "baseline_pr_mse": _metric(summary, "pseudorange_mse", "baseline"),
                "raw_wls_pr_mse": _metric(summary, "pseudorange_mse", "raw_wls"),
                "output_dir": str(trip_output_dir),
            },
        )

    trip_summary = pd.DataFrame(rows)
    if not trip_summary.empty:
        trip_summary = trip_summary.sort_values("trip").reset_index(drop=True)
    residual_failures = _guard_residual_failure_rows(trip_summary)
    guard_enabled_failures = [row for row in residual_failures if bool(row["phone_guard_enabled"])]
    payload = {
        "data_root": str(Path(data_root)),
        "trips": list(trips),
        "trip_count": int(len(trips)),
        "completed_trip_count": int(len(trip_summary)),
        "error_count": int(len(errors)),
        "errors": errors,
        "start_epoch": int(start_epoch),
        "max_epochs": int(max_epochs),
        "multi_gnss": bool(multi_gnss),
        "dual_frequency": bool(dual_frequency),
        "observation_mask": bool(observation_mask),
        "tdcp_use_drift": str(tdcp_use_drift),
        "require_guard_clean": bool(require_guard_clean),
        "guard_thresholds": _guard_thresholds(),
        "max_doppler_weighted_rms_mps": (
            float(trip_summary["doppler_weighted_rms_mps"].max()) if not trip_summary.empty else float("nan")
        ),
        "max_tdcp_weighted_rms_m": (
            float(trip_summary["tdcp_weighted_rms_m"].max()) if not trip_summary.empty else float("nan")
        ),
        "residual_threshold_failure_count": int(len(residual_failures)),
        "guard_enabled_residual_threshold_failure_count": int(len(guard_enabled_failures)),
        "guard_disabled_residual_threshold_failure_count": int(len(residual_failures) - len(guard_enabled_failures)),
        "residual_threshold_failures": residual_failures,
        "passed": bool(not errors and (not require_guard_clean or not guard_enabled_failures)),
    }
    if not trip_summary.empty:
        worst_doppler = trip_summary.loc[trip_summary["doppler_weighted_rms_mps"].idxmax()]
        worst_tdcp = trip_summary.loc[trip_summary["tdcp_weighted_rms_m"].idxmax()]
        payload["worst_doppler_trip"] = str(worst_doppler["trip"])
        payload["worst_tdcp_trip"] = str(worst_tdcp["trip"])
    return trip_summary, payload


def _build_diagnostic_batch_args(
    *,
    data_root: Path,
    trip: str,
    start_epoch: int,
    max_epochs: int,
    multi_gnss: bool,
    dual_frequency: bool,
    observation_mask: bool,
) -> dict[str, object]:
    return {
        "max_epochs": max_epochs_for_build(max_epochs),
        "start_epoch": int(start_epoch),
        "constellation_type": 1,
        "signal_type": "GPS_L1_CA",
        "weight_mode": "sin2el",
        "multi_gnss": bool(multi_gnss),
        "use_tdcp": True,
        "tdcp_consistency_threshold_m": 1.5,
        "tdcp_weight_scale": DEFAULT_TDCP_WEIGHT_SCALE,
        "tdcp_geometry_correction": DEFAULT_TDCP_GEOMETRY_CORRECTION,
        "data_root": Path(data_root),
        "trip": str(trip),
        "apply_observation_mask": bool(observation_mask),
        "observation_min_cn0_dbhz": OBS_MASK_MIN_CN0_DBHZ,
        "observation_min_elevation_deg": OBS_MASK_MIN_ELEVATION_DEG,
        "pseudorange_residual_mask_m": OBS_MASK_RESIDUAL_THRESHOLD_M,
        "pseudorange_residual_mask_l5_m": OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
        "doppler_residual_mask_mps": OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
        "pseudorange_doppler_mask_m": OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
        "dual_frequency": bool(dual_frequency),
    }


def vd_factor_guard_segment_rows(
    data_root: Path,
    trip: str,
    *,
    start_epoch: int,
    max_epochs: int,
    chunk_epochs: int,
    multi_gnss: bool,
    dual_frequency: bool,
    observation_mask: bool,
) -> list[dict[str, object]]:
    trip_dir = Path(data_root) / trip
    batch = build_trip_arrays(
        trip_dir,
        **_build_diagnostic_batch_args(
            data_root=Path(data_root),
            trip=trip,
            start_epoch=start_epoch,
            max_epochs=max_epochs,
            multi_gnss=multi_gnss,
            dual_frequency=dual_frequency,
            observation_mask=observation_mask,
        ),
    )
    baseline_state, _baseline_sse, _baseline_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        batch.kaggle_wls,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    raw_wls = run_wls(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
        fallback_xyz=batch.kaggle_wls,
    )
    raw_wls[:, :3] = repair_baseline_wls(batch.times_ms, raw_wls[:, :3])
    context = _build_solver_execution_context(Path(trip).name, batch, baseline_state)
    n_epoch = int(batch.sat_ecef.shape[0])
    chunk_size = n_epoch if chunk_epochs <= 0 or n_epoch <= chunk_epochs else int(chunk_epochs)
    break_mask = _factor_break_mask(context.clock_jump, batch.dt, n_epoch)
    stitched = np.zeros((n_epoch, 3 + batch.n_clock), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for chunk_start in range(0, n_epoch, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_epoch)
        baseline_chunk_state, _baseline_chunk_sse, _baseline_chunk_weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef[chunk_start:chunk_end],
            batch.pseudorange[chunk_start:chunk_end],
            batch.weights[chunk_start:chunk_end],
            batch.kaggle_wls[chunk_start:chunk_end],
            sys_kind=(batch.sys_kind[chunk_start:chunk_end] if batch.sys_kind is not None else None),
            n_clock=batch.n_clock,
        )
        raw_chunk_state, _raw_chunk_sse, _raw_chunk_weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef[chunk_start:chunk_end],
            batch.pseudorange[chunk_start:chunk_end],
            batch.weights[chunk_start:chunk_end],
            raw_wls[chunk_start:chunk_end, :3],
            sys_kind=(batch.sys_kind[chunk_start:chunk_end] if batch.sys_kind is not None else None),
            n_clock=batch.n_clock,
        )
        dt_chunk = batch.dt[chunk_start:chunk_end] if batch.dt is not None else np.zeros(chunk_end - chunk_start)
        for seg_start, seg_end in _segment_ranges(chunk_start, chunk_end, break_mask):
            local_start = seg_start - chunk_start
            local_end = seg_end - chunk_start
            seg_dt = dt_chunk[local_start:local_end]
            seg_state = _seed_vd_state(
                raw_chunk_state[local_start:local_end],
                baseline_chunk_state[local_start:local_end],
                seg_dt,
                n_clock=batch.n_clock,
                clock_drift_mps=(
                    context.clock_drift_seed_mps[seg_start:seg_end]
                    if context.clock_drift_seed_mps is not None
                    else None
                ),
            )
            if seg_start == chunk_start and chunk_start > 0 and not (
                break_mask is not None and bool(break_mask[chunk_start])
            ):
                seg_state[0, :3] = stitched[chunk_start - 1, :3]
                seg_state[0, 6 : 6 + batch.n_clock] = stitched[chunk_start - 1, 3 : 3 + batch.n_clock]
            tdcp_meas = None
            tdcp_weights = None
            if batch.tdcp_meas is not None and seg_end - seg_start > 1:
                tdcp_meas = batch.tdcp_meas[seg_start : seg_end - 1]
                tdcp_weights = batch.tdcp_weights[seg_start : seg_end - 1] if batch.tdcp_weights is not None else None
            doppler_rms, doppler_count = _vd_seed_doppler_rms(
                batch.sat_ecef[seg_start:seg_end],
                seg_state,
                batch.sat_vel[seg_start:seg_end] if batch.sat_vel is not None else None,
                batch.doppler[seg_start:seg_end] if batch.doppler is not None else None,
                batch.doppler_weights[seg_start:seg_end] if batch.doppler_weights is not None else None,
                batch.sat_clock_drift_mps[seg_start:seg_end] if batch.sat_clock_drift_mps is not None else None,
                batch.n_clock,
            )
            tdcp_rms, tdcp_count = _vd_seed_tdcp_rms(
                batch.sat_ecef[seg_start:seg_end],
                seg_state,
                tdcp_meas,
                tdcp_weights,
                batch.sys_kind[seg_start:seg_end] if batch.sys_kind is not None else None,
                seg_dt,
                n_clock=batch.n_clock,
                tdcp_use_drift=context.tdcp_use_drift,
            )
            reason = _guard_reason(
                doppler_count=doppler_count,
                doppler_rms=doppler_rms,
                tdcp_count=tdcp_count,
                tdcp_rms=tdcp_rms,
            )
            phone_guard_enabled = _vd_seed_factor_guard_enabled_for_phone(Path(trip).name)
            rows.append(
                {
                    "trip": trip,
                    "phone": Path(trip).name,
                    "phone_guard_enabled": bool(phone_guard_enabled),
                    "chunk_start_epoch": int(chunk_start),
                    "chunk_end_epoch": int(chunk_end),
                    "segment_start_epoch": int(seg_start),
                    "segment_end_epoch": int(seg_end),
                    "segment_epochs": int(seg_end - seg_start),
                    "doppler_count": int(doppler_count),
                    "doppler_rms_mps": float(doppler_rms),
                    "tdcp_count": int(tdcp_count),
                    "tdcp_rms_m": float(tdcp_rms),
                    "would_reject": bool(reason),
                    "effective_reject": bool(reason and phone_guard_enabled),
                    "reject_reason": reason,
                },
            )
            stitched[seg_start:seg_end] = raw_chunk_state[local_start:local_end]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_arg(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--trip", action="append", dest="trips")
    parser.add_argument("--start-epoch", type=int, default=0)
    _add_max_epochs_arg(parser, help_text="0 means full trip")
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--observation-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tdcp-use-drift", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--chunk-epochs", type=int, default=0, help="emit guard-segment diagnostics using this chunk size")
    parser.add_argument(
        "--require-guard-clean",
        action="store_true",
        help="fail when a guard-enabled phone exceeds the VD seed residual guard thresholds",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trips = tuple(args.trips) if args.trips else DEFAULT_VD_FACTOR_RESIDUAL_TRIPS
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_vd_factor_residual_audit")
    trip_summary, payload = vd_factor_residual_audit(
        Path(args.data_root).resolve(),
        trips,
        start_epoch=int(args.start_epoch),
        max_epochs=_nonnegative_max_epochs(args),
        output_dir=out_dir,
        multi_gnss=bool(args.multi_gnss),
        dual_frequency=bool(args.dual_frequency),
        observation_mask=bool(args.observation_mask),
        tdcp_use_drift=str(args.tdcp_use_drift),
        top=int(args.top),
        require_guard_clean=bool(args.require_guard_clean),
        verbose=bool(args.verbose),
    )
    trip_summary.to_csv(out_dir / "trip_summary.csv", index=False)
    segment_rows: list[dict[str, object]] = []
    if int(args.chunk_epochs) > 0:
        for trip in trips:
            segment_rows.extend(
                vd_factor_guard_segment_rows(
                    Path(args.data_root).resolve(),
                    trip,
                    start_epoch=int(args.start_epoch),
                    max_epochs=_nonnegative_max_epochs(args),
                    chunk_epochs=int(args.chunk_epochs),
                    multi_gnss=bool(args.multi_gnss),
                    dual_frequency=bool(args.dual_frequency),
                    observation_mask=bool(args.observation_mask),
                ),
            )
        segment_frame = pd.DataFrame(segment_rows)
        segment_frame.to_csv(out_dir / "guard_segment_summary.csv", index=False)
        payload.update(guard_segment_summary_payload(segment_frame))
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir, label="audit_dir")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
