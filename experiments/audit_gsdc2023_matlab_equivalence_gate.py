#!/usr/bin/env python3
"""Run the strict MATLAB-equivalence gate for GSDC2023 migration parity."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.audit_gsdc2023_factor_mask_parity import (  # noqa: E402
    DEFAULT_FACTOR_MASK_PARITY_TRIPS,
    factor_mask_parity_audit,
)
from experiments.audit_gsdc2023_matlab_parity import _audit_split  # noqa: E402
from experiments.audit_gsdc2023_residual_value_parity import (  # noqa: E402
    DEFAULT_RESIDUAL_PARITY_TRIPS,
    residual_value_parity_audit,
)
from experiments.compare_gsdc2023_phone_data_raw_bridge_counts import (  # noqa: E402
    build_comparison_frames,
)
from experiments.gsdc2023_audit_cli import (  # noqa: E402
    add_data_root_arg as _add_data_root_arg,
    add_max_epochs_arg as _add_max_epochs_arg,
    add_multi_gnss_arg as _add_multi_gnss_arg,
    add_output_dir_arg as _add_output_dir_arg,
    nonnegative_max_epochs as _nonnegative_max_epochs,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT  # noqa: E402


DEFAULT_EQUIVALENCE_TRIPS: tuple[str, ...] = tuple(
    sorted(set(DEFAULT_FACTOR_MASK_PARITY_TRIPS) | set(DEFAULT_RESIDUAL_PARITY_TRIPS)),
)

AssetAuditFn = Callable[..., pd.DataFrame]
FactorAuditFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]
ResidualAuditFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]
CountAuditFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    summary: dict[str, Any]
    details: str | None = None

    def row(self) -> dict[str, Any]:
        row = {"gate": self.name, "passed": bool(self.passed)}
        row.update(self.summary)
        if self.details:
            row["details"] = self.details
        return row


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _asset_gate(
    data_root: Path,
    datasets: Sequence[str],
    *,
    include_imu_sync: bool,
    strict_ref_height: bool,
    asset_audit_fn: AssetAuditFn = _audit_split,
) -> tuple[pd.DataFrame, GateResult]:
    frames = []
    for split in datasets:
        frame = asset_audit_fn(data_root, split, include_imu_sync=include_imu_sync)
        if not frame.empty:
            frame = frame.copy()
            frame.insert(0, "split", split)
            frames.append(frame)
    audit = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if audit.empty:
        summary = {"dataset_count": len(datasets), "trip_count": 0, "error": "no settings rows found"}
        return audit, GateResult("assets", False, summary)

    base_ready = int(audit["base_correction_ready"].sum()) if "base_correction_ready" in audit else 0
    imu_ready = None
    if include_imu_sync and "imu_sync_ready" in audit:
        imu_ready = int(audit["imu_sync_ready"].sum())
    ref_height_present = int(audit["ref_height_present"].sum()) if "ref_height_present" in audit else 0
    ground_truth_present = int(audit["ground_truth_present"].sum()) if "ground_truth_present" in audit else 0
    trip_count = int(len(audit))
    passed = base_ready == trip_count and ground_truth_present == trip_count
    if include_imu_sync and imu_ready is not None:
        passed = passed and imu_ready == trip_count
    if strict_ref_height:
        passed = passed and ref_height_present == trip_count

    summary = {
        "dataset_count": int(len(datasets)),
        "trip_count": trip_count,
        "base_correction_ready": base_ready,
        "ground_truth_present": ground_truth_present,
        "imu_sync_checked": bool(include_imu_sync),
        "imu_sync_ready": imu_ready,
        "ref_height_present": ref_height_present,
        "strict_ref_height": bool(strict_ref_height),
    }
    return audit, GateResult("assets", bool(passed), summary)


def _factor_gate(
    data_root: Path,
    trips: Sequence[str],
    *,
    max_epochs: int,
    multi_gnss: bool,
    min_symmetric_parity: float,
    verbose: bool = False,
    factor_audit_fn: FactorAuditFn = factor_mask_parity_audit,
) -> tuple[pd.DataFrame, pd.DataFrame, GateResult]:
    trip_summary, field_summary, payload = factor_audit_fn(
        data_root,
        trips,
        max_epochs=max_epochs,
        multi_gnss=multi_gnss,
        min_symmetric_parity=min_symmetric_parity,
        verbose=verbose,
    )
    summary = {
        "trip_count": int(payload.get("trip_count", len(trips)) or 0),
        "completed_trip_count": int(payload.get("completed_trip_count", len(trip_summary)) or 0),
        "error_count": int(payload.get("error_count", 0) or 0),
        "overall_min_symmetric_parity": payload.get("overall_min_symmetric_parity"),
        "total_matlab_only": int(payload.get("total_matlab_only", 0) or 0),
        "total_bridge_only": int(payload.get("total_bridge_only", 0) or 0),
        "threshold": float(min_symmetric_parity),
    }
    return trip_summary, field_summary, GateResult("factor_mask", bool(payload.get("passed", False)), summary)


def _residual_gate(
    data_root: Path,
    trips: Sequence[str],
    *,
    max_epochs: int,
    multi_gnss: bool,
    apply_observation_mask: bool,
    include_inactive_observations: bool,
    max_abs_delta_threshold_m: float,
    p95_abs_delta_threshold_m: float | None,
    verbose: bool = False,
    residual_audit_fn: ResidualAuditFn = residual_value_parity_audit,
) -> tuple[pd.DataFrame, pd.DataFrame, GateResult]:
    trip_summary, max_rows, payload = residual_audit_fn(
        data_root,
        trips,
        max_epochs=max_epochs,
        multi_gnss=multi_gnss,
        apply_observation_mask=apply_observation_mask,
        include_inactive_observations=include_inactive_observations,
        max_abs_delta_threshold_m=max_abs_delta_threshold_m,
        p95_abs_delta_threshold_m=p95_abs_delta_threshold_m,
        verbose=verbose,
    )
    total_matlab_only = (
        int(pd.to_numeric(trip_summary.get("matlab_only_count"), errors="coerce").fillna(0).sum())
        if "matlab_only_count" in trip_summary
        else 0
    )
    total_bridge_only = (
        int(pd.to_numeric(trip_summary.get("bridge_only_count"), errors="coerce").fillna(0).sum())
        if "bridge_only_count" in trip_summary
        else 0
    )
    summary = {
        "trip_count": int(payload.get("trip_count", len(trips)) or 0),
        "completed_trip_count": int(payload.get("completed_trip_count", len(trip_summary)) or 0),
        "error_count": int(payload.get("error_count", 0) or 0),
        "errors": payload.get("errors", []),
        "total_matlab_only": total_matlab_only,
        "total_bridge_only": total_bridge_only,
        "overall_max_abs_delta": payload.get("overall_max_abs_delta"),
        "overall_p95_abs_delta_max": payload.get("overall_p95_abs_delta_max"),
        "max_abs_delta_threshold_m": float(max_abs_delta_threshold_m),
        "p95_abs_delta_threshold_m": p95_abs_delta_threshold_m,
        "apply_observation_mask": bool(apply_observation_mask),
        "include_inactive_observations": bool(include_inactive_observations),
        "internal_delta_failure_count": int(payload.get("internal_delta_failure_count", 0) or 0),
        "internal_delta_failures": payload.get("internal_delta_failures", []),
        "internal_delta_thresholds": payload.get("internal_delta_thresholds", {}),
        "worst_trip": payload.get("worst_trip"),
        "worst_field": payload.get("worst_field"),
    }
    passed = bool(payload.get("passed", False)) and total_matlab_only == 0 and total_bridge_only == 0
    return trip_summary, max_rows, GateResult("residual_values", passed, summary)


def _count_gate(
    data_root: Path,
    trips: Sequence[str],
    *,
    max_epochs: int,
    multi_gnss: bool,
    count_audit_fn: CountAuditFn = build_comparison_frames,
) -> tuple[pd.DataFrame, pd.DataFrame, GateResult]:
    comparison, trip_summary, payload = count_audit_fn(
        data_root,
        ["train"],
        trips=list(trips),
        max_epochs=max_epochs,
        multi_gnss=multi_gnss,
    )
    matched_abs_delta_total = int(payload.get("matched_abs_delta_total", 0) or 0)
    bridge_errors = int(payload.get("bridge_errors", 0) or 0)
    phone_errors = int(payload.get("phone_errors", 0) or 0)
    trip_count = int(payload.get("trip_count", len(trips)) or 0)
    trips_with_phone_data = int(payload.get("trips_with_phone_data", 0) or 0)
    count_parity_ratio = payload.get("count_parity_ratio")
    passed = (
        bridge_errors == 0
        and phone_errors == 0
        and trip_count == len(trips)
        and trips_with_phone_data == trip_count
        and matched_abs_delta_total == 0
        and count_parity_ratio == 1.0
    )
    summary = {
        "trip_count": trip_count,
        "trips_with_phone_data": trips_with_phone_data,
        "bridge_errors": bridge_errors,
        "phone_errors": phone_errors,
        "matched_rows": int(payload.get("matched_rows", 0) or 0),
        "matched_abs_delta_total": matched_abs_delta_total,
        "count_parity_ratio": count_parity_ratio,
    }
    return comparison, trip_summary, GateResult("raw_bridge_counts", bool(passed), summary)


def run_equivalence_gate(
    data_root: Path,
    output_dir: Path,
    *,
    trips: Sequence[str],
    max_epochs: int,
    count_max_epochs: int,
    factor_multi_gnss: bool,
    residual_multi_gnss: bool,
    residual_observation_mask: bool,
    residual_include_inactive_observations: bool,
    count_multi_gnss: bool,
    asset_datasets: Sequence[str],
    quick_assets: bool,
    strict_ref_height: bool,
    min_symmetric_parity: float,
    max_abs_delta_threshold_m: float,
    p95_abs_delta_threshold_m: float | None,
    verbose: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gates: list[GateResult] = []

    if verbose:
        print("[1/4] asset gate", file=sys.stderr, flush=True)
    asset_frame, asset_result = _asset_gate(
        data_root,
        asset_datasets,
        include_imu_sync=not quick_assets,
        strict_ref_height=strict_ref_height,
    )
    asset_frame.to_csv(output_dir / "asset_audit.csv", index=False)
    gates.append(asset_result)

    if verbose:
        print("[2/4] factor-mask gate", file=sys.stderr, flush=True)
    factor_trip_summary, factor_field_summary, factor_result = _factor_gate(
        data_root,
        trips,
        max_epochs=max_epochs,
        multi_gnss=factor_multi_gnss,
        min_symmetric_parity=min_symmetric_parity,
        verbose=verbose,
    )
    factor_trip_summary.to_csv(output_dir / "factor_mask_trip_summary.csv", index=False)
    factor_field_summary.to_csv(output_dir / "factor_mask_field_summary.csv", index=False)
    gates.append(factor_result)

    if verbose:
        print("[3/4] residual-value gate", file=sys.stderr, flush=True)
    residual_trip_summary, residual_max_rows, residual_result = _residual_gate(
        data_root,
        trips,
        max_epochs=max_epochs,
        multi_gnss=residual_multi_gnss,
        apply_observation_mask=residual_observation_mask,
        include_inactive_observations=residual_include_inactive_observations,
        max_abs_delta_threshold_m=max_abs_delta_threshold_m,
        p95_abs_delta_threshold_m=p95_abs_delta_threshold_m,
        verbose=verbose,
    )
    residual_trip_summary.to_csv(output_dir / "residual_value_trip_summary.csv", index=False)
    residual_max_rows.to_csv(output_dir / "residual_value_max_rows.csv", index=False)
    gates.append(residual_result)

    if verbose:
        print("[4/4] raw-bridge count gate", file=sys.stderr, flush=True)
    count_comparison, count_trip_summary, count_result = _count_gate(
        data_root,
        trips,
        max_epochs=count_max_epochs,
        multi_gnss=count_multi_gnss,
    )
    count_comparison.to_csv(output_dir / "raw_bridge_count_comparison.csv", index=False)
    count_trip_summary.to_csv(output_dir / "raw_bridge_count_trip_summary.csv", index=False)
    gates.append(count_result)

    gate_rows = pd.DataFrame([gate.row() for gate in gates])
    gate_rows.to_csv(output_dir / "gate_summary.csv", index=False)
    passed = all(gate.passed for gate in gates)
    payload = {
        "passed": bool(passed),
        "equivalence_claim": "matlab_equivalent" if passed else "not_proven",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "trips": list(trips),
        "trip_count": int(len(trips)),
        "max_epochs": int(max_epochs),
        "count_max_epochs": int(count_max_epochs),
        "factor_multi_gnss": bool(factor_multi_gnss),
        "residual_multi_gnss": bool(residual_multi_gnss),
        "residual_observation_mask": bool(residual_observation_mask),
        "residual_include_inactive_observations": bool(residual_include_inactive_observations),
        "count_multi_gnss": bool(count_multi_gnss),
        "asset_datasets": list(asset_datasets),
        "quick_assets": bool(quick_assets),
        "strict_ref_height": bool(strict_ref_height),
        "gates": {gate.name: {"passed": gate.passed, **gate.summary} for gate in gates},
    }
    _write_json(output_dir / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_arg(parser, default_root=DEFAULT_ROOT)
    parser.add_argument(
        "--trip",
        action="append",
        dest="trips",
        help="trip in split/course/phone form; repeatable. Defaults to the built-in MATLAB parity export set.",
    )
    _add_max_epochs_arg(parser, help_text="0 uses each trip's full settings window")
    parser.add_argument(
        "--count-max-epochs",
        type=int,
        default=0,
        help="epoch limit for raw bridge count parity; default 0 because MATLAB count exports are full-window",
    )
    _add_multi_gnss_arg(parser, default=False, help_text="factor/count scope; default matches GPS-only MATLAB exports")
    parser.add_argument(
        "--residual-multi-gnss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="residual-value scope; default false because current MATLAB residual exports are GPS-only",
    )
    parser.add_argument(
        "--residual-observation-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep factor-observation mask while estimating residual common biases",
    )
    parser.add_argument(
        "--residual-include-inactive-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include pre-mask diagnostics rows using common biases from active factors",
    )
    parser.add_argument("--asset-datasets", nargs="*", default=["train"])
    parser.add_argument("--quick-assets", action="store_true", help="skip expensive IMU sync parsing in asset gate")
    parser.add_argument("--strict-ref-height", action="store_true", help="require ref_hight.mat/ref_height.mat coverage")
    parser.add_argument("--min-symmetric-parity", type=float, default=1.0)
    parser.add_argument("--max-abs-delta-threshold-m", type=float, default=1.0e-4)
    parser.add_argument("--p95-abs-delta-threshold-m", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", help="print gate and trip progress to stderr")
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trips = tuple(args.trips) if args.trips else DEFAULT_EQUIVALENCE_TRIPS
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolved_output_root(args) / f"gsdc2023_matlab_equivalence_gate_{stamp}"
    payload = run_equivalence_gate(
        Path(args.data_root).resolve(),
        output_dir,
        trips=trips,
        max_epochs=_nonnegative_max_epochs(args),
        count_max_epochs=max(int(args.count_max_epochs), 0),
        factor_multi_gnss=bool(args.multi_gnss),
        residual_multi_gnss=bool(args.residual_multi_gnss),
        residual_observation_mask=bool(args.residual_observation_mask),
        residual_include_inactive_observations=bool(args.residual_include_inactive_observations),
        count_multi_gnss=bool(args.multi_gnss),
        asset_datasets=tuple(args.asset_datasets),
        quick_assets=bool(args.quick_assets),
        strict_ref_height=bool(args.strict_ref_height),
        min_symmetric_parity=float(args.min_symmetric_parity),
        max_abs_delta_threshold_m=float(args.max_abs_delta_threshold_m),
        p95_abs_delta_threshold_m=args.p95_abs_delta_threshold_m,
        verbose=bool(args.verbose),
    )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    print(f"equivalence_dir={output_dir}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
