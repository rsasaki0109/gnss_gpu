#!/usr/bin/env python3
"""Summarize MATLAB ``preprocessing.m`` coverage in the Python raw bridge.

The table produced here is a gap audit, not a numerical parity proof. It maps
the major MATLAB preprocessing stages and ``phone_data.mat`` artifacts to the
current Python raw-bridge behavior, then optionally scans local trips to attach
file/readability evidence and raw-bridge observation counts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT, build_trip_arrays, collect_matlab_parity_audit
from experiments.validate_gsdc2023_phone_data import TripValidationResult, validate_trip


@dataclass(frozen=True)
class PreprocessingGapRow:
    stage_id: str
    matlab_stage: str
    matlab_artifact: str
    python_coverage: str
    python_evidence: str
    status: str
    gap: str
    next_action: str


@dataclass(frozen=True)
class TripSpec:
    split: str
    course: str
    phone: str
    trip: str
    trip_dir: Path
    source: str
    idx_start: int | None = None
    idx_end: int | None = None


def preprocessing_gap_rows() -> list[PreprocessingGapRow]:
    """Static coverage table for ``ref/gsdc2023/preprocessing.m``."""

    return [
        PreprocessingGapRow(
            stage_id="raw_gnss_conversion",
            matlab_stage="ReadGnssLogFile converts supplemental/gnss_log.txt to GobsPhone",
            matlab_artifact="obs.P/D/L, sat metadata, residual fields",
            python_coverage="partial",
            python_evidence="raw bridge reads device_gnss.csv columns directly",
            status="partial",
            gap="No Python phone_data.mat writer and no gnss_log.txt parser for full MATLAB obs struct.",
            next_action="Choose whether raw CSV remains canonical or add a phone_data-compatible preprocessor.",
        ),
        PreprocessingGapRow(
            stage_id="baseline_filter_repair",
            matlab_stage="Filter BiasUncertaintyNanos, interpolate NaNs, remove moving-median outliers",
            matlab_artifact="timebl, posbl",
            python_coverage="implemented in raw path",
            python_evidence="_repair_baseline_wls plus baseline source/gated source modes",
            status="partial",
            gap="The Python repair is behaviorally similar but not a line-by-line MATLAB port.",
            next_action="Add a small train-trip fixture comparing repaired baseline samples if exact parity matters.",
        ),
        PreprocessingGapRow(
            stage_id="base_rinex_preload",
            matlab_stage="Trim base RINEX obs around phone time span and eliminate invalid samples",
            matlab_artifact="obsb",
            python_coverage="file audit plus on-demand RINEX read",
            python_evidence="collect_matlab_parity_audit and --base-correction RINEX loader",
            status="experimental",
            gap="Python does not persist obsb and only loads the subset needed for pseudorange correction.",
            next_action="Keep on-demand loading unless an obsb fixture is needed for MATLAB regression tests.",
        ),
        PreprocessingGapRow(
            stage_id="base_pseudorange_correction",
            matlab_stage="correct_pseudorange applies base-station residual correction",
            matlab_artifact="obs.L1/L5 residual-corrected pseudorange",
            python_coverage="optional raw bridge flag",
            python_evidence="validate_fgo_gsdc2023_raw.py --base-correction",
            status="experimental",
            gap="GPS base correction matches MATLAB on two pixel5 smokes: L1 sub-millimeter on 2020 MTV and L1/L5 about 1e-5 m on 2021 LAX. Local train brdc.*n files contain GPS nav only, so Galileo/QZSS correction parity is data-blocked.",
            next_action="If a tree with Galileo/QZSS broadcast nav is available, run export_base_correction_series.m plus compare_gsdc2023_base_correction_series.py on those slots.",
        ),
        PreprocessingGapRow(
            stage_id="ground_truth_alignment",
            matlab_stage="Load train ground_truth.csv and save gt.mat",
            matlab_artifact="gt.mat: posgt, timegt",
            python_coverage="implemented for validation",
            python_evidence="validate_gsdc2023_phone_data.py reads gt.mat or ground_truth.csv",
            status="partial",
            gap="Python validates against GT but does not write MATLAB-compatible gt.mat by default.",
            next_action="Add a gt.mat writer only if downstream MATLAB artifact parity is required.",
        ),
        PreprocessingGapRow(
            stage_id="clock_repair",
            matlab_stage="Phone-specific clock, drift, Samsung blocklist, and jump repairs",
            matlab_artifact="obs clock/drift fields and XX flags",
            python_coverage="partial raw bridge heuristics",
            python_evidence="clock aid, clock jump segmentation, TDCP phone policies",
            status="partial",
            gap="Several MATLAB phone-specific corrections are approximated or replaced by raw-bridge policies.",
            next_action="Audit one problematic phone family at a time before changing global clock behavior.",
        ),
        PreprocessingGapRow(
            stage_id="observation_masking",
            matlab_stage="Apply exobs/exobs_residuals masking and residual outlier exclusion",
            matlab_artifact="filtered obs matrices and residual masks",
            python_coverage="optional raw bridge flag",
            python_evidence="validate_fgo_gsdc2023_raw.py --observation-mask; compare_gsdc2023_phone_data_raw_bridge_counts.py GPS-only count parity",
            status="experimental",
            gap="C/N0, multipath, State bit, range, elevation, pseudorange-Doppler, pseudorange residual, Doppler residual, and ADR-Doppler TDCP consistency masks are connected. GPS-only MATLAB factor-count parity is exact on train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4; multi-GNSS requires a matching MATLAB export scope.",
            next_action="Use compare_gsdc2023_phone_data_raw_bridge_counts.py in its default GPS-only scope for MATLAB factor-count parity, and --multi-gnss only for separate bridge coverage audits.",
        ),
        PreprocessingGapRow(
            stage_id="dual_frequency_model",
            matlab_stage="Build L1 and L5 bands and use dual-frequency observations where available",
            matlab_artifact="obs.L1 and obs.L5",
            python_coverage="optional raw bridge flag",
            python_evidence="validate_fgo_gsdc2023_raw.py --dual-frequency; exact GPS L1/L5 factor-count parity on exported phone_data_factor_counts.csv",
            status="experimental",
            gap="L1/L5 observations are separate slots and dual-frequency raw bridge/native FGO now use MATLAB's seven signal-clock indices for GPS/Galileo/BeiDou L1/L5. QZSS is mapped to the GPS clock family because MATLAB's sysfreq2sigtype does not assign a QZSS-specific index. Current MATLAB factor-count exports are GPS-only, so multi-GNSS count deltas are scope differences rather than parity failures.",
            next_action="For multi-GNSS parity, regenerate MATLAB factor counts with the same constellation scope before comparing counts or residual values.",
        ),
        PreprocessingGapRow(
            stage_id="imu_preprocessing",
            matlab_stage="deviceimu2imu loads, calibrates, synchronizes acc/gyro/mag",
            matlab_artifact="acc, gyro, mag",
            python_coverage="partial loading and stop detection",
            python_evidence="load_device_imu_measurements, process_device_imu, project_stop_to_epochs",
            status="partial",
            gap="Python does not expose MATLAB-compatible calibrated IMU structs.",
            next_action="Decide whether IMU parity target is artifact compatibility or graph-factor behavior.",
        ),
        PreprocessingGapRow(
            stage_id="imu_preintegration",
            matlab_stage="GTSAM preintegrated IMU factors and bias evolution",
            matlab_artifact="fgo_gnss_imu graph inputs",
            python_coverage="optional weak body/ECEF delta priors plus bias telemetry; native accel-bias scaffold",
            python_evidence="preintegrate_processed_imu + validate_fgo_gsdc2023_raw.py --imu-prior --imu-frame {body,ecef} --imu-accel-bias-state --factor-dt-max-s; imu_acc/gyro_bias_mean_norm metrics; fgo_gnss_lm_vd optional [bax,bay,baz] state",
            status="experimental",
            gap="Native FGO can consume epoch delta priors, ECEF mode applies yaw/mounting/gravity approximation, raw BiasX/Y/Z are synchronized as telemetry, and long-gap factor gating follows MATLAB time_diff_th. Raw bridge can opt into a native accelerometer-bias state for first-order IMU delta correction, but it remains off by default and gyro bias / pose attitude optimization are still missing.",
            next_action="Add a real pose/bias IMU state before enabling IMU priors by default.",
        ),
        PreprocessingGapRow(
            stage_id="height_constraints",
            matlab_stage="Use relative and absolute height references when present",
            matlab_artifact="ref_hight.mat and relative-height factors",
            python_coverage="relative height plus opt-in absolute-height path",
            python_evidence="raw bridge --absolute-height loads ref_hight.mat/ref_height.mat; native VD relative/absolute height factors have standalone direction regressions",
            status="experimental",
            gap="Local audit found no ref_hight.mat/ref_height.mat artifact, so MATLAB numerical parity is not validated on real reference-height data.",
            next_action="Run --absolute-height on a tree with ref_hight.mat and compare nearest-reference counts/residuals against MATLAB.",
        ),
        PreprocessingGapRow(
            stage_id="submission_offset",
            matlab_stage="Apply phone position offsets and write submission artifacts",
            matlab_artifact="result_gnss.mat, result_gnss_imu.mat, submission.csv",
            python_coverage="implemented for raw bridge output",
            python_evidence="--position-offset and run_raw_bridge_batch.py submission path",
            status="implemented",
            gap="Python output is not a MATLAB result_gnss.mat clone.",
            next_action="Keep CSV submission path canonical unless MATLAB .mat outputs are needed for comparison.",
        ),
    ]


def gap_rows_to_dicts(rows: Iterable[PreprocessingGapRow]) -> list[dict[str, str]]:
    return [asdict(row) for row in rows]


def rows_to_dataframe(rows: Iterable[PreprocessingGapRow]) -> pd.DataFrame:
    return pd.DataFrame(gap_rows_to_dicts(rows))


def _escape_markdown(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def rows_to_markdown(rows: Iterable[PreprocessingGapRow]) -> str:
    dicts = gap_rows_to_dicts(rows)
    if not dicts:
        return ""
    headers = list(dicts[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in dicts:
        lines.append("| " + " | ".join(_escape_markdown(row.get(header)) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _split_settings_trips(data_root: Path, split: str) -> list[TripSpec]:
    settings_path = data_root / f"settings_{split}.csv"
    if not settings_path.is_file():
        return []
    settings = pd.read_csv(settings_path)
    specs: list[TripSpec] = []
    if "Course" not in settings.columns or "Phone" not in settings.columns:
        return specs
    seen: set[tuple[str, str]] = set()
    for row in settings.itertuples(index=False):
        course = str(getattr(row, "Course", "")).strip()
        phone = str(getattr(row, "Phone", "")).strip()
        if not course or not phone or (course, phone) in seen:
            continue
        seen.add((course, phone))
        trip = f"{split}/{course}/{phone}"
        idx_start_value = getattr(row, "IdxStart", None)
        idx_end_value = getattr(row, "IdxEnd", None)
        idx_start = None if pd.isna(idx_start_value) else int(idx_start_value)
        idx_end = None if pd.isna(idx_end_value) else int(idx_end_value)
        specs.append(
            TripSpec(
                split=split,
                course=course,
                phone=phone,
                trip=trip,
                trip_dir=data_root / split / course / phone,
                source=f"settings_{split}.csv",
                idx_start=idx_start,
                idx_end=idx_end,
            ),
        )
    return specs


def _split_directory_trips(data_root: Path, split: str) -> list[TripSpec]:
    split_dir = data_root / split
    if not split_dir.is_dir():
        return []
    specs: list[TripSpec] = []
    marker_names = {
        "phone_data.mat",
        "device_gnss.csv",
        "ground_truth.csv",
        "device_imu.csv",
    }
    for phone_dir in sorted(path for path in split_dir.glob("*/*") if path.is_dir()):
        if not any((phone_dir / marker).is_file() for marker in marker_names):
            continue
        course = phone_dir.parent.name
        phone = phone_dir.name
        trip = f"{split}/{course}/{phone}"
        specs.append(
            TripSpec(
                split=split,
                course=course,
                phone=phone,
                trip=trip,
                trip_dir=phone_dir,
                source="directory_scan",
            ),
        )
    return specs


def discover_trip_specs(data_root: Path, datasets: Iterable[str]) -> list[TripSpec]:
    specs: list[TripSpec] = []
    for split in datasets:
        split = str(split).strip()
        if not split:
            continue
        split_specs = _split_settings_trips(data_root, split)
        if not split_specs:
            split_specs = _split_directory_trips(data_root, split)
        specs.extend(split_specs)
    return specs


def _score(metrics: object | None) -> float | None:
    if metrics is None:
        return None
    return getattr(metrics, "score_m", None)


def _count(result: TripValidationResult | None, freq: str, key: str) -> int | None:
    if result is None:
        return None
    return result.counts_by_freq.get(freq, {}).get(key)


def _total_count(result: TripValidationResult | None, key: str) -> int | None:
    if result is None:
        return None
    values = [
        result.counts_by_freq.get(freq, {}).get(key)
        for freq in ("L1", "L5")
    ]
    values = [int(value) for value in values if value is not None]
    if not values:
        return None
    return int(sum(values))


def _present(*paths: Path) -> bool:
    return any(path.is_file() for path in paths)


def _bridge_counts(
    data_root: Path,
    spec: TripSpec,
    *,
    max_epochs: int,
) -> dict[str, object]:
    batch = build_trip_arrays(
        spec.trip_dir,
        max_epochs=(max_epochs if max_epochs > 0 else 1_000_000_000),
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        use_tdcp=True,
        tdcp_consistency_threshold_m=1.5,
        apply_base_correction=False,
        data_root=data_root,
        trip=spec.trip,
        apply_observation_mask=True,
        dual_frequency=True,
    )
    return {
        "bridge_epochs": int(batch.times_ms.size),
        "bridge_sat_slots": int(batch.n_sat_slots),
        "bridge_p_count": int((batch.weights > 0.0).sum()),
        "bridge_d_count": (
            None if batch.doppler_weights is None else int((batch.doppler_weights > 0.0).sum())
        ),
        "bridge_tdcp_count": (
            None if batch.tdcp_weights is None else int((batch.tdcp_weights > 0.0).sum())
        ),
        "bridge_raw_mask_count": int(batch.observation_mask_count),
        "bridge_pr_residual_mask_count": int(batch.residual_mask_count),
        "bridge_doppler_residual_mask_count": int(batch.doppler_residual_mask_count),
        "bridge_pseudorange_doppler_mask_count": int(batch.pseudorange_doppler_mask_count),
        "bridge_tdcp_consistency_mask_count": int(batch.tdcp_consistency_mask_count),
    }


def _trip_record(
    data_root: Path,
    spec: TripSpec,
    *,
    include_validation: bool,
    include_bridge_counts: bool,
    bridge_max_epochs: int,
) -> dict[str, object]:
    trip_dir = spec.trip_dir
    audit: dict[str, object] = {}
    audit_error = None
    try:
        audit = collect_matlab_parity_audit(data_root, spec.trip)
    except Exception as exc:  # noqa: BLE001
        audit_error = str(exc)

    result: TripValidationResult | None = None
    validation_error = None
    if include_validation and _present(trip_dir / "phone_data.mat", trip_dir / "device_gnss.csv"):
        try:
            result = validate_trip(trip_dir)
        except Exception as exc:  # noqa: BLE001
            validation_error = str(exc)

    bridge_counts: dict[str, object] = {}
    bridge_count_error = None
    if include_bridge_counts and (trip_dir / "device_gnss.csv").is_file():
        try:
            bridge_counts = _bridge_counts(data_root, spec, max_epochs=bridge_max_epochs)
        except Exception as exc:  # noqa: BLE001
            bridge_count_error = str(exc)

    phone_p_count = _total_count(result, "P")
    phone_d_count = _total_count(result, "D")
    phone_l_count = _total_count(result, "L")
    bridge_p_count = bridge_counts.get("bridge_p_count")
    bridge_d_count = bridge_counts.get("bridge_d_count")
    bridge_tdcp_count = bridge_counts.get("bridge_tdcp_count")

    record: dict[str, object] = {
        "trip": spec.trip,
        "dataset_split": spec.split,
        "course": spec.course,
        "phone": spec.phone,
        "source": spec.source,
        "trip_dir_present": trip_dir.is_dir(),
        "phone_data_present": (trip_dir / "phone_data.mat").is_file(),
        "raw_device_gnss_present": (trip_dir / "device_gnss.csv").is_file(),
        "raw_gnss_log_present": _present(
            trip_dir / "supplemental" / "gnss_log.txt",
            trip_dir / "gnss_log.txt",
        ),
        "device_imu_present": (trip_dir / "device_imu.csv").is_file(),
        "ground_truth_csv_present": (trip_dir / "ground_truth.csv").is_file(),
        "gt_mat_present": (trip_dir / "gt.mat").is_file(),
        "result_gnss_present": (trip_dir / "result_gnss.mat").is_file(),
        "result_gnss_imu_present": (trip_dir / "result_gnss_imu.mat").is_file(),
        "settings_csv_present": audit.get("settings_csv_present"),
        "setting_row_present": audit.get("setting_row_present"),
        "base_name": audit.get("base_name"),
        "rinex_type": audit.get("rinex_type"),
        "base_correction_status": audit.get("base_correction_status"),
        "base_correction_ready": audit.get("base_correction_ready"),
        "base_obs_file_present": audit.get("base_obs_file_present"),
        "broadcast_nav_present": audit.get("broadcast_nav_present"),
        "base_position_csv_present": audit.get("base_position_csv_present"),
        "base_offset_csv_present": audit.get("base_offset_csv_present"),
        "ref_height_present": audit.get("ref_height_present"),
        "gnss_elapsed_present": audit.get("gnss_elapsed_present"),
        "imu_sync_ready": audit.get("imu_sync_ready"),
        "stop_epoch_count": audit.get("stop_epoch_count"),
        "obs_epochs": None if result is None else result.obs_epochs,
        "baseline_epochs": None if result is None else result.baseline_epochs,
        "gt_epochs": None if result is None else result.gt_epochs,
        "nsat": None if result is None else result.nsat,
        "dt_s": None if result is None else result.dt_s,
        "l1_p_count": _count(result, "L1", "P"),
        "l1_d_count": _count(result, "L1", "D"),
        "l1_l_count": _count(result, "L1", "L"),
        "l5_p_count": _count(result, "L5", "P"),
        "l5_d_count": _count(result, "L5", "D"),
        "l5_l_count": _count(result, "L5", "L"),
        "phone_p_count": phone_p_count,
        "phone_d_count": phone_d_count,
        "phone_l_count": phone_l_count,
        **bridge_counts,
        "bridge_minus_phone_p_count": (
            None if phone_p_count is None or bridge_p_count is None else int(bridge_p_count) - int(phone_p_count)
        ),
        "bridge_minus_phone_d_count": (
            None if phone_d_count is None or bridge_d_count is None else int(bridge_d_count) - int(phone_d_count)
        ),
        "bridge_minus_phone_tdcp_count": (
            None if phone_l_count is None or bridge_tdcp_count is None else int(bridge_tdcp_count) - int(phone_l_count)
        ),
        "baseline_score_m": None if result is None else _score(result.baseline_metrics),
        "result_gnss_score_m": None if result is None else _score(result.result_gnss_metrics),
        "result_gnss_imu_score_m": None if result is None else _score(result.result_gnss_imu_metrics),
        "audit_error": audit_error,
        "validation_error": validation_error,
        "bridge_count_error": bridge_count_error,
    }
    return record


def trip_gap_records(
    data_root: Path,
    datasets: Iterable[str],
    *,
    limit: int = 0,
    include_validation: bool = True,
    include_bridge_counts: bool = False,
    bridge_max_epochs: int = 0,
) -> pd.DataFrame:
    specs = discover_trip_specs(data_root, datasets)
    if limit > 0:
        specs = specs[:limit]
    rows = [
        _trip_record(
            data_root,
            spec,
            include_validation=include_validation,
            include_bridge_counts=include_bridge_counts,
            bridge_max_epochs=bridge_max_epochs,
        )
        for spec in specs
    ]
    return pd.DataFrame(rows)


def summary_from_records(records: pd.DataFrame | None = None) -> dict[str, object]:
    rows = rows_to_dataframe(preprocessing_gap_rows())
    status_counts = {
        str(key): int(value)
        for key, value in rows["status"].value_counts(dropna=False).to_dict().items()
    }
    summary: dict[str, object] = {
        "static_stage_count": int(len(rows)),
        "static_status_counts": status_counts,
    }
    if records is not None:
        summary["trip_count"] = int(len(records))
        for col in (
            "phone_data_present",
            "raw_device_gnss_present",
            "raw_gnss_log_present",
            "device_imu_present",
            "base_correction_ready",
            "ref_height_present",
            "gt_mat_present",
            "ground_truth_csv_present",
        ):
            if col in records.columns:
                summary[col] = int(records[col].fillna(False).astype(bool).sum())
        if "base_correction_status" in records.columns:
            summary["base_correction_status_counts"] = {
                str(key): int(value)
                for key, value in records["base_correction_status"].value_counts(dropna=False).to_dict().items()
            }
        for col in (
            "bridge_raw_mask_count",
            "bridge_pr_residual_mask_count",
            "bridge_doppler_residual_mask_count",
            "bridge_pseudorange_doppler_mask_count",
            "bridge_tdcp_consistency_mask_count",
        ):
            if col in records.columns:
                values = pd.to_numeric(records[col], errors="coerce")
                if values.notna().any():
                    summary[f"{col}_sum"] = int(values.fillna(0).sum())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=["train", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--scan-trips", action="store_true", help="scan local trips and write trip_gap.csv")
    parser.add_argument("--limit", type=int, default=0, help="limit trip scan rows; 0 means no limit")
    parser.add_argument("--no-validation", action="store_true", help="skip phone_data/raw baseline validation during trip scan")
    parser.add_argument("--bridge-counts", action="store_true", help="attach raw-bridge observation/mask counts to trip_gap.csv")
    parser.add_argument("--bridge-max-epochs", type=int, default=0, help="epoch limit for --bridge-counts; 0 means all")
    parser.add_argument("--print-markdown", action="store_true", help="also print the static Markdown table")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_dir = args.output_dir.resolve() / f"gsdc2023_preprocessing_gap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = preprocessing_gap_rows()
    rows_to_dataframe(rows).to_csv(out_dir / "preprocessing_gap.csv", index=False)
    markdown = rows_to_markdown(rows)
    (out_dir / "preprocessing_gap.md").write_text(markdown, encoding="utf-8")

    records = None
    if args.scan_trips:
        records = trip_gap_records(
            data_root,
            args.datasets,
            limit=max(args.limit, 0),
            include_validation=not args.no_validation,
            include_bridge_counts=args.bridge_counts,
            bridge_max_epochs=max(args.bridge_max_epochs, 0),
        )
        records.to_csv(out_dir / "trip_gap.csv", index=False)

    summary = summary_from_records(records)
    summary["data_root"] = str(data_root)
    summary["datasets"] = list(args.datasets)
    summary["trip_scan_enabled"] = bool(args.scan_trips)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"audit_dir={out_dir}")
    if args.print_markdown:
        print()
        print(markdown)


__all__ = [
    "PreprocessingGapRow",
    "TripSpec",
    "discover_trip_specs",
    "gap_rows_to_dicts",
    "preprocessing_gap_rows",
    "rows_to_dataframe",
    "rows_to_markdown",
    "summary_from_records",
    "trip_gap_records",
]


if __name__ == "__main__":
    main()
