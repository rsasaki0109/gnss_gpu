#!/usr/bin/env python3
"""Build a GSDC2023 test submission from raw bridge selected positions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.gsdc2023_chunk_selection import (  # noqa: E402
    DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT,
)
from experiments.gsdc2023_output import (  # noqa: E402
    export_bridge_outputs,
    has_valid_bridge_outputs,
    load_bridge_metrics,
)
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    BridgeConfig,
    CT_RBPF_FGO_SOURCE,
    DD_CARRIER_FGO_SOURCE,
    DEFAULT_CT_RBPF_MOTION_SIGMA_M,
    DEFAULT_ROOT,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    FACTOR_DT_MAX_S,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    validate_raw_gsdc2023_trip,
)


KEY_COLUMNS = ["tripId", "UnixTimeMillis"]
COORDINATE_COLUMNS = ["LatitudeDegrees", "LongitudeDegrees"]
DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4.csv")


def ordered_trip_ids(sample_submission: pd.DataFrame) -> list[str]:
    if "tripId" not in sample_submission.columns:
        raise ValueError("sample submission is missing tripId")
    return [str(trip) for trip in sample_submission["tripId"].drop_duplicates()]


def normalize_sample_trip_id(trip: str) -> str:
    trip = trip.strip()
    return trip.removeprefix("test/")


def bridge_trip_id(sample_trip_id: str) -> str:
    return f"test/{normalize_sample_trip_id(sample_trip_id)}"


def bridge_output_dir(bridge_output_root: Path, sample_trip_id: str) -> Path:
    return bridge_output_root / normalize_sample_trip_id(sample_trip_id)


def load_cached_bridge_trip(
    bridge_output_root: Path,
    sample_trip_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    trip_dir = bridge_output_dir(bridge_output_root, sample_trip_id)
    if not has_valid_bridge_outputs(trip_dir):
        return None
    return pd.read_csv(trip_dir / "bridge_positions.csv"), load_bridge_metrics(trip_dir)


def submission_from_bridge_tables(
    sample_submission: pd.DataFrame,
    bridge_tables: dict[str, pd.DataFrame],
    *,
    allow_partial: bool = False,
    interpolate_missing: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_columns = set(KEY_COLUMNS + COORDINATE_COLUMNS).difference(sample_submission.columns)
    if missing_columns:
        raise ValueError(f"sample submission is missing columns: {sorted(missing_columns)}")

    output = sample_submission.copy()
    patched_rows = 0
    interpolated_rows = 0
    missing_rows = 0
    rows_by_trip: dict[str, int] = {}
    missing_by_trip: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for trip_id in ordered_trip_ids(sample_submission):
        trip_mask = output["tripId"] == trip_id
        bridge = bridge_tables.get(trip_id)
        if bridge is None:
            if allow_partial:
                missing = int(trip_mask.sum())
                missing_rows += missing
                missing_by_trip[trip_id] = missing
                continue
            raise ValueError(f"missing bridge table for {trip_id}")

        required_bridge_columns = set(["UnixTimeMillis"] + COORDINATE_COLUMNS)
        bridge_missing = required_bridge_columns.difference(bridge.columns)
        if bridge_missing:
            raise ValueError(f"bridge table for {trip_id} is missing columns: {sorted(bridge_missing)}")
        if bridge["UnixTimeMillis"].duplicated().any():
            raise ValueError(f"bridge table for {trip_id} has duplicate UnixTimeMillis")

        keyed = output.loc[trip_mask, KEY_COLUMNS].copy()
        patch_columns = ["UnixTimeMillis"] + COORDINATE_COLUMNS
        if "SelectedSource" in bridge.columns:
            patch_columns.append("SelectedSource")
        patch = keyed.merge(bridge[patch_columns], on="UnixTimeMillis", how="left", validate="one_to_one")
        missing_mask = patch[COORDINATE_COLUMNS].isna().any(axis=1)
        if missing_mask.any() and interpolate_missing:
            bridge_sorted = bridge[patch_columns].sort_values("UnixTimeMillis")
            bridge_times = bridge_sorted["UnixTimeMillis"].to_numpy(dtype=float)
            missing_times = patch.loc[missing_mask, "UnixTimeMillis"].to_numpy(dtype=float)
            if bridge_times.size < 1:
                if not allow_partial:
                    raise ValueError(f"bridge table for {trip_id} cannot interpolate {int(missing_mask.sum())} timestamp(s)")
            else:
                for column in COORDINATE_COLUMNS:
                    values = bridge_sorted[column].to_numpy(dtype=float)
                    if not np.isfinite(values).all():
                        raise ValueError(f"bridge table for {trip_id} has non-finite {column} values")
                    # np.interp clamps out-of-range timestamps to the nearest endpoint.
                    patch.loc[missing_mask, column] = np.interp(missing_times, bridge_times, values)
                if "SelectedSource" in patch.columns:
                    patch.loc[missing_mask, "SelectedSource"] = "interpolated"
                interpolated = int(missing_mask.sum())
                interpolated_rows += interpolated
                missing_mask = patch[COORDINATE_COLUMNS].isna().any(axis=1)
        if missing_mask.any():
            missing = int(missing_mask.sum())
            if not allow_partial:
                raise ValueError(f"bridge table for {trip_id} is missing {missing} sample timestamp(s)")
            missing_rows += missing
            missing_by_trip[trip_id] = missing

        replace_mask = trip_mask.copy()
        if missing_mask.any():
            trip_indices = output.index[trip_mask].to_numpy()
            replace_mask.loc[trip_indices[missing_mask.to_numpy()]] = False
        output.loc[replace_mask, COORDINATE_COLUMNS] = patch.loc[
            ~missing_mask,
            COORDINATE_COLUMNS,
        ].to_numpy()
        replaced = int((~missing_mask).sum())
        patched_rows += replaced
        rows_by_trip[trip_id] = replaced

        if "SelectedSource" in patch.columns:
            for source, count in patch.loc[~missing_mask, "SelectedSource"].astype(str).value_counts().items():
                source_counts[str(source)] = source_counts.get(str(source), 0) + int(count)

    lat = output["LatitudeDegrees"].to_numpy(dtype=float)
    lon = output["LongitudeDegrees"].to_numpy(dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    coordinate_sanity_pass = bool(
        finite.all()
        and (lat >= 30.0).all()
        and (lat <= 40.0).all()
        and (lon >= -130.0).all()
        and (lon <= -110.0).all()
    )
    summary = {
        "rows": int(len(output)),
        "patched_rows": patched_rows,
        "interpolated_rows": interpolated_rows,
        "missing_rows": missing_rows,
        "rows_by_trip": rows_by_trip,
        "missing_by_trip": missing_by_trip,
        "selected_source_counts": dict(sorted(source_counts.items())),
        "coordinate_sanity_pass": coordinate_sanity_pass,
        "nonfinite_latlon_rows": int(np.count_nonzero(~finite)),
        "latitude_min": float(np.nanmin(lat)),
        "latitude_max": float(np.nanmax(lat)),
        "longitude_min": float(np.nanmin(lon)),
        "longitude_max": float(np.nanmax(lon)),
    }
    return output, summary


def build_config(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        motion_sigma_m=args.motion_sigma_m,
        factor_dt_max_s=args.factor_dt_max_s,
        fgo_iters=args.fgo_iters,
        weight_mode=getattr(args, "weight_mode", "sin2el"),
        fgo_weight_mode=(
            None
            if getattr(args, "fgo_weight_mode", "same") == "same"
            else getattr(args, "fgo_weight_mode", "same")
        ),
        fgo_robust_kernel=getattr(args, "fgo_robust_kernel", "huber"),
        fgo_cauchy_c_m=float(getattr(args, "fgo_cauchy_c_m", 4.0)),
        fgo_cauchy_outer_iters=int(getattr(args, "fgo_cauchy_outer_iters", 3)),
        per_type_kernel_enabled=bool(getattr(args, "per_type_kernel", False)),
        per_type_kernel_huber_enabled=bool(getattr(args, "per_type_kernel_huber", True)),
        per_type_kernel_motion_enabled=bool(getattr(args, "per_type_kernel_motion", False)),
        fgo_huber_k_pr=float(getattr(args, "fgo_huber_k_pr", 0.0)),
        gate_fgo_low_baseline_mse_pr_max=getattr(args, "gate_fgo_low_baseline_mse_pr_max", None),
        gate_fgo_baseline_mse_pr_min=getattr(args, "gate_fgo_baseline_mse_pr_min", None),
        pairwise_consistency_enabled=bool(getattr(args, "pairwise_consistency", False)),
        pairwise_consistency_mad_threshold_m=float(getattr(args, "pairwise_consistency_mad_threshold_m", 3.5)),
        pairwise_consistency_min_obs_after_filter=int(getattr(args, "pairwise_consistency_min_obs_after_filter", 5)),
        max_clique_filter_enabled=bool(getattr(args, "max_clique_filter", False)),
        max_clique_filter_pair_threshold_m=float(getattr(args, "max_clique_filter_pair_threshold_m", 3.0)),
        max_clique_filter_min_clique_size=int(getattr(args, "max_clique_filter_min_clique_size", 5)),
        hatch_smoothing_enabled=bool(getattr(args, "hatch_smoothing", False)),
        hatch_smoothing_n=int(getattr(args, "hatch_smoothing_n", 100)),
        use_rtklib_tropo=bool(getattr(args, "use_rtklib_tropo", False)),
        position_source=args.position_source,
        chunk_epochs=args.chunk_epochs,
        gated_baseline_threshold=args.gated_threshold,
        use_vd=args.vd,
        multi_gnss=args.multi_gnss,
        tdcp_enabled=args.tdcp,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        dual_frequency=args.dual_frequency,
        ct_rbpf_fgo_enabled=args.ct_rbpf_fgo or args.position_source == CT_RBPF_FGO_SOURCE,
        ct_rbpf_motion_sigma_m=args.ct_rbpf_motion_sigma_m,
        dd_carrier_fgo_enabled=args.dd_carrier_fgo or args.position_source == DD_CARRIER_FGO_SOURCE,
        dd_carrier_base_obs_template=args.dd_carrier_base_obs_template,
        dd_carrier_require_base_obs_template=args.dd_carrier_require_base_obs_template,
        dd_carrier_tow_snap_tolerance_s=args.dd_carrier_tow_snap_tolerance_s,
        dd_carrier_min_dd_pairs=args.dd_carrier_min_dd_pairs,
        dd_carrier_smooth_corrections=args.dd_carrier_smooth_corrections,
        dd_carrier_min_anchor_coverage=args.dd_carrier_min_anchor_coverage,
        fgo_raw_wls_proxy_rescue_enabled=args.fgo_raw_wls_proxy_rescue,
        fgo_raw_wls_proxy_rescue_phones=tuple(
            item.strip().lower()
            for item in args.fgo_raw_wls_proxy_rescue_phones.split(",")
            if item.strip()
        ),
        fgo_raw_wls_proxy_rescue_mse_ratio_max=args.fgo_raw_wls_proxy_rescue_mse_ratio_max,
        fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max=args.fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max,
        fgo_raw_wls_proxy_rescue_quality_delta_max=args.fgo_raw_wls_proxy_rescue_quality_delta_max,
        fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max=args.fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max,
    )


def run_one_bridge_trip(
    *,
    data_root: Path,
    sample_trip_id: str,
    max_epochs: int,
    start_epoch: int,
    config: BridgeConfig,
    bridge_output_root: Path | None,
    resume_existing: bool,
) -> dict[str, Any]:
    start = time.time()
    if bridge_output_root is not None and resume_existing:
        cached = load_cached_bridge_trip(bridge_output_root, sample_trip_id)
        if cached is not None:
            table, payload = cached
            return {
                "trip_id": sample_trip_id,
                "bridge_table": table,
                "metrics": payload,
                "cached": True,
                "elapsed_s": time.time() - start,
            }

    result = validate_raw_gsdc2023_trip(
        data_root,
        bridge_trip_id(sample_trip_id),
        max_epochs=max_epochs,
        start_epoch=start_epoch,
        config=config,
    )
    if bridge_output_root is not None:
        export_bridge_outputs(bridge_output_dir(bridge_output_root, sample_trip_id), result)
    return {
        "trip_id": sample_trip_id,
        "bridge_table": result.positions_table(),
        "metrics": result.metrics_payload(),
        "cached": False,
        "elapsed_s": time.time() - start,
    }


def _run_one_bridge_trip_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return run_one_bridge_trip(**payload)


def run_bridge_submission(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_path = args.sample_submission or (args.data_root / "sample_submission.csv")
    sample = pd.read_csv(sample_path)
    trips = ordered_trip_ids(sample)
    if args.trip:
        wanted = {normalize_sample_trip_id(trip) for trip in args.trip}
        trips = [trip for trip in trips if normalize_sample_trip_id(trip) in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no test trips selected")

    config = build_config(args)
    bridge_tables: dict[str, pd.DataFrame] = {}
    metrics: list[dict[str, Any]] = []
    total = len(trips)
    jobs = max(1, int(args.jobs))
    task_payloads = [
        {
            "data_root": args.data_root,
            "sample_trip_id": trip_id,
            "max_epochs": args.max_epochs,
            "start_epoch": args.start_epoch,
            "config": config,
            "bridge_output_root": args.bridge_output_root,
            "resume_existing": bool(args.resume_existing),
        }
        for trip_id in trips
    ]

    completed: list[dict[str, Any]] = []
    if jobs == 1:
        for payload in task_payloads:
            completed.append(run_one_bridge_trip(**payload))
            item = completed[-1]
            trip_id = str(item["trip_id"])
            metric_payload = item["metrics"]
            mode = "cached" if item["cached"] else "solved"
            print(
                f"[{len(completed)}/{total}] {trip_id} {mode} "
                f"epochs={metric_payload.get('n_epochs')} "
                f"sources={metric_payload.get('selected_source_counts')} "
                f"done in {float(item['elapsed_s']):.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_run_one_bridge_trip_from_payload, payload) for payload in task_payloads]
            for future in as_completed(futures):
                item = future.result()
                completed.append(item)
                trip_id = str(item["trip_id"])
                metric_payload = item["metrics"]
                mode = "cached" if item["cached"] else "solved"
                print(
                    f"[{len(completed)}/{total}] {trip_id} {mode} "
                    f"epochs={metric_payload.get('n_epochs')} "
                    f"sources={metric_payload.get('selected_source_counts')} "
                    f"done in {float(item['elapsed_s']):.1f}s",
                    flush=True,
                )

    for item in sorted(completed, key=lambda row: trips.index(str(row["trip_id"]))):
        trip_id = str(item["trip_id"])
        bridge_tables[trip_id] = item["bridge_table"]
        payload = item["metrics"]
        print(
            f"[assemble] {trip_id} cached={bool(item['cached'])}",
            flush=True,
        )
        metrics.append(payload)

    output, summary = submission_from_bridge_tables(
        sample,
        bridge_tables,
        allow_partial=args.allow_partial,
        interpolate_missing=args.interpolate_missing,
    )
    summary.update(
        {
            "data_root": str(args.data_root),
            "sample_submission": str(sample_path),
            "output": str(args.output),
            "bridge_output_root": None if args.bridge_output_root is None else str(args.bridge_output_root),
            "processed_trips": len(trips),
            "total_sample_trips": len(ordered_trip_ids(sample)),
            "allow_partial": bool(args.allow_partial),
            "interpolate_missing": bool(args.interpolate_missing),
            "jobs": jobs,
            "resume_existing": bool(args.resume_existing),
            "cached_trips": sum(1 for item in completed if bool(item["cached"])),
            "config": {
                "position_source": args.position_source,
                "chunk_epochs": args.chunk_epochs,
                "max_epochs": args.max_epochs,
                "dual_frequency": bool(args.dual_frequency),
                "ct_rbpf_fgo": bool(args.ct_rbpf_fgo or args.position_source == CT_RBPF_FGO_SOURCE),
                "ct_rbpf_motion_sigma_m": args.ct_rbpf_motion_sigma_m,
                "dd_carrier_fgo": bool(args.dd_carrier_fgo or args.position_source == DD_CARRIER_FGO_SOURCE),
                "dd_carrier_base_obs_template": args.dd_carrier_base_obs_template,
                "dd_carrier_require_base_obs_template": bool(args.dd_carrier_require_base_obs_template),
                "dd_carrier_tow_snap_tolerance_s": args.dd_carrier_tow_snap_tolerance_s,
                "dd_carrier_min_dd_pairs": args.dd_carrier_min_dd_pairs,
                "dd_carrier_smooth_corrections": bool(args.dd_carrier_smooth_corrections),
                "dd_carrier_min_anchor_coverage": float(args.dd_carrier_min_anchor_coverage),
                "fgo_raw_wls_proxy_rescue": bool(args.fgo_raw_wls_proxy_rescue),
                "fgo_raw_wls_proxy_rescue_phones": args.fgo_raw_wls_proxy_rescue_phones,
                "fgo_raw_wls_proxy_rescue_mse_ratio_max": args.fgo_raw_wls_proxy_rescue_mse_ratio_max,
                "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max": (
                    args.fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max
                ),
                "fgo_raw_wls_proxy_rescue_quality_delta_max": args.fgo_raw_wls_proxy_rescue_quality_delta_max,
                "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max": (
                    args.fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max
                ),
            },
            "trip_metrics": metrics,
        },
    )
    return output, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--bridge-output-root", type=Path)
    parser.add_argument("--jobs", type=int, default=1, help="number of test trips to process concurrently")
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reuse valid bridge outputs under --bridge-output-root instead of recomputing them",
    )
    parser.add_argument("--trip", action="append", default=[], help="sample tripId or test/... trip; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--interpolate-missing",
        action="store_true",
        help="linearly interpolate bridge coordinates for missing sample timestamps, clamping edge timestamps",
    )
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--motion-sigma-m", type=float, default=0.2)
    parser.add_argument("--factor-dt-max-s", type=float, default=FACTOR_DT_MAX_S)
    parser.add_argument("--fgo-iters", type=int, default=8)
    parser.add_argument(
        "--position-source",
        choices=("auto", "gated", "fgo", CT_RBPF_FGO_SOURCE, DD_CARRIER_FGO_SOURCE),
        default="gated",
    )
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--gated-threshold", type=float, default=GATED_BASELINE_THRESHOLD_DEFAULT)
    parser.add_argument(
        "--weight-mode",
        choices=("sin2el", "cn0", "taroz_sn"),
        default="sin2el",
        help="Gate/WLS pseudorange weight model.",
    )
    parser.add_argument(
        "--fgo-weight-mode",
        choices=("sin2el", "cn0", "taroz_sn", "same"),
        default="same",
        help="FGO-only weight model; 'same' uses --weight-mode.",
    )
    parser.add_argument(
        "--fgo-robust-kernel",
        choices=("huber", "cauchy"),
        default="huber",
        help="Robust kernel for FGO PR factor. 'cauchy' wraps the solver in Python-side IRLS.",
    )
    parser.add_argument("--fgo-cauchy-c-m", type=float, default=4.0)
    parser.add_argument("--fgo-cauchy-outer-iters", type=int, default=3)
    parser.add_argument(
        "--per-type-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable taroz per-Type Huber/motion overrides.",
    )
    parser.add_argument(
        "--per-type-kernel-huber",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--per-type-kernel-motion",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--fgo-huber-k-pr", type=float, default=0.0)
    parser.add_argument(
        "--gate-fgo-low-baseline-mse-pr-max",
        type=float,
        default=None,
        help="Override gate fgo mse_pr_max (default 9.3) to relax for Cauchy.",
    )
    parser.add_argument(
        "--gate-fgo-baseline-mse-pr-min",
        type=float,
        default=None,
        help="Override gate baseline mse_pr threshold for low-baseline FGO guard.",
    )
    parser.add_argument(
        "--pairwise-consistency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Step 3: TEASER-light pairwise consistency pre-filter on PR.",
    )
    parser.add_argument("--pairwise-consistency-mad-threshold-m", type=float, default=3.5)
    parser.add_argument("--pairwise-consistency-min-obs-after-filter", type=int, default=5)
    parser.add_argument(
        "--max-clique-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="lever 4: TEASER max-clique consensus pre-filter on PR.",
    )
    parser.add_argument("--max-clique-filter-pair-threshold-m", type=float, default=3.0)
    parser.add_argument("--max-clique-filter-min-clique-size", type=int, default=5)
    parser.add_argument(
        "--hatch-smoothing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hatch carrier-phase smoothing on raw PR before WLS/FGO.",
    )
    parser.add_argument("--hatch-smoothing-n", type=int, default=100)
    parser.add_argument(
        "--use-rtklib-tropo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Swap Android tropo for Saastamoinen recompute in raw PR.",
    )
    parser.add_argument("--vd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument(
        "--tdcp-geometry-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TDCP_GEOMETRY_CORRECTION,
    )
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ct-rbpf-fgo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add the CT-RBPF/FGO candidate source (fgo_ct_rbpf) to gated/direct selection",
    )
    parser.add_argument("--ct-rbpf-motion-sigma-m", type=float, default=DEFAULT_CT_RBPF_MOTION_SIGMA_M)
    parser.add_argument(
        "--dd-carrier-fgo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add sparse DD-carrier anchored FGO candidate source (fgo_dd_carrier)",
    )
    parser.add_argument(
        "--dd-carrier-base-obs-template",
        default=None,
        help="course-relative template such as '{base}_1hz.obs'",
    )
    parser.add_argument("--dd-carrier-require-base-obs-template", action="store_true")
    parser.add_argument("--dd-carrier-tow-snap-tolerance-s", type=float, default=0.6)
    parser.add_argument("--dd-carrier-min-dd-pairs", type=int, default=4)
    parser.add_argument("--dd-carrier-smooth-corrections", action="store_true")
    parser.add_argument(
        "--dd-carrier-min-anchor-coverage",
        type=float,
        default=DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT,
        help="per-trip DD-carrier anchor coverage gate (0..1); chunks emit DD-carrier only when accepted_anchor_epochs/n_epoch>=threshold",
    )
    parser.add_argument("--fgo-raw-wls-proxy-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fgo-raw-wls-proxy-rescue-phones", default="pixel4")
    parser.add_argument("--fgo-raw-wls-proxy-rescue-mse-ratio-max", type=float, default=1.20)
    parser.add_argument("--fgo-raw-wls-proxy-rescue-gap-step-p95-ratio-max", type=float, default=1.25)
    parser.add_argument("--fgo-raw-wls-proxy-rescue-quality-delta-max", type=float, default=-0.35)
    parser.add_argument("--fgo-raw-wls-proxy-rescue-mse-delta-vs-baseline-max", type=float, default=0.0)
    parser.add_argument(
        "--hampel-postprocess",
        action="store_true",
        help="apply Hampel filter to the final lat/lng trajectory after gate selection",
    )
    parser.add_argument("--hampel-postprocess-window", type=int, default=21)
    parser.add_argument("--hampel-postprocess-k", type=float, default=2.5)
    parser.add_argument("--hampel-postprocess-passes", type=int, default=3)
    parser.add_argument("--hampel-postprocess-mad-floor-deg", type=float, default=5e-7)
    parser.add_argument(
        "--accel-smoother",
        action="store_true",
        help="apply motion-acceleration smoother (after Hampel) on the final trajectory",
    )
    parser.add_argument("--accel-smoother-accel-max", type=float, default=3.0)
    parser.add_argument("--accel-smoother-passes", type=int, default=2)
    parser.add_argument(
        "--stop-snap",
        action="store_true",
        help="apply stationary-segment median snap (after Hampel + accel) on the final trajectory",
    )
    parser.add_argument("--stop-snap-move-threshold-m", type=float, default=2.0)
    parser.add_argument("--stop-snap-min-run-length", type=int, default=10)
    parser.add_argument(
        "--heading-smoother",
        action="store_true",
        help="apply heading-consistency smoother (after Hampel/accel/snap) on the final trajectory",
    )
    parser.add_argument("--heading-smoother-max-dps", type=float, default=45.0)
    parser.add_argument(
        "--kalman-smoother",
        action="store_true",
        help="apply 1D RTS Kalman CV smoother (after Hampel/accel/snap/heading) on the final trajectory",
    )
    parser.add_argument("--kalman-smoother-sigma-a", type=float, default=1.0)
    parser.add_argument("--kalman-smoother-sigma-z", type=float, default=1.0)
    args = parser.parse_args(argv)

    output, summary = run_bridge_submission(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, "hampel_postprocess", False):
        from experiments.postprocess_gsdc2023_submission_hampel import (
            apply_hampel_to_submission,
        )
        output, hampel_stats = apply_hampel_to_submission(
            output,
            window=int(getattr(args, "hampel_postprocess_window", 21)),
            k=float(getattr(args, "hampel_postprocess_k", 2.5)),
            mad_floor_deg=float(getattr(args, "hampel_postprocess_mad_floor_deg", 5e-7)),
            passes=int(getattr(args, "hampel_postprocess_passes", 3)),
        )
        summary["hampel_postprocess"] = hampel_stats
    if getattr(args, "accel_smoother", False):
        from experiments.postprocess_gsdc2023_submission_accel_smooth import (
            apply_accel_smoothing_to_submission,
        )
        output, accel_stats = apply_accel_smoothing_to_submission(
            output,
            accel_max=float(getattr(args, "accel_smoother_accel_max", 3.0)),
            passes=int(getattr(args, "accel_smoother_passes", 2)),
        )
        summary["accel_smoother"] = accel_stats
    if getattr(args, "stop_snap", False):
        from experiments.postprocess_gsdc2023_submission_stop_snap import (
            apply_stop_snap_to_submission,
        )
        output, snap_stats = apply_stop_snap_to_submission(
            output,
            move_threshold_m=float(getattr(args, "stop_snap_move_threshold_m", 2.0)),
            min_run_length=int(getattr(args, "stop_snap_min_run_length", 10)),
        )
        summary["stop_snap"] = snap_stats
    if getattr(args, "heading_smoother", False):
        from experiments.postprocess_gsdc2023_submission_heading import (
            apply_heading_smoothing_to_submission,
        )
        output, hdg_stats = apply_heading_smoothing_to_submission(
            output,
            heading_max_dps=float(getattr(args, "heading_smoother_max_dps", 45.0)),
        )
        summary["heading_smoother"] = hdg_stats
    if getattr(args, "kalman_smoother", False):
        from experiments.postprocess_gsdc2023_submission_kalman import (
            apply_kalman_smoothing_to_submission,
        )
        output, kf_stats = apply_kalman_smoothing_to_submission(
            output,
            sigma_a=float(getattr(args, "kalman_smoother_sigma_a", 1.0)),
            sigma_z=float(getattr(args, "kalman_smoother_sigma_z", 1.0)),
        )
        summary["kalman_smoother"] = kf_stats
    output.to_csv(args.output, index=False)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote: {args.output}")
    print(f"wrote: {summary_path}")
    print(
        f"summary: patched={summary['patched_rows']}/{summary['rows']} "
        f"interpolated={summary['interpolated_rows']} "
        f"missing={summary['missing_rows']} sanity={summary['coordinate_sanity_pass']} "
        f"sources={summary['selected_source_counts']}",
    )
    if "hampel_postprocess" in summary:
        hp = summary["hampel_postprocess"]
        print(
            f"hampel: passes={hp['passes']} rows_changed={hp['rows_changed']}/"
            f"{hp['rows_total']} per_pass_changed={hp['per_pass_changed']}",
        )
    if "accel_smoother" in summary:
        ac = summary["accel_smoother"]
        print(
            f"accel : passes={ac['passes']} accel_max={ac['accel_max']} "
            f"rows_changed={ac['rows_changed']}/{ac['rows_total']} "
            f"per_pass_changed={ac['per_pass_changed']}",
        )
    if "stop_snap" in summary:
        sn = summary["stop_snap"]
        print(
            f"snap  : move_threshold={sn['move_threshold_m']} min_run={sn['min_run_length']} "
            f"rows_changed={sn['rows_changed']}/{sn['rows_total']} runs={sn['runs']}",
        )
    if "heading_smoother" in summary:
        hg = summary["heading_smoother"]
        print(
            f"hdg   : max_dps={hg['heading_max_dps']} "
            f"rows_changed={hg['rows_changed']}/{hg['rows_total']}",
        )
    if "kalman_smoother" in summary:
        kf = summary["kalman_smoother"]
        print(
            f"kalman: sigma_a={kf['sigma_a']} sigma_z={kf['sigma_z']} "
            f"rows_changed={kf['rows_changed']}/{kf['rows_total']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
