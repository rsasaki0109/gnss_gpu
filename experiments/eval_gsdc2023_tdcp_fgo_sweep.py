#!/usr/bin/env python3
"""Sweep GSDC2023 TDCP/FGO knobs on train trips."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.eval_gsdc2023_ct_rbpf_fgo import discover_train_trips, parse_float_list, score_delta  # noqa: E402
from experiments.gsdc2023_chunk_selection import (  # noqa: E402
    GATED_CANDIDATE_QUALITY_MARGIN,
    GATED_FGO_BASELINE_GAP_P95_FLOOR_M,
)
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    BridgeConfig,
    DEFAULT_MOTION_SIGMA_M,
    DEFAULT_ROOT,
    FACTOR_DT_MAX_S,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    validate_raw_gsdc2023_trip,
)
from experiments.gsdc2023_tdcp import (  # noqa: E402
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
)


DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_tdcp_fgo_sweep_20260520.csv")


def parse_bool_list(text: str) -> list[bool]:
    values: list[bool] = []
    truthy = {"1", "true", "yes", "on", "y"}
    falsy = {"0", "false", "no", "off", "n"}
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in truthy:
            values.append(True)
        elif item in falsy:
            values.append(False)
        else:
            raise argparse.ArgumentTypeError(f"expected boolean value, got {raw!r}")
    if not values:
        raise argparse.ArgumentTypeError("expected at least one boolean")
    return values


def variant_name(*, scale: float, threshold: float, geometry: bool) -> str:
    geom = "geom" if geometry else "nogeom"
    return f"tdcp_s{scale:g}_thr{threshold:g}_{geom}".replace(".", "p")


def tdcp_variant_grid(scales: list[float], thresholds: list[float], geometries: list[bool]) -> list[tuple[float, float, bool]]:
    variants = [
        (float(scale), float(threshold), bool(geometry))
        for scale in scales
        for threshold in thresholds
        for geometry in geometries
    ]
    default = (DEFAULT_TDCP_WEIGHT_SCALE, DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M, DEFAULT_TDCP_GEOMETRY_CORRECTION)
    if default in variants:
        variants.remove(default)
        variants.insert(0, default)
    return variants


def _as_records(payload: dict[str, object]) -> list[dict[str, object]]:
    records = payload.get("chunk_selection_records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _float_value(payload: dict[str, object] | None, key: str) -> float | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def candidate_guard_diagnostics(payload: dict[str, object], source: str) -> dict[str, object]:
    count = 0
    raw_wls_mse_block = 0
    baseline_gap_ok = 0
    quality_margin_ok = 0
    mse_values: list[float] = []
    quality_values: list[float] = []
    baseline_gap_p95_values: list[float] = []
    baseline_gap_step_p95_ratios: list[float] = []
    raw_wls_mse_delta: list[float] = []
    raw_wls_mse_ratios: list[float] = []
    mse_delta_vs_baseline: list[float] = []
    quality_delta_vs_baseline: list[float] = []
    mse_delta_vs_fgo: list[float] = []
    quality_delta_vs_fgo: list[float] = []

    for record in _as_records(payload):
        candidates = record.get("candidates")
        if not isinstance(candidates, dict):
            continue
        quality = candidates.get(source)
        baseline = candidates.get("baseline")
        raw_wls = candidates.get("raw_wls")
        fgo = candidates.get("fgo")
        if not isinstance(quality, dict) or not isinstance(baseline, dict):
            continue
        count += 1

        quality_mse = _float_value(quality, "mse_pr")
        quality_score = _float_value(quality, "quality_score")
        baseline_mse = _float_value(baseline, "mse_pr")
        baseline_score = _float_value(baseline, "quality_score")
        baseline_step_p95 = _float_value(baseline, "step_p95_m")
        baseline_gap_p95 = _float_value(quality, "baseline_gap_p95_m")
        raw_wls_mse = _float_value(raw_wls, "mse_pr") if isinstance(raw_wls, dict) else None
        fgo_mse = _float_value(fgo, "mse_pr") if isinstance(fgo, dict) else None
        fgo_score = _float_value(fgo, "quality_score") if isinstance(fgo, dict) else None

        if quality_mse is not None and raw_wls_mse is not None and quality_mse > raw_wls_mse:
            raw_wls_mse_block += 1
        if quality_mse is not None:
            mse_values.append(quality_mse)
        if quality_score is not None:
            quality_values.append(quality_score)
        if baseline_gap_p95 is not None:
            baseline_gap_p95_values.append(baseline_gap_p95)
        if baseline_gap_p95 is not None and baseline_step_p95 is not None:
            if baseline_gap_p95 <= max(baseline_step_p95, GATED_FGO_BASELINE_GAP_P95_FLOOR_M):
                baseline_gap_ok += 1
            baseline_gap_step_p95_ratios.append(baseline_gap_p95 / max(baseline_step_p95, 1.0e-9))
        if (
            quality_mse is not None
            and baseline_mse is not None
            and quality_score is not None
            and baseline_score is not None
            and quality_mse <= baseline_mse * 1.12
            and quality_score + GATED_CANDIDATE_QUALITY_MARGIN < baseline_score
        ):
            quality_margin_ok += 1
        if quality_mse is not None and baseline_mse is not None:
            mse_delta_vs_baseline.append(quality_mse - baseline_mse)
        if quality_mse is not None and raw_wls_mse is not None:
            raw_wls_mse_delta.append(quality_mse - raw_wls_mse)
            raw_wls_mse_ratios.append(quality_mse / max(raw_wls_mse, 1.0e-9))
        if quality_score is not None and baseline_score is not None:
            quality_delta_vs_baseline.append(quality_score - baseline_score)
        if quality_mse is not None and fgo_mse is not None:
            mse_delta_vs_fgo.append(quality_mse - fgo_mse)
        if quality_score is not None and fgo_score is not None:
            quality_delta_vs_fgo.append(quality_score - fgo_score)

    prefix = source.replace("fgo_", "")
    return {
        f"{prefix}_candidate_chunks": count,
        f"{prefix}_raw_wls_mse_block_chunks": raw_wls_mse_block,
        f"{prefix}_baseline_gap_ok_chunks": baseline_gap_ok,
        f"{prefix}_quality_margin_ok_chunks": quality_margin_ok,
        f"{prefix}_mean_mse_pr": _mean(mse_values),
        f"{prefix}_mean_quality_score": _mean(quality_values),
        f"{prefix}_mean_baseline_gap_p95_m": _mean(baseline_gap_p95_values),
        f"{prefix}_mean_baseline_gap_step_p95_ratio": _mean(baseline_gap_step_p95_ratios),
        f"{prefix}_mean_mse_delta_vs_raw_wls": _mean(raw_wls_mse_delta),
        f"{prefix}_mean_mse_ratio_vs_raw_wls": _mean(raw_wls_mse_ratios),
        f"{prefix}_mean_mse_delta_vs_baseline": _mean(mse_delta_vs_baseline),
        f"{prefix}_mean_quality_delta_vs_baseline": _mean(quality_delta_vs_baseline),
        f"{prefix}_mean_mse_delta_vs_fgo": _mean(mse_delta_vs_fgo),
        f"{prefix}_mean_quality_delta_vs_fgo": _mean(quality_delta_vs_fgo),
    }


def metrics_row(variant: str, trip: str, payload: dict[str, object], base_payload: dict[str, object] | None) -> dict[str, object]:
    counts = payload.get("selected_source_counts")
    selected_counts = counts if isinstance(counts, dict) else {}
    row: dict[str, object] = {
        "variant": variant,
        "trip": trip,
        "n_epochs": payload.get("n_epochs"),
        "selected_source_mode": payload.get("selected_source_mode"),
        "selected_score_m": payload.get("selected_score_m"),
        "baseline_score_m": payload.get("kaggle_wls_score_m"),
        "raw_wls_score_m": payload.get("raw_wls_score_m"),
        "fgo_score_m": payload.get("fgo_score_m"),
        "selected_mse_pr": payload.get("selected_mse_pr"),
        "baseline_mse_pr": payload.get("baseline_mse_pr"),
        "raw_wls_mse_pr": payload.get("raw_wls_mse_pr"),
        "fgo_mse_pr": payload.get("fgo_mse_pr"),
        "tdcp_consistency_mask_count": payload.get("tdcp_consistency_mask_count"),
        "tdcp_weight_scale": payload.get("tdcp_weight_scale"),
        "tdcp_geometry_correction_applied": payload.get("tdcp_geometry_correction_applied"),
        "tdcp_geometry_correction_count": payload.get("tdcp_geometry_correction_count"),
        "selected_baseline_epochs": int(selected_counts.get("baseline", 0) or 0),
        "selected_raw_wls_epochs": int(selected_counts.get("raw_wls", 0) or 0),
        "selected_fgo_epochs": int(selected_counts.get("fgo", 0) or 0),
        "selected_fgo_no_tdcp_epochs": int(selected_counts.get("fgo_no_tdcp", 0) or 0),
        "selected_fgo_tdcp_scale_epochs": int(selected_counts.get("fgo_tdcp_scale", 0) or 0),
        "tdcp_scale_candidate_enabled": payload.get("tdcp_scale_candidate_enabled"),
        "tdcp_scale_candidate_weight_scale": payload.get("tdcp_scale_candidate_weight_scale"),
        "fgo_raw_wls_proxy_rescue_enabled": payload.get("fgo_raw_wls_proxy_rescue_enabled"),
        "fgo_raw_wls_proxy_rescue_mse_ratio_max": payload.get("fgo_raw_wls_proxy_rescue_mse_ratio_max"),
        "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max": payload.get(
            "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max",
        ),
        "fgo_raw_wls_proxy_rescue_quality_delta_max": payload.get("fgo_raw_wls_proxy_rescue_quality_delta_max"),
        "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max": payload.get(
            "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max",
        ),
    }
    row.update(candidate_guard_diagnostics(payload, "fgo"))
    row.update(candidate_guard_diagnostics(payload, "fgo_tdcp_scale"))
    if base_payload is not None:
        row["delta_selected_score_m_vs_default"] = score_delta(
            row["selected_score_m"],
            base_payload.get("selected_score_m"),
        )
        row["delta_selected_mse_pr_vs_default"] = score_delta(
            row["selected_mse_pr"],
            base_payload.get("selected_mse_pr"),
        )
    return row


def build_config(
    args: argparse.Namespace,
    *,
    tdcp_scale: float,
    tdcp_threshold: float,
    tdcp_geometry: bool,
) -> BridgeConfig:
    return BridgeConfig(
        motion_sigma_m=args.motion_sigma_m,
        factor_dt_max_s=args.factor_dt_max_s,
        fgo_iters=args.fgo_iters,
        position_source=args.position_source,
        chunk_epochs=args.chunk_epochs,
        gated_baseline_threshold=args.gated_threshold,
        use_vd=args.vd,
        multi_gnss=args.multi_gnss,
        tdcp_enabled=args.tdcp,
        tdcp_scale_candidate_enabled=args.tdcp_scale_candidate,
        tdcp_scale_candidate_weight_scale=args.tdcp_scale_candidate_weight_scale,
        tdcp_scale_candidate_phones=tuple(
            item.strip().lower()
            for item in args.tdcp_scale_candidate_phones.split(",")
            if item.strip()
        ),
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
        tdcp_consistency_threshold_m=tdcp_threshold,
        tdcp_weight_scale=tdcp_scale,
        tdcp_geometry_correction=tdcp_geometry,
        dual_frequency=args.dual_frequency,
    )


def run_eval(args: argparse.Namespace) -> pd.DataFrame:
    trips = discover_train_trips(args.data_root)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no train trips found")

    variants = tdcp_variant_grid(
        args.tdcp_weight_scale,
        args.tdcp_consistency_threshold_m,
        args.tdcp_geometry_correction,
    )
    rows: list[dict[str, object]] = []
    total = len(trips)
    for idx, trip in enumerate(trips, start=1):
        started = time.time()
        default_payload: dict[str, object] | None = None
        for scale, threshold, geometry in variants:
            config = build_config(
                args,
                tdcp_scale=float(scale),
                tdcp_threshold=float(threshold),
                tdcp_geometry=bool(geometry),
            )
            payload = validate_raw_gsdc2023_trip(
                args.data_root,
                trip,
                max_epochs=args.max_epochs,
                start_epoch=args.start_epoch,
                config=config,
            ).metrics_payload()
            name = variant_name(scale=float(scale), threshold=float(threshold), geometry=bool(geometry))
            is_default = (
                float(scale) == DEFAULT_TDCP_WEIGHT_SCALE
                and float(threshold) == DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M
                and bool(geometry) == DEFAULT_TDCP_GEOMETRY_CORRECTION
            )
            if is_default:
                default_payload = payload
            rows.append(metrics_row(name, trip, payload, default_payload))
        print(f"[{idx}/{total}] {trip} variants={len(variants)} done in {time.time() - started:.1f}s", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trip", action="append", default=[], help="train/.../phone trip; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--motion-sigma-m", type=float, default=DEFAULT_MOTION_SIGMA_M)
    parser.add_argument("--factor-dt-max-s", type=float, default=FACTOR_DT_MAX_S)
    parser.add_argument("--fgo-iters", type=int, default=8)
    parser.add_argument("--position-source", choices=("auto", "gated", "fgo"), default="gated")
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--gated-threshold", type=float, default=GATED_BASELINE_THRESHOLD_DEFAULT)
    parser.add_argument("--vd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-scale-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tdcp-scale-candidate-weight-scale", type=float, default=1.0e-7)
    parser.add_argument("--tdcp-scale-candidate-phones", default="pixel4,pixel4xl,mi8")
    parser.add_argument("--fgo-raw-wls-proxy-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fgo-raw-wls-proxy-rescue-phones", default="pixel4")
    parser.add_argument("--fgo-raw-wls-proxy-rescue-mse-ratio-max", type=float, default=1.20)
    parser.add_argument("--fgo-raw-wls-proxy-rescue-gap-step-p95-ratio-max", type=float, default=1.25)
    parser.add_argument("--fgo-raw-wls-proxy-rescue-quality-delta-max", type=float, default=-0.35)
    parser.add_argument("--fgo-raw-wls-proxy-rescue-mse-delta-vs-baseline-max", type=float, default=0.0)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-weight-scale", type=parse_float_list, default=[DEFAULT_TDCP_WEIGHT_SCALE])
    parser.add_argument("--tdcp-consistency-threshold-m", type=parse_float_list, default=[DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M])
    parser.add_argument("--tdcp-geometry-correction", type=parse_bool_list, default=[DEFAULT_TDCP_GEOMETRY_CORRECTION])
    args = parser.parse_args()

    frame = run_eval(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote: {args.output}")

    delta = pd.to_numeric(frame.get("delta_selected_score_m_vs_default"), errors="coerce")
    if delta.notna().any():
        print(f"score wins vs default: {int((delta < 0).sum())}/{int(delta.notna().sum())}")
        print(f"mean score delta vs default: {float(delta.mean()):.4f}m")


if __name__ == "__main__":
    main()
