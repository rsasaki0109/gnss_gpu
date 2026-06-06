#!/usr/bin/env python3
"""Benchmark taroz FGO preset and ablations on GSDC2023 train trips."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_PYTHON = _REPO / "python"
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

from experiments.eval_gsdc2023_ct_rbpf_fgo import discover_train_trips, score_delta  # noqa: E402
from experiments.gsdc2023_bridge_config import (  # noqa: E402
    TAROZ_CLOCK_DRIFT_SIGMA_M,
    TAROZ_FGO_WEIGHT_MODE,
    TAROZ_HEIGHT_SIGMA_M,
    TAROZ_STOP_ATTITUDE_SIGMA_RAD,
    TAROZ_STOP_POSITION_SIGMA_M,
    TAROZ_STOP_VELOCITY_SIGMA_MPS,
    BridgeConfig,
    apply_taroz_fgo_preset,
    apply_taroz_gnss_only_preset,
)
from experiments.gsdc2023_per_type_kernel import (  # noqa: E402
    PerTypeKernel,
    per_type_kernel_for,
    trip_type_from_data_root,
)
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_MOTION_SIGMA_M,
    DEFAULT_ROOT,
    FACTOR_DT_MAX_S,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    validate_raw_gsdc2023_trip,
)
from experiments.gsdc2023_tdcp import DEFAULT_TDCP_GEOMETRY_CORRECTION  # noqa: E402


DEFAULT_VARIANTS = (
    "baseline",
    "taroz_weights",
    "taroz_pr",
    "taroz_pr_d",
    "taroz_pr_d_l",
    "taroz_gnss_only",
    "taroz_phone_aware",
    "taroz_full",
)
TAROZ_GNSS_ABLATION_VARIANTS = (
    "taroz_gnss_no_motion",
    "taroz_gnss_no_clock",
    "taroz_gnss_no_tdcp",
    "taroz_gnss_no_clock_no_tdcp",
)
TAROZ_FULL_ABLATION_VARIANTS = (
    "taroz_full_no_motion",
    "taroz_full_no_clock",
    "taroz_full_clock0p2",
    "taroz_full_clock0p5",
    "taroz_full_clock1p0",
    "taroz_full_no_stop",
    "taroz_full_no_height",
    "taroz_full_no_priors",
)
VARIANT_CHOICES = DEFAULT_VARIANTS + TAROZ_GNSS_ABLATION_VARIANTS + TAROZ_FULL_ABLATION_VARIANTS
DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_taroz_fgo_benchmark.csv")


def parse_variants(text: str) -> tuple[str, ...]:
    variants = tuple(item.strip() for item in text.split(",") if item.strip())
    if not variants:
        raise argparse.ArgumentTypeError("expected at least one variant")
    unsupported = [variant for variant in variants if variant not in VARIANT_CHOICES]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported variant(s): {', '.join(unsupported)}; choices={', '.join(VARIANT_CHOICES)}",
        )
    return variants


def _phone_from_trip(trip: str) -> str:
    parts = Path(trip).parts
    return parts[-1] if parts else ""


def kernel_for_trip(data_root: Path, trip: str) -> tuple[str, PerTypeKernel]:
    trip_type = trip_type_from_data_root(data_root, trip)
    return trip_type, per_type_kernel_for(trip_type, phone=_phone_from_trip(trip))


def use_taroz_gnss_only_for_phone(phone: str) -> bool:
    return str(phone).lower().startswith("pixel")


def base_config_from_args(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        motion_sigma_m=float(args.motion_sigma_m),
        factor_dt_max_s=float(args.factor_dt_max_s),
        fgo_iters=int(args.fgo_iters),
        weight_mode=str(args.weight_mode),
        fgo_weight_mode=None if args.fgo_weight_mode == "same" else str(args.fgo_weight_mode),
        position_source=str(args.position_source),
        chunk_epochs=int(args.chunk_epochs),
        gated_baseline_threshold=float(args.gated_threshold),
        use_vd=bool(args.vd),
        multi_gnss=bool(args.multi_gnss),
        tdcp_enabled=bool(args.tdcp),
        tdcp_geometry_correction=bool(args.tdcp_geometry_correction),
        dual_frequency=bool(args.dual_frequency),
        apply_observation_mask=bool(args.apply_observation_mask),
    )


def config_for_variant(
    variant: str,
    base_config: BridgeConfig,
    kernel: PerTypeKernel,
    *,
    phone: str = "",
) -> BridgeConfig:
    """Build a per-trip config for one taroz ablation variant."""
    if variant == "baseline":
        return base_config

    cfg = replace(base_config, fgo_weight_mode=TAROZ_FGO_WEIGHT_MODE)
    if variant == "taroz_weights":
        return cfg
    if variant == "taroz_pr":
        return replace(cfg, fgo_huber_k_pr=kernel.pr_huber_k)
    if variant == "taroz_pr_d":
        return replace(
            cfg,
            fgo_huber_k_pr=kernel.pr_huber_k,
            fgo_huber_k_doppler=kernel.doppler_huber_k,
        )
    if variant == "taroz_pr_d_l":
        return replace(
            cfg,
            fgo_huber_k_pr=kernel.pr_huber_k,
            fgo_huber_k_doppler=kernel.doppler_huber_k,
            fgo_huber_k_tdcp=kernel.carrier_huber_k,
        )
    if variant == "taroz_gnss_only":
        return apply_taroz_gnss_only_preset(base_config)
    if variant == "taroz_phone_aware":
        if use_taroz_gnss_only_for_phone(phone):
            return apply_taroz_gnss_only_preset(base_config)
        return cfg
    if variant == "taroz_gnss_no_motion":
        return replace(
            apply_taroz_gnss_only_preset(base_config),
            per_type_kernel_motion_enabled=False,
        )
    if variant == "taroz_gnss_no_clock":
        return replace(
            apply_taroz_gnss_only_preset(base_config),
            clock_drift_sigma_m=base_config.clock_drift_sigma_m,
            clock_use_average_drift=base_config.clock_use_average_drift,
        )
    if variant == "taroz_gnss_no_tdcp":
        return replace(
            apply_taroz_gnss_only_preset(base_config),
            tdcp_enabled=False,
        )
    if variant == "taroz_gnss_no_clock_no_tdcp":
        return replace(
            apply_taroz_gnss_only_preset(base_config),
            clock_drift_sigma_m=base_config.clock_drift_sigma_m,
            clock_use_average_drift=base_config.clock_use_average_drift,
            tdcp_enabled=False,
        )
    if variant == "taroz_full":
        return apply_taroz_fgo_preset(base_config)
    if variant == "taroz_full_no_motion":
        return replace(
            apply_taroz_fgo_preset(base_config),
            per_type_kernel_motion_enabled=False,
        )
    if variant == "taroz_full_no_clock":
        return replace(
            apply_taroz_fgo_preset(base_config),
            clock_drift_sigma_m=base_config.clock_drift_sigma_m,
        )
    if variant == "taroz_full_clock0p2":
        return replace(apply_taroz_fgo_preset(base_config), clock_drift_sigma_m=0.2)
    if variant == "taroz_full_clock0p5":
        return replace(apply_taroz_fgo_preset(base_config), clock_drift_sigma_m=0.5)
    if variant == "taroz_full_clock1p0":
        return replace(apply_taroz_fgo_preset(base_config), clock_drift_sigma_m=1.0)
    if variant == "taroz_full_no_stop":
        return replace(
            apply_taroz_fgo_preset(base_config),
            stop_velocity_sigma_mps=base_config.stop_velocity_sigma_mps,
            stop_position_sigma_m=base_config.stop_position_sigma_m,
            stop_attitude_sigma_rad=base_config.stop_attitude_sigma_rad,
        )
    if variant == "taroz_full_no_height":
        return replace(
            apply_taroz_fgo_preset(base_config),
            graph_relative_height=base_config.graph_relative_height,
            relative_height_sigma_m=base_config.relative_height_sigma_m,
            apply_absolute_height=base_config.apply_absolute_height,
            absolute_height_sigma_m=base_config.absolute_height_sigma_m,
        )
    if variant == "taroz_full_no_priors":
        return replace(
            apply_taroz_fgo_preset(base_config),
            clock_drift_sigma_m=base_config.clock_drift_sigma_m,
            stop_velocity_sigma_mps=base_config.stop_velocity_sigma_mps,
            stop_position_sigma_m=base_config.stop_position_sigma_m,
            stop_attitude_sigma_rad=base_config.stop_attitude_sigma_rad,
            graph_relative_height=base_config.graph_relative_height,
            relative_height_sigma_m=base_config.relative_height_sigma_m,
            apply_absolute_height=base_config.apply_absolute_height,
            absolute_height_sigma_m=base_config.absolute_height_sigma_m,
        )
    raise ValueError(f"unsupported variant: {variant}")


def apply_matlab_residual_diagnostics_mask_config(
    cfg: BridgeConfig,
    *,
    data_root: Path,
    trip: str,
    enabled: bool,
) -> BridgeConfig:
    if not enabled:
        return cfg
    diagnostics_path = data_root / trip / "phone_data_residual_diagnostics.csv"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"MATLAB residual diagnostics mask not found: {diagnostics_path}")
    return replace(cfg, matlab_residual_diagnostics_mask_path=diagnostics_path)


def load_trip_file(path: Path) -> list[str]:
    trips: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        trips.append(item)
    return trips


def select_trips(args: argparse.Namespace) -> list[str]:
    if args.trip:
        trips = list(args.trip)
    elif args.trip_file is not None:
        trips = load_trip_file(args.trip_file)
    else:
        trips = discover_train_trips(args.data_root)

    if args.phone:
        wanted = tuple(phone.lower() for phone in args.phone)
        trips = [trip for trip in trips if any(phone in _phone_from_trip(trip).lower() for phone in wanted)]

    if args.trip_type:
        wanted_types = set(args.trip_type)
        trips = [
            trip for trip in trips
            if trip_type_from_data_root(args.data_root, trip) in wanted_types
        ]

    if args.sample_per_type > 0:
        sampled: list[str] = []
        seen_by_type: dict[str, int] = {}
        for trip in trips:
            trip_type = trip_type_from_data_root(args.data_root, trip)
            count = seen_by_type.get(trip_type, 0)
            if count >= args.sample_per_type:
                continue
            sampled.append(trip)
            seen_by_type[trip_type] = count + 1
        trips = sampled

    if args.limit > 0:
        trips = trips[: args.limit]
    return trips


def _metric(payload: dict[str, object], group: str, name: str) -> object:
    value = payload.get(group)
    if not isinstance(value, dict):
        return None
    return value.get(name)


def _source_count(payload: dict[str, object], source: str) -> int:
    counts = payload.get("selected_source_counts")
    if not isinstance(counts, dict):
        return 0
    return int(counts.get(source, 0) or 0)


def result_row(
    *,
    variant: str,
    trip: str,
    trip_type: str,
    kernel: PerTypeKernel,
    config: BridgeConfig,
    payload: dict[str, object],
    baseline_payload: dict[str, object] | None,
    elapsed_s: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "trip": trip,
        "trip_type": trip_type,
        "phone": _phone_from_trip(trip),
        "status": "ok",
        "elapsed_s": float(elapsed_s),
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
        "selected_p50_m": _metric(payload, "selected_metrics", "p50_m"),
        "selected_p95_m": _metric(payload, "selected_metrics", "p95_m"),
        "fgo_p50_m": _metric(payload, "fgo_metrics", "p50_m"),
        "fgo_p95_m": _metric(payload, "fgo_metrics", "p95_m"),
        "selected_baseline_epochs": _source_count(payload, "baseline"),
        "selected_raw_wls_epochs": _source_count(payload, "raw_wls"),
        "selected_fgo_epochs": _source_count(payload, "fgo"),
        "selected_auto_epochs": _source_count(payload, "auto"),
        "selected_fgo_no_tdcp_epochs": _source_count(payload, "fgo_no_tdcp"),
        "fgo_iters": payload.get("fgo_iters"),
        "failed_chunks": payload.get("failed_chunks"),
        "vd_seed_guard_skipped_segments": payload.get("vd_seed_guard_skipped_segments"),
        "vd_seed_guard_skipped_epochs": payload.get("vd_seed_guard_skipped_epochs"),
        "factor_dt_gap_count": payload.get("factor_dt_gap_count"),
        "tdcp_consistency_mask_count": payload.get("tdcp_consistency_mask_count"),
        "tdcp_geometry_correction_count": payload.get("tdcp_geometry_correction_count"),
        "kernel_pr_huber_k": kernel.pr_huber_k,
        "kernel_doppler_huber_k": kernel.doppler_huber_k,
        "kernel_tdcp_huber_k": kernel.carrier_huber_k,
        "kernel_motion_sigma_m": kernel.motion_sigma_m,
        "config_motion_sigma_m": config.motion_sigma_m,
        "config_fgo_weight_mode": config.fgo_weight_mode or config.weight_mode,
        "config_fgo_huber_k_pr": config.fgo_huber_k_pr,
        "config_fgo_huber_k_doppler": config.fgo_huber_k_doppler,
        "config_fgo_huber_k_tdcp": config.fgo_huber_k_tdcp,
        "config_per_type_kernel": config.per_type_kernel_enabled,
        "config_per_type_motion": config.per_type_kernel_motion_enabled,
        "config_clock_drift_sigma_m": config.clock_drift_sigma_m,
        "config_clock_use_average_drift": config.clock_use_average_drift,
        "config_stop_velocity_sigma_mps": config.stop_velocity_sigma_mps,
        "config_stop_position_sigma_m": config.stop_position_sigma_m,
        "config_stop_attitude_sigma_rad": config.stop_attitude_sigma_rad,
        "config_graph_relative_height": config.graph_relative_height,
        "config_relative_height_sigma_m": config.relative_height_sigma_m,
        "config_apply_absolute_height": config.apply_absolute_height,
        "config_absolute_height_sigma_m": config.absolute_height_sigma_m,
        "config_tdcp_enabled": config.tdcp_enabled,
        "config_tdcp_geometry_correction": config.tdcp_geometry_correction,
        "config_apply_observation_mask": config.apply_observation_mask,
        "config_matlab_residual_diagnostics_mask_path": (
            str(config.matlab_residual_diagnostics_mask_path)
            if config.matlab_residual_diagnostics_mask_path is not None
            else None
        ),
    }
    if baseline_payload is not None:
        row["delta_selected_score_m_vs_baseline"] = score_delta(
            row["selected_score_m"],
            baseline_payload.get("selected_score_m"),
        )
        row["delta_selected_mse_pr_vs_baseline"] = score_delta(
            row["selected_mse_pr"],
            baseline_payload.get("selected_mse_pr"),
        )
        row["delta_fgo_score_m_vs_baseline"] = score_delta(
            row["fgo_score_m"],
            baseline_payload.get("fgo_score_m"),
        )
        row["delta_fgo_mse_pr_vs_baseline"] = score_delta(
            row["fgo_mse_pr"],
            baseline_payload.get("fgo_mse_pr"),
        )
    return row


def error_row(
    *,
    variant: str,
    trip: str,
    trip_type: str,
    kernel: PerTypeKernel,
    config: BridgeConfig,
    elapsed_s: float,
    error: Exception,
) -> dict[str, object]:
    return {
        "variant": variant,
        "trip": trip,
        "trip_type": trip_type,
        "phone": _phone_from_trip(trip),
        "status": "error",
        "elapsed_s": float(elapsed_s),
        "error_type": type(error).__name__,
        "error": str(error),
        "kernel_pr_huber_k": kernel.pr_huber_k,
        "kernel_doppler_huber_k": kernel.doppler_huber_k,
        "kernel_tdcp_huber_k": kernel.carrier_huber_k,
        "kernel_motion_sigma_m": kernel.motion_sigma_m,
        "config_motion_sigma_m": config.motion_sigma_m,
        "config_fgo_weight_mode": config.fgo_weight_mode or config.weight_mode,
        "config_fgo_huber_k_pr": config.fgo_huber_k_pr,
        "config_fgo_huber_k_doppler": config.fgo_huber_k_doppler,
        "config_fgo_huber_k_tdcp": config.fgo_huber_k_tdcp,
        "config_per_type_kernel": config.per_type_kernel_enabled,
        "config_per_type_motion": config.per_type_kernel_motion_enabled,
        "config_tdcp_enabled": config.tdcp_enabled,
        "config_tdcp_geometry_correction": config.tdcp_geometry_correction,
        "config_apply_observation_mask": config.apply_observation_mask,
    }


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ok = frame[frame["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    numeric_cols = [
        "selected_score_m",
        "selected_mse_pr",
        "fgo_score_m",
        "fgo_mse_pr",
        "delta_selected_score_m_vs_baseline",
        "delta_selected_mse_pr_vs_baseline",
        "delta_fgo_score_m_vs_baseline",
        "delta_fgo_mse_pr_vs_baseline",
        "elapsed_s",
    ]
    for col in numeric_cols:
        if col not in ok.columns:
            ok[col] = np.nan
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    grouped = ok.groupby(["variant", "trip_type"], dropna=False)
    summary = grouped.agg(
        trip_count=("trip", "nunique"),
        run_count=("trip", "size"),
        mean_selected_score_m=("selected_score_m", "mean"),
        mean_selected_mse_pr=("selected_mse_pr", "mean"),
        mean_fgo_score_m=("fgo_score_m", "mean"),
        mean_fgo_mse_pr=("fgo_mse_pr", "mean"),
        mean_delta_selected_score_m_vs_baseline=("delta_selected_score_m_vs_baseline", "mean"),
        mean_delta_selected_mse_pr_vs_baseline=("delta_selected_mse_pr_vs_baseline", "mean"),
        mean_delta_fgo_score_m_vs_baseline=("delta_fgo_score_m_vs_baseline", "mean"),
        mean_delta_fgo_mse_pr_vs_baseline=("delta_fgo_mse_pr_vs_baseline", "mean"),
        total_elapsed_s=("elapsed_s", "sum"),
    ).reset_index()
    return summary.sort_values(["trip_type", "variant"]).reset_index(drop=True)


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    trips = select_trips(args)
    if not trips:
        raise RuntimeError("no train trips selected")

    variants = tuple(args.variants)
    base_config = base_config_from_args(args)
    rows: list[dict[str, object]] = []
    total_runs = len(trips) * len(variants)
    run_idx = 0
    for trip_idx, trip in enumerate(trips, start=1):
        trip_type, kernel = kernel_for_trip(args.data_root, trip)
        baseline_payload: dict[str, object] | None = None
        for variant in variants:
            run_idx += 1
            variant_cfg = config_for_variant(variant, base_config, kernel, phone=_phone_from_trip(trip))
            cfg = variant_cfg
            started = time.time()
            try:
                cfg = apply_matlab_residual_diagnostics_mask_config(
                    variant_cfg,
                    data_root=args.data_root,
                    trip=trip,
                    enabled=bool(args.use_matlab_residual_diagnostics_mask),
                )
                payload = validate_raw_gsdc2023_trip(
                    args.data_root,
                    trip,
                    max_epochs=int(args.max_epochs),
                    start_epoch=int(args.start_epoch),
                    config=cfg,
                ).metrics_payload()
                elapsed = time.time() - started
                if variant == "baseline":
                    baseline_payload = payload
                rows.append(
                    result_row(
                        variant=variant,
                        trip=trip,
                        trip_type=trip_type,
                        kernel=kernel,
                        config=cfg,
                        payload=payload,
                        baseline_payload=baseline_payload,
                        elapsed_s=elapsed,
                    ),
                )
                print(
                    f"[{run_idx}/{total_runs}] {trip} {variant} "
                    f"score={payload.get('selected_score_m')} mse={payload.get('selected_mse_pr')} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - exercised by CLI/data failures.
                elapsed = time.time() - started
                rows.append(
                    error_row(
                        variant=variant,
                        trip=trip,
                        trip_type=trip_type,
                        kernel=kernel,
                        config=cfg,
                        elapsed_s=elapsed,
                        error=exc,
                    ),
                )
                print(f"[{run_idx}/{total_runs}] {trip} {variant} ERROR {exc}", flush=True)
                if not args.keep_going:
                    raise
        print(f"[trip {trip_idx}/{len(trips)}] {trip} done", flush=True)
    return pd.DataFrame(rows)


def write_outputs(frame: pd.DataFrame, output: Path, *, args: argparse.Namespace) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = summarize_results(frame)
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    payload = {
        "data_root": str(args.data_root),
        "variants": list(args.variants),
        "rows": int(len(frame)),
        "ok_rows": int(frame["status"].eq("ok").sum()) if "status" in frame.columns else 0,
        "error_rows": int(frame["status"].eq("error").sum()) if "status" in frame.columns else 0,
        "output": str(output),
        "summary_output": str(summary_path),
        "taroz_constants": {
            "fgo_weight_mode": TAROZ_FGO_WEIGHT_MODE,
            "clock_drift_sigma_m": TAROZ_CLOCK_DRIFT_SIGMA_M,
            "stop_velocity_sigma_mps": TAROZ_STOP_VELOCITY_SIGMA_MPS,
            "stop_position_sigma_m": TAROZ_STOP_POSITION_SIGMA_M,
            "stop_attitude_sigma_rad": TAROZ_STOP_ATTITUDE_SIGMA_RAD,
            "height_sigma_m": TAROZ_HEIGHT_SIGMA_M,
        },
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def add_trip_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trip", action="append", default=[], help="train/<course>/<phone>; repeatable")
    parser.add_argument("--trip-file", type=Path, default=None, help="newline-delimited trip list")
    parser.add_argument("--phone", action="append", default=[], help="phone substring filter; repeatable")
    parser.add_argument("--trip-type", action="append", default=[], help="settings Type filter; repeatable")
    parser.add_argument("--sample-per-type", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=parse_variants, default=DEFAULT_VARIANTS)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--motion-sigma-m", type=float, default=DEFAULT_MOTION_SIGMA_M)
    parser.add_argument("--factor-dt-max-s", type=float, default=FACTOR_DT_MAX_S)
    parser.add_argument("--fgo-iters", type=int, default=8)
    parser.add_argument("--position-source", choices=("auto", "gated", "fgo"), default="gated")
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--gated-threshold", type=float, default=GATED_BASELINE_THRESHOLD_DEFAULT)
    parser.add_argument("--weight-mode", choices=("sin2el", "cn0", "taroz_sn"), default="sin2el")
    parser.add_argument("--fgo-weight-mode", choices=("same", "sin2el", "cn0", "taroz_sn"), default="same")
    parser.add_argument("--vd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--tdcp-geometry-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TDCP_GEOMETRY_CORRECTION,
    )
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-observation-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--use-matlab-residual-diagnostics-mask",
        action="store_true",
        help="apply per-trip Taroz phone_data_residual_diagnostics.csv factor-finite flags before FGO",
    )
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    add_trip_selection_args(parser)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    frame = run_benchmark(args)
    summary_path = write_outputs(frame, args.output, args=args)
    print(f"wrote rows: {args.output}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
