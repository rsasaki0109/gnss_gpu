#!/usr/bin/env python3
"""Sweep raw-bridge solver settings and summarize train/test trip diagnostics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import BridgeConfig, DEFAULT_ROOT, validate_raw_gsdc2023_trip


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    motion_sigma_m: float
    clock_drift_sigma_m: float
    use_vd: bool
    multi_gnss: bool
    tdcp_enabled: bool
    tdcp_consistency_threshold_m: float


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("expected at least one numeric value")
    return tuple(values)


def _trip_slug(trip: str) -> str:
    return trip.replace("/", "__")


def _build_cases(
    motion_sigmas: tuple[float, ...],
    clock_drift_sigmas: tuple[float, ...],
    tdcp_thresholds: tuple[float, ...],
) -> list[DiagnosticCase]:
    cases: list[DiagnosticCase] = []

    for motion_sigma_m in motion_sigmas:
        cases.append(
            DiagnosticCase(
                name=f"legacy_gps_m{motion_sigma_m:g}",
                motion_sigma_m=motion_sigma_m,
                clock_drift_sigma_m=0.0,
                use_vd=False,
                multi_gnss=False,
                tdcp_enabled=False,
                tdcp_consistency_threshold_m=0.0,
            ),
        )

    for motion_sigma_m in motion_sigmas:
        for clock_drift_sigma_m in clock_drift_sigmas:
            cases.append(
                DiagnosticCase(
                    name=f"vd_gps_m{motion_sigma_m:g}_cd{clock_drift_sigma_m:g}",
                    motion_sigma_m=motion_sigma_m,
                    clock_drift_sigma_m=clock_drift_sigma_m,
                    use_vd=True,
                    multi_gnss=False,
                    tdcp_enabled=False,
                    tdcp_consistency_threshold_m=0.0,
                ),
            )
            cases.append(
                DiagnosticCase(
                    name=f"vd_multi_m{motion_sigma_m:g}_cd{clock_drift_sigma_m:g}",
                    motion_sigma_m=motion_sigma_m,
                    clock_drift_sigma_m=clock_drift_sigma_m,
                    use_vd=True,
                    multi_gnss=True,
                    tdcp_enabled=False,
                    tdcp_consistency_threshold_m=0.0,
                ),
            )
            for tdcp_threshold_m in tdcp_thresholds:
                cases.append(
                    DiagnosticCase(
                        name=(
                            f"vd_multi_tdcp_m{motion_sigma_m:g}_cd{clock_drift_sigma_m:g}"
                            f"_t{tdcp_threshold_m:g}"
                        ),
                        motion_sigma_m=motion_sigma_m,
                        clock_drift_sigma_m=clock_drift_sigma_m,
                        use_vd=True,
                        multi_gnss=True,
                        tdcp_enabled=True,
                        tdcp_consistency_threshold_m=tdcp_threshold_m,
                    ),
                )
    return cases


def _score_value(result_dict: dict, key: str) -> float:
    value = result_dict.get(key)
    if value is None:
        return float("inf")
    return float(value)


def _result_row(case: DiagnosticCase, result) -> dict[str, object]:
    payload = result.metrics_payload()
    selected_metrics = payload["selected_metrics"] or {}
    baseline_metrics = payload["kaggle_wls_metrics"] or {}
    raw_metrics = payload["raw_wls_metrics"] or {}
    fgo_metrics = payload["fgo_metrics"] or {}
    return {
        "case": case.name,
        "trip": result.trip,
        "epochs": payload["n_epochs"],
        "max_sats": payload["max_sats"],
        "motion_sigma_m": case.motion_sigma_m,
        "clock_drift_sigma_m": case.clock_drift_sigma_m,
        "use_vd": case.use_vd,
        "multi_gnss": case.multi_gnss,
        "tdcp_enabled": case.tdcp_enabled,
        "tdcp_consistency_threshold_m": case.tdcp_consistency_threshold_m,
        "selected_source_mode": payload["selected_source_mode"],
        "selected_mse_pr": payload["selected_mse_pr"],
        "baseline_mse_pr": payload["baseline_mse_pr"],
        "raw_wls_mse_pr": payload["raw_wls_mse_pr"],
        "fgo_mse_pr": payload["fgo_mse_pr"],
        "selected_score_m": payload["selected_score_m"],
        "baseline_score_m": payload["kaggle_wls_score_m"],
        "raw_wls_score_m": payload["raw_wls_score_m"],
        "fgo_score_m": payload["fgo_score_m"],
        "selected_rms_2d_m": selected_metrics.get("rms_2d_m"),
        "baseline_rms_2d_m": baseline_metrics.get("rms_2d_m"),
        "raw_wls_rms_2d_m": raw_metrics.get("rms_2d_m"),
        "fgo_rms_2d_m": fgo_metrics.get("rms_2d_m"),
        "fgo_minus_baseline_score_m": _score_value(payload, "fgo_score_m") - _score_value(payload, "kaggle_wls_score_m"),
        "fgo_minus_raw_score_m": _score_value(payload, "fgo_score_m") - _score_value(payload, "raw_wls_score_m"),
        "failed_chunks": payload["failed_chunks"],
        "fgo_iters": payload["fgo_iters"],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _print_top(rows: list[dict[str, object]], *, top_k: int) -> None:
    ranked = sorted(rows, key=lambda row: float(row["fgo_score_m"]) if row["fgo_score_m"] is not None else float("inf"))
    print(f"Top {min(top_k, len(ranked))} by fgo_score_m")
    for row in ranked[:top_k]:
        print(
            f"  {row['case']}: "
            f"fgo_score={row['fgo_score_m']:.3f} "
            f"baseline={row['baseline_score_m']:.3f} "
            f"raw={row['raw_wls_score_m']:.3f} "
            f"delta_base={row['fgo_minus_baseline_score_m']:.3f} "
            f"delta_raw={row['fgo_minus_raw_score_m']:.3f} "
            f"iters={row['fgo_iters']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", required=True, help="relative trip path under data root")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--position-source", default="fgo", choices=("baseline", "raw_wls", "fgo", "auto", "gated"))
    parser.add_argument("--motion-sigmas", default="0.3,1.0,3.0")
    parser.add_argument("--clock-drift-sigmas", default="0.3,1.0,3.0")
    parser.add_argument("--tdcp-thresholds", default="1.0,1.5,2.5")
    parser.add_argument("--fgo-iters", type=int, default=8)
    parser.add_argument("--signal-type", default="GPS_L1_CA")
    parser.add_argument("--constellation-type", type=int, default=1)
    parser.add_argument("--weight-mode", default="sin2el", choices=("sin2el", "cn0"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--results-csv", type=Path, default=None)
    args = parser.parse_args()

    motion_sigmas = _parse_float_list(args.motion_sigmas)
    clock_drift_sigmas = _parse_float_list(args.clock_drift_sigmas)
    tdcp_thresholds = _parse_float_list(args.tdcp_thresholds)
    cases = _build_cases(motion_sigmas, clock_drift_sigmas, tdcp_thresholds)

    rows: list[dict[str, object]] = []
    for case in cases:
        print(f"[run] {case.name}", flush=True)
        result = validate_raw_gsdc2023_trip(
            args.data_root,
            args.trip,
            max_epochs=args.max_epochs,
            start_epoch=args.start_epoch,
            config=BridgeConfig(
                motion_sigma_m=case.motion_sigma_m,
                clock_drift_sigma_m=case.clock_drift_sigma_m,
                fgo_iters=args.fgo_iters,
                signal_type=args.signal_type,
                constellation_type=args.constellation_type,
                weight_mode=args.weight_mode,
                position_source=args.position_source,
                chunk_epochs=args.chunk_epochs,
                use_vd=case.use_vd,
                multi_gnss=case.multi_gnss,
                tdcp_enabled=case.tdcp_enabled,
                tdcp_consistency_threshold_m=case.tdcp_consistency_threshold_m,
            ),
        )
        row = _result_row(case, result)
        rows.append(row)
        print(
            f"  fgo_score={row['fgo_score_m']:.3f} "
            f"baseline={row['baseline_score_m']:.3f} "
            f"raw={row['raw_wls_score_m']:.3f} "
            f"delta_base={row['fgo_minus_baseline_score_m']:.3f}",
            flush=True,
        )

    if args.results_csv is None:
        slug = _trip_slug(args.trip)
        args.results_csv = _REPO / "experiments" / "results" / f"gsdc2023_bridge_diag_{slug}.csv"
    _write_csv(args.results_csv, rows)
    _print_top(rows, top_k=args.top_k)
    print(f"results_csv={args.results_csv}")


if __name__ == "__main__":
    main()
