#!/usr/bin/env python3
# ruff: noqa: E402
"""Sweep realtime PPC fusion over fixed validation segments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_realtime_fusion import run_fusion_eval
from gnss_gpu.io.ppc import PPCDatasetLoader

RESULTS_DIR = _SCRIPT_DIR / "results"


@dataclass(frozen=True)
class SegmentSpec:
    city: str
    run: str
    start_epoch: int


@dataclass(frozen=True)
class FusionConfig:
    height_release_min_dd_shift_m: float
    height_release_on_dd_shift: bool
    height_reference_max_dd_rms_m: float
    dd_anchor_blend_alpha: float
    dd_anchor_high_blend_alpha: float
    dd_anchor_high_requires_untrusted_height: bool
    dd_anchor_high_min_shift_m: float
    dd_anchor_high_max_robust_rms_m: float
    last_velocity_max_age_s: float


POSITIVE_SEGMENTS = (
    SegmentSpec("tokyo", "run1", 1463),
    SegmentSpec("tokyo", "run2", 808),
    SegmentSpec("tokyo", "run3", 774),
    SegmentSpec("nagoya", "run1", 0),
    SegmentSpec("nagoya", "run2", 983),
    SegmentSpec("nagoya", "run3", 235),
)

HOLDOUT_SEGMENTS = (
    SegmentSpec("tokyo", "run1", 1663),
    SegmentSpec("tokyo", "run2", 1008),
    SegmentSpec("tokyo", "run3", 974),
    SegmentSpec("nagoya", "run1", 200),
    SegmentSpec("nagoya", "run2", 1183),
    SegmentSpec("nagoya", "run3", 35),
)


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse_float_list(raw: str, label: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"at least one {label} value is required")
    return values


def _config_label(config: FusionConfig) -> str:
    return (
        f"dd{config.dd_anchor_blend_alpha:g}_"
        f"ddhi{config.dd_anchor_high_blend_alpha:g}_"
        f"{'ddhiuntrusted_' if config.dd_anchor_high_requires_untrusted_height else ''}"
        f"hishift{config.dd_anchor_high_min_shift_m:g}_"
        f"hirms{config.dd_anchor_high_max_robust_rms_m:g}_"
        f"height_release_{config.height_release_min_dd_shift_m:g}m_"
        f"{'height_anydd_' if config.height_release_on_dd_shift else ''}"
        f"{'hrefmax' + format(config.height_reference_max_dd_rms_m, 'g') + '_' if np.isfinite(config.height_reference_max_dd_rms_m) else ''}"
        f"age{config.last_velocity_max_age_s:g}s"
    )


def _segments_for_preset(preset: str, data_root: Path, start_epoch: int) -> tuple[SegmentSpec, ...]:
    if preset == "positive":
        return POSITIVE_SEGMENTS
    if preset == "holdout":
        return HOLDOUT_SEGMENTS
    if preset == "all-runs":
        run_dirs = sorted(path for path in data_root.rglob("run*") if PPCDatasetLoader.is_run_directory(path))
        return tuple(SegmentSpec(path.parent.name, path.name, int(start_epoch)) for path in run_dirs)
    raise ValueError(f"unknown segment preset: {preset}")


def _fusion_kwargs(
    *,
    height_hold_release_min_dd_shift_m: float,
    height_hold_release_on_dd_shift: bool,
    height_hold_reference_max_dd_rms_m: float,
    dd_anchor_blend_alpha: float,
    dd_anchor_high_blend_alpha: float,
    dd_anchor_high_requires_untrusted_height: bool,
    dd_anchor_high_min_shift_m: float,
    dd_anchor_high_max_robust_rms_m: float,
    rsp_correction: bool,
    last_velocity_max_age_s: float,
) -> dict[str, object]:
    return {
        "tdcp_min_sats": 5,
        "tdcp_max_postfit_rms_m": 0.5,
        "tdcp_max_cycle_jump": 20000.0,
        "tdcp_max_velocity_mps": 50.0,
        "carrier_phase_sign": 1.0,
        "receiver_motion_sign": -1.0,
        "dd_huber_k_m": 1.0,
        "dd_trim_m": 1.5,
        "dd_min_kept_pairs": 5,
        "dd_max_shift_m": 200.0,
        "dd_anchor_blend_alpha": float(dd_anchor_blend_alpha),
        "dd_anchor_high_blend_alpha": float(dd_anchor_high_blend_alpha),
        "dd_anchor_high_requires_untrusted_height": bool(
            dd_anchor_high_requires_untrusted_height
        ),
        "dd_anchor_high_min_shift_m": float(dd_anchor_high_min_shift_m),
        "dd_anchor_high_max_robust_rms_m": float(dd_anchor_high_max_robust_rms_m),
        "dd_interpolate_base_epochs": True,
        "widelane": True,
        "widelane_min_epochs": 5,
        "widelane_max_std_cycles": 0.75,
        "widelane_ratio_threshold": 3.0,
        "widelane_min_fix_rate": 0.3,
        "widelane_min_kept_pairs": 3,
        "widelane_max_shift_m": 5.0,
        "widelane_max_robust_rms_m": 0.8,
        "widelane_veto_rms_band_min_m": 0.15,
        "widelane_veto_rms_band_max_m": 0.35,
        "widelane_veto_min_kept_pairs": 4,
        "widelane_anchor_blend_alpha": 1.0,
        "height_hold_alpha": 1.0,
        "height_hold_release_on_last_velocity": True,
        "height_hold_release_on_dd_shift": bool(height_hold_release_on_dd_shift),
        "height_hold_release_min_dd_shift_m": float(height_hold_release_min_dd_shift_m),
        "height_hold_reference_max_dd_rms_m": float(height_hold_reference_max_dd_rms_m),
        "rsp_correction": bool(rsp_correction),
        "rsp_n_particles": 64,
        "rsp_spread_m": 1.0,
        "rsp_sigma_m": 1.0,
        "rsp_huber_k_m": 1.0,
        "rsp_stein_steps": 1,
        "rsp_stein_step_size": 0.1,
        "rsp_repulsion_scale": 0.25,
        "rsp_min_dd_shift_m": 0.85,
        "rsp_max_dd_shift_m": 1.4,
        "rsp_min_dd_rms_m": 0.45,
        "rsp_max_dd_rms_m": 0.7,
        "rsp_random_seed": 42,
        "last_velocity_max_age_s": float(last_velocity_max_age_s),
    }


def _segment_row(
    spec: SegmentSpec,
    config: str,
    summary_rows: list[dict[str, object]],
    per_epoch: list[dict[str, object]],
) -> dict[str, object]:
    wls = summary_rows[0]
    fused = summary_rows[1]
    n_pass = sum(1 for row in per_epoch if bool(row["fused_ppc_pass"]))
    return {
        "city": spec.city,
        "run": spec.run,
        "start_epoch": int(spec.start_epoch),
        "segment": f"{spec.city}/{spec.run}@{spec.start_epoch}",
        "config": config,
        "n_epochs": int(fused["n_epochs"]),
        "ppc_score_pct": float(fused["ppc_score_pct"]),
        "ppc_pass_distance_m": float(fused["ppc_pass_distance_m"]),
        "ppc_total_distance_m": float(fused["ppc_total_distance_m"]),
        "ppc_epoch_pass_pct": float(fused["ppc_epoch_pass_pct"]),
        "ppc_pass_epochs": int(n_pass),
        "rms_2d": float(fused["rms_2d"]),
        "p50": float(fused["p50"]),
        "p95": float(fused["p95"]),
        "max_2d": float(fused["max_2d"]),
        "wls_ppc_score_pct": float(wls["ppc_score_pct"]),
        "wls_rms_2d": float(wls["rms_2d"]),
        "tdcp_used_epochs": int(fused["tdcp_used_epochs"]),
        "dd_pr_anchor_epochs": int(fused["dd_pr_anchor_epochs"]),
        "dd_anchor_blend_alpha": float(fused["dd_anchor_blend_alpha"]),
        "dd_anchor_high_blend_alpha": float(fused["dd_anchor_high_blend_alpha"]),
        "dd_anchor_high_requires_untrusted_height": bool(
            fused["dd_anchor_high_requires_untrusted_height"]
        ),
        "dd_anchor_high_min_shift_m": float(fused["dd_anchor_high_min_shift_m"]),
        "dd_anchor_high_max_robust_rms_m": float(fused["dd_anchor_high_max_robust_rms_m"]),
        "dd_anchor_high_blend_epochs": int(fused["dd_anchor_high_blend_epochs"]),
        "widelane_anchor_epochs": int(fused["widelane_anchor_epochs"]),
        "rsp_correction_epochs": int(fused["rsp_correction_epochs"]),
        "height_hold_release_min_dd_shift_m": float(fused["height_hold_release_min_dd_shift_m"]),
        "height_hold_release_on_dd_shift": bool(fused["height_hold_release_on_dd_shift"]),
        "height_hold_reference_max_dd_rms_m": float(
            fused["height_hold_reference_max_dd_rms_m"]
        ),
        "height_hold_reference_trusted": bool(fused["height_hold_reference_trusted"]),
        "last_velocity_max_age_s": float(fused["last_velocity_max_age_s"]),
    }


def _summarize_configs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for config in sorted({str(row["config"]) for row in rows}):
        selected = [row for row in rows if row["config"] == config]
        scores = np.array([float(row["ppc_score_pct"]) for row in selected], dtype=np.float64)
        total_pass_distance = float(sum(float(row["ppc_pass_distance_m"]) for row in selected))
        total_distance = float(sum(float(row["ppc_total_distance_m"]) for row in selected))
        total_pass_epochs = int(sum(int(row["ppc_pass_epochs"]) for row in selected))
        total_epochs = int(sum(int(row["n_epochs"]) for row in selected))
        summary.append(
            {
                "config": config,
                "n_segments": len(selected),
                "aggregate_ppc_score_pct": (
                    100.0 * total_pass_distance / total_distance if total_distance > 0.0 else 0.0
                ),
                "mean_ppc_score_pct": float(np.mean(scores)),
                "median_ppc_score_pct": float(np.median(scores)),
                "min_ppc_score_pct": float(np.min(scores)),
                "aggregate_epoch_pass_pct": 100.0 * total_pass_epochs / max(total_epochs, 1),
                "mean_rms_2d": float(np.mean([float(row["rms_2d"]) for row in selected])),
                "mean_p95": float(np.mean([float(row["p95"]) for row in selected])),
                "total_pass_distance_m": total_pass_distance,
                "total_distance_m": total_distance,
                "total_pass_epochs": total_pass_epochs,
                "total_epochs": total_epochs,
                "rsp_correction_epochs": int(
                    sum(int(row["rsp_correction_epochs"]) for row in selected)
                ),
                "height_hold_release_min_dd_shift_m": float(
                    selected[0]["height_hold_release_min_dd_shift_m"]
                ),
                "height_hold_release_on_dd_shift": bool(
                    selected[0]["height_hold_release_on_dd_shift"]
                ),
                "height_hold_reference_max_dd_rms_m": float(
                    selected[0]["height_hold_reference_max_dd_rms_m"]
                ),
                "height_hold_reference_untrusted_segments": int(
                    sum(
                        1
                        for row in selected
                        if not bool(row["height_hold_reference_trusted"])
                    )
                ),
                "dd_anchor_blend_alpha": float(selected[0]["dd_anchor_blend_alpha"]),
                "dd_anchor_high_blend_alpha": float(selected[0]["dd_anchor_high_blend_alpha"]),
                "dd_anchor_high_requires_untrusted_height": bool(
                    selected[0]["dd_anchor_high_requires_untrusted_height"]
                ),
                "dd_anchor_high_min_shift_m": float(
                    selected[0]["dd_anchor_high_min_shift_m"]
                ),
                "dd_anchor_high_max_robust_rms_m": float(
                    selected[0]["dd_anchor_high_max_robust_rms_m"]
                ),
                "dd_anchor_high_blend_epochs": int(
                    sum(int(row["dd_anchor_high_blend_epochs"]) for row in selected)
                ),
                "last_velocity_max_age_s": float(selected[0]["last_velocity_max_age_s"]),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep realtime PPC fusion over validation segments")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segment-preset", choices=("positive", "holdout", "all-runs"), default="positive")
    parser.add_argument("--start-epoch", type=int, default=1300, help="Used only with --segment-preset all-runs")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--systems", type=str, default="G,E,J")
    parser.add_argument(
        "--height-release-min-dd-shift-ms",
        type=str,
        default="0.4",
        help="Comma-separated stale-velocity height-release DD shift thresholds",
    )
    parser.add_argument(
        "--height-release-on-dd-shift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Release height hold on any accepted DD-PR shift above the release threshold",
    )
    parser.add_argument(
        "--height-reference-max-dd-rms-ms",
        type=str,
        default="inf",
        help="Comma-separated initial DD RMS ceilings for trusting the height reference",
    )
    parser.add_argument(
        "--dd-anchor-blend-alphas",
        type=str,
        default="0.3",
        help="Comma-separated DD-PR anchor blend values",
    )
    parser.add_argument(
        "--dd-anchor-high-blend-alphas",
        type=str,
        default="0.3",
        help="Comma-separated elevated DD-PR anchor blend values",
    )
    parser.add_argument(
        "--dd-anchor-high-requires-untrusted-height",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use elevated DD-PR blend only on segments with untrusted initial height",
    )
    parser.add_argument(
        "--dd-anchor-high-min-shift-ms",
        type=str,
        default="inf",
        help="Comma-separated DD shift thresholds for elevated DD blend",
    )
    parser.add_argument(
        "--dd-anchor-high-max-robust-rms-ms",
        type=str,
        default="0.7",
        help="Comma-separated DD robust RMS ceilings for elevated DD blend",
    )
    parser.add_argument(
        "--rsp-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use gated reservoir Stein DD likelihood correction",
    )
    parser.add_argument(
        "--last-velocity-max-age-ss",
        type=str,
        default="8.0",
        help="Comma-separated stale velocity bridge age limits",
    )
    parser.add_argument("--results-prefix", type=str, default="ppc_realtime_fusion_sweep")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    segments = _segments_for_preset(args.segment_preset, data_root, args.start_epoch)
    if not segments:
        raise FileNotFoundError(f"no PPC run directories found under: {data_root}")
    thresholds = _parse_float_list(args.height_release_min_dd_shift_ms, "height-release threshold")
    dd_alphas = _parse_float_list(args.dd_anchor_blend_alphas, "DD anchor blend alpha")
    height_reference_max_rms_values = _parse_float_list(
        args.height_reference_max_dd_rms_ms,
        "height-reference DD RMS ceiling",
    )
    dd_high_alphas = _parse_float_list(
        args.dd_anchor_high_blend_alphas,
        "elevated DD anchor blend alpha",
    )
    dd_high_min_shifts = _parse_float_list(
        args.dd_anchor_high_min_shift_ms,
        "elevated DD anchor minimum shift",
    )
    dd_high_max_rms_values = _parse_float_list(
        args.dd_anchor_high_max_robust_rms_ms,
        "elevated DD anchor robust RMS ceiling",
    )
    last_velocity_ages = _parse_float_list(args.last_velocity_max_age_ss, "last-velocity age")
    configs = tuple(
        FusionConfig(
            height_release_min_dd_shift_m=threshold,
            height_release_on_dd_shift=bool(args.height_release_on_dd_shift),
            height_reference_max_dd_rms_m=height_reference_max_rms,
            dd_anchor_blend_alpha=dd_alpha,
            dd_anchor_high_blend_alpha=dd_high_alpha,
            dd_anchor_high_requires_untrusted_height=bool(
                args.dd_anchor_high_requires_untrusted_height
            ),
            dd_anchor_high_min_shift_m=dd_high_min_shift,
            dd_anchor_high_max_robust_rms_m=dd_high_max_rms,
            last_velocity_max_age_s=last_velocity_age,
        )
        for threshold in thresholds
        for height_reference_max_rms in height_reference_max_rms_values
        for dd_alpha in dd_alphas
        for dd_high_alpha in dd_high_alphas
        for dd_high_min_shift in dd_high_min_shifts
        for dd_high_max_rms in dd_high_max_rms_values
        for last_velocity_age in last_velocity_ages
    )
    max_epochs = args.max_epochs if args.max_epochs and args.max_epochs > 0 else None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC Realtime Fusion Sweep")
    print("=" * 72)
    print(f"  Data root : {data_root}")
    print(f"  Preset    : {args.segment_preset}")
    print(f"  Segments  : {len(segments)}")
    print(f"  Max epochs: {max_epochs if max_epochs is not None else 'full'}")
    print(f"  Systems   : {','.join(systems)}")
    print(f"  Configs   : {len(configs)}")
    print(flush=True)

    rows: list[dict[str, object]] = []
    data_cache: dict[SegmentSpec, dict] = {}
    for spec in segments:
        run_dir = data_root / spec.city / spec.run
        if not PPCDatasetLoader.is_run_directory(run_dir):
            raise FileNotFoundError(f"not a PPC run directory: {run_dir}")
        print(f"[segment] {spec.city}/{spec.run} start={spec.start_epoch}", flush=True)
        if spec not in data_cache:
            data_cache[spec] = PPCDatasetLoader(run_dir).load_experiment_data(
                max_epochs=max_epochs,
                start_epoch=spec.start_epoch,
                systems=systems,
                include_sat_velocity=True,
            )
        data = data_cache[spec]
        for config_spec in configs:
            config = _config_label(config_spec)
            summary_rows, per_epoch, _arrays = run_fusion_eval(
                data,
                run_dir,
                systems,
                **_fusion_kwargs(
                    height_hold_release_min_dd_shift_m=config_spec.height_release_min_dd_shift_m,
                    height_hold_release_on_dd_shift=config_spec.height_release_on_dd_shift,
                    height_hold_reference_max_dd_rms_m=(
                        config_spec.height_reference_max_dd_rms_m
                    ),
                    dd_anchor_blend_alpha=config_spec.dd_anchor_blend_alpha,
                    dd_anchor_high_blend_alpha=config_spec.dd_anchor_high_blend_alpha,
                    dd_anchor_high_requires_untrusted_height=(
                        config_spec.dd_anchor_high_requires_untrusted_height
                    ),
                    dd_anchor_high_min_shift_m=config_spec.dd_anchor_high_min_shift_m,
                    dd_anchor_high_max_robust_rms_m=(
                        config_spec.dd_anchor_high_max_robust_rms_m
                    ),
                    rsp_correction=args.rsp_correction,
                    last_velocity_max_age_s=config_spec.last_velocity_max_age_s,
                ),
            )
            row = _segment_row(spec, config, summary_rows, per_epoch)
            rows.append(row)
            print(
                f"    {config:<22} ppc={row['ppc_score_pct']:.2f}%"
                f" pass={row['ppc_pass_epochs']}/{row['n_epochs']}"
                f" rms={row['rms_2d']:.2f}m p95={row['p95']:.2f}m"
                f" rsp={row['rsp_correction_epochs']}",
                flush=True,
            )
        print(flush=True)

    summary_rows = _summarize_configs(rows)
    run_path = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    summary_path = RESULTS_DIR / f"{args.results_prefix}_configs.csv"
    _write_rows(rows, run_path)
    _write_rows(summary_rows, summary_path)

    print("=" * 72)
    for row in summary_rows:
        print(
            f"  {row['config']:<22} aggregate={row['aggregate_ppc_score_pct']:.2f}%"
            f" mean={row['mean_ppc_score_pct']:.2f}%"
            f" pass={row['total_pass_epochs']}/{row['total_epochs']}"
            f" rsp={row['rsp_correction_epochs']}"
        )
    print(f"  Saved: {run_path}")
    print(f"  Saved: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
