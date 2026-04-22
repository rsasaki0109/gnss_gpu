"""CLI output helpers for PF smoother evaluation runs."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def select_pf_smoother_variants(
    *,
    compare_both: bool,
    use_smoother: bool,
) -> list[tuple[str, bool]]:
    if compare_both:
        return [("forward_only", False), ("with_smoother", True)]
    return [("forward_only" if not use_smoother else "with_smoother", bool(use_smoother))]


def print_pf_smoother_run_header(
    run_name: str,
    *,
    print_func: Callable[..., None] = print,
) -> None:
    print_func(f"\n{'='*60}\n  {run_name}\n{'='*60}")


def print_pf_smoother_variant_start(
    *,
    label: str,
    predict_guide: str,
    position_update_sigma: float | None,
    sigma_pos_tdcp: float | None,
    use_smoother: bool,
    print_func: Callable[..., None] = print,
) -> None:
    print_func(
        f"  [{label}] guide={predict_guide} PU={position_update_sigma} "
        f"sp_tdcp={sigma_pos_tdcp} smooth={use_smoother}...",
        end=" ",
        flush=True,
    )


def pf_smoother_variant_metrics(
    out: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int, float]:
    forward_metrics = out["forward_metrics"]
    smoothed_metrics = out["smoothed_metrics"]
    n_epochs = int(forward_metrics["n_epochs"]) if forward_metrics else 0
    ms_per_epoch = float(out["elapsed_ms"]) / n_epochs if n_epochs else 0.0
    return forward_metrics, smoothed_metrics, n_epochs, ms_per_epoch


def build_pf_smoother_variant_metric_lines(
    out: Mapping[str, Any],
    forward_metrics: Mapping[str, Any] | None,
    smoothed_metrics: Mapping[str, Any] | None,
    *,
    n_epochs: int,
    ms_per_epoch: float,
) -> list[str]:
    lines: list[str] = []
    if forward_metrics:
        lines.append(
            f"FWD P50={forward_metrics['p50']:.2f}m RMS={forward_metrics['rms_2d']:.2f}m "
            f"({n_epochs} ep, {ms_per_epoch:.2f}ms/ep)"
        )
    if smoothed_metrics:
        lines.append(
            f"       SMTH P50={smoothed_metrics['p50']:.2f}m "
            f"RMS={smoothed_metrics['rms_2d']:.2f}m"
        )
        if _int(out, "n_tail_guard_applied") > 0:
            lines.append(f"       tail guard applied: {_int(out, 'n_tail_guard_applied')} epochs")
        if _int(out, "n_widelane_forward_guard_applied") > 0:
            lines.append(
                "       wide-lane forward guard applied: "
                f"{_int(out, 'n_widelane_forward_guard_applied')} epochs"
            )
        if _int(out, "n_stop_segment_epochs_applied") > 0:
            stop_info = out.get("stop_segment_info") or {}
            lines.append(
                "       stop segment constant applied: "
                f"{_int(out, 'n_stop_segment_epochs_applied')} epochs "
                f"segments={stop_info.get('segments_applied', 0)}"
            )
        if _int(out, "n_stop_segment_static_epochs_applied") > 0:
            static_info = out.get("stop_segment_static_info") or {}
            lines.append(
                "       stop segment static GNSS applied: "
                f"{_int(out, 'n_stop_segment_static_epochs_applied')} epochs "
                f"segments={static_info.get('segments_applied', 0)}"
            )
        if out.get("fgo_local_applied"):
            info = out.get("fgo_local_info") or {}
            lines.append(
                f"       local FGO window: {info.get('window')} "
                f"solve={info.get('solve_window')}"
            )
            lambda_info = info.get("lambda") or {}
            if lambda_info:
                lines.append(
                    "       local FGO lambda: "
                    f"fixed={lambda_info.get('n_fixed', 0)} "
                    f"obs={lambda_info.get('n_fixed_observations', 0)} "
                    f"by_system={lambda_info.get('fixed_by_system', {})}"
                )
    return lines


def print_pf_smoother_variant_metrics(
    out: Mapping[str, Any],
    forward_metrics: Mapping[str, Any] | None,
    smoothed_metrics: Mapping[str, Any] | None,
    *,
    n_epochs: int,
    ms_per_epoch: float,
    print_func: Callable[..., None] = print,
) -> None:
    for line in build_pf_smoother_variant_metric_lines(
        out,
        forward_metrics,
        smoothed_metrics,
        n_epochs=n_epochs,
        ms_per_epoch=ms_per_epoch,
    ):
        print_func(line)


def write_pf_smoother_result_csv(
    rows: Sequence[Mapping[str, object]],
    out_csv: Path,
    *,
    print_func: Callable[..., None] = print,
) -> None:
    if not rows:
        return
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print_func(f"\nSaved: {out_csv}")


def _int(values: Mapping[str, Any], key: str) -> int:
    return int(values.get(key, 0) or 0)
