#!/usr/bin/env python3
"""Audit existing gici-open reference outputs without integrating GPL code."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)

GICI_VARIANTS = (
    ("xd_gici_def", "gici_tc_esdfix"),
    ("xd_gici_z", "gici_full_zeroarm"),
    ("xd_gici_r", "gici_full_ratio25"),
    ("xd_gici_lp", "gici_full_loosepr"),
    ("xd_gici_lh", "gici_full_loosephase"),
    ("xd_gici_r4", "gici_full_ratio40"),
    ("xd_gici_combo", "gici_full_combo"),
    ("xd_gici_c4", "gici_full_combo4"),
    ("xd_gici_lprlph", "gici_full_lprlph"),
    ("xd_gici_zr", "gici_full_zr"),
    ("xd_gici_oa", "gici_full_onarm"),
    ("xd_gici_la", "gici_full_lowacc"),
    ("xd_gici_hs", "gici_full_hisnr"),
    ("xd_gici_hs45", "gici_full_hisnr45"),
    ("xd_gici_hs30", "gici_full_hisnr30"),
    ("xd_gici_he", "gici_full_hielev"),
    ("xd_gici_ir", "gici_full_imurot"),
    ("xd_gici_mb", "gici_full_himuba"),
    ("xd_gici_w5", "gici_full_window5"),
)


def parse_selected_counts(raw: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in str(raw or "").split(","):
        item = item.strip().strip('"')
        if not item or ":" not in item:
            continue
        label, count_text = item.rsplit(":", 1)
        label = label.removesuffix("+rnk")
        if not label:
            continue
        try:
            counts[label] += int(count_text)
        except ValueError:
            continue
    return counts


def _float_or_none(value: str | None) -> float | None:
    try:
        out = float(value if value is not None else "")
    except ValueError:
        return None
    return out if out == out else None


def audit_variant_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "rows": 0,
            "valid_rows": 0,
            "output_added_rows": 0,
            "status4_rows": 0,
            "status5_rows": 0,
            "median_residual_rms": "",
        }
    rows = 0
    valid_rows = 0
    output_added_rows = 0
    status_counts: Counter[int] = Counter()
    residuals: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            if row.get("final_valid") == "1":
                valid_rows += 1
            if row.get("output_added") == "1":
                output_added_rows += 1
            status = _float_or_none(row.get("final_status"))
            if status is not None:
                status_counts[int(status)] += 1
            residual = _float_or_none(row.get("final_residual_rms"))
            if residual is not None:
                residuals.append(residual)
    return {
        "exists": True,
        "rows": rows,
        "valid_rows": valid_rows,
        "output_added_rows": output_added_rows,
        "status4_rows": status_counts[4],
        "status5_rows": status_counts[5],
        "median_residual_rms": statistics.median(residuals) if residuals else "",
    }


def load_phase43_run(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.DictReader(fh), {})


def build_audit(
    *,
    results_dir: Path,
    phase43_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    label_to_dir = dict(GICI_VARIANTS)
    gici_labels = set(label_to_dir)
    for city, run in RUNS:
        key = f"{city}_{run}"
        phase43_path = results_dir / f"{phase43_prefix}_{key}_full_runs.csv"
        run_row = load_phase43_run(phase43_path)
        selected = parse_selected_counts(run_row.get("rtkdiag_pf_selected_counts", ""))
        total_selected = sum(selected.values())
        gici_selected = sum(count for label, count in selected.items() if label in gici_labels)
        summary_rows.append(
            {
                "city": city,
                "run": run,
                "phase43_exists": bool(run_row),
                "phase43_honest_ppc_pct": run_row.get("honest_ppc_pct", ""),
                "phase43_honest_pass_m": run_row.get("honest_pass_m", ""),
                "phase43_total_selected": total_selected,
                "phase43_gici_selected": gici_selected,
                "phase43_gici_selected_frac": (gici_selected / total_selected) if total_selected else "",
            },
        )
        for label, variant_dir in GICI_VARIANTS:
            diag_path = results_dir / "libgnss_diag_phase19" / variant_dir / f"{key}_full.csv"
            diag = audit_variant_csv(diag_path)
            selected_rows.append(
                {
                    "city": city,
                    "run": run,
                    "label": label,
                    "variant_dir": variant_dir,
                    "phase43_selected_count": selected[label],
                    **diag,
                },
            )
    return summary_rows, selected_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--phase43-prefix", default="ppc_ctrbpf_fgo_phase43_prod")
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("experiments/results/gici_reference_pool_phase56_summary.csv"),
    )
    parser.add_argument(
        "--out-variants",
        type=Path,
        default=Path("experiments/results/gici_reference_pool_phase56_variants.csv"),
    )
    args = parser.parse_args(argv)

    summary_rows, selected_rows = build_audit(
        results_dir=args.results_dir,
        phase43_prefix=args.phase43_prefix,
    )
    write_csv(args.out_summary, summary_rows)
    write_csv(args.out_variants, selected_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
