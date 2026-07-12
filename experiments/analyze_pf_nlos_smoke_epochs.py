#!/usr/bin/env python3
"""Summarize per-epoch PF internal diagnostics from PPC NLOS smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def _read_epochs(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    mask_epochs = [r for r in rows if _truthy(r.get("pf_nlos_mask_applied"))]
    emit_vals = [
        v
        for r in rows
        if (v := _float_or_none(r.get("emit_to_ref_m"))) is not None
    ]
    mask_emit_vals = [
        v
        for r in mask_epochs
        if (v := _float_or_none(r.get("emit_to_ref_m"))) is not None
    ]
    return {
        "epochs": len(rows),
        "mask_applied_epochs": len(mask_epochs),
        "emit_to_ref_median_m": _median(emit_vals),
        "mask_emit_to_ref_median_m": _median(mask_emit_vals),
        "emit_to_ref_mean_m": _mean(emit_vals),
        "mask_emit_to_ref_mean_m": _mean(mask_emit_vals),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _epoch_key(row: dict[str, str]) -> str:
    tow = row.get("tow") or row.get("gps_tow") or row.get("time") or ""
    epoch = row.get("epoch") or row.get("epoch_idx") or ""
    return f"{epoch}:{tow}"


def _paired_delta(
    baseline_rows: list[dict[str, str]],
    mask_rows: list[dict[str, str]],
) -> dict[str, object]:
    base_by_key = {_epoch_key(r): r for r in baseline_rows}
    deltas: list[float] = []
    mask_only: list[float] = []
    for row in mask_rows:
        if not _truthy(row.get("pf_nlos_mask_applied")):
            continue
        key = _epoch_key(row)
        base = base_by_key.get(key)
        if base is None:
            continue
        b_err = _float_or_none(base.get("emit_to_ref_m"))
        m_err = _float_or_none(row.get("emit_to_ref_m"))
        if b_err is None or m_err is None:
            continue
        delta = m_err - b_err
        deltas.append(delta)
        mask_only.append(m_err)
    improved = sum(1 for d in deltas if d < 0)
    worsened = sum(1 for d in deltas if d > 0)
    return {
        "paired_epochs": len(deltas),
        "emit_to_ref_delta_mean_m": _mean(deltas),
        "emit_to_ref_delta_median_m": _median(deltas),
        "epochs_improved": improved,
        "epochs_worsened": worsened,
        "epochs_unchanged": len(deltas) - improved - worsened,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-prefix", required=True)
    parser.add_argument("--mask-prefix", required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write summary JSON (default: {mask-prefix}_epoch_analysis.json)",
    )
    args = parser.parse_args(argv)

    baseline_path = RESULTS_DIR / f"{args.baseline_prefix}_internal_epochs.csv"
    mask_path = RESULTS_DIR / f"{args.mask_prefix}_internal_epochs.csv"
    baseline_rows = _read_epochs(baseline_path)
    mask_rows = _read_epochs(mask_path)
    if not baseline_rows and not mask_rows:
        print(
            f"[error] no internal epoch CSVs under {RESULTS_DIR}\n"
            f"  expected: {baseline_path.name}, {mask_path.name}",
            file=sys.stderr,
        )
        return 2

    summary: dict[str, object] = {
        "baseline_csv": str(baseline_path),
        "mask_csv": str(mask_path),
        "baseline": _summarize(baseline_rows),
        "mask_soft": _summarize(mask_rows),
        "paired": _paired_delta(baseline_rows, mask_rows),
    }
    out_json = args.out_json or (RESULTS_DIR / f"{args.mask_prefix}_epoch_analysis.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
