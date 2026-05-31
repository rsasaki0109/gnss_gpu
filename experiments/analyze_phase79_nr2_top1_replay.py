#!/usr/bin/env python3
"""Summarize Phase79 top1 overlay replay results."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from audit_nr2_profile_span_oracle import _distance_weights, _float, _write_csv  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _first_row(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    return rows[0]


def _pass(row: dict[str, str], pass_m: float) -> bool:
    error = _float(row, "emit_to_ref_m")
    return math.isfinite(error) and error <= pass_m


def _base_label(row: dict[str, str]) -> str:
    label = row.get("rtkdiag_selected_base_label", "")
    if label:
        return label
    return row.get("rtkdiag_selected_label", "").removesuffix("+rnk")


def _run_summary(
    *,
    scope: str,
    run_csv: Path,
    base_pass_m: float,
    total_m: float,
) -> dict[str, Any]:
    row = _first_row(run_csv)
    pass_m = float(row["honest_pass_m"])
    gain_m = pass_m - base_pass_m
    return {
        "scope": scope,
        "run_csv": str(run_csv),
        "honest_ppc_pct": float(row["honest_ppc_pct"]),
        "honest_pass_m": pass_m,
        "honest_total_m": float(row["honest_total_m"]),
        "gain_m": gain_m,
        "n2_delta_pp": 100.0 * gain_m / total_m if total_m > 0.0 else "",
        "official_delta_pp": 100.0 * gain_m / total_m / 6.0 if total_m > 0.0 else "",
    }


def _matched_gain(
    *,
    base_rows: list[dict[str, str]],
    replay_rows: list[dict[str, str]],
    trigger_tows: set[float],
    pass_m: float,
) -> tuple[float, float, Counter[str]]:
    base_by_tow = {round(float(row["tow"]), 1): row for row in base_rows}
    weights = _distance_weights(replay_rows)
    trigger_gain_m = 0.0
    nontrigger_gain_m = 0.0
    labels: Counter[str] = Counter()
    for weight, row in zip(weights, replay_rows):
        tow = round(float(row["tow"]), 1)
        base_row = base_by_tow.get(tow)
        if base_row is None:
            continue
        replay_pass = _pass(row, pass_m)
        base_pass = _pass(base_row, pass_m)
        delta_m = 0.0
        if replay_pass and not base_pass:
            delta_m = weight
        elif base_pass and not replay_pass:
            delta_m = -weight
        if tow in trigger_tows:
            trigger_gain_m += delta_m
            labels[_base_label(row)] += 1
        else:
            nontrigger_gain_m += delta_m
    return trigger_gain_m, nontrigger_gain_m, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-run-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase70_osmroad_overlay_nagoya_run2_full_runs.csv"),
    )
    parser.add_argument(
        "--base-internal-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase70_osmroad_overlay_nagoya_run2_full_internal_epochs.csv"),
    )
    parser.add_argument(
        "--native-run-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase79_nr2_top1_overlay_replay_runs.csv"),
    )
    parser.add_argument(
        "--native-internal-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase79_nr2_top1_overlay_replay_internal_epochs.csv"),
    )
    parser.add_argument(
        "--order-run-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase79_nr2_top1_overlay_order_replay_runs.csv"),
    )
    parser.add_argument(
        "--order-internal-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase79_nr2_top1_overlay_order_replay_internal_epochs.csv"),
    )
    parser.add_argument(
        "--alias-run-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase79_nr2_top1_overlay_alias_replay_runs.csv"),
    )
    parser.add_argument(
        "--alias-internal-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase79_nr2_top1_overlay_alias_replay_internal_epochs.csv"),
    )
    parser.add_argument(
        "--trigger-epochs-csv",
        type=Path,
        default=Path("experiments/results/phase79_nr2_top1_label_pair_overlay_trigger_epochs.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("experiments/results/phase79_nr2_top1_overlay_replay"),
    )
    parser.add_argument("--pass-m", type=float, default=0.5)
    args = parser.parse_args(argv)

    base_run = _first_row(args.base_run_csv)
    base_pass_m = float(base_run["honest_pass_m"])
    total_m = float(base_run["honest_total_m"])
    base_rows = _read_csv(args.base_internal_csv)
    trigger_tows = {
        round(float(row["tow"]), 1)
        for row in _read_csv(args.trigger_epochs_csv)
        if str(row.get("triggered", "")).lower() == "true"
    }

    summary_rows: list[dict[str, Any]] = [
        {
            "scope": "phase70_base",
            "run_csv": str(args.base_run_csv),
            "honest_ppc_pct": float(base_run["honest_ppc_pct"]),
            "honest_pass_m": base_pass_m,
            "honest_total_m": total_m,
            "gain_m": 0.0,
            "n2_delta_pp": 0.0,
            "official_delta_pp": 0.0,
            "triggered_epochs": len(trigger_tows),
        },
    ]
    label_rows: list[dict[str, Any]] = []
    for scope, run_csv, internal_csv in (
        ("native_label", args.native_run_csv, args.native_internal_csv),
        ("phase79_high_score_bypass", args.order_run_csv, args.order_internal_csv),
        ("alias_bypass_diagnostic", args.alias_run_csv, args.alias_internal_csv),
    ):
        replay_rows = _read_csv(internal_csv)
        trigger_gain_m, nontrigger_gain_m, labels = _matched_gain(
            base_rows=base_rows,
            replay_rows=replay_rows,
            trigger_tows=trigger_tows,
            pass_m=float(args.pass_m),
        )
        row = _run_summary(
            scope=scope,
            run_csv=run_csv,
            base_pass_m=base_pass_m,
            total_m=total_m,
        )
        row["trigger_gain_m"] = trigger_gain_m
        row["nontrigger_gain_m"] = nontrigger_gain_m
        row["trigger_selected_labels"] = ",".join(f"{label}:{count}" for label, count in labels.most_common())
        summary_rows.append(row)
        for label, count in labels.most_common():
            label_rows.append({"scope": scope, "label": label, "trigger_epochs": count})

    _write_csv(Path(f"{args.out_prefix}_summary.csv"), summary_rows)
    _write_csv(Path(f"{args.out_prefix}_trigger_label_counts.csv"), label_rows)
    for row in summary_rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
