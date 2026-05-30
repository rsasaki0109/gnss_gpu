#!/usr/bin/env python3
"""Build a Phase80 multi label-pair ranker overlay for n/r2 replay."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from audit_nr2_profile_span_oracle import _distance_weights, _float, _truth_xyz, _write_csv  # noqa: E402
from build_phase79_nr2_top1_label_pair_overlay import _base_label, _read_csv, _target_passes_rule  # noqa: E402
from exp_ppc_ctrbpf_fgo import _load_hybrid_pos_file, _load_rtk_diag_file  # noqa: E402
from run_phase73_nr2_selector_variant import PHASE71_LABEL_DIRS  # noqa: E402


def _target_label_to_dir(target_label: str) -> Path:
    parts = target_label.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"cannot convert target label to directory: {target_label}")
    return Path("experiments/results") / parts[0] / parts[1]


def _runtime_label_for_target_dir(target_dir: Path) -> str:
    target_resolved = (Path.cwd() / target_dir).resolve()
    for label, path in PHASE71_LABEL_DIRS:
        if (Path.cwd() / path).resolve() == target_resolved:
            return label
    raise ValueError(f"target directory is not in Phase71 runtime label set: {target_dir}")


def _xyz_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [
            _float(row, f"{prefix}_x"),
            _float(row, f"{prefix}_y"),
            _float(row, f"{prefix}_z"),
        ],
        dtype=np.float64,
    )


def _read_rules(path: Path, max_rules: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gain = _float(row, "best_gain_m")
            if not math.isfinite(gain) or gain <= 0.0:
                continue
            target_dir = _target_label_to_dir(row["target_label"])
            runtime_label = _runtime_label_for_target_dir(target_dir)
            rows.append(
                {
                    "rank": len(rows),
                    "source_label": row["selected_label"],
                    "target_profile_label": row["target_label"],
                    "target_runtime_label": runtime_label,
                    "target_dir": target_dir,
                    "best_gain_m": gain,
                    "offset_min_m": float(row["offset_min_m"]),
                    "offset_max_m": float(row["offset_max_m"]),
                    "family_span_min_m": float(row["family_span_min_m"]),
                    "agreement_1m_max": float(row["agreement_1m_max"]),
                    "target_rms_max": float(row["target_rms_max"]),
                },
            )
            if len(rows) >= max_rules:
                break
    if not rows:
        raise ValueError(f"no positive rules loaded from {path}")
    return rows


def _build_trigger_rows(args: argparse.Namespace, rules: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    internal_rows = _read_csv(args.internal_epochs_csv)
    weights = _distance_weights(internal_rows)
    loaded_targets: dict[str, tuple[dict[float, tuple[float, float, float]], dict[float, dict[str, str]]]] = {}
    for rule in rules:
        key = str(rule["target_profile_label"])
        if key in loaded_targets:
            continue
        target_dir = Path(rule["target_dir"])
        target_pos, _target_status = _load_hybrid_pos_file(target_dir / f"{args.city}_{args.run}_full.pos")
        target_diag = _load_rtk_diag_file(target_dir / f"{args.city}_{args.run}_full.csv")
        loaded_targets[key] = (target_pos, target_diag)

    trigger_rows: list[dict[str, Any]] = []
    chosen_by_row: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(internal_rows):
        selected = _base_label(row)
        tow = round(float(row["tow"]), 1)
        current_pos = _xyz_from_row(row, "pf_before_emit")
        truth = _truth_xyz(row)
        current_error = _float(row, "emit_to_ref_m")
        current_pass = math.isfinite(current_error) and current_error <= float(args.pass_m)
        for rule in rules:
            if selected != str(rule["source_label"]):
                continue
            target_pos, target_diag = loaded_targets[str(rule["target_profile_label"])]
            pos = target_pos.get(tow)
            if pos is None:
                continue
            rule_args = SimpleNamespace(
                ratio_min=args.ratio_min,
                residual_rms_max=args.residual_rms_max,
                status5_residual_rms_max=args.status5_residual_rms_max,
                offset_min_m=rule["offset_min_m"],
                offset_max_m=rule["offset_max_m"],
                family_span_min_m=rule["family_span_min_m"],
                agreement_1m_max=rule["agreement_1m_max"],
                target_rms_max=rule["target_rms_max"],
            )
            passed, metrics = _target_passes_rule(
                row=row,
                current_pos=current_pos,
                target_pos=np.asarray(pos, dtype=np.float64),
                target_diag=target_diag.get(tow),
                args=rule_args,
            )
            if not passed:
                continue

            target_error = (
                float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - truth))
                if np.all(np.isfinite(truth))
                else float("nan")
            )
            target_pass = math.isfinite(target_error) and target_error <= float(args.pass_m)
            delta_m = 0.0
            if target_pass and not current_pass:
                delta_m = weights[row_index]
            elif current_pass and not target_pass:
                delta_m = -weights[row_index]
            p_pass = float(args.p_pass_start) - float(rule["rank"])
            out = {
                "row_index": row_index,
                "epoch": int(float(row["epoch"])),
                "tow": f"{tow:.1f}",
                "weight_m": weights[row_index],
                "source_label": str(rule["source_label"]),
                "target_profile_label": str(rule["target_profile_label"]),
                "target_runtime_label": str(rule["target_runtime_label"]),
                "rule_rank": int(rule["rank"]),
                "p_pass": p_pass,
                "current_error_m": current_error,
                "target_error_m": target_error,
                "current_pass": current_pass,
                "target_pass": target_pass,
                "delta_m": delta_m,
                **metrics,
            }
            trigger_rows.append(out)
            current = chosen_by_row.get(row_index)
            if current is None or p_pass > float(current["p_pass"]):
                chosen_by_row[row_index] = out

    full_total_m = sum(weights)
    base_pass_m = sum(
        weight
        for weight, row in zip(weights, internal_rows)
        if math.isfinite(_float(row, "emit_to_ref_m")) and _float(row, "emit_to_ref_m") <= float(args.pass_m)
    )
    chosen = list(chosen_by_row.values())
    gain_m = sum(float(row["delta_m"]) for row in chosen)
    target_counts = Counter(str(row["target_runtime_label"]) for row in chosen)
    summary_rows = [
        {
            "scope": "phase80_combo_overlay",
            "rules_used": len(rules),
            "base_pass_m": base_pass_m,
            "full_total_m": full_total_m,
            "base_score_pct": 100.0 * base_pass_m / full_total_m if full_total_m > 0.0 else "",
            "triggered_rows": len(trigger_rows),
            "chosen_overrides": len(chosen),
            "good_epochs": sum(1 for row in chosen if bool(row["target_pass"]) and not bool(row["current_pass"])),
            "bad_epochs": sum(1 for row in chosen if bool(row["current_pass"]) and not bool(row["target_pass"])),
            "oracle_eval_gain_m": gain_m,
            "oracle_eval_n2_delta_pp": 100.0 * gain_m / full_total_m if full_total_m > 0.0 else "",
            "oracle_eval_official_delta_pp": 100.0 * gain_m / full_total_m / 6.0 if full_total_m > 0.0 else "",
            "chosen_runtime_labels": ",".join(f"{k}:{v}" for k, v in target_counts.most_common()),
        },
    ]
    return trigger_rows, summary_rows


def _write_overlay_predictions(
    *,
    base_predictions: Path,
    out_csv: Path,
    trigger_rows: list[dict[str, Any]],
    run_id: str,
) -> tuple[int, int]:
    updates: dict[tuple[float, str], float] = {}
    for row in trigger_rows:
        key = (round(float(row["tow"]), 1), str(row["target_runtime_label"]))
        updates[key] = max(float(row["p_pass"]), updates.get(key, float("-inf")))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    updated_rows = 0
    added_rows = 0
    seen: set[tuple[float, str]] = set()
    with base_predictions.open(newline="", encoding="utf-8") as src, out_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        if not {"run_id", "tow", "label", "p_pass"}.issubset(fieldnames):
            raise SystemExit(f"unexpected prediction columns: {fieldnames}")
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            try:
                row_run_id = str(row["run_id"])
                tow = round(float(row["tow"]), 1)
                label = str(row["label"])
            except (KeyError, ValueError):
                writer.writerow(row)
                continue
            key = (tow, label)
            if row_run_id == run_id and key in updates:
                row["p_pass"] = f"{updates[key]:.6f}"
                seen.add(key)
                updated_rows += 1
            writer.writerow(row)

        for tow, label in sorted(set(updates) - seen):
            row = {field: "" for field in fieldnames}
            row["run_id"] = run_id
            row["tow"] = f"{tow:.1f}"
            row["label"] = label
            if "is_pass_50cm" in row:
                row["is_pass_50cm"] = "0"
            if "path_weight" in row:
                row["path_weight"] = "0.0"
            row["p_pass"] = f"{updates[(tow, label)]:.6f}"
            writer.writerow(row)
            added_rows += 1
    return updated_rows, added_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-sweep-csv",
        type=Path,
        default=Path("experiments/results/phase78_nr2_label_pair_detector_pair_sweep.csv"),
    )
    parser.add_argument(
        "--internal-epochs-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase70_osmroad_overlay_nagoya_run2_full_internal_epochs.csv"),
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path("/tmp/selector_ranker_predictions_phase79_phase71_osmroad_overlay_v5.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/tmp/selector_ranker_predictions_phase80_nr2_top3_label_pair_overlay.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("experiments/results/phase80_nr2_top3_label_pair_overlay"),
    )
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--run-id", default="nagoya_run2")
    parser.add_argument("--max-rules", type=int, default=3)
    parser.add_argument("--p-pass-start", type=float, default=999.0)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    args = parser.parse_args(argv)

    rules = _read_rules(args.pair_sweep_csv, int(args.max_rules))
    trigger_rows, summary_rows = _build_trigger_rows(args, rules)
    updated_rows, added_rows = _write_overlay_predictions(
        base_predictions=args.base_predictions,
        out_csv=args.out_csv,
        trigger_rows=trigger_rows,
        run_id=str(args.run_id),
    )

    for row in summary_rows:
        row["prediction_csv"] = str(args.out_csv)
        row["prediction_rows_updated"] = updated_rows
        row["prediction_rows_added"] = added_rows
    _write_csv(Path(f"{args.out_prefix}_rules.csv"), rules)
    _write_csv(Path(f"{args.out_prefix}_summary.csv"), summary_rows)
    _write_csv(Path(f"{args.out_prefix}_trigger_epochs.csv"), trigger_rows)
    print(f"saved: {args.out_csv}")
    print(
        "rules_used="
        f"{len(rules)} triggered_rows={len(trigger_rows)} "
        f"updated_rows={updated_rows} added_rows={added_rows}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
