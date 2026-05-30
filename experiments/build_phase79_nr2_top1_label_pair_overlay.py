#!/usr/bin/env python3
"""Build Phase79 top1 label-pair ranker overlay predictions."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from audit_nr2_profile_span_oracle import _distance_weights, _float, _truth_xyz, _write_csv  # noqa: E402
from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _diag_float,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
)


def _base_label(row: dict[str, str]) -> str:
    label = row.get("rtkdiag_selected_base_label", "")
    if label:
        return label
    return row.get("rtkdiag_selected_label", "").removesuffix("+rnk")


def _xyz_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [
            _float(row, f"{prefix}_x"),
            _float(row, f"{prefix}_y"),
            _float(row, f"{prefix}_z"),
        ],
        dtype=np.float64,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _target_passes_rule(
    *,
    row: dict[str, str],
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    target_diag: dict[str, str] | None,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float]]:
    if not _rtkdiag_candidate_gate(
        target_diag,
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    ):
        return False, {}
    if not np.all(np.isfinite(current_pos)) or not np.all(np.isfinite(target_pos)):
        return False, {}

    offset_m = float(np.linalg.norm(target_pos - current_pos))
    family_span_m = _float(row, "rtkdiag_candidate_family_span_m")
    agreement_1m = _float(row, "rtkdiag_candidate_agreement_count_1m")
    target_rms = _diag_float(target_diag, "final_residual_rms") if target_diag else float("nan")
    target_ratio = _diag_float(target_diag, "final_ratio") if target_diag else float("nan")
    target_status = _diag_float(target_diag, "final_status") if target_diag else float("nan")

    if not math.isfinite(offset_m) or offset_m < float(args.offset_min_m) or offset_m > float(args.offset_max_m):
        return False, {}
    if not math.isfinite(family_span_m) or family_span_m < float(args.family_span_min_m):
        return False, {}
    if math.isfinite(agreement_1m) and agreement_1m > float(args.agreement_1m_max):
        return False, {}
    if math.isfinite(target_rms) and target_rms > float(args.target_rms_max):
        return False, {}
    return True, {
        "offset_m": offset_m,
        "family_span_m": family_span_m,
        "agreement_1m": agreement_1m,
        "target_rms": target_rms,
        "target_ratio": target_ratio,
        "target_status": target_status,
    }


def _build_trigger_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    internal_rows = _read_csv(args.internal_epochs_csv)
    weights = _distance_weights(internal_rows)
    target_pos, _target_status = _load_hybrid_pos_file(args.target_dir / f"{args.city}_{args.run}_full.pos")
    target_diag = _load_rtk_diag_file(args.target_dir / f"{args.city}_{args.run}_full.csv")

    trigger_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(internal_rows):
        selected = _base_label(row)
        if selected != str(args.source_label):
            continue
        tow = round(float(row["tow"]), 1)
        pos = target_pos.get(tow)
        if pos is None:
            continue
        current_pos = _xyz_from_row(row, "pf_before_emit")
        passed, metrics = _target_passes_rule(
            row=row,
            current_pos=current_pos,
            target_pos=np.asarray(pos, dtype=np.float64),
            target_diag=target_diag.get(tow),
            args=args,
        )
        if not passed:
            continue

        truth = _truth_xyz(row)
        current_error = _float(row, "emit_to_ref_m")
        target_error = (
            float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - truth))
            if np.all(np.isfinite(truth))
            else float("nan")
        )
        current_pass = math.isfinite(current_error) and current_error <= float(args.pass_m)
        target_pass = math.isfinite(target_error) and target_error <= float(args.pass_m)
        delta_m = 0.0
        if target_pass and not current_pass:
            delta_m = weights[row_index]
        elif current_pass and not target_pass:
            delta_m = -weights[row_index]
        trigger_rows.append(
            {
                "row_index": row_index,
                "epoch": int(float(row["epoch"])),
                "tow": f"{tow:.1f}",
                "weight_m": weights[row_index],
                "source_label": str(args.source_label),
                "target_label": str(args.target_label),
                "triggered": True,
                "current_error_m": current_error,
                "target_error_m": target_error,
                "current_pass": current_pass,
                "target_pass": target_pass,
                "delta_m": delta_m,
                **metrics,
            },
        )

    total_m = sum(weights)
    base_pass_m = sum(
        weight
        for weight, row in zip(weights, internal_rows)
        if math.isfinite(_float(row, "emit_to_ref_m")) and _float(row, "emit_to_ref_m") <= float(args.pass_m)
    )
    gain_m = sum(float(row["delta_m"]) for row in trigger_rows)
    summary_rows = [
        {
            "scope": "phase79_trigger_overlay",
            "base_pass_m": base_pass_m,
            "full_total_m": total_m,
            "base_score_pct": 100.0 * base_pass_m / total_m if total_m > 0.0 else "",
            "source_label": str(args.source_label),
            "target_label": str(args.target_label),
            "target_dir": str(args.target_dir),
            "offset_min_m": float(args.offset_min_m),
            "offset_max_m": float(args.offset_max_m),
            "family_span_min_m": float(args.family_span_min_m),
            "agreement_1m_max": float(args.agreement_1m_max),
            "target_rms_max": float(args.target_rms_max),
            "triggered_epochs": len(trigger_rows),
            "good_epochs": sum(1 for row in trigger_rows if bool(row["target_pass"]) and not bool(row["current_pass"])),
            "bad_epochs": sum(1 for row in trigger_rows if bool(row["current_pass"]) and not bool(row["target_pass"])),
            "oracle_eval_gain_m": gain_m,
            "oracle_eval_n2_delta_pp": 100.0 * gain_m / total_m if total_m > 0.0 else "",
            "oracle_eval_official_delta_pp": 100.0 * gain_m / total_m / 6.0 if total_m > 0.0 else "",
        },
    ]
    return trigger_rows, summary_rows


def _write_overlay_predictions(
    *,
    base_predictions: Path,
    out_csv: Path,
    trigger_rows: list[dict[str, Any]],
    run_id: str,
    target_label: str,
    source_label: str,
    p_pass: float,
    source_p_pass: float | None,
) -> tuple[int, int, int]:
    trigger_tows = {round(float(row["tow"]), 1) for row in trigger_rows}
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    target_updates = 0
    source_updates = 0
    added_rows = 0
    seen_target_tows: set[float] = set()
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
            if row_run_id == run_id and tow in trigger_tows and label == target_label:
                row["p_pass"] = f"{float(p_pass):.6f}"
                seen_target_tows.add(tow)
                target_updates += 1
            elif (
                source_p_pass is not None
                and row_run_id == run_id
                and tow in trigger_tows
                and label == source_label
            ):
                row["p_pass"] = f"{float(source_p_pass):.6f}"
                source_updates += 1
            writer.writerow(row)

        missing_tows = sorted(trigger_tows - seen_target_tows)
        for tow in missing_tows:
            row = {field: "" for field in fieldnames}
            row["run_id"] = run_id
            row["tow"] = f"{tow:.1f}"
            row["label"] = target_label
            if "is_pass_50cm" in row:
                row["is_pass_50cm"] = "0"
            if "path_weight" in row:
                row["path_weight"] = "0.0"
            row["p_pass"] = f"{float(p_pass):.6f}"
            writer.writerow(row)
            added_rows += 1
    return target_updates, source_updates, added_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path("/tmp/selector_ranker_predictions_phase71_osmroad_overlay_v5.csv"),
    )
    parser.add_argument(
        "--internal-epochs-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase70_osmroad_overlay_nagoya_run2_full_internal_epochs.csv"),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("experiments/results/libgnss_diag_phase19/gici_full_hisnr"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/tmp/selector_ranker_predictions_phase79_nr2_top1_label_pair_overlay.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("experiments/results/phase79_nr2_top1_label_pair_overlay"),
    )
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--run-id", default="nagoya_run2")
    parser.add_argument("--source-label", default="xd_gici_r")
    parser.add_argument("--target-label", default="xd_gici_hs")
    parser.add_argument("--p-pass", type=float, default=999.0)
    parser.add_argument("--source-p-pass", type=float, default=None)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--offset-min-m", type=float, default=0.0)
    parser.add_argument("--offset-max-m", type=float, default=3.832)
    parser.add_argument("--family-span-min-m", type=float, default=0.0)
    parser.add_argument("--agreement-1m-max", type=float, default=99.0)
    parser.add_argument("--target-rms-max", type=float, default=0.1)
    args = parser.parse_args(argv)

    trigger_rows, summary_rows = _build_trigger_rows(args)
    target_updates, source_updates, added_rows = _write_overlay_predictions(
        base_predictions=args.base_predictions,
        out_csv=args.out_csv,
        trigger_rows=trigger_rows,
        run_id=str(args.run_id),
        target_label=str(args.target_label),
        source_label=str(args.source_label),
        p_pass=float(args.p_pass),
        source_p_pass=args.source_p_pass,
    )

    for row in summary_rows:
        row["prediction_csv"] = str(args.out_csv)
        row["target_prediction_updates"] = target_updates
        row["source_prediction_updates"] = source_updates
        row["prediction_rows_added"] = added_rows
    _write_csv(Path(f"{args.out_prefix}_summary.csv"), summary_rows)
    _write_csv(Path(f"{args.out_prefix}_trigger_epochs.csv"), trigger_rows)
    print(f"saved: {args.out_csv}")
    print(
        "triggered_epochs="
        f"{len(trigger_rows)} target_updates={target_updates} "
        f"source_updates={source_updates} added_rows={added_rows}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
