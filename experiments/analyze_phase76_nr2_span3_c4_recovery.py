#!/usr/bin/env python3
"""Focused recovery audit for the n/r2 residual span #3 (`xd_gici_c4`)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_phase61_no_oracle_span_bias import _ecef_delta_to_enu  # noqa: E402
from exp_ppc_ctrbpf_fgo import _load_hybrid_pos_file, _load_rtk_diag_file, _rtkdiag_candidate_gate  # noqa: E402


DEFAULT_INTERNAL = Path(
    "experiments/results/ppc_phase70_osmroad_overlay_nagoya_run2_full_internal_epochs.csv",
)
DEFAULT_SUMMARY = Path("experiments/results/phase76_nr2_span3_c4_recovery_summary.csv")
DEFAULT_EPOCHS = Path("experiments/results/phase76_nr2_span3_c4_recovery_epochs.csv")
DEFAULT_ROAD_SUMMARY = Path("experiments/results/phase76_nr2_span3_road_variant_sweep_summary.csv")

DEFAULT_SOURCES = {
    "xd_gici_c4": "experiments/results/libgnss_diag_phase19/gici_full_combo4",
    "xd_gici_z": "experiments/results/libgnss_diag_phase19/gici_full_zeroarm",
    "xd_gici_oa": "experiments/results/libgnss_diag_phase19/gici_full_onarm",
    "xd_gici_def": "experiments/results/libgnss_diag_phase19/gici_tc_esdfix",
    "xd_gici_combo": "experiments/results/libgnss_diag_phase19/gici_full_combo",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def _int(row: dict[str, str], key: str) -> int | str:
    value = _float(row, key)
    return int(value) if math.isfinite(value) else ""


def _xyz(row: dict[str, str], prefix: str = "ref") -> np.ndarray:
    return np.array(
        [_float(row, f"{prefix}_x"), _float(row, f"{prefix}_y"), _float(row, f"{prefix}_z")],
        dtype=np.float64,
    )


def _distance_weights(rows: list[dict[str, str]]) -> list[float]:
    xyz = [_xyz(row) for row in rows]
    weights: list[float] = []
    for idx, pos in enumerate(xyz):
        if idx == 0 or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(xyz[idx - 1])):
            weights.append(0.0)
        else:
            weights.append(float(np.linalg.norm(pos - xyz[idx - 1])))
    return weights


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _mean(values: list[float]) -> float | str:
    vals = _finite(values)
    return float(sum(vals) / len(vals)) if vals else ""


def _median(values: list[float]) -> float | str:
    vals = _finite(values)
    return float(median(vals)) if vals else ""


def _p95(values: list[float]) -> float | str:
    vals = sorted(_finite(values))
    if not vals:
        return ""
    idx = min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)
    return float(vals[idx])


def _score(
    *,
    mode: str,
    source_label: str,
    errors: list[float],
    weights: list[float],
    pass_m: float,
    note: str = "",
    correction_enu: np.ndarray | None = None,
    available_epochs: int | None = None,
    gated_epochs: int | None = None,
) -> dict[str, Any]:
    total = sum(weights)
    passed = sum(weight for error, weight in zip(errors, weights) if math.isfinite(error) and error <= pass_m)
    row: dict[str, Any] = {
        "mode": mode,
        "source_label": source_label,
        "span_total_m": total,
        "pass_m": passed,
        "fail_m": total - passed,
        "score_pct": 100.0 * passed / total if total > 0.0 else "",
        "available_epochs": sum(1 for error in errors if math.isfinite(error)) if available_epochs is None else available_epochs,
        "gated_epochs": "" if gated_epochs is None else gated_epochs,
        "pass_epochs": sum(1 for error in errors if math.isfinite(error) and error <= pass_m),
        "mean_error_m": _mean(errors),
        "median_error_m": _median(errors),
        "p95_error_m": _p95(errors),
        "errors_m": ",".join("" if not math.isfinite(error) else f"{error:.3f}" for error in errors),
        "note": note,
    }
    if correction_enu is not None:
        row["correction_e_m"] = float(correction_enu[0])
        row["correction_n_m"] = float(correction_enu[1])
        row["correction_u_m"] = float(correction_enu[2])
    return row


def _split_sources(spec: str) -> dict[str, Path]:
    if not spec:
        return {label: Path(path) for label, path in DEFAULT_SOURCES.items()}
    out: dict[str, Path] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        label, raw_path = item.split("=", 1)
        out[label.strip()] = Path(raw_path.strip())
    return out


def _load_sources(spec: str, *, city: str, run: str) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for label, base in _split_sources(spec).items():
        pos_path = base / f"{city}_{run}_full.pos"
        diag_path = base / f"{city}_{run}_full.csv"
        if not pos_path.is_file():
            print(f"skip missing source: {label} {pos_path}")
            continue
        pos, _status = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path) if diag_path.is_file() else {}
        sources[label] = {
            "pos": pos,
            "diag": diag,
            "pos_path": str(pos_path),
            "diag_path": str(diag_path),
        }
    return sources


def _road_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for row in _read_csv(path)[:limit]:
        out.append(
            {
                "mode": "road_variant",
                "source_label": row.get("source_label", ""),
                "span_total_m": row.get("span_total_m", ""),
                "pass_m": row.get("pass_m", ""),
                "fail_m": row.get("fail_m", ""),
                "score_pct": row.get("score_pct", ""),
                "available_epochs": row.get("available_epochs", ""),
                "gated_epochs": "",
                "pass_epochs": "",
                "mean_error_m": row.get("mean_error_m", ""),
                "median_error_m": row.get("median_error_m", ""),
                "p95_error_m": row.get("p95_error_m", ""),
                "alpha": row.get("alpha", ""),
                "height_offset_m": row.get("height_offset_m", ""),
                "note": "best rows imported from phase76 road variant sweep",
            },
        )
    return out


def analyze(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_csv(args.internal_epochs_csv)
    weights_all = _distance_weights(rows)
    rows_by_epoch = {int(float(row["epoch"])): row for row in rows}
    span = [rows_by_epoch[epoch] for epoch in range(args.start_epoch, args.end_epoch + 1)]
    weights = [weights_all[int(float(row["epoch"]))] for row in span]
    sources = _load_sources(args.sources, city=args.city, run=args.run)

    summary_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []

    current_errors = [_float(row, "emit_to_ref_m") for row in span]
    summary_rows.append(
        _score(
            mode="current_emit",
            source_label="emitted_position",
            errors=current_errors,
            weights=weights,
            pass_m=args.pass_m,
            note="current Phase71-style output; all rows are selected xd_gici_c4",
        ),
    )

    for label, source in sources.items():
        raw_errors: list[float] = []
        gated_mask: list[bool] = []
        positions: list[np.ndarray | None] = []
        truths: list[np.ndarray] = []
        enu_errors: list[np.ndarray] = []
        for row in span:
            tow = round(float(row["tow"]), 1)
            truth = _xyz(row)
            truths.append(truth)
            pos = source["pos"].get(tow)
            if pos is None or not np.all(np.isfinite(pos)):
                raw_errors.append(float("nan"))
                gated_mask.append(False)
                positions.append(None)
                enu_errors.append(np.array([float("nan")] * 3))
                continue
            pos_xyz = np.asarray(pos, dtype=np.float64)
            positions.append(pos_xyz)
            error = float(np.linalg.norm(pos_xyz - truth))
            raw_errors.append(error)
            enu = _ecef_delta_to_enu(pos_xyz - truth, truth)
            enu_errors.append(enu)
            diag_row = source["diag"].get(tow)
            gated_mask.append(
                _rtkdiag_candidate_gate(
                    diag_row,
                    ratio_min=args.ratio_min,
                    residual_rms_max=args.residual_rms_max,
                    status5_residual_rms_max=args.status5_residual_rms_max,
                ),
            )

        summary_rows.append(
            _score(
                mode="raw_candidate",
                source_label=label,
                errors=raw_errors,
                weights=weights,
                pass_m=args.pass_m,
                note="raw candidate position over span",
                available_epochs=sum(1 for error in raw_errors if math.isfinite(error)),
                gated_epochs=sum(1 for value in gated_mask if value),
            ),
        )

        for mode, require_gated in [("bias_oracle_all", False), ("bias_oracle_gated", True)]:
            usable = [
                idx
                for idx, pos in enumerate(positions)
                if pos is not None and (not require_gated or gated_mask[idx])
            ]
            if len(usable) < args.min_bias_epochs:
                continue
            deltas = np.asarray([positions[idx] - truths[idx] for idx in usable], dtype=np.float64)
            correction_ecef = np.median(deltas, axis=0)
            correction_enu = -np.median(np.asarray([enu_errors[idx] for idx in usable], dtype=np.float64), axis=0)
            corrected_errors: list[float] = []
            for pos, truth in zip(positions, truths):
                if pos is None:
                    corrected_errors.append(float("nan"))
                else:
                    corrected_errors.append(float(np.linalg.norm((pos - correction_ecef) - truth)))
            summary_rows.append(
                _score(
                    mode=mode,
                    source_label=label,
                    errors=corrected_errors,
                    weights=weights,
                    pass_m=args.pass_m,
                    note="truth-derived constant-bias upper bound; not deployable",
                    correction_enu=correction_enu,
                    available_epochs=len(usable),
                    gated_epochs=sum(1 for value in gated_mask if value),
                ),
            )

        for row, weight, current_error, raw_error, gated in zip(span, weights, current_errors, raw_errors, gated_mask):
            epoch_rows.append(
                {
                    "source_label": label,
                    "epoch": _int(row, "epoch"),
                    "tow": _float(row, "tow"),
                    "weight_m": weight,
                    "current_error_m": current_error,
                    "raw_candidate_error_m": raw_error,
                    "raw_pass_50cm": bool(math.isfinite(raw_error) and raw_error <= args.pass_m),
                    "candidate_gated": bool(gated),
                    "current_selected_base_label": row.get("rtkdiag_selected_base_label", ""),
                    "current_diag_status": _int(row, "rtkdiag_selected_diag_status"),
                    "current_diag_rms": _float(row, "rtkdiag_selected_diag_rms"),
                    "current_agreement_1m": _int(row, "rtkdiag_candidate_agreement_count_1m"),
                    "current_family_span_m": _float(row, "rtkdiag_candidate_family_span_m"),
                },
            )

    summary_rows.extend(_road_rows(args.road_summary_csv, limit=args.road_top))
    summary_rows.sort(
        key=lambda row: (
            0 if row["mode"] == "current_emit" else 1,
            -float(row["pass_m"]) if str(row.get("pass_m", "")) else 0.0,
            str(row["mode"]),
            str(row["source_label"]),
        ),
    )
    return summary_rows, epoch_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-epochs-csv", type=Path, default=DEFAULT_INTERNAL)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--epochs-out", type=Path, default=DEFAULT_EPOCHS)
    parser.add_argument("--road-summary-csv", type=Path, default=DEFAULT_ROAD_SUMMARY)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--start-epoch", type=int, default=2308)
    parser.add_argument("--end-epoch", type=int, default=2338)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--sources", default="")
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--min-bias-epochs", type=int, default=20)
    parser.add_argument("--road-top", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows, epoch_rows = analyze(args)
    _write_csv(args.summary_out, summary_rows)
    _write_csv(args.epochs_out, epoch_rows)


if __name__ == "__main__":
    main()
