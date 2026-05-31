#!/usr/bin/env python3
"""Focused low-satellite bridge audit for the n/r2 residual span #2."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_INTERNAL = Path(
    "experiments/results/ppc_phase70_osmroad_overlay_nagoya_run2_full_internal_epochs.csv",
)
DEFAULT_SUMMARY = Path("experiments/results/phase75_nr2_low_sat_bridge_span2_summary.csv")
DEFAULT_EPOCHS = Path("experiments/results/phase75_nr2_low_sat_bridge_span2_epochs.csv")

SOURCE_PREFIXES = (
    "pf_after_rtkdiag",
    "pf_epoch_end",
    "pf_after_hybrid",
    "pf_before_emit",
)


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


def _truth_xyz(row: dict[str, str]) -> np.ndarray:
    return np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)


def _source_xyz(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [
            _float(row, f"{prefix}_x"),
            _float(row, f"{prefix}_y"),
            _float(row, f"{prefix}_z"),
        ],
        dtype=np.float64,
    )


def _distance_weights(rows: list[dict[str, str]]) -> list[float]:
    xyz = [_truth_xyz(row) for row in rows]
    out: list[float] = []
    for idx, pos in enumerate(xyz):
        if idx == 0 or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(xyz[idx - 1])):
            out.append(0.0)
        else:
            out.append(float(np.linalg.norm(pos - xyz[idx - 1])))
    return out


def _mean(values: list[float]) -> float | str:
    vals = [value for value in values if math.isfinite(value)]
    return float(sum(vals) / len(vals)) if vals else ""


def _max(values: list[float]) -> float | str:
    vals = [value for value in values if math.isfinite(value)]
    return float(max(vals)) if vals else ""


def _score(
    *,
    mode: str,
    source: str,
    errors: list[float],
    weights: list[float],
    pass_m: float,
    note: str = "",
    lo_row: dict[str, str] | None = None,
    hi_row: dict[str, str] | None = None,
    lo_source: str = "",
    hi_source: str = "",
) -> dict[str, Any]:
    passed = sum(weight for error, weight in zip(errors, weights) if math.isfinite(error) and error <= pass_m)
    total = sum(weights)
    row: dict[str, Any] = {
        "mode": mode,
        "source": source,
        "lo_epoch": _int(lo_row, "epoch") if lo_row else "",
        "hi_epoch": _int(hi_row, "epoch") if hi_row else "",
        "lo_source": lo_source,
        "hi_source": hi_source,
        "span_total_m": total,
        "pass_m": passed,
        "fail_m": total - passed,
        "score_pct": 100.0 * passed / total if total > 0.0 else "",
        "pass_epochs": sum(1 for error in errors if math.isfinite(error) and error <= pass_m),
        "available_epochs": sum(1 for error in errors if math.isfinite(error)),
        "mean_error_m": _mean(errors),
        "max_error_m": _max(errors),
        "errors_m": ",".join("" if not math.isfinite(error) else f"{error:.3f}" for error in errors),
        "note": note,
    }
    if lo_row is not None:
        row.update({f"lo_{key}": value for key, value in _anchor_fields(lo_row).items()})
    if hi_row is not None:
        row.update({f"hi_{key}": value for key, value in _anchor_fields(hi_row).items()})
    return row


def _anchor_fields(row: dict[str, str]) -> dict[str, Any]:
    return {
        "tow": _float(row, "tow"),
        "emitted_source": row.get("emitted_source", ""),
        "emit_to_ref_m": _float(row, "emit_to_ref_m"),
        "n_sat_used_pr": _int(row, "n_sat_used_pr"),
        "rtkdiag_label": row.get("rtkdiag_selected_base_label") or row.get("rtkdiag_selected_label", ""),
        "rtkdiag_status": _int(row, "rtkdiag_selected_diag_status"),
        "rtkdiag_sats": _int(row, "rtkdiag_selected_diag_sats"),
        "rtkdiag_ratio": _float(row, "rtkdiag_selected_diag_ratio"),
        "rtkdiag_rms": _float(row, "rtkdiag_selected_diag_rms"),
        "agreement_1m": _int(row, "rtkdiag_candidate_agreement_count_1m"),
        "family_span_m": _float(row, "rtkdiag_candidate_family_span_m"),
        "rtkdiag_velocity_mps": _float(row, "rtkdiag_selected_velocity_mps"),
        "tdcp_velocity_delta_mps": _float(row, "rtkdiag_selected_to_tdcp_velocity_mps"),
    }


def _span_rows(
    rows_by_epoch: dict[int, dict[str, str]],
    start_epoch: int,
    end_epoch: int,
) -> list[dict[str, str]]:
    return [rows_by_epoch[epoch] for epoch in range(start_epoch, end_epoch + 1)]


def _interp_errors(
    *,
    span: list[dict[str, str]],
    lo_row: dict[str, str],
    hi_row: dict[str, str],
    lo_pos: np.ndarray,
    hi_pos: np.ndarray,
) -> list[float]:
    lo_tow = _float(lo_row, "tow")
    hi_tow = _float(hi_row, "tow")
    if not np.all(np.isfinite(lo_pos)) or not np.all(np.isfinite(hi_pos)):
        return [float("nan")] * len(span)
    if not math.isfinite(lo_tow) or not math.isfinite(hi_tow) or hi_tow <= lo_tow:
        return [float("nan")] * len(span)

    errors: list[float] = []
    for row in span:
        u = (_float(row, "tow") - lo_tow) / (hi_tow - lo_tow)
        pred = (1.0 - u) * lo_pos + u * hi_pos
        errors.append(float(np.linalg.norm(pred - _truth_xyz(row))))
    return errors


def _current_errors(span: list[dict[str, str]]) -> list[float]:
    return [_float(row, "emit_to_ref_m") for row in span]


def _source_errors(span: list[dict[str, str]], source: str) -> list[float]:
    errors: list[float] = []
    for row in span:
        pos = _source_xyz(row, source)
        if np.all(np.isfinite(pos)):
            errors.append(float(np.linalg.norm(pos - _truth_xyz(row))))
        else:
            errors.append(float("nan"))
    return errors


def _select_nearest(
    rows_by_epoch: dict[int, dict[str, str]],
    *,
    start_epoch: int,
    end_epoch: int,
    before: int,
    after: int,
    predicate: Callable[[dict[str, str]], bool],
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    lo = None
    for epoch in range(start_epoch - 1, max(-1, start_epoch - before - 1), -1):
        row = rows_by_epoch.get(epoch)
        if row is not None and predicate(row):
            lo = row
            break

    hi = None
    for epoch in range(end_epoch + 1, end_epoch + after + 1):
        row = rows_by_epoch.get(epoch)
        if row is not None and predicate(row):
            hi = row
            break
    return lo, hi


def _non_bridge(row: dict[str, str]) -> bool:
    return not row.get("emitted_source", "").startswith("pf_bridge")


def _finite_source(row: dict[str, str], source: str) -> bool:
    return bool(np.all(np.isfinite(_source_xyz(row, source))))


def _simple_sats_rms(row: dict[str, str]) -> bool:
    return (
        _non_bridge(row)
        and _finite_source(row, "pf_after_rtkdiag")
        and _float(row, "n_sat_used_pr") >= 15.0
        and _float(row, "rtkdiag_selected_diag_rms") <= 1.0
    )


def _agreement_status4(row: dict[str, str]) -> bool:
    return (
        _non_bridge(row)
        and _finite_source(row, "pf_after_rtkdiag")
        and _float(row, "rtkdiag_selected_diag_status") == 4.0
        and _float(row, "rtkdiag_selected_diag_rms") <= 0.6
        and _float(row, "rtkdiag_candidate_agreement_count_1m") >= 10.0
    )


def _scan_bridge_pairs(
    *,
    rows_by_epoch: dict[int, dict[str, str]],
    span: list[dict[str, str]],
    weights: list[float],
    start_epoch: int,
    end_epoch: int,
    before: int,
    after: int,
    pass_m: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    lo_min = max(0, start_epoch - before)
    hi_max = end_epoch + after
    for source in SOURCE_PREFIXES:
        for lo_epoch in range(lo_min, start_epoch):
            lo_row = rows_by_epoch.get(lo_epoch)
            if lo_row is None or not _finite_source(lo_row, source):
                continue
            lo_pos = _source_xyz(lo_row, source)
            for hi_epoch in range(end_epoch + 1, hi_max + 1):
                hi_row = rows_by_epoch.get(hi_epoch)
                if hi_row is None or not _finite_source(hi_row, source):
                    continue
                errors = _interp_errors(
                    span=span,
                    lo_row=lo_row,
                    hi_row=hi_row,
                    lo_pos=lo_pos,
                    hi_pos=_source_xyz(hi_row, source),
                )
                rec = _score(
                    mode="actual_window_scan",
                    source=source,
                    errors=errors,
                    weights=weights,
                    pass_m=pass_m,
                    lo_row=lo_row,
                    hi_row=hi_row,
                    lo_source=source,
                    hi_source=source,
                    note="same-source interpolation over nearby internal PF states",
                )
                rec["lo_anchor_error_m"] = float(np.linalg.norm(lo_pos - _truth_xyz(lo_row)))
                rec["hi_anchor_error_m"] = float(np.linalg.norm(_source_xyz(hi_row, source) - _truth_xyz(hi_row)))
                out.append(rec)
    out.sort(
        key=lambda row: (
            -float(row["pass_m"]),
            float(row["max_error_m"]) if row["max_error_m"] != "" else float("inf"),
            abs(int(row["lo_epoch"]) - start_epoch) + abs(int(row["hi_epoch"]) - end_epoch),
        ),
    )
    return out


def _epoch_details(
    *,
    mode: str,
    source: str,
    span: list[dict[str, str]],
    weights: list[float],
    errors: list[float],
    pass_m: float,
    lo_epoch: int | str = "",
    hi_epoch: int | str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, weight, error in zip(span, weights, errors):
        rows.append(
            {
                "mode": mode,
                "source": source,
                "lo_epoch": lo_epoch,
                "hi_epoch": hi_epoch,
                "epoch": _int(row, "epoch"),
                "tow": _float(row, "tow"),
                "weight_m": weight,
                "error_m": error,
                "passed_50cm": bool(math.isfinite(error) and error <= pass_m),
                "current_emitted_source": row.get("emitted_source", ""),
                "current_emit_to_ref_m": _float(row, "emit_to_ref_m"),
                "n_sat_used_pr": _int(row, "n_sat_used_pr"),
                "hybrid_available": row.get("hybrid_available", ""),
                "rtkdiag_candidate_available": row.get("rtkdiag_candidate_available", ""),
                "rtkdiag_selected_base_label": row.get("rtkdiag_selected_base_label", ""),
            },
        )
    return rows


def analyze(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_csv(args.internal_epochs_csv)
    rows_by_epoch = {int(float(row["epoch"])): row for row in rows}
    weights_all = _distance_weights(rows)
    span = _span_rows(rows_by_epoch, args.start_epoch, args.end_epoch)
    weights = [weights_all[int(float(row["epoch"]))] for row in span]
    summary_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []

    current_errors = _current_errors(span)
    current = _score(
        mode="current_emit",
        source="emitted_position",
        errors=current_errors,
        weights=weights,
        pass_m=args.pass_m,
        note="current Phase71-style output for this low-satellite span",
    )
    summary_rows.append(current)
    epoch_rows.extend(
        _epoch_details(
            mode="current_emit",
            source="emitted_position",
            span=span,
            weights=weights,
            errors=current_errors,
            pass_m=args.pass_m,
        ),
    )

    for source in SOURCE_PREFIXES:
        errors = _source_errors(span, source)
        summary_rows.append(
            _score(
                mode="in_span_state",
                source=source,
                errors=errors,
                weights=weights,
                pass_m=args.pass_m,
                note="direct in-span internal state, no bridge",
            ),
        )

    truth_anchor_pairs = [(1249, 1256), (1248, 1258), (1249, 1264)]
    for lo_epoch, hi_epoch in truth_anchor_pairs:
        lo_row = rows_by_epoch[lo_epoch]
        hi_row = rows_by_epoch[hi_epoch]
        errors = _interp_errors(
            span=span,
            lo_row=lo_row,
            hi_row=hi_row,
            lo_pos=_truth_xyz(lo_row),
            hi_pos=_truth_xyz(hi_row),
        )
        mode = "truth_bridge"
        source = "ref_truth"
        rec = _score(
            mode=mode,
            source=source,
            errors=errors,
            weights=weights,
            pass_m=args.pass_m,
            lo_row=lo_row,
            hi_row=hi_row,
            lo_source=source,
            hi_source=source,
            note="upper bound only; uses reference trajectory anchors",
        )
        summary_rows.append(rec)
        epoch_rows.extend(
            _epoch_details(
                mode=mode,
                source=source,
                span=span,
                weights=weights,
                errors=errors,
                pass_m=args.pass_m,
                lo_epoch=lo_epoch,
                hi_epoch=hi_epoch,
            ),
        )

    fixed_actual = [
        ("pf_after_rtkdiag", 1215, 1278, "best known actual internal-state bridge; recovers only the long-weight last epoch"),
        ("pf_epoch_end", 1215, 1278, "same anchors after epoch-end state"),
        ("pf_after_rtkdiag", 1228, 1298, "alternate actual bridge with weaker anchors"),
        ("pf_after_hybrid", 1245, 1277, "hybrid-state bridge; last epoch is near the 50cm boundary"),
    ]
    for source, lo_epoch, hi_epoch, note in fixed_actual:
        lo_row = rows_by_epoch[lo_epoch]
        hi_row = rows_by_epoch[hi_epoch]
        errors = _interp_errors(
            span=span,
            lo_row=lo_row,
            hi_row=hi_row,
            lo_pos=_source_xyz(lo_row, source),
            hi_pos=_source_xyz(hi_row, source),
        )
        mode = "actual_bridge_fixed"
        rec = _score(
            mode=mode,
            source=source,
            errors=errors,
            weights=weights,
            pass_m=args.pass_m,
            lo_row=lo_row,
            hi_row=hi_row,
            lo_source=source,
            hi_source=source,
            note=note,
        )
        summary_rows.append(rec)
        epoch_rows.extend(
            _epoch_details(
                mode=mode,
                source=source,
                span=span,
                weights=weights,
                errors=errors,
                pass_m=args.pass_m,
                lo_epoch=lo_epoch,
                hi_epoch=hi_epoch,
            ),
        )

    for rule_name, predicate in [
        ("nearest_sats15_rms1_non_bridge", _simple_sats_rms),
        ("nearest_status4_rms06_agree10", _agreement_status4),
    ]:
        lo_row, hi_row = _select_nearest(
            rows_by_epoch,
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
            before=args.search_before,
            after=args.search_after,
            predicate=predicate,
        )
        if lo_row is None or hi_row is None:
            summary_rows.append(
                {
                    "mode": "observable_rule_bridge",
                    "source": "pf_after_rtkdiag",
                    "lo_epoch": _int(lo_row, "epoch") if lo_row else "",
                    "hi_epoch": _int(hi_row, "epoch") if hi_row else "",
                    "span_total_m": sum(weights),
                    "pass_m": 0.0,
                    "fail_m": sum(weights),
                    "score_pct": 0.0,
                    "note": f"{rule_name}: missing anchor",
                },
            )
            continue
        errors = _interp_errors(
            span=span,
            lo_row=lo_row,
            hi_row=hi_row,
            lo_pos=_source_xyz(lo_row, "pf_after_rtkdiag"),
            hi_pos=_source_xyz(hi_row, "pf_after_rtkdiag"),
        )
        mode = "observable_rule_bridge"
        rec = _score(
            mode=mode,
            source="pf_after_rtkdiag",
            errors=errors,
            weights=weights,
            pass_m=args.pass_m,
            lo_row=lo_row,
            hi_row=hi_row,
            lo_source="pf_after_rtkdiag",
            hi_source="pf_after_rtkdiag",
            note=f"{rule_name}: nearest anchors by observable diagnostics",
        )
        summary_rows.append(rec)
        epoch_rows.extend(
            _epoch_details(
                mode=mode,
                source="pf_after_rtkdiag",
                span=span,
                weights=weights,
                errors=errors,
                pass_m=args.pass_m,
                lo_epoch=_int(lo_row, "epoch"),
                hi_epoch=_int(hi_row, "epoch"),
            ),
        )

    scanned = _scan_bridge_pairs(
        rows_by_epoch=rows_by_epoch,
        span=span,
        weights=weights,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        before=args.search_before,
        after=args.search_after,
        pass_m=args.pass_m,
    )
    summary_rows.extend(scanned[: args.top_scan])
    return summary_rows, epoch_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-epochs-csv", type=Path, default=DEFAULT_INTERNAL)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--epochs-out", type=Path, default=DEFAULT_EPOCHS)
    parser.add_argument("--start-epoch", type=int, default=1250)
    parser.add_argument("--end-epoch", type=int, default=1255)
    parser.add_argument("--search-before", type=int, default=80)
    parser.add_argument("--search-after", type=int, default=80)
    parser.add_argument("--top-scan", type=int, default=24)
    parser.add_argument("--pass-m", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows, epoch_rows = analyze(args)
    _write_csv(args.summary_out, summary_rows)
    _write_csv(args.epochs_out, epoch_rows)


if __name__ == "__main__":
    main()
