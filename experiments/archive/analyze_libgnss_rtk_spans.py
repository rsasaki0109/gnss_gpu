#!/usr/bin/env python3
"""Summarize libgnss++ RTK diagnostics over PPC failure spans.

This is a bridge between PPC-level failure classification and libgnss++ RTK
internals.  It joins:

* PPC internal epoch logs with reference ECEF coordinates.
* libgnss++ ``.pos`` output.
* libgnss++ diagnostics CSV.
* optional DD residual CSV from ``gnss_solve --dd-residuals-csv``.

The output highlights whether a span is mainly FLOAT/AR, residual/outlier, or
candidate-selection trouble, and lists the worst DD satellite pairs in that
span.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _tow_key(value: object) -> float:
    return round(_float(value), 1)


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _median(values: list[float]) -> float:
    vals = _finite(values)
    return float(median(vals)) if vals else float("nan")


def _p95(values: list[float]) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    idx = int(math.ceil(0.95 * len(vals))) - 1
    return vals[max(0, min(idx, len(vals) - 1))]


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((aa - bb) ** 2 for aa, bb in zip(a, b)))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _counter_text(counter: Counter[object], limit: int = 8) -> str:
    return ";".join(f"{key}:{count}" for key, count in counter.most_common(limit))


def _load_reference_epochs(path: Path) -> dict[float, tuple[float, float, float]]:
    refs: dict[float, tuple[float, float, float]] = {}
    for row in _read_csv(path):
        ref = (_float(row.get("ref_x")), _float(row.get("ref_y")), _float(row.get("ref_z")))
        if all(math.isfinite(value) for value in ref):
            refs[_tow_key(row.get("tow"))] = ref
    return refs


def _load_pos(path: Path) -> dict[float, dict[str, object]]:
    rows: dict[float, dict[str, object]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            xyz = (_float(parts[2]), _float(parts[3]), _float(parts[4]))
            if not all(math.isfinite(value) for value in xyz):
                continue
            rows[_tow_key(parts[1])] = {
                "xyz": xyz,
                "status": _int(parts[8]),
                "sats": _int(parts[9]),
                "ratio": _float(parts[11]),
            }
    return rows


def _load_diag(path: Path) -> dict[float, dict[str, str]]:
    return {_tow_key(row.get("tow")): row for row in _read_csv(path)}


def _load_debug(path: Path | None) -> dict[float, dict[str, str]]:
    if path is None:
        return {}
    return {_tow_key(row.get("tow")): row for row in _read_csv(path)}


def _load_dd(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    records: list[dict[str, object]] = []
    for row in _read_csv(path):
        residual = _float(row.get("residual_m"))
        abs_residual = _float(row.get("abs_residual_m"))
        if not math.isfinite(residual) or not math.isfinite(abs_residual):
            continue
        records.append(
            {
                "tow": _tow_key(row.get("tow")),
                "kind": row.get("kind", ""),
                "frequency_index": _int(row.get("frequency_index")),
                "reference_satellite": row.get("reference_satellite", ""),
                "satellite": row.get("satellite", ""),
                "residual_m": residual,
                "abs_residual_m": abs_residual,
                "suppressed": _int(row.get("suppressed_by_outlier_threshold")),
                "status": _int(row.get("status")),
            }
        )
    return records


def _span_tows(span: dict[str, str], refs: dict[float, tuple[float, float, float]]) -> list[float]:
    start = _tow_key(span.get("start_tow"))
    end = _tow_key(span.get("end_tow"))
    return sorted(tow for tow in refs if start <= tow <= end)


def _summarize_span(
    span: dict[str, str],
    refs: dict[float, tuple[float, float, float]],
    pos: dict[float, dict[str, object]],
    diag: dict[float, dict[str, str]],
    debug: dict[float, dict[str, str]],
    dd_records: list[dict[str, object]],
    *,
    pair_abs_min_m: float,
    top_pairs: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    tows = _span_tows(span, refs)
    pos_errors: list[float] = []
    pos_status = Counter()
    pos_ratios: list[float] = []
    diag_status = Counter()
    diag_reasons = Counter()
    diag_sats: list[float] = []
    diag_ratios: list[float] = []
    diag_rms: list[float] = []
    diag_absmax: list[float] = []
    output_added = 0
    ar_attempted = 0
    full_lambda_solved = 0
    full_ratios: list[float] = []
    pair_counts: list[float] = []
    max_ambiguity_variances: list[float] = []
    selected_fixed = 0
    final_fixed_applied = 0
    reject_reasons = Counter()
    ar_skip_reasons = Counter()

    for tow in tows:
        pos_row = pos.get(tow)
        if pos_row is not None:
            pos_errors.append(_dist(pos_row["xyz"], refs[tow]))  # type: ignore[arg-type]
            pos_status[pos_row["status"]] += 1
            pos_ratios.append(float(pos_row["ratio"]))
        diag_row = diag.get(tow)
        if diag_row is not None:
            diag_status[_int(diag_row.get("final_status"))] += 1
            diag_reasons[str(diag_row.get("rejection_reason") or "none")] += 1
            diag_sats.append(_float(diag_row.get("final_sats")))
            diag_ratios.append(_float(diag_row.get("final_ratio")))
            diag_rms.append(_float(diag_row.get("final_residual_rms")))
            diag_absmax.append(_float(diag_row.get("final_residual_abs_max")))
            output_added += 1 if _int(diag_row.get("output_added")) else 0
        debug_row = debug.get(tow)
        if debug_row is not None:
            ar_attempted += 1 if _int(debug_row.get("ar_attempted")) else 0
            full_lambda_solved += 1 if _int(debug_row.get("full_lambda_solved")) else 0
            full_ratios.append(_float(debug_row.get("full_ratio")))
            pair_counts.append(_float(debug_row.get("pair_count")))
            max_ambiguity_variances.append(_float(debug_row.get("max_ambiguity_variance")))
            is_selected_fixed = _int(debug_row.get("selected_fixed")) != 0
            is_final_fixed = _int(debug_row.get("final_fixed_applied")) != 0
            selected_fixed += 1 if is_selected_fixed else 0
            final_fixed_applied += 1 if is_final_fixed else 0
            reason = str(debug_row.get("reject_reason") or "none")
            if is_selected_fixed or is_final_fixed:
                reason = "none"
            reject_reasons[reason] += 1
            ar_skip_reasons[str(debug_row.get("ar_skip_reason") or "none")] += 1

    start = _tow_key(span.get("start_tow"))
    end = _tow_key(span.get("end_tow"))
    span_dd = [record for record in dd_records if start <= float(record["tow"]) <= end]
    dd_by_kind: dict[str, list[float]] = defaultdict(list)
    for record in span_dd:
        dd_by_kind[str(record["kind"])].append(float(record["abs_residual_m"]))
    suppressed = sum(int(record["suppressed"]) for record in span_dd)
    dd_status = Counter(record["status"] for record in span_dd)

    grouped: dict[tuple[object, object, object, object], list[dict[str, object]]] = defaultdict(list)
    for record in span_dd:
        if float(record["abs_residual_m"]) >= pair_abs_min_m or int(record["suppressed"]):
            key = (
                record["kind"],
                record["frequency_index"],
                record["reference_satellite"],
                record["satellite"],
            )
            grouped[key].append(record)

    pair_rows: list[dict[str, object]] = []
    for (kind, freq, ref_sat, sat), rows in grouped.items():
        abs_values = [float(row["abs_residual_m"]) for row in rows]
        pair_rows.append(
            {
                "span_id": span.get("span_id"),
                "root_cause": span.get("root_cause"),
                "kind": kind,
                "frequency_index": freq,
                "reference_satellite": ref_sat,
                "satellite": sat,
                "pair": f"{ref_sat}-{sat}",
                "rows": len(rows),
                "suppressed_rows": sum(int(row["suppressed"]) for row in rows),
                "median_abs_residual_m": _median(abs_values),
                "p95_abs_residual_m": _p95(abs_values),
                "max_abs_residual_m": max(abs_values) if abs_values else float("nan"),
            }
        )
    pair_rows.sort(
        key=lambda row: (
            -int(row["suppressed_rows"]),
            -float(row["max_abs_residual_m"]),
            -int(row["rows"]),
        )
    )
    pair_rows = pair_rows[:top_pairs]

    summary = {
        "span_id": span.get("span_id"),
        "root_cause": span.get("root_cause"),
        "start_epoch": span.get("start_epoch"),
        "end_epoch": span.get("end_epoch"),
        "start_tow": span.get("start_tow"),
        "end_tow": span.get("end_tow"),
        "n_ref_epochs": len(tows),
        "pos_epochs": len(pos_errors),
        "pos_error_median_m": _median(pos_errors),
        "pos_error_p95_m": _p95(pos_errors),
        "pos_status_counts": _counter_text(pos_status),
        "pos_ratio_median": _median(pos_ratios),
        "diag_epochs": sum(diag_status.values()),
        "diag_output_added_epochs": output_added,
        "diag_final_status_counts": _counter_text(diag_status),
        "diag_rejection_reason_counts": _counter_text(diag_reasons),
        "diag_final_sats_median": _median(diag_sats),
        "diag_final_ratio_median": _median(diag_ratios),
        "diag_final_residual_rms_median_m": _median(diag_rms),
        "diag_final_residual_absmax_p95_m": _p95(diag_absmax),
        "debug_epochs": sum(reject_reasons.values()),
        "ar_attempted_epochs": ar_attempted,
        "full_lambda_solved_epochs": full_lambda_solved,
        "full_ratio_median": _median(full_ratios),
        "pair_count_median": _median(pair_counts),
        "max_ambiguity_variance_median": _median(max_ambiguity_variances),
        "selected_fixed_epochs": selected_fixed,
        "final_fixed_applied_epochs": final_fixed_applied,
        "reject_reason_counts": _counter_text(reject_reasons),
        "ar_skip_reason_counts": _counter_text(ar_skip_reasons),
        "dd_rows": len(span_dd),
        "dd_epochs": len({record["tow"] for record in span_dd}),
        "dd_status_counts": _counter_text(dd_status),
        "dd_suppressed_rows": suppressed,
        "dd_phase_median_abs_m": _median(dd_by_kind.get("phase", [])),
        "dd_phase_p95_abs_m": _p95(dd_by_kind.get("phase", [])),
        "dd_code_median_abs_m": _median(dd_by_kind.get("code", [])),
        "dd_code_p95_abs_m": _p95(dd_by_kind.get("code", [])),
        "top_dd_pairs": ";".join(
            f"{row['pair']}/{row['kind']}f{row['frequency_index']}:"
            f"n={row['rows']},sup={row['suppressed_rows']},max={float(row['max_abs_residual_m']):.2f}"
            for row in pair_rows[:5]
        ),
    }
    return summary, pair_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("spans_csv", type=Path)
    parser.add_argument("--internal-epochs", type=Path, required=True)
    parser.add_argument("--pos", type=Path, required=True)
    parser.add_argument("--diag", type=Path, required=True)
    parser.add_argument("--debug-epoch-log", type=Path, default=None)
    parser.add_argument("--dd-residuals", type=Path, default=None)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--top-n-spans", type=int, default=0)
    parser.add_argument("--span-id", action="append", default=[])
    parser.add_argument("--pair-abs-min-m", type=float, default=3.0)
    parser.add_argument("--top-pairs-per-span", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spans = _read_csv(args.spans_csv)
    if args.span_id:
        allowed = set(args.span_id)
        spans = [span for span in spans if str(span.get("span_id")) in allowed]
    elif args.top_n_spans > 0:
        spans = spans[: args.top_n_spans]

    refs = _load_reference_epochs(args.internal_epochs)
    pos = _load_pos(args.pos)
    diag = _load_diag(args.diag)
    debug = _load_debug(args.debug_epoch_log)
    dd_records = _load_dd(args.dd_residuals)

    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for span in spans:
        summary, pairs = _summarize_span(
            span,
            refs,
            pos,
            diag,
            debug,
            dd_records,
            pair_abs_min_m=args.pair_abs_min_m,
            top_pairs=args.top_pairs_per_span,
        )
        summary_rows.append(summary)
        pair_rows.extend(pairs)

    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    pairs_path = args.out_prefix.with_name(args.out_prefix.name + "_dd_pairs.csv")
    _write_csv(summary_path, summary_rows)
    _write_csv(pairs_path, pair_rows)

    print(f"wrote {summary_path}")
    print(f"wrote {pairs_path}")
    for row in summary_rows[:10]:
        print(
            f"span {row['span_id']} {row['root_cause']}: "
            f"pos_med={float(row['pos_error_median_m']):.3f}m "
            f"status={row['diag_final_status_counts']} "
            f"ratio_med={float(row['diag_final_ratio_median']):.3f} "
            f"dd_phase_p95={float(row['dd_phase_p95_abs_m']):.3f}m "
            f"supp={row['dd_suppressed_rows']}"
        )


if __name__ == "__main__":
    main()
