#!/usr/bin/env python3
"""Check whether nearby good candidates can estimate no-oracle span bias."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
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

from analyze_phase43_span_oracle import _load_pool, _split_csv_values  # noqa: E402
from analyze_phase61_no_oracle_span_bias import _ecef_delta_to_enu  # noqa: E402
from exp_ppc_ctrbpf_fgo import _diag_float, _rtkdiag_candidate_gate  # noqa: E402


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
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("nan")


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _median(values: list[float]) -> float | str:
    values = _finite(values)
    return float(median(values)) if values else ""


def _mean(values: list[float]) -> float | str:
    values = _finite(values)
    return float(sum(values) / len(values)) if values else ""


def _norm(vec: tuple[float, float, float] | np.ndarray) -> float:
    vals = [float(v) for v in vec]
    if any(not math.isfinite(v) for v in vals):
        return float("nan")
    return float(math.sqrt(sum(v * v for v in vals)))


def _cosine(a: tuple[float, float, float] | np.ndarray, b: tuple[float, float, float] | np.ndarray) -> float:
    av = [float(v) for v in a]
    bv = [float(v) for v in b]
    if any(not math.isfinite(v) for v in (*av, *bv)):
        return float("nan")
    na = _norm(np.asarray(av))
    nb = _norm(np.asarray(bv))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(sum(x * y for x, y in zip(av, bv)) / (na * nb))


def _xyz(row: dict[str, str], prefix: str) -> np.ndarray | None:
    xyz = np.array([_float(row, f"{prefix}_x"), _float(row, f"{prefix}_y"), _float(row, f"{prefix}_z")], dtype=np.float64)
    return xyz if np.all(np.isfinite(xyz)) else None


def _ref(row: dict[str, str]) -> np.ndarray | None:
    xyz = np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)
    return xyz if np.all(np.isfinite(xyz)) else None


def _span_rows_to_analyze(span_rows: list[dict[str, str]], top: int, include_recoverable: bool) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in span_rows:
        gain = _float(row, "gated_oracle_gain_m")
        if include_recoverable or (math.isfinite(gain) and gain <= 1e-9):
            selected.append(row)
        if len(selected) >= top:
            break
    return selected


def _epoch_region(epoch: int, start: int, end: int, context_epochs: int) -> str | None:
    if start <= epoch <= end:
        return "span"
    if start - context_epochs <= epoch < start:
        return "pre"
    if end < epoch <= end + context_epochs:
        return "post"
    return None


def _label_family(label: str) -> str:
    if label.startswith("xd_gici"):
        return "gici"
    if label.startswith("xd_fgo"):
        return "fgo"
    if label.startswith("xd_"):
        return "other_xd"
    return "legacy"


def _format_counts(counter: Counter[str], n: int = 8) -> str:
    return ",".join(f"{key}:{value}" for key, value in counter.most_common(n))


def _oracle_correction_for_span(rows: list[dict[str, str]]) -> np.ndarray:
    biases: list[np.ndarray] = []
    for row in rows:
        selected = _xyz(row, "pf_epoch_end")
        truth = _ref(row)
        if selected is None or truth is None:
            continue
        biases.append(_ecef_delta_to_enu(selected - truth, truth))
    if not biases:
        return np.array([float("nan")] * 3, dtype=np.float64)
    return -np.median(np.asarray(biases, dtype=np.float64), axis=0)


def analyze(
    *,
    internal_rows: list[dict[str, str]],
    span_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    top: int,
    include_recoverable: bool,
    context_epochs: int,
    pass_m: float,
    near_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_epoch = {_int(row, "epoch"): row for row in internal_rows}
    summary_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for span_index, span in enumerate(_span_rows_to_analyze(span_rows, top, include_recoverable), 1):
        start = _int(span, "start_epoch")
        end = _int(span, "end_epoch")
        target_epochs = range(start - context_epochs, end + context_epochs + 1)
        span_internal = [rows_by_epoch[e] for e in range(start, end + 1) if e in rows_by_epoch]
        oracle_correction = _oracle_correction_for_span(span_internal)
        by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_region_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        epoch_good_any: dict[str, set[int]] = defaultdict(set)
        epoch_near_any: dict[str, set[int]] = defaultdict(set)
        epoch_gated_any: dict[str, set[int]] = defaultdict(set)

        for epoch in target_epochs:
            row = rows_by_epoch.get(epoch)
            if row is None:
                continue
            region = _epoch_region(epoch, start, end, context_epochs)
            if region is None:
                continue
            truth = _ref(row)
            selected = _xyz(row, "pf_epoch_end")
            if truth is None or selected is None:
                continue
            tow = round(float(row["tow"]), 1)
            for cand in candidates:
                label = str(cand["label"])
                pos = cand["pos"].get(tow)
                if pos is None or not np.all(np.isfinite(pos)):
                    continue
                diag = cand["diag"].get(tow)
                gated = _rtkdiag_candidate_gate(
                    diag,
                    ratio_min=ratio_min,
                    residual_rms_max=residual_rms_max,
                    status5_residual_rms_max=status5_residual_rms_max,
                )
                if gated:
                    epoch_gated_any[region].add(epoch)
                pos_xyz = np.asarray(pos, dtype=np.float64)
                err = float(np.linalg.norm(pos_xyz - truth))
                if err <= pass_m:
                    epoch_good_any[region].add(epoch)
                if err <= near_m:
                    epoch_near_any[region].add(epoch)
                if not gated or err > near_m:
                    continue
                offset = _ecef_delta_to_enu(pos_xyz - selected, truth)
                rec = {
                    "label": label,
                    "family": _label_family(label),
                    "epoch": epoch,
                    "error_m": err,
                    "good": err <= pass_m,
                    "near": err <= near_m,
                    "offset_e_m": float(offset[0]),
                    "offset_n_m": float(offset[1]),
                    "offset_u_m": float(offset[2]),
                    "offset_norm_m": _norm(offset),
                    "offset_to_oracle_cos": _cosine(offset, oracle_correction),
                    "offset_to_oracle_error_m": _norm(np.asarray(offset, dtype=np.float64) - oracle_correction),
                    "diag_status": _diag_float(diag, "final_status") if diag else float("nan"),
                    "diag_rms": _diag_float(diag, "final_residual_rms") if diag else float("nan"),
                    "diag_ratio": _diag_float(diag, "ambiguity_ratio") if diag else float("nan"),
                }
                by_region[region].append(rec)
                by_region_label[(region, label)].append(rec)

        for region in ("pre", "span", "post"):
            recs = by_region.get(region, [])
            good_recs = [r for r in recs if bool(r["good"])]
            near_recs = [r for r in recs if bool(r["near"])]
            source = good_recs if good_recs else near_recs
            summary_rows.append(
                {
                    "span_index": span_index,
                    "span_label": span.get("label", ""),
                    "start_epoch": start,
                    "end_epoch": end,
                    "region": region,
                    "context_epochs": context_epochs,
                    "oracle_correction_e_m": float(oracle_correction[0]),
                    "oracle_correction_n_m": float(oracle_correction[1]),
                    "oracle_correction_u_m": float(oracle_correction[2]),
                    "gated_candidate_epochs_any": len(epoch_gated_any.get(region, set())),
                    "good_anchor_epochs_any": len(epoch_good_any.get(region, set())),
                    "near_anchor_epochs_any": len(epoch_near_any.get(region, set())),
                    "near_records": len(near_recs),
                    "good_records": len(good_recs),
                    "source_records_for_offset": len(source),
                    "median_offset_e_m": _median([r["offset_e_m"] for r in source]),
                    "median_offset_n_m": _median([r["offset_n_m"] for r in source]),
                    "median_offset_u_m": _median([r["offset_u_m"] for r in source]),
                    "median_offset_to_oracle_error_m": _median([r["offset_to_oracle_error_m"] for r in source]),
                    "median_offset_to_oracle_cos": _median([r["offset_to_oracle_cos"] for r in source]),
                    "good_labels": _format_counts(Counter(str(r["label"]) for r in good_recs)),
                    "near_labels": _format_counts(Counter(str(r["label"]) for r in near_recs)),
                    "good_families": _format_counts(Counter(str(r["family"]) for r in good_recs)),
                    "near_families": _format_counts(Counter(str(r["family"]) for r in near_recs)),
                },
            )

        for (region, label), recs in by_region_label.items():
            good_recs = [r for r in recs if bool(r["good"])]
            if not recs:
                continue
            label_rows.append(
                {
                    "span_index": span_index,
                    "span_label": span.get("label", ""),
                    "start_epoch": start,
                    "end_epoch": end,
                    "region": region,
                    "label": label,
                    "family": _label_family(label),
                    "near_records": len(recs),
                    "good_records": len(good_recs),
                    "median_error_m": _median([r["error_m"] for r in recs]),
                    "median_offset_e_m": _median([r["offset_e_m"] for r in recs]),
                    "median_offset_n_m": _median([r["offset_n_m"] for r in recs]),
                    "median_offset_u_m": _median([r["offset_u_m"] for r in recs]),
                    "median_offset_to_oracle_error_m": _median([r["offset_to_oracle_error_m"] for r in recs]),
                    "median_offset_to_oracle_cos": _median([r["offset_to_oracle_cos"] for r in recs]),
                    "median_diag_rms": _median([r["diag_rms"] for r in recs]),
                    "median_diag_ratio": _median([r["diag_ratio"] for r in recs]),
                },
            )

    label_rows.sort(
        key=lambda row: (
            int(row["span_index"]),
            str(row["region"]),
            -int(row["good_records"]),
            float(row["median_offset_to_oracle_error_m"]) if row["median_offset_to_oracle_error_m"] != "" else float("inf"),
        ),
    )
    return summary_rows, label_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("span_oracle_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidate-dirs", required=True)
    parser.add_argument("--top", type=int, default=1)
    parser.add_argument("--include-recoverable", action="store_true")
    parser.add_argument("--context-epochs", type=int, default=300)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--near-m", type=float, default=1.0)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/phase64_nearby_anchor_offsets"))
    args = parser.parse_args(argv)

    candidates = _load_pool(_split_csv_values(args.labels), _split_csv_values(args.candidate_dirs), args.city, args.run)
    if not candidates:
        raise SystemExit("no candidates loaded")
    summary_rows, label_rows = analyze(
        internal_rows=_read_csv(args.internal_epochs_csv),
        span_rows=_read_csv(args.span_oracle_csv),
        candidates=candidates,
        top=int(args.top),
        include_recoverable=bool(args.include_recoverable),
        context_epochs=int(args.context_epochs),
        pass_m=float(args.pass_m),
        near_m=float(args.near_m),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary_rows)
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_labels.csv"), label_rows)
    print(f"loaded candidates: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
