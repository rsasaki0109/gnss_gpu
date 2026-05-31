#!/usr/bin/env python3
"""Find RTK improvement points from PPC RTKDiag internal epoch logs.

The script classifies epochs where the current RTKDiag candidate policy falls
back to hybrid/PF output, then looks at the underlying libgnss++ candidate
files to decide whether the problem is candidate generation, diagnostics
gating, hybrid-distance gating, or fallback emission.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import median


DEFAULT_LABELS = (
    "fgo_v14_snr38",
    "full_ratio15_lock3_trustedseed_rtkout3oGem3",
    "dev_demo5_trusted_o3",
    "n2_nobds",
    "fgo_v1",
    "full_ratio15_lock3_trustedseed_rtkout3mlc1",
    "full_ratio15_lock3_trustedseed_rtkout5",
    "libgnss_ext_subset",
)


def _float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", "", "nan"}:
        return False
    return bool(text)


def _tow_key(value: object) -> float:
    return round(_float(value), 1)


def _dist(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float:
    if a is None or b is None:
        return float("nan")
    return math.sqrt(sum((aa - bb) ** 2 for aa, bb in zip(a, b)))


def _percent(n: int, total: int) -> float:
    return 100.0 * float(n) / max(int(total), 1)


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _median(values: list[float]) -> float:
    vals = _finite(values)
    return float(median(vals)) if vals else float("nan")


def _p95(values: list[float]) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    idx = int(math.ceil(0.95 * len(vals))) - 1
    return float(vals[max(0, min(idx, len(vals) - 1))])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
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


def _load_pos(path: Path) -> dict[float, tuple[float, float, float]]:
    positions: dict[float, tuple[float, float, float]] = {}
    if not path.exists():
        return positions
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            status = _int(parts[8])
            if status == 0:
                continue
            xyz = (_float(parts[2]), _float(parts[3]), _float(parts[4]))
            if not all(math.isfinite(v) for v in xyz):
                continue
            positions[_tow_key(parts[1])] = xyz
    return positions


def _load_diag(path: Path) -> dict[float, dict[str, str]]:
    rows: dict[float, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows[_tow_key(row.get("tow"))] = row
    return rows


def _diag_gate(row: dict[str, str] | None, *, ratio_min: float, residual_rms_max: float) -> bool:
    if row is None:
        return False
    return (
        _int(row.get("output_added")) == 1
        and _int(row.get("final_status")) == 4
        and _float(row.get("final_ratio")) >= ratio_min
        and _float(row.get("final_residual_rms")) <= residual_rms_max
    )


def _candidate_sort(row: dict[str, str]) -> tuple[float, float]:
    return (_float(row.get("final_residual_rms")), -_float(row.get("final_ratio")))


def _nearest_position_error(
    positions: dict[float, tuple[float, float, float]],
    tow: float,
    ref: tuple[float, float, float],
    max_dt: float,
) -> tuple[float, float]:
    best_dt = float("nan")
    best_error = float("nan")
    for key, pos in positions.items():
        dt = abs(key - tow)
        if dt <= max_dt and (not math.isfinite(best_dt) or dt < best_dt):
            best_dt = dt
            best_error = _dist(pos, ref)
    return best_dt, best_error


def _load_candidates(
    base_dir: Path,
    labels: tuple[str, ...],
    city: str,
    run: str,
) -> dict[str, dict[str, object]]:
    stem = f"{city}_{run}_full"
    out: dict[str, dict[str, object]] = {}
    for label in labels:
        label_dir = base_dir / label
        out[label] = {
            "pos": _load_pos(label_dir / f"{stem}.pos"),
            "diag": _load_diag(label_dir / f"{stem}.csv"),
        }
    return out


def _classify_row(
    row: dict[str, str],
    candidates: dict[str, dict[str, object]],
    *,
    ratio_min: float,
    residual_rms_max: float,
    hybrid_gate_m: float,
    pass_m: float,
    fail_m: float,
    nearby_dt_s: float,
) -> dict[str, object]:
    tow = _tow_key(row.get("tow"))
    ref = (_float(row.get("ref_x")), _float(row.get("ref_y")), _float(row.get("ref_z")))
    hybrid = None
    if _bool(row.get("hybrid_available")):
        hybrid = (
            _float(row.get("pf_after_hybrid_x")),
            _float(row.get("pf_after_hybrid_y")),
            _float(row.get("pf_after_hybrid_z")),
        )
    if hybrid is None or not all(math.isfinite(v) for v in hybrid):
        hybrid = None

    pos_count = 0
    diag_count = 0
    gate_count = 0
    hybrid_gate_count = 0
    good_pos_count = 0
    good_diag_count = 0
    good_hybrid_rejected_count = 0
    best_label = ""
    best_error = float("nan")
    best_diag_label = ""
    best_diag_error = float("nan")
    best_hybrid_label = ""
    best_hybrid_error = float("nan")
    best_nearby_label = ""
    best_nearby_dt = float("nan")
    best_nearby_error = float("nan")
    selected: list[tuple[tuple[float, float], str, float]] = []

    for label, loaded in candidates.items():
        positions = loaded["pos"]
        diag = loaded["diag"]
        assert isinstance(positions, dict)
        assert isinstance(diag, dict)
        pos = positions.get(tow)
        diag_row = diag.get(tow)
        if diag_row is not None:
            diag_count += 1
        if pos is not None:
            pos_count += 1
            error = _dist(pos, ref)
            if not math.isfinite(best_error) or error < best_error:
                best_label = label
                best_error = error
            if error <= pass_m:
                good_pos_count += 1
        else:
            dt, near_error = _nearest_position_error(positions, tow, ref, nearby_dt_s)
            if math.isfinite(near_error) and (
                not math.isfinite(best_nearby_error) or near_error < best_nearby_error
            ):
                best_nearby_label = label
                best_nearby_dt = dt
                best_nearby_error = near_error

        gate_ok = _diag_gate(diag_row, ratio_min=ratio_min, residual_rms_max=residual_rms_max)
        if not gate_ok:
            continue
        gate_count += 1
        if pos is None:
            continue
        error = _dist(pos, ref)
        if not math.isfinite(best_diag_error) or error < best_diag_error:
            best_diag_label = label
            best_diag_error = error
        if error <= pass_m:
            good_diag_count += 1
        candidate_to_hybrid = _dist(pos, hybrid)
        if math.isfinite(candidate_to_hybrid) and candidate_to_hybrid > hybrid_gate_m and error <= pass_m:
            good_hybrid_rejected_count += 1
        if (
            hybrid_gate_m <= 0.0
            or not math.isfinite(candidate_to_hybrid)
            or candidate_to_hybrid <= hybrid_gate_m
        ):
            hybrid_gate_count += 1
            if not math.isfinite(best_hybrid_error) or error < best_hybrid_error:
                best_hybrid_label = label
                best_hybrid_error = error
            selected.append((_candidate_sort(diag_row), label, error))

    selected.sort(key=lambda item: item[0])
    policy_pick_label = selected[0][1] if selected else ""
    policy_pick_error = selected[0][2] if selected else float("nan")
    emit_error = _float(row.get("emit_to_ref_m"))
    emitted_source = str(row.get("emitted_source", ""))

    if gate_count == 0 and good_pos_count > 0:
        root = "libgnss_diag_gate_blocks_cm_candidate"
    elif gate_count == 0 and math.isfinite(best_error) and best_error <= fail_m:
        root = "libgnss_diag_gate_blocks_meter_candidate"
    elif gate_count == 0 and math.isfinite(best_nearby_error) and best_nearby_error <= pass_m:
        root = "gnss_gpu_candidate_time_grid_gap"
    elif pos_count == 0:
        root = "libgnss_candidate_generation_gap"
    elif hybrid_gate_count == 0 and good_diag_count > 0:
        root = "gnss_gpu_hybrid_gate_blocks_cm_candidate"
    elif hybrid_gate_count == 0 and math.isfinite(best_diag_error) and best_diag_error <= fail_m:
        root = "gnss_gpu_hybrid_gate_blocks_meter_candidate"
    elif hybrid_gate_count == 0:
        root = "libgnss_candidates_not_meter_class"
    elif math.isfinite(policy_pick_error) and policy_pick_error > fail_m:
        root = "libgnss_bad_candidate_survived_gate"
    elif emit_error > fail_m and emitted_source.startswith("pf"):
        root = "gnss_gpu_pf_fallback_bad"
    elif emit_error > fail_m:
        root = "gnss_gpu_hybrid_fallback_bad"
    else:
        root = "fallback_ok_or_low_impact"

    return {
        "epoch": _int(row.get("epoch")),
        "tow": tow,
        "emitted_source": emitted_source,
        "emit_to_ref_m": emit_error,
        "hybrid_to_ref_m": _float(row.get("hybrid_to_ref_m")),
        "pf_after_hybrid_to_ref_m": _float(row.get("pf_after_hybrid_to_ref_m")),
        "pf_after_rtkdiag_to_ref_m": _float(row.get("pf_after_rtkdiag_to_ref_m")),
        "rtkdiag_gated_options_logged": _int(row.get("rtkdiag_gated_options")),
        "pos_candidate_count": pos_count,
        "diag_row_count": diag_count,
        "diag_gate_count": gate_count,
        "hybrid_gate_count": hybrid_gate_count,
        "good_pos_count": good_pos_count,
        "good_diag_count": good_diag_count,
        "good_hybrid_rejected_count": good_hybrid_rejected_count,
        "best_pos_label": best_label,
        "best_pos_to_ref_m": best_error,
        "best_diag_label": best_diag_label,
        "best_diag_to_ref_m": best_diag_error,
        "best_hybrid_label": best_hybrid_label,
        "best_hybrid_to_ref_m": best_hybrid_error,
        "policy_pick_label": policy_pick_label,
        "policy_pick_to_ref_m": policy_pick_error,
        "best_nearby_label": best_nearby_label,
        "best_nearby_dt_s": best_nearby_dt,
        "best_nearby_to_ref_m": best_nearby_error,
        "root_cause": root,
    }


def _span_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: int(r["epoch"]))
    spans: list[dict[str, object]] = []
    start = 0
    span_id = 1
    for idx in range(1, len(ordered) + 1):
        contiguous = idx < len(ordered) and int(ordered[idx]["epoch"]) == int(ordered[idx - 1]["epoch"]) + 1
        same_cause = idx < len(ordered) and ordered[idx]["root_cause"] == ordered[idx - 1]["root_cause"]
        if contiguous and same_cause:
            continue
        chunk = ordered[start:idx]
        causes = Counter(str(r["root_cause"]) for r in chunk)
        sources = Counter(str(r["emitted_source"]) for r in chunk)
        labels = Counter(str(r["policy_pick_label"]) for r in chunk if str(r["policy_pick_label"]))
        spans.append(
            {
                "span_id": span_id,
                "root_cause": causes.most_common(1)[0][0],
                "start_epoch": chunk[0]["epoch"],
                "end_epoch": chunk[-1]["epoch"],
                "start_tow": chunk[0]["tow"],
                "end_tow": chunk[-1]["tow"],
                "n_epochs": len(chunk),
                "emitted_sources": ";".join(f"{k}:{v}" for k, v in sources.most_common()),
                "top_policy_label": labels.most_common(1)[0][0] if labels else "",
                "emit_median_m": _median([_float(r.get("emit_to_ref_m")) for r in chunk]),
                "emit_p95_m": _p95([_float(r.get("emit_to_ref_m")) for r in chunk]),
                "best_pos_median_m": _median([_float(r.get("best_pos_to_ref_m")) for r in chunk]),
                "best_hybrid_median_m": _median([_float(r.get("best_hybrid_to_ref_m")) for r in chunk]),
                "policy_pick_median_m": _median([_float(r.get("policy_pick_to_ref_m")) for r in chunk]),
                "diag_gate_median": _median([_float(r.get("diag_gate_count")) for r in chunk]),
                "hybrid_gate_median": _median([_float(r.get("hybrid_gate_count")) for r in chunk]),
            }
        )
        span_id += 1
        start = idx
    spans.sort(key=lambda row: int(row["n_epochs"]), reverse=True)
    return spans


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = len(rows)
    root_counts = Counter(str(row["root_cause"]) for row in rows)
    source_counts = Counter(str(row["emitted_source"]) for row in rows)
    out: list[dict[str, object]] = [
        {
            "group": "all_fallback",
            "value": "all",
            "n_epochs": total,
            "epoch_pct": 100.0,
            "emit_median_m": _median([_float(r.get("emit_to_ref_m")) for r in rows]),
            "emit_p95_m": _p95([_float(r.get("emit_to_ref_m")) for r in rows]),
            "best_pos_median_m": _median([_float(r.get("best_pos_to_ref_m")) for r in rows]),
            "policy_pick_median_m": _median([_float(r.get("policy_pick_to_ref_m")) for r in rows]),
        }
    ]
    for cause, count in root_counts.most_common():
        chunk = [r for r in rows if r["root_cause"] == cause]
        out.append(
            {
                "group": "root_cause",
                "value": cause,
                "n_epochs": count,
                "epoch_pct": _percent(count, total),
                "emit_median_m": _median([_float(r.get("emit_to_ref_m")) for r in chunk]),
                "emit_p95_m": _p95([_float(r.get("emit_to_ref_m")) for r in chunk]),
                "best_pos_median_m": _median([_float(r.get("best_pos_to_ref_m")) for r in chunk]),
                "policy_pick_median_m": _median([_float(r.get("policy_pick_to_ref_m")) for r in chunk]),
            }
        )
    for source, count in source_counts.most_common():
        chunk = [r for r in rows if r["emitted_source"] == source]
        out.append(
            {
                "group": "emitted_source",
                "value": source,
                "n_epochs": count,
                "epoch_pct": _percent(count, total),
                "emit_median_m": _median([_float(r.get("emit_to_ref_m")) for r in chunk]),
                "emit_p95_m": _p95([_float(r.get("emit_to_ref_m")) for r in chunk]),
                "best_pos_median_m": _median([_float(r.get("best_pos_to_ref_m")) for r in chunk]),
                "policy_pick_median_m": _median([_float(r.get("policy_pick_to_ref_m")) for r in chunk]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--candidate-base-dir", type=Path, default=Path("experiments/results/libgnss_diag_phase10"))
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--method-contains", default="rtkdiag_pf")
    parser.add_argument("--out-prefix", default="ppc_rtk_improvement_points")
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--hybrid-gate-m", type=float, default=10.0)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--fail-m", type=float, default=3.0)
    parser.add_argument("--nearby-dt-s", type=float, default=0.5)
    args = parser.parse_args()

    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    candidates = _load_candidates(args.candidate_base_dir, labels, args.city, args.run)
    rows = _read_csv(args.internal_epochs_csv)
    target = [
        row
        for row in rows
        if args.method_contains in row.get("method", "")
        and row.get("emitted_source") != "rtkdiag_candidate"
    ]
    classified = [
        _classify_row(
            row,
            candidates,
            ratio_min=args.ratio_min,
            residual_rms_max=args.residual_rms_max,
            hybrid_gate_m=args.hybrid_gate_m,
            pass_m=args.pass_m,
            fail_m=args.fail_m,
            nearby_dt_s=args.nearby_dt_s,
        )
        for row in target
    ]

    out_dir = Path("experiments/results")
    prefix = out_dir / args.out_prefix
    epoch_path = Path(f"{prefix}_epochs.csv")
    summary_path = Path(f"{prefix}_summary.csv")
    spans_path = Path(f"{prefix}_spans.csv")
    _write_csv(classified, epoch_path)
    _write_csv(_summary_rows(classified), summary_path)
    _write_csv(_span_rows(classified), spans_path)

    print(f"fallback epochs: {len(classified)}")
    print(f"wrote: {epoch_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {spans_path}")
    for cause, count in Counter(str(row["root_cause"]) for row in classified).most_common():
        print(f"{cause}: {count} ({_percent(count, len(classified)):.1f}%)")


if __name__ == "__main__":
    main()
