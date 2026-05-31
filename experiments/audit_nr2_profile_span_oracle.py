#!/usr/bin/env python3
"""Oracle-check every existing nagoya/run2 profile over residual-gap spans."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
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
    except ValueError:
        return float("nan")


def _truth_xyz(row: dict[str, str]) -> np.ndarray:
    return np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)


def _distance_weights(rows: list[dict[str, str]]) -> list[float]:
    xyz = [_truth_xyz(row) for row in rows]
    weights: list[float] = []
    for i, pos in enumerate(xyz):
        if i == 0 or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(xyz[i - 1])):
            weights.append(0.0)
        else:
            weights.append(float(np.linalg.norm(pos - xyz[i - 1])))
    return weights


def _label_from_dir(path: Path, search_root: Path) -> str:
    try:
        rel = path.relative_to(search_root)
    except ValueError:
        rel = path
    return str(rel).strip("./").replace("/", "__")


def _excluded_by_contains(label: str, path: Path, tokens: list[str]) -> bool:
    haystack = f"{label}\n{path}".lower()
    return any(token and token.lower() in haystack for token in tokens)


def _discover_candidates(
    search_roots: list[Path],
    *,
    city: str,
    run: str,
    exclude_label_contains: list[str],
) -> tuple[list[dict[str, Any]], int]:
    suffix = f"{city}_{run}_full.pos"
    candidates: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    excluded = 0
    for root in search_roots:
        for pos_path in sorted(root.rglob(suffix)):
            base = pos_path.parent
            if base in seen_dirs:
                continue
            seen_dirs.add(base)
            label = _label_from_dir(base, root)
            if _excluded_by_contains(label, base, exclude_label_contains):
                excluded += 1
                continue
            diag_path = base / f"{city}_{run}_full.csv"
            pos, _status = _load_hybrid_pos_file(pos_path)
            if not pos:
                continue
            diag = _load_rtk_diag_file(diag_path) if diag_path.is_file() else {}
            candidates.append(
                {
                    "label": label,
                    "dir": str(base),
                    "pos": pos,
                    "diag": diag,
                    "has_diag": bool(diag),
                },
            )
    candidates.sort(key=lambda item: str(item["label"]))
    return candidates, excluded


def _top_counts(counter: Counter[str], limit: int = 8) -> str:
    return ",".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def _mean(values: list[float]) -> float | str:
    vals = [value for value in values if math.isfinite(value)]
    return float(sum(vals) / len(vals)) if vals else ""


def audit(
    *,
    internal_rows: list[dict[str, str]],
    span_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    top: int,
    pass_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    weights = _distance_weights(internal_rows)
    rows_by_epoch = {int(float(row["epoch"])): row for row in internal_rows}
    spans_out: list[dict[str, Any]] = []
    span_profiles: list[dict[str, Any]] = []
    profile_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "profile_label": "",
            "profile_dir": "",
            "has_diag": False,
            "available_epochs": 0,
            "pass_m": 0.0,
            "gated_pass_m": 0.0,
            "sum_error_m": 0.0,
            "error_count": 0,
        },
    )

    for span_index, span in enumerate(span_rows[:top], start=1):
        start = int(float(span["start_epoch"]))
        end = int(float(span["end_epoch"]))
        epoch_rows = [rows_by_epoch[i] for i in range(start, end + 1) if i in rows_by_epoch]
        current_pass_m = 0.0
        total_m = 0.0
        all_oracle_pass_m = 0.0
        gated_oracle_pass_m = 0.0
        all_best_errors: list[float] = []
        gated_best_errors: list[float] = []
        all_best_labels: Counter[str] = Counter()
        gated_best_labels: Counter[str] = Counter()
        per_profile: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "profile_label": "",
                "profile_dir": "",
                "has_diag": False,
                "available_epochs": 0,
                "pass_m": 0.0,
                "gated_pass_m": 0.0,
                "sum_error_m": 0.0,
                "error_count": 0,
            },
        )

        for row in epoch_rows:
            epoch = int(float(row["epoch"]))
            tow = round(float(row["tow"]), 1)
            truth = _truth_xyz(row)
            weight = weights[epoch]
            total_m += weight
            current_error = _float(row, "emit_to_ref_m")
            if math.isfinite(current_error) and current_error <= pass_m:
                current_pass_m += weight

            best_all: tuple[str, float] | None = None
            best_gated: tuple[str, float] | None = None
            for cand in candidates:
                pos = cand["pos"].get(tow)
                if pos is None or not np.all(np.isfinite(pos)):
                    continue
                label = str(cand["label"])
                dist = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - truth))
                rec = per_profile[label]
                rec["profile_label"] = label
                rec["profile_dir"] = cand["dir"]
                rec["has_diag"] = bool(cand["has_diag"])
                rec["available_epochs"] += 1
                rec["sum_error_m"] += dist
                rec["error_count"] += 1
                if dist <= pass_m:
                    rec["pass_m"] += weight
                diag_row = cand["diag"].get(tow)
                gated = _rtkdiag_candidate_gate(
                    diag_row,
                    ratio_min=ratio_min,
                    residual_rms_max=residual_rms_max,
                    status5_residual_rms_max=status5_residual_rms_max,
                )
                if gated and dist <= pass_m:
                    rec["gated_pass_m"] += weight
                if best_all is None or dist < best_all[1]:
                    best_all = (label, dist)
                if gated and (best_gated is None or dist < best_gated[1]):
                    best_gated = (label, dist)

            if best_all is not None:
                all_best_labels[best_all[0]] += 1
                all_best_errors.append(best_all[1])
                if best_all[1] <= pass_m:
                    all_oracle_pass_m += weight
            if best_gated is not None:
                gated_best_labels[best_gated[0]] += 1
                gated_best_errors.append(best_gated[1])
                if best_gated[1] <= pass_m:
                    gated_oracle_pass_m += weight

        current_fail_m = total_m - current_pass_m
        spans_out.append(
            {
                "span_index": span_index,
                "label": span.get("label", ""),
                "family": span.get("family", ""),
                "start_epoch": start,
                "end_epoch": end,
                "start_tow": span.get("start_tow", ""),
                "end_tow": span.get("end_tow", ""),
                "n_epochs": len(epoch_rows),
                "total_m": total_m,
                "current_pass_m": current_pass_m,
                "current_fail_m": current_fail_m,
                "all_profile_oracle_pass_m": all_oracle_pass_m,
                "all_profile_oracle_gain_m": all_oracle_pass_m - current_pass_m,
                "gated_profile_oracle_pass_m": gated_oracle_pass_m,
                "gated_profile_oracle_gain_m": gated_oracle_pass_m - current_pass_m,
                "all_best_mean_error_m": _mean(all_best_errors),
                "gated_best_mean_error_m": _mean(gated_best_errors),
                "all_best_labels": _top_counts(all_best_labels),
                "gated_best_labels": _top_counts(gated_best_labels),
            },
        )

        for rec in per_profile.values():
            profile_key = str(rec["profile_label"])
            profile_totals[profile_key]["profile_label"] = profile_key
            profile_totals[profile_key]["profile_dir"] = rec["profile_dir"]
            profile_totals[profile_key]["has_diag"] = rec["has_diag"]
            profile_totals[profile_key]["available_epochs"] += rec["available_epochs"]
            profile_totals[profile_key]["pass_m"] += rec["pass_m"]
            profile_totals[profile_key]["gated_pass_m"] += rec["gated_pass_m"]
            profile_totals[profile_key]["sum_error_m"] += rec["sum_error_m"]
            profile_totals[profile_key]["error_count"] += rec["error_count"]
            span_profiles.append(
                {
                    "span_index": span_index,
                    "span_label": span.get("label", ""),
                    "profile_label": profile_key,
                    "profile_dir": rec["profile_dir"],
                    "has_diag": rec["has_diag"],
                    "available_epochs": rec["available_epochs"],
                    "pass_m": rec["pass_m"],
                    "gated_pass_m": rec["gated_pass_m"],
                    "mean_error_m": rec["sum_error_m"] / rec["error_count"] if rec["error_count"] else "",
                },
            )

    profiles_out = []
    for rec in profile_totals.values():
        profiles_out.append(
            {
                "profile_label": rec["profile_label"],
                "profile_dir": rec["profile_dir"],
                "has_diag": rec["has_diag"],
                "available_epochs": rec["available_epochs"],
                "pass_m": rec["pass_m"],
                "gated_pass_m": rec["gated_pass_m"],
                "mean_error_m": rec["sum_error_m"] / rec["error_count"] if rec["error_count"] else "",
            },
        )
    profiles_out.sort(key=lambda row: (float(row["pass_m"]), -float(row["mean_error_m"] or 1.0e9)), reverse=True)
    span_profiles.sort(key=lambda row: (int(row["span_index"]), -float(row["pass_m"])))
    return spans_out, profiles_out, span_profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("spans_csv", type=Path)
    parser.add_argument("--search-root", action="append", type=Path, default=[Path("experiments/results")])
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument(
        "--exclude-label-contains",
        action="append",
        default=[],
        help="Skip candidate profiles whose label or path contains this case-insensitive token.",
    )
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/nr2_profile_span_oracle"))
    args = parser.parse_args(argv)

    candidates, excluded = _discover_candidates(
        args.search_root,
        city=args.city,
        run=args.run,
        exclude_label_contains=list(args.exclude_label_contains),
    )
    if not candidates:
        raise SystemExit("no profile candidates found")
    print(f"loaded profile candidates: {len(candidates)}")
    if excluded:
        print(f"excluded profile candidates: {excluded}")
    spans_out, profiles_out, span_profiles = audit(
        internal_rows=_read_csv(args.internal_epochs_csv),
        span_rows=_read_csv(args.spans_csv),
        candidates=candidates,
        top=int(args.top),
        pass_m=float(args.pass_m),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    _write_csv(Path(f"{args.out_prefix}_spans.csv"), spans_out)
    _write_csv(Path(f"{args.out_prefix}_profiles.csv"), profiles_out)
    _write_csv(Path(f"{args.out_prefix}_span_profiles.csv"), span_profiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
