#!/usr/bin/env python3
"""Replay label-pair offset corrections on Phase43 internal diagnostics."""

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

from analyze_phase43_span_oracle import _load_pool, _split_csv_values  # noqa: E402
from exp_ppc_ctrbpf_fgo import _rtkdiag_candidate_gate  # noqa: E402


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


def _distance_weights(rows: list[dict[str, str]]) -> list[float]:
    xyz = [(_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")) for row in rows]
    weights: list[float] = []
    for i, pos in enumerate(xyz):
        if i == 0:
            weights.append(0.0)
            continue
        prev = xyz[i - 1]
        if any(not math.isfinite(v) for v in (*prev, *pos)):
            weights.append(0.0)
        else:
            weights.append(float(math.dist(prev, pos)))
    return weights


def _base_label(row: dict[str, str]) -> str:
    label = row.get("rtkdiag_selected_base_label", "")
    if label:
        return label
    return row.get("rtkdiag_selected_label", "").removesuffix("+rnk")


def _xyz(row: dict[str, str], prefix: str) -> np.ndarray | None:
    xyz = np.array([_float(row, f"{prefix}_x"), _float(row, f"{prefix}_y"), _float(row, f"{prefix}_z")], dtype=np.float64)
    return xyz if np.all(np.isfinite(xyz)) else None


def _truth(row: dict[str, str]) -> np.ndarray | None:
    xyz = np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)
    return xyz if np.all(np.isfinite(xyz)) else None


def _candidate_by_label(candidates: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for cand in candidates:
        if str(cand["label"]) == label:
            return cand
    raise SystemExit(f"candidate label not loaded: {label}")


def _prepare_records(
    *,
    rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    selected_label: str,
    anchor_label: str,
    pass_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> list[dict[str, Any]]:
    weights = _distance_weights(rows)
    selected_cand = _candidate_by_label(candidates, selected_label)
    anchor_cand = _candidate_by_label(candidates, anchor_label)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        truth = _truth(row)
        current_error = _float(row, "emit_to_ref_m")
        base_pos = _xyz(row, "pf_epoch_end")
        selected = _base_label(row)
        tow = round(float(row["tow"]), 1)
        selected_pos = selected_cand["pos"].get(tow)
        anchor_pos = anchor_cand["pos"].get(tow)
        anchor_diag = anchor_cand["diag"].get(tow)
        anchor_gated = _rtkdiag_candidate_gate(
            anchor_diag,
            ratio_min=ratio_min,
            residual_rms_max=residual_rms_max,
            status5_residual_rms_max=status5_residual_rms_max,
        )
        offset_norm = float("nan")
        offset: np.ndarray | None = None
        if (
            selected_pos is not None
            and anchor_pos is not None
            and np.all(np.isfinite(selected_pos))
            and np.all(np.isfinite(anchor_pos))
        ):
            offset = np.asarray(anchor_pos, dtype=np.float64) - np.asarray(selected_pos, dtype=np.float64)
            offset_norm = float(np.linalg.norm(offset))
        records.append(
            {
                "epoch": int(float(row["epoch"])),
                "tow": tow,
                "weight": weights[i],
                "truth": truth,
                "base_pos": base_pos,
                "current_error": current_error,
                "current_pass": math.isfinite(current_error) and current_error <= pass_m,
                "selected": selected,
                "family_span": _float(row, "rtkdiag_candidate_family_span_m"),
                "selected_agreement": _float(row, "rtkdiag_candidate_agreement_count_1m"),
                "anchor_gated": anchor_gated,
                "offset": offset,
                "offset_norm": offset_norm,
            },
        )
    return records


def _replay(
    *,
    records: list[dict[str, Any]],
    selected_label: str,
    scale: float,
    family_span_min_m: float,
    selected_agreement_max: float,
    offset_norm_min_m: float,
    offset_norm_max_m: float,
    pass_m: float,
) -> dict[str, Any]:
    total = 0.0
    base_pass = 0.0
    replay_pass = 0.0
    overrides = 0
    good = 0
    bad = 0
    corrected_errors: list[float] = []
    for rec in records:
        weight = float(rec["weight"])
        total += weight
        if bool(rec["current_pass"]):
            base_pass += weight
        error = float(rec["current_error"])
        eligible = (
            rec["selected"] == selected_label
            and bool(rec["anchor_gated"])
            and rec["offset"] is not None
            and rec["base_pos"] is not None
            and rec["truth"] is not None
            and math.isfinite(float(rec["family_span"]))
            and float(rec["family_span"]) >= family_span_min_m
            and (
                not math.isfinite(float(rec["selected_agreement"]))
                or float(rec["selected_agreement"]) <= selected_agreement_max
            )
            and math.isfinite(float(rec["offset_norm"]))
            and offset_norm_min_m <= float(rec["offset_norm"]) <= offset_norm_max_m
        )
        if eligible:
            pos = np.asarray(rec["base_pos"], dtype=np.float64) + scale * np.asarray(rec["offset"], dtype=np.float64)
            error = float(np.linalg.norm(pos - np.asarray(rec["truth"], dtype=np.float64)))
            corrected_errors.append(error)
            overrides += 1
            chosen_pass = error <= pass_m
            if chosen_pass and not bool(rec["current_pass"]):
                good += 1
            elif bool(rec["current_pass"]) and not chosen_pass:
                bad += 1
        if math.isfinite(error) and error <= pass_m:
            replay_pass += weight
    corrected_errors.sort()
    median_corrected = corrected_errors[len(corrected_errors) // 2] if corrected_errors else ""
    p95_corrected = corrected_errors[min(len(corrected_errors) - 1, int(math.ceil(0.95 * len(corrected_errors))) - 1)] if corrected_errors else ""
    return {
        "scale": scale,
        "family_span_min_m": family_span_min_m,
        "selected_agreement_max": selected_agreement_max,
        "offset_norm_min_m": offset_norm_min_m,
        "offset_norm_max_m": offset_norm_max_m,
        "base_pass_m": base_pass,
        "replay_pass_m": replay_pass,
        "gain_m": replay_pass - base_pass,
        "total_m": total,
        "base_score_pct": 100.0 * base_pass / total if total > 0.0 else "",
        "replay_score_pct": 100.0 * replay_pass / total if total > 0.0 else "",
        "overrides": overrides,
        "good_overrides": good,
        "bad_overrides": bad,
        "median_corrected_error_m": median_corrected,
        "p95_corrected_error_m": p95_corrected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidate-dirs", required=True)
    parser.add_argument("--selected-label", default="xd_gici_hs")
    parser.add_argument("--anchor-label", default="xd_gici_c4")
    parser.add_argument("--scales", default="0,0.5,0.75,1,1.25,1.5,1.6,1.75,2,2.25")
    parser.add_argument("--family-span-min-values", default="0,20,40,60,80,100")
    parser.add_argument("--selected-agreement-max-values", default="2,4,6,8,10,12,20,99")
    parser.add_argument("--offset-norm-min-values", default="0,0.5,1.0")
    parser.add_argument("--offset-norm-max-values", default="2,5,10,999")
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/results/phase65_label_pair_offset_sweep.csv"))
    args = parser.parse_args(argv)

    candidates = _load_pool(_split_csv_values(args.labels), _split_csv_values(args.candidate_dirs), args.city, args.run)
    records = _prepare_records(
        rows=_read_csv(args.internal_epochs_csv),
        candidates=candidates,
        selected_label=str(args.selected_label),
        anchor_label=str(args.anchor_label),
        pass_m=float(args.pass_m),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    out: list[dict[str, Any]] = []
    for scale in (float(v) for v in _split_csv_values(args.scales)):
        for span_min in (float(v) for v in _split_csv_values(args.family_span_min_values)):
            for agree_max in (float(v) for v in _split_csv_values(args.selected_agreement_max_values)):
                for offset_min in (float(v) for v in _split_csv_values(args.offset_norm_min_values)):
                    for offset_max in (float(v) for v in _split_csv_values(args.offset_norm_max_values)):
                        if offset_max < offset_min:
                            continue
                        out.append(
                            _replay(
                                records=records,
                                selected_label=str(args.selected_label),
                                scale=scale,
                                family_span_min_m=span_min,
                                selected_agreement_max=agree_max,
                                offset_norm_min_m=offset_min,
                                offset_norm_max_m=offset_max,
                                pass_m=float(args.pass_m),
                            ),
                        )
    out.sort(key=lambda row: float(row["gain_m"]), reverse=True)
    _write_csv(args.out_csv, out)
    print(f"loaded candidates: {len(candidates)}")
    if out:
        print("best:", out[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
