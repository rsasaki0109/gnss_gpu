#!/usr/bin/env python3
"""Replay truth-free consensus guards on Phase43 internal diagnostics."""

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
    label = row.get("rtkdiag_selected_label", "").removesuffix("+rnk")
    return label or row.get("emitted_source", "")


def _candidate_options(
    candidates: list[dict[str, Any]],
    tow: float,
    *,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand in candidates:
        pos = cand["pos"].get(tow)
        if pos is None or not np.all(np.isfinite(pos)):
            continue
        diag = cand["diag"].get(tow)
        if not _rtkdiag_candidate_gate(
            diag,
            ratio_min=ratio_min,
            residual_rms_max=residual_rms_max,
            status5_residual_rms_max=status5_residual_rms_max,
        ):
            continue
        out.append(
            {
                "label": str(cand["label"]),
                "pos": np.asarray(pos, dtype=np.float64),
                "diag_rms": _diag_float(diag, "final_residual_rms") if diag else float("nan"),
                "status": int(_diag_float(diag, "final_status")) if diag else 0,
            },
        )
    return out


def _agreement_counts(options: list[dict[str, Any]], radius_m: float) -> list[int]:
    counts: list[int] = []
    for opt in options:
        pos = opt["pos"]
        count = 0
        for other in options:
            if float(np.linalg.norm(pos - other["pos"])) <= radius_m:
                count += 1
        counts.append(count)
    return counts


def _choose_consensus(
    options: list[dict[str, Any]],
    *,
    radius_m: float,
    min_agreement: int,
    current_label: str,
    exclude_current: bool,
) -> dict[str, Any] | None:
    if exclude_current:
        options = [opt for opt in options if opt["label"] != current_label]
    if not options:
        return None
    counts = _agreement_counts(options, radius_m)
    ranked = []
    for opt, count in zip(options, counts):
        if count < min_agreement:
            continue
        rms = opt["diag_rms"]
        ranked.append((count, -float(rms) if math.isfinite(rms) else float("-inf"), opt["label"], opt))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    chosen = dict(ranked[0][3])
    chosen["agreement_count"] = ranked[0][0]
    return chosen


def replay(
    *,
    rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    selected_labels: set[str],
    family_span_min_m: float,
    selected_agreement_max: float,
    radius_m: float,
    min_agreement: int,
    exclude_current: bool,
    pass_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> dict[str, Any]:
    weights = _distance_weights(rows)
    base_pass = 0.0
    replay_pass = 0.0
    total = sum(weights)
    overrides = 0
    good_overrides = 0
    bad_overrides = 0
    override_labels: dict[str, int] = {}
    for i, row in enumerate(rows):
        truth = np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)
        current_error = _float(row, "emit_to_ref_m")
        weight = weights[i]
        if math.isfinite(current_error) and current_error <= pass_m:
            base_pass += weight
        selected = _base_label(row)
        use_current = True
        chosen_error = current_error
        family_span = _float(row, "rtkdiag_candidate_family_span_m")
        selected_agreement = _float(row, "rtkdiag_candidate_agreement_count_1m")
        if (
            selected in selected_labels
            and math.isfinite(family_span)
            and family_span >= family_span_min_m
            and (not math.isfinite(selected_agreement) or selected_agreement <= selected_agreement_max)
        ):
            tow = round(float(row["tow"]), 1)
            options = _candidate_options(
                candidates,
                tow,
                ratio_min=ratio_min,
                residual_rms_max=residual_rms_max,
                status5_residual_rms_max=status5_residual_rms_max,
            )
            chosen = _choose_consensus(
                options,
                radius_m=radius_m,
                min_agreement=min_agreement,
                current_label=selected,
                exclude_current=exclude_current,
            )
            if chosen is not None:
                chosen_error = float(np.linalg.norm(chosen["pos"] - truth))
                use_current = False
                overrides += 1
                override_labels[chosen["label"]] = override_labels.get(chosen["label"], 0) + 1
                current_pass = math.isfinite(current_error) and current_error <= pass_m
                chosen_pass = chosen_error <= pass_m
                if chosen_pass and not current_pass:
                    good_overrides += 1
                elif current_pass and not chosen_pass:
                    bad_overrides += 1
        if use_current:
            chosen_error = current_error
        if math.isfinite(chosen_error) and chosen_error <= pass_m:
            replay_pass += weight
    return {
        "base_pass_m": base_pass,
        "replay_pass_m": replay_pass,
        "gain_m": replay_pass - base_pass,
        "total_m": total,
        "base_score_pct": 100.0 * base_pass / total if total > 0.0 else "",
        "replay_score_pct": 100.0 * replay_pass / total if total > 0.0 else "",
        "overrides": overrides,
        "good_overrides": good_overrides,
        "bad_overrides": bad_overrides,
        "override_labels": ",".join(f"{k}:{v}" for k, v in sorted(override_labels.items())),
    }


def prepare_records(
    *,
    rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    min_agreement_values: list[int],
    radius_m: float,
    exclude_current: bool,
    pass_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> list[dict[str, Any]]:
    weights = _distance_weights(rows)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        truth = np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)
        current_error = _float(row, "emit_to_ref_m")
        selected = _base_label(row)
        tow = round(float(row["tow"]), 1)
        options = _candidate_options(
            candidates,
            tow,
            ratio_min=ratio_min,
            residual_rms_max=residual_rms_max,
            status5_residual_rms_max=status5_residual_rms_max,
        )
        if exclude_current:
            options = [opt for opt in options if opt["label"] != selected]
        counts = _agreement_counts(options, radius_m) if options else []
        ranked: list[tuple[int, float, str, dict[str, Any]]] = []
        for opt, count in zip(options, counts):
            rms = opt["diag_rms"]
            ranked.append((count, -float(rms) if math.isfinite(rms) else float("-inf"), opt["label"], opt))
        ranked.sort(reverse=True)
        best_by_min: dict[int, dict[str, Any] | None] = {}
        for min_agreement in min_agreement_values:
            chosen = None
            for count, _neg_rms, _label, opt in ranked:
                if count >= min_agreement:
                    chosen = dict(opt)
                    chosen["agreement_count"] = count
                    chosen["error_m"] = float(np.linalg.norm(opt["pos"] - truth))
                    break
            best_by_min[min_agreement] = chosen
        records.append(
            {
                "weight": weights[i],
                "current_error": current_error,
                "current_pass": math.isfinite(current_error) and current_error <= pass_m,
                "selected": selected,
                "family_span": _float(row, "rtkdiag_candidate_family_span_m"),
                "selected_agreement": _float(row, "rtkdiag_candidate_agreement_count_1m"),
                "best_by_min": best_by_min,
            },
        )
    return records


def replay_prepared(
    *,
    records: list[dict[str, Any]],
    selected_labels: set[str],
    family_span_min_m: float,
    selected_agreement_max: float,
    min_agreement: int,
    pass_m: float,
) -> dict[str, Any]:
    base_pass = 0.0
    replay_pass = 0.0
    total = 0.0
    overrides = 0
    good_overrides = 0
    bad_overrides = 0
    override_labels: dict[str, int] = {}
    for rec in records:
        weight = float(rec["weight"])
        total += weight
        if bool(rec["current_pass"]):
            base_pass += weight
        chosen_error = float(rec["current_error"])
        selected = str(rec["selected"])
        family_span = float(rec["family_span"])
        selected_agreement = float(rec["selected_agreement"])
        if (
            selected in selected_labels
            and math.isfinite(family_span)
            and family_span >= family_span_min_m
            and (not math.isfinite(selected_agreement) or selected_agreement <= selected_agreement_max)
        ):
            chosen = rec["best_by_min"].get(min_agreement)
            if chosen is not None:
                chosen_error = float(chosen["error_m"])
                overrides += 1
                override_labels[str(chosen["label"])] = override_labels.get(str(chosen["label"]), 0) + 1
                chosen_pass = chosen_error <= pass_m
                if chosen_pass and not bool(rec["current_pass"]):
                    good_overrides += 1
                elif bool(rec["current_pass"]) and not chosen_pass:
                    bad_overrides += 1
        if math.isfinite(chosen_error) and chosen_error <= pass_m:
            replay_pass += weight
    return {
        "base_pass_m": base_pass,
        "replay_pass_m": replay_pass,
        "gain_m": replay_pass - base_pass,
        "total_m": total,
        "base_score_pct": 100.0 * base_pass / total if total > 0.0 else "",
        "replay_score_pct": 100.0 * replay_pass / total if total > 0.0 else "",
        "overrides": overrides,
        "good_overrides": good_overrides,
        "bad_overrides": bad_overrides,
        "override_labels": ",".join(f"{k}:{v}" for k, v in sorted(override_labels.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidate-dirs", required=True)
    parser.add_argument("--selected-labels", default="xd_gici_w5,xd_gici_ir,xd_gici_z,xd_gici_zr,xd_gici_mb,xd_gici_r4,xd_gici_combo,xd_gici_he,xd_gici_la")
    parser.add_argument("--family-span-min-values", default="10,20,30,40,50,75,100")
    parser.add_argument("--selected-agreement-max-values", default="4,8,12,16,20,99")
    parser.add_argument("--radius-m", type=float, default=1.0)
    parser.add_argument("--min-agreement-values", default="3,5,8,10,12,15")
    parser.add_argument("--include-current", action="store_true")
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/results/phase59_consensus_guard_sweep.csv"))
    args = parser.parse_args(argv)

    candidates = _load_pool(_split_csv_values(args.labels), _split_csv_values(args.candidate_dirs), args.city, args.run)
    rows = _read_csv(args.internal_epochs_csv)
    selected_labels = set(_split_csv_values(args.selected_labels))
    min_agreement_values = [int(float(v)) for v in _split_csv_values(args.min_agreement_values)]
    records = prepare_records(
        rows=rows,
        candidates=candidates,
        min_agreement_values=min_agreement_values,
        radius_m=float(args.radius_m),
        exclude_current=not bool(args.include_current),
        pass_m=float(args.pass_m),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    out: list[dict[str, Any]] = []
    for span_min in (float(v) for v in _split_csv_values(args.family_span_min_values)):
        for agree_max in (float(v) for v in _split_csv_values(args.selected_agreement_max_values)):
            for min_agree in min_agreement_values:
                result = replay_prepared(
                    records=records,
                    selected_labels=selected_labels,
                    family_span_min_m=span_min,
                    selected_agreement_max=agree_max,
                    min_agreement=min_agree,
                    pass_m=float(args.pass_m),
                )
                out.append(
                    {
                        "family_span_min_m": span_min,
                        "selected_agreement_max": agree_max,
                        "radius_m": float(args.radius_m),
                        "min_agreement": min_agree,
                        "exclude_current": not bool(args.include_current),
                        **result,
                    },
                )
    out.sort(key=lambda row: float(row["gain_m"]), reverse=True)
    _write_csv(args.out_csv, out)
    print(f"loaded candidates: {len(candidates)}")
    if out:
        print("best:", out[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
