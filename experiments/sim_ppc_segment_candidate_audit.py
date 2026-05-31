#!/usr/bin/env python3
"""Audit candidate availability on oracle-miss segments.

Given segments from ``sim_ppc_oracle_miss_diagnosis.py``, compare the best
truth-distance among:

* gated candidates under the current policy
* all loaded candidates ignoring the RTKDiag gate

If ungated candidates are good but gated candidates are absent/bad, the next
axis is gate/trajectory selection.  If ungated candidates are also bad/missing,
the next axis is candidate generation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    CTRBPFConfig,
    _apply_rtkdiag_run_index_policy,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _parse_label_list,
    _rtkdiag_candidate_gate,
    _rtkdiag_fixed_output_ok,
    _rtkdiag_local_ungate_labels,
    _rtkdiag_local_ungate_labels_for_tow,
)
from gnss_gpu.ppc_score import ppc_segment_distances  # noqa: E402
from sim_ppc_oracle_miss_diagnosis import _load_candidates, _run_phase_rows  # noqa: E402

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


def _segment_rows(path: Path, top: int, kinds: set[str]) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if not kinds or str(r.get("kind", "")) in kinds]
    rows.sort(key=lambda r: float(r.get("weight_m", "0") or 0.0), reverse=True)
    return rows[:top]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    p.add_argument("--phase-runs-csv", type=Path, required=True)
    p.add_argument("--segments-csv", type=Path, required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--kinds", default="pool_miss,no_gated_candidate")
    p.add_argument("--threshold-m", type=float, default=0.5)
    p.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_segment_candidate_audit.csv")
    args = p.parse_args()

    kinds = {k.strip() for k in str(args.kinds).split(",") if k.strip()}
    phase_by_run = {
        (str(r["city"]), str(r["run"])): r
        for r in _run_phase_rows(args.phase_runs_csv)
    }
    segs = _segment_rows(args.segments_csv, int(args.top), kinds)
    out_rows: list[dict[str, object]] = []
    cache = {}
    for seg in segs:
        city = str(seg["city"])
        run = str(seg["run"])
        key = (city, run)
        if key not in cache:
            phase_row = phase_by_run.get(key)
            if phase_row is None:
                raise SystemExit(f"missing phase row for {city}/{run}")
            labels = _parse_label_list(str(phase_row["rtkdiag_candidate_labels"]))
            loaded, missing = _load_candidates(city, run, labels)
            kept = _filter_rtkdiag_candidates_by_policy(loaded, city=city, run=run, policy=str(args.policy))
            cfg = _apply_rtkdiag_run_index_policy(
                CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
                city=city,
                run=run,
                policy=str(args.policy),
            )
            ref = _load_full_reference(args.data_root / city / run / "reference.csv")
            truth = np.asarray([p for _tow, p in ref], dtype=np.float64)
            weights = ppc_segment_distances(truth)
            cache[key] = {
                "loaded": loaded,
                "kept": kept,
                "missing": missing,
                "ratio_min": float(cfg.rtkdiag_candidate_ratio_min),
                "rms_max": float(cfg.rtkdiag_candidate_residual_rms_max),
                "local_ungate_windows": tuple(cfg.rtkdiag_candidate_local_ungate_windows),
                "local_ungate_tow_windows": tuple(cfg.rtkdiag_candidate_local_ungate_tow_windows),
                "ref": ref,
                "truth": truth,
                "weights": weights,
            }
        data = cache[key]
        start = int(seg["start_idx"])
        end = int(seg["end_idx"])
        best_gated_dists = []
        best_all_dists = []
        gated_counts = []
        avail_counts = []
        best_gated_labels = Counter()
        best_all_labels = Counter()
        weight_total = 0.0
        gated_pass_weight = 0.0
        all_pass_weight = 0.0
        for idx in range(start, end + 1):
            tow = round(float(data["ref"][idx][0]), 1)
            truth_i = data["truth"][idx]
            weight = float(data["weights"][idx])
            weight_total += weight
            local_ungate_labels = _rtkdiag_local_ungate_labels(
                data["local_ungate_windows"],
                int(idx),
            )
            if local_ungate_labels is None:
                local_ungate_labels = _rtkdiag_local_ungate_labels_for_tow(
                    data["local_ungate_tow_windows"],
                    float(tow),
                )
            all_opts = []
            gated_opts = []
            for label, cand_pos, cand_diag in data["kept"]:
                cand = cand_pos.get(tow)
                if cand is None or not np.all(np.isfinite(cand)) or np.all(np.asarray(cand) == 0.0):
                    continue
                dist = float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - truth_i))
                all_opts.append((label, dist))
                row = cand_diag.get(tow)
                gate_ok = _rtkdiag_candidate_gate(
                    row,
                    ratio_min=float(data["ratio_min"]),
                    residual_rms_max=float(data["rms_max"]),
                )
                local_ungate_ok = (
                    local_ungate_labels is not None
                    and _rtkdiag_fixed_output_ok(row)
                    and (not local_ungate_labels or label in local_ungate_labels)
                )
                if gate_ok or local_ungate_ok:
                    gated_opts.append((label, dist))
            avail_counts.append(len(all_opts))
            gated_counts.append(len(gated_opts))
            if all_opts:
                label, dist = min(all_opts, key=lambda x: x[1])
                best_all_labels[label] += 1
                best_all_dists.append(dist)
                if dist <= float(args.threshold_m):
                    all_pass_weight += weight
            if gated_opts:
                label, dist = min(gated_opts, key=lambda x: x[1])
                best_gated_labels[label] += 1
                best_gated_dists.append(dist)
                if dist <= float(args.threshold_m):
                    gated_pass_weight += weight
        mean_best_gated = float(np.mean(best_gated_dists)) if best_gated_dists else float("inf")
        mean_best_all = float(np.mean(best_all_dists)) if best_all_dists else float("inf")
        pass_gain_ungated = all_pass_weight - gated_pass_weight
        if not best_all_dists:
            diagnosis = "no_candidate_position"
        elif pass_gain_ungated > 1.0:
            diagnosis = "gate_too_strict"
        elif mean_best_all <= 2.0 and mean_best_gated > 2.0:
            diagnosis = "gate_too_strict_low_weight"
        else:
            diagnosis = "candidate_generation_needed"
        out_rows.append({
            "city": city,
            "run": run,
            "kind": seg["kind"],
            "start_idx": start,
            "end_idx": end,
            "start_tow": seg["start_tow"],
            "end_tow": seg["end_tow"],
            "epochs": int(seg["epochs"]),
            "weight_m": float(seg["weight_m"]),
            "mean_available_candidates": float(np.mean(avail_counts)) if avail_counts else 0.0,
            "mean_gated_candidates": float(np.mean(gated_counts)) if gated_counts else 0.0,
            "best_gated_mean_error_m": mean_best_gated,
            "best_all_mean_error_m": mean_best_all,
            "gated_pass_weight_m": gated_pass_weight,
            "all_ungated_pass_weight_m": all_pass_weight,
            "ungated_extra_pass_weight_m": pass_gain_ungated,
            "diagnosis": diagnosis,
            "top_best_gated_labels": ",".join(f"{k}:{v}" for k, v in best_gated_labels.most_common(5)),
            "top_best_all_labels": ",".join(f"{k}:{v}" for k, v in best_all_labels.most_common(5)),
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"saved: {args.out_csv}")
    print("Top audited segments:")
    for row in out_rows[:12]:
        print(
            f"  {row['city']}/{row['run']} {row['kind']} weight={float(row['weight_m']):.1f}m "
            f"avail={float(row['mean_available_candidates']):.1f} gated={float(row['mean_gated_candidates']):.1f} "
            f"best_all={float(row['best_all_mean_error_m']):.2f}m "
            f"best_gated={float(row['best_gated_mean_error_m']):.2f}m "
            f"ungated_gain={float(row['ungated_extra_pass_weight_m']):.1f}m "
            f"{row['diagnosis']}"
        )


if __name__ == "__main__":
    main()
