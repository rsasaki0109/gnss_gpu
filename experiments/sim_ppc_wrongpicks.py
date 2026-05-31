#!/usr/bin/env python3
"""Analyse wrong-pick epochs: which features predict the truth-best?

For each run, walk through epochs where the gated_oracle picks a candidate
different from the policy. Inspect what features distinguish truth-best
from policy-pick. This informs whether a learned ranker could close the
gated_gap +1.60pp.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
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
    _diag_float,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
)

from sim_ppc_selector_sweep import (  # noqa: E402
    _CANDIDATES_PHASE11V,
    _DIAG_ROOT,
    _eligible_for_run,
    _FULL_RUNS,
)

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


def _load_candidates_for_run(city, run):
    out = []
    for label, dir_name, restrict in _CANDIDATES_PHASE11V:
        if not _eligible_for_run(city, run, restrict):
            continue
        pos_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
        diag_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        pos, _ = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        out.append((label, pos, diag))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11x")
    parser.add_argument("--max-rows", type=int, default=80,
                        help="Max wrong-pick rows to print per run")
    args = parser.parse_args()

    feature_diffs: dict[str, list[float]] = {
        "delta_residual_rms": [], "delta_ratio": [],
        "delta_nrows": [], "delta_abs_max": [],
        "delta_pdop": [], "delta_sats": [],
        "policy_dist_to_truth_m": [], "best_dist_to_truth_m": [],
    }

    for city, run in _FULL_RUNS:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        truth_by_t = {round(float(t), 1): pos for t, pos in ref}
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        variant = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run, policy=str(args.policy), city=city,
        )
        ratio_min = float(variant.rtkdiag_candidate_ratio_min)
        rms_max = float(variant.rtkdiag_candidate_residual_rms_max)
        sort_mode = str(variant.rtkdiag_candidate_select_mode)
        kept = _filter_rtkdiag_candidates_by_policy(
            _load_candidates_for_run(city, run),
            city=city, run=run, policy=str(args.policy),
        )
        wrongs = 0
        wrong_pass_diff_m = 0.0
        for tow, truth in ref:
            t_key = round(float(tow), 1)
            options: list[tuple[str, np.ndarray, dict]] = []
            for label, cand_pos, cand_diag in kept:
                row = cand_diag.get(t_key)
                if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                    continue
                cand = cand_pos.get(t_key)
                if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                    continue
                options.append((label, np.asarray(cand, dtype=np.float64), row))
            if not options:
                continue
            # Policy pick.
            policy_idx = min(range(len(options)),
                             key=lambda i: _rtkdiag_candidate_sort_key(options[i][2], mode=sort_mode))
            # Oracle pick (closest to truth).
            errs = [float(np.linalg.norm(p - truth)) for _, p, _ in options]
            best_idx = int(np.argmin(errs))
            if policy_idx == best_idx:
                continue
            wrongs += 1
            wrong_pass_diff_m += errs[policy_idx] - errs[best_idx]
            policy_row = options[policy_idx][2]
            best_row = options[best_idx][2]
            feature_diffs["delta_residual_rms"].append(_diag_float(best_row, "final_residual_rms") - _diag_float(policy_row, "final_residual_rms"))
            feature_diffs["delta_ratio"].append(_diag_float(best_row, "final_ratio") - _diag_float(policy_row, "final_ratio"))
            feature_diffs["delta_nrows"].append(_diag_float(best_row, "final_update_rows") - _diag_float(policy_row, "final_update_rows"))
            feature_diffs["delta_abs_max"].append(_diag_float(best_row, "final_residual_abs_max") - _diag_float(policy_row, "final_residual_abs_max"))
            feature_diffs["delta_pdop"].append(_diag_float(best_row, "final_pdop") - _diag_float(policy_row, "final_pdop"))
            feature_diffs["delta_sats"].append(_diag_float(best_row, "final_sats") - _diag_float(policy_row, "final_sats"))
            feature_diffs["policy_dist_to_truth_m"].append(errs[policy_idx])
            feature_diffs["best_dist_to_truth_m"].append(errs[best_idx])

        gate_total = sum(1 for _ in feature_diffs["delta_residual_rms"][-wrongs:]) if wrongs > 0 else 0
        print(f"\n{city}/{run}: wrong picks={wrongs}, sum_dist_diff={wrong_pass_diff_m:.1f}m")
        # Show top 10 by dist_diff.
        # Reconstruct rank using saved arrays.
        if wrongs > 0:
            recent_pol = feature_diffs["policy_dist_to_truth_m"][-wrongs:]
            recent_best = feature_diffs["best_dist_to_truth_m"][-wrongs:]
            order = sorted(range(wrongs), key=lambda i: recent_pol[i] - recent_best[i], reverse=True)
            top_n = min(args.max_rows, wrongs)
            shown = 0
            for j in order:
                d = recent_pol[j] - recent_best[j]
                if d < 0.5:
                    break
                shown += 1
                if shown >= top_n:
                    break
            print(f"  wrongs with dist_diff>=0.5m: {shown}/{wrongs}")

    # Aggregate feature stats.
    n = len(feature_diffs["delta_residual_rms"])
    print(f"\n========== aggregate over {n} wrong picks ==========")
    for key in ("delta_residual_rms", "delta_ratio", "delta_nrows", "delta_abs_max", "delta_pdop", "delta_sats"):
        arr = np.asarray(feature_diffs[key], dtype=np.float64)
        if arr.size == 0:
            continue
        # Filter out NaN / inf for stats.
        m = np.isfinite(arr)
        a = arr[m]
        print(f"  {key:24s}: mean={a.mean():+.4f}, median={np.median(a):+.4f}, std={a.std():.4f}, +pos%={100*(a>0).mean():.1f}%")
    pol = np.asarray(feature_diffs["policy_dist_to_truth_m"], dtype=np.float64)
    bst = np.asarray(feature_diffs["best_dist_to_truth_m"], dtype=np.float64)
    diff = pol - bst
    big = diff >= 0.5
    print(f"\n  wrong picks with policy-best dist >= 0.5m: {big.sum()} / {n}, sum_dist_diff={diff[big].sum():.1f}m")
    print(f"  total wrong-pick distance loss: {diff.sum():.1f}m  (cap at 50m: {np.minimum(diff, 50).sum():.1f}m)")


if __name__ == "__main__":
    main()
