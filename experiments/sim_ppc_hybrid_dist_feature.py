#!/usr/bin/env python3
"""Diagnostic: is distance-to-hybrid a useful selector feature?

For each gated epoch (per city/run/policy=phase11dd):
  - oracle pick = truth-closest gated candidate (current 74.45% pool ceiling)
  - composite pick = current selector mode
  - hybrid-closest pick = candidate nearest to hybrid pos

Report agreement rates and PPC scores for each selector.
If hybrid-closest agrees with oracle frequently, dist-to-hybrid is a useful
feature to fold into composite. Otherwise it's noise.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402

from sim_ppc_selector_sweep import (  # noqa: E402
    _CANDIDATES_PHASE11V,
    _DIAG_ROOT,
    _eligible_for_run,
    _FULL_RUNS,
)

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


def _simulate(city, run, hybrid_pos, candidates, ref, ratio_min, rms_max, sort_mode):
    """For each ref epoch, compute oracle/composite/hybrid-closest selectors."""
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    n = len(ref)
    est_oracle = np.zeros((n, 3))
    est_composite = np.zeros((n, 3))
    est_hybrid_close = np.zeros((n, 3))

    n_gated = 0
    n_oracle_eq_comp = 0
    n_oracle_eq_hyb = 0
    n_comp_eq_hyb = 0

    # Per-epoch dist-to-truth distributions
    dist_oracle = []
    dist_composite = []
    dist_hyb_closest = []

    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        # default hybrid floor
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est_oracle[i] = est_composite[i] = est_hybrid_close[i] = np.asarray(hp)

        gated_pos = []
        gated_diag = []
        gated_label = []
        for label, cand_pos, cand_diag in candidates:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            gated_pos.append(np.asarray(cand))
            gated_diag.append(row)
            gated_label.append(label)

        if not gated_pos:
            continue

        n_gated += 1
        gated_pos_arr = np.stack(gated_pos)

        # Oracle: truth-closest
        oracle_idx = int(np.argmin(np.linalg.norm(gated_pos_arr - truth[i], axis=1)))
        est_oracle[i] = gated_pos_arr[oracle_idx]
        dist_oracle.append(np.linalg.norm(est_oracle[i] - truth[i]))

        # Composite: lowest sort_key (key is a tuple; use Python lexicographic min)
        comp_keys = [_rtkdiag_candidate_sort_key(d, mode=sort_mode) for d in gated_diag]
        comp_idx = min(range(len(comp_keys)), key=lambda j: comp_keys[j])
        est_composite[i] = gated_pos_arr[comp_idx]
        dist_composite.append(np.linalg.norm(est_composite[i] - truth[i]))

        # Hybrid-closest: nearest to hybrid_pos (if hybrid available)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            hp_arr = np.asarray(hp)
            hyb_idx = int(np.argmin(np.linalg.norm(gated_pos_arr - hp_arr, axis=1)))
        else:
            hyb_idx = comp_idx  # no hybrid available, fall back to composite
        est_hybrid_close[i] = gated_pos_arr[hyb_idx]
        dist_hyb_closest.append(np.linalg.norm(est_hybrid_close[i] - truth[i]))

        if oracle_idx == comp_idx:
            n_oracle_eq_comp += 1
        if oracle_idx == hyb_idx:
            n_oracle_eq_hyb += 1
        if comp_idx == hyb_idx:
            n_comp_eq_hyb += 1

    s_oracle = score_ppc2024(est_oracle, truth)
    s_composite = score_ppc2024(est_composite, truth)
    s_hyb_closest = score_ppc2024(est_hybrid_close, truth)

    print(f"\n{city}/{run}: ratio>={ratio_min} rms<={rms_max} sort={sort_mode} gated={n_gated}")
    print(f"  oracle      ppc={s_oracle.score_pct:.4f}% pass={s_oracle.pass_distance_m:.1f}/{s_oracle.total_distance_m:.1f}")
    print(f"  composite   ppc={s_composite.score_pct:.4f}% pass={s_composite.pass_distance_m:.1f}")
    print(f"  hyb-closest ppc={s_hyb_closest.score_pct:.4f}% pass={s_hyb_closest.pass_distance_m:.1f}")
    if n_gated:
        print(f"  agree(oracle, composite)   = {100*n_oracle_eq_comp/n_gated:.2f}%")
        print(f"  agree(oracle, hyb-closest) = {100*n_oracle_eq_hyb/n_gated:.2f}%")
        print(f"  agree(composite, hyb)      = {100*n_comp_eq_hyb/n_gated:.2f}%")
    if dist_oracle:
        print(f"  dist median: oracle={np.median(dist_oracle):.2f}m comp={np.median(dist_composite):.2f}m hyb-close={np.median(dist_hyb_closest):.2f}m")
        print(f"  dist p90:    oracle={np.percentile(dist_oracle,90):.2f}m comp={np.percentile(dist_composite,90):.2f}m hyb-close={np.percentile(dist_hyb_closest,90):.2f}m")

    return {
        "city": city, "run": run, "n_gated": n_gated,
        "oracle_pass": s_oracle.pass_distance_m,
        "composite_pass": s_composite.pass_distance_m,
        "hyb_closest_pass": s_hyb_closest.pass_distance_m,
        "total": s_oracle.total_distance_m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path,
                        default=Path("experiments/results/libgnss_rtk_pos_v5"))
    parser.add_argument("--policy", type=str, default="phase11dd")
    args = parser.parse_args()

    rows = []
    for city, run in _FULL_RUNS:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
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
        rows.append(_simulate(city, run, hybrid_pos, kept, ref,
                              ratio_min, rms_max, sort_mode))

    print("\n=== Aggregate ===")
    pass_o = sum(r["oracle_pass"] for r in rows)
    pass_c = sum(r["composite_pass"] for r in rows)
    pass_h = sum(r["hyb_closest_pass"] for r in rows)
    total = sum(r["total"] for r in rows)
    print(f"  oracle      ppc={100*pass_o/total:.4f}% pass={pass_o:.1f}/{total:.1f}")
    print(f"  composite   ppc={100*pass_c/total:.4f}% pass={pass_c:.1f}")
    print(f"  hyb-closest ppc={100*pass_h/total:.4f}% pass={pass_h:.1f}")


if __name__ == "__main__":
    main()
