#!/usr/bin/env python3
"""Sweep: composite_key * (1 + beta * dist_to_hybrid^2) per-run.

Diagnostic shows hyb-closest agrees with oracle more often than composite for
several runs (t/r1 37.7%, t/r2 43.9%, n/r1 35.7%) — but pure hyb-closest is
worse because when it disagrees with oracle, the pick is far. Adding hybrid
distance as a *tiebreaker* on top of composite may capture some of the gap.

For each (city, run, beta), pick candidate minimizing
    composite_key_scalar * (1 + beta * dist_hyb^2)
where composite_key_scalar = first element of the existing sort tuple.
Report PPC for beta in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0}.

beta=0 reproduces current composite. Improvement at any beta>0 is signal.
"""

from __future__ import annotations

import argparse
import sys
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
_BETAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


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


def _composite_scalar(diag, mode):
    """Extract scalar from composite sort tuple (first element)."""
    k = _rtkdiag_candidate_sort_key(diag, mode=mode)
    if isinstance(k, tuple):
        return float(k[0])
    return float(k)


def _simulate(city, run, hybrid_pos, candidates, ref, ratio_min, rms_max, sort_mode, beta):
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    n = len(ref)
    est = np.zeros((n, 3))

    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp)

        gated_pos = []
        gated_diag = []
        for label, cand_pos, cand_diag in candidates:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            gated_pos.append(np.asarray(cand))
            gated_diag.append(row)

        if not gated_pos:
            continue

        gated_pos_arr = np.stack(gated_pos)
        comp_scalars = np.asarray([_composite_scalar(d, sort_mode) for d in gated_diag])

        # If hybrid available, weight by (1 + beta * dist^2). Else pure composite.
        if beta > 0 and hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            d2 = np.sum((gated_pos_arr - np.asarray(hp))**2, axis=1)
            keys = comp_scalars * (1.0 + beta * d2)
        else:
            keys = comp_scalars

        idx = int(np.argmin(keys))
        est[i] = gated_pos_arr[idx]

    s = score_ppc2024(est, truth)
    return float(s.score_pct), float(s.pass_distance_m), float(s.total_distance_m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path,
                        default=Path("experiments/results/libgnss_rtk_pos_v5"))
    parser.add_argument("--policy", type=str, default="phase11dd")
    parser.add_argument("--betas", type=str, default=",".join(str(b) for b in _BETAS))
    args = parser.parse_args()

    betas = [float(s) for s in args.betas.split(",") if s.strip()]
    per_run_results = {}  # (city,run) -> [(beta, ppc, pass, total)]
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
        print(f"\n{city}/{run}: ratio>={ratio_min} rms<={rms_max} sort={sort_mode}")
        results = []
        for beta in betas:
            ppc, p, t = _simulate(city, run, hybrid_pos, kept, ref,
                                  ratio_min, rms_max, sort_mode, beta)
            results.append((beta, ppc, p, t))
            print(f"  beta={beta:>8.4f}: ppc={ppc:.4f}% pass={p:.1f}/{t:.1f}")
        per_run_results[(city, run)] = results

    # Best per run
    print("\n=== Best beta per run ===")
    best_total_pass = 0.0
    best_total_total = 0.0
    base_total_pass = 0.0
    for (city, run), results in per_run_results.items():
        results_sorted = sorted(results, key=lambda r: -r[2])
        best_beta, best_ppc, best_pass, total = results_sorted[0]
        # baseline = beta=0
        base_pass = next(p for (b, _, p, _) in results if b == 0.0)
        delta = best_pass - base_pass
        print(f"  {city}/{run}: best beta={best_beta} ppc={best_ppc:.4f}% (vs beta=0 +{delta:+.1f}m)")
        best_total_pass += best_pass
        best_total_total += total
        base_total_pass += base_pass
    print(f"\nBest per-run aggregate: ppc={100*best_total_pass/best_total_total:.4f}% (vs all-beta=0 baseline)")
    print(f"All-beta=0 aggregate:   ppc={100*base_total_pass/best_total_total:.4f}%")
    print(f"Delta:                  +{100*(best_total_pass-base_total_pass)/best_total_total:.4f}pp")


if __name__ == "__main__":
    main()
