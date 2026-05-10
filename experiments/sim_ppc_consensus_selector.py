#!/usr/bin/env python3
"""Offline consensus selector on Phase 11x candidate pool.

Selector_gap = +1.60pp (gated_oracle 63.21% vs Phase 11x 61.73%) cannot
be closed by single-feature sort keys alone (saturated). Hypothesis: a
candidate that lies inside the cluster of other gated candidates is more
likely to be truth-close. We measure each candidate's "agreement count"
(number of other gated candidates within agreement_radius_m), break ties
with the policy's existing sort key.

Modes:
- agreement: pick the candidate with most neighbours within radius
- median: pick the candidate closest to the median of all gated positions
- hybrid_anchor: pick the candidate closest to hybrid_pos (uses hybrid as
  anchor; small gain when hybrid is reliable, none when not)
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

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("datasets/PPC-Dataset-data")


@dataclass
class Result:
    city: str
    run: str
    mode: str
    radius_m: float
    ppc_pct: float
    pass_m: float
    total_m: float


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


def _select_consensus(positions: list[np.ndarray],
                      diags: list[dict],
                      sort_mode: str,
                      mode: str,
                      radius_m: float,
                      hybrid_pos: np.ndarray | None) -> int:
    """Return index of the chosen candidate from list."""
    if len(positions) == 1:
        return 0
    pts = np.asarray(positions, dtype=np.float64)
    if mode == "agreement":
        # Count neighbours within radius.
        diffs = pts[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        counts = (dists <= radius_m).sum(axis=1)  # includes self
        max_count = int(counts.max())
        # Tie-break with sort_key (smallest tuple wins).
        cands = [i for i in range(len(positions)) if int(counts[i]) == max_count]
        if len(cands) == 1:
            return cands[0]
        return min(cands, key=lambda i: _rtkdiag_candidate_sort_key(diags[i], mode=sort_mode))
    if mode == "median":
        median = np.median(pts, axis=0)
        d = np.linalg.norm(pts - median, axis=1)
        return int(np.argmin(d))
    if mode == "hybrid_anchor":
        if hybrid_pos is None:
            return min(range(len(positions)),
                       key=lambda i: _rtkdiag_candidate_sort_key(diags[i], mode=sort_mode))
        d = np.linalg.norm(pts - hybrid_pos, axis=1)
        return int(np.argmin(d))
    raise ValueError(f"unknown mode {mode}")


def _simulate(city, run, hybrid_pos_dict, candidates, ref, ratio_min, rms_max,
              sort_mode, mode, radius_m) -> Result:
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos_dict.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
            hp_arr = np.asarray(hp, dtype=np.float64)
        else:
            hp_arr = None
        gated_pos: list[np.ndarray] = []
        gated_diag: list[dict] = []
        for label, cand_pos, cand_diag in candidates:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            gated_pos.append(np.asarray(cand, dtype=np.float64))
            gated_diag.append(row)
        if gated_pos:
            idx = _select_consensus(gated_pos, gated_diag,
                                    sort_mode=sort_mode, mode=mode, radius_m=radius_m,
                                    hybrid_pos=hp_arr)
            est[i] = gated_pos[idx]
    score = score_ppc2024(est, truth)
    return Result(
        city=city, run=run, mode=mode, radius_m=radius_m,
        ppc_pct=float(score.score_pct),
        pass_m=float(score.pass_distance_m),
        total_m=float(score.total_distance_m),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11x")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_consensus_sweep_phase11x.csv")
    args = parser.parse_args()

    modes_radii = [
        ("agreement", 0.5),
        ("agreement", 1.0),
        ("agreement", 2.0),
        ("agreement", 5.0),
        ("agreement", 10.0),
        ("median", 0.0),
        ("hybrid_anchor", 0.0),
    ]

    rows: list[Result] = []
    for city, run in _FULL_RUNS:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        variant = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run, policy=str(args.policy), city=city,
        )
        # Verify city plumbing.
        _ = variant
        ratio_min = float(variant.rtkdiag_candidate_ratio_min)
        rms_max = float(variant.rtkdiag_candidate_residual_rms_max)
        sort_mode = str(variant.rtkdiag_candidate_select_mode)
        kept = _filter_rtkdiag_candidates_by_policy(
            _load_candidates_for_run(city, run),
            city=city, run=run, policy=str(args.policy),
        )
        print(f"\n{city}/{run}: ratio_min={ratio_min} rms_max={rms_max} sort_mode={sort_mode}")
        for mode, radius_m in modes_radii:
            res = _simulate(city, run, hybrid_pos, kept, ref, ratio_min, rms_max,
                            sort_mode=sort_mode, mode=mode, radius_m=radius_m)
            rows.append(res)
            tag = f"{mode}({radius_m})" if mode == "agreement" else mode
            print(f"  {tag:<22s}: ppc={res.ppc_pct:.4f}% (pass {res.pass_m:.1f})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "mode", "radius_m", "ppc_pct", "pass_m", "total_m"])
        for r in rows:
            w.writerow([r.city, r.run, r.mode, r.radius_m, f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}"])

    # Aggregate per-mode-uniform.
    print("\nAggregate per mode (uniform across runs):")
    by_mr: dict[tuple[str, float], list[Result]] = {}
    for r in rows:
        by_mr.setdefault((r.mode, r.radius_m), []).append(r)
    for (mode, radius_m), items in by_mr.items():
        ps = sum(r.pass_m for r in items)
        ts = sum(r.total_m for r in items)
        tag = f"{mode}({radius_m})" if mode == "agreement" else mode
        print(f"  {tag:<22s}: ppc={100*ps/ts:.4f}% (pass {ps:.1f}/{ts:.1f})")

    print("\nPer-run-best (oracle on consensus mode):")
    pass_sum = 0.0
    total_sum = 0.0
    best_summary: list[str] = []
    for city, run in _FULL_RUNS:
        sub = [r for r in rows if r.city == city and r.run == run]
        best = max(sub, key=lambda r: r.pass_m)
        pass_sum += best.pass_m
        total_sum += best.total_m
        tag = f"{best.mode}({best.radius_m})" if best.mode == "agreement" else best.mode
        best_summary.append(f"{city}/{run}={tag} ppc={best.ppc_pct:.4f}%")
    for s in best_summary:
        print(f"  {s}")
    print(f"  aggregate: ppc={100*pass_sum/total_sum:.4f}% (pass {pass_sum:.1f}/{total_sum:.1f})")


if __name__ == "__main__":
    main()
