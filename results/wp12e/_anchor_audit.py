#!/usr/bin/env python3
"""WP12e anchor supply audit: FIX/FLOAT coverage vs reference truth."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "python"))

from score_vs_inuex35 import load_reference_grid  # noqa: E402
from wp11_run_tc_fgo import parse_rover_tows_from_obs, resolve_data_root, resolve_run_dir  # noqa: E402
from wp5_run_anchored_fgo import (  # noqa: E402
    anchor_sigma_m,
    classify_anchor_status,
    load_rtk_pos_extended,
    nearest_anchor_distance_epochs,
)

BASELINE_POS = {
    "run1": ROOT / "results/wp10/sweep/run1/a0_baseline_no_wp10.pos",
    "run2": ROOT / "results/wp10/sweep/run2/b0_baseline_no_wp10.pos",
    "run3": ROOT / "results/wp10/sweep/run3/b0_baseline_no_wp10.pos",
}

K_VALUES = (5, 10, 25, 50, 100, 200)


def _truth_err_m(ecef: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(ecef) - np.asarray(ref)))


def _segment_stats(errs: list[float]) -> dict[str, float | int]:
    if not errs:
        return {"n": 0}
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "rms_m": float(math.sqrt(np.mean(arr * arr))),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.quantile(arr, 0.9)),
        "lt50cm_pct": float(100.0 * np.mean(arr < 0.5)),
    }


def audit_run(run: str) -> dict[str, object]:
    data_root = resolve_data_root()
    run_dir = resolve_run_dir(data_root, f"tokyo/{run}")
    tows = parse_rover_tows_from_obs(run_dir / "rover.obs")
    rtk = load_rtk_pos_extended(BASELINE_POS[run])
    status_by_tow = {k: (v.ecef, v.status) for k, v in rtk.items()}
    anchor_class = classify_anchor_status(tows, status_by_tow)
    ref = load_reference_grid("tokyo", run, data_root=data_root)

    fix_idx = np.flatnonzero(anchor_class == 2)
    float_idx = np.flatnonzero(anchor_class == 1)
    n_rover = int(tows.size)
    phase2_start = 1000  # ~200 s @ 5 Hz

    fix_errs: list[float] = []
    float_errs: list[float] = []
    fix_errs_tail: list[float] = []
    for i in np.concatenate([fix_idx, float_idx]):
        tow = round(float(tows[i]), 1)
        hit = ref.get(tow)
        if hit is None:
            continue
        err = _truth_err_m(rtk[tow].ecef, hit)
        if anchor_class[i] == 2:
            fix_errs.append(err)
            if i >= phase2_start:
                fix_errs_tail.append(err)
        else:
            float_errs.append(err)

    # Temporal FIX distribution (100-epoch bins).
    bins: list[dict[str, int | float]] = []
    for start in range(0, n_rover, 100):
        end = min(start + 100, n_rover)
        sl = anchor_class[start:end]
        bins.append(
            {
                "epoch_start": start,
                "epoch_end": end - 1,
                "n_fix": int(np.count_nonzero(sl == 2)),
                "n_float": int(np.count_nonzero(sl == 1)),
            }
        )

    # Anchor desert: fraction of rover epochs with anchor within k.
    coverage: dict[str, float] = {}
    for k in K_VALUES:
        dist_fix = nearest_anchor_distance_epochs(anchor_class, include_fix=True, include_float=False)
        dist_dense = nearest_anchor_distance_epochs(
            anchor_class, include_fix=True, include_float=True
        )
        coverage[f"fix_within_{k}ep_pct"] = float(
            100.0 * np.mean(np.isfinite(dist_fix) & (dist_fix <= k))
        )
        coverage[f"fix_float_within_{k}ep_pct"] = float(
            100.0 * np.mean(np.isfinite(dist_dense) & (dist_dense <= k))
        )

    # Calibrated FLOAT sigma sample (median truth error).
    float_sigmas = [
        anchor_sigma_m(rtk[round(float(tows[i]), 1)], 1, fix_sigma_m=0.15, float_sigma_m=3.0)
        for i in float_idx
        if round(float(tows[i]), 1) in rtk
    ]

    # <50cm candidate epochs not near any FIX (desert overlap).
    lt50_candidate = 0
    lt50_no_fix_near_50 = 0
    dist_fix = nearest_anchor_distance_epochs(anchor_class, include_fix=True, include_float=False)
    for i in range(n_rover):
        tow = round(float(tows[i]), 1)
        hit = ref.get(tow)
        if hit is None:
            continue
        err = _truth_err_m(rtk[tow].ecef, hit) if tow in rtk else float("inf")
        if err < 0.5:
            lt50_candidate += 1
            if not (np.isfinite(dist_fix[i]) and dist_fix[i] <= 50):
                lt50_no_fix_near_50 += 1

    return {
        "run": run,
        "n_rover_epochs": n_rover,
        "n_fix_anchors": int(fix_idx.size),
        "n_float_anchors": int(float_idx.size),
        "fix_pct_of_rover": float(100.0 * fix_idx.size / n_rover),
        "float_pct_of_rover": float(100.0 * float_idx.size / n_rover),
        "fix_after_ep1000": int(np.count_nonzero(fix_idx >= phase2_start)),
        "fix_after_ep1000_pct_of_fix": float(
            100.0 * np.count_nonzero(fix_idx >= phase2_start) / max(1, fix_idx.size)
        ),
        "fix_truth": _segment_stats(fix_errs),
        "fix_truth_tail_ep1000plus": _segment_stats(fix_errs_tail),
        "float_truth": _segment_stats(float_errs),
        "float_sigma_median": float(np.median(float_sigmas)) if float_sigmas else None,
        "temporal_bins_100ep": bins,
        "anchor_coverage_pct": coverage,
        "lt50_rtk_candidate_epochs": lt50_candidate,
        "lt50_rtk_no_fix_within_50ep": lt50_no_fix_near_50,
    }


def main() -> int:
    out_dir = ROOT / "results/wp12e"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"runs": [audit_run(r) for r in ("run1", "run2", "run3")]}
    path = out_dir / "anchor_audit.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in summary["runs"]:
        print(
            f"{row['run']}: FIX={row['n_fix_anchors']} FLOAT={row['n_float_anchors']} "
            f"fix_tail={row['fix_after_ep1000']} "
            f"fix_RMS={row['fix_truth'].get('rms_m')} float_RMS={row['float_truth'].get('rms_m')} "
            f"within50ep(dense)={row['anchor_coverage_pct']['fix_float_within_50ep_pct']:.1f}%",
            flush=True,
        )
    print(f"Wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
