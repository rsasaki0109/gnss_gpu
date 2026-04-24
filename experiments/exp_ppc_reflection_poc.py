#!/usr/bin/env python3
"""Proof-of-concept: enable BVH first-order multipath reflection in the PPC
ray-tracing simulator and record per-satellite reflection-presence + excess
delay at every epoch of one run.  Aggregate to per-epoch and per-window
diagnostics, and check whether reflection density correlates with demo5
FIX failures.

This is a focused experimental script, scoped to one (city, run) at a
time.  Output columns are minimal — the goal is to test the hypothesis
that adding reflection-aware features helps explain §7.16 failures
(Tokyo run2 false-high w7/w9 + hidden-high w23-w27).

Outputs under `experiments/results/`:

- `ppc_reflection_poc_<city>_<run>_per_sat.csv`:
  city, run, epoch, gps_tow, sat_id, los_gt, reflection_present,
  excess_delay_m
- `ppc_reflection_poc_<city>_<run>_per_epoch.csv`:
  city, run, epoch, gps_tow, sat_count, los_count, nlos_count,
  reflection_count, excess_delay_m_max, excess_delay_m_p90
- `ppc_reflection_poc_<city>_<run>_per_window.csv`:
  city, run, window_index, epoch_count, reflection_count_mean,
  reflection_count_max, excess_delay_m_max_mean, excess_delay_m_max_max
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_pf3d_residual_analysis import _reference_state_at_truth
from exp_urbannav_baseline import load_real_data
from fetch_plateau_subset import PRESET_URLS, expand_meshes
from gnss_gpu.io.plateau import load_plateau
from scan_ppc_plateau_segments import derive_mesh_codes, ensure_subset


RESULTS_DIR = _SCRIPT_DIR / "results"


def _parse_systems(spec: str) -> tuple[str, ...]:
    return tuple(part.strip().upper() for part in spec.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reflection-aware sim PoC for one PPC run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESET_URLS), default="tokyo23")
    parser.add_argument("--systems", default="G,R,J,E")
    parser.add_argument("--plateau-zone", type=int, default=9)
    parser.add_argument("--mesh-radius", type=int, default=1)
    parser.add_argument("--epoch-stride", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=0, help="0 = all")
    parser.add_argument("--subset-root", type=Path, default=Path("/tmp/plateau_segment_cache"))
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    parser.add_argument("--results-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = _parse_systems(args.systems)
    city = args.run_dir.parent.name
    run = args.run_dir.name
    prefix = args.results_prefix or f"ppc_reflection_poc_{city}_{run}"
    print(f"loading real data for {city}/{run}...", flush=True)
    t0 = time.monotonic()
    data = load_real_data(args.run_dir, max_epochs=(args.max_epochs or None), systems=systems)
    print(f"  loaded in {time.monotonic() - t0:.1f}s; epochs={len(data['times'])}, sats per epoch ~{len(data['sat_ecef'][0])}", flush=True)

    ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)
    if ground_truth.ndim != 2 or ground_truth.shape[1] != 3:
        raise SystemExit(f"unexpected ground_truth shape: {ground_truth.shape}")

    print("deriving mesh codes...", flush=True)
    mesh_codes = derive_mesh_codes(ground_truth)
    unique_meshes = sorted(set(mesh_codes))
    expanded = expand_meshes(unique_meshes, args.mesh_radius)
    print(f"  {len(unique_meshes)} unique meshes, expanded to {len(expanded)}", flush=True)

    zip_url = PRESET_URLS[args.preset]
    print("fetching / loading PLATEAU subset...", flush=True)
    t0 = time.monotonic()
    subset_dir = ensure_subset(zip_url, expanded, args.subset_root)
    model = load_plateau(subset_dir, zone=args.plateau_zone)
    accelerator = model  # BuildingModel directly: BVH multipath binding is not compiled.
    print(f"  BuildingModel loaded in {time.monotonic() - t0:.1f}s ({len(model.triangles)} triangles, brute-force multipath)", flush=True)

    epoch_rows: list[dict[str, object]] = []
    per_sat_rows: list[dict[str, object]] = []
    n_epochs = len(data["times"])

    print("per-epoch ray tracing (LoS + multipath reflection)...", flush=True)
    t0 = time.monotonic()
    for i in range(0, n_epochs, args.epoch_stride):
        sat_ecef = np.asarray(data["sat_ecef"][i], dtype=np.float64)
        if sat_ecef.size == 0:
            continue
        pseudoranges = np.asarray(data["pseudoranges"][i], dtype=np.float64)
        weights = np.asarray(data["weights"][i], dtype=np.float64)
        sat_ids = list(data.get("used_prns", [])[i])
        truth_state = _reference_state_at_truth(sat_ecef, pseudoranges, weights, ground_truth[i])
        rx = truth_state[:3]
        los = np.asarray(accelerator.check_los(rx, sat_ecef), dtype=bool)
        excess_delays, _refl_pts = accelerator.compute_multipath(rx, sat_ecef)
        excess_delays = np.asarray(excess_delays, dtype=np.float64)
        reflection_present = excess_delays > 1.0  # > 1 m excess delay counts as a reflection
        sat_count = int(sat_ecef.shape[0])
        los_count = int(los.sum())
        nlos_count = sat_count - los_count
        refl_count = int(reflection_present.sum())
        ed_max = float(excess_delays.max()) if excess_delays.size else 0.0
        ed_p90 = float(np.percentile(excess_delays, 90)) if excess_delays.size else 0.0
        epoch_rows.append({
            "city": city, "run": run, "epoch": i,
            "gps_tow": float(data["times"][i]),
            "sat_count": sat_count, "los_count": los_count, "nlos_count": nlos_count,
            "reflection_count": refl_count,
            "excess_delay_m_max": ed_max, "excess_delay_m_p90": ed_p90,
        })
        for s, sat_id in enumerate(sat_ids):
            per_sat_rows.append({
                "city": city, "run": run, "epoch": i,
                "gps_tow": float(data["times"][i]),
                "sat_id": sat_id,
                "los_gt": bool(los[s]),
                "reflection_present": bool(reflection_present[s]),
                "excess_delay_m": float(excess_delays[s]),
            })
        if (i // args.epoch_stride) % 1000 == 0 and i > 0:
            elapsed = time.monotonic() - t0
            rate = (i / args.epoch_stride) / elapsed
            remaining = (n_epochs / args.epoch_stride - i / args.epoch_stride) / max(rate, 1e-3)
            print(f"  epoch {i}/{n_epochs}  rate={rate:.0f}/s  eta={remaining:.0f}s", flush=True)
    print(f"  done in {time.monotonic() - t0:.1f}s", flush=True)

    epoch_df = pd.DataFrame(epoch_rows)
    per_sat_df = pd.DataFrame(per_sat_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_path = RESULTS_DIR / f"{prefix}_per_epoch.csv"
    per_sat_path = RESULTS_DIR / f"{prefix}_per_sat.csv"
    epoch_df.to_csv(epoch_path, index=False)
    per_sat_df.to_csv(per_sat_path, index=False)
    print(f"saved: {epoch_path} ({len(epoch_df)} rows)")
    print(f"saved: {per_sat_path} ({len(per_sat_df)} rows)")

    # Per-window aggregation
    if not epoch_df.empty:
        t_run_start = epoch_df["gps_tow"].min()
        epoch_df["window_index"] = ((epoch_df["gps_tow"] - t_run_start) // args.window_duration_s).astype(int)
        win = epoch_df.groupby("window_index").agg(
            epoch_count=("gps_tow", "size"),
            reflection_count_mean=("reflection_count", "mean"),
            reflection_count_max=("reflection_count", "max"),
            excess_delay_m_max_mean=("excess_delay_m_max", "mean"),
            excess_delay_m_max_max=("excess_delay_m_max", "max"),
            excess_delay_m_p90_mean=("excess_delay_m_p90", "mean"),
            nlos_count_mean=("nlos_count", "mean"),
            sat_count_mean=("sat_count", "mean"),
        ).reset_index()
        win.insert(0, "run", run)
        win.insert(0, "city", city)
        win_path = RESULTS_DIR / f"{prefix}_per_window.csv"
        win.to_csv(win_path, index=False)
        print(f"saved: {win_path} ({len(win)} rows)")

    print("\nepoch-level summary:")
    print(epoch_df.describe()[["sat_count", "los_count", "reflection_count", "excess_delay_m_max"]].to_string())


if __name__ == "__main__":
    main()
