#!/usr/bin/env python3
"""UTD edge-diffraction candidate features for one PPC run.

This is the first implementation step toward the UTD direction.  It
does not compute full wedge diffraction coefficients.  Instead it
extracts PLATEAU building edges that can act as diffraction candidates
and scores how close those edges are to each receiver-to-satellite ray.
The resulting features are deployable and can be merged into the
existing §7.16 window stack to test whether UTD-style edge information
adds signal beyond LoS, reflection, and antenna proxies.

Outputs under `experiments/results/`:

- `ppc_utd_edges_s<stride>_<city>_<run>_per_epoch.csv`
- `ppc_utd_edges_s<stride>_<city>_<run>_per_window.csv`
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
from utd_edge_features import epoch_utd_summary, extract_diffraction_edges


RESULTS_DIR = _SCRIPT_DIR / "results"


def _parse_systems(spec: str) -> tuple[str, ...]:
    return tuple(part.strip().upper() for part in spec.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UTD edge-diffraction candidate features")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESET_URLS), default="tokyo23")
    parser.add_argument("--systems", default="G,R,J,E")
    parser.add_argument("--plateau-zone", type=int, default=9)
    parser.add_argument("--mesh-radius", type=int, default=1)
    parser.add_argument("--epoch-stride", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=0, help="0 = all")
    parser.add_argument("--subset-root", type=Path, default=Path("/tmp/plateau_segment_cache"))
    parser.add_argument("--window-duration-s", type=float, default=60.0)
    parser.add_argument("--route-margin-m", type=float, default=300.0)
    parser.add_argument("--edge-quantization-m", type=float, default=0.05)
    parser.add_argument("--min-edge-length-m", type=float, default=2.0)
    parser.add_argument("--min-dihedral-deg", type=float, default=20.0)
    parser.add_argument("--exclude-boundary-edges", action="store_true")
    parser.add_argument("--edge-voxel-size-m", type=float, default=2.0)
    parser.add_argument("--max-candidate-edges", type=int, default=200_000)
    parser.add_argument("--max-edge-range-m", type=float, default=250.0)
    parser.add_argument("--max-ray-edge-distance-m", type=float, default=25.0)
    parser.add_argument("--max-excess-path-m", type=float, default=80.0)
    parser.add_argument("--wavelength-m", type=float, default=0.1902936728)
    parser.add_argument("--score-excess-scale-m", type=float, default=20.0)
    parser.add_argument("--score-distance-scale-m", type=float, default=10.0)
    parser.add_argument("--results-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = _parse_systems(args.systems)
    city = args.run_dir.parent.name
    run = args.run_dir.name
    prefix = args.results_prefix or f"ppc_utd_edges_s{args.epoch_stride}_{city}_{run}"

    print(f"loading real data for {city}/{run}...", flush=True)
    t0 = time.monotonic()
    data = load_real_data(args.run_dir, max_epochs=(args.max_epochs or None), systems=systems)
    print(f"  loaded in {time.monotonic() - t0:.1f}s; epochs={len(data['times'])}", flush=True)

    ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)
    mesh_codes = derive_mesh_codes(ground_truth)
    expanded = expand_meshes(sorted(set(mesh_codes)), args.mesh_radius)
    print(f"  meshes: {len(set(mesh_codes))} unique, {len(expanded)} expanded", flush=True)

    print("loading PLATEAU + BVH...", flush=True)
    t0 = time.monotonic()
    subset_dir = ensure_subset(PRESET_URLS[args.preset], expanded, args.subset_root)
    model = load_plateau(subset_dir, zone=args.plateau_zone)
    from gnss_gpu.bvh import BVHAccelerator

    accelerator = BVHAccelerator.from_building_model(model)
    print(f"  BVH built in {time.monotonic() - t0:.1f}s ({accelerator.n_triangles} triangles)", flush=True)

    print("extracting candidate diffraction edges...", flush=True)
    t0 = time.monotonic()
    edges = extract_diffraction_edges(
        model.triangles,
        route_ecef=ground_truth,
        route_margin_m=args.route_margin_m,
        quantization_m=args.edge_quantization_m,
        min_edge_length_m=args.min_edge_length_m,
        min_dihedral_deg=args.min_dihedral_deg,
        include_boundary_edges=not args.exclude_boundary_edges,
        voxel_size_m=args.edge_voxel_size_m,
        max_edges=args.max_candidate_edges,
    )
    boundary_count = int(np.count_nonzero(edges.is_boundary))
    print(
        "  edges="
        f"{edges.size} boundary={boundary_count} welded={edges.size - boundary_count} "
        f"in {time.monotonic() - t0:.1f}s",
        flush=True,
    )

    epoch_rows: list[dict[str, object]] = []
    n_epochs = len(data["times"])
    candidate_kwargs = {
        "max_edge_range_m": args.max_edge_range_m,
        "max_ray_edge_distance_m": args.max_ray_edge_distance_m,
        "max_excess_path_m": args.max_excess_path_m,
        "wavelength_m": args.wavelength_m,
        "score_excess_scale_m": args.score_excess_scale_m,
        "score_distance_scale_m": args.score_distance_scale_m,
    }

    print("per-epoch UTD edge candidate scoring...", flush=True)
    t0 = time.monotonic()
    for i in range(0, n_epochs, args.epoch_stride):
        sat_ecef = np.asarray(data["sat_ecef"][i], dtype=np.float64)
        if sat_ecef.size == 0:
            continue
        pseudoranges = np.asarray(data["pseudoranges"][i], dtype=np.float64)
        weights = np.asarray(data["weights"][i], dtype=np.float64)
        truth_state = _reference_state_at_truth(sat_ecef, pseudoranges, weights, ground_truth[i])
        rx = truth_state[:3]
        is_los = np.asarray(accelerator.check_los(rx, sat_ecef), dtype=bool)
        summary = epoch_utd_summary(rx, sat_ecef, is_los, edges, **candidate_kwargs)
        summary.update({
            "city": city,
            "run": run,
            "epoch": i,
            "gps_tow": float(data["times"][i]),
            "utd_edge_count_total": edges.size,
            "utd_edge_boundary_count": boundary_count,
        })
        epoch_rows.append(summary)
        if i > 0 and (i // max(args.epoch_stride, 1)) % 250 == 0:
            elapsed = time.monotonic() - t0
            done = i / max(args.epoch_stride, 1)
            rate = done / max(elapsed, 1e-3)
            remaining = (n_epochs / max(args.epoch_stride, 1) - done) / max(rate, 1e-3)
            print(f"  epoch {i}/{n_epochs}  rate={rate:.1f}/s  eta={remaining:.0f}s", flush=True)
    print(f"  done in {time.monotonic() - t0:.1f}s", flush=True)

    epoch_df = pd.DataFrame(epoch_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_path = RESULTS_DIR / f"{prefix}_per_epoch.csv"
    epoch_df.to_csv(epoch_path, index=False)
    print(f"saved: {epoch_path} ({len(epoch_df)} rows)")

    if epoch_df.empty:
        return

    t_start = epoch_df["gps_tow"].min()
    epoch_df["window_index"] = ((epoch_df["gps_tow"] - t_start) // args.window_duration_s).astype(int)
    win = epoch_df.groupby("window_index").agg(
        epoch_count=("gps_tow", "size"),
        sat_count_mean=("sat_count", "mean"),
        nlos_count_mean=("nlos_count", "mean"),
        utd_candidate_sat_count_mean=("utd_candidate_sat_count", "mean"),
        utd_candidate_sat_count_max=("utd_candidate_sat_count", "max"),
        utd_candidate_nlos_sat_count_mean=("utd_candidate_nlos_sat_count", "mean"),
        utd_candidate_nlos_sat_count_max=("utd_candidate_nlos_sat_count", "max"),
        utd_candidate_count_total_mean=("utd_candidate_count_total", "mean"),
        utd_candidate_count_total_max=("utd_candidate_count_total", "max"),
        utd_candidate_count_nlos_mean=("utd_candidate_count_nlos", "mean"),
        utd_candidate_count_nlos_max=("utd_candidate_count_nlos", "max"),
        utd_min_excess_path_m_mean=("utd_min_excess_path_m", "mean"),
        utd_min_excess_path_m_min=("utd_min_excess_path_m", "min"),
        utd_min_edge_distance_m_mean=("utd_min_edge_distance_m", "mean"),
        utd_min_edge_distance_m_min=("utd_min_edge_distance_m", "min"),
        utd_min_fresnel_v_mean=("utd_min_fresnel_v", "mean"),
        utd_min_fresnel_v_min=("utd_min_fresnel_v", "min"),
        utd_score_sum_mean=("utd_score_sum", "mean"),
        utd_score_sum_max=("utd_score_sum", "max"),
        utd_score_nlos_sum_mean=("utd_score_nlos_sum", "mean"),
        utd_score_nlos_sum_max=("utd_score_nlos_sum", "max"),
        utd_edge_count_total=("utd_edge_count_total", "max"),
        utd_edge_boundary_count=("utd_edge_boundary_count", "max"),
    ).reset_index()
    win.insert(0, "run", run)
    win.insert(0, "city", city)
    win_path = RESULTS_DIR / f"{prefix}_per_window.csv"
    win.to_csv(win_path, index=False)
    print(f"saved: {win_path} ({len(win)} rows)")

    print("\nepoch-level UTD candidate summary:")
    cols = [
        "utd_candidate_sat_count",
        "utd_candidate_count_total",
        "utd_candidate_count_nlos",
        "utd_score_sum",
        "utd_score_nlos_sum",
    ]
    print(epoch_df[cols].describe().round(3).to_string())


if __name__ == "__main__":
    main()
