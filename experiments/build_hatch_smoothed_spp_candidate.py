#!/usr/bin/env python3
"""A-lite Doppler smoothing: trusted-arc Hatch filter + SPP candidate generator.

Within continuous carrier-phase arcs (no cycle slips), smooths pseudorange
using the Hatch filter:

    P_smooth[k] = (1/n) * P[k] + ((n-1)/n) * (P_smooth[k-1] + (L[k] - L[k-1]))

Then computes per-epoch SPP positions with robust IRLS on the smoothed PR.

Output is candidate-dir compatible with PPC selector pool:
    {output_dir}/{city}_{run}_full.pos
    {output_dir}/{city}_{run}_full.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
sys.path.insert(0, str(REPO / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.robust_spp import robust_spp  # noqa: E402

C_LIGHT = 299_792_458.0

# L1-band wavelengths (meters)
WAVELENGTH = {
    "G": 0.190293673,  # GPS L1 1575.42 MHz
    "E": 0.190293673,  # Galileo E1
    "J": 0.190293673,  # QZSS L1
    "C": 0.192039486,  # BDS B1I 1561.098 MHz
    "R": 0.187094,     # GLONASS G1 approx (FDMA, ±1% per sat)
}


def detect_trusted_arcs(
    cp_per_epoch: list[np.ndarray],
    pr_per_epoch: list[np.ndarray],
    sat_ids_per_epoch: list[list[str]],
    wavelengths: dict[str, float],
    min_arc_epochs: int,
    max_resid_jump_m: float,
    max_gap_epochs: int,
):
    """Build per-sat trusted-arc list and per-sat-per-epoch (PR, CP_m) cache.

    A cycle slip is declared when the PR-CP residual (after subtracting per-epoch
    median across sats to absorb receiver clock drift) jumps by > max_resid_jump_m
    between consecutive epochs of the same sat.
    """
    n_epochs = len(pr_per_epoch)
    sat_obs: dict[str, dict[int, tuple[float, float]]] = {}
    # per-epoch median(PR - CP_m) for receiver clock drift absorption
    median_resid: list[float] = [float("nan")] * n_epochs
    for ep_idx in range(n_epochs):
        cp_arr = cp_per_epoch[ep_idx]
        pr_arr = pr_per_epoch[ep_idx]
        sat_ids = sat_ids_per_epoch[ep_idx]
        residuals = []
        for s_idx, sat_id in enumerate(sat_ids):
            cp = float(cp_arr[s_idx])
            pr = float(pr_arr[s_idx])
            if np.isnan(cp) or np.isnan(pr):
                continue
            sys_code = sat_id[0]
            wl = wavelengths.get(sys_code, 0.190)
            cp_m = cp * wl
            sat_obs.setdefault(sat_id, {})[ep_idx] = (pr, cp_m)
            residuals.append(pr - cp_m)
        if residuals:
            median_resid[ep_idx] = float(np.median(residuals))

    sat_arcs: dict[str, list[tuple[int, int]]] = {}
    for sat_id, obs_dict in sat_obs.items():
        sorted_eps = sorted(obs_dict.keys())
        arcs: list[tuple[int, int]] = []
        arc_start = None
        prev_ep = None
        prev_relative_resid = None
        for ep in sorted_eps:
            pr, cp_m = obs_dict[ep]
            mr = median_resid[ep]
            if np.isnan(mr):
                relative_resid = pr - cp_m
            else:
                relative_resid = (pr - cp_m) - mr
            slip = False
            if prev_ep is not None:
                if ep - prev_ep > max_gap_epochs:
                    slip = True
                elif prev_relative_resid is not None and abs(
                    relative_resid - prev_relative_resid
                ) > max_resid_jump_m:
                    slip = True
            if slip or arc_start is None:
                if arc_start is not None and prev_ep is not None:
                    if prev_ep - arc_start + 1 >= min_arc_epochs:
                        arcs.append((arc_start, prev_ep))
                arc_start = ep
            prev_ep = ep
            prev_relative_resid = relative_resid
        if arc_start is not None and prev_ep is not None:
            if prev_ep - arc_start + 1 >= min_arc_epochs:
                arcs.append((arc_start, prev_ep))
        if arcs:
            sat_arcs[sat_id] = arcs
    return sat_obs, sat_arcs


def hatch_smooth(
    sat_obs: dict[str, dict[int, tuple[float, float]]],
    sat_arcs: dict[str, list[tuple[int, int]]],
    smoothing_const_n: int,
):
    """Hatch filter per trusted arc. Returns dict[(sat_id, epoch_idx) -> smoothed_PR]."""
    smoothed: dict[tuple[str, int], float] = {}
    for sat_id, arcs in sat_arcs.items():
        obs = sat_obs[sat_id]
        for arc_start, arc_end in arcs:
            P_prev = None
            L_prev = None
            n_in_arc = 0
            for ep in range(arc_start, arc_end + 1):
                if ep not in obs:
                    # data hole inside detected arc — break safely
                    P_prev = None
                    L_prev = None
                    n_in_arc = 0
                    continue
                pr, cp_m = obs[ep]
                n_in_arc += 1
                n_eff = min(n_in_arc, smoothing_const_n)
                if P_prev is None or L_prev is None:
                    P_smooth = pr
                else:
                    P_smooth = (1.0 / n_eff) * pr + (
                        (n_eff - 1) / n_eff
                    ) * (P_prev + (cp_m - L_prev))
                smoothed[(sat_id, ep)] = P_smooth
                P_prev = P_smooth
                L_prev = cp_m
    return smoothed


def ecef_to_llh(x: float, y: float, z: float):
    """WGS84 ECEF → geodetic."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1 - f)
    e2 = 2 * f - f * f
    ep2 = (a * a - b * b) / (b * b)
    p = np.sqrt(x * x + y * y)
    th = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return float(np.degrees(lat)), float(np.degrees(lon)), float(h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["tokyo", "nagoya"])
    parser.add_argument("--run", required=True, choices=["run1", "run2", "run3"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-arc-epochs", type=int, default=30)
    parser.add_argument("--max-resid-jump-m", type=float, default=0.5)
    parser.add_argument("--max-gap-epochs", type=int, default=3)
    parser.add_argument("--smoothing-n", type=int, default=100)
    parser.add_argument("--spp-irls-c", type=float, default=15.0)
    parser.add_argument(
        "--gps-week",
        type=int,
        default=2324,
        help="GPS week to write in the .pos header (PPC 2024 = 2324)",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="G,E,J",
        help="Comma-separated constellation codes (G/E/J/C/R)",
    )
    args = parser.parse_args()

    run_dir = args.data_root / args.city / args.run
    print(f"Loading {run_dir} ...", flush=True)
    loader = PPCDatasetLoader(run_dir)
    systems_tuple = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    data = loader.load_experiment_data(systems=systems_tuple)
    n_epochs = data["n_epochs"]
    print(f"  n_epochs: {n_epochs}", flush=True)

    print("Detecting trusted arcs ...", flush=True)
    sat_obs, sat_arcs = detect_trusted_arcs(
        data["carrier_phase"],
        data["pseudoranges"],
        data["used_prns"],
        WAVELENGTH,
        min_arc_epochs=args.min_arc_epochs,
        max_resid_jump_m=args.max_resid_jump_m,
        max_gap_epochs=args.max_gap_epochs,
    )
    total_arcs = sum(len(arcs) for arcs in sat_arcs.values())
    total_arc_epochs = sum(
        end - start + 1 for arcs in sat_arcs.values() for (start, end) in arcs
    )
    print(
        f"  sats with arcs: {len(sat_arcs)}, total arcs: {total_arcs}, "
        f"total arc-sat-epoch coverage: {total_arc_epochs}",
        flush=True,
    )

    print("Hatch smoothing ...", flush=True)
    smoothed_pr = hatch_smooth(sat_obs, sat_arcs, smoothing_const_n=args.smoothing_n)
    print(f"  smoothed measurements: {len(smoothed_pr)}", flush=True)

    print("Computing SPP positions on smoothed PR ...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pos_path = args.output_dir / f"{args.city}_{args.run}_full.pos"
    csv_path = args.output_dir / f"{args.city}_{args.run}_full.csv"

    times = data["times"]
    pos_lines: list[str] = []
    pos_lines.append("% LibGNSS++ Position Solution (A-lite Hatch SPP)\n")
    pos_lines.append(
        "% GPS_Week GPS_TOW X(m) Y(m) Z(m) Lat(deg) Lon(deg) Height(m) "
        "Status NumSat PDOP Ratio Baseline(m)\n"
    )

    csv_lines: list[str] = []
    csv_lines.append(
        "epoch_index,gps_week,tow,final_valid,final_status,final_sats,"
        "final_ratio,final_pdop,final_baseline_m,final_residual_rms,"
        "final_residual_abs_max,candidate_jump_m,drift_from_fixed_m,"
        "fixed_drift_guard_m,spp_valid,spp_sats,spp_pdop,esd_std_pe,"
        "esd_std_pn,esd_std_pu,output_added\n"
    )

    n_smoothed_used_total = 0
    n_valid = 0
    # seed initial position from reference origin to avoid the geometric-center init
    # which sits ~26 000 km off Earth and rarely converges in 10 IRLS iters.
    prev_xyz = np.asarray(data["origin_ecef"], dtype=np.float64).copy()
    for ep_idx in range(n_epochs):
        sat_ecef = data["sat_ecef"][ep_idx]
        pr_orig = data["pseudoranges"][ep_idx]
        weights = data["weights"][ep_idx]
        sat_ids = data["used_prns"][ep_idx]
        tow = float(times[ep_idx])

        if pr_orig.size == 0 or sat_ecef.size == 0:
            csv_lines.append(
                f"{ep_idx},{args.gps_week},{tow:.3f},0,2,0,0.00,0.00,0.0,"
                f"nan,nan,0.0,0.0,0.4,0,0,0.00,0.0,0.0,0.0,0\n"
            )
            continue

        pr_smoothed = np.array(pr_orig, dtype=np.float64, copy=True)
        n_smoothed_here = 0
        for s_idx, sat_id in enumerate(sat_ids):
            key = (sat_id, ep_idx)
            if key in smoothed_pr:
                pr_smoothed[s_idx] = smoothed_pr[key]
                n_smoothed_here += 1
        n_smoothed_used_total += n_smoothed_here

        mask = (~np.isnan(pr_smoothed)) & (~np.isnan(sat_ecef[:, 0]))
        if int(mask.sum()) < 5:
            csv_lines.append(
                f"{ep_idx},{args.gps_week},{tow:.3f},0,2,{int(mask.sum())},"
                f"0.00,0.00,0.0,nan,nan,0.0,0.0,0.4,0,0,0.00,0.0,0.0,0.0,0\n"
            )
            continue

        sat_used = sat_ecef[mask]
        pr_used = pr_smoothed[mask]
        w_used = weights[mask] if weights is not None and weights.size else None

        pos = robust_spp(
            sat_used,
            pr_used,
            weights=w_used,
            init_pos=prev_xyz,
            weight_func="cauchy",
            threshold=args.spp_irls_c,
        )
        if pos is None:
            csv_lines.append(
                f"{ep_idx},{args.gps_week},{tow:.3f},0,2,{int(mask.sum())},"
                f"0.00,0.00,0.0,nan,nan,0.0,0.0,0.4,0,0,0.00,0.0,0.0,0.0,0\n"
            )
            continue

        ranges = np.linalg.norm(sat_used - pos[None, :], axis=1)
        cb = float(np.median(pr_used - ranges))
        resid = pr_used - ranges - cb
        rms = float(np.sqrt(np.mean(resid ** 2)))
        abs_max = float(np.max(np.abs(resid)))
        n_sat = int(mask.sum())
        # rough PDOP via H^T H trace (geometric only)
        los = (sat_used - pos[None, :]) / ranges[:, None]
        H = np.column_stack([-los, np.ones(n_sat)])
        try:
            cov = np.linalg.inv(H.T @ H)
            pdop = float(np.sqrt(cov[0, 0] + cov[1, 1] + cov[2, 2]))
        except np.linalg.LinAlgError:
            pdop = 99.0

        lat_deg, lon_deg, h_m = ecef_to_llh(pos[0], pos[1], pos[2])

        pos_lines.append(
            f"{args.gps_week} {tow:.3f} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} "
            f"{lat_deg:.9f} {lon_deg:.9f} {h_m:.4f} 2 {n_sat} {pdop:.2f} 0.0 0.0\n"
        )
        csv_lines.append(
            f"{ep_idx},{args.gps_week},{tow:.3f},1,2,{n_sat},0.00,{pdop:.2f},0.0,"
            f"{rms:.4f},{abs_max:.4f},0.0,0.0,0.4,1,{n_sat},{pdop:.2f},"
            f"0.0,0.0,0.0,1\n"
        )
        prev_xyz = pos
        n_valid += 1

    with open(pos_path, "w") as f:
        f.writelines(pos_lines)
    with open(csv_path, "w") as f:
        f.writelines(csv_lines)
    print(f"Wrote {pos_path} (valid epochs: {n_valid}/{n_epochs})", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(
        f"Smoothed PR substitutions: {n_smoothed_used_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
