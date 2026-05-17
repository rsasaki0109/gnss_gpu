#!/usr/bin/env python3
"""A-lite Doppler smoothing + atmosphere-corrected MultiGNSS SPP candidate.

v2 additions over v1:
- Klobuchar ionosphere correction (Klobuchar coefs from base.nav header)
- Saastamoinen tropospheric correction
- Sagnac/Earth-rotation correction
- MultiGNSSSolver with inter-system bias estimation
- Elevation-mask weighting

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
from gnss_gpu.io.nav_rinex import read_gps_klobuchar_from_nav_header  # noqa: E402
from gnss_gpu.multi_gnss import MultiGNSSSolver  # noqa: E402
from gnss_gpu.spp import correct_pseudoranges  # noqa: E402

# L1-band wavelengths (meters) — same as v1
WAVELENGTH = {
    "G": 0.190293673,
    "E": 0.190293673,
    "J": 0.190293673,
    "C": 0.192039486,
    "R": 0.187094,
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
    n_epochs = len(pr_per_epoch)
    sat_obs: dict[str, dict[int, tuple[float, float]]] = {}
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
            relative_resid = (pr - cp_m) - mr if not np.isnan(mr) else (pr - cp_m)
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


def hatch_smooth(sat_obs, sat_arcs, smoothing_const_n: int):
    smoothed: dict[tuple[str, int], float] = {}
    for sat_id, arcs in sat_arcs.items():
        obs = sat_obs[sat_id]
        for arc_start, arc_end in arcs:
            P_prev = None
            L_prev = None
            n_in_arc = 0
            for ep in range(arc_start, arc_end + 1):
                if ep not in obs:
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
    parser.add_argument("--gps-week", type=int, default=2324)
    parser.add_argument("--systems", type=str, default="G,E,J")
    parser.add_argument("--el-mask-deg", type=float, default=10.0)
    parser.add_argument(
        "--skip-hatch",
        action="store_true",
        help="Diagnostic: bypass Hatch smoothing (atmosphere-only SPP)",
    )
    parser.add_argument(
        "--smoothed-sats-only",
        action="store_true",
        help="Use only sats with smoothed PR (= continuous CP = LOS proxy).",
    )
    parser.add_argument(
        "--min-smoothed-sats",
        type=int,
        default=5,
        help="Fallback to all sats if fewer than this many smoothed sats are available.",
    )
    args = parser.parse_args()

    run_dir = args.data_root / args.city / args.run
    print(f"Loading {run_dir} ...", flush=True)
    loader = PPCDatasetLoader(run_dir)
    systems_tuple = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    data = loader.load_experiment_data(systems=systems_tuple)
    n_epochs = data["n_epochs"]
    print(f"  n_epochs: {n_epochs}", flush=True)

    iono_alpha, iono_beta = read_gps_klobuchar_from_nav_header(run_dir / "base.nav")
    iono_alpha = list(iono_alpha) if iono_alpha else None
    iono_beta = list(iono_beta) if iono_beta else None
    print(
        f"  Klobuchar: alpha={iono_alpha is not None}, beta={iono_beta is not None}",
        flush=True,
    )

    # System list for MultiGNSSSolver
    system_id_map = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4}
    enabled_systems = sorted({system_id_map[s] for s in systems_tuple if s in system_id_map})
    solver = MultiGNSSSolver(systems=enabled_systems)
    print(f"  MultiGNSSSolver systems: {enabled_systems}", flush=True)

    smoothed_pr: dict[tuple[str, int], float] = {}
    if not args.skip_hatch:
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
            f"sat-epoch coverage: {total_arc_epochs}",
            flush=True,
        )
        print("Hatch smoothing ...", flush=True)
        smoothed_pr = hatch_smooth(sat_obs, sat_arcs, smoothing_const_n=args.smoothing_n)
        print(f"  smoothed measurements: {len(smoothed_pr)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pos_path = args.output_dir / f"{args.city}_{args.run}_full.pos"
    csv_path = args.output_dir / f"{args.city}_{args.run}_full.csv"

    times = data["times"]
    el_mask_rad = float(np.radians(args.el_mask_deg))

    pos_lines: list[str] = [
        "% LibGNSS++ Position Solution (A-lite Hatch + atmo-corrected MultiGNSS SPP)\n",
        "% GPS_Week GPS_TOW X(m) Y(m) Z(m) Lat(deg) Lon(deg) Height(m) "
        "Status NumSat PDOP Ratio Baseline(m)\n",
    ]
    csv_lines: list[str] = [
        "epoch_index,gps_week,tow,final_valid,final_status,final_sats,"
        "final_ratio,final_pdop,final_baseline_m,final_residual_rms,"
        "final_residual_abs_max,candidate_jump_m,drift_from_fixed_m,"
        "fixed_drift_guard_m,spp_valid,spp_sats,spp_pdop,esd_std_pe,"
        "esd_std_pn,esd_std_pu,output_added\n"
    ]

    n_valid = 0
    n_smoothed_used_total = 0
    prev_xyz = np.asarray(data["origin_ecef"], dtype=np.float64).copy()
    for ep_idx in range(n_epochs):
        sat_ecef = data["sat_ecef"][ep_idx]
        pr_orig = data["pseudoranges"][ep_idx]
        snr_weights = data["weights"][ep_idx]
        sat_ids = data["used_prns"][ep_idx]
        system_ids = data["system_ids"][ep_idx]
        tow = float(times[ep_idx])

        if pr_orig.size == 0:
            csv_lines.append(
                f"{ep_idx},{args.gps_week},{tow:.3f},0,2,0,0.00,0.00,0.0,"
                f"nan,nan,0.0,0.0,0.4,0,0,0.00,0.0,0.0,0.0,0\n"
            )
            continue

        # apply Hatch substitutions, optionally restrict to smoothed sats only
        pr_smoothed = np.array(pr_orig, dtype=np.float64, copy=True)
        is_smoothed_mask = np.zeros(len(sat_ids), dtype=bool)
        n_smoothed_here = 0
        for s_idx, sat_id in enumerate(sat_ids):
            key = (sat_id, ep_idx)
            if key in smoothed_pr:
                pr_smoothed[s_idx] = smoothed_pr[key]
                is_smoothed_mask[s_idx] = True
                n_smoothed_here += 1
        n_smoothed_used_total += n_smoothed_here

        mask = (~np.isnan(pr_smoothed)) & (~np.isnan(sat_ecef[:, 0]))
        if args.smoothed_sats_only and is_smoothed_mask.sum() >= args.min_smoothed_sats:
            mask = mask & is_smoothed_mask
        if int(mask.sum()) < 5:
            csv_lines.append(
                f"{ep_idx},{args.gps_week},{tow:.3f},0,2,{int(mask.sum())},"
                f"0.00,0.00,0.0,nan,nan,0.0,0.0,0.4,0,0,0.00,0.0,0.0,0.0,0\n"
            )
            continue

        sat_used = sat_ecef[mask]
        pr_used = pr_smoothed[mask]
        sysid_used = system_ids[mask].astype(np.int32, copy=False)
        snr_w = snr_weights[mask]

        # atmosphere + Sagnac correction
        corrected_pr, elev_w = correct_pseudoranges(
            sat_used,
            pr_used,
            receiver_approx=prev_xyz,
            gps_time=tow,
            iono_alpha=iono_alpha,
            iono_beta=iono_beta,
            el_mask_rad=el_mask_rad,
        )
        # combine SNR weight × elevation weight
        combined_w = np.maximum(snr_w * elev_w, 1e-6)
        # drop completely zero-weight sats
        keep = combined_w > 0.01
        if int(keep.sum()) < 5:
            csv_lines.append(
                f"{ep_idx},{args.gps_week},{tow:.3f},0,2,{int(keep.sum())},"
                f"0.00,0.00,0.0,nan,nan,0.0,0.0,0.4,0,0,0.00,0.0,0.0,0.0,0\n"
            )
            continue

        try:
            pos, clk_biases, n_iter = solver.solve(
                sat_used[keep],
                corrected_pr[keep],
                sysid_used[keep],
                combined_w[keep],
            )
        except Exception:
            csv_lines.append(
                f"{ep_idx},{args.gps_week},{tow:.3f},0,2,{int(keep.sum())},"
                f"0.00,0.00,0.0,nan,nan,0.0,0.0,0.4,0,0,0.00,0.0,0.0,0.0,0\n"
            )
            continue

        if n_iter <= 0 or not np.all(np.isfinite(pos)) or np.all(pos == 0.0):
            csv_lines.append(
                f"{ep_idx},{args.gps_week},{tow:.3f},0,2,{int(keep.sum())},"
                f"0.00,0.00,0.0,nan,nan,0.0,0.0,0.4,0,0,0.00,0.0,0.0,0.0,0\n"
            )
            continue

        # compute residuals using each sat's clock bias
        sat_kept = sat_used[keep]
        cpr_kept = corrected_pr[keep]
        sys_kept = sysid_used[keep]
        ranges = np.linalg.norm(sat_kept - pos[None, :], axis=1)
        per_sat_cb = np.array([clk_biases.get(int(s), 0.0) for s in sys_kept])
        resid = cpr_kept - ranges - per_sat_cb
        rms = float(np.sqrt(np.mean(resid ** 2)))
        abs_max = float(np.max(np.abs(resid)))

        # PDOP from geometry matrix
        n_eff = int(keep.sum())
        los = (sat_kept - pos[None, :]) / np.maximum(ranges[:, None], 1.0)
        H = np.column_stack([-los, np.ones(n_eff)])
        try:
            cov = np.linalg.inv(H.T @ H)
            pdop = float(np.sqrt(cov[0, 0] + cov[1, 1] + cov[2, 2]))
        except np.linalg.LinAlgError:
            pdop = 99.0

        lat_deg, lon_deg, h_m = ecef_to_llh(pos[0], pos[1], pos[2])
        pos_lines.append(
            f"{args.gps_week} {tow:.3f} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} "
            f"{lat_deg:.9f} {lon_deg:.9f} {h_m:.4f} 2 {n_eff} {pdop:.2f} 0.0 0.0\n"
        )
        csv_lines.append(
            f"{ep_idx},{args.gps_week},{tow:.3f},1,2,{n_eff},0.00,{pdop:.2f},0.0,"
            f"{rms:.4f},{abs_max:.4f},0.0,0.0,0.4,1,{n_eff},{pdop:.2f},"
            f"0.0,0.0,0.0,1\n"
        )
        prev_xyz = pos
        n_valid += 1

    with open(pos_path, "w") as f:
        f.writelines(pos_lines)
    with open(csv_path, "w") as f:
        f.writelines(csv_lines)
    print(
        f"Wrote {pos_path} (valid: {n_valid}/{n_epochs}, "
        f"smoothed PR subst: {n_smoothed_used_total})",
        flush=True,
    )


if __name__ == "__main__":
    main()
