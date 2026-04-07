#!/usr/bin/env python3
"""Evaluate forward-backward particle smoothing on the gnssplusplus UrbanNav PF stack.

Pipeline per epoch (forward): predict (SPP or TDCP guide) → ``correct_clock_bias`` →
``update`` → optional ``position_update`` → ``store_epoch`` when smoothing is enabled.

After the forward pass, ``smooth()`` runs a backward PF and averages with forward
estimates. Metrics are computed on epochs aligned with UrbanNav ground-truth time tags
(same convention as ``exp_position_update_eval.py``).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (
    _PROJECT_ROOT / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "python",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate import compute_metrics, ecef_to_lla
from exp_urbannav_baseline import load_or_generate_data
from exp_urbannav_pf3d import PF_SIGMA_CB, PF_SIGMA_POS
from gnss_gpu import ParticleFilterDevice
from gnss_gpu.imu import ComplementaryHeadingFilter, load_imu_csv
from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics

RESULTS_DIR = _SCRIPT_DIR / "results"


def load_pf_smoother_dataset(run_dir: Path, rover_source: str = "trimble") -> dict[str, object]:
    """Load RINEX / UrbanNav ground-truth once for repeated PF runs (sweeps).

    Returns a dict with keys: ``epochs``, ``spp_lookup``, ``gt``, ``our_times``,
    ``first_pos``, ``init_cb``.  If ``imu.csv`` exists in *run_dir*, also
    includes ``imu_data`` (raw dict from :func:`load_imu_csv`).
    """
    from libgnsspp import preprocess_spp_file, solve_spp_file

    obs_path = str(run_dir / f"rover_{rover_source}.obs")
    nav_path = str(run_dir / "base.nav")

    epochs = preprocess_spp_file(obs_path, nav_path)
    sol = solve_spp_file(obs_path, nav_path)
    spp_records = [r for r in sol.records() if r.is_valid()]
    spp_lookup = {round(r.time.tow, 1): np.array(r.position_ecef_m) for r in spp_records}

    data = load_or_generate_data(run_dir, systems=("G", "E", "J"), urban_rover=rover_source)
    gt = data["ground_truth"]
    our_times = data["times"]

    first_pos = np.array(spp_records[0].position_ecef_m[:3], dtype=np.float64)
    init_meas = None
    for sol_epoch, measurements in epochs:
        if sol_epoch.is_valid() and len(measurements) >= 4:
            init_meas = measurements
            first_pos = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
            break
    if init_meas is None:
        raise RuntimeError(f"No valid epoch for init in {run_dir}")

    init_cb = float(
        np.median(
            [
                m.corrected_pseudorange
                - np.linalg.norm(np.asarray(m.satellite_ecef, dtype=np.float64) - first_pos)
                for m in init_meas
            ]
        )
    )
    result = {
        "epochs": epochs,
        "spp_lookup": spp_lookup,
        "gt": gt,
        "our_times": our_times,
        "first_pos": first_pos,
        "init_cb": init_cb,
    }

    # Load IMU data if available
    imu_path = run_dir / "imu.csv"
    if imu_path.exists():
        result["imu_data"] = load_imu_csv(imu_path)
        print(f"  [IMU] loaded {len(result['imu_data']['tow'])} samples from {imu_path}")
    else:
        result["imu_data"] = None

    return result


def _spp_heading_from_velocity(spp_vel_ecef: np.ndarray, lat: float, lon: float) -> float | None:
    """Compute heading (radians from north, clockwise) from SPP velocity in ECEF."""
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    # ECEF to ENU rotation
    ve = -sin_lon * spp_vel_ecef[0] + cos_lon * spp_vel_ecef[1]
    vn = (-sin_lat * cos_lon * spp_vel_ecef[0]
          - sin_lat * sin_lon * spp_vel_ecef[1]
          + cos_lat * spp_vel_ecef[2])
    speed = math.sqrt(ve ** 2 + vn ** 2)
    if speed < 0.5:  # too slow to get reliable heading
        return None
    return math.atan2(ve, vn)  # heading from north, clockwise


def run_pf_with_optional_smoother(
    run_dir: Path,
    run_name: str,
    *,
    n_particles: int,
    sigma_pos: float,
    sigma_pr: float,
    position_update_sigma: float | None,
    predict_guide: str,
    use_smoother: bool,
    rover_source: str = "trimble",
    max_epochs: int = 0,
    skip_valid_epochs: int = 0,
    sigma_pos_tdcp: float | None = None,
    sigma_pos_tdcp_tight: float | None = None,
    tdcp_tight_rms_max_m: float = 1.0e9,
    dataset: dict[str, object] | None = None,
    resampling: str = "megopolis",
    tdcp_elevation_weight: bool = False,
    tdcp_el_sin_floor: float = 0.1,
    tdcp_rms_threshold: float = 3.0,
    residual_downweight: bool = False,
    residual_threshold: float = 15.0,
    pr_accel_downweight: bool = False,
    pr_accel_threshold: float = 5.0,
    use_gmm: bool = False,
    gmm_w_los: float = 0.7,
    gmm_mu_nlos: float = 15.0,
    gmm_sigma_nlos: float = 30.0,
    doppler_position_update: bool = False,
    doppler_pu_sigma: float = 5.0,
    imu_tight_coupling: bool = False,
    tdcp_position_update: bool = False,
    tdcp_pu_sigma: float = 0.5,
    tdcp_pu_rms_max: float = 3.0,
    mupf: bool = False,
    mupf_sigma_cycles: float = 0.05,
    mupf_snr_min: float = 25.0,
    mupf_elev_min: float = 0.15,
    mupf_dd: bool = False,
    mupf_dd_sigma_cycles: float = 0.05,
) -> dict[str, object]:
    if dataset is None:
        ds = load_pf_smoother_dataset(run_dir, rover_source)
    else:
        ds = dataset

    epochs = ds["epochs"]
    spp_lookup = ds["spp_lookup"]
    gt = ds["gt"]
    our_times = ds["our_times"]
    first_pos = np.asarray(ds["first_pos"], dtype=np.float64)
    init_cb = float(ds["init_cb"])

    # --- IMU setup (for predict_guide in {"imu", "imu_spp_blend"}) ---
    imu_filter: ComplementaryHeadingFilter | None = None
    n_imu_used = 0
    n_imu_fallback = 0
    if predict_guide in ("imu", "imu_spp_blend"):
        imu_data = ds.get("imu_data")
        if imu_data is None:
            imu_path = run_dir / "imu.csv"
            if imu_path.exists():
                imu_data = load_imu_csv(imu_path)
            else:
                raise RuntimeError(
                    f"predict_guide={predict_guide} requires IMU data but "
                    f"imu.csv not found in {run_dir}"
                )
        imu_filter = ComplementaryHeadingFilter(imu_data, alpha=0.05)
        # Initialize heading from first two SPP positions
        tow_keys = sorted(spp_lookup.keys())
        if len(tow_keys) >= 2:
            p0 = spp_lookup[tow_keys[0]][:3]
            p1 = spp_lookup[tow_keys[1]][:3]
            lat0, lon0, _ = ecef_to_lla(float(p0[0]), float(p0[1]), float(p0[2]))
            dt_init = tow_keys[1] - tow_keys[0]
            if dt_init > 0:
                spp_vel_init = (p1 - p0) / dt_init
                h = _spp_heading_from_velocity(spp_vel_init, lat0, lon0)
                if h is not None:
                    imu_filter.heading = h

    # --- DD carrier phase setup ---
    dd_computer = None
    n_dd_used = 0
    n_dd_skip = 0
    if mupf_dd:
        from gnss_gpu.dd_carrier import DDCarrierComputer
        # Try common base station filenames
        base_obs_path = None
        for name in ("base_trimble.obs", "base.obs"):
            p = run_dir / name
            if p.exists():
                base_obs_path = p
                break
        if base_obs_path is not None:
            dd_computer = DDCarrierComputer(base_obs_path)
            print(f"  [DD] base_pos = {dd_computer.base_position}")
        else:
            print(f"  [DD] WARNING: no base station RINEX found in {run_dir}, DD disabled")

    pf = ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=sigma_pos,
        sigma_cb=PF_SIGMA_CB,
        sigma_pr=sigma_pr,
        resampling=resampling,
        seed=42,
    )
    pf.initialize(first_pos, clock_bias=init_cb, spread_pos=10.0, spread_cb=100.0)

    if use_smoother:
        pf.enable_smoothing()

    forward_aligned: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    aligned_indices: list[int] = []
    n_stored = 0
    n_tdcp_used = 0
    n_tdcp_fallback = 0
    n_imu_tight_used = 0
    n_imu_tight_skip = 0
    # PR acceleration weighting: need per-satellite PR history
    pr_history: dict[int, list[float]] = {}  # prn -> [pr(t-2), pr(t-1)]

    prev_tow = None
    prev_measurements: list | None = None
    prev_estimate: np.ndarray | None = None
    prev_pf_estimate: np.ndarray | None = None
    t0 = time.perf_counter()
    epochs_done = 0

    for sol_epoch, measurements in epochs:
        if not sol_epoch.is_valid() or len(measurements) < 4:
            continue
        if max_epochs and epochs_done >= skip_valid_epochs + max_epochs:
            break

        tow = sol_epoch.time.tow
        tow_key = round(tow, 1)
        dt = tow - prev_tow if prev_tow else 0.1

        velocity = None
        imu_velocity = None
        used_tdcp = False
        tdcp_rms = float("nan")
        used_imu = False

        if prev_tow is not None and dt > 0:
            # --- IMU-based velocity ---
            if predict_guide in ("imu", "imu_spp_blend") and imu_filter is not None:
                # Get current PF estimate for ECEF -> geodetic
                cur_est = np.asarray(pf.estimate()[:3], dtype=np.float64)
                lat_r, lon_r, _ = ecef_to_lla(
                    float(cur_est[0]), float(cur_est[1]), float(cur_est[2])
                )

                # Correct heading drift with SPP velocity if available
                pk = round(prev_tow, 1)
                spp_fd_vel_ecef: np.ndarray | None = None
                if tow_key in spp_lookup and pk in spp_lookup:
                    ddx = spp_lookup[tow_key][:3] - spp_lookup[pk][:3]
                    if np.all(np.isfinite(ddx)):
                        spp_fd_vel_ecef = ddx / dt
                        spp_heading = _spp_heading_from_velocity(
                            spp_fd_vel_ecef, lat_r, lon_r
                        )
                        if spp_heading is not None:
                            imu_filter.correct_heading_spp(spp_heading)

                # Get IMU velocity in ENU
                vel_enu = imu_filter.get_velocity_enu(prev_tow, tow)
                speed_enu = float(np.linalg.norm(vel_enu[:2]))

                if speed_enu > 0.01:
                    # Convert ENU velocity to ECEF
                    imu_vel_ecef = ComplementaryHeadingFilter.velocity_enu_to_ecef(
                        vel_enu, lat_r, lon_r
                    )

                    if predict_guide == "imu":
                        velocity = imu_vel_ecef
                        imu_velocity = imu_vel_ecef
                        used_imu = True
                        n_imu_used += 1
                    elif predict_guide == "imu_spp_blend":
                        # Blend IMU + SPP velocity (average)
                        if spp_fd_vel_ecef is not None:
                            spp_speed = float(np.linalg.norm(spp_fd_vel_ecef))
                            if spp_speed < 50:
                                velocity = 0.5 * imu_vel_ecef + 0.5 * spp_fd_vel_ecef
                            else:
                                # SPP velocity unreasonable, use IMU only
                                velocity = imu_vel_ecef
                        else:
                            # No SPP velocity, use IMU only
                            velocity = imu_vel_ecef
                        imu_velocity = velocity
                        used_imu = True
                        n_imu_used += 1
                else:
                    # IMU reports near-zero speed, fall back to SPP
                    n_imu_fallback += 1

            # --- TDCP-based velocity ---
            if predict_guide in ("tdcp", "tdcp_adaptive") and prev_measurements is not None:
                spp_pos_pre = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
                pk = round(prev_tow, 1)
                spp_fd_vel: np.ndarray | None = None
                if tow_key in spp_lookup and pk in spp_lookup:
                    ddx = spp_lookup[tow_key][:3] - spp_lookup[pk][:3]
                    if np.all(np.isfinite(ddx)):
                        spp_fd_vel = ddx / dt
                if np.all(np.isfinite(spp_pos_pre)):
                    tv, t_rms = estimate_velocity_from_tdcp_with_metrics(
                        spp_pos_pre,
                        prev_measurements,
                        measurements,
                        dt=dt,
                        elevation_weight=tdcp_elevation_weight,
                        el_sin_floor=tdcp_el_sin_floor,
                    )
                    if tv is not None and spp_fd_vel is not None:
                        if float(np.linalg.norm(tv - spp_fd_vel)) > 6.0:
                            tv = None
                    if tv is not None:
                        if predict_guide == "tdcp_adaptive" and t_rms >= tdcp_rms_threshold:
                            # Adaptive mode: postfit RMS too large, fall back
                            n_tdcp_fallback += 1
                        else:
                            velocity = tv
                            used_tdcp = True
                            tdcp_rms = float(t_rms)
                            n_tdcp_used += 1
                elif predict_guide == "tdcp_adaptive":
                    n_tdcp_fallback += 1

            # --- SPP finite-difference fallback ---
            if velocity is None and tow_key in spp_lookup:
                pk = round(prev_tow, 1)
                if pk in spp_lookup:
                    vel = (spp_lookup[tow_key][:3] - spp_lookup[pk][:3]) / dt
                    if np.all(np.isfinite(vel)) and np.linalg.norm(vel) < 50:
                        velocity = vel

        sig_predict = float(sigma_pos)
        if used_tdcp and sigma_pos_tdcp is not None:
            sig_predict = float(sigma_pos_tdcp)
        if (
            used_tdcp
            and sigma_pos_tdcp_tight is not None
            and np.isfinite(tdcp_rms)
            and tdcp_rms < float(tdcp_tight_rms_max_m)
        ):
            sig_predict = float(sigma_pos_tdcp_tight)

        pf.predict(velocity=velocity, dt=dt, sigma_pos=sig_predict)

        sat_ecef = np.array([m.satellite_ecef for m in measurements])
        pr = np.array([m.corrected_pseudorange for m in measurements])
        w = np.array([m.weight for m in measurements])

        # --- Residual-based adaptive downweighting ---
        if residual_downweight:
            spp_pos3 = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
            if np.isfinite(spp_pos3).all() and np.linalg.norm(spp_pos3) > 1e6:
                # Estimate clock bias as median(PR - range)
                ranges = np.linalg.norm(sat_ecef - spp_pos3, axis=1)
                cb_est = float(np.median(pr - ranges))
                for i_m in range(len(measurements)):
                    residual = abs(pr[i_m] - ranges[i_m] - cb_est)
                    w[i_m] *= 1.0 / (1.0 + (residual / residual_threshold) ** 2)

        # --- Pseudorange acceleration downweighting ---
        if pr_accel_downweight:
            for i_m in range(len(measurements)):
                prn = int(getattr(measurements[i_m], "prn", 0))
                cur_pr = float(pr[i_m])
                hist = pr_history.get(prn, [])
                if len(hist) >= 2:
                    accel = abs(cur_pr - 2.0 * hist[-1] + hist[-2])
                    w[i_m] *= 1.0 / (1.0 + (accel / pr_accel_threshold) ** 2)
                # Update history (keep last 2)
                hist.append(cur_pr)
                if len(hist) > 2:
                    hist.pop(0)
                pr_history[prn] = hist

        pf.correct_clock_bias(sat_ecef, pr)
        if use_gmm:
            pf.update_gmm(sat_ecef, pr, weights=w,
                          w_los=gmm_w_los, mu_nlos=gmm_mu_nlos, sigma_nlos=gmm_sigma_nlos)
        else:
            pf.update(sat_ecef, pr, weights=w)

        # --- MUPF: carrier phase AFV update (after pseudorange) ---
        if mupf:
            # Collect carrier phase from gnssplusplus measurements
            # Filter: only use high-quality satellites (C/N0 + elevation)
            cp_cycles = []
            cp_sat_ecef = []
            cp_weights = []
            for m in measurements:
                cp = float(getattr(m, "carrier_phase", 0.0))
                if cp == 0.0 or not np.isfinite(cp) or abs(cp) < 1e3:
                    continue
                snr = float(getattr(m, "snr", 0.0))
                elev = float(getattr(m, "elevation", 0.0))
                # Skip low SNR (likely NLOS/multipath)
                if snr < mupf_snr_min and snr > 0:
                    continue
                # Skip low elevation (likely NLOS)
                if 0 < elev < mupf_elev_min:
                    continue
                # Also check pseudorange residual — large residual = likely NLOS
                spp_pos_check = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
                sat_pos = np.array(m.satellite_ecef, dtype=np.float64)
                if np.isfinite(spp_pos_check).all() and np.linalg.norm(spp_pos_check) > 1e6:
                    rng = np.linalg.norm(sat_pos - spp_pos_check)
                    cb_est_m = float(np.median(pr - np.linalg.norm(sat_ecef - spp_pos_check, axis=1)))
                    res = abs(m.corrected_pseudorange - rng - cb_est_m)
                    if res > 30.0:  # large residual = NLOS
                        continue
                cp_cycles.append(cp)
                cp_sat_ecef.append(m.satellite_ecef)
                cp_weights.append(m.weight)
            if len(cp_cycles) >= 4:
                cp_sat = np.array(cp_sat_ecef, dtype=np.float64)
                cp_arr = np.array(cp_cycles, dtype=np.float64)
                cp_w = np.array(cp_weights, dtype=np.float64)
                # Multi-step carrier phase AFV: progressively tighten sigma
                # Step 1: very loose (sigma=2.0 cycles ≈ 38cm) → coarse narrowing
                # Step 2: medium (sigma=0.5 cycles ≈ 10cm) → medium narrowing
                # Step 3: tight (sigma=target) → final precision
                mupf_sigmas = [2.0, 0.5, mupf_sigma_cycles]
                for sig in mupf_sigmas:
                    pf.resample_if_needed()
                    pf.update_carrier_afv(cp_sat, cp_arr, weights=cp_w,
                                          sigma_cycles=sig)

        # --- MUPF-DD: Double-Differenced carrier phase AFV update ---
        if mupf_dd and dd_computer is not None:
            pf_est = np.asarray(pf.estimate()[:3], dtype=np.float64)
            dd_result = dd_computer.compute_dd(tow, measurements, pf_est)
            if dd_result is not None and dd_result.n_dd >= 3:
                # Resample to concentrate particles before DD-AFV
                pf.resample_if_needed()
                pf.update_dd_carrier_afv(dd_result, sigma_cycles=mupf_dd_sigma_cycles)
                n_dd_used += 1
            else:
                n_dd_skip += 1

        spp_pos = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
        if position_update_sigma is not None:
            if np.isfinite(spp_pos).all() and np.linalg.norm(spp_pos) > 1e6:
                pf.position_update(spp_pos, sigma_pos=position_update_sigma)

        # --- IMU tight-coupling dead-reckoning position update ---
        if (
            imu_tight_coupling
            and prev_pf_estimate is not None
            and imu_velocity is not None
            and np.isfinite(imu_velocity).all()
            and dt > 0
        ):
            imu_predicted_pos = prev_pf_estimate + imu_velocity * dt
            if len(measurements) > 0:
                ranges = np.linalg.norm(sat_ecef - spp_pos, axis=1)
                valid_mask = np.isfinite(ranges) & np.isfinite(pr)
                if np.any(valid_mask):
                    cb_est = float(np.median((pr - ranges)[valid_mask]))
                    residuals = np.abs((pr - ranges - cb_est)[valid_mask])
                    spp_residual_rms = float(np.sqrt(np.mean(residuals**2)))
                else:
                    spp_residual_rms = float("inf")
            else:
                spp_residual_rms = float("inf")

            n_sats = len(measurements)
            if n_sats < 6 or spp_residual_rms > 20.0:
                imu_pu_sigma = 3.0
            elif n_sats < 8 or spp_residual_rms > 10.0:
                imu_pu_sigma = 8.0
            else:
                imu_pu_sigma = 30.0

            if np.all(np.isfinite(imu_predicted_pos)):
                pf.position_update(imu_predicted_pos, sigma_pos=imu_pu_sigma)
                n_imu_tight_used += 1
            else:
                n_imu_tight_skip += 1
        elif imu_tight_coupling:
            n_imu_tight_skip += 1

        # Doppler-predicted position update: propagate previous estimate by velocity
        if doppler_position_update and velocity is not None and prev_estimate is not None and dt > 0:
            doppler_predicted_pos = prev_estimate + velocity * dt
            pf.position_update(doppler_predicted_pos, sigma_pos=doppler_pu_sigma)

        # TDCP displacement position update: cm-level carrier phase constraint
        # This is the key to matching FGO performance without FGO.
        # TDCP velocity * dt = cm-level displacement from carrier phase.
        # Apply as tight position_update when TDCP RMS is good.
        if tdcp_position_update and used_tdcp and prev_estimate is not None and dt > 0:
            if np.isfinite(tdcp_rms) and tdcp_rms < tdcp_pu_rms_max:
                tdcp_displacement = velocity * dt  # velocity here is TDCP velocity
                tdcp_predicted_pos = prev_estimate + tdcp_displacement
                if np.all(np.isfinite(tdcp_predicted_pos)):
                    pf.position_update(tdcp_predicted_pos, sigma_pos=tdcp_pu_sigma)

        if use_smoother:
            spp_ref = (
                spp_pos
                if position_update_sigma is not None
                and np.isfinite(spp_pos).all()
                and np.linalg.norm(spp_pos) > 1e6
                else None
            )
            pf.store_epoch(sat_ecef, pr, w, velocity, dt, spp_ref=spp_ref)
            n_stored += 1

        if epochs_done >= skip_valid_epochs:
            gt_idx = np.argmin(np.abs(our_times - tow))
            if abs(our_times[gt_idx] - tow) < 0.05:
                forward_aligned.append(np.asarray(pf.estimate()[:3], dtype=np.float64).copy())
                all_gt.append(np.asarray(gt[gt_idx], dtype=np.float64).copy())
                if use_smoother:
                    aligned_indices.append(n_stored - 1)

        prev_tow = tow
        prev_measurements = list(measurements)
        prev_estimate = np.asarray(pf.estimate()[:3], dtype=np.float64).copy()
        prev_pf_estimate = np.asarray(pf.estimate()[:3], dtype=np.float64).copy()
        epochs_done += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if predict_guide == "tdcp_adaptive":
        total_tdcp = n_tdcp_used + n_tdcp_fallback
        print(
            f"  [tdcp_adaptive] TDCP used {n_tdcp_used}/{total_tdcp} epochs, "
            f"fallback {n_tdcp_fallback}/{total_tdcp} (rms_threshold={tdcp_rms_threshold:.1f}m)"
        )

    if predict_guide in ("imu", "imu_spp_blend"):
        total_imu = n_imu_used + n_imu_fallback
        print(
            f"  [{predict_guide}] IMU used {n_imu_used}/{total_imu} epochs, "
            f"fallback {n_imu_fallback}/{total_imu}"
        )

    if imu_tight_coupling:
        total_tight = n_imu_tight_used + n_imu_tight_skip
        print(
            f"  [imu_tight] IMU position_update used {n_imu_tight_used}/{total_tight} epochs, "
            f"skip {n_imu_tight_skip}/{total_tight}"
        )

    if mupf_dd:
        total_dd = n_dd_used + n_dd_skip
        print(
            f"  [mupf_dd] DD-AFV used {n_dd_used}/{total_dd} epochs, "
            f"skip {n_dd_skip}/{total_dd}"
        )

    forward_pos_full = np.array(forward_aligned, dtype=np.float64)
    gt_arr = np.array(all_gt, dtype=np.float64)

    result: dict[str, object] = {
        "run": run_name,
        "n_particles": n_particles,
        "predict_guide": predict_guide,
        "position_update_sigma": position_update_sigma,
        "use_smoother": use_smoother,
        "skip_valid_epochs": skip_valid_epochs,
        "sigma_pos_tdcp": sigma_pos_tdcp,
        "sigma_pos_tdcp_tight": sigma_pos_tdcp_tight,
        "tdcp_tight_rms_max_m": tdcp_tight_rms_max_m,
        "tdcp_elevation_weight": tdcp_elevation_weight,
        "tdcp_el_sin_floor": tdcp_el_sin_floor,
        "tdcp_rms_threshold": tdcp_rms_threshold,
        "doppler_position_update": doppler_position_update,
        "doppler_pu_sigma": doppler_pu_sigma,
        "imu_tight_coupling": imu_tight_coupling,
        "n_imu_tight_used": n_imu_tight_used,
        "n_imu_tight_skip": n_imu_tight_skip,
        "n_tdcp_used": n_tdcp_used,
        "n_tdcp_fallback": n_tdcp_fallback,
        "n_imu_used": n_imu_used,
        "n_imu_fallback": n_imu_fallback,
        "elapsed_ms": elapsed_ms,
        "forward_metrics": None,
        "smoothed_metrics": None,
    }

    if len(forward_pos_full) == 0:
        return result

    result["forward_metrics"] = compute_metrics(forward_pos_full, gt_arr[: len(forward_pos_full)])

    if use_smoother and n_stored > 0:
        smoothed_full, _forward_stored = pf.smooth(
            position_update_sigma=position_update_sigma,
        )
        if aligned_indices:
            idx = np.asarray(aligned_indices, dtype=np.int64)
            smoothed_aligned_arr = smoothed_full[idx]
            result["smoothed_metrics"] = compute_metrics(
                smoothed_aligned_arr, gt_arr[: len(smoothed_aligned_arr)]
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PF + optional forward-backward smoother (gnssplusplus stack)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default="Odaiba")
    parser.add_argument("--n-particles", type=int, default=100_000)
    parser.add_argument("--sigma-pos", type=float, default=PF_SIGMA_POS)
    parser.add_argument("--sigma-pr", type=float, default=3.0)
    parser.add_argument(
        "--position-update-sigma",
        type=float,
        default=3.0,
        help="SPP soft constraint (m); use negative to disable",
    )
    parser.add_argument(
        "--predict-guide",
        choices=("spp", "tdcp", "tdcp_adaptive", "imu", "imu_spp_blend"),
        default="spp",
    )
    parser.add_argument("--smoother", action="store_true", help="Enable forward-backward smooth")
    parser.add_argument("--compare-both", action="store_true", help="Run with and without smoother")
    parser.add_argument("--max-epochs", type=int, default=0, help="Limit valid epochs (0 = no limit)")
    parser.add_argument(
        "--skip-valid-epochs",
        type=int,
        default=0,
        help="Process (burn-in) this many valid epochs before recording metrics; "
        "total processed = skip + max-epochs when max-epochs > 0",
    )
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument(
        "--sigma-pos-tdcp",
        type=float,
        default=None,
        help="When TDCP velocity is accepted, use this predict sigma_pos (m); "
        "omit to use --sigma-pos for all epochs",
    )
    parser.add_argument(
        "--sigma-pos-tdcp-tight",
        type=float,
        default=None,
        help="If set and TDCP postfit RMS < --tdcp-tight-rms-max, use this sigma_pos",
    )
    parser.add_argument(
        "--tdcp-tight-rms-max",
        type=float,
        default=1.0e9,
        help="postfit RMS threshold (m) for --sigma-pos-tdcp-tight (default: disabled)",
    )
    parser.add_argument(
        "--tdcp-rms-threshold",
        type=float,
        default=3.0,
        help="Postfit RMS threshold (m) for tdcp_adaptive mode; "
        "epochs with RMS >= threshold fall back to Doppler/random-walk predict",
    )
    parser.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="WLS row weight sin(el)^2 when measurements expose elevation (TDCP guide only)",
    )
    parser.add_argument(
        "--tdcp-el-sin-floor",
        type=float,
        default=0.1,
        help="Floor on sin(elevation) when --tdcp-elevation-weight is set",
    )
    parser.add_argument(
        "--residual-downweight",
        action="store_true",
        help="Downweight satellites with large SPP residuals (Cauchy-like)",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=15.0,
        help="Residual threshold (m) for Cauchy downweighting",
    )
    parser.add_argument(
        "--pr-accel-downweight",
        action="store_true",
        help="Downweight satellites with large pseudorange acceleration (multipath indicator)",
    )
    parser.add_argument(
        "--pr-accel-threshold",
        type=float,
        default=5.0,
        help="PR acceleration threshold (m) for Cauchy downweighting",
    )
    parser.add_argument("--gmm", action="store_true", help="Use GMM likelihood (LOS+NLOS mixture)")
    parser.add_argument("--gmm-w-los", type=float, default=0.7, help="GMM LOS weight")
    parser.add_argument("--gmm-mu-nlos", type=float, default=15.0, help="GMM NLOS mean bias (m)")
    parser.add_argument("--gmm-sigma-nlos", type=float, default=30.0, help="GMM NLOS sigma (m)")
    parser.add_argument(
        "--doppler-position-update",
        action="store_true",
        help="Apply a second position_update using Doppler-predicted position (prev_estimate + velocity*dt)",
    )
    parser.add_argument(
        "--doppler-pu-sigma",
        type=float,
        default=5.0,
        help="Sigma (m) for Doppler-predicted position_update constraint",
    )
    parser.add_argument(
        "--imu-tight-coupling",
        action="store_true",
        help="Apply IMU dead-reckoning position_update after SPP in each epoch",
    )
    parser.add_argument(
        "--tdcp-position-update",
        action="store_true",
        help="Apply TDCP displacement as tight position_update (carrier-phase constraint)",
    )
    parser.add_argument("--tdcp-pu-sigma", type=float, default=0.5,
                        help="Sigma for TDCP displacement position_update (m)")
    parser.add_argument("--tdcp-pu-rms-max", type=float, default=3.0,
                        help="Max TDCP postfit RMS to apply displacement PU (m)")
    parser.add_argument("--mupf", action="store_true",
                        help="Multiple Update PF: carrier phase AFV update after pseudorange")
    parser.add_argument("--mupf-sigma-cycles", type=float, default=0.05,
                        help="Carrier phase AFV sigma in cycles (default 0.05 ≈ 1cm)")
    parser.add_argument("--mupf-snr-min", type=float, default=25.0,
                        help="Min C/N0 (dB-Hz) for carrier phase in MUPF")
    parser.add_argument("--mupf-elev-min", type=float, default=0.15,
                        help="Min elevation (rad) for carrier phase in MUPF (~8.6 deg)")
    parser.add_argument("--mupf-dd", action="store_true",
                        help="Use Double-Differenced carrier phase AFV (requires base station RINEX)")
    parser.add_argument("--mupf-dd-sigma-cycles", type=float, default=0.05,
                        help="DD carrier phase AFV sigma in cycles (default 0.05)")
    args = parser.parse_args()

    pos_sigma = args.position_update_sigma
    if pos_sigma < 0:
        pos_sigma = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows: list[dict[str, object]] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*60}\n  {run_name}\n{'='*60}")
        variants: list[tuple[str, bool]] = []
        if args.compare_both:
            variants.append(("forward_only", False))
            variants.append(("with_smoother", True))
        else:
            variants.append(("forward_only" if not args.smoother else "with_smoother", args.smoother))

        for label, use_sm in variants:
            print(
                f"  [{label}] guide={args.predict_guide} PU={pos_sigma} "
                f"sp_tdcp={args.sigma_pos_tdcp} smooth={use_sm}...",
                end=" ",
                flush=True,
            )
            out = run_pf_with_optional_smoother(
                run_dir,
                run_name,
                n_particles=args.n_particles,
                sigma_pos=args.sigma_pos,
                sigma_pr=args.sigma_pr,
                position_update_sigma=pos_sigma,
                predict_guide=args.predict_guide,
                use_smoother=use_sm,
                rover_source=args.urban_rover,
                max_epochs=args.max_epochs,
                skip_valid_epochs=args.skip_valid_epochs,
                sigma_pos_tdcp=args.sigma_pos_tdcp,
                sigma_pos_tdcp_tight=args.sigma_pos_tdcp_tight,
                tdcp_tight_rms_max_m=args.tdcp_tight_rms_max,
                tdcp_elevation_weight=args.tdcp_elevation_weight,
                tdcp_el_sin_floor=args.tdcp_el_sin_floor,
                tdcp_rms_threshold=args.tdcp_rms_threshold,
                residual_downweight=args.residual_downweight,
                residual_threshold=args.residual_threshold,
                pr_accel_downweight=args.pr_accel_downweight,
                pr_accel_threshold=args.pr_accel_threshold,
                use_gmm=args.gmm,
                gmm_w_los=args.gmm_w_los,
                gmm_mu_nlos=args.gmm_mu_nlos,
                gmm_sigma_nlos=args.gmm_sigma_nlos,
                doppler_position_update=args.doppler_position_update,
                doppler_pu_sigma=args.doppler_pu_sigma,
                imu_tight_coupling=args.imu_tight_coupling,
                tdcp_position_update=args.tdcp_position_update,
                tdcp_pu_sigma=args.tdcp_pu_sigma,
                tdcp_pu_rms_max=args.tdcp_pu_rms_max,
                mupf=args.mupf,
                mupf_sigma_cycles=args.mupf_sigma_cycles,
                mupf_snr_min=args.mupf_snr_min,
                mupf_elev_min=args.mupf_elev_min,
                mupf_dd=args.mupf_dd,
                mupf_dd_sigma_cycles=args.mupf_dd_sigma_cycles,
            )
            fm = out["forward_metrics"]
            sm = out["smoothed_metrics"]
            ep_n = int(fm["n_epochs"]) if fm else 0
            ms_ep = float(out["elapsed_ms"]) / ep_n if ep_n else 0.0
            if fm:
                print(
                    f"FWD P50={fm['p50']:.2f}m RMS={fm['rms_2d']:.2f}m "
                    f"({ep_n} ep, {ms_ep:.2f}ms/ep)"
                )
            if sm:
                print(
                    f"       SMTH P50={sm['p50']:.2f}m RMS={sm['rms_2d']:.2f}m"
                )
            rows.append({
                "run": run_name,
                "variant": label,
                "predict_guide": args.predict_guide,
                "sigma_pos": args.sigma_pos,
                "sigma_pos_tdcp": args.sigma_pos_tdcp,
                "sigma_pos_tdcp_tight": args.sigma_pos_tdcp_tight,
                "tdcp_tight_rms_max_m": args.tdcp_tight_rms_max,
                "skip_valid_epochs": args.skip_valid_epochs,
                "tdcp_elevation_weight": args.tdcp_elevation_weight,
                "tdcp_el_sin_floor": args.tdcp_el_sin_floor,
                "tdcp_rms_threshold": args.tdcp_rms_threshold,
                "residual_downweight": args.residual_downweight,
                "residual_threshold": args.residual_threshold,
                "pr_accel_downweight": args.pr_accel_downweight,
                "pr_accel_threshold": args.pr_accel_threshold,
                "position_update_sigma": pos_sigma if pos_sigma is not None else "off",
                "doppler_position_update": args.doppler_position_update,
                "doppler_pu_sigma": args.doppler_pu_sigma,
                "imu_tight_coupling": args.imu_tight_coupling,
                "mupf": args.mupf,
                "mupf_dd": args.mupf_dd,
                "smoother": use_sm,
                "n_particles": args.n_particles,
                "forward_p50": fm["p50"] if fm else None,
                "forward_p95": fm["p95"] if fm else None,
                "forward_rms_2d": fm["rms_2d"] if fm else None,
                "smoothed_p50": sm["p50"] if sm else None,
                "smoothed_p95": sm["p95"] if sm else None,
                "smoothed_rms_2d": sm["rms_2d"] if sm else None,
                "n_epochs": ep_n,
                "ms_per_epoch": ms_ep,
            })

    out_csv = RESULTS_DIR / "pf_smoother_eval.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
