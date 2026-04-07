#!/usr/bin/env python3
"""Evaluate wide-lane ambiguity fixed pseudorange correction on the PF stack.

Pipeline (forward): predict (SPP or TDCP guide) -> correct_clock_bias -> update
with measurements where wide-lane-fixed PR replaces code PR when available, then
optional position_update and smoothing.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python",
    _PROJECT_ROOT / "python",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate import compute_metrics
from gnss_gpu import ParticleFilterDevice
from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.wide_lane import WidelaneResolver, L1_FREQ, L2_FREQ, LAMBDA_1, LAMBDA_2
from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics
from exp_urbannav_pf3d import PF_SIGMA_CB, PF_SIGMA_POS

RESULTS_DIR = _SCRIPT_DIR / "results"
_SYSTEM_ID_TO_CHAR = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}


def _to_sat_id(system_id: int, prn: int) -> str:
    """Format libgnsspp system/prn pair to RINEX satellite id."""
    return f"{_SYSTEM_ID_TO_CHAR.get(system_id, 'G')}{int(prn):02d}"


def _build_urban_maps(data: dict[str, object]) -> dict[float, dict[str, dict[str, float]]]:
    time_series = data.get("times")
    l1_series = data.get("l1_carrier_per_epoch", [])
    l2pr_series = data.get("l2_pr_per_epoch", [])
    l2car_series = data.get("l2_carrier_per_epoch", [])

    maps: dict[float, dict[str, dict[str, float]]] = {}
    if time_series is None or len(time_series) == 0:
        return maps

    for i, tow in enumerate(time_series):
        key = round(float(tow), 1)
        l1_map = l1_series[i] if i < len(l1_series) else {}
        l2pr_map = l2pr_series[i] if i < len(l2pr_series) else {}
        l2car_map = l2car_series[i] if i < len(l2car_series) else {}
        maps[key] = {
            "l1_carrier": {str(k): float(v) for k, v in (l1_map or {}).items() if np.isfinite(float(v))},
            "l2_pr": {str(k): float(v) for k, v in (l2pr_map or {}).items() if np.isfinite(float(v))},
            "l2_carrier": {str(k): float(v) for k, v in (l2car_map or {}).items() if np.isfinite(float(v))},
        }
    return maps


def load_widelane_dataset(
    run_dir: Path,
    rover_source: str = "trimble",
    systems: tuple[str, ...] = ("G", "E", "J"),
) -> dict[str, object]:
    """Load both gnssplusplus epoch stream and wide-lane-ready UrbanNav maps."""
    from libgnsspp import preprocess_spp_file, solve_spp_file

    obs_path = str(run_dir / f"rover_{rover_source}.obs")
    nav_path = str(run_dir / "base.nav")

    epochs = preprocess_spp_file(obs_path, nav_path)
    sol = solve_spp_file(obs_path, nav_path)
    spp_records = [r for r in sol.records() if r.is_valid()]
    spp_lookup = {round(r.time.tow, 1): np.array(r.position_ecef_m) for r in spp_records}

    loader = UrbanNavLoader(run_dir)
    data = loader.load_experiment_data(
        rover_source=rover_source,
        systems=systems,
        return_l1_carrier_per_epoch=True,
        return_l2_pr_per_epoch=True,
        return_l2_carrier_per_epoch=True,
    )

    gt = data["ground_truth"]
    our_times = np.asarray(data["times"], dtype=np.float64)
    urban_maps = _build_urban_maps(data)

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

    return {
        "epochs": epochs,
        "spp_lookup": spp_lookup,
        "gt": gt,
        "our_times": our_times,
        "first_pos": first_pos,
        "init_cb": init_cb,
        "urban_maps": urban_maps,
    }


def run_pf_with_widelane(
    run_dir: Path,
    run_name: str,
    *,
    n_particles: int,
    sigma_pos: float,
    sigma_pr: float,
    position_update_sigma: float | None,
    predict_guide: str,
    rover_source: str = "trimble",
    max_epochs: int = 0,
    skip_valid_epochs: int = 0,
    dataset: dict[str, object] | None = None,
    resampling: str = "megopolis",
    tdcp_elevation_weight: bool = False,
    tdcp_el_sin_floor: float = 0.1,
    tdcp_rms_threshold: float = 3.0,
    verbose: bool = False,
) -> dict[str, object]:
    ds = load_widelane_dataset(run_dir, rover_source) if dataset is None else dataset
    epochs = ds["epochs"]
    spp_lookup = ds["spp_lookup"]
    gt = ds["gt"]
    our_times = ds["our_times"]
    first_pos = np.asarray(ds["first_pos"], dtype=np.float64)
    init_cb = float(ds["init_cb"])
    urban_maps = ds["urban_maps"]

    pf = ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=sigma_pos,
        sigma_cb=PF_SIGMA_CB,
        sigma_pr=sigma_pr,
        resampling=resampling,
        seed=42,
    )
    pf.initialize(first_pos, clock_bias=init_cb, spread_pos=10.0, spread_cb=100.0)

    pf.enable_smoothing()

    forward_aligned: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    aligned_indices: list[int] = []

    total_pr_count = 0
    total_wl_candidates = 0
    total_wl_fixed = 0
    total_wl_replaced = 0
    fixed_sat_ids = set[str]()
    n_stored = 0
    epoch_counter = 0

    prev_tow = None
    prev_measurements: list | None = None
    t0 = time.perf_counter()
    epochs_done = 0
    n_tdcp_used = 0
    n_tdcp_fallback = 0

    resolver = WidelaneResolver()
    _carrier_smooth_state: dict[int, tuple[float, float]] = {}  # prn -> (smoothed_pr, prev_carrier_m)

    for sol_epoch, measurements in epochs:
        if not sol_epoch.is_valid() or len(measurements) < 4:
            continue
        if max_epochs and epochs_done >= skip_valid_epochs + max_epochs:
            break

        tow = sol_epoch.time.tow
        tow_key = round(tow, 1)
        obs_maps = urban_maps.get(tow_key, {})
        l1_map = obs_maps.get("l1_carrier", {})
        l2pr_map = obs_maps.get("l2_pr", {})
        l2car_map = obs_maps.get("l2_carrier", {})

        dt = tow - prev_tow if prev_tow else 0.1

        # ------------------------------------------------------------
        # Predict
        # ------------------------------------------------------------
        velocity = None
        if prev_tow is not None and dt > 0 and predict_guide == "spp":
            pk = round(prev_tow, 1)
            if tow_key in spp_lookup and pk in spp_lookup:
                velocity = (spp_lookup[tow_key][:3] - spp_lookup[pk][:3]) / dt
                if np.linalg.norm(velocity) > 50:
                    velocity = None
        if prev_tow is not None and dt > 0 and predict_guide in ("tdcp", "tdcp_adaptive") and prev_measurements is not None:
            spp_pos_pre = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
            pk = round(prev_tow, 1)
            tv: np.ndarray | None = None
            tdcp_rms = float("nan")
            if np.all(np.isfinite(spp_pos_pre)) and tow_key in spp_lookup and pk in spp_lookup:
                tv, tdcp_rms = estimate_velocity_from_tdcp_with_metrics(
                    spp_pos_pre,
                    prev_measurements,
                    measurements,
                    dt=dt,
                    elevation_weight=tdcp_elevation_weight,
                    el_sin_floor=tdcp_el_sin_floor,
                )
            if tv is not None and np.linalg.norm(tv) < 60:
                if predict_guide == "tdcp_adaptive" and tdcp_rms >= tdcp_rms_threshold:
                    n_tdcp_fallback += 1
                    tv = None
                else:
                    velocity = tv
                    n_tdcp_used += 1
            else:
                if predict_guide == "tdcp_adaptive":
                    n_tdcp_fallback += 1

        if velocity is None and prev_tow is not None and dt > 0:
            pk = round(prev_tow, 1)
            if tow_key in spp_lookup and pk in spp_lookup:
                vel = (spp_lookup[tow_key][:3] - spp_lookup[pk][:3]) / dt
                if np.linalg.norm(vel) < 50:
                    velocity = vel

        pf.predict(velocity=velocity, dt=dt, sigma_pos=float(sigma_pos))

        # ------------------------------------------------------------
        # Correct measurements (wide-lane replacement)
        # ------------------------------------------------------------
        sat_ecef = np.array([m.satellite_ecef for m in measurements], dtype=np.float64)
        sat_ids = [
            _to_sat_id(int(getattr(m, "system_id", 0)), int(getattr(m, "prn", 0)))
            for m in measurements
        ]
        corrected_pr = np.array([m.corrected_pseudorange for m in measurements], dtype=np.float64)
        weights = np.array([m.weight for m in measurements], dtype=np.float64)

        wl_fixed_epoch = 0
        wl_replaced_epoch = 0
        wl_candidates_epoch = 0

        # Carrier-phase smoothing (divergence-free, dual-freq iono-free)
        # Use iono-free carrier combination delta to smooth code PR:
        #   smoothed_PR(t) = alpha * code_PR(t) + (1-alpha) * (smoothed_PR(t-1) + delta_iono_free_carrier)
        # where delta_iono_free_carrier = (f1^2*dL1 - f2^2*dL2)/(f1^2-f2^2) * wavelengths
        # This avoids ambiguity resolution entirely.
        alpha = 0.2  # weight on raw code PR per epoch (lower = more smoothing)
        f1sq = L1_FREQ * L1_FREQ
        f2sq = L2_FREQ * L2_FREQ

        for i, m in enumerate(measurements):
            sat_id = sat_ids[i]
            prn = int(getattr(m, "prn", 0))

            l1 = l1_map.get(sat_id)
            # Fallback: use gnssplusplus carrier_phase (L1) if RINEX L1 unavailable
            if l1 is None:
                cp = float(getattr(m, "carrier_phase", 0.0))
                if cp != 0.0 and np.isfinite(cp):
                    l1 = cp
            l2c = l2car_map.get(sat_id)
            l2p = l2pr_map.get(sat_id)

            wl_candidates_epoch += 1
            total_pr_count += 1

            if l1 is None:
                # No carrier at all — reset smoother for this sat, use code PR as-is
                _carrier_smooth_state.pop(prn, None)
                continue

            # Carrier range in meters: iono-free if L2 available, else L1-only
            if l2c is not None:
                carrier_if_m = (f1sq * l1 * LAMBDA_1 - f2sq * l2c * LAMBDA_2) / (f1sq - f2sq)
            else:
                carrier_if_m = l1 * LAMBDA_1

            prev_state = _carrier_smooth_state.get(prn)
            code_pr_val = corrected_pr[i]

            if prev_state is None:
                # Initialize: smoothed = code PR
                _carrier_smooth_state[prn] = (code_pr_val, carrier_if_m)
                continue

            prev_smoothed, prev_carrier = prev_state
            delta_carrier = carrier_if_m - prev_carrier

            # Divergence-free Hatch filter
            smoothed = alpha * code_pr_val + (1.0 - alpha) * (prev_smoothed + delta_carrier)

            # Sanity: if smoothed diverges too far from code, reset
            if abs(smoothed - code_pr_val) > 50.0:
                _carrier_smooth_state[prn] = (code_pr_val, carrier_if_m)
                continue

            _carrier_smooth_state[prn] = (smoothed, carrier_if_m)
            corrected_pr[i] = smoothed
            wl_replaced_epoch += 1
            wl_fixed_epoch += 1
            fixed_sat_ids.add(sat_id)

        total_wl_candidates += wl_candidates_epoch
        total_wl_fixed += wl_fixed_epoch
        total_wl_replaced += wl_replaced_epoch

        if verbose:
            print(
                f"  [ep] tow={tow:.1f} wl_fixed={wl_fixed_epoch:2d} "
                f"wl_replaced={wl_replaced_epoch:2d} wl_candidates={wl_candidates_epoch:2d}"
            )

        pf.correct_clock_bias(sat_ecef, corrected_pr)
        pf.update(sat_ecef, corrected_pr, weights=weights)

        spp_pos = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
        if position_update_sigma is not None:
            if np.isfinite(spp_pos).all() and np.linalg.norm(spp_pos) > 1e6:
                pf.position_update(spp_pos, sigma_pos=position_update_sigma)

        spp_ref = (
            spp_pos
            if position_update_sigma is not None
            and np.isfinite(spp_pos).all()
            and np.linalg.norm(spp_pos) > 1e6
            else None
        )
        pf.store_epoch(sat_ecef, corrected_pr, weights, velocity, dt, spp_ref=spp_ref)
        n_stored += 1

        if epochs_done >= skip_valid_epochs:
            gt_idx = np.argmin(np.abs(our_times - tow))
            if abs(our_times[gt_idx] - tow) < 0.05:
                forward_aligned.append(np.asarray(pf.estimate()[:3], dtype=np.float64).copy())
                all_gt.append(np.asarray(gt[gt_idx], dtype=np.float64).copy())
                aligned_indices.append(n_stored - 1)

        prev_tow = tow
        prev_measurements = list(measurements)
        epochs_done += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if predict_guide == "tdcp_adaptive":
        total_tdcp = n_tdcp_used + n_tdcp_fallback
        print(
            f"  [tdcp_adaptive] TDCP used {n_tdcp_used}/{total_tdcp} epochs, "
            f"fallback {n_tdcp_fallback}/{total_tdcp}"
        )

    forward_pos_full = np.array(forward_aligned, dtype=np.float64)
    gt_arr = np.array(all_gt, dtype=np.float64)

    result: dict[str, object] = {
        "run": run_name,
        "n_particles": n_particles,
        "predict_guide": predict_guide,
        "position_update_sigma": position_update_sigma,
        "tdcp_rms_threshold": tdcp_rms_threshold,
        "tdcp_elevation_weight": tdcp_elevation_weight,
        "tdcp_el_sin_floor": tdcp_el_sin_floor,
        "skip_valid_epochs": skip_valid_epochs,
        "n_tdcp_used": n_tdcp_used,
        "n_tdcp_fallback": n_tdcp_fallback,
        "elapsed_ms": elapsed_ms,
        "n_wl_candidates": total_wl_candidates,
        "n_wl_fixed": total_wl_fixed,
        "n_wl_replaced": total_wl_replaced,
        "n_wl_fixed_sats": len(fixed_sat_ids),
        "total_pr_count": total_pr_count,
        "forward_metrics": None,
        "smoothed_metrics": None,
    }

    if len(forward_pos_full) == 0:
        return result

    result["forward_metrics"] = compute_metrics(forward_pos_full, gt_arr[: len(forward_pos_full)])

    smoothed_full, _forward_stored = pf.smooth(position_update_sigma=position_update_sigma)
    if aligned_indices:
        idx = np.asarray(aligned_indices, dtype=np.int64)
        smoothed_aligned_arr = smoothed_full[idx]
        result["smoothed_metrics"] = compute_metrics(
            smoothed_aligned_arr, gt_arr[: len(smoothed_aligned_arr)]
        )

    return result



def main() -> None:
    parser = argparse.ArgumentParser(description="PF + wide-lane resolver evaluation")
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
    parser.add_argument("--predict-guide", choices=("spp", "tdcp", "tdcp_adaptive"), default="spp")
    parser.add_argument("--max-epochs", type=int, default=0, help="Limit valid epochs (0 = no limit)")
    parser.add_argument(
        "--skip-valid-epochs",
        type=int,
        default=0,
        help="Skip this many valid epochs before recording metrics",
    )
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument(
        "--tdcp-rms-threshold",
        type=float,
        default=3.0,
        help="Postfit RMS threshold for tdcp_adaptive fallback",
    )
    parser.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="WLS row weight ∝ sin(elevation)^2 when measurements expose elevation",
    )
    parser.add_argument(
        "--tdcp-el-sin-floor",
        type=float,
        default=0.1,
        help="Floor on sin(elevation) when --tdcp-elevation-weight is set",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch wide-lane stats")

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

        out = run_pf_with_widelane(
            run_dir,
            run_name,
            n_particles=args.n_particles,
            sigma_pos=args.sigma_pos,
            sigma_pr=args.sigma_pr,
            position_update_sigma=pos_sigma,
            predict_guide=args.predict_guide,
            rover_source=args.urban_rover,
            max_epochs=args.max_epochs,
            skip_valid_epochs=args.skip_valid_epochs,
            tdcp_elevation_weight=args.tdcp_elevation_weight,
            tdcp_el_sin_floor=args.tdcp_el_sin_floor,
            tdcp_rms_threshold=args.tdcp_rms_threshold,
            verbose=args.verbose,
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
            print(f"     SMTH P50={sm['p50']:.2f}m RMS={sm['rms_2d']:.2f}m")

        total_wl = out["n_wl_candidates"]
        total_fixed = out["n_wl_fixed"]
        total_replaced = out["n_wl_replaced"]
        fixed_ratio = float(total_replaced) / float(total_wl) if total_wl else 0.0
        print(
            "WL stats: fixed_sats={}/{} measurements, "
            "replaced={} of {} ({:.1%}), fixed_epochs={}".format(
                out["n_wl_fixed_sats"],
                total_wl,
                total_replaced,
                total_wl,
                fixed_ratio,
                out["n_wl_fixed"],
            )
        )

        rows.append(
            {
                "run": run_name,
                "predict_guide": args.predict_guide,
                "sigma_pos": args.sigma_pos,
                "sigma_pr": args.sigma_pr,
                "position_update_sigma": pos_sigma if pos_sigma is not None else "off",
                "urban_rover": args.urban_rover,
                "n_particles": args.n_particles,
                "forward_p50": fm["p50"] if fm else None,
                "forward_p95": fm["p95"] if fm else None,
                "forward_rms_2d": fm["rms_2d"] if fm else None,
                "smoothed_p50": sm["p50"] if sm else None,
                "smoothed_p95": sm["p95"] if sm else None,
                "smoothed_rms_2d": sm["rms_2d"] if sm else None,
                "n_epochs": ep_n,
                "ms_per_epoch": ms_ep,
                "n_tdcp_used": out["n_tdcp_used"],
                "n_tdcp_fallback": out["n_tdcp_fallback"],
                "n_wl_fixed_sats": out["n_wl_fixed_sats"],
                "n_wl_candidates": out["n_wl_candidates"],
                "n_wl_fixed": out["n_wl_fixed"],
                "n_wl_replaced": out["n_wl_replaced"],
                "wl_replaced_ratio": fixed_ratio,
            }
        )

    out_csv = RESULTS_DIR / "widelane_eval.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
