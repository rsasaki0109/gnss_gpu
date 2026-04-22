#!/usr/bin/env python3
"""Evaluate particle smoothing on the same PF stack as exp_pf_smoother_eval.

Default **genealogy** smoother: systematic resampling with GPU-exported ancestor
indices + backward lineage sampling (see ``gnss_gpu.particle_ffbsi.genealogy_smooth_sample``).

Optional **marginal** mode: legacy FFBSi-style backward kernel (ignores resampling
genealogy; can mis-rank parents under ESS resampling).

**Memory:** storing (T, N, 4) doubles is ~32 * T * N bytes. Use ``--n-particles``
in the 4k–16k range for full-sequence histories unless you have ample RAM.
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
    _PROJECT_ROOT / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "python",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate import compute_metrics
from exp_pf_smoother_eval import load_pf_smoother_dataset
from exp_urbannav_pf3d import PF_SIGMA_CB, PF_SIGMA_POS
from gnss_gpu.particle_ffbsi import ffbsi_smooth_sample, genealogy_smooth_sample
from gnss_gpu import ParticleFilterDevice
from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics

RESULTS_DIR = _SCRIPT_DIR / "results"


def run_pf_with_ffbsi(
    run_dir: Path,
    run_name: str,
    *,
    n_particles: int,
    sigma_pos: float,
    sigma_pr: float,
    position_update_sigma: float | None,
    predict_guide: str,
    n_ffbsi_paths: int,
    smoother_mode: str = "genealogy",
    rover_source: str = "trimble",
    max_epochs: int = 0,
    sigma_pos_tdcp: float | None = None,
    sigma_pos_tdcp_tight: float | None = None,
    tdcp_tight_rms_max_m: float = 1.0e9,
    dataset: dict[str, object] | None = None,
    ffbsi_seed: int = 12345,
    tdcp_elevation_weight: bool = False,
    tdcp_el_sin_floor: float = 0.1,
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

    if smoother_mode not in ("genealogy", "marginal"):
        raise ValueError("smoother_mode must be 'genealogy' or 'marginal'")
    # Genealogy smoother needs systematic resample indices from the device.
    pf = ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=sigma_pos,
        sigma_cb=PF_SIGMA_CB,
        sigma_pr=sigma_pr,
        resampling="systematic",
        seed=42,
    )
    pf.initialize(first_pos, clock_bias=init_cb, spread_pos=10.0, spread_cb=100.0)

    hist_X: list[np.ndarray] = []
    hist_LW: list[np.ndarray] = []
    hist_vel: list[np.ndarray] = []
    hist_dt: list[float] = []
    hist_sig_pos: list[float] = []
    hist_anc: list[np.ndarray] = []

    forward_aligned: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    aligned_indices: list[int] = []

    prev_tow = None
    prev_measurements: list | None = None
    t0 = time.perf_counter()
    epochs_done = 0
    n_hist = 0

    for sol_epoch, measurements in epochs:
        if not sol_epoch.is_valid() or len(measurements) < 4:
            continue
        if max_epochs and epochs_done >= max_epochs:
            break

        tow = sol_epoch.time.tow
        tow_key = round(tow, 1)
        dt = tow - prev_tow if prev_tow else 0.1

        velocity = None
        used_tdcp = False
        tdcp_rms = float("nan")
        if prev_tow is not None and dt > 0:
            if predict_guide == "tdcp" and prev_measurements is not None:
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
                        velocity = tv
                        used_tdcp = True
                        tdcp_rms = float(t_rms)
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

        vel_arr = (
            np.asarray(velocity, dtype=np.float64).ravel()
            if velocity is not None
            else np.zeros(3, dtype=np.float64)
        )
        if vel_arr.shape != (3,):
            vel_arr = np.resize(vel_arr, 3)

        hist_vel.append(vel_arr.copy())
        hist_dt.append(float(dt))
        hist_sig_pos.append(float(sig_predict))

        pf.predict(velocity=velocity, dt=dt, sigma_pos=sig_predict)

        sat_ecef = np.array([m.satellite_ecef for m in measurements])
        pr = np.array([m.corrected_pseudorange for m in measurements])
        w = np.array([m.weight for m in measurements])

        pf.correct_clock_bias(sat_ecef, pr)
        pf.update(sat_ecef, pr, weights=w, resample=False)

        spp_pos = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
        if position_update_sigma is not None:
            if np.isfinite(spp_pos).all() and np.linalg.norm(spp_pos) > 1e6:
                pf.position_update(spp_pos, sigma_pos=position_update_sigma)

        hist_X.append(pf.get_particles().copy())
        hist_LW.append(pf.get_log_weights().copy())
        n_hist += 1
        did_rs = pf.resample_if_needed()
        if did_rs:
            anc_row = pf.get_resample_ancestors().astype(np.int64, copy=False)
        else:
            anc_row = np.arange(n_particles, dtype=np.int64)
        hist_anc.append(anc_row)

        gt_idx = np.argmin(np.abs(our_times - tow))
        if abs(our_times[gt_idx] - tow) < 0.05:
            forward_aligned.append(np.asarray(pf.estimate()[:3], dtype=np.float64).copy())
            all_gt.append(np.asarray(gt[gt_idx], dtype=np.float64).copy())
            aligned_indices.append(n_hist - 1)

        prev_tow = tow
        prev_measurements = list(measurements)
        epochs_done += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000

    forward_pos_full = np.array(forward_aligned, dtype=np.float64)
    gt_arr = np.array(all_gt, dtype=np.float64)

    result: dict[str, object] = {
        "run": run_name,
        "n_particles": n_particles,
        "n_ffbsi_paths": n_ffbsi_paths,
        "predict_guide": predict_guide,
        "position_update_sigma": position_update_sigma,
        "sigma_pos_tdcp": sigma_pos_tdcp,
        "sigma_pos_tdcp_tight": sigma_pos_tdcp_tight,
        "tdcp_tight_rms_max_m": tdcp_tight_rms_max_m,
        "tdcp_elevation_weight": tdcp_elevation_weight,
        "tdcp_el_sin_floor": tdcp_el_sin_floor,
        "smoother_mode": smoother_mode,
        "elapsed_ms": elapsed_ms,
        "forward_metrics": None,
        "ffbsi_metrics": None,
    }

    if len(forward_pos_full) == 0 or n_hist == 0:
        return result

    result["forward_metrics"] = compute_metrics(forward_pos_full, gt_arr[: len(forward_pos_full)])

    X = np.stack(hist_X, axis=0)
    LW = np.stack(hist_LW, axis=0)
    V = np.stack(hist_vel, axis=0)
    D = np.asarray(hist_dt, dtype=np.float64)
    S = np.asarray(hist_sig_pos, dtype=np.float64)
    A = np.stack(hist_anc, axis=0)

    if n_ffbsi_paths > 0:
        if smoother_mode == "genealogy":
            paths = np.stack(
                [
                    genealogy_smooth_sample(
                        LW, X, A, np.random.default_rng(ffbsi_seed + k)
                    )
                    for k in range(n_ffbsi_paths)
                ],
                axis=0,
            )
        else:
            paths = np.stack(
                [
                    ffbsi_smooth_sample(
                        LW, X, V, D, S, float(pf.sigma_cb), np.random.default_rng(ffbsi_seed + k)
                    )
                    for k in range(n_ffbsi_paths)
                ],
                axis=0,
            )
        smoothed_full = paths.mean(axis=0)
        if aligned_indices:
            idx = np.asarray(aligned_indices, dtype=np.int64)
            sm_align = smoothed_full[idx]
            result["ffbsi_metrics"] = compute_metrics(
                sm_align, gt_arr[: len(sm_align)]
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PF + FFBSi smoothing (particle backward sampling)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default="Odaiba")
    parser.add_argument("--n-particles", type=int, default=8192)
    parser.add_argument("--sigma-pos", type=float, default=PF_SIGMA_POS)
    parser.add_argument("--sigma-pr", type=float, default=3.0)
    parser.add_argument(
        "--position-update-sigma",
        type=float,
        default=3.0,
        help="SPP soft constraint (m); use negative to disable",
    )
    parser.add_argument("--predict-guide", choices=("spp", "tdcp"), default="spp")
    parser.add_argument("--n-ffbsi-paths", type=int, default=8, help="Monte Carlo smoother trajectories (averaged)")
    parser.add_argument(
        "--smoother",
        choices=("genealogy", "marginal"),
        default="genealogy",
        help="genealogy: ancestor-consistent (default); marginal: legacy Gaussian FFBSi",
    )
    parser.add_argument("--ffbsi-seed", type=int, default=12345)
    parser.add_argument("--max-epochs", type=int, default=0, help="Limit valid epochs (0 = no limit)")
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument("--sigma-pos-tdcp", type=float, default=None)
    parser.add_argument("--sigma-pos-tdcp-tight", type=float, default=None)
    parser.add_argument("--tdcp-tight-rms-max", type=float, default=1.0e9)
    parser.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="TDCP WLS: weight rows by sin(el)^2 when elevation is present",
    )
    parser.add_argument(
        "--tdcp-el-sin-floor",
        type=float,
        default=0.1,
        help="Minimum sin(elevation) under elevation weighting",
    )
    args = parser.parse_args()

    pos_sigma = args.position_update_sigma
    if pos_sigma < 0:
        pos_sigma = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows: list[dict[str, object]] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*60}\n  {run_name} (smoother={args.smoother})\n{'='*60}")
        print(
            f"  N={args.n_particles} paths={args.n_ffbsi_paths} guide={args.predict_guide} "
            f"PU={pos_sigma}...",
            end=" ",
            flush=True,
        )
        out = run_pf_with_ffbsi(
            run_dir,
            run_name,
            n_particles=args.n_particles,
            sigma_pos=args.sigma_pos,
            sigma_pr=args.sigma_pr,
            position_update_sigma=pos_sigma,
            predict_guide=args.predict_guide,
            n_ffbsi_paths=args.n_ffbsi_paths,
            smoother_mode=args.smoother,
            rover_source=args.urban_rover,
            max_epochs=args.max_epochs,
            sigma_pos_tdcp=args.sigma_pos_tdcp,
            sigma_pos_tdcp_tight=args.sigma_pos_tdcp_tight,
            tdcp_tight_rms_max_m=args.tdcp_tight_rms_max,
            ffbsi_seed=args.ffbsi_seed,
            tdcp_elevation_weight=args.tdcp_elevation_weight,
            tdcp_el_sin_floor=args.tdcp_el_sin_floor,
        )
        fm = out["forward_metrics"]
        sm = out["ffbsi_metrics"]
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
            "smoother": args.smoother,
            "predict_guide": args.predict_guide,
            "sigma_pos": args.sigma_pos,
            "sigma_pos_tdcp": args.sigma_pos_tdcp,
            "sigma_pos_tdcp_tight": args.sigma_pos_tdcp_tight,
            "tdcp_tight_rms_max_m": args.tdcp_tight_rms_max,
            "tdcp_elevation_weight": args.tdcp_elevation_weight,
            "tdcp_el_sin_floor": args.tdcp_el_sin_floor,
            "position_update_sigma": pos_sigma if pos_sigma is not None else "off",
            "n_particles": args.n_particles,
            "n_ffbsi_paths": args.n_ffbsi_paths,
            "forward_p50": fm["p50"] if fm else None,
            "forward_p95": fm["p95"] if fm else None,
            "forward_rms_2d": fm["rms_2d"] if fm else None,
            "ffbsi_p50": sm["p50"] if sm else None,
            "ffbsi_p95": sm["p95"] if sm else None,
            "ffbsi_rms_2d": sm["rms_2d"] if sm else None,
            "n_epochs": ep_n,
            "ms_per_epoch": ms_ep,
        })

    out_csv = RESULTS_DIR / "pf_ffbsi_eval.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
