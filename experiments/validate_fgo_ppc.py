#!/usr/bin/env python3
"""Validate GPU FGO on PPC-Dataset (taroz/PPC-Dataset) RINEX runs.

Uses the RTKLIB export_spp_meas pipeline for observation model alignment,
then runs WLS + FGO and compares against PPC reference.csv ground truth.

Usage:
    PYTHONPATH=python python3 experiments/validate_fgo_ppc.py
    PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --run tokyo/run1 --max-epochs 300
    PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --all
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
for _p in (_REPO, _EXPERIMENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.fgo import fgo_gnss_lm, fgo_gnss_lm_vd  # noqa: E402
from gnss_gpu.io.rinex import read_rinex_obs  # noqa: E402
from gnss_gpu.io.nav_rinex import (  # noqa: E402
    _datetime_to_gps_seconds_of_week,
    _datetime_to_gps_week,
    read_gps_klobuchar_from_nav_header,
    read_nav_rinex,
)
from gnss_gpu.spp import correct_pseudoranges  # noqa: E402
from gtsam_public_dataset import SYS_ID_TO_KIND  # noqa: E402

C_LIGHT = 299792458.0

# Default PPC-Dataset location
_DEFAULT_PPC_ROOT = _REPO.parent / "ref" / "PPC-Dataset" / "PPC-Dataset"

# All 6 runs
ALL_RUNS = [
    "tokyo/run1", "tokyo/run2", "tokyo/run3",
    "nagoya/run1", "nagoya/run2", "nagoya/run3",
]


def _default_export_spp_meas() -> Path | None:
    envp = os.environ.get("RTKLIB_EXPORT_SPP_MEAS")
    if envp:
        p = Path(envp)
        if p.is_file():
            return p
    guess = (
        _REPO.parent / "ref" / "RTKLIB-demo5" / "app" / "consapp"
        / "rnx2rtkp" / "gcc" / "export_spp_meas"
    )
    return guess if guess.is_file() else None


def _load_ppc_reference(ref_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load PPC reference.csv as (tow, ecef)."""
    tow_list, ecef_list = [], []
    with open(ref_path, newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            tow_list.append(float(row["GPS TOW (s)"]))
            ecef_list.append([
                float(row["ECEF X (m)"]),
                float(row["ECEF Y (m)"]),
                float(row["ECEF Z (m)"]),
            ])
    return np.array(tow_list), np.array(ecef_list)


def _nearest_ref_error_2d(
    tow: float, ref_tow: np.ndarray, ref_ecef: np.ndarray, est: np.ndarray,
) -> float:
    i = int(np.argmin(np.abs(ref_tow - tow)))
    d = est[:3] - ref_ecef[i]
    return float(np.linalg.norm(d[:2]))


def _nearest_ref_error_3d(
    tow: float, ref_tow: np.ndarray, ref_ecef: np.ndarray, est: np.ndarray,
) -> float:
    i = int(np.argmin(np.abs(ref_tow - tow)))
    d = est[:3] - ref_ecef[i]
    return float(np.linalg.norm(d))


def _run_rtklib_export(
    exe: Path, obs_p: Path, nav_p: Path, el_mask_deg: float,
) -> dict[tuple[int, float, str], dict[str, float]]:
    """Run export_spp_meas and parse CSV.

    Supports both old (GPS-only) and new multi-GNSS format (with ``sys_id`` column).
    """
    import subprocess
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix="_ppc_spp.csv", text=True)
    os.close(fd)
    tmp_p = Path(tmp)
    cmd = [str(exe), str(obs_p), str(nav_p), "-m", str(el_mask_deg)]
    with open(tmp_p, "w") as fp:
        subprocess.run(cmd, check=True, stdout=fp, stderr=subprocess.PIPE, text=True)
    out: dict[tuple[int, float, str], dict[str, float]] = {}
    with open(tmp_p, newline="") as f:
        for row in csv.DictReader(f):
            wk = int(row["gps_week"])
            tow = round(float(row["gps_tow"]), 4)
            sid = row["sat_id"].strip()
            d: dict[str, float] = {
                "prange_m": float(row["prange_m"]),
                "iono_m": float(row["iono_m"]),
                "trop_m": float(row["trop_m"]),
                "sat_clk_m": float(row["sat_clk_m"]),
                "satx": float(row["satx"]),
                "saty": float(row["saty"]),
                "satz": float(row["satz"]),
            }
            if "el_rad" in row and "var_total" in row:
                d["el_rad"] = float(row["el_rad"])
                d["var_total"] = float(row["var_total"])
            if "svx" in row:
                d["svx"] = float(row["svx"])
                d["svy"] = float(row["svy"])
                d["svz"] = float(row["svz"])
            if "rx_vx" in row:
                d["rx_vx"] = float(row["rx_vx"])
                d["rx_vy"] = float(row["rx_vy"])
                d["rx_vz"] = float(row["rx_vz"])
            # Multi-GNSS sys_id column (backward compatible)
            if "sys_id" in row and row["sys_id"]:
                d["sys_id"] = row["sys_id"].strip()  # type: ignore[assignment]
            out[(wk, tow, sid)] = d
    tmp_p.unlink(missing_ok=True)
    return out


def run_fgo_on_ppc(
    run_dir: Path,
    *,
    max_epochs: int = 300,
    el_mask_deg: float = 15.0,
    motion_sigma_m: float = 0.0,
    fgo_iters: int = 8,
    export_spp: Path | None = None,
    use_doppler: bool = False,
    multi_gnss: bool = False,
    use_vd: bool = False,
    clock_drift_sigma_m: float = 1.0,
) -> dict:
    """Run RTKLIB-aligned FGO on a PPC run and return results."""
    obs_p = run_dir / "rover.obs"
    nav_p = run_dir / "base.nav"
    ref_p = run_dir / "reference.csv"
    for need in (obs_p, nav_p, ref_p):
        if not need.is_file():
            raise FileNotFoundError(f"Missing: {need}")

    ref_tow, ref_ecef = _load_ppc_reference(ref_p)

    # Parse RINEX for epoch/satellite structure
    rinex = read_rinex_obs(obs_p)

    # Get RTKLIB measurements
    rtk_meas = None
    if export_spp is not None:
        rtk_meas = _run_rtklib_export(export_spp, obs_p, nav_p, el_mask_deg)

    # Determine which satellite system prefixes to accept
    _accept_prefixes = ("G",)
    if multi_gnss and rtk_meas is not None:
        _accept_prefixes = ("G", "E", "J")

    # Build epoch list from RINEX (also extract Doppler D1C if needed)
    epochs_data: list[tuple[float, list[str], np.ndarray, int]] = []
    doppler_data: list[dict[str, float]] = []  # per-epoch: sat_id -> D1C
    max_sats = 0
    for ep in rinex.epochs:
        pr_map: dict[str, float] = {}
        dop_map: dict[str, float] = {}
        for sat, obs in ep.observations.items():
            if not any(sat.startswith(p) for p in _accept_prefixes):
                continue
            if "C1C" in obs and obs["C1C"] and obs["C1C"] != 0.0:
                pr_map[sat] = obs["C1C"]
                d1c = obs.get("D1C", 0.0)
                if d1c and d1c != 0.0:
                    dop_map[sat] = float(d1c)
        if len(pr_map) < 4:
            continue
        tow = _datetime_to_gps_seconds_of_week(ep.time)
        wk = _datetime_to_gps_week(ep.time)
        sats = sorted(pr_map.keys())
        pr = np.array([pr_map[s] for s in sats], dtype=np.float64)
        epochs_data.append((tow, sats, pr, wk))
        doppler_data.append(dop_map)
        max_sats = max(max_sats, len(sats))
        if len(epochs_data) >= max_epochs:
            break

    n_epoch = len(epochs_data)
    if n_epoch < 5:
        raise RuntimeError(f"Only {n_epoch} valid epochs")

    # Build padded arrays
    sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
    weights = np.zeros((n_epoch, max_sats), dtype=np.float64)

    # Build sys_kind for multi-GNSS ISB
    has_multi = multi_gnss and rtk_meas is not None
    n_clock = 3 if has_multi else 1
    sys_kind_arr: np.ndarray | None = None
    if has_multi:
        sys_kind_arr = np.zeros((n_epoch, max_sats), dtype=np.int32)
        for t_sk, (_tow_sk, sats_sk, _pr_sk, _wk_sk) in enumerate(epochs_data):
            for si_sk, sid_sk in enumerate(sats_sk):
                prefix = sid_sk[0] if sid_sk else "G"
                sys_kind_arr[t_sk, si_sk] = SYS_ID_TO_KIND.get(prefix, 0)

    approx0 = rinex.header.approx_position.copy()

    for t, (tow, sats, pr_raw, wk) in enumerate(epochs_data):
        ns = len(sats)
        rx_est = approx0.astype(np.float64, copy=True)

        for _pass in range(2):
            pr_tmp = np.zeros(ns, dtype=np.float64)
            w_tmp = np.zeros(ns, dtype=np.float64)
            sat_buf = np.zeros((ns, 3), dtype=np.float64)

            for si, sid in enumerate(sats):
                if rtk_meas is not None:
                    row = rtk_meas.get((wk, round(float(tow), 4), sid))
                    if row is None:
                        continue
                    sat_buf[si, 0] = row["satx"]
                    sat_buf[si, 1] = row["saty"]
                    sat_buf[si, 2] = row["satz"]
                    pr_clean = (
                        row["prange_m"]
                        - row["iono_m"]
                        - row["trop_m"]
                        - row["sat_clk_m"]
                    )
                    pr_tmp[si] = pr_clean
                    if "el_rad" in row:
                        sin_el = max(math.sin(row["el_rad"]), 0.1)
                        w_tmp[si] = sin_el * sin_el
                    else:
                        w_tmp[si] = 0.5

            idx = np.flatnonzero(w_tmp > 0)
            if idx.size >= 4:
                st, _ = wls_position(
                    sat_buf[idx, :].reshape(-1), pr_tmp[idx], w_tmp[idx], 25, 1e-9,
                )
                rx_est = np.asarray(st[:3], dtype=np.float64).copy()

        sat_ecef[t, :ns] = sat_buf
        pseudorange[t, :ns] = pr_tmp
        weights[t, :ns] = w_tmp

    # WLS (single-clock for position seed; multi-clock handled by FGO)
    wls_state = np.zeros((n_epoch, 4), dtype=np.float64)
    for t2 in range(n_epoch):
        w = weights[t2]
        idx = np.flatnonzero(w > 0)
        if idx.size < 4:
            continue
        st, _ = wls_position(
            sat_ecef[t2, idx].reshape(-1), pseudorange[t2, idx], w[idx], 25, 1e-9,
        )
        wls_state[t2] = st

    # Compute Doppler-derived motion displacement from RTKLIB receiver velocity
    # RTKLIB pntpos estvel() uses Sagnac-corrected Doppler → accurate ECEF velocity
    motion_disp = None
    if use_doppler and motion_sigma_m > 0 and rtk_meas is not None:
        motion_disp = np.zeros((n_epoch, 3), dtype=np.float64)
        for t_d in range(n_epoch - 1):
            tow_d = epochs_data[t_d][0]
            wk_d = epochs_data[t_d][3]
            sats_d = epochs_data[t_d][1]
            # Get receiver velocity from any satellite row at this epoch
            rx_vel = None
            for sid in sats_d:
                row = rtk_meas.get((wk_d, round(float(tow_d), 4), sid))
                if row is not None and "rx_vx" in row:
                    rx_vel = np.array([row["rx_vx"], row["rx_vy"], row["rx_vz"]])
                    break
            if rx_vel is None:
                continue
            dt_ep = epochs_data[t_d + 1][0] - epochs_data[t_d][0]
            if 0 < dt_ep < 10:
                motion_disp[t_d] = rx_vel * dt_ep

    # FGO — choose between standard and VD solver
    if use_vd:
        # --- Velocity-Doppler (VD) solver ---
        # State: [x,y,z, vx,vy,vz, c0,...,c_{K-1}, drift] -> 7 + n_clock columns
        fgo_state = np.zeros((n_epoch, 7 + n_clock), dtype=np.float64)
        fgo_state[:, :3] = wls_state[:, :3]  # position from WLS
        fgo_state[:, 6] = wls_state[:, 3]    # GPS clock bias

        # Store RTKLIB receiver velocity and initialize velocity state
        rx_vel_per_epoch = np.zeros((n_epoch, 3), dtype=np.float64)
        if rtk_meas is not None:
            for t_v in range(n_epoch):
                tow_v = epochs_data[t_v][0]
                wk_v = epochs_data[t_v][3]
                sats_v = epochs_data[t_v][1]
                for sid in sats_v:
                    row = rtk_meas.get((wk_v, round(float(tow_v), 4), sid))
                    if row is not None and "rx_vx" in row:
                        rv = np.array([row["rx_vx"], row["rx_vy"], row["rx_vz"]])
                        fgo_state[t_v, 3] = rv[0]
                        fgo_state[t_v, 4] = rv[1]
                        fgo_state[t_v, 5] = rv[2]
                        rx_vel_per_epoch[t_v] = rv
                        break

        # Build dt array (inter-epoch time differences)
        dt_arr = np.zeros(n_epoch, dtype=np.float64)
        for t_dt in range(n_epoch - 1):
            dt_arr[t_dt] = epochs_data[t_dt + 1][0] - epochs_data[t_dt][0]
            if dt_arr[t_dt] <= 0 or dt_arr[t_dt] > 30:
                dt_arr[t_dt] = 1.0  # fallback

        # Build satellite velocity and Doppler pseudorange-rate arrays
        sat_vel_arr = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
        doppler_arr = np.zeros((n_epoch, max_sats), dtype=np.float64)
        doppler_w_arr = np.zeros((n_epoch, max_sats), dtype=np.float64)

        if rtk_meas is not None:
            for t_d in range(n_epoch):
                tow_d = epochs_data[t_d][0]
                wk_d = epochs_data[t_d][3]
                sats_d = epochs_data[t_d][1]
                # Get receiver velocity for this epoch
                rx_vel = np.zeros(3)
                for sid in sats_d:
                    row = rtk_meas.get((wk_d, round(float(tow_d), 4), sid))
                    if row is not None and "rx_vx" in row:
                        rx_vel = np.array([row["rx_vx"], row["rx_vy"], row["rx_vz"]])
                        break

                for si, sid in enumerate(sats_d):
                    row = rtk_meas.get((wk_d, round(float(tow_d), 4), sid))
                    if row is None or "svx" not in row:
                        continue
                    sv = np.array([row["svx"], row["svy"], row["svz"]])
                    sat_vel_arr[t_d, si] = sv
                    # Compute pseudorange-rate (Doppler range rate):
                    # Kernel convention: e = (rx - sat) / r (sat-to-rx)
                    # pred = dot(e, sv - rv) + drift
                    # So doppler_obs should be dot((rx-sat)/r, sv-rv)
                    sat_pos = sat_ecef[t_d, si]
                    rx_pos = fgo_state[t_d, :3]
                    diff = rx_pos - sat_pos  # rx - sat (sat-to-rx direction)
                    rng = np.linalg.norm(diff)
                    if rng < 1e3:
                        continue
                    unit_vec = diff / rng
                    range_rate = np.dot(sv - rx_vel, unit_vec)
                    doppler_arr[t_d, si] = range_rate
                    # Weight same as pseudorange (sin^2 elevation)
                    doppler_w_arr[t_d, si] = weights[t_d, si]

        iters, mse_pr = fgo_gnss_lm_vd(
            sat_ecef, pseudorange, weights, fgo_state,
            sys_kind=sys_kind_arr,
            n_clock=n_clock,
            motion_sigma_m=motion_sigma_m,
            clock_drift_sigma_m=clock_drift_sigma_m,
            max_iter=fgo_iters, tol=1e-7,
            sat_vel=sat_vel_arr,
            doppler=doppler_arr,
            doppler_weights=doppler_w_arr,
            dt=dt_arr,
        )
    else:
        # --- Standard solver ---
        fgo_state = np.zeros((n_epoch, 3 + n_clock), dtype=np.float64)
        fgo_state[:, :4] = wls_state  # xyz + GPS clock from WLS
        iters, mse_pr = fgo_gnss_lm(
            sat_ecef, pseudorange, weights, fgo_state,
            sys_kind=sys_kind_arr,
            n_clock=n_clock,
            motion_sigma_m=motion_sigma_m, max_iter=fgo_iters, tol=1e-7,
            motion_displacement=motion_disp,
        )

    # Compute errors (skip epochs where WLS failed to converge)
    err_wls_2d, err_fgo_2d = [], []
    err_wls_3d, err_fgo_3d = [], []
    for t3 in range(n_epoch):
        if np.linalg.norm(wls_state[t3, :3]) < 1e3:
            continue  # WLS didn't converge
        tow = epochs_data[t3][0]
        err_wls_2d.append(_nearest_ref_error_2d(tow, ref_tow, ref_ecef, wls_state[t3]))
        err_fgo_2d.append(_nearest_ref_error_2d(tow, ref_tow, ref_ecef, fgo_state[t3]))
        err_wls_3d.append(_nearest_ref_error_3d(tow, ref_tow, ref_ecef, wls_state[t3]))
        err_fgo_3d.append(_nearest_ref_error_3d(tow, ref_tow, ref_ecef, fgo_state[t3]))

    rms_wls_2d = float(np.sqrt(np.mean(np.square(err_wls_2d))))
    rms_fgo_2d = float(np.sqrt(np.mean(np.square(err_fgo_2d))))
    rms_wls_3d = float(np.sqrt(np.mean(np.square(err_wls_3d))))
    rms_fgo_3d = float(np.sqrt(np.mean(np.square(err_fgo_3d))))
    p95_fgo_2d = float(np.percentile(err_fgo_2d, 95))

    return {
        "run": str(run_dir.relative_to(run_dir.parents[2])) if run_dir.parents[2].exists() else run_dir.name,
        "n_epoch": n_epoch,
        "max_sats": max_sats,
        "n_clock": n_clock,
        "fgo_iters": iters,
        "rms_wls_2d": rms_wls_2d,
        "rms_fgo_2d": rms_fgo_2d,
        "rms_wls_3d": rms_wls_3d,
        "rms_fgo_3d": rms_fgo_3d,
        "p95_fgo_2d": p95_fgo_2d,
        "export_spp": str(export_spp) if export_spp else "(off)",
        "multi_gnss": multi_gnss,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ppc-root", type=Path, default=None, help="PPC-Dataset root dir")
    p.add_argument("--run", type=str, default="tokyo/run1", help="Run subdir (e.g. tokyo/run1)")
    p.add_argument("--all", action="store_true", help="Run all 6 PPC runs")
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--elev", type=float, default=15.0)
    p.add_argument("--motion-sigma-m", type=float, default=0.0)
    p.add_argument("--fgo-iters", type=int, default=8)
    p.add_argument("--no-rtklib", action="store_true", help="Skip RTKLIB export_spp_meas")
    p.add_argument("--doppler", action="store_true", help="Use Doppler velocity for motion model")
    p.add_argument("--vd", action="store_true",
                   help="Use fgo_gnss_lm_vd solver (velocity-Doppler)")
    p.add_argument("--clock-drift-sigma-m", type=float, default=1.0,
                   help="Clock drift sigma for VD solver (default: 1.0)")
    p.add_argument("--multi-gnss", action="store_true",
                   help="Use GPS+Galileo+QZSS with ISB (n_clock=3)")
    args = p.parse_args()

    ppc_root = args.ppc_root or _DEFAULT_PPC_ROOT
    if not ppc_root.is_dir():
        print(f"PPC-Dataset not found: {ppc_root}")
        print("Download from: https://github.com/taroz/PPC-Dataset")
        sys.exit(1)

    export_spp = None if args.no_rtklib else _default_export_spp_meas()

    runs = ALL_RUNS if args.all else [args.run]

    results = []
    for run_name in runs:
        run_dir = ppc_root / run_name
        if not run_dir.is_dir():
            print(f"  SKIP {run_name}: not found")
            continue
        try:
            r = run_fgo_on_ppc(
                run_dir,
                max_epochs=args.max_epochs,
                el_mask_deg=args.elev,
                motion_sigma_m=args.motion_sigma_m,
                fgo_iters=args.fgo_iters,
                export_spp=export_spp,
                use_doppler=args.doppler,
                multi_gnss=args.multi_gnss,
                use_vd=args.vd,
                clock_drift_sigma_m=args.clock_drift_sigma_m,
            )
            results.append(r)
            gnss_tag = f"clk={r['n_clock']}" if r.get("multi_gnss") else ""
            print(f"  {r['run']:20s}  epochs={r['n_epoch']:4d}  sats={r['max_sats']:2d}  "
                  f"WLS 2D={r['rms_wls_2d']:7.2f}m  FGO 2D={r['rms_fgo_2d']:7.2f}m  "
                  f"P95={r['p95_fgo_2d']:7.2f}m  3D={r['rms_fgo_3d']:7.2f}m  {gnss_tag}")
        except Exception as e:
            print(f"  {run_name}: ERROR {e}")

    if len(results) > 1:
        print()
        print("Summary:")
        all_wls = [r["rms_wls_2d"] for r in results]
        all_fgo = [r["rms_fgo_2d"] for r in results]
        print(f"  Avg WLS 2D: {np.mean(all_wls):.2f}m  Avg FGO 2D: {np.mean(all_fgo):.2f}m")


if __name__ == "__main__":
    main()
