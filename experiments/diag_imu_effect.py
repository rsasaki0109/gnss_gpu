#!/usr/bin/env python3
"""D3 sanity check: does the IMU prior actually perturb the VD solve?

Loads tokyo/run1 once, builds the first 1000-epoch chunk exactly like
``validate_fgo_ppc.run_fgo_on_ppc_native`` (Doppler in-repo + Huber), then
runs ``fgo_gnss_lm_vd`` with imu_position_sigma_m/imu_velocity_sigma_mps at
0 (disabled) vs the D3 defaults (5.0 m / 2.0 m/s) and diffs the resulting
state + mse_pr, plus reports whether ``imu_delta_p``/``imu_delta_v`` slices
are actually non-None/non-degenerate for this chunk.

Usage:
    set PYTHONPATH=python
    python experiments/diag_imu_effect.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.fgo import fgo_gnss_lm_vd  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gsdc2023_imu import imu_preintegration_segment_with_bias_jacobians  # noqa: E402
from ppc_imu_adapter import load_ppc_imu_preintegration  # noqa: E402
from validate_fgo_ppc import (  # noqa: E402
    C_LIGHT,
    _doppler_hz_to_range_rate,
    _per_satellite_wavelength_m,
)

RUN_DIR = Path("E:/datasets/PPC-Dataset-data/tokyo/run1")
N_EPOCH = 300


def main() -> None:
    t0 = time.time()
    loader = PPCDatasetLoader(RUN_DIR)
    data = loader.load_experiment_data(include_sat_velocity=True, max_epochs=N_EPOCH)
    print(f"Loaded {data['n_epochs']} epochs in {time.time() - t0:.1f}s")

    n_epoch = int(data["n_epochs"])
    times = np.asarray(data["times"], dtype=np.float64)
    sat_counts = np.asarray(data["satellite_counts"], dtype=np.int32)
    max_sats = int(sat_counts.max())
    used_prns = data["used_prns"]

    sat_ecef = np.zeros((n_epoch, max_sats, 3))
    pseudorange = np.zeros((n_epoch, max_sats))
    weights = np.zeros((n_epoch, max_sats))
    sat_vel = np.zeros((n_epoch, max_sats, 3))
    doppler_rr = np.zeros((n_epoch, max_sats))
    doppler_w = np.zeros((n_epoch, max_sats))
    sat_clock_drift = np.zeros((n_epoch, max_sats))
    wavelength_arr = _per_satellite_wavelength_m(used_prns, max_sats)
    for t in range(n_epoch):
        ns = int(sat_counts[t])
        sat_ecef[t, :ns] = np.asarray(data["sat_ecef"][t], dtype=np.float64)
        pseudorange[t, :ns] = np.asarray(data["pseudoranges"][t], dtype=np.float64)
        weights[t, :ns] = np.asarray(data["weights"][t], dtype=np.float64)
        sat_vel[t, :ns] = np.asarray(data["sat_velocity"][t], dtype=np.float64)
        dop_hz = np.asarray(data["doppler_hz"][t], dtype=np.float64)
        doppler_rr[t, :ns] = _doppler_hz_to_range_rate(dop_hz, wavelength_m=wavelength_arr[t, :ns])
        valid = np.isfinite(dop_hz) & (dop_hz != 0.0) & np.isfinite(wavelength_arr[t, :ns])
        doppler_w[t, :ns] = np.where(valid, weights[t, :ns], 0.0)
        clk_drift = np.asarray(data["clock_drift"][t], dtype=np.float64)
        sat_clock_drift[t, :ns] = clk_drift * C_LIGHT

    wls_state = np.zeros((n_epoch, 4))
    for t in range(n_epoch):
        idx = np.flatnonzero(weights[t] > 0)
        if idx.size < 4:
            continue
        st, _ = wls_position(sat_ecef[t, idx].reshape(-1), pseudorange[t, idx], weights[t, idx], 25, 1e-9)
        wls_state[t] = st

    dt_arr = np.zeros(n_epoch)
    for t in range(n_epoch - 1):
        dt_arr[t] = float(times[t + 1] - times[t])
        if dt_arr[t] <= 0 or dt_arr[t] > 30:
            dt_arr[t] = 0.2

    imu_preint = load_ppc_imu_preintegration(RUN_DIR, times, wls_state[:, :3])
    imu_delta_p, imu_delta_v, imu_delta_angle, imu_delta_t, *_rest = (
        imu_preintegration_segment_with_bias_jacobians(imu_preint, 0, n_epoch)
    )
    print(f"\nimu_delta_p is None: {imu_delta_p is None}")
    if imu_delta_p is not None:
        print(f"  imu_delta_p shape={imu_delta_p.shape} "
              f"finite_frac={np.isfinite(imu_delta_p).mean():.3f} "
              f"abs_mean={np.nanmean(np.abs(imu_delta_p)):.4f} "
              f"abs_max={np.nanmax(np.abs(imu_delta_p)):.4f}")
    if imu_delta_v is not None:
        print(f"  imu_delta_v shape={imu_delta_v.shape} "
              f"finite_frac={np.isfinite(imu_delta_v).mean():.3f} "
              f"abs_mean={np.nanmean(np.abs(imu_delta_v)):.4f} "
              f"abs_max={np.nanmax(np.abs(imu_delta_v)):.4f}")

    def make_state():
        st = np.zeros((n_epoch, 8))
        st[:, :3] = wls_state[:, :3]
        st[:, 6] = wls_state[:, 3]
        return st

    common_kwargs = dict(
        n_clock=1, motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        max_iter=8, tol=1e-7, dt=dt_arr,
        sat_vel=sat_vel, doppler=doppler_rr, doppler_weights=doppler_w,
        sat_clock_drift=sat_clock_drift, doppler_huber_k=5.0,
    )

    st_no_imu = make_state()
    iters0, mse0 = fgo_gnss_lm_vd(sat_ecef, pseudorange, weights, st_no_imu, **common_kwargs)
    print(f"\n(no IMU)   iters={iters0} mse={mse0:.6g}")

    for pos_sigma, vel_sigma in ((5.0, 2.0), (0.5, 0.2), (0.05, 0.05), (0.02, 0.1)):
        st_imu = make_state()
        iters1, mse1 = fgo_gnss_lm_vd(
            sat_ecef, pseudorange, weights, st_imu, **common_kwargs,
            imu_delta_p=imu_delta_p, imu_delta_v=imu_delta_v, imu_delta_t=imu_delta_t,
            imu_position_sigma_m=pos_sigma, imu_velocity_sigma_mps=vel_sigma,
        )
        pos_diff = np.linalg.norm(st_imu[:, :3] - st_no_imu[:, :3], axis=1)
        print(
            f"(with IMU, pos_sigma={pos_sigma:.3f} vel_sigma={vel_sigma:.3f}) "
            f"iters={iters1} mse={mse1:.6g}  "
            f"pos_diff mean={pos_diff.mean():.4f}m max={pos_diff.max():.4f}m"
        )


if __name__ == "__main__":
    main()
