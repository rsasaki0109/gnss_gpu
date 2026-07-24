#!/usr/bin/env python3
"""Temporary diagnostic for TASK_C2 iters=-1 bug."""
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO / "experiments"))

import numpy as np
from validate_fgo_ppc import run_fgo_on_ppc_native, _chunk_ranges
from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.fgo import fgo_gnss_lm_vd
from gnss_gpu import wls_position

run_dir = Path(r"E:/datasets/PPC-Dataset-data/tokyo/run1")

for ms in (0.0, 1.0):
    r = run_fgo_on_ppc_native(
        run_dir,
        max_epochs=120,
        motion_sigma_m=ms,
        fgo_iters=8,
        clock_drift_sigma_m=1.0,
        doppler_mode="off",
    )
    print(
        f"max_ep=120 motion_sigma={ms}: iters={r['fgo_iters']} mse={r['fgo_mse_pr']:.4g} "
        f"WLS={r['rms_wls_2d']:.2f} FGO={r['rms_fgo_2d']:.2f}"
    )

data = PPCDatasetLoader(run_dir).load_experiment_data(max_epochs=2000)
n_epoch = int(data["n_epochs"])
print(f"Loaded {n_epoch} epochs for chunk test")
times = np.asarray(data["times"], dtype=np.float64)
sat_counts = np.asarray(data["satellite_counts"], dtype=np.int32)
max_sats = int(sat_counts.max())
sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
weights = np.zeros((n_epoch, max_sats), dtype=np.float64)
for t in range(n_epoch):
    ns = int(sat_counts[t])
    sat_ecef[t, :ns] = np.asarray(data["sat_ecef"][t], dtype=np.float64)
    pseudorange[t, :ns] = np.asarray(data["pseudoranges"][t], dtype=np.float64)
    weights[t, :ns] = np.asarray(data["weights"][t], dtype=np.float64)
wls_state = np.zeros((n_epoch, 4), dtype=np.float64)
for t in range(n_epoch):
    w = weights[t]
    idx = np.flatnonzero(w > 0)
    if idx.size < 4:
        continue
    st, _ = wls_position(
        sat_ecef[t, idx].reshape(-1), pseudorange[t, idx], w[idx], 25, 1e-9
    )
    wls_state[t] = st
fgo_state = np.zeros((n_epoch, 8), dtype=np.float64)
fgo_state[:, :3] = wls_state[:, :3]
fgo_state[:, 6] = wls_state[:, 3]
dt_arr = np.zeros(n_epoch, dtype=np.float64)
fallback_dt = float(data.get("dt", 0.2))
for t in range(n_epoch - 1):
    dt_arr[t] = float(times[t + 1] - times[t])
    if dt_arr[t] <= 0 or dt_arr[t] > 30:
        dt_arr[t] = fallback_dt
for ms in (0.0, 1.0):
    st = fgo_state.copy()
    iters, mse = fgo_gnss_lm_vd(
        sat_ecef,
        pseudorange,
        weights,
        st,
        n_clock=1,
        motion_sigma_m=ms,
        clock_drift_sigma_m=1.0,
        max_iter=8,
        tol=1e-7,
        dt=dt_arr,
    )
    print(
        f"direct 2000ep motion_sigma={ms}: n_state={n_epoch * 8} "
        f"iters={iters} mse={mse:.4g}"
    )
for start, end in _chunk_ranges(11676, 2000):
    seg_n = end - start
    print(f"chunk [{start}:{end}] n_epoch={seg_n} n_state={seg_n * 8}")
