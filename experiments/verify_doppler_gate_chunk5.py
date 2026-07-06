#!/usr/bin/env python3
"""TASK_D D1 verification: does robust Doppler gating fix the chunk-5 blow-up?

Loads the PPC tokyo/run1 dataset ONCE (avoids repeated slow reloads), then
runs the native VD solver on chunk 5 (epochs 5000:6000) three ways:
  (a) no Doppler (motion-only baseline)
  (b) in-repo Doppler, no gating/huber (reproduces the WP3a blow-up)
  (b') in-repo Doppler + D1 robust gating (+ optional Huber kernel)

Usage:
    set PYTHONPATH=python
    python experiments/verify_doppler_gate_chunk5.py
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
from validate_fgo_ppc import (  # noqa: E402
    C_LIGHT,
    _doppler_hz_to_range_rate,
    _gate_doppler_outliers_per_epoch,
    _load_ppc_reference,
    _nearest_ref_error_2d,
)

RUN_DIR = Path("E:/datasets/PPC-Dataset-data/tokyo/run1")
CHUNK_START, CHUNK_END = 5000, 6000


def rms_2d(fgo_state, times, ref_tow, ref_ecef) -> float:
    errs = [
        _nearest_ref_error_2d(float(times[t]), ref_tow, ref_ecef, fgo_state[t])
        for t in range(fgo_state.shape[0])
    ]
    return float(np.sqrt(np.mean(np.square(errs))))


def main() -> None:
    t0 = time.time()
    ref_tow, ref_ecef = _load_ppc_reference(RUN_DIR / "reference.csv")
    loader = PPCDatasetLoader(RUN_DIR)
    data = loader.load_experiment_data(include_sat_velocity=True)
    print(f"Loaded {data['n_epochs']} epochs in {time.time() - t0:.1f}s")

    n_epoch_full = int(data["n_epochs"])
    end = min(CHUNK_END, n_epoch_full)
    start = min(CHUNK_START, end)
    n_epoch = end - start
    print(f"Chunk [{start}:{end}) -> {n_epoch} epochs")

    times = np.asarray(data["times"][start:end], dtype=np.float64)
    sat_counts = np.asarray(data["satellite_counts"][start:end], dtype=np.int32)
    max_sats = int(sat_counts.max())

    sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
    weights = np.zeros((n_epoch, max_sats), dtype=np.float64)
    sat_vel = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    doppler_rr = np.zeros((n_epoch, max_sats), dtype=np.float64)
    doppler_w = np.zeros((n_epoch, max_sats), dtype=np.float64)
    sat_clock_drift = np.zeros((n_epoch, max_sats), dtype=np.float64)
    for i, t in enumerate(range(start, end)):
        ns = int(sat_counts[i])
        sat_ecef[i, :ns] = np.asarray(data["sat_ecef"][t], dtype=np.float64)
        pseudorange[i, :ns] = np.asarray(data["pseudoranges"][t], dtype=np.float64)
        weights[i, :ns] = np.asarray(data["weights"][t], dtype=np.float64)
        sat_vel[i, :ns] = np.asarray(data["sat_velocity"][t], dtype=np.float64)
        dop_hz = np.asarray(data["doppler_hz"][t], dtype=np.float64)
        doppler_rr[i, :ns] = _doppler_hz_to_range_rate(dop_hz)
        valid_dop = np.isfinite(dop_hz) & (dop_hz != 0.0)
        doppler_w[i, :ns] = np.where(valid_dop, weights[i, :ns], 0.0)
        clk_drift = np.asarray(data["clock_drift"][t], dtype=np.float64)
        sat_clock_drift[i, :ns] = clk_drift * C_LIGHT

    wls_state = np.zeros((n_epoch, 4), dtype=np.float64)
    for t in range(n_epoch):
        idx = np.flatnonzero(weights[t] > 0)
        if idx.size < 4:
            continue
        st, _ = wls_position(sat_ecef[t, idx].reshape(-1), pseudorange[t, idx], weights[t, idx], 25, 1e-9)
        wls_state[t] = st

    dt_arr = np.zeros(n_epoch, dtype=np.float64)
    for t in range(n_epoch - 1):
        dt_arr[t] = float(times[t + 1] - times[t])
        if dt_arr[t] <= 0 or dt_arr[t] > 30:
            dt_arr[t] = 0.2

    def make_state():
        st = np.zeros((n_epoch, 8), dtype=np.float64)
        st[:, :3] = wls_state[:, :3]
        st[:, 6] = wls_state[:, 3]
        return st

    print(f"\nWLS-only 2D RMS (chunk 5): {rms_2d(wls_state, times, ref_tow, ref_ecef):.2f} m\n")

    # (a) no Doppler
    st_a = make_state()
    iters_a, mse_a = fgo_gnss_lm_vd(
        sat_ecef, pseudorange, weights, st_a,
        n_clock=1, motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        max_iter=8, tol=1e-7, dt=dt_arr,
    )
    print(f"(a) no-Doppler:            iters={iters_a} mse={mse_a:.4g} "
          f"2D RMS={rms_2d(st_a, times, ref_tow, ref_ecef):.2f} m")

    # (b) in-repo Doppler, no gating/huber -- reproduces the blow-up
    st_b = make_state()
    iters_b, mse_b = fgo_gnss_lm_vd(
        sat_ecef, pseudorange, weights, st_b,
        n_clock=1, motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        max_iter=8, tol=1e-7, dt=dt_arr,
        sat_vel=sat_vel, doppler=doppler_rr, doppler_weights=doppler_w,
        sat_clock_drift=sat_clock_drift,
    )
    print(f"(b) doppler, no gate:     iters={iters_b} mse={mse_b:.4g} "
          f"2D RMS={rms_2d(st_b, times, ref_tow, ref_ecef):.2f} m")

    # (b') in-repo Doppler + D1 robust gating
    gated_w, stats = _gate_doppler_outliers_per_epoch(
        sat_ecef, sat_vel, sat_clock_drift, doppler_rr, doppler_w, make_state(),
        gate_sigma=3.0,
    )
    print(f"    gate stats: {stats}")
    st_bp = make_state()
    iters_bp, mse_bp = fgo_gnss_lm_vd(
        sat_ecef, pseudorange, weights, st_bp,
        n_clock=1, motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        max_iter=8, tol=1e-7, dt=dt_arr,
        sat_vel=sat_vel, doppler=doppler_rr, doppler_weights=gated_w,
        sat_clock_drift=sat_clock_drift,
    )
    print(f"(b') doppler + gate:      iters={iters_bp} mse={mse_bp:.4g} "
          f"2D RMS={rms_2d(st_bp, times, ref_tow, ref_ecef):.2f} m")

    # (b'') in-repo Doppler + native Huber kernel only (no gating)
    st_bh = make_state()
    iters_bh, mse_bh = fgo_gnss_lm_vd(
        sat_ecef, pseudorange, weights, st_bh,
        n_clock=1, motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        max_iter=8, tol=1e-7, dt=dt_arr,
        sat_vel=sat_vel, doppler=doppler_rr, doppler_weights=doppler_w,
        sat_clock_drift=sat_clock_drift, doppler_huber_k=5.0,
    )
    print(f"(b'') doppler + huber_k=5: iters={iters_bh} mse={mse_bh:.4g} "
          f"2D RMS={rms_2d(st_bh, times, ref_tow, ref_ecef):.2f} m")

    # (b''') gate + huber combined
    st_bgh = make_state()
    iters_bgh, mse_bgh = fgo_gnss_lm_vd(
        sat_ecef, pseudorange, weights, st_bgh,
        n_clock=1, motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        max_iter=8, tol=1e-7, dt=dt_arr,
        sat_vel=sat_vel, doppler=doppler_rr, doppler_weights=gated_w,
        sat_clock_drift=sat_clock_drift, doppler_huber_k=5.0,
    )
    print(f"(b''') gate + huber_k=5:   iters={iters_bgh} mse={mse_bgh:.4g} "
          f"2D RMS={rms_2d(st_bgh, times, ref_tow, ref_ecef):.2f} m")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
