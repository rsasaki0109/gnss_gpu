#!/usr/bin/env python3
"""TASK_D D1 diagnostic: dump per-epoch Doppler residuals for PPC tokyo/run1 chunk 5
(epochs 5000:6000, GPS-only) against WLS-derived receiver velocity, to find the
offending satellites/epochs behind the variant (b) full-run blow-up
(FGO 2D 546m vs 94.5m without Doppler; chunk 5 mse_pr=9.25e5).

Usage:
    set PYTHONPATH=python
    python experiments/diag_doppler_chunk5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from validate_fgo_ppc import _doppler_hz_to_range_rate, C_LIGHT  # noqa: E402

RUN_DIR = Path("E:/datasets/PPC-Dataset-data/tokyo/run1")
CHUNK_START, CHUNK_END = 5000, 6000


def main() -> None:
    loader = PPCDatasetLoader(RUN_DIR)
    data = loader.load_experiment_data(include_sat_velocity=True)
    n_epoch = int(data["n_epochs"])
    print(f"n_epoch total={n_epoch}")
    end = min(CHUNK_END, n_epoch)
    start = min(CHUNK_START, end)

    times = np.asarray(data["times"], dtype=np.float64)
    sat_counts = np.asarray(data["satellite_counts"], dtype=np.int32)

    worst: list[tuple[float, int, str, float, float, float]] = []
    per_sat_abs_resid: dict[str, list[float]] = {}
    n_epochs_checked = 0
    n_dop_obs = 0

    for t in range(start, end):
        ns = int(sat_counts[t])
        sat_ecef = np.asarray(data["sat_ecef"][t], dtype=np.float64)
        pr = np.asarray(data["pseudoranges"][t], dtype=np.float64)
        w = np.asarray(data["weights"][t], dtype=np.float64)
        used_ids = data["used_prns"][t]
        sat_vel = np.asarray(data["sat_velocity"][t], dtype=np.float64)
        clk_drift = np.asarray(data["clock_drift"][t], dtype=np.float64)
        dop_hz = np.asarray(data["doppler_hz"][t], dtype=np.float64)

        idx = np.flatnonzero(w > 0)
        if idx.size < 4:
            continue
        st, _ = wls_position(sat_ecef[idx].reshape(-1), pr[idx], w[idx], 25, 1e-9)
        rx_pos = st[:3]

        # Estimate receiver velocity + clock drift by linear LSQ over Doppler obs,
        # matching the native kernel convention exactly (fgo.cu doppler_prediction_vd):
        #   los = (sat - rx) / |sat - rx|          (rx -> sat direction)
        #   pred = drift + los.(sat_vel - rx_vel) - sat_clk_drift   (Sagnac term dropped, <~few m/s)
        # => rhs = rr_obs - sat_clk_drift - los.sat_vel = drift - los.rx_vel
        rows = []
        rhs = []
        sats_this = []
        for i in range(ns):
            if not np.isfinite(dop_hz[i]) or dop_hz[i] == 0.0:
                continue
            rr_obs = _doppler_hz_to_range_rate(np.array([dop_hz[i]]))[0]
            diff = sat_ecef[i] - rx_pos  # rx -> sat, matches native kernel `los`
            rng = np.linalg.norm(diff)
            if rng < 1e3:
                continue
            los = diff / rng
            sat_clk_drift_ms = clk_drift[i] * C_LIGHT if np.isfinite(clk_drift[i]) else 0.0
            rhs_val = rr_obs - sat_clk_drift_ms - float(np.dot(los, sat_vel[i]))
            # unknowns: [-rx_vx, -rx_vy, -rx_vz, drift] s.t. dot(los, -rx_vel) + drift = rhs
            rows.append([-los[0], -los[1], -los[2], 1.0])
            rhs.append(rhs_val)
            sats_this.append(used_ids[i])
        if len(rows) < 4:
            continue
        A = np.array(rows)
        b = np.array(rhs)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = A @ sol - b
        n_epochs_checked += 1
        for sid, r in zip(sats_this, resid):
            n_dop_obs += 1
            per_sat_abs_resid.setdefault(sid, []).append(abs(float(r)))
            worst.append((float(times[t]), t, sid, float(r), float(b[list(sats_this).index(sid)]), 0.0))

    print(f"Checked {n_epochs_checked} epochs, {n_dop_obs} Doppler obs in [{start}:{end})")
    resid_all = np.array([abs(w[3]) for w in worst])
    print(f"|residual| median={np.median(resid_all):.4g} mean={np.mean(resid_all):.4g} "
          f"p95={np.percentile(resid_all, 95):.4g} max={np.max(resid_all):.4g}")

    print("\nPer-satellite median/mean/max |residual| (m/s), sorted by max:")
    rows_out = []
    for sid, vals in per_sat_abs_resid.items():
        arr = np.array(vals)
        rows_out.append((sid, len(arr), float(np.median(arr)), float(np.mean(arr)), float(np.max(arr))))
    rows_out.sort(key=lambda r: -r[4])
    for sid, n, med, mean, mx in rows_out[:20]:
        print(f"  {sid:4s} n={n:4d} median={med:8.3f} mean={mean:8.3f} max={mx:10.3f}")

    print("\nTop 20 worst individual (time, sat, residual) entries:")
    worst.sort(key=lambda w: -abs(w[3]))
    for tow, t, sid, r, rhs_val, _ in worst[:20]:
        print(f"  t={t:5d} tow={tow:10.2f} sat={sid:4s} resid={r:10.3f} m/s  rhs={rhs_val:10.3f}")


if __name__ == "__main__":
    main()
