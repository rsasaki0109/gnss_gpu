#!/usr/bin/env python3
"""TASK_E (WP3c) work item 1: elevation-mask sweep on a representative epoch range.

Runs the native GRECJ (multi-GNSS) FGO backbone (PR + motion, Doppler off --
matches the D2 baseline in WP3B_REPORT.md so the elevation-mask effect is
isolated from the D1 Doppler-Huber fix) over a representative epoch window
at elevation_mask_deg in {0 (off), 10, 15, 20} and reports WLS/FGO 2D & 3D
RMS plus the fraction of observations masked, so the best cutoff can be
picked before committing to a full run.

Usage:
    set PYTHONPATH=python
    set PYTHONUNBUFFERED=1
    python experiments/sweep_elevation_mask_wp3c.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
for _p in (_REPO, _EXPERIMENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from validate_fgo_ppc import run_fgo_on_ppc_native  # noqa: E402

RUN_DIR = Path("E:/datasets/PPC-Dataset-data/tokyo/run1")
MAX_EPOCHS = 2000  # representative window (~1/6 of the full 11928-epoch run)
CHUNK_EPOCHS = 250  # n_clock=5 perf workaround per WP3B_REPORT.md D2
ELEVATION_SWEEP_DEG = (0.0, 10.0, 15.0, 20.0)


def main() -> None:
    print(f"Run: {RUN_DIR}  max_epochs={MAX_EPOCHS}  chunk_epochs={CHUNK_EPOCHS}\n")
    rows = []
    for el_deg in ELEVATION_SWEEP_DEG:
        t0 = time.time()
        r = run_fgo_on_ppc_native(
            RUN_DIR,
            max_epochs=MAX_EPOCHS,
            motion_sigma_m=1.0,
            fgo_iters=8,
            doppler_mode="off",
            systems=("G", "R", "E", "C", "J"),
            chunk_epochs=CHUNK_EPOCHS,
            elevation_mask_deg=el_deg,
        )
        dt = time.time() - t0
        elev_stats = r.get("elevation_mask_stats") or {}
        n_masked = elev_stats.get("n_obs_masked", 0)
        n_total = elev_stats.get("n_obs_total", 0)
        pct_masked = 100.0 * n_masked / n_total if n_total else 0.0
        row = {
            "elevation_mask_deg": el_deg,
            "n_epoch": r["n_epoch"],
            "median_sats": r["median_sats"],
            "wls_2d": r["rms_wls_2d"],
            "fgo_2d": r["rms_fgo_2d"],
            "fgo_3d": r["rms_fgo_3d"],
            "iters": r["fgo_iters"],
            "mse_pr": r["fgo_mse_pr"],
            "pct_obs_masked": pct_masked,
            "runtime_s": dt,
        }
        rows.append(row)
        print(
            f"  el_mask={el_deg:4.0f} deg  n_ep={row['n_epoch']:5d}  med_sats={row['median_sats']:5.1f}  "
            f"masked={pct_masked:5.1f}%  WLS2D={row['wls_2d']:7.2f}m  FGO2D={row['fgo_2d']:7.2f}m  "
            f"FGO3D={row['fgo_3d']:7.2f}m  iters={row['iters']}  mse={row['mse_pr']:.4g}  "
            f"({dt:.0f}s)",
            flush=True,
        )

    print("\nSummary (best FGO 2D wins):")
    best = min(rows, key=lambda r: r["fgo_2d"])
    for row in rows:
        marker = "  <== best" if row is best else ""
        print(
            f"  el_mask={row['elevation_mask_deg']:4.0f} deg  FGO2D={row['fgo_2d']:7.2f}m  "
            f"FGO3D={row['fgo_3d']:7.2f}m{marker}"
        )


if __name__ == "__main__":
    main()
