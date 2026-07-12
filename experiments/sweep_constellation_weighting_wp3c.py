#!/usr/bin/env python3
"""TASK_E (WP3c) work item 2b: tune per-constellation sigma scaling on data.

Compares the task's a-priori starting weights (BDS=1.5x, GLO=2.0x sigma)
against sigma-scale factors *derived from* the work item 3 per-constellation
WLS residual RMS breakdown (``experiments/diag_constellation_wiring_and_residuals.py``),
on the same representative 2000-epoch GRECJ window used by the elevation-mask
sweep (``experiments/sweep_elevation_mask_wp3c.py``), all with the chosen
20 deg elevation mask already applied.

Usage:
    set PYTHONPATH=python
    set PYTHONUNBUFFERED=1
    python experiments/sweep_constellation_weighting_wp3c.py
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

from validate_fgo_ppc import DEFAULT_CONSTELLATION_SIGMA_SCALE, run_fgo_on_ppc_native  # noqa: E402

RUN_DIR = Path("E:/datasets/PPC-Dataset-data/tokyo/run1")
MAX_EPOCHS = 2000
CHUNK_EPOCHS = 250
ELEVATION_MASK_DEG = 20.0

# Derived from the work item 3 diagnostic's "BEFORE" per-constellation WLS
# residual RMS (2000-epoch window, no mask/weighting): rms_C=24.99,
# rms_E=30.09, rms_G=28.97, rms_J=25.07, rms_R=36.52 (metres); scale = rms /
# rms_G. Unlike the a-priori DEFAULT_CONSTELLATION_SIGMA_SCALE, this treats
# BeiDou/QZSS as *cleaner* than GPS on this dataset (scale < 1.0) and
# down-weights GLONASS only mildly (~1.26x, not 2x).
DATA_TUNED_SIGMA_SCALE = {
    "C": 24.99 / 28.97,
    "E": 30.09 / 28.97,
    "G": 1.0,
    "J": 25.07 / 28.97,
    "R": 36.52 / 28.97,
}


def _run(label: str, sigma_scale: dict[str, float] | None) -> dict:
    t0 = time.time()
    r = run_fgo_on_ppc_native(
        RUN_DIR,
        max_epochs=MAX_EPOCHS,
        motion_sigma_m=1.0,
        fgo_iters=8,
        doppler_mode="off",
        systems=("G", "R", "E", "C", "J"),
        chunk_epochs=CHUNK_EPOCHS,
        elevation_mask_deg=ELEVATION_MASK_DEG,
        constellation_weighting=True,
        constellation_sigma_scale=sigma_scale,
    )
    dt = time.time() - t0
    print(
        f"  {label:24s}  WLS2D={r['rms_wls_2d']:7.2f}m  FGO2D={r['rms_fgo_2d']:7.2f}m  "
        f"FGO3D={r['rms_fgo_3d']:7.2f}m  mse={r['fgo_mse_pr']:.4g}  ({dt:.0f}s)",
        flush=True,
    )
    return r


def main() -> None:
    print(f"Run: {RUN_DIR}  max_epochs={MAX_EPOCHS}  elevation_mask={ELEVATION_MASK_DEG} deg\n")
    print(f"  a-priori DEFAULT_CONSTELLATION_SIGMA_SCALE = {DEFAULT_CONSTELLATION_SIGMA_SCALE}")
    print(f"  data-tuned DATA_TUNED_SIGMA_SCALE          = "
          f"{ {k: round(v, 3) for k, v in DATA_TUNED_SIGMA_SCALE.items()} }\n")

    r_default = _run("default (BDS1.5/GLO2.0)", DEFAULT_CONSTELLATION_SIGMA_SCALE)
    r_tuned = _run("data-tuned (from WLS RMS)", DATA_TUNED_SIGMA_SCALE)

    print("\nSummary:")
    print(f"  default weights:    FGO2D={r_default['rms_fgo_2d']:.2f}m  FGO3D={r_default['rms_fgo_3d']:.2f}m")
    print(f"  data-tuned weights: FGO2D={r_tuned['rms_fgo_2d']:.2f}m  FGO3D={r_tuned['rms_fgo_3d']:.2f}m")


if __name__ == "__main__":
    main()
