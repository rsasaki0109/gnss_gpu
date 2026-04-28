#!/usr/bin/env python3
"""Debug helper: show gnss_gpu preprocessed PR vs RTKLIB ECEF on one epoch (gtsam public clip).

At the RTKLIB single-point solution for epoch 0, computes residual RMS
``sqrt(mean((PR - rho - mean(PR-rho))^2))`` using gnss_gpu ephemeris + pipeline from
``gtsam_public_dataset``. Large RMS ⇒ SPP prep differs from demo5 (not FGO).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
for p in (_REPO, _EXPERIMENTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gtsam_public_dataset import build_public_gtsam_arrays  # noqa: E402

C = 299792458.0
OM = 7.2921151467e-5


def _geom(rx: np.ndarray, sat: np.ndarray) -> float:
    x, y, z = rx
    sx, sy, sz = sat
    dx0, dy0, dz0 = x - sx, y - sy, z - sz
    r0 = math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    th = OM * (r0 / C)
    sxr = sx * math.cos(th) + sy * math.sin(th)
    syr = -sx * math.sin(th) + sy * math.cos(th)
    dx, dy, dz = x - sxr, y - syr, z - sz
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def main() -> None:
    guess = _EXPERIMENTS.parents[1] / "ref" / "gtsam_gnss" / "examples" / "data"
    batch = build_public_gtsam_arrays(guess, 5)
    rtk = np.array([-3810233.3449, 3567867.2770, 3652898.4534])
    t = 0
    w = batch.weights[t]
    idx = np.flatnonzero(w > 0)
    pr = batch.pseudorange[t, idx]
    rho = np.array([_geom(rtk, batch.sat_ecef[t, i]) for i in idx])
    d = pr - rho
    cb = float(np.mean(d))
    rms = float(np.sqrt(np.mean((d - cb) ** 2)))
    print("Epoch 0, sats:", idx.size)
    print("  RTKLIB pos (rnx2rtkp -p0 -sys G first line):", rtk)
    print("  Mean (PR - rho) at RTK [m] (≈ receiver clock):", cb)
    print("  RMS after removing mean [m] (should be ≪10 if models matched demo5):", rms)


if __name__ == "__main__":
    main()
