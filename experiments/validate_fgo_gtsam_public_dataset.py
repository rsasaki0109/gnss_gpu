#!/usr/bin/env python3
"""Validate GPU-assembled GNSS FGO on public gtsam_gnss example RINEX (+ reference).

Data source (open): ``gtsam_gnss/examples/data`` — rover observation, navigation,
and ``reference.csv`` distributed with the `gtsam_gnss` repository / ION example
set (Taro Suzuki et al.). No proprietary logs are used.

Prerequisite: clone or copy that ``examples/data`` tree (this workspace keeps it
under ``gnss_gpu_ws/ref/gtsam_gnss/examples/data``). Pass ``--data-dir`` if yours
is elsewhere.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Repo root: .../gnss_gpu
_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.fgo import fgo_gnss_lm  # noqa: E402
from gtsam_public_dataset import build_public_gtsam_arrays, nearest_ref_error  # noqa: E402


def _default_export_spp_meas() -> Path | None:
    """Auto-detect RTKLIB ``export_spp_meas`` binary."""
    envp = os.environ.get("RTKLIB_EXPORT_SPP_MEAS")
    if envp:
        p = Path(envp)
        if p.is_file():
            return p
    guess = (
        Path(__file__).resolve().parents[2]
        / "ref"
        / "RTKLIB-demo5"
        / "app"
        / "consapp"
        / "rnx2rtkp"
        / "gcc"
        / "export_spp_meas"
    )
    if guess.is_file():
        return guess
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing rover_1Hz.obs, base.nav, reference.csv",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=180,
        help="Window length (dense Hessian cost grows as O(T^3) on the CPU solve)",
    )
    p.add_argument(
        "--fgo-iters",
        type=int,
        default=8,
        help="Gauss–Newton steps (use motion_sigma_m=0 to verify PR-only refinement)",
    )
    p.add_argument(
        "--fgo-tol",
        type=float,
        default=1e-7,
        help="Stop when ‖Δx‖_2 falls below this (full state vector)",
    )
    p.add_argument(
        "--motion-sigma-m",
        type=float,
        default=0.0,
        help="optional position random-walk σ [m]; 0 = PR-only (recommended for first validation)",
    )
    p.add_argument(
        "--fgo-huber-k",
        type=float,
        default=0.0,
        help="Huber IRLS threshold on Mahalanobis |sqrt(w)*res|; 0 = pure WLS (default)",
    )
    p.add_argument(
        "--rtklib-export-spp",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to RTKLIB export_spp_meas binary (auto-detected if omitted)",
    )
    p.add_argument(
        "--no-rtklib-spp-export",
        action="store_true",
        help="Force legacy gnss_gpu SPP path even if export_spp_meas is available",
    )
    p.add_argument(
        "--elev",
        type=float,
        default=15.0,
        help="Elevation mask in degrees (default: 15.0)",
    )
    p.add_argument(
        "--weight-mode",
        choices=["sin2el", "rtklib"],
        default="sin2el",
        help="Weight strategy: sin2el (default, best FGO accuracy) or rtklib (match pntpos exactly)",
    )
    args = p.parse_args()

    # Resolve RTKLIB export_spp_meas path
    if args.no_rtklib_spp_export:
        export_spp: Path | None = None
    elif args.rtklib_export_spp is not None:
        export_spp = args.rtklib_export_spp
    else:
        export_spp = _default_export_spp_meas()

    here = Path(__file__).resolve()
    guess = here.parents[2] / "ref" / "gtsam_gnss" / "examples" / "data"
    data_dir = args.data_dir or guess
    obs_p = data_dir / "rover_1Hz.obs"
    nav_p = data_dir / "base.nav"
    ref_p = data_dir / "reference.csv"
    for need in (obs_p, nav_p, ref_p):
        if not need.is_file():
            print(f"Missing public dataset file: {need}")
            print("Clone https://github.com/taroz/gtsam_gnss and point --data-dir to examples/data")
            sys.exit(1)

    try:
        batch_kw: dict = {}
        if export_spp is not None:
            batch_kw["rtklib_export_spp_exe"] = export_spp
            batch_kw["el_mask_deg"] = args.elev
            batch_kw["weight_mode"] = args.weight_mode
        batch = build_public_gtsam_arrays(data_dir, args.max_epochs, **batch_kw)
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
        sys.exit(1)

    epochs_data = batch.epochs_data
    n_epoch = batch.n_epoch
    max_sats = batch.max_sats
    sat_ecef = batch.sat_ecef
    pseudorange = batch.pseudorange
    weights = batch.weights
    ref_tow, ref_ecef = batch.ref_tow, batch.ref_ecef
    state = np.zeros((n_epoch, 4), dtype=np.float64)

    # Per-epoch CPU WLS matches `wls_position` (reference GNSS least-squares).
    # The GPU `wls_batch` kernel can stop early while a significant nonlinear
    # gradient remains; using it as an FGO warm-start biases the comparison.
    try:
        wls_state = np.zeros((n_epoch, 4), dtype=np.float64)
        for t in range(n_epoch):
            mask = weights[t] > 0
            idx = np.flatnonzero(mask)
            if idx.size < 4:
                continue
            st, _ = wls_position(
                sat_ecef[t, idx, :].reshape(-1),
                pseudorange[t, idx],
                weights[t, idx],
                25,
                1e-9,
            )
            wls_state[t, :] = st
    except Exception as e:
        print("wls_position failed:", e)
        sys.exit(1)

    err_wls = []
    for t in range(n_epoch):
        tow = epochs_data[t][0]
        err_wls.append(
            nearest_ref_error(tow, ref_tow, ref_ecef, wls_state[t]),
        )
    rms_wls = float(np.sqrt(np.mean(np.square(err_wls))))

    state[:, :] = wls_state
    try:
        iters, mse_pr = fgo_gnss_lm(
            sat_ecef,
            pseudorange,
            weights,
            state,
            motion_sigma_m=args.motion_sigma_m,
            max_iter=args.fgo_iters,
            tol=args.fgo_tol,
            huber_k=args.fgo_huber_k,
        )
    except RuntimeError as e:
        print("fgo_gnss_lm:", e)
        sys.exit(1)

    err_fgo = []
    for t in range(n_epoch):
        tow = epochs_data[t][0]
        err_fgo.append(nearest_ref_error(tow, ref_tow, ref_ecef, state[t]))
    rms_fgo = float(np.sqrt(np.mean(np.square(err_fgo))))

    print("GPU-assembled FGO (public gtsam_gnss RINEX subset)")
    print(f"  export_spp  : {export_spp if export_spp else '(off — gnss_gpu SPP)'}")
    print(f"  epochs      : {n_epoch}")
    print(f"  max sats /ep: {max_sats}")
    print(f"  GN iters    : {iters}")
    print(f"  wMSE pr     : {mse_pr:.4f} (weighted, model units)")
    print(f"  RMS 2D vs ref (WLS epoch batch): {rms_wls:.3f} m")
    print(f"  RMS 2D vs ref (FGO + RW prior) : {rms_fgo:.3f} m")
    if rms_fgo < rms_wls - 1e-6:
        print(f"  -> Improvement vs independent WLS: {(1.0 - rms_fgo / rms_wls) * 100:.1f}%")
    elif abs(rms_fgo - rms_wls) <= 1e-3:
        print("  -> FGO matches WLS on this clip (expected for motion_sigma_m=0 near a WLS stationary point).")
    else:
        print("  (FGO degraded vs WLS — try motion_sigma_m=0, smaller --fgo-iters, or future LM line search)")


if __name__ == "__main__":
    main()
