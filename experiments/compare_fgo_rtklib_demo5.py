#!/usr/bin/env python3
"""Compare gnss_gpu FGO to RTKLIB demo5 ``rnx2rtkp`` on the same public gtsam_gnss data.

Uses ``rover_1Hz.obs`` + ``base.nav`` from ``gtsam_gnss/examples/data``. Builds the
same preprocessed pseudoranges / weights as ``validate_fgo_gtsam_public_dataset``.

RTKLIB must be built separately, e.g.::

    git clone --depth 1 -b demo5 https://github.com/rtklibexplorer/RTKLIB.git \\
        gnss_gpu_ws/ref/RTKLIB-demo5
    make -C gnss_gpu_ws/ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc

Point ``--rnx2rtkp`` to the resulting ``rnx2rtkp`` binary (or set env
``RTKLIB_RNX2RTKP``). The same build produces ``export_spp_meas`` in that
directory; pass ``--rtklib-export-spp`` or set ``RTKLIB_EXPORT_SPP_MEAS`` so FGO
uses RTKLIB ``pntpos``-compatible pseudorange / sat ECEF (see
``gtsam_public_dataset.build_public_gtsam_arrays``).

**Caveat:** Without ``export_spp_meas``, gnss_gpu applies ``correct_pseudoranges``
+ ``Ephemeris`` while demo5 uses ``pntpos`` / ``prange`` — those observation
models differ. Driving the batch from ``export_spp_meas`` aligns gnss_gpu WLS/FGO
with demo5 SPP terms.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent
_REPO = _EXPERIMENTS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.fgo import fgo_gnss_lm  # noqa: E402
from gtsam_public_dataset import build_public_gtsam_arrays, nearest_ref_error  # noqa: E402


def _default_rnx2rtkp() -> Path | None:
    envp = os.environ.get("RTKLIB_RNX2RTKP")
    if envp:
        p = Path(envp)
        if p.is_file():
            return p
    guess = (
        _EXPERIMENTS.parents[1]
        / "ref"
        / "RTKLIB-demo5"
        / "app"
        / "consapp"
        / "rnx2rtkp"
        / "gcc"
        / "rnx2rtkp"
    )
    if guess.is_file():
        return guess
    return None


def _default_export_spp_meas(rnx2rtkp: Path | None) -> Path | None:
    envp = os.environ.get("RTKLIB_EXPORT_SPP_MEAS")
    if envp:
        p = Path(envp)
        if p.is_file():
            return p
    if rnx2rtkp and rnx2rtkp.is_file():
        guess = rnx2rtkp.parent / "export_spp_meas"
        if guess.is_file():
            return guess
    return None


def _run_rnx2rtkp(
    rnx2rtkp: Path,
    obs: Path,
    nav: Path,
    out_pos: Path,
    *,
    mode: int,
    elev_deg: float,
) -> None:
    cmd = [
        str(rnx2rtkp),
        "-p",
        str(mode),
        "-sys",
        "G",
        "-e",
        "-m",
        str(elev_deg),
        "-o",
        str(out_pos),
        str(obs),
        str(nav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _parse_rtklib_pos(path: Path) -> dict[float, tuple[float, float, float, int]]:
    """Map GPS time-of-week (s) -> (x,y,z,Q). SOW rounded to ms for dict keys."""
    out: dict[float, tuple[float, float, float, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                sow_raw = float(parts[1])
                sow = round(sow_raw * 1000.0) / 1000.0
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                q = int(parts[5])
            except ValueError:
                continue
            out[sow] = (x, y, z, q)
    return out


def _rtk_at_tow(
    rtk_sol: dict[float, tuple[float, float, float, int]], tow: float
) -> tuple[float, float, float, int] | None:
    towk = round(float(tow) * 1000.0) / 1000.0
    r = rtk_sol.get(towk)
    if r is not None:
        return r
    for k, v in rtk_sol.items():
        if abs(k - tow) < 0.02:
            return v
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--max-epochs", type=int, default=180)
    p.add_argument("--rnx2rtkp", type=Path, default=None, help="Path to demo5 rnx2rtkp")
    p.add_argument(
        "--rtk-mode",
        type=int,
        default=0,
        help="rnx2rtkp -p mode: 0=single (per-epoch SPP), 2=kinematic",
    )
    p.add_argument("--elev", type=float, default=15.0, help="Elevation mask (deg)")
    p.add_argument(
        "--rtklib-export-spp",
        type=Path,
        default=None,
        help="RTKLIB export_spp_meas binary (default: next to rnx2rtkp or RTKLIB_EXPORT_SPP_MEAS)",
    )
    p.add_argument(
        "--no-rtklib-spp-export",
        action="store_true",
        help="Use gnss_gpu Ephemeris/correct_pseudoranges instead of export_spp_meas",
    )
    p.add_argument("--motion-sigma-m", type=float, default=0.0)
    p.add_argument("--fgo-iters", type=int, default=8)
    p.add_argument("--fgo-tol", type=float, default=1e-7)
    p.add_argument("--fgo-huber-k", type=float, default=0.0)
    p.add_argument(
        "--weight-mode",
        choices=["sin2el", "rtklib"],
        default="sin2el",
        help="Weight strategy: sin2el (default, best FGO accuracy) or rtklib (match pntpos exactly)",
    )
    args = p.parse_args()

    rnx = args.rnx2rtkp or _default_rnx2rtkp()
    if not rnx or not rnx.is_file():
        print("rnx2rtkp not found. Build demo5 (see docstring) or pass --rnx2rtkp / set RTKLIB_RNX2RTKP")
        sys.exit(1)

    guess = _EXPERIMENTS.parents[1] / "ref" / "gtsam_gnss" / "examples" / "data"
    data_dir = args.data_dir or guess

    export_spp = None
    if not args.no_rtklib_spp_export:
        export_spp = args.rtklib_export_spp or _default_export_spp_meas(rnx)

    batch_kw: dict = {}
    if export_spp is not None:
        batch_kw["rtklib_export_spp_exe"] = export_spp
        batch_kw["el_mask_deg"] = args.elev
        batch_kw["weight_mode"] = args.weight_mode
    batch = build_public_gtsam_arrays(data_dir, args.max_epochs, **batch_kw)
    obs_p = data_dir / "rover_1Hz.obs"
    nav_p = data_dir / "base.nav"

    with tempfile.NamedTemporaryFile(suffix=".pos", delete=False) as tf:
        out_pos = Path(tf.name)
    try:
        _run_rnx2rtkp(rnx, obs_p, nav_p, out_pos, mode=args.rtk_mode, elev_deg=args.elev)
        rtk_sol = _parse_rtklib_pos(out_pos)
    finally:
        out_pos.unlink(missing_ok=True)

    state = np.zeros((batch.n_epoch, 4), dtype=np.float64)
    wls_state = np.zeros((batch.n_epoch, 4), dtype=np.float64)
    for t in range(batch.n_epoch):
        w = batch.weights[t]
        mask = w > 0
        idx = np.flatnonzero(mask)
        if idx.size < 4:
            continue
        st, _ = wls_position(
            batch.sat_ecef[t, idx, :].reshape(-1),
            batch.pseudorange[t, idx],
            w[idx],
            25,
            1e-9,
        )
        wls_state[t, :] = st
    state[:, :] = wls_state

    iters, mse_pr = fgo_gnss_lm(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        state,
        motion_sigma_m=args.motion_sigma_m,
        max_iter=args.fgo_iters,
        tol=args.fgo_tol,
        huber_k=args.fgo_huber_k,
    )

    diffs_2d: list[float] = []
    err_fgo: list[float] = []
    err_rtk: list[float] = []
    matched = 0
    rtk_missing = 0
    for t in range(batch.n_epoch):
        tow = batch.epochs_data[t][0]
        err_fgo.append(nearest_ref_error(tow, batch.ref_tow, batch.ref_ecef, state[t]))

        r = _rtk_at_tow(rtk_sol, tow)
        if r is None:
            rtk_missing += 1
            continue
        x, y, z, _q = r
        rtk_arr = np.array([x, y, z, 0.0], dtype=np.float64)
        err_rtk.append(nearest_ref_error(tow, batch.ref_tow, batch.ref_ecef, rtk_arr))
        d = state[t, :2] - rtk_arr[:2]
        diffs_2d.append(float(np.linalg.norm(d)))
        matched += 1

    rms_fgo = float(np.sqrt(np.mean(np.square(err_fgo))))
    rms_rtk = float(np.sqrt(np.mean(np.square(err_rtk)))) if err_rtk else float("nan")
    rms_pair = float(np.sqrt(np.mean(np.square(diffs_2d)))) if diffs_2d else float("nan")
    mean_pair = float(np.mean(diffs_2d)) if diffs_2d else float("nan")

    print("Public gtsam_gnss clip: gnss_gpu FGO vs RTKLIB demo5 rnx2rtkp")
    print(f"  data_dir    : {data_dir}")
    print(f"  epochs      : {batch.n_epoch}")
    print(f"  export_spp  : {export_spp if export_spp else '(off — gnss_gpu SPP)'}")
    print(f"  rnx2rtkp    : {rnx}")
    print(f"  rtk -p mode : {args.rtk_mode}  elev mask: {args.elev} deg  -sys G -e")
    print(f"  FGO iters   : {iters}  wMSE pr: {mse_pr:.4f}  motion_sigma_m: {args.motion_sigma_m}")
    print(f"  RMS 2D vs reference.csv (FGO)     : {rms_fgo:.3f} m")
    print(f"  RMS 2D vs reference.csv (RTKLIB)  : {rms_rtk:.3f} m  (n={len(err_rtk)})")
    print(f"  RMS 2D ||FGO − RTKLIB|| (matched) : {rms_pair:.3f} m  mean={mean_pair:.3f} m (n={matched})")
    if rtk_missing:
        print(f"  (RTKLIB rows missing for {rtk_missing} gnss_gpu epochs — time alignment?)")
    print()
    if not np.isnan(rms_rtk) and rms_fgo > 2.0 * rms_rtk and rms_fgo > 5.0:
        print(
            "  Note: FGO/WLS RMS vs reference is much worse than RTKLIB here — likely gnss_gpu"
            " preprocessing (ephemeris, iono/tropo, code DCBs) does not match RTKLIB yet."
            " Tune ``correct_pseudoranges`` / ephemeris to narrow ||FGO−RTK|| before judging the FGO solver."
        )
    elif not np.isnan(rms_rtk) and abs(rms_fgo - rms_rtk) < 2.0:
        print(
            "  Similar reference RMS: pipelines are in the same ballpark vs reference.csv on this clip."
        )
    else:
        print(
            "  Nonzero ||FGO−RTK|| is normal (different measurement models); use reference RMS"
            " to see whether both are reasonable vs ground truth."
        )


if __name__ == "__main__":
    main()
