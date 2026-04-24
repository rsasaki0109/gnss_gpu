#!/usr/bin/env python3
# ruff: noqa: E402
"""Per-epoch weighted-prior FGO smoother for libgnss++ trajectories.

Builds a GTSAM position-only factor graph with:

- Per-epoch prior factors whose sigma depends on the source
  (RTK-FIXED: 3 cm, RTK-FLOAT: 30 cm, SPP: 5 m, missing: 50 m).
- Between-factors using IMU-derived adjacent-position deltas (causal
  strapdown propagated between rover epochs) with 1 m sigma.

The endpoint-only prior mode of ``python/gnss_gpu/local_fgo.py`` is
ineffective when initial positions already match motion deltas -- the
optimizer converges to the identity. Per-epoch priors with a source-
weighted sigma give the FGO something to solve: SPP points get pulled
toward neighbouring RTK anchors while still having to respect the
IMU-measured motion.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

try:
    import gtsam
except ImportError:
    gtsam = None  # type: ignore

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.ppc_score import score_ppc2024

_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_RTK_DIR = _SCRIPT_DIR / "results" / "libgnss_rtk_pos_best"
_SPP_DIR = _SCRIPT_DIR / "results" / "libgnss_spp_pos"

_RUNS = (("tokyo", "run1"), ("tokyo", "run2"), ("tokyo", "run3"),
         ("nagoya", "run1"), ("nagoya", "run2"), ("nagoya", "run3"))

SIGMA_FIXED = 0.03
SIGMA_FLOAT = 0.30
SIGMA_SPP = 5.0
SIGMA_MISSING = 50.0
SIGMA_MOTION = 0.5  # per-epoch (0.2 s) motion-delta sigma


def _load_pos(path: Path) -> dict[float, tuple[np.ndarray, int]]:
    out: dict[float, tuple[np.ndarray, int]] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            tow = round(float(parts[1]), 1)
            ecef = np.array(
                [float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=np.float64,
            )
            status = int(parts[8])
            out[tow] = (ecef, status)
    return out


def _load_reference(path: Path) -> list[tuple[float, np.ndarray]]:
    out: list[tuple[float, np.ndarray]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tow = round(float(row[0]), 1)
            out.append(
                (
                    tow,
                    np.array(
                        [float(row[5]), float(row[6]), float(row[7])],
                        dtype=np.float64,
                    ),
                )
            )
    return out


def _load_imu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tows: list[float] = []
    acc: list[list[float]] = []
    gyro: list[list[float]] = []
    d2r = math.pi / 180.0
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tows.append(float(row[0]))
            acc.append([float(row[2]), float(row[3]), float(row[4])])
            gyro.append([float(row[5]) * d2r, float(row[6]) * d2r, float(row[7]) * d2r])
    return np.asarray(tows), np.asarray(acc), np.asarray(gyro)


def _motion_deltas_from_pos(init: np.ndarray) -> np.ndarray:
    """Simple baseline: use adjacent initial-position differences."""
    return np.diff(init, axis=0)


def _motion_deltas_from_rtk(
    ref: list[tuple[float, np.ndarray]],
    rtk: dict[float, tuple[np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Produce motion deltas and per-segment sigma from RTK fixes.

    Segment between adjacent rover epochs is modelled as
    ``delta[i] = v[i] * (t[i+1] - t[i])`` where ``v[i]`` is the velocity
    interpolated from the two nearest RTK fixes. When an epoch has no
    bracketing RTK fixes the sigma is loosened so the graph relaxes that
    segment toward the prior-only solution.
    """
    n = len(ref)
    rtk_tows = np.array(sorted(rtk.keys()), dtype=np.float64)
    rtk_pos = np.array([rtk[t][0] for t in rtk_tows], dtype=np.float64)
    # Linearly interpolate position as a function of time, then derive per-epoch velocity.
    deltas = np.zeros((n - 1, 3), dtype=np.float64)
    sigmas = np.full(n - 1, SIGMA_MOTION, dtype=np.float64)
    for i in range(n - 1):
        t0 = ref[i][0]
        t1 = ref[i + 1][0]
        idx = int(np.searchsorted(rtk_tows, t1))
        if idx >= 1 and idx < rtk_tows.size:
            # bracketing case
            ta = rtk_tows[idx - 1]
            tb = rtk_tows[idx]
            gap = float(tb - ta)
            if gap > 0 and gap <= 30.0:
                v = (rtk_pos[idx] - rtk_pos[idx - 1]) / gap
                deltas[i] = v * (t1 - t0)
                # sigma grows with interpolation gap
                sigmas[i] = max(SIGMA_MOTION, 0.05 * gap)
                continue
        # fallback: use local init diff via None -> zero delta with big sigma
        deltas[i] = np.zeros(3, dtype=np.float64)
        sigmas[i] = 20.0
    return deltas, sigmas


def _build_initial(
    ref: list[tuple[float, np.ndarray]],
    rtk: dict[float, tuple[np.ndarray, int]],
    spp: dict[float, tuple[np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(ref)
    init = np.zeros((n, 3), dtype=np.float64)
    src = np.full(n, -1, dtype=np.int32)  # -1 missing, 0 SPP, 3 RTK-FLOAT, 4 RTK-FIXED
    for i, (tow, _t) in enumerate(ref):
        if tow in rtk:
            init[i] = rtk[tow][0]
            src[i] = rtk[tow][1]
        elif tow in spp:
            init[i] = spp[tow][0]
            src[i] = 0
    # fill missing by neighbour search
    for i in range(n):
        if src[i] == -1:
            for j in range(1, n):
                if i - j >= 0 and src[i - j] != -1:
                    init[i] = init[i - j]
                    break
                if i + j < n and src[i + j] != -1:
                    init[i] = init[i + j]
                    break
    return init, src


def _prior_sigma_for(src: int) -> float:
    if src == 4:
        return SIGMA_FIXED
    if src == 3:
        return SIGMA_FLOAT
    if src == 0:
        return SIGMA_SPP
    return SIGMA_MISSING


def _fgo_smooth_window(
    init: np.ndarray,
    src: np.ndarray,
    motion_deltas: np.ndarray,
    motion_sigmas: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    assert gtsam is not None
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    keys = [gtsam.symbol("x", i) for i in range(start, end + 1)]
    for local_i, k in enumerate(keys):
        idx = start + local_i
        values.insert(k, gtsam.Point3(*init[idx]))
        sigma = _prior_sigma_for(int(src[idx]))
        noise = gtsam.noiseModel.Isotropic.Sigma(3, sigma)
        graph.add(gtsam.PriorFactorPoint3(k, gtsam.Point3(*init[idx]), noise))
    for local_i in range(len(keys) - 1):
        idx = start + local_i
        delta = motion_deltas[idx]
        msigma = float(motion_sigmas[idx])
        noise = gtsam.noiseModel.Isotropic.Sigma(3, msigma)
        graph.add(
            gtsam.BetweenFactorPoint3(
                keys[local_i], keys[local_i + 1], gtsam.Point3(*delta), noise,
            )
        )
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(20)
    params.setRelativeErrorTol(1e-5)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()
    out = np.zeros_like(init[start:end + 1])
    for local_i, k in enumerate(keys):
        out[local_i] = np.asarray(result.atPoint3(k), dtype=np.float64)
    return out


def _smooth_full(
    init: np.ndarray,
    src: np.ndarray,
    motion_deltas: np.ndarray,
    motion_sigmas: np.ndarray,
    window_size: int = 400,
    overlap: int = 50,
) -> np.ndarray:
    n = init.shape[0]
    out = init.copy()
    w_start = 0
    while w_start < n:
        w_end = min(w_start + window_size - 1, n - 1)
        if w_end - w_start < 5:
            break
        smoothed = _fgo_smooth_window(out, src, motion_deltas, motion_sigmas, w_start, w_end)
        out[w_start:w_end + 1] = smoothed
        if w_end >= n - 1:
            break
        w_start += max(window_size - overlap, 1)
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description="FGO smoother on libgnss++ + SPP")
    parser.add_argument("--window-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    if gtsam is None:
        raise SystemExit("gtsam is not available — install gtsam Python bindings")

    agg_pass = agg_total = 0.0
    print(f"{'Run':<15} {'n_ref':>6} {'init %':>7} {'fgo %':>7} {'Δ':>6}")
    for city, run in _RUNS:
        seg = _DATA_ROOT / city / run
        ref = _load_reference(seg / "reference.csv")
        truth = np.array([r[1] for r in ref], dtype=np.float64)
        rtk = _load_pos(_RTK_DIR / f"{city}_{run}_best.pos")
        spp = _load_pos(_SPP_DIR / f"{city}_{run}_full.pos")
        init, src = _build_initial(ref, rtk, spp)
        motion_deltas, motion_sigmas = _motion_deltas_from_rtk(ref, rtk)
        out = _smooth_full(init, src, motion_deltas, motion_sigmas,
                           window_size=int(args.window_size),
                           overlap=int(args.overlap))
        s_init = score_ppc2024(init, truth)
        s_fgo = score_ppc2024(out, truth)
        print(
            f"  {city}/{run:<7} {len(ref):>6} "
            f"{s_init.score_pct:>6.2f}% {s_fgo.score_pct:>6.2f}% "
            f"{s_fgo.score_pct - s_init.score_pct:+6.2f}"
        )
        agg_pass += s_fgo.pass_distance_m
        agg_total += s_fgo.total_distance_m

    print()
    print(f"FGO-smoothed honest aggregate : {100 * agg_pass / agg_total:6.2f}%")
    print("(previous best: 40.04% / TURING: 85.60%)")


if __name__ == "__main__":
    _main()
