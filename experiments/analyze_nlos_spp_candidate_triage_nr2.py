"""Offline triage gate for a 3D-NLOS-rejected SPP candidate on nagoya/run2.

This is the fast go/no-go step (Plan Step 5) BEFORE any full six-run replay.
For each high-NLOS-fraction epoch it removes the NLOS-flagged satellites and
re-solves a robust SPP position, then compares its 2D error against truth with
the error of the source .pos pick. If the NLOS-excluded re-solve does not turn
clearly more fail->pass than pass->fail, the whole lever is killed here.

    PYTHONPATH=python python3 experiments/analyze_nlos_spp_candidate_triage_nr2.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.robust_spp import robust_spp

WGS84_A = 6378137.0
WGS84_E2 = 2 * (1 / 298.257223563) - (1 / 298.257223563) ** 2

DATA_DIR = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/nagoya/run2")
NLOS_CSV = Path("experiments/results/nlos_masks/nagoya_run2_per_epoch_nlos.csv")
SOURCE_POS = Path("experiments/results/libgnss_diag_phase19/gici_full_hisnr/nagoya_run2_full.pos")


def ecef_to_llh(x: float, y: float, z: float) -> tuple[float, float]:
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        s = math.sin(lat)
        n = WGS84_A / math.sqrt(1 - WGS84_E2 * s * s)
        lat = math.atan2(z + WGS84_E2 * n * s, p)
    return lat, lon


def enu_2d_error(est: np.ndarray, truth: np.ndarray) -> float:
    lat, lon = ecef_to_llh(*truth)
    s_lat, c_lat, s_lon, c_lon = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    d = est - truth
    east = -s_lon * d[0] + c_lon * d[1]
    north = -s_lat * c_lon * d[0] - s_lat * s_lon * d[1] + c_lat * d[2]
    return math.hypot(east, north)


def load_nlos_mask(path: Path) -> dict[float, set[str]]:
    """tow(rounded 1dp) -> set of NLOS prn strings (is_los==0)."""
    mask: dict[float, set[str]] = {}
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split(",")
        i_tow, i_prn, i_los = header.index("tow"), header.index("prn"), header.index("is_los")
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if int(parts[i_los]) == 0:
                mask.setdefault(round(float(parts[i_tow]), 1), set()).add(parts[i_prn])
    return mask


def load_pos(path: Path) -> dict[float, np.ndarray]:
    """tow(rounded 1dp) -> ECEF xyz from a libgnss .pos file."""
    pos: dict[float, np.ndarray] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("%") or not line.strip():
                continue
            c = line.split()
            pos[round(float(c[1]), 1)] = np.array([float(c[2]), float(c[3]), float(c[4])])
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-nlos-frac", type=float, default=0.5)
    ap.add_argument("--min-los-sats", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--systems", default="G,E,J,C")
    args = ap.parse_args()

    print(f"Loading nagoya/run2 raw measurements (systems={args.systems}) ...")
    loader = PPCDatasetLoader(DATA_DIR)
    data = loader.load_experiment_data(
        systems=tuple(args.systems.split(",")), max_epochs=args.max_epochs
    )
    times = data["times"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    used_prns = data["used_prns"]
    truth = data["ground_truth"]
    print(f"  loaded {len(times)} epochs, median {data['n_satellites']} sats, "
          f"constellations={data['constellations']}")

    nlos = load_nlos_mask(NLOS_CSV)
    src = load_pos(SOURCE_POS)
    print(f"  NLOS mask epochs={len(nlos)}, source .pos epochs={len(src)}")

    triggered = 0
    solved = 0
    skipped_lowsat = 0
    no_source = 0
    src_errs, nlos_errs = [], []
    good = bad = wash = 0

    for i in range(len(times)):
        tow = round(float(times[i]), 1)
        prns = used_prns[i]
        nlos_set = nlos.get(tow, set())
        if not nlos_set:
            continue
        los_idx = [j for j, p in enumerate(prns) if p not in nlos_set]
        nlos_frac = 1.0 - len(los_idx) / max(len(prns), 1)
        if nlos_frac < args.min_nlos_frac:
            continue
        triggered += 1
        if tow not in src:
            no_source += 1
            continue
        if len(los_idx) < args.min_los_sats:
            skipped_lowsat += 1
            continue
        idx = np.array(los_idx)
        sol = robust_spp(
            sat_ecef[i][idx], pseudoranges[i][idx],
            weights=weights[i][idx], init_pos=src[tow],
            weight_func="cauchy", min_satellites=args.min_los_sats,
        )
        if sol is None:
            skipped_lowsat += 1
            continue
        solved += 1
        src_e = enu_2d_error(src[tow], truth[i])
        nlos_e = enu_2d_error(sol, truth[i])
        src_errs.append(src_e)
        nlos_errs.append(nlos_e)
        src_pass, nlos_pass = src_e <= 1.0, nlos_e <= 1.0
        if nlos_pass and not src_pass:
            good += 1
        elif src_pass and not nlos_pass:
            bad += 1
        else:
            wash += 1

    src_errs = np.array(src_errs)
    nlos_errs = np.array(nlos_errs)
    print("\n=== NLOS-SPP candidate offline triage (nagoya/run2) ===")
    print(f"min_nlos_frac={args.min_nlos_frac}  min_los_sats={args.min_los_sats}")
    print(f"triggered epochs            : {triggered}")
    print(f"  re-solved                 : {solved}")
    print(f"  skipped (<min LOS / fail) : {skipped_lowsat}")
    print(f"  no source .pos row        : {no_source}")
    if solved:
        print(f"median 2D err  source .pos  : {np.median(src_errs):.2f} m")
        print(f"median 2D err  NLOS-SPP     : {np.median(nlos_errs):.2f} m")
        print(f"<=1m pass  source / NLOS-SPP: {int(np.sum(src_errs<=1))} / {int(np.sum(nlos_errs<=1))}")
        print(f"transitions  good(fail->pass)={good}  bad(pass->fail)={bad}  wash={wash}")
        print(f"NET good-bad                : {good - bad}")
        verdict = "PROCEED" if (good - bad) >= 20 and np.median(nlos_errs) < 1.5 else "KILL"
        print(f"\nVERDICT: {verdict}  (proceed if good-bad>=20 and median NLOS-SPP err<1.5m)")


if __name__ == "__main__":
    main()
