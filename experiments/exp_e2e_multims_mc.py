#!/usr/bin/env python3
"""Open Sky Monte-Carlo sweep for multi-ms E2E positioning."""
import argparse
import csv
import math
from pathlib import Path
import sys, time
import numpy as np
from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.acquisition import Acquisition
from gnss_gpu._gnss_gpu import wls_position
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.e2e_helpers import (
    C_LIGHT, DEFAULT_CODE_LOCK_MAX_ERROR_M, acquisition_code_phase_to_pseudorange,
    compute_e2e_wls_weights, pseudorange_to_code_phase_chips,
    refine_acquisition_code_lags_dll_batch)
RX_TRUE = np.array([-3963426.8, 3350882.2, 3694865.5], dtype=np.float64)
CSV_COLUMNS = "n_coherent_ms,noise_floor_db,seed,n_acquired,n_used,position_error_m,runtime_s".split(",")

def ecef_to_lla(x, y, z):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(10):
        s = math.sin(lat)
        n = a / math.sqrt(1 - e2 * s * s)
        lat = math.atan2(z + e2 * n * s, p)
    s = math.sin(lat)
    n = a / math.sqrt(1 - e2 * s * s)
    c = math.cos(lat)
    alt = p / c - n if abs(c) > 1e-10 else abs(z) - n * (1 - e2)
    return lat, lon, alt
def parse_csv_list(parser, option, value, cast):
    try:
        out = [cast(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError:
        parser.error(f"{option} must be a comma-separated list")
    if not out:
        parser.error(f"{option} must not be empty")
    return out
def parse_args():
    p = argparse.ArgumentParser(
        description="Open Sky Monte-Carlo sweep for multi-ms coherent refinement.")
    p.add_argument("--n-coherent-ms-list", default="1,5,10", help="comma-separated ints (default: 1,5,10)")
    p.add_argument("--noise-floor-db-list", default="-30,-10,0", help="comma-separated floats (default: -30,-10,0)")
    p.add_argument("--n-trials", type=int, default=10, help="seeds = 1..n_trials (default: 10)")
    p.add_argument("--sampling-freq", type=float, default=2.6e6, help="Hz (default: 2.6e6)")
    p.add_argument("--dll-gain", type=float, default=0.22)
    p.add_argument("--pll-gain", type=float, default=0.18)
    p.add_argument("--n-iter", type=int, default=15)
    p.add_argument("--correlator-spacing", type=float, default=0.5)
    p.add_argument("--gain-schedule", choices=["constant", "cn0_weighted"], default="constant")
    p.add_argument("--csv", default=None, help="optional output CSV")
    p.add_argument("--gps-tow", type=float, default=273500.0, help="ephemeris epoch (default: 273500.0)")
    p.add_argument("--nav-path", default="experiments/data/urbannav/Odaiba/base.nav")
    raw, argv, i = sys.argv[1:], [], 0
    while i < len(raw):
        if raw[i] in ("--n-coherent-ms-list", "--noise-floor-db-list") and i + 1 < len(raw):
            argv.append(f"{raw[i]}={raw[i + 1]}"); i += 2
        else:
            argv.append(raw[i]); i += 1
    args = p.parse_args(argv)
    args.n_coherent_ms_list = parse_csv_list(p, "--n-coherent-ms-list", args.n_coherent_ms_list, int)
    args.noise_floor_db_list = parse_csv_list(p, "--noise-floor-db-list", args.noise_floor_db_list, float)
    if any(n < 1 for n in args.n_coherent_ms_list):
        p.error("--n-coherent-ms-list entries must be >= 1")
    if args.n_trials < 1:
        p.error("--n-trials must be >= 1")
    return args

def open_sky_geometry(nav_path, gps_tow):
    eph = Ephemeris(read_nav_rinex_multi(nav_path))
    gps_prns = [p for p in eph.available_prns
                if (isinstance(p, str) and p.startswith("G")) or isinstance(p, int)]
    sat_ecef, sat_clk, used_prns = eph.compute(gps_tow, prn_list=gps_prns)
    prns = [int(p[1:]) if isinstance(p, str) and p.startswith("G")
            else p if isinstance(p, int) else 0 for p in used_prns]

    lat, lon, _ = ecef_to_lla(*RX_TRUE)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    r_enu = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])
    visible = []
    for i in range(len(prns)):
        enu = r_enu @ (sat_ecef[i] - RX_TRUE)
        el = math.atan2(enu[2], math.sqrt(enu[0] ** 2 + enu[1] ** 2))
        if el > math.radians(10):
            visible.append(i)
    return sat_ecef[visible], sat_clk[visible], [prns[i] for i in visible]


def run_trial(args, sat_ecef, sat_clk, prns, n_ms, noise_db, seed):
    t0 = time.perf_counter()
    sat_clock_m = np.asarray(sat_clk, dtype=np.float64) * C_LIGHT
    approx_raw_pr = np.linalg.norm(sat_ecef - RX_TRUE, axis=1) - sat_clock_m
    samples_per_ms = int(round(args.sampling_freq * 1e-3))
    channels = [{
        "prn": int(prns[i]),
        "code_phase": float(pseudorange_to_code_phase_chips(approx_raw_pr[i])),
        "carrier_phase": 0.0,
        "doppler_hz": 0.0,
        "amplitude": 1.0,
        "nav_bit": 1,
    } for i in range(len(prns))]

    sim = SignalSimulator(sampling_freq=args.sampling_freq,
                          noise_floor_db=noise_db, noise_seed=seed)
    iq = sim.generate_epoch(channels, n_samples=int(n_ms) * samples_per_ms)
    signal_i_acq = iq[: 2 * samples_per_ms][0::2].copy()
    signal_i_full = iq[0::2].copy()
    acq = Acquisition(args.sampling_freq, intermediate_freq=0,
                      doppler_range=5000, doppler_step=500, threshold=2.0)
    acq_results = acq.acquire(signal_i_acq, prn_list=prns)
    candidates = [(i, r) for i, r in enumerate(acq_results) if r["acquired"]]

    if candidates:
        lag_refs, prompt_pow, dll_abs = refine_acquisition_code_lags_dll_batch(
            signal_i_full,
            [int(prns[i]) for i, _ in candidates],
            [r["code_phase"] for _, r in candidates],
            [r["doppler_hz"] for _, r in candidates],
            args.sampling_freq,
            intermediate_freq=0.0,
            n_iter=args.n_iter,
            dll_gain=args.dll_gain,
            pll_gain=args.pll_gain,
            correlator_spacing=args.correlator_spacing,
            return_lock_metrics=True,
            gain_schedule=args.gain_schedule,
        )
    else:
        lag_refs = prompt_pow = dll_abs = np.array([], dtype=np.float64)

    pseudoranges, sat_used, pp, snr, dll = [], [], [], [], []
    for j, (i, r) in enumerate(candidates):
        pr_raw = acquisition_code_phase_to_pseudorange(
            float(lag_refs[j]), args.sampling_freq, approx_raw_pr[i])
        if abs(pr_raw - approx_raw_pr[i]) > DEFAULT_CODE_LOCK_MAX_ERROR_M:
            continue
        pseudoranges.append(pr_raw + sat_clock_m[i])
        sat_used.append(sat_ecef[i])
        pp.append(prompt_pow[j])
        snr.append(float(r["snr"]))
        dll.append(dll_abs[j])

    err = None
    if len(pseudoranges) >= 4:
        try:
            weights = compute_e2e_wls_weights(pp, snr, dll)
            pos, _ = wls_position(
                np.array(sat_used).flatten(), np.array(pseudoranges), weights)
            err = float(np.linalg.norm(pos[:3] - RX_TRUE))
        except Exception:
            err = None
    return {
        "n_acquired": len(candidates),
        "n_used": len(pseudoranges),
        "position_error_m": err,
        "runtime_s": time.perf_counter() - t0,
    }


def stats(errors):
    if not errors:
        return None, None, None
    arr = np.asarray(errors, dtype=np.float64)
    p25, p75 = np.percentile(arr, [25, 75])
    return float(np.median(arr)), float(p25), float(p75)


def fmt(value):
    return "N/A" if value is None else f"{value:.2f}"


def write_row(writer, n_ms, noise_db, seed, result):
    if writer is None:
        return
    err = result["position_error_m"]
    writer.writerow({
        "n_coherent_ms": n_ms,
        "noise_floor_db": f"{noise_db:g}",
        "seed": seed,
        "n_acquired": result["n_acquired"],
        "n_used": result["n_used"],
        "position_error_m": "" if err is None else f"{err:.9g}",
        "runtime_s": f"{result['runtime_s']:.6f}",
    })


def main():
    args = parse_args()
    sat_ecef, sat_clk, prns = open_sky_geometry(args.nav_path, args.gps_tow)
    print(f"Visible satellites: {len(prns)} (el > 10 deg)")
    print(f"PRNs: {prns}")

    csv_file = writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    summaries = []
    total = len(args.n_coherent_ms_list) * len(args.noise_floor_db_list) * args.n_trials
    idx = 0
    try:
        for n_ms in args.n_coherent_ms_list:
            for noise_db in args.noise_floor_db_list:
                errors = []
                skipped = 0
                for seed in range(1, args.n_trials + 1):
                    print(f"[{idx + 1}/{total}] N={n_ms} noise={noise_db:g} seed={seed}",
                          file=sys.stderr, flush=True)
                    result = run_trial(args, sat_ecef, sat_clk, prns, n_ms, noise_db, seed)
                    idx += 1
                    write_row(writer, n_ms, noise_db, seed, result)
                    if result["position_error_m"] is None:
                        skipped += 1
                    else:
                        errors.append(result["position_error_m"])
                median, p25, p75 = stats(errors)
                summaries.append((n_ms, noise_db, median, p25, p75, len(errors), skipped))
    finally:
        if csv_file is not None:
            csv_file.close()

    for n_ms, noise_db, median, p25, p75, n_used, skipped in summaries:
        print(f"N={n_ms:<4} noise={noise_db:<6g} trials={args.n_trials:<3} "
              f"skipped={skipped:<3} median={fmt(median)} m  "
              f"IQR=[{fmt(p25)}, {fmt(p75)}]")

    print("\n| N_ms | noise [dB] | median [m] | p25 [m] | p75 [m] | n_used | n_skipped |")
    print("|-----:|-----------:|-----------:|--------:|--------:|-------:|----------:|")
    for n_ms, noise_db, median, p25, p75, n_used, skipped in summaries:
        print(f"| {n_ms:>4d} | {noise_db:>10g} | {fmt(median):>10} | "
              f"{fmt(p25):>7} | {fmt(p75):>7} | {n_used:>6d} | {skipped:>9d} |")


if __name__ == "__main__":
    main()
