#!/usr/bin/env python3
"""End-to-end: signal generation → acquisition → positioning.

Generates IQ signals with known receiver position + satellite geometry,
runs acquisition to recover code phase → pseudorange, then solves for
position via WLS. Compares with ground truth.

Two scenarios:
  1. Open sky (no buildings) — expect sub-meter positioning error
  2. Urban canyon (PLATEAU buildings) — expect degraded accuracy from NLOS/multipath
"""

import argparse
import math
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from e2e_utils import (
        C_LIGHT,
        DEFAULT_CODE_LOCK_MAX_ERROR_M,
        acquisition_code_phase_to_pseudorange,
        compute_e2e_wls_weights,
        dump_e2e_diagnostics_csv,
        pseudorange_to_code_phase_chips,
        refine_acquisition_code_lags_dll_batch,
        refine_acquisition_code_lags_diagnostic_batch,
    )
except ImportError:
    from experiments.e2e_utils import (
        C_LIGHT,
        DEFAULT_CODE_LOCK_MAX_ERROR_M,
        acquisition_code_phase_to_pseudorange,
        compute_e2e_wls_weights,
        dump_e2e_diagnostics_csv,
        pseudorange_to_code_phase_chips,
        refine_acquisition_code_lags_dll_batch,
        refine_acquisition_code_lags_diagnostic_batch,
    )

from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla
from gnss_gpu.acquisition import Acquisition
from gnss_gpu._gnss_gpu import wls_position
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator


def _diagnostics_slug(name):
    slug = str(name).strip().lower()
    if "(" in slug and ")" in slug:
        start = slug.find("(")
        end = slug.find(")", start + 1)
        inner = slug[start + 1:end].strip()
        if inner:
            slug = inner
    slug = slug.replace(" ", "_").replace("-", "_")
    slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in slug)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "scenario"


def _scenario_diagnostics_path(path, scenario_name):
    base = Path(path)
    return base.with_name(f"{base.stem}_{_diagnostics_slug(scenario_name)}.csv")


def run_scenario(name, rx_ecef_true, sat_ecef, prn_list, sat_clk=None, building_model=None,
                 noise_floor_db=-30, sampling_freq=2.6e6,
                 dll_gain=0.22, pll_gain=0.18, n_iter=15, correlator_spacing=0.5,
                 diagnostics_csv=None, gain_schedule="constant", n_coherent_ms=1):
    """Run one E2E scenario: generate → acquire → position.

    Pseudorange construction:
      Acquisition code_phase has 1ms ambiguity (~300km). We resolve it around
      the approximate raw pseudorange predicted by geometry + satellite clock.
      The resulting pseudorange carries acquisition quantization error and any
      multipath-induced code-phase bias.
    """
    n_sat = len(prn_list)
    ranges_true = np.linalg.norm(sat_ecef - rx_ecef_true, axis=1)
    sat_clock_m = np.zeros(n_sat, dtype=np.float64) if sat_clk is None else np.asarray(
        sat_clk, dtype=np.float64) * C_LIGHT
    approx_raw_pr = ranges_true - sat_clock_m

    # --- Step 1: Generate IQ signal ---
    samples_per_ms = int(round(sampling_freq * 1e-3))
    n_samples_total = int(n_coherent_ms) * samples_per_ms
    excess_delays = np.zeros(n_sat)
    if building_model is not None:
        usim = UrbanSignalSimulator(
            building_model=building_model,
            sampling_freq=sampling_freq,
            noise_floor_db=noise_floor_db,
            nlos_attenuation_db=8.0,
            fresnel_coeff=0.5,
        )
        result = usim.compute_epoch(
            rx_ecef=rx_ecef_true, sat_ecef=sat_ecef, sat_clk=sat_clk, prn_list=prn_list,
            n_samples=n_samples_total)
        iq = result["iq"]
        is_los = result["is_los"]
        excess_delays = result["excess_delays"]
        n_channels = len(result["channels"])
    else:
        channels = []
        for i in range(n_sat):
            channels.append({
                "prn": int(prn_list[i]),
                "code_phase": float(pseudorange_to_code_phase_chips(approx_raw_pr[i])),
                "carrier_phase": 0.0,
                "doppler_hz": 0.0,
                "amplitude": 1.0,
                "nav_bit": 1,
            })
        sim = SignalSimulator(sampling_freq=sampling_freq, noise_floor_db=noise_floor_db)
        iq = sim.generate_epoch(channels, n_samples=n_samples_total)
        is_los = np.ones(n_sat, dtype=bool)
        n_channels = n_sat

    iq_full = iq
    iq_acq = iq_full[: 2 * samples_per_ms]
    signal_i_acq = iq_acq[0::2].copy()
    signal_i_full = iq_full[0::2].copy()

    # --- Step 2: Acquisition ---
    acq = Acquisition(sampling_freq=sampling_freq, intermediate_freq=0,
                      doppler_range=5000, doppler_step=500, threshold=2.0)
    acq_results = acq.acquire(signal_i_acq, prn_list=prn_list)

    acquired_idx = []
    pseudoranges = []
    sat_ecef_acquired = []

    candidates = [(i, r) for i, r in enumerate(acq_results) if r["acquired"]]
    candidate_prns = [int(prn_list[i]) for i, _ in candidates]
    candidate_lags = [r["code_phase"] for _, r in candidates]
    candidate_dopplers = [r["doppler_hz"] for _, r in candidates]
    if candidates:
        lag_refs, prompt_pow, dll_abs = refine_acquisition_code_lags_dll_batch(
            signal_i_full,
            candidate_prns,
            candidate_lags,
            candidate_dopplers,
            sampling_freq,
            intermediate_freq=0.0,
            n_iter=n_iter,
            dll_gain=dll_gain,
            pll_gain=pll_gain,
            correlator_spacing=correlator_spacing,
            return_lock_metrics=True,
            gain_schedule=gain_schedule,
        )
    else:
        lag_refs = np.array([], dtype=np.float64)
        prompt_pow = np.array([], dtype=np.float64)
        dll_abs = np.array([], dtype=np.float64)

    if diagnostics_csv is not None:
        diagnostics = refine_acquisition_code_lags_diagnostic_batch(
            signal_i_full,
            candidate_prns,
            candidate_lags,
            candidate_dopplers,
            sampling_freq,
            intermediate_freq=0.0,
            n_iter=n_iter,
            dll_gain=dll_gain,
            pll_gain=pll_gain,
            correlator_spacing=correlator_spacing,
            gain_schedule=gain_schedule,
        )
        diagnostics_path = _scenario_diagnostics_path(diagnostics_csv, name)
        dump_e2e_diagnostics_csv(diagnostics_path, diagnostics)
        print(f"  Diagnostics CSV: {diagnostics_path}")

    accepted_pp = []
    accepted_snr = []
    accepted_dll = []
    for j, (i, r) in enumerate(candidates):
        lag_ref = float(lag_refs[j])
        pr_raw = acquisition_code_phase_to_pseudorange(
            lag_ref, sampling_freq, approx_raw_pr[i])
        if abs(pr_raw - approx_raw_pr[i]) > DEFAULT_CODE_LOCK_MAX_ERROR_M:
            continue
        pr = pr_raw + sat_clock_m[i]

        acquired_idx.append(i)
        pseudoranges.append(pr)
        sat_ecef_acquired.append(sat_ecef[i])
        accepted_pp.append(prompt_pow[j])
        accepted_snr.append(float(r["snr"]))
        accepted_dll.append(dll_abs[j])

    n_acquired = len(acquired_idx)

    # --- Step 3: WLS Positioning ---
    pos_error = float("nan")
    pos_ecef = None

    if n_acquired >= 4:
        sat_flat = np.array(sat_ecef_acquired).flatten()
        pr_arr = np.array(pseudoranges)
        weights = compute_e2e_wls_weights(accepted_pp, accepted_snr, accepted_dll)

        try:
            result_pos, iters = wls_position(sat_flat, pr_arr, weights)
            pos_ecef = result_pos[:3]
            pos_error = float(np.linalg.norm(pos_ecef - rx_ecef_true))
        except Exception as e:
            pos_error = float("nan")

    lat, lon, alt = ecef_to_lla(*rx_ecef_true)

    return {
        "name": name,
        "n_sat": n_sat,
        "n_acquired": n_acquired,
        "n_channels": n_channels,
        "pos_error_m": pos_error,
        "pos_ecef": pos_ecef,
        "rx_true": rx_ecef_true,
        "is_los": is_los,
        "excess_delays": excess_delays,
        "n_los": int(np.sum(is_los[:n_sat])),
        "n_nlos": int(np.sum(~is_los[:n_sat])),
        "lat": math.degrees(lat),
        "lon": math.degrees(lon),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="E2E positioning: signal sim → acquisition → DLL/PLL → weighted WLS.",
    )
    p.add_argument(
        "--dll-gain", type=float, default=0.22,
        help="DLL code-phase step gain per iteration (default: 0.22).",
    )
    p.add_argument(
        "--pll-gain", type=float, default=0.18,
        help="PLL carrier-phase step gain per iteration (0 disables PLL).",
    )
    p.add_argument(
        "--n-iter", type=int, default=15,
        help="Post-acquisition DLL/PLL iterations on the 1 ms buffer (default: 15).",
    )
    p.add_argument(
        "--correlator-spacing", type=float, default=0.5,
        help="Early/late spacing in chips (default: 0.5).",
    )
    p.add_argument(
        "--diagnostics-csv", default=None,
        help="Optional per-scenario channel diagnostics CSV path stem.",
    )
    p.add_argument(
        "--gain-schedule",
        choices=["constant", "cn0_weighted"],
        default="constant",
        help="DLL/PLL gain schedule (default: constant).",
    )
    p.add_argument(
        "--n-coherent-ms",
        type=int,
        default=1,
        help="coherent integration length in milliseconds (>=1). "
             "Acquisition still runs on the first 1 ms; refinement uses the full span.",
    )
    args = p.parse_args()
    if args.n_coherent_ms < 1:
        p.error("--n-coherent-ms must be >= 1")
    return args


def main():
    args = parse_args()
    out_dir = os.path.join(os.path.dirname(__file__), "results", "e2e_positioning")
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Tracking refine: n_iter={args.n_iter} dll_gain={args.dll_gain} "
        f"pll_gain={args.pll_gain} correlator_spacing={args.correlator_spacing} "
        f"gain_schedule={args.gain_schedule}",
    )

    # Load real ephemeris
    nav_path = "experiments/data/urbannav/Odaiba/base.nav"
    print(f"Loading ephemeris: {nav_path}")
    nav_messages = read_nav_rinex_multi(nav_path)
    eph = Ephemeris(nav_messages)
    gps_prns = [p for p in eph.available_prns
                if (isinstance(p, str) and p.startswith("G")) or isinstance(p, int)]

    # Compute satellite positions at a specific time
    gps_tow = 273500.0
    sat_ecef, sat_clk, used_prns = eph.compute(gps_tow, prn_list=gps_prns)
    prn_ints = []
    for p in used_prns:
        if isinstance(p, str) and p.startswith("G"):
            prn_ints.append(int(p[1:]))
        elif isinstance(p, int):
            prn_ints.append(p)
        else:
            prn_ints.append(0)

    # Filter to visible satellites (elevation > 10°)
    rx_true = np.array([-3963426.8, 3350882.2, 3694865.5])  # Odaiba
    lat, lon, _ = ecef_to_lla(*rx_true)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    R_enu = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat],
    ])

    visible = []
    for i in range(len(prn_ints)):
        diff = sat_ecef[i] - rx_true
        enu = R_enu @ diff
        el = math.atan2(enu[2], math.sqrt(enu[0]**2 + enu[1]**2))
        if el > math.radians(10):
            visible.append(i)

    sat_ecef_vis = sat_ecef[visible]
    sat_clk_vis = sat_clk[visible]
    prn_vis = [prn_ints[i] for i in visible]
    print(f"Visible satellites: {len(prn_vis)} (el > 10°)")
    print(f"PRNs: {prn_vis}")

    # --- Scenario 1: Open sky ---
    print("\n=== Scenario 1: Open Sky ===")
    r1 = run_scenario(
        "Open Sky", rx_true, sat_ecef_vis, prn_vis, sat_clk=sat_clk_vis,
        building_model=None, noise_floor_db=-30,
        dll_gain=args.dll_gain, pll_gain=args.pll_gain, n_iter=args.n_iter,
        correlator_spacing=args.correlator_spacing,
        diagnostics_csv=args.diagnostics_csv,
        gain_schedule=args.gain_schedule,
        n_coherent_ms=args.n_coherent_ms,
    )
    print(f"  Acquired: {r1['n_acquired']}/{r1['n_sat']}")
    print(f"  Position error: {r1['pos_error_m']:.2f} m")

    # --- Scenario 2: Urban canyon ---
    print("\n=== Scenario 2: Urban (PLATEAU Odaiba) ===")
    loader = PlateauLoader(zone=9)
    building = loader.load_directory("experiments/data/plateau_odaiba")
    bvh = BVHAccelerator.from_building_model(building)
    print(f"  {len(building.triangles)} triangles")

    r2 = run_scenario(
        "Urban (Odaiba)", rx_true, sat_ecef_vis, prn_vis, sat_clk=sat_clk_vis,
        building_model=bvh, noise_floor_db=-30,
        dll_gain=args.dll_gain, pll_gain=args.pll_gain, n_iter=args.n_iter,
        correlator_spacing=args.correlator_spacing,
        diagnostics_csv=args.diagnostics_csv,
        gain_schedule=args.gain_schedule,
        n_coherent_ms=args.n_coherent_ms,
    )
    print(f"  LOS: {r2['n_los']}, NLOS: {r2['n_nlos']}")
    print(f"  Acquired: {r2['n_acquired']}/{r2['n_sat']}")
    print(f"  Position error: {r2['pos_error_m']:.2f} m")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#1a1a2e")

    for ax, r in zip(axes, [r1, r2]):
        ax.set_facecolor("#16213e")
        tc = "#e0e0e0"

        # Position error bar
        color = "#00d4aa" if r["pos_error_m"] < 50 else "#ffd93d" if r["pos_error_m"] < 200 else "#ff6b6b"
        ax.barh(["Position\nError"], [r["pos_error_m"]], color=color, height=0.4)
        ax.text(r["pos_error_m"] + 1, 0, f'{r["pos_error_m"]:.1f} m', color=tc, va="center", fontsize=12)

        # Stats
        stats = (
            f"Satellites: {r['n_sat']}\n"
            f"Acquired: {r['n_acquired']}\n"
            f"LOS: {r['n_los']}  NLOS: {r['n_nlos']}"
        )
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=11,
                color=tc, va="top", ha="right", fontfamily="monospace",
                bbox=dict(facecolor="#0f3460", edgecolor="#334", boxstyle="round,pad=0.5"))

        ax.set_title(r["name"], color="white", fontsize=14, fontweight="bold")
        ax.set_xlabel("Position Error [m]", color=tc, fontsize=11)
        ax.tick_params(colors=tc, labelsize=10)
        ax.grid(True, axis="x", color="#333355", alpha=0.3)
        for s in ax.spines.values():
            s.set_color("#333355")

    fig.suptitle("End-to-End: Signal Generation → Acquisition → WLS Positioning",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "e2e_positioning.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Summary table
    print(f"\n{'Scenario':<25} {'Sats':>5} {'Acq':>5} {'LOS':>5} {'NLOS':>5} {'Error [m]':>10}")
    print("-" * 60)
    for r in [r1, r2]:
        err = f"{r['pos_error_m']:.1f}" if not math.isnan(r['pos_error_m']) else "N/A"
        print(f"{r['name']:<25} {r['n_sat']:>5} {r['n_acquired']:>5} {r['n_los']:>5} {r['n_nlos']:>5} {err:>10}")


if __name__ == "__main__":
    main()
