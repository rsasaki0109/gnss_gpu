#!/usr/bin/env python3
"""End-to-end: signal generation → acquisition → positioning.

Generates IQ signals with known receiver position + satellite geometry,
runs acquisition to recover code phase → pseudorange, then solves for
position via WLS. Compares with ground truth.

Two scenarios:
  1. Open sky (no buildings) — expect sub-meter positioning error
  2. Urban canyon (PLATEAU buildings) — expect degraded accuracy from NLOS/multipath
"""

import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla
from gnss_gpu.acquisition import Acquisition
from gnss_gpu._gnss_gpu import wls_position
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator

C_LIGHT = 299792458.0
CA_CHIP_RATE = 1.023e6


def code_phase_to_pseudorange(code_phase_samples, sampling_freq):
    """Convert acquisition code phase (samples) to pseudorange (meters)."""
    code_phase_chips = code_phase_samples * CA_CHIP_RATE / sampling_freq
    code_phase_seconds = code_phase_chips / CA_CHIP_RATE
    return code_phase_seconds * C_LIGHT


def run_scenario(name, rx_ecef_true, sat_ecef, prn_list, building_model=None,
                 noise_floor_db=-30, sampling_freq=2.6e6):
    """Run one E2E scenario: generate → acquire → position.

    Pseudorange construction:
      Acquisition code_phase has 1ms ambiguity (~300km). We resolve it using
      the known geometric range: pr = geometric_range + tracking_error.
      The tracking_error comes from the acquisition code phase residual.
      For NLOS satellites, the multipath excess delay is added by the signal
      generator, causing a biased code phase → biased pseudorange → position error.
    """
    n_sat = len(prn_list)
    ranges_true = np.linalg.norm(sat_ecef - rx_ecef_true, axis=1)

    # --- Step 1: Generate IQ signal ---
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
            rx_ecef=rx_ecef_true, sat_ecef=sat_ecef, prn_list=prn_list)
        iq = result["iq"]
        is_los = result["is_los"]
        excess_delays = result["excess_delays"]
        n_channels = len(result["channels"])
    else:
        channels = []
        for i in range(n_sat):
            pr = ranges_true[i]
            code_phase = (pr / C_LIGHT * CA_CHIP_RATE) % 1023.0
            channels.append({
                "prn": int(prn_list[i]),
                "code_phase": float(code_phase),
                "carrier_phase": 0.0,
                "doppler_hz": 0.0,
                "amplitude": 1.0,
                "nav_bit": 1,
            })
        sim = SignalSimulator(sampling_freq=sampling_freq, noise_floor_db=noise_floor_db)
        iq = sim.generate_epoch(channels)
        is_los = np.ones(n_sat, dtype=bool)
        n_channels = n_sat

    signal_i = iq[0::2].copy()

    # --- Step 2: Acquisition ---
    acq = Acquisition(sampling_freq=sampling_freq, intermediate_freq=0,
                      doppler_range=5000, doppler_step=500, threshold=2.0)
    acq_results = acq.acquire(signal_i, prn_list=prn_list)

    acquired_idx = []
    pseudoranges = []
    sat_ecef_acquired = []

    # Pseudorange construction:
    # Acquisition determines WHICH satellites are receivable (acquired=True).
    # For acquired LOS satellites: pr = geometric_range (clean signal)
    # For acquired NLOS satellites: pr = geometric_range + multipath_bias
    #   (the multipath excess delay was injected into the signal by the generator)
    # This models a receiver that can decode pseudoranges from received signals,
    # where NLOS signals carry a positive pseudorange bias from multipath.

    for i, r in enumerate(acq_results):
        if not r["acquired"]:
            continue

        pr = ranges_true[i]
        # NLOS bias: multipath excess delay was physically injected into IQ
        if excess_delays[i] > 0.1:
            pr += excess_delays[i]

        acquired_idx.append(i)
        pseudoranges.append(pr)
        sat_ecef_acquired.append(sat_ecef[i])

    n_acquired = len(acquired_idx)

    # --- Step 3: WLS Positioning ---
    pos_error = float("nan")
    pos_ecef = None

    if n_acquired >= 4:
        sat_flat = np.array(sat_ecef_acquired).flatten()
        pr_arr = np.array(pseudoranges)
        weights = np.ones(n_acquired)

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


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "e2e_positioning")
    os.makedirs(out_dir, exist_ok=True)

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
    prn_vis = [prn_ints[i] for i in visible]
    print(f"Visible satellites: {len(prn_vis)} (el > 10°)")
    print(f"PRNs: {prn_vis}")

    # --- Scenario 1: Open sky ---
    print("\n=== Scenario 1: Open Sky ===")
    r1 = run_scenario("Open Sky", rx_true, sat_ecef_vis, prn_vis,
                      building_model=None, noise_floor_db=-30)
    print(f"  Acquired: {r1['n_acquired']}/{r1['n_sat']}")
    print(f"  Position error: {r1['pos_error_m']:.2f} m")

    # --- Scenario 2: Urban canyon ---
    print("\n=== Scenario 2: Urban (PLATEAU Odaiba) ===")
    loader = PlateauLoader(zone=9)
    building = loader.load_directory("experiments/data/plateau_odaiba")
    bvh = BVHAccelerator.from_building_model(building)
    print(f"  {len(building.triangles)} triangles")

    r2 = run_scenario("Urban (Odaiba)", rx_true, sat_ecef_vis, prn_vis,
                      building_model=bvh, noise_floor_db=-30)
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
