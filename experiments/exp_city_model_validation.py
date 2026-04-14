#!/usr/bin/env python3
"""Validate PLATEAU 3D city model against real GNSS C/N0 observations.

Uses UrbanNav RINEX observations (C/N0 per satellite) + PLATEAU BVH
ray-tracing to detect inconsistencies between the 3D model and reality.
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi, _datetime_to_gps_seconds_of_week
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.city_model_validator import validate_epoch, EpochValidation
from gnss_gpu.urban_signal_sim import ecef_to_lla


def load_reference(csv_path, step=100):
    """Load ground truth trajectory."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    positions, times = [], []
    for i in range(0, len(rows), step):
        r = rows[i]
        positions.append([float(r[" ECEF X (m)"]),
                          float(r[" ECEF Y (m)"]),
                          float(r[" ECEF Z (m)"])])
        times.append(float(r["GPS TOW (s)"]))
    return np.array(positions), np.array(times)


def find_cn0(obs_dict, system_prefix):
    """Extract L1-band C/N0 [dB-Hz] from a RINEX observation dict.

    RINEX uses different observation codes per constellation:
      GPS:     S1C (C/A code)
      Galileo: S1X (pilot), S1B (data)
      BeiDou:  S1I
      QZSS:    S1C, S1L, S1Z
    We try them in priority order and return the first positive value.
    """
    for key in ["S1C", "S1X", "S1I", "S1L", "S1Z"]:
        if key in obs_dict and obs_dict[key] > 0:
            return obs_dict[key]
    return float("nan")


def run_validation(area_name, obs_path, nav_path, ref_path,
                   plateau_dir, max_epochs=30, step=400):
    """Run validation for one area."""
    print(f"\n=== {area_name} ===")

    # Load RINEX observations
    print(f"Loading RINEX: {obs_path}")
    rinex = read_rinex_obs(obs_path)
    print(f"  {len(rinex.epochs)} epochs")

    # Load ephemeris
    print(f"Loading NAV: {nav_path}")
    nav = read_nav_rinex_multi(nav_path)
    eph = Ephemeris(nav)

    # Load reference trajectory
    ref_pos, ref_times = load_reference(ref_path, step=step)
    print(f"  Reference: {len(ref_pos)} positions")

    # Load PLATEAU
    print(f"Loading PLATEAU: {plateau_dir}")
    loader = PlateauLoader(zone=9)
    building = loader.load_directory(plateau_dir)
    bvh = BVHAccelerator.from_building_model(building)
    print(f"  {len(building.triangles)} triangles, {bvh.n_nodes} BVH nodes")

    # Match RINEX epochs to reference positions (nearest time)
    results = []
    epoch_indices = np.linspace(0, len(rinex.epochs) - 1,
                                min(max_epochs, len(rinex.epochs)), dtype=int)

    for fi, ei in enumerate(epoch_indices):
        ep = rinex.epochs[ei]

        # Convert epoch time to GPS TOW using the canonical helper
        gps_tow = _datetime_to_gps_seconds_of_week(ep.time)

        # Find nearest reference position (interpolate if within 2s)
        time_diffs = np.abs(ref_times - gps_tow)
        nearest_idx = np.argmin(time_diffs)
        if time_diffs[nearest_idx] > 10.0:  # skip if >10s gap
            continue
        if (time_diffs[nearest_idx] < 0.5 or nearest_idx == 0
                or nearest_idx == len(ref_times) - 1):
            rx_ecef = ref_pos[nearest_idx]
        else:
            # Linear interpolation between two nearest reference positions
            if gps_tow > ref_times[nearest_idx] and nearest_idx < len(ref_times) - 1:
                i0, i1 = nearest_idx, nearest_idx + 1
            else:
                i0, i1 = max(0, nearest_idx - 1), nearest_idx
            dt_span = ref_times[i1] - ref_times[i0]
            if dt_span > 0:
                alpha = (gps_tow - ref_times[i0]) / dt_span
                alpha = max(0.0, min(1.0, alpha))
                rx_ecef = (1 - alpha) * ref_pos[i0] + alpha * ref_pos[i1]
            else:
                rx_ecef = ref_pos[nearest_idx]

        # Get GPS satellites with C/N0
        gps_sats = []
        for sat_id in ep.satellites:
            sat_id_clean = sat_id.strip()
            if sat_id_clean.startswith("G") and len(sat_id_clean) >= 2:
                prn_str = sat_id_clean[0] + sat_id_clean[1:].strip().zfill(2)
                obs = ep.observations.get(sat_id, {})
                cn0 = find_cn0(obs, "G")
                if cn0 != cn0:  # NaN
                    continue
                gps_sats.append((prn_str, cn0))

        if len(gps_sats) < 4:
            continue

        # Compute satellite positions
        prn_labels = [s[0] for s in gps_sats]
        cn0_vals = np.array([s[1] for s in gps_sats])

        sat_ecef_list = []
        valid_idx = []
        for i, prn in enumerate(prn_labels):
            sat_pos, sat_clk, used = eph.compute(gps_tow, prn_list=[prn])
            if len(used) > 0:
                sat_ecef_list.append(sat_pos[0])
                valid_idx.append(i)

        if len(valid_idx) < 4:
            continue

        sat_ecef = np.array(sat_ecef_list)
        prn_valid = [prn_labels[i] for i in valid_idx]
        cn0_valid = cn0_vals[valid_idx]

        # Validate
        val = validate_epoch(rx_ecef, sat_ecef, prn_valid, cn0_valid,
                             bvh, time=ep.time)
        results.append(val)

        if (fi + 1) % 10 == 0 or fi == 0:
            print(f"  [{fi+1}/{len(epoch_indices)}] score={val.model_score:.2f} "
                  f"ok={val.n_consistent} miss={val.n_missing} "
                  f"phantom={val.n_phantom} amb={val.n_ambiguous}")

    return results


def plot_results(all_results, output_path):
    """Plot validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    for ax_row, (area_name, results) in zip(axes, all_results.items()):
        if not results:
            continue

        # Left: model score over time
        ax = ax_row[0]
        scores = [r.model_score for r in results]
        epochs = range(len(scores))
        ax.plot(epochs, scores, "-o", color="#008866", markersize=3, linewidth=1.5)
        ax.axhline(1.0, color="#ccc", linestyle="--", linewidth=0.5)
        ax.axhline(0.8, color="#ffd93d", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Model Consistency Score", fontsize=10)
        ax.set_title(f"{area_name} — Score over time (mean={np.mean(scores):.2f})",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Right: stacked bar of categories
        ax = ax_row[1]
        consistent = [r.n_consistent for r in results]
        missing = [r.n_missing for r in results]
        phantom = [r.n_phantom for r in results]
        ambiguous = [r.n_ambiguous for r in results]

        ax.bar(epochs, consistent, color="#00d4aa", label="Consistent", width=1)
        ax.bar(epochs, missing, bottom=consistent, color="#ff6b6b",
               label="Missing building", width=1)
        ax.bar(epochs, phantom,
               bottom=[c + m for c, m in zip(consistent, missing)],
               color="#6c5ce7", label="Phantom building", width=1)
        ax.bar(epochs, ambiguous,
               bottom=[c + m + p for c, m, p in zip(consistent, missing, phantom)],
               color="#ddd", label="Ambiguous", width=1)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Satellite count", fontsize=10)
        ax.set_title(f"{area_name} — Classification breakdown", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("3D City Model Validation: PLATEAU predictions vs GNSS C/N0 observations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "city_model_validation")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    # Odaiba
    all_results["Odaiba"] = run_validation(
        "Odaiba",
        obs_path="experiments/data/urbannav/Odaiba/rover_trimble.obs",
        nav_path="experiments/data/urbannav/Odaiba/base.nav",
        ref_path="experiments/data/urbannav/Odaiba/reference.csv",
        plateau_dir="experiments/data/plateau_odaiba",
        max_epochs=40, step=50,
    )

    # Shinjuku
    all_results["Shinjuku"] = run_validation(
        "Shinjuku",
        obs_path="experiments/data/urbannav/Shinjuku/rover_trimble.obs",
        nav_path="experiments/data/urbannav/Shinjuku/base.nav",
        ref_path="experiments/data/urbannav/Shinjuku/reference.csv",
        plateau_dir="experiments/data/plateau_shinjuku",
        max_epochs=40, step=50,
    )

    # Plot
    plot_results(all_results, os.path.join(out_dir, "city_model_validation.png"))

    # Summary
    print("\n=== Summary ===")
    print(f"{'Area':<12} {'Epochs':>7} {'Score':>7} {'Consistent':>11} {'Missing':>8} {'Phantom':>8}")
    print("-" * 60)
    for area, results in all_results.items():
        if not results:
            print(f"{area:<12} no data")
            continue
        n = len(results)
        avg_score = np.mean([r.model_score for r in results])
        total_c = sum(r.n_consistent for r in results)
        total_m = sum(r.n_missing for r in results)
        total_p = sum(r.n_phantom for r in results)
        print(f"{area:<12} {n:>7} {avg_score:>6.2f} {total_c:>11} {total_m:>8} {total_p:>8}")


if __name__ == "__main__":
    main()
