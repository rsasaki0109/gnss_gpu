#!/usr/bin/env python3
"""Use-case demo: urban multipath assessment at a fixed site.

Simulates a receiver standing still in an urban canyon (real UrbanNav Odaiba
GNSS + PLATEAU building mesh) and renders a per-satellite sky-map assessment
of the kind an engineer would use to judge an antenna/installation site:
which sky sectors are blocked (NLOS), how large the resulting pseudorange
multipath bias is, and how much C/N0 degrades relative to open-sky.

Reuses :func:`gnss_gpu.scenario.run_scenario` (see ``examples/demo_scenario_engine.py``
for the minimal wiring this is based on) -- no physics is reimplemented here,
only stats aggregation and plotting.

Run:
    PYTHONPATH=python python examples/demo_urban_multipath.py

Output:
    results/use_cases/urban_multipath.png
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from gnss_gpu.scenario import ScenarioConfig, ScenarioResult, run_scenario  # noqa: E402

GPS_EPOCH = datetime(1980, 1, 6)

# Receiver point: the Odaiba UrbanNav reference fix, offset ~15 m
# north-east toward the surrounding building block to push the receiver
# into the street canyon (the raw reference fix sits closer to open sky
# over the road). Found by a short manual sweep: offsets of ~30 m+ in this
# direction land the receiver fully inside building shadow (NLOS fraction
# 1.0, no LOS satellites left to contrast against); this smaller offset
# gives a genuine street-canyon mix (~29% NLOS at a 9-epoch probe).
_RX_OFFSET_LAT_DEG = 0.00015
_RX_OFFSET_LON_DEG = 0.00015

_DURATION_S = 59.0  # ~60 epochs at 1 Hz (t=0..59)
_STEP_S = 1.0

_OUTPUT_PNG = _REPO_ROOT / "results" / "use_cases" / "urban_multipath.png"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _first_reference_fix(reference_csv: Path) -> tuple[datetime, float, float, float]:
    """Read (start_time_utc, lat_deg, lon_deg, alt_m) from the first row."""
    with open(reference_csv, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        row = next(reader)

    week = int(float(row["GPS Week"]))
    tow = float(row["GPS TOW (s)"])
    lat = float(row["Latitude (deg)"])
    lon = float(row["Longitude (deg)"])
    alt = float(row["Ellipsoid Height (m)"])
    start_time = GPS_EPOCH + timedelta(weeks=week, seconds=tow)
    return start_time, lat, lon, alt


# ---------------------------------------------------------------------------
# Stats computation (pure function of a ScenarioResult -- unit-testable
# without a GPU, a PLATEAU mesh, or network access; see
# tests/test_demo_urban_multipath.py).
# ---------------------------------------------------------------------------


def compute_multipath_stats(result: ScenarioResult, top_n: int = 5) -> dict:
    """Aggregate per-satellite / overall multipath + C/N0 stats from a
    :class:`~gnss_gpu.scenario.ScenarioResult`.

    Returns a dict with:
      - ``n_epochs``, ``n_rows``: epoch count and total satellite-epoch rows.
      - ``nlos_fraction``: fraction of satellite-epoch rows that are NLOS.
      - ``mean_multipath_excess_nlos_m`` / ``max_multipath_excess_nlos_m``:
        multipath excess delay stats restricted to NLOS rows (NaN if none).
      - ``mean_cn0_los_dbhz`` / ``mean_cn0_nlos_dbhz``: mean C/N0 split by
        LOS/NLOS (NaN if a group is empty).
      - ``per_satellite``: list of per-sat_id dicts (n_epochs, n_nlos,
        nlos_fraction, mean_multipath_excess_m, max_multipath_excess_m,
        mean_cn0_dbhz), sorted by ``mean_multipath_excess_m`` descending.
      - ``worst_offenders``: the first ``top_n`` entries of ``per_satellite``.
    """
    arrays = result.to_arrays()
    n_rows = int(arrays["is_los"].size)

    is_los = arrays["is_los"]
    multipath = arrays["multipath_excess_m"]
    cn0 = arrays["cn0_dbhz"]
    sat_id = arrays["sat_id"]

    nlos_mask = ~is_los
    n_nlos = int(nlos_mask.sum())

    def _mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float("nan")

    def _max(x: np.ndarray) -> float:
        return float(np.max(x)) if x.size else float("nan")

    stats: dict = {
        "n_epochs": result.n_epochs,
        "n_rows": n_rows,
        "n_los": int(is_los.sum()),
        "n_nlos": n_nlos,
        "nlos_fraction": (n_nlos / n_rows) if n_rows else float("nan"),
        "mean_multipath_excess_nlos_m": _mean(multipath[nlos_mask]),
        "max_multipath_excess_nlos_m": _max(multipath[nlos_mask]),
        "mean_cn0_los_dbhz": _mean(cn0[is_los]),
        "mean_cn0_nlos_dbhz": _mean(cn0[nlos_mask]),
    }

    per_sat = []
    for sid in np.unique(sat_id):
        m = sat_id == sid
        sat_nlos = nlos_mask[m]
        per_sat.append(
            {
                "sat_id": str(sid),
                "n_epochs": int(m.sum()),
                "n_nlos": int(sat_nlos.sum()),
                "nlos_fraction": float(sat_nlos.mean()) if m.any() else float("nan"),
                "mean_multipath_excess_m": _mean(multipath[m]),
                "max_multipath_excess_m": _max(multipath[m]),
                "mean_cn0_dbhz": _mean(cn0[m]),
            }
        )
    per_sat.sort(key=lambda d: d["mean_multipath_excess_m"], reverse=True)

    stats["per_satellite"] = per_sat
    stats["worst_offenders"] = per_sat[:top_n]
    return stats


def build_satellite_tracks(result: ScenarioResult) -> dict[str, dict[str, np.ndarray]]:
    """Group per-epoch arrays by ``sat_id`` for plotting a track per satellite.

    Returns ``{sat_id: {"az_deg": ..., "el_deg": ..., "is_los": ...,
    "multipath_excess_m": ..., "cn0_dbhz": ...}}``, each array epoch-ordered.
    """
    arrays = result.to_arrays()
    tracks: dict[str, dict[str, np.ndarray]] = {}
    order = np.argsort(arrays["epoch_index"], kind="stable")
    for key in ("sat_id", "elevation_rad", "azimuth_rad", "is_los", "multipath_excess_m", "cn0_dbhz"):
        arrays[key] = arrays[key][order]

    for sid in np.unique(arrays["sat_id"]):
        m = arrays["sat_id"] == sid
        tracks[str(sid)] = {
            "az_deg": np.degrees(arrays["azimuth_rad"][m]),
            "el_deg": np.degrees(arrays["elevation_rad"][m]),
            "is_los": arrays["is_los"][m],
            "multipath_excess_m": arrays["multipath_excess_m"][m],
            "cn0_dbhz": arrays["cn0_dbhz"][m],
        }
    return tracks


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_polar_ax(ax, title: str) -> None:
    """Match the az/el polar convention used by gnss_gpu.viz.plots.plot_skyplot:
    theta zero at North, clockwise, zenith at the plot centre."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90", "75", "60", "45", "30", "15", "0"])
    ax.set_title(title, va="bottom", pad=20)


def render_figure(result: ScenarioResult, stats: dict, out_path: Path):
    """Render the two-panel sky-map figure and save it to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = build_satellite_tracks(result)

    fig, (ax_los, ax_cn0) = plt.subplots(
        1, 2, figsize=(15, 7.5), subplot_kw={"projection": "polar"}
    )

    _setup_polar_ax(ax_los, "LOS / NLOS tracks, sized by multipath excess")
    _setup_polar_ax(ax_cn0, "C/N0 along track")

    cn0_all = np.concatenate([t["cn0_dbhz"] for t in tracks.values()]) if tracks else np.array([0.0])
    cn0_vmin, cn0_vmax = float(np.min(cn0_all)), float(np.max(cn0_all))
    sc_cn0 = None

    for sid, tr in sorted(tracks.items()):
        theta = np.radians(tr["az_deg"])
        r = 90.0 - tr["el_deg"]
        is_los = tr["is_los"]
        excess = tr["multipath_excess_m"]

        # Left panel: thin track line + LOS(blue)/NLOS(red) markers sized by
        # multipath excess (a small floor so LOS/zero-excess points stay visible).
        ax_los.plot(theta, r, "-", color="0.75", linewidth=0.6, zorder=1)
        colors = np.where(is_los, "tab:blue", "tab:red")
        sizes = 18.0 + 6.0 * np.clip(excess, 0.0, 60.0)
        ax_los.scatter(theta, r, c=colors, s=sizes, edgecolors="k", linewidths=0.3, zorder=5)

        # Annotate the satellite label at its first epoch, and flag the worst
        # NLOS excess sample on that track directly on the plot.
        ax_los.annotate(
            sid, (theta[0], r[0]), fontsize=7, ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 6),
        )
        if np.any(~is_los) and np.max(excess[~is_los]) > 0:
            k = int(np.argmax(np.where(~is_los, excess, -np.inf)))
            ax_los.annotate(
                f"{excess[k]:.0f} m", (theta[k], r[k]), fontsize=6, color="darkred",
                ha="left", va="top", textcoords="offset points", xytext=(4, -4),
            )

        # Right panel: track coloured by C/N0.
        ax_cn0.plot(theta, r, "-", color="0.75", linewidth=0.6, zorder=1)
        sc_cn0 = ax_cn0.scatter(
            theta, r, c=tr["cn0_dbhz"], cmap="RdYlGn", vmin=cn0_vmin, vmax=cn0_vmax,
            s=26, edgecolors="k", linewidths=0.3, zorder=5,
        )
        ax_cn0.annotate(
            sid, (theta[0], r[0]), fontsize=7, ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 6),
        )

    los_handle = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:blue",
                             markeredgecolor="k", markersize=7, label="LOS")
    nlos_handle = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:red",
                              markeredgecolor="k", markersize=7, label="NLOS (larger = more excess delay)")
    ax_los.legend(handles=[los_handle, nlos_handle], loc="lower right",
                  bbox_to_anchor=(1.15, -0.08), fontsize=8)

    if sc_cn0 is not None:
        fig.colorbar(sc_cn0, ax=ax_cn0, pad=0.1, label="C/N0 [dB-Hz]")

    stats_text = (
        f"NLOS fraction: {stats['nlos_fraction']:.1%}  "
        f"({stats['n_nlos']}/{stats['n_rows']} sat-epochs)\n"
        f"Multipath excess (NLOS): mean {stats['mean_multipath_excess_nlos_m']:.1f} m, "
        f"max {stats['max_multipath_excess_nlos_m']:.1f} m\n"
        f"C/N0: LOS mean {stats['mean_cn0_los_dbhz']:.1f} dB-Hz, "
        f"NLOS mean {stats['mean_cn0_nlos_dbhz']:.1f} dB-Hz "
        f"(Delta {stats['mean_cn0_los_dbhz'] - stats['mean_cn0_nlos_dbhz']:.1f} dB-Hz)"
    )
    fig.text(
        0.5, 0.02, stats_text, ha="center", va="bottom", fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.85},
    )

    rx = result.epochs[0].rx_ecef if result.epochs else None
    fig.suptitle(
        "Urban multipath assessment -- fixed site sky map (Odaiba UrbanNav)"
        + (f"\nrx_ecef=({rx[0]:.1f}, {rx[1]:.1f}, {rx[2]:.1f}) m" if rx is not None else ""),
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(stats: dict) -> None:
    print(f"epochs simulated: {stats['n_epochs']}")
    print(f"total satellite-epoch rows: {stats['n_rows']}  (LOS {stats['n_los']}, NLOS {stats['n_nlos']})")
    print(f"NLOS fraction: {stats['nlos_fraction']:.3f}")
    print(
        f"multipath excess on NLOS sats: mean={stats['mean_multipath_excess_nlos_m']:.2f} m "
        f"max={stats['max_multipath_excess_nlos_m']:.2f} m"
    )
    print(
        f"C/N0: LOS mean={stats['mean_cn0_los_dbhz']:.2f} dB-Hz  "
        f"NLOS mean={stats['mean_cn0_nlos_dbhz']:.2f} dB-Hz"
    )
    print("worst offenders (top 5 by mean multipath excess):")
    for sat in stats["worst_offenders"]:
        print(
            f"  {sat['sat_id']}: mean={sat['mean_multipath_excess_m']:.2f} m "
            f"max={sat['max_multipath_excess_m']:.2f} m "
            f"nlos_fraction={sat['nlos_fraction']:.2f} "
            f"mean_cn0={sat['mean_cn0_dbhz']:.1f} dB-Hz "
            f"(n={sat['n_epochs']})"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    odaiba_dir = _REPO_ROOT / "experiments" / "data" / "urbannav" / "Odaiba"
    plateau_dir = _REPO_ROOT / "experiments" / "data" / "plateau_odaiba"
    nav_file = odaiba_dir / "base.nav"
    reference_csv = odaiba_dir / "reference.csv"

    if not (odaiba_dir.is_dir() and plateau_dir.is_dir() and nav_file.is_file()):
        print(
            "demo_urban_multipath: real UrbanNav Odaiba data or the PLATEAU "
            "Odaiba mesh is not available locally "
            f"(expected {odaiba_dir} and {plateau_dir}); skipping. See "
            "experiments/data/README.md (or the fetch_urbannav_subset / "
            "fetch_plateau_subset scripts) to download them."
        )
        return 0

    start_time, lat0, lon0, alt0 = _first_reference_fix(reference_csv)
    lat = lat0 + _RX_OFFSET_LAT_DEG
    lon = lon0 + _RX_OFFSET_LON_DEG
    print(
        f"Odaiba base fix: lat={lat0:.6f} lon={lon0:.6f} alt={alt0:.2f} m\n"
        f"receiver site (offset toward buildings): lat={lat:.6f} lon={lon:.6f} alt={alt0:.2f} m"
    )

    config = ScenarioConfig(
        nav_file=str(nav_file),
        lat_deg=lat,
        lon_deg=lon,
        alt_m=alt0,
        start_time=start_time,
        duration_s=_DURATION_S,
        step_s=_STEP_S,
        constellations=["G"],
        plateau_dir=str(plateau_dir),
        elevation_mask_deg=10.0,
        diffraction_model="utd",
        seed=0,
    )

    print(f"running scenario engine over {int(_DURATION_S / _STEP_S) + 1} epochs (diffraction_model=utd) ...")
    t0 = time.time()
    result = run_scenario(config)
    elapsed_s = time.time() - t0
    print(f"scenario engine wall time: {elapsed_s:.1f} s")

    stats = compute_multipath_stats(result)
    print_report(stats)

    render_figure(result, stats, _OUTPUT_PNG)
    print(f"saved figure: {_OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
