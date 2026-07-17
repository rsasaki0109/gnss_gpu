"""Use-case demo: route positioning-quality evaluation.

Given a vehicle route through a city, predict where along the route GNSS
will be good or bad -- aimed at someone planning an autonomous-driving test
course who wants to know, before ever driving it, which stretches of a
candidate route will starve the receiver of usable satellites.

Loads the real Odaiba UrbanNav ground-truth trajectory as the route, runs
:func:`gnss_gpu.scenario.run_scenario` in route mode over the PLATEAU mesh
with diffraction on, then derives a per-epoch positioning-quality metric
(HDOP from LOS satellites -> expected horizontal position error) and renders
one summary PNG. If the Odaiba data or PLATEAU mesh is not available
locally, prints a message and exits cleanly (same pattern as
``demo_scenario_engine.py``).
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from gnss_gpu.coverage_map import _ecef_to_lla_deg_vec  # noqa: E402
from gnss_gpu.scenario import ScenarioConfig, ScenarioResult, run_scenario  # noqa: E402

GPS_EPOCH = datetime(1980, 1, 6)
HPE_UERE_M = 5.0  # expected horizontal position error = HDOP * UERE
MIN_LOS_FOR_AVAILABILITY = 4


def _load_route(reference_csv: Path, step_s: float, max_epochs: int) -> list[tuple[datetime, float, float, float]]:
    """Read (time_utc, lat_deg, lon_deg, alt_m) rows, subsampled to ~1/``step_s``."""
    with open(reference_csv, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = list(reader)
    if not rows:
        return []

    dt0 = float(rows[1]["GPS TOW (s)"]) - float(rows[0]["GPS TOW (s)"]) if len(rows) > 1 else step_s
    stride = max(1, int(round(step_s / dt0))) if dt0 > 0 else 1

    route: list[tuple[datetime, float, float, float]] = []
    for row in rows[::stride]:
        week = int(float(row["GPS Week"]))
        tow = float(row["GPS TOW (s)"])
        t = GPS_EPOCH + timedelta(weeks=week, seconds=tow)
        route.append((t, float(row["Latitude (deg)"]), float(row["Longitude (deg)"]), float(row["Ellipsoid Height (m)"])))
        if len(route) >= max_epochs:
            break
    return route


def _epoch_hdop(elevation_rad: np.ndarray, azimuth_rad: np.ndarray, is_los: np.ndarray) -> float:
    """HDOP from the LOS satellites of one epoch (same (H^T H)^-1 formula as
    :func:`gnss_gpu.coverage_map._dop_from_mask`, specialized to one receiver)."""
    idx = np.where(is_los)[0]
    if idx.size < MIN_LOS_FOR_AVAILABILITY:
        return float("nan")
    el = elevation_rad[idx]
    az = azimuth_rad[idx]
    east = np.cos(el) * np.sin(az)
    north = np.cos(el) * np.cos(az)
    up = np.sin(el)
    design = np.stack([east, north, up, np.ones_like(east)], axis=1)
    try:
        q = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return float("nan")
    diag = np.diagonal(q)
    if not np.all(np.isfinite(diag)):
        return float("nan")
    return float(np.sqrt(max(diag[0] + diag[1], 0.0)))


def compute_route_metrics(result: ScenarioResult, uere_m: float = HPE_UERE_M) -> dict[str, np.ndarray]:
    """Per-epoch route positioning-quality metrics from a route-mode :class:`ScenarioResult`.

    Pure function of ``result`` (no I/O, no GPU) so it is testable against a
    tiny synthetic scenario. Returns arrays keyed by ``lat_deg``, ``lon_deg``,
    ``time_s`` (seconds since the first epoch), ``n_visible``, ``n_los``,
    ``hdop``, ``expected_hpe_m`` and ``available`` (``n_los >= 4``).
    """
    n = result.n_epochs
    lat_deg = np.full(n, np.nan)
    lon_deg = np.full(n, np.nan)
    time_s = np.full(n, np.nan)
    n_visible = np.zeros(n, dtype=np.int64)
    n_los = np.zeros(n, dtype=np.int64)
    hdop = np.full(n, np.nan)

    t0 = result.epochs[0].time_utc if n else None
    for i, ep in enumerate(result.epochs):
        lat_i, lon_i = _ecef_to_lla_deg_vec(ep.rx_ecef[None, :])
        lat_deg[i] = float(lat_i[0])
        lon_deg[i] = float(lon_i[0])
        time_s[i] = (ep.time_utc - t0).total_seconds()
        n_visible[i] = ep.n_sat
        n_los[i] = int(np.count_nonzero(ep.is_los))
        hdop[i] = _epoch_hdop(ep.elevation_rad, ep.azimuth_rad, ep.is_los)

    expected_hpe_m = hdop * float(uere_m)
    available = n_los >= MIN_LOS_FOR_AVAILABILITY
    return {
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "time_s": time_s,
        "n_visible": n_visible,
        "n_los": n_los,
        "hdop": hdop,
        "expected_hpe_m": expected_hpe_m,
        "available": available,
    }


def _render_png(metrics: dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite_hpe = np.isfinite(metrics["expected_hpe_m"])
    # Cap the color scale at the smaller of the 90th percentile and a fixed
    # 20 m ceiling: near-singular (exactly-4-satellite) urban-canyon
    # geometries can spike HDOP two orders of magnitude above the rest of
    # the route, and an uncapped linear scale would wash out the good/bad
    # contrast everywhere else. Those points still get the top color
    # (clipped), just off the linear scale.
    vmax = float(np.nanpercentile(metrics["expected_hpe_m"], 90.0)) if finite_hpe.any() else 1.0
    vmax = float(np.clip(vmax, 1.0, 20.0))

    fig, (ax_map, ax_time) = plt.subplots(2, 1, figsize=(8.0, 9.5))

    sc = ax_map.scatter(
        metrics["lon_deg"], metrics["lat_deg"], c=metrics["expected_hpe_m"],
        cmap="viridis_r", vmin=0.0, vmax=vmax, s=22,
    )
    fig.colorbar(sc, ax=ax_map, label=f"Expected HPE [m] (clipped at {vmax:.0f} m)")
    ax_map.set_xlabel("Longitude [deg]")
    ax_map.set_ylabel("Latitude [deg]")
    ax_map.set_title("Route positioning quality -- Odaiba (expected HPE)")
    ax_map.set_aspect("equal", adjustable="datalim")

    ax_time.plot(metrics["time_s"], metrics["n_los"], color="tab:blue", label="n LOS satellites")
    ax_time.set_xlabel("Time since route start [s]")
    ax_time.set_ylabel("n LOS satellites", color="tab:blue")
    ax_time.tick_params(axis="y", labelcolor="tab:blue")
    ax_time.axhline(MIN_LOS_FOR_AVAILABILITY, color="tab:blue", linestyle=":", linewidth=1)

    ax_hpe = ax_time.twinx()
    ax_hpe.plot(metrics["time_s"], metrics["expected_hpe_m"], color="tab:red", label="Expected HPE [m]")
    ax_hpe.set_ylabel("Expected HPE [m]", color="tab:red")
    ax_hpe.tick_params(axis="y", labelcolor="tab:red")
    ax_time.set_title("Timeline: LOS satellite count and expected HPE")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _print_summary(metrics: dict[str, np.ndarray]) -> None:
    hpe = metrics["expected_hpe_m"]
    finite = np.isfinite(hpe)
    availability_fraction = float(metrics["available"].mean()) if metrics["available"].size else float("nan")
    print(f"epochs: {hpe.size}, availability fraction (n_LOS >= {MIN_LOS_FOR_AVAILABILITY}): {availability_fraction:.3f}")

    if finite.any():
        worst_threshold = np.nanpercentile(hpe, 90.0)
        worst_mean = float(np.nanmean(hpe[hpe >= worst_threshold]))
        print(f"worst 10% expected HPE: mean={worst_mean:.2f} m (>= {worst_threshold:.2f} m)")

        worst_idx = int(np.nanargmax(hpe))
        print(
            f"worst segment: t={metrics['time_s'][worst_idx]:.1f}s "
            f"lat={metrics['lat_deg'][worst_idx]:.6f} lon={metrics['lon_deg'][worst_idx]:.6f} "
            f"expected_hpe={hpe[worst_idx]:.2f} m n_los={metrics['n_los'][worst_idx]}"
        )
    else:
        print("no epoch had >= 4 LOS satellites; HDOP/HPE undefined everywhere")


def main() -> int:
    odaiba_dir = _REPO_ROOT / "experiments" / "data" / "urbannav" / "Odaiba"
    plateau_dir = _REPO_ROOT / "experiments" / "data" / "plateau_odaiba"
    nav_file = odaiba_dir / "base.nav"
    reference_csv = odaiba_dir / "reference.csv"

    if not (odaiba_dir.is_dir() and plateau_dir.is_dir() and nav_file.is_file() and reference_csv.is_file()):
        print(
            "demo_route_accuracy: real UrbanNav Odaiba data or the PLATEAU "
            "Odaiba mesh is not available locally "
            f"(expected {odaiba_dir} and {plateau_dir}); skipping the route "
            "accuracy demo. See experiments/data/README.md to download them."
        )
        return 0

    route = _load_route(reference_csv, step_s=2.0, max_epochs=120)
    if not route:
        print("demo_route_accuracy: reference.csv had no rows; skipping.")
        return 0

    print(f"route: {len(route)} epochs over {(route[-1][0] - route[0][0]).total_seconds():.0f} s")

    config = ScenarioConfig(
        nav_file=str(nav_file),
        route=route,
        epoch_times=[r[0] for r in route],
        constellations=["G"],
        plateau_dir=str(plateau_dir),
        elevation_mask_deg=10.0,
        diffraction_model="knife_edge",
        seed=0,
    )

    print(f"running scenario engine over {len(route)} route epochs (PLATEAU mesh + diffraction) ...")
    result = run_scenario(config)
    metrics = compute_route_metrics(result)

    out_path = _REPO_ROOT / "results" / "use_cases" / "route_accuracy.png"
    _render_png(metrics, out_path)
    print(f"wrote {out_path}")
    _print_summary(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
