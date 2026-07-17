"""Use-case demo: RTK base-station placement evaluation.

Ranks candidate RTK base-station sites in a city area by predicted GNSS
quality (from :func:`gnss_gpu.coverage_map.run_coverage_map`) and by
*common-view* satellite count against a rover route (from
:func:`gnss_gpu.scenario.run_scenario`) -- the metric that actually decides
whether a base can differentially correct a given rover, since RTK requires
satellites both receivers track in the clear at the same time.

Pipeline, over the real Odaiba UrbanNav dataset + PLATEAU mesh when both are
present locally (experiments/data/urbannav/Odaiba,
experiments/data/plateau_odaiba):

1. Sweep a ~25x25 cell coverage map (:mod:`gnss_gpu.coverage_map`) around the
   start of the rover's recorded route.
2. Filter to candidate sites that sit on open ground (not inside a building
   footprint) with 100% >=4-satellite availability over the swept epochs,
   then rank by (mean LOS satellites desc, HDOP asc).
3. For the top ~5 candidates, run the scenario engine
   (:mod:`gnss_gpu.scenario`) on a ~30-epoch subsample of the rover's
   recorded route and on each candidate (fixed point, same epochs), and
   score each candidate by the mean per-epoch count of LOS satellites shared
   with the rover (matched by sat_id).
4. Write a heatmap PNG (mean LOS satellites) with the rover route and the
   top-5 candidates marked, and print a ranked table.

If either input is missing, prints a clear message and exits instead of
failing.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnss_gpu.coverage_map import CoverageMapConfig, CoverageMapResult, run_coverage_map  # noqa: E402
from gnss_gpu.scenario import EpochRecord, ScenarioConfig, run_scenario  # noqa: E402
from demo_scenario_engine import _first_reference_fix  # noqa: E402

GPS_EPOCH = datetime(1980, 1, 6)


# ---------------------------------------------------------------------------
# Pure logic (importable, no GPU/mesh/network) -- exercised directly by
# tests/test_demo_rtk_base_placement.py.
# ---------------------------------------------------------------------------


def filter_and_rank_candidates(result: CoverageMapResult, top_n: int = 5) -> list[dict]:
    """Rank :func:`~gnss_gpu.coverage_map.run_coverage_map` cells as RTK
    base-station candidates.

    A cell qualifies when it sits on open ground -- ``mean_los``/``hdop`` are
    finite, i.e. not masked NaN for standing inside a building footprint --
    and has 100% >=4-satellite availability over the swept epochs (a base
    that sometimes drops below 4 satellites cannot reliably anchor RTK).
    Qualifying cells are ranked by (``mean_los`` desc, ``hdop`` asc): more
    line-of-sight satellites first, tighter geometry as the tiebreaker.

    Returns up to ``top_n`` dicts with ``row``, ``col``, ``lat``, ``lon``,
    ``mean_los``, ``hdop``.
    """
    mean_los = result.mean_los
    hdop = result.hdop
    availability = result.availability
    n_north, n_east = mean_los.shape

    candidates: list[dict] = []
    for row in range(n_north):
        for col in range(n_east):
            los = mean_los[row, col]
            if not np.isfinite(los):
                continue  # inside a building footprint
            h = hdop[row, col]
            if not np.isfinite(h):
                continue
            if availability[row, col] != 1.0:
                continue
            candidates.append(
                {
                    "row": row,
                    "col": col,
                    "lat": float(result.cell_lat_deg[row, col]),
                    "lon": float(result.cell_lon_deg[row, col]),
                    "mean_los": float(los),
                    "hdop": float(h),
                }
            )

    candidates.sort(key=lambda c: (-c["mean_los"], c["hdop"]))
    return candidates[:top_n]


def common_view_score(base_epochs: list[EpochRecord], rover_epochs: list[EpochRecord]) -> float:
    """Mean, over epoch-matched pairs, of the LOS satellites shared between
    ``base_epochs[i]`` and ``rover_epochs[i]`` (matched by ``sat_id``).

    RTK needs satellites both receivers track in the clear at the same
    epoch, so raw LOS count at the base alone is not enough -- this is the
    metric that determines whether a candidate site can actually support RTK
    against a given rover route. ``base_epochs`` and ``rover_epochs`` are
    assumed to already be epoch-aligned (same length, same epoch times);
    only the shorter length is used if they differ.
    """
    n = min(len(base_epochs), len(rover_epochs))
    if n == 0:
        return 0.0
    counts = np.empty(n, dtype=np.float64)
    for i in range(n):
        base_los = set(base_epochs[i].sat_id[base_epochs[i].is_los].tolist())
        rover_los = set(rover_epochs[i].sat_id[rover_epochs[i].is_los].tolist())
        counts[i] = len(base_los & rover_los)
    return float(counts.mean())


# ---------------------------------------------------------------------------
# Rover route loading
# ---------------------------------------------------------------------------


def _load_reference_route(
    reference_csv: Path, max_points: int | None = None
) -> list[tuple[datetime, float, float, float]]:
    """Read the full (time, lat_deg, lon_deg, alt_m) rover ground-truth route.

    ``max_points`` evenly subsamples by row index down to about that many
    points -- used both to shrink the plotted route and to build the ~30
    epoch subsample fed to :func:`~gnss_gpu.scenario.run_scenario` for
    common-view scoring.
    """
    rows: list[tuple[datetime, float, float, float]] = []
    with open(reference_csv, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            week = int(float(row["GPS Week"]))
            tow = float(row["GPS TOW (s)"])
            lat = float(row["Latitude (deg)"])
            lon = float(row["Longitude (deg)"])
            alt = float(row["Ellipsoid Height (m)"])
            rows.append((GPS_EPOCH + timedelta(weeks=week, seconds=tow), lat, lon, alt))

    if max_points is not None and len(rows) > max_points:
        idx = sorted(set(np.linspace(0, len(rows) - 1, max_points).round().astype(int).tolist()))
        rows = [rows[i] for i in idx]
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_plot(
    result: CoverageMapResult,
    candidates: list[dict],
    route: list[tuple[datetime, float, float, float]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = result.mean_los
    lon = result.cell_lon_deg
    lat = result.cell_lat_deg

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    im = ax.imshow(
        values,
        origin="lower",
        cmap="viridis",
        extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="Mean LOS satellites")

    route_lon = [r[2] for r in route]
    route_lat = [r[1] for r in route]
    ax.plot(
        route_lon, route_lat, color="white", linewidth=1.2, alpha=0.85,
        label="rover route", zorder=3,
    )

    cand_lon = [c["lon"] for c in candidates]
    cand_lat = [c["lat"] for c in candidates]
    ax.scatter(
        cand_lon, cand_lat, marker="*", s=220, color="red", edgecolor="black",
        linewidth=0.8, zorder=4, label="top-5 candidates",
    )
    for rank, c in enumerate(candidates, start=1):
        ax.annotate(
            f"#{rank} cv={c.get('common_view', float('nan')):.1f}",
            (c["lon"], c["lat"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"),
        )

    # Keep the grid prominent even though the recorded route runs well beyond
    # it; only the portion of the route near the candidate area is relevant.
    lon_pad = (float(lon.max()) - float(lon.min())) * 0.15
    lat_pad = (float(lat.max()) - float(lat.min())) * 0.15
    ax.set_xlim(float(lon.min()) - lon_pad, float(lon.max()) + lon_pad)
    ax.set_ylim(float(lat.min()) - lat_pad, float(lat.max()) + lat_pad)

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("RTK base-station placement -- mean LOS satellites")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    odaiba_dir = _REPO_ROOT / "experiments" / "data" / "urbannav" / "Odaiba"
    plateau_dir = _REPO_ROOT / "experiments" / "data" / "plateau_odaiba"
    nav_file = odaiba_dir / "base.nav"
    reference_csv = odaiba_dir / "reference.csv"

    if not (
        odaiba_dir.is_dir()
        and plateau_dir.is_dir()
        and nav_file.is_file()
        and reference_csv.is_file()
    ):
        print(
            "demo_rtk_base_placement: real UrbanNav Odaiba data or the PLATEAU "
            "Odaiba mesh is not available locally "
            f"(expected {odaiba_dir} and {plateau_dir}); skipping the RTK base "
            "placement demo. See experiments/data/README.md (or the "
            "fetch_urbannav_subset / fetch_plateau_subset scripts) to download "
            "them."
        )
        return 0

    t_start = time.perf_counter()

    start_time, center_lat, center_lon, center_alt = _first_reference_fix(reference_csv)
    print(f"Odaiba grid center: {start_time.isoformat()}Z, lat={center_lat:.6f} lon={center_lon:.6f}")

    # --- Step 1: coverage map over the candidate area -----------------------
    receiver_height_m = 1.5
    cov_config = CoverageMapConfig(
        nav_file=str(nav_file),
        center_lat_deg=center_lat,
        center_lon_deg=center_lon,
        ground_alt_m=center_alt,
        receiver_height_m=receiver_height_m,
        extent_east_m=250.0,
        extent_north_m=250.0,
        cell_size_m=10.0,
        start_time=start_time,
        duration_s=270.0,
        step_s=30.0,
        constellations=["G"],
        plateau_dir=str(plateau_dir),
        elevation_mask_deg=10.0,
        uere_m=5.0,
    )
    n_east = round(cov_config.extent_east_m / cov_config.cell_size_m)
    n_north = round(cov_config.extent_north_m / cov_config.cell_size_m)
    print(f"running coverage map: {n_north}x{n_east} cells ...")

    t0 = time.perf_counter()
    coverage_result = run_coverage_map(cov_config)
    coverage_wall_s = time.perf_counter() - t0
    print(
        f"coverage map: {coverage_result.shape[0]}x{coverage_result.shape[1]} cells, "
        f"{len(coverage_result.epoch_times)} epochs, {coverage_wall_s:.2f} s"
    )

    # --- Step 2: filter + rank candidates ------------------------------------
    candidates = filter_and_rank_candidates(coverage_result, top_n=5)
    if not candidates:
        print(
            "no candidate cells passed the open-ground / 100% availability "
            "filter; cannot rank RTK base-station sites."
        )
        return 0
    print(
        f"{len(candidates)} candidate site(s) passed the filter "
        f"(of {coverage_result.mean_los.size} cells)"
    )

    # --- Step 3: common-view score vs a subsampled rover route --------------
    full_route = _load_reference_route(reference_csv)
    plot_route = _load_reference_route(reference_csv, max_points=400)
    rover_route = _load_reference_route(reference_csv, max_points=30)
    rover_epoch_times = [r[0] for r in rover_route]
    print(
        f"rover route: {len(full_route)} raw fixes -> {len(rover_route)} epochs "
        "for common-view scoring"
    )

    t1 = time.perf_counter()
    rover_config = ScenarioConfig(
        nav_file=str(nav_file),
        route=rover_route,
        epoch_times=rover_epoch_times,
        constellations=["G"],
        plateau_dir=str(plateau_dir),
        elevation_mask_deg=10.0,
        diffraction_model=None,
        max_reflection_paths=0,
        seed=0,
    )
    rover_result = run_scenario(rover_config)

    candidate_alt_m = center_alt + receiver_height_m
    for cand in candidates:
        cand_config = ScenarioConfig(
            nav_file=str(nav_file),
            lat_deg=cand["lat"],
            lon_deg=cand["lon"],
            alt_m=candidate_alt_m,
            epoch_times=rover_epoch_times,
            constellations=["G"],
            plateau_dir=str(plateau_dir),
            elevation_mask_deg=10.0,
            diffraction_model=None,
            max_reflection_paths=0,
            seed=0,
        )
        cand_result = run_scenario(cand_config)
        cand["common_view"] = common_view_score(cand_result.epochs, rover_result.epochs)
    scenario_wall_s = time.perf_counter() - t1

    # --- Step 4: report -------------------------------------------------------
    print()
    header = f"{'rank':>4}  {'lat':>11}  {'lon':>12}  {'mean_los':>8}  {'hdop':>6}  {'common_view':>11}"
    print(header)
    for i, cand in enumerate(candidates, start=1):
        print(
            f"{i:>4}  {cand['lat']:>11.6f}  {cand['lon']:>12.6f}  "
            f"{cand['mean_los']:>8.2f}  {cand['hdop']:>6.2f}  {cand['common_view']:>11.2f}"
        )

    out_path = _REPO_ROOT / "results" / "use_cases" / "rtk_base_placement.png"
    _render_plot(coverage_result, candidates, plot_route, out_path)

    total_wall_s = time.perf_counter() - t_start
    print()
    print(f"PNG: {out_path}")
    print(
        f"wall time: coverage map {coverage_wall_s:.2f} s, common-view scoring "
        f"{scenario_wall_s:.2f} s, total {total_wall_s:.2f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
