"""Demo of the GPU area-sweep GNSS coverage/accuracy prediction map
(gnss_gpu.coverage_map).

Sweeps a ~30x30 cell grid x 10 epochs over the real Odaiba UrbanNav dataset +
PLATEAU mesh when both are present locally
(experiments/data/urbannav/Odaiba, experiments/data/plateau_odaiba). Writes a
docs-quality PNG and a self-contained deck.gl HTML heatmap into
results/coverage_map/. If either input is missing, prints a clear message and
exits instead of failing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnss_gpu.coverage_map import (  # noqa: E402
    CoverageMapConfig,
    run_coverage_map,
    to_deckgl_html,
    to_png,
)
from demo_scenario_engine import _first_reference_fix  # noqa: E402


def main() -> int:
    odaiba_dir = _REPO_ROOT / "experiments" / "data" / "urbannav" / "Odaiba"
    plateau_dir = _REPO_ROOT / "experiments" / "data" / "plateau_odaiba"
    nav_file = odaiba_dir / "base.nav"
    reference_csv = odaiba_dir / "reference.csv"

    if not (odaiba_dir.is_dir() and plateau_dir.is_dir() and nav_file.is_file()):
        print(
            "demo_coverage_map: real UrbanNav Odaiba data or the PLATEAU "
            "Odaiba mesh is not available locally "
            f"(expected {odaiba_dir} and {plateau_dir}); skipping the "
            "coverage map demo. See experiments/data/README.md (or the "
            "fetch_urbannav_subset / fetch_plateau_subset scripts) to "
            "download them."
        )
        return 0

    start_time, lat, lon, alt = _first_reference_fix(reference_csv)
    print(f"Odaiba grid center: {start_time.isoformat()}Z, lat={lat:.6f} lon={lon:.6f}")

    config = CoverageMapConfig(
        nav_file=str(nav_file),
        center_lat_deg=lat,
        center_lon_deg=lon,
        ground_alt_m=alt,
        receiver_height_m=1.5,
        extent_east_m=300.0,
        extent_north_m=300.0,
        cell_size_m=10.0,
        start_time=start_time,
        duration_s=270.0,
        step_s=30.0,
        constellations=["G"],
        plateau_dir=str(plateau_dir),
        elevation_mask_deg=10.0,
        uere_m=5.0,
    )
    n_east = round(config.extent_east_m / config.cell_size_m)
    n_north = round(config.extent_north_m / config.cell_size_m)

    print(f"running coverage map: {n_north}x{n_east} cells ...")
    t0 = time.perf_counter()
    result = run_coverage_map(config)
    wall_s = time.perf_counter() - t0

    n_cells = result.mean_visible.size
    n_building = int(np.sum(np.isnan(result.mean_visible)))
    finite_hpe = result.expected_hpe_m[np.isfinite(result.expected_hpe_m)]

    print(f"grid: {result.shape[0]}x{result.shape[1]} cells, epochs: {len(result.epoch_times)}")
    print(f"cells inside a building footprint (masked NaN): {n_building}/{n_cells}")
    print(f"mean visible satellites: {np.nanmean(result.mean_visible):.2f}")
    print(f"mean LOS satellites:     {np.nanmean(result.mean_los):.2f}")
    print(f"mean LOS fraction:       {np.nanmean(result.los_fraction):.3f}")
    print(f"mean availability (>=4 LOS): {np.nanmean(result.availability):.3f}")
    if finite_hpe.size:
        print(
            f"expected HPE [m]: mean={finite_hpe.mean():.2f} "
            f"median={np.median(finite_hpe):.2f} p90={np.percentile(finite_hpe, 90):.2f}"
        )
    else:
        print("expected HPE [m]: no cell had >=4 LOS satellites in any epoch")
    print(f"wall time: {wall_s:.2f} s for {n_cells} cells x {len(result.epoch_times)} epochs")

    out_dir = _REPO_ROOT / "results" / "coverage_map"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "odaiba_expected_hpe.png"
    html_path = out_dir / "odaiba_expected_hpe_deckgl.html"
    to_png(result, png_path, metric="expected_hpe_m")
    to_deckgl_html(result, html_path, metric="expected_hpe_m")

    print(f"PNG:  {png_path}")
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
