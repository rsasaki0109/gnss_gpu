"""Minimal demo of the unified scenario engine (gnss_gpu.scenario).

Runs 10 epochs of the scenario engine over the real Odaiba UrbanNav dataset
+ PLATEAU mesh when both are present locally
(experiments/data/urbannav/Odaiba, experiments/data/plateau_odaiba). If
either is missing, prints a clear message and exits instead of failing.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from gnss_gpu.scenario import ScenarioConfig, run_scenario  # noqa: E402

GPS_EPOCH = datetime(1980, 1, 6)


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


def main() -> int:
    odaiba_dir = _REPO_ROOT / "experiments" / "data" / "urbannav" / "Odaiba"
    plateau_dir = _REPO_ROOT / "experiments" / "data" / "plateau_odaiba"
    nav_file = odaiba_dir / "base.nav"
    reference_csv = odaiba_dir / "reference.csv"

    if not (odaiba_dir.is_dir() and plateau_dir.is_dir() and nav_file.is_file()):
        print(
            "demo_scenario_engine: real UrbanNav Odaiba data or the PLATEAU "
            "Odaiba mesh is not available locally "
            f"(expected {odaiba_dir} and {plateau_dir}); skipping the real-data "
            "demo. See experiments/data/README.md (or the fetch_urbannav_subset "
            "/ fetch_plateau_subset scripts) to download them."
        )
        return 0

    start_time, lat, lon, alt = _first_reference_fix(reference_csv)
    print(f"Odaiba fix: {start_time.isoformat()}Z, lat={lat:.6f} lon={lon:.6f} alt={alt:.2f} m")

    config = ScenarioConfig(
        nav_file=str(nav_file),
        lat_deg=lat,
        lon_deg=lon,
        alt_m=alt,
        start_time=start_time,
        duration_s=9.0,
        step_s=1.0,
        constellations=["G"],
        plateau_dir=str(plateau_dir),
        elevation_mask_deg=10.0,
        diffraction_model="knife_edge",
        seed=0,
    )

    print("running scenario engine over 10 epochs ...")
    result = run_scenario(config)
    arrays = result.to_arrays()

    n_sat_per_epoch = [ep.n_sat for ep in result.epochs]
    mean_visible = sum(n_sat_per_epoch) / len(n_sat_per_epoch) if n_sat_per_epoch else 0.0
    los_fraction = float(arrays["is_los"].mean()) if arrays["is_los"].size else float("nan")
    mean_cn0 = float(arrays["cn0_dbhz"].mean()) if arrays["cn0_dbhz"].size else float("nan")

    print(f"epochs simulated: {result.n_epochs}")
    print(f"mean visible satellites per epoch: {mean_visible:.2f}")
    print(f"LOS fraction: {los_fraction:.3f}")
    print(f"mean C/N0: {mean_cn0:.1f} dB-Hz")
    print(f"total satellite-epoch rows: {arrays['is_los'].size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
