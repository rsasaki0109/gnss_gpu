"""Dataset loading for PF smoother evaluation runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from gnss_gpu.imu import load_imu_csv


def load_pf_smoother_dataset(
    run_dir: Path,
    rover_source: str = "trimble",
    *,
    urban_data_loader: Callable[..., dict[str, Any]],
    preprocess_spp_file_func: Callable[[str, str], Iterable[tuple[Any, Any]]] | None = None,
    solve_spp_file_func: Callable[[str, str], Any] | None = None,
    imu_loader: Callable[[Path], dict[str, np.ndarray]] = load_imu_csv,
    systems: tuple[str, ...] = ("G", "E", "J"),
) -> dict[str, object]:
    """Load RINEX, UrbanNav ground truth, SPP lookup, and optional IMU data."""

    if preprocess_spp_file_func is None or solve_spp_file_func is None:
        from libgnsspp import preprocess_spp_file, solve_spp_file

        preprocess_spp_file_func = preprocess_spp_file
        solve_spp_file_func = solve_spp_file

    obs_path = str(run_dir / f"rover_{rover_source}.obs")
    nav_path = str(run_dir / "base.nav")

    epochs = preprocess_spp_file_func(obs_path, nav_path)
    sol = solve_spp_file_func(obs_path, nav_path)
    spp_records = [r for r in sol.records() if r.is_valid()]
    if not spp_records:
        raise RuntimeError(f"No valid SPP solution records for {run_dir}")
    spp_lookup = {round(r.time.tow, 1): np.array(r.position_ecef_m) for r in spp_records}

    data = urban_data_loader(run_dir, systems=systems, urban_rover=rover_source)
    gt = data["ground_truth"]
    our_times = data["times"]

    first_pos = np.array(spp_records[0].position_ecef_m[:3], dtype=np.float64)
    init_meas = None
    for sol_epoch, measurements in epochs:
        if sol_epoch.is_valid() and len(measurements) >= 4:
            init_meas = measurements
            first_pos = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
            break
    if init_meas is None:
        raise RuntimeError(f"No valid epoch for init in {run_dir}")

    init_cb = float(
        np.median(
            [
                m.corrected_pseudorange
                - np.linalg.norm(np.asarray(m.satellite_ecef, dtype=np.float64) - first_pos)
                for m in init_meas
            ]
        )
    )
    result = {
        "epochs": epochs,
        "spp_lookup": spp_lookup,
        "gt": gt,
        "our_times": our_times,
        "first_pos": first_pos,
        "init_cb": init_cb,
    }

    imu_path = run_dir / "imu.csv"
    if imu_path.exists():
        result["imu_data"] = imu_loader(imu_path)
        print(f"  [IMU] loaded {len(result['imu_data']['tow'])} samples from {imu_path}")
    else:
        result["imu_data"] = None

    return result
