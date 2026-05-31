import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "exp_urban_shadow_lab.py"
_SPEC = importlib.util.spec_from_file_location("exp_urban_shadow_lab", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_LAB = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LAB
_SPEC.loader.exec_module(_LAB)


def test_cpu_los_batch_blocks_box_and_leaves_vertical_ray_clear():
    triangles = _LAB._box_triangles(center=(20.0, 0.0, 10.0), width=10.0, depth=20.0, height=20.0)
    rx = np.array([[0.0, 0.0, 1.5]], dtype=np.float64)
    sat = np.array(
        [
            [
                [100.0, 0.0, 10.0],
                [0.0, 0.0, 100.0],
            ]
        ],
        dtype=np.float64,
    )

    los = _LAB._cpu_los_batch(rx, sat, triangles)

    assert los.shape == (1, 2)
    assert not bool(los[0, 0])
    assert bool(los[0, 1])


def test_shadow_run_cpu_only_writes_epoch_metrics(tmp_path):
    args = argparse.Namespace(
        out_dir=tmp_path,
        n_epochs=8,
        particles_per_epoch=4,
        length_m=160.0,
        block_depth_m=36.0,
        road_half_width_m=12.0,
        building_width_m=24.0,
        base_height_m=28.0,
        height_wave_m=18.0,
        n_blocks_per_side=3,
        rx_height_m=1.6,
        sat_range_m=10000.0,
        seed=123,
        cpu_only=True,
    )

    summary, rows = _LAB.run(args)

    assert summary.n_epochs == 8
    assert summary.n_sat == len(_LAB.DEFAULT_SATELLITES)
    assert summary.route_backend == "numpy_moller_trumbore"
    assert summary.particle_backend == "numpy_moller_trumbore"
    assert 0.0 <= summary.mean_blocked_ratio <= 1.0
    assert any(row.n_nlos > 0 for row in rows)
    assert (tmp_path / "urban_shadow_epoch_summary.csv").exists()
    assert (tmp_path / "urban_shadow_summary.json").exists()
    assert (tmp_path / "urban_shadow_report.html").exists()
