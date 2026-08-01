from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/compose_ppc_safe_trajectory.py"
SPEC = importlib.util.spec_from_file_location("compose_ppc_safe_trajectory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compose_never_inherits_primary_fix_and_overlays_safe_fix(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pos"
    primary.write_text(
        "% pos\n0 1.0 10 0 0 0 0 0 4\n0 1.2 20 0 0 0 0 0 4\n",
        encoding="utf-8",
    )
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "tow,shadow_fixed,x,y,z\n1.0,0,11,0,0\n1.2,1,21,0,0\n1.4,0,30,0,0\n",
        encoding="utf-8",
    )

    rows = MODULE.compose(primary, tracker)

    assert [row["status"] for row in rows] == [3, 4, 3]
    assert [row["source"] for row in rows] == [
        "tracker_float_fallback",
        "safe_imu_pf_fgo_fixed",
        "tracker_float_fallback",
    ]
    assert [row["x"] for row in rows] == [11.0, 21.0, 30.0]


def test_compose_causally_reanchors_tracker_across_invalid_primary() -> None:
    primary = {
        1.0: np.array([6_400_000.0, 0.0, 0.0]),
        1.2: np.array([8_000_000.0, 0.0, 0.0]),
        1.4: np.array([8_000_001.0, 0.0, 0.0]),
    }
    tracker = {
        1.0: {"fixed": False, "position": (6_300_000.0, 0.0, 0.0)},
        1.2: {"fixed": False, "position": (6_300_001.0, 0.0, 0.0)},
        1.4: {"fixed": False, "position": (6_300_003.0, 0.0, 0.0)},
    }
    original_read_estimates = MODULE.read_estimates
    original_read_tracker = MODULE.read_safe_tracker
    MODULE.read_estimates = lambda _: (primary, {})
    MODULE.read_safe_tracker = lambda _: tracker
    try:
        rows = MODULE.compose(Path("primary.pos"), Path("tracker.csv"))
    finally:
        MODULE.read_estimates = original_read_estimates
        MODULE.read_safe_tracker = original_read_tracker

    assert [row["source"] for row in rows] == [
        "primary_float",
        "causal_tracker_integrity_bridge",
        "causal_tracker_integrity_bridge",
    ]
    assert [row["x"] for row in rows] == [6_400_000.0, 6_400_001.0, 6_400_003.0]
