"""Tests for physical reflection-path wiring in UrbanSignalSimulator.

These use a pure-Python stub building model so the test runs without the
compiled CUDA ray-trace extension. ``check_los`` / ``compute_multipath`` are
stubbed in Python, while ``compute_reflection_paths`` delegates to a real
``BuildingModel`` (whose reflection-path method is itself pure Python).
"""

import numpy as np

from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.urban_signal_sim import (
    CA_CHIP_RATE,
    C_LIGHT,
    UrbanSignalSimulator,
)


class _CaptureSignalGenerator:
    def __init__(self, sampling_freq=2.6e6):
        self.sampling_freq = sampling_freq
        self.channels = None

    def generate_epoch(self, channels, n_samples=None):
        self.channels = [dict(ch) for ch in channels]
        count = int(n_samples) if n_samples is not None else int(self.sampling_freq * 1e-3)
        return np.zeros(2 * count, dtype=np.float32)


class _StubBuildingModel:
    """Pure-Python building model: LOS/multipath stubbed, reflections delegated."""

    def __init__(self, mesh_model, los_flags):
        self._mesh = mesh_model
        self._los_flags = list(los_flags)

    def check_los(self, rx_ecef, sat_ecef):
        sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        return np.asarray(self._los_flags[: sats.shape[0]], dtype=bool)

    def compute_multipath(self, rx_ecef, sat_ecef):
        sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        n = sats.shape[0]
        return np.zeros(n, dtype=np.float64), np.zeros((n, 3), dtype=np.float64)

    def compute_reflection_paths(self, rx_ecef, sat_ecef, max_paths=4):
        return self._mesh.compute_reflection_paths(rx_ecef, sat_ecef, max_paths=max_paths)


def _wall_mesh():
    # Reflection geometry proven in test_reflection_paths.py: a thin vertical
    # wall yields exactly one first-order reflection for this rx/sat pair.
    return BuildingModel.create_box(
        np.array([0.05, 0.0, 0.0], dtype=np.float64), 0.1, 20.0, 20.0
    )


def _reflection_geometry():
    rx = np.array([-10.0, -3.0, 0.0], dtype=np.float64)
    sats = np.array([[-20.0, 9.0, 0.0]], dtype=np.float64)
    return rx, sats


def test_default_mode_keeps_single_los_channel_without_buildings():
    rx = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    sats = np.array([[6378137.0 + 20200000.0, 0.0, 0.0]], dtype=np.float64)

    usim = UrbanSignalSimulator(building_model=None, elevation_mask_deg=-90.0)
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[1], n_samples=16)

    assert len(result["channels"]) == 1
    assert result["n_reflection_paths"] == 0
    assert result["reflection_paths"] == [[]]


def test_opt_in_reflection_paths_add_replica_channel_and_code_phase():
    rx, sats = _reflection_geometry()
    model = _StubBuildingModel(_wall_mesh(), los_flags=[False])
    usim = UrbanSignalSimulator(
        building_model=model,
        elevation_mask_deg=-90.0,
        fresnel_coeff=0.5,
        max_reflection_paths=2,
    )
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[3], n_samples=16)

    assert result["visible"][0]
    assert len(result["reflection_paths"]) == 1
    assert len(result["reflection_paths"][0]) >= 1
    assert result["n_reflection_paths"] >= 1
    assert len(result["channels"]) == 1 + result["n_reflection_paths"]

    path = result["reflection_paths"][0][0]
    replica = result["channels"][1]
    pr = float(np.linalg.norm(sats[0] - rx))
    expected_code_phase = (((pr + float(path.excess_delay)) / C_LIGHT) * CA_CHIP_RATE) % 1023.0

    np.testing.assert_allclose(replica["code_phase"], expected_code_phase, rtol=0.0, atol=1e-9)
    # NLOS direct amplitude is attenuated; replica scaled by fresnel_coeff.
    np.testing.assert_allclose(replica["amplitude"], result["channels"][0]["amplitude"] * 0.5)


def test_reflection_path_keys_are_populated_in_opt_in_mode():
    rx, sats = _reflection_geometry()
    model = _StubBuildingModel(_wall_mesh(), los_flags=[True])
    usim = UrbanSignalSimulator(
        building_model=model,
        elevation_mask_deg=-90.0,
        max_reflection_paths=1,
    )
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[7], n_samples=16)

    assert "reflection_paths" in result
    assert "n_reflection_paths" in result
    assert len(result["reflection_paths"][0]) >= 1
    assert result["n_reflection_paths"] >= 1


def test_reflection_paths_are_empty_when_opted_out_with_buildings():
    rx, sats = _reflection_geometry()
    model = _StubBuildingModel(_wall_mesh(), los_flags=[True])
    usim = UrbanSignalSimulator(
        building_model=model,
        elevation_mask_deg=-90.0,
        max_reflection_paths=0,
    )
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[9], n_samples=16)

    assert result["n_reflection_paths"] == 0
    assert all(paths == [] for paths in result["reflection_paths"])
