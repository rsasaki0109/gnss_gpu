"""Tests for second-order (double-bounce) reflection wiring in UrbanSignalSimulator."""

import numpy as np

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
    def __init__(self, triangles, los_flags):
        self.triangles = np.asarray(triangles, dtype=np.float64)
        self._los_flags = list(los_flags)

    def check_los(self, rx_ecef, sat_ecef):
        sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        return np.asarray(self._los_flags[: sats.shape[0]], dtype=bool)

    def compute_multipath(self, rx_ecef, sat_ecef):
        sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        n = sats.shape[0]
        return np.zeros(n, dtype=np.float64), np.zeros((n, 3), dtype=np.float64)


def _corner_mesh():
    # Two orthogonal walls forming a corner; a verified double bounce exists for
    # rx=[0,0,0], sat=[0,2,0] (see test_double_reflection.py).
    return np.array(
        [
            [[5.0, -10.0, -5.0], [5.0, 10.0, -5.0], [5.0, 0.0, 5.0]],
            [[-10.0, 5.0, -5.0], [10.0, 5.0, -5.0], [0.0, 5.0, 5.0]],
        ],
        dtype=np.float64,
    )


def _geometry():
    return np.array([0.0, 0.0, 0.0]), np.array([[0.0, 2.0, 0.0]])


def test_double_reflection_opt_out_keeps_single_channel():
    rx, sats = _geometry()
    model = _StubBuildingModel(_corner_mesh(), los_flags=[True])
    usim = UrbanSignalSimulator(
        building_model=model, elevation_mask_deg=-90.0,
        max_double_reflection_paths=0)
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[1], n_samples=16)

    assert result["n_double_reflection_paths"] == 0
    assert result["double_reflection_paths"] == [[]]
    assert len(result["channels"]) == 1


def test_double_reflection_adds_replica_with_squared_coeff():
    rx, sats = _geometry()
    model = _StubBuildingModel(_corner_mesh(), los_flags=[True])
    usim = UrbanSignalSimulator(
        building_model=model, elevation_mask_deg=-90.0,
        fresnel_coeff=0.5, max_double_reflection_paths=4)
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[3], n_samples=16)

    assert result["visible"][0]
    assert len(result["double_reflection_paths"][0]) >= 1
    assert result["n_double_reflection_paths"] >= 1
    assert len(result["channels"]) == 1 + result["n_double_reflection_paths"]

    path = result["double_reflection_paths"][0][0]
    assert path.excess_delay > 0.0
    assert len(path.points) == 2

    replica = result["channels"][1]
    pr = float(np.linalg.norm(sats[0] - rx))
    expected_code_phase = (
        ((pr + float(path.excess_delay)) / C_LIGHT) * CA_CHIP_RATE
    ) % 1023.0
    np.testing.assert_allclose(
        replica["code_phase"], expected_code_phase, rtol=0.0, atol=1e-9)
    # Two specular bounces => coefficient squared (0.5 * 0.5 = 0.25).
    np.testing.assert_allclose(
        replica["amplitude"], result["channels"][0]["amplitude"] * 0.25)
