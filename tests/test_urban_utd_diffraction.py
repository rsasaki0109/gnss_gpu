"""Tests for the UTD diffraction model switch in UrbanSignalSimulator."""

from types import SimpleNamespace

import numpy as np

from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
from gnss_gpu.utd_diffraction import UTDDiffractionPath
from gnss_gpu.diffraction import DiffractionPath


class _CaptureSignalGenerator:
    def __init__(self, sampling_freq=2.6e6):
        self.sampling_freq = sampling_freq
        self.channels = None

    def generate_epoch(self, channels, n_samples=None):
        self.channels = [dict(ch) for ch in channels]
        count = int(n_samples) if n_samples is not None else int(self.sampling_freq * 1e-3)
        return np.zeros(2 * count, dtype=np.float32)


class _StubBuildingModel:
    def __init__(self, los_flags):
        self._los_flags = list(los_flags)
        self.triangles = None

    def check_los(self, rx_ecef, sat_ecef):
        sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        return np.asarray(self._los_flags[: sats.shape[0]], dtype=bool)


def _edge_set():
    start = np.array([[0.0, 0.0, -100.0]])
    end = np.array([[0.0, 0.0, 100.0]])
    return SimpleNamespace(
        start=start, end=end, midpoint=0.5 * (start + end), size=1,
        face_dir_a=np.array([[0.0, -1.0, 0.0]]),
        face_dir_b=np.array([[-1.0, 0.0, 0.0]]),  # 90-degree convex corner
        wedge_n=np.array([1.5]))


_RX = np.array([50.0, 6.0, 0.0])
_SATS = np.array([[-1.0e7, 0.0, 0.0]])


def _run(model_name):
    usim = UrbanSignalSimulator(
        building_model=_StubBuildingModel(los_flags=[True]),
        elevation_mask_deg=-90.0,
        diffraction_edges=_edge_set(),
        max_diffraction_paths=2,
        diffraction_model=model_name,
        diffraction_path_kwargs=dict(
            max_ray_edge_distance_m=30.0, max_excess_path_m=200.0))
    usim.sim = _CaptureSignalGenerator()
    return usim.compute_epoch(_RX, _SATS, prn_list=[5], n_samples=16)


def test_utd_model_produces_utd_paths():
    result = _run("utd")
    paths = result["diffraction_paths"][0]
    assert len(paths) >= 1
    assert all(isinstance(p, UTDDiffractionPath) for p in paths)
    # Direct channel + one replica per diffraction path.
    assert len(result["channels"]) == 1 + len(paths)
    assert result["n_diffraction_paths"] == len(paths)


def test_knife_edge_model_produces_knife_edge_paths():
    result = _run("knife_edge")
    paths = result["diffraction_paths"][0]
    assert len(paths) >= 1
    assert all(isinstance(p, DiffractionPath) for p in paths)


def test_utd_and_knife_edge_amplitudes_differ_for_wedge():
    utd = _run("utd")["diffraction_paths"][0][0]
    ke = _run("knife_edge")["diffraction_paths"][0][0]
    # Same geometry, different physics (n=1.5 wedge vs knife-edge): amplitudes differ.
    assert abs(utd.amplitude - ke.amplitude) > 1e-4
