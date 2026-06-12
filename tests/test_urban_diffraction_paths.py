"""Tests for knife-edge diffraction-path wiring in UrbanSignalSimulator.

Uses a capture signal generator and either precomputed diffraction edges or a
pure-Python stub building model, so the suite runs without the compiled CUDA
ray-trace extension.
"""

from types import SimpleNamespace

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
    """LOS stubbed in Python; exposes triangles for lazy edge extraction."""

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


def _edges():
    start = np.array([[100.0, 0.0, 0.0]])
    end = np.array([[100.0, 0.0, 50.0]])
    return SimpleNamespace(
        start=start,
        end=end,
        midpoint=0.5 * (start + end),
        length_m=np.array([50.0]),
        dihedral_deg=np.array([90.0]),
        is_boundary=np.array([False]),
        size=1,
    )


def _geometry():
    rx = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sats = np.array([[200.0, 0.0, 20.0]], dtype=np.float64)
    return rx, sats


def test_diffraction_opt_out_keeps_single_channel():
    rx, sats = _geometry()
    usim = UrbanSignalSimulator(
        building_model=None,
        elevation_mask_deg=-90.0,
        max_diffraction_paths=0,
        diffraction_edges=_edges(),
    )
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[1], n_samples=16)

    assert result["n_diffraction_paths"] == 0
    assert result["diffraction_paths"] == [[]]
    assert len(result["channels"]) == 1


def test_precomputed_edges_add_diffraction_replica():
    rx, sats = _geometry()
    usim = UrbanSignalSimulator(
        building_model=None,
        elevation_mask_deg=-90.0,
        max_diffraction_paths=2,
        diffraction_edges=_edges(),
    )
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[5], n_samples=16)

    assert result["visible"][0]
    assert len(result["diffraction_paths"][0]) >= 1
    assert result["n_diffraction_paths"] >= 1
    assert len(result["channels"]) == 1 + result["n_diffraction_paths"]

    path = result["diffraction_paths"][0][0]
    assert 0.0 < path.amplitude <= 1.0
    assert path.attenuation_db >= 0.0

    replica = result["channels"][1]
    pr = float(np.linalg.norm(sats[0] - rx))
    expected_code_phase = (
        ((pr + float(path.excess_delay)) / C_LIGHT) * CA_CHIP_RATE
    ) % 1023.0
    np.testing.assert_allclose(
        replica["code_phase"], expected_code_phase, rtol=0.0, atol=1e-9
    )
    # Replica amplitude = direct amplitude * knife-edge amplitude.
    np.testing.assert_allclose(
        replica["amplitude"],
        result["channels"][0]["amplitude"] * float(path.amplitude),
    )


def test_lazy_edge_extraction_from_building_triangles():
    rx, sats = _geometry()
    # A tall thin wall around x=100 provides a vertical roof/side edge near the
    # rx->sat ray; extract_diffraction_edges keeps welded box edges.
    mesh = BuildingModel.create_box(
        np.array([100.0, 0.0, 25.0], dtype=np.float64), 2.0, 40.0, 50.0
    )
    model = _StubBuildingModel(mesh.triangles, los_flags=[False])
    usim = UrbanSignalSimulator(
        building_model=model,
        elevation_mask_deg=-90.0,
        max_diffraction_paths=4,
        diffraction_edge_kwargs={
            "include_boundary_edges": True,
            "min_edge_length_m": 1.0,
            "min_dihedral_deg": 20.0,
        },
        diffraction_path_kwargs={"max_ray_edge_distance_m": 30.0},
    )
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[11], n_samples=16)

    edges = usim._get_diffraction_edges()
    assert edges is not None and int(edges.size) > 0
    assert result["n_diffraction_paths"] >= 1
    assert len(result["channels"]) == 1 + result["n_diffraction_paths"]
