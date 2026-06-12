"""Tests for reflection+diffraction composite wiring in UrbanSignalSimulator."""

from types import SimpleNamespace

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


# Verified composite scene (see test_reflection_diffraction.py): wall x=5, an
# edge straddling the rx-image ray (RD) and one near the sat-image ray (DR).
_WALL = np.array(
    [[[5.0, -50.0, -50.0], [5.0, 50.0, -50.0], [5.0, 0.0, 50.0]]], dtype=float)
_RX = np.array([0.0, 0.0, 0.0])
_SATS = np.array([[-20.0, 60.0, 0.0]])


def _edge_set():
    start = np.array([[0.0, 20.0, -3.0], [3.0, 6.0, -3.0]])
    end = np.array([[0.0, 20.0, 3.0], [3.0, 6.0, 3.0]])
    return SimpleNamespace(
        start=start, end=end, midpoint=0.5 * (start + end), size=2)


def test_opt_out_keeps_single_channel():
    model = _StubBuildingModel(_WALL, los_flags=[True])
    usim = UrbanSignalSimulator(
        building_model=model, elevation_mask_deg=-90.0,
        diffraction_edges=_edge_set(), max_reflection_diffraction_paths=0)
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(_RX, _SATS, prn_list=[1], n_samples=16)

    assert result["n_reflection_diffraction_paths"] == 0
    assert result["reflection_diffraction_paths"] == [[]]
    assert len(result["channels"]) == 1


def test_adds_composite_replica_with_product_coeff():
    model = _StubBuildingModel(_WALL, los_flags=[True])
    usim = UrbanSignalSimulator(
        building_model=model, elevation_mask_deg=-90.0,
        fresnel_coeff=0.5, diffraction_edges=_edge_set(),
        max_reflection_diffraction_paths=4)
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(_RX, _SATS, prn_list=[7], n_samples=16)

    paths = result["reflection_diffraction_paths"][0]
    assert len(paths) >= 1
    assert result["n_reflection_diffraction_paths"] == len(paths)
    # LOS direct channel + one replica per composite path.
    assert len(result["channels"]) == 1 + len(paths)

    direct = result["channels"][0]
    replicas = result["channels"][1:]
    for ch, path in zip(replicas, paths):
        # Amplitude = direct * fresnel_coeff * knife-edge amplitude.
        expected_amp = direct["amplitude"] * 0.5 * path.amplitude
        assert ch["amplitude"] == expected_amp
        # Code phase shifted by the composite excess delay.
        mp_pr = (direct["code_phase"] / CA_CHIP_RATE * C_LIGHT) % C_LIGHT  # sanity
        expected_code = (
            (direct["code_phase"]
             + (path.excess_delay / C_LIGHT) * CA_CHIP_RATE) % 1023.0)
        assert ch["code_phase"] == expected_code


def test_material_uses_reflection_coefficient():
    model = _StubBuildingModel(_WALL, los_flags=[True])
    usim = UrbanSignalSimulator(
        building_model=model, elevation_mask_deg=-90.0,
        reflector_material="concrete", diffraction_edges=_edge_set(),
        max_reflection_diffraction_paths=4)
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(_RX, _SATS, prn_list=[9], n_samples=16)
    paths = result["reflection_diffraction_paths"][0]
    assert len(paths) >= 1
    # With a real material the per-path reflection coefficient is applied, so
    # replica amplitude stays positive and below the diffraction-only amplitude.
    replicas = result["channels"][1:]
    direct_amp = result["channels"][0]["amplitude"]
    for ch, path in zip(replicas, paths):
        assert 0.0 < ch["amplitude"] < direct_amp * path.amplitude + 1e-12
