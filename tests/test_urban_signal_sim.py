"""Tests for urban signal simulation helpers."""

import numpy as np
import pytest

from gnss_gpu.urban_signal_sim import (
    UrbanSignalSimulator,
    C_LIGHT,
    CA_CHIP_RATE,
    GPS_L1_WAVELENGTH,
)


class _CaptureSignalGenerator:
    def __init__(self, sampling_freq: float = 2.6e6) -> None:
        self.sampling_freq = float(sampling_freq)
        self.channels = None

    def generate_epoch(self, channels, n_samples=None):
        self.channels = list(channels)
        if n_samples is None:
            n_samples = int(self.sampling_freq * 1e-3)
        return np.zeros(2 * int(n_samples), dtype=np.float32)


def test_compute_epoch_applies_satellite_clock_to_pseudorange():
    """Satellite clock bias should shift the generated code phase."""
    usim = UrbanSignalSimulator(elevation_mask_deg=0.0)
    capture = _CaptureSignalGenerator()
    usim.sim = capture

    rx = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    sat = np.array([[6378137.0 + 20200000.0, 0.0, 0.0]], dtype=np.float64)
    sat_clk = np.array([1.0e-6], dtype=np.float64)

    result = usim.compute_epoch(
        rx_ecef=rx,
        sat_ecef=sat,
        sat_clk=sat_clk,
        prn_list=[1],
        n_samples=32,
    )

    assert result["visible"][0]
    assert capture.channels is not None
    assert len(capture.channels) == 1

    geom_range = float(np.linalg.norm(sat[0] - rx))
    expected_pr = geom_range - C_LIGHT * sat_clk[0]
    expected_code_phase = ((expected_pr / C_LIGHT) * CA_CHIP_RATE) % 1023.0
    expected_carrier_phase = (
        (expected_pr / GPS_L1_WAVELENGTH) * 2.0 * np.pi
    ) % (2.0 * np.pi)

    assert capture.channels[0]["code_phase"] == pytest.approx(expected_code_phase)
    assert capture.channels[0]["carrier_phase"] == pytest.approx(expected_carrier_phase)


def test_compute_epoch_rejects_mismatched_satellite_clock_length():
    usim = UrbanSignalSimulator(elevation_mask_deg=0.0)
    usim.sim = _CaptureSignalGenerator()

    rx = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    sats = np.array([
        [6378137.0 + 20200000.0, 0.0, 0.0],
        [6378137.0 + 21000000.0, 1.0, 0.0],
    ], dtype=np.float64)

    with pytest.raises(ValueError, match="sat_clk must have one entry per satellite"):
        usim.compute_epoch(
            rx_ecef=rx,
            sat_ecef=sats,
            sat_clk=np.array([0.0], dtype=np.float64),
            prn_list=[1, 2],
            n_samples=16,
        )
