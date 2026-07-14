import numpy as np

from gnss_gpu.doppler import doppler_velocity
from gnss_gpu.doppler_signals import (
    C_LIGHT_MPS,
    GPS_L1_WAVELENGTH_M,
    carrier_frequency_hz,
    doppler_wavelengths_m,
    fit_constellation_clock_drifts,
    normalize_constellation_clock_drifts,
    normalize_doppler_to_reference,
)


def test_signal_frequencies_cover_g_e_j_c_r():
    assert carrier_frequency_hz("G01", "D1C") == 1575.42e6
    assert carrier_frequency_hz("E11", "D7Q") == 1207.14e6
    assert carrier_frequency_hz("J02", "D6L") == 1278.75e6
    assert carrier_frequency_hz("C19", "D2I") == 1561.098e6
    assert carrier_frequency_hz(
        "R05", "D1C", glonass_frequency_channels={"R05": -7}
    ) == 1602.0e6 - 7 * 0.5625e6


def test_unknown_glonass_channel_abstains():
    assert np.isnan(carrier_frequency_hz("R05", "D1C"))


def test_reference_normalization_preserves_range_rate():
    sats = ["G01", "E11", "J02", "C19", "R05"]
    codes = ["D1C", "D7Q", "D5Q", "D2I", "D2C"]
    wavelengths = doppler_wavelengths_m(
        sats, codes, glonass_frequency_channels={"R05": 3}
    )
    doppler = np.array([-1000.0, 800.0, -500.0, 1200.0, -700.0])
    equivalent = normalize_doppler_to_reference(doppler, wavelengths)
    assert np.allclose(equivalent * GPS_L1_WAVELENGTH_M, doppler * wavelengths)


def test_mixed_signal_doppler_solver_uses_per_row_wavelengths():
    sat = np.array([[20e6, 0, 0], [0, 20e6, 0], [0, 0, 20e6], [-20e6, -20e6, -20e6]])
    sat_vel = np.zeros_like(sat)
    rx = np.zeros(3)
    wavelengths = C_LIGHT_MPS / np.array([1575.42e6, 1207.14e6, 1176.45e6, 1561.098e6])
    truth = np.array([3.0, -2.0, 1.0, 0.4])
    los = sat / np.linalg.norm(sat, axis=1)[:, None]
    range_rate = -los @ truth[:3] + truth[3]
    doppler = range_rate / wavelengths
    solved, _iterations = doppler_velocity(
        sat, sat_vel, doppler, rx, wavelength=wavelengths
    )
    assert np.allclose(solved, truth, atol=1e-8)


def test_constellation_clock_drift_fit_recovers_velocity_and_each_clock():
    directions = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0],
         [0, -1, 0], [0, 0, -1], [1, 1, 1], [-1, 1, -1]], dtype=float
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sat = directions * 21e6
    sat_vel = np.zeros_like(sat)
    groups = np.array([0, 0, 0, 0, 2, 2, 2, 2])
    velocity = np.array([4.0, -1.5, 0.8])
    clocks = np.where(groups == 0, 2.0, -3.5)
    wavelengths = C_LIGHT_MPS / np.array(
        [1575.42e6, 1575.42e6, 1227.60e6, 1176.45e6,
         1575.42e6, 1207.14e6, 1176.45e6, 1278.75e6]
    )
    signed_range_rate = -directions @ velocity + clocks
    doppler = -signed_range_rate / wavelengths
    fit = fit_constellation_clock_drifts(
        sat, sat_vel, doppler, wavelengths, np.zeros(3), groups
    )
    assert np.allclose(fit.velocity_ecef_mps, velocity, atol=1e-9)
    assert np.allclose(fit.clock_drifts_mps, [2.0, -3.5], atol=1e-9)
    assert fit.residual_rms_mps < 1e-9


def test_clock_normalization_preserves_velocity_with_one_reference_clock():
    directions = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0],
         [0, -1, 0], [0, 0, -1], [1, 1, 1], [-1, 1, -1]], dtype=float
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sat = directions * 21e6
    groups = np.array([0, 0, 0, 0, 2, 2, 2, 2])
    velocity = np.array([2.0, 1.0, -0.5])
    clocks = np.where(groups == 0, 5.0, -4.0)
    wavelengths = np.linspace(0.19, 0.25, len(groups))
    doppler = -(-directions @ velocity + clocks) / wavelengths
    equivalent, fit = normalize_constellation_clock_drifts(
        sat, np.zeros_like(sat), doppler, wavelengths, np.zeros(3), groups
    )
    normalized_range_rate = -equivalent * GPS_L1_WAVELENGTH_M
    recovered_clock = normalized_range_rate + directions @ velocity
    assert np.max(recovered_clock) - np.min(recovered_clock) < 1e-9
    assert np.allclose(fit.velocity_ecef_mps, velocity, atol=1e-9)
