from dataclasses import asdict, replace

import numpy as np
import pytest

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.code_delay_cpu import _averaged_template
from gnss_gpu.noncoherent_acquisition_cpu import (
    CaNoncoherentSearch, acquire_ca_noncoherent, calibrate_ca_noncoherent, detect_ca_noncoherent,
)
from gnss_gpu.validation.tracking_residuals import generate_ca_code

FS, N = 2046000., 2046


def iq_rows(signals):
    return np.stack([np.column_stack((z.real, z.imag)).ravel() for z in signals])


def config(k=4, windows=1, grid=(-500., 0., 500.)):
    return CaNoncoherentSearch(CaSearch(FS, grid), k, windows)


def test_accumulated_fft_matches_independent_direct_circular_inner_products():
    rng = np.random.default_rng(734)
    z = rng.normal(size=(4, N))+1j*rng.normal(size=(4, N))
    search = config(2, 2, (0.,))
    local = _averaged_template(generate_ca_code(1), 2, 0.)
    # A direct reference avoids sharing FFT, normalization or peak-selection code.
    powers = np.array([[sum(abs(np.vdot(np.roll(local, lag), row))**2 for row in z[2*w:2*w+2])
                       for lag in range(N)] for w in range(2)])
    energies = np.array([np.sum(abs(z[2*w:2*w+2])**2) for w in range(2)])
    scores = 2*N*powers/(float(local@local)*energies[:, None])
    window, lag = np.unravel_index(np.argmax(scores), scores.shape)
    found = acquire_ca_noncoherent(iq_rows(z), search)
    assert found.window_index == window
    assert found.delay_samples == (lag+N//2) % N-N//2
    assert found.statistic == pytest.approx(float(scores[window, lag]), rel=2e-14)


def test_independent_period_peaks_are_not_added_as_if_they_shared_a_delay():
    local = _averaged_template(generate_ca_code(1), 2, 0.)
    same = acquire_ca_noncoherent(iq_rows([local, local]), config(2, grid=(0.,)))
    different = acquire_ca_noncoherent(iq_rows([local, np.roll(local, 200)]), config(2, grid=(0.,)))
    assert same.statistic == pytest.approx(2*N)
    assert different.statistic < .6*same.statistic


def test_whole_period_phase_and_nav_flips_preserve_noncoherent_score():
    local = _averaged_template(generate_ca_code(1), 2, 17.)
    ordinary = iq_rows([local]*4)
    flipped = iq_rows([local, -local, 1j*local, -1j*local])
    a, b = (acquire_ca_noncoherent(iq, config(grid=(0.,))) for iq in (ordinary, flipped))
    assert a.delay_samples == b.delay_samples == 17
    assert a.statistic == pytest.approx(b.statistic, rel=2e-14)
    assert abs(np.sum([local, -local, 1j*local, -1j*local], axis=0)).max() == 0.


@pytest.mark.parametrize("carrier,delay", [(-500., -12), (500., 37)])
def test_unknown_carrier_and_delay_are_recovered(carrier, delay):
    local = _averaged_template(generate_ca_code(1), 2, delay)
    z = local*np.exp(2j*np.pi*carrier*np.arange(N)/FS)
    found = acquire_ca_noncoherent(iq_rows([z, -z, z, -z]), config())
    assert found.carrier_hz == carrier and found.delay_samples == delay


@pytest.mark.parametrize("scale", [1e-250, 1e250])
def test_large_small_scale_and_input_preservation(scale):
    iq = np.random.default_rng(91).normal(size=(4, 2*N))
    saved = iq.copy()
    a = acquire_ca_noncoherent(iq, config())
    b = acquire_ca_noncoherent(iq*scale, config())
    assert (a.carrier_hz, a.delay_samples, a.window_index) == (b.carrier_hz, b.delay_samples, b.window_index)
    assert a.statistic == pytest.approx(b.statistic, rel=2e-14)
    np.testing.assert_array_equal(iq, saved)


def test_zero_and_partially_zero_windows_are_handled():
    assert acquire_ca_noncoherent(np.zeros((4, 2*N)), config()) is None
    iq = np.random.default_rng(3).normal(size=(4, 2*N)); iq[:2] = 0.
    assert acquire_ca_noncoherent(iq, config(2, 2)).window_index == 1


@pytest.mark.parametrize("iq", [np.zeros((3, 2*N)), np.zeros((4, 2*N-1)),
                                   np.zeros((4, 2*N), dtype=complex), np.full((4, 2*N), np.nan)])
def test_invalid_iq(iq):
    with pytest.raises(ValueError):
        acquire_ca_noncoherent(iq, config())


@pytest.mark.parametrize("k,windows", [(0, 1), (1, 0), (1.5, 1)])
def test_invalid_window_configuration(k, windows):
    with pytest.raises(ValueError):
        config(k, windows)


def test_threshold_covers_full_decision_and_strict_ties_retain_candidate():
    cfg = config(2, 2)
    iq = np.random.default_rng(71).normal(size=(4, 2*N))
    candidate = acquire_ca_noncoherent(iq, cfg)
    threshold = calibrate_ca_noncoherent([candidate.statistic]*99, cfg)
    found = detect_ca_noncoherent(iq, threshold)
    assert not found.accepted and found.candidate == candidate
    assert threshold.config.windows_per_decision == 2
    with pytest.raises(ValueError, match="metadata"):
        replace(threshold, config=CaNoncoherentSearch(CaSearch(4092000., (0.,)), 2, 2))


@pytest.mark.parametrize("samples_per_chip", [3, 8, 20])
def test_sampling_geometries_including_uncached_reference(samples_per_chip):
    local = _averaged_template(generate_ca_code(7), samples_per_chip, -13.)
    cfg = CaNoncoherentSearch(CaSearch(samples_per_chip*1023000., (0.,), prn=7), 2)
    found = acquire_ca_noncoherent(iq_rows([local, -local]), cfg)
    assert found.delay_samples == -13
    assert found.statistic == pytest.approx(2*len(local))


def test_bit_flip_inside_each_period_can_still_destroy_the_target_projection():
    local = _averaged_template(generate_ca_code(1), 2, 0.)
    altered = local.copy(); altered[N//2:] *= -1
    ordinary = acquire_ca_noncoherent(iq_rows([local]*4), config(grid=(0.,)))
    within_period = acquire_ca_noncoherent(iq_rows([altered]*4), config(grid=(0.,)))
    assert within_period.statistic < .2*ordinary.statistic
