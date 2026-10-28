import numpy as np
import pytest

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch, acquire_ca_bit_edge, calibrate_ca_bit_edge, detect_ca_bit_edge
from gnss_gpu.code_delay_cpu import _averaged_template
from gnss_gpu.validation.tracking_residuals import generate_ca_code

FS, N = 2046000., 2046


def interleave(z):
    return np.stack([np.column_stack((row.real, row.imag)).ravel() for row in z])


def test_unknown_within_period_bit_edge_and_carrier_are_recovered():
    local = _averaged_template(generate_ca_code(1), 2, 37.)
    time = np.arange(20*N).reshape(20, N)/FS
    z = local*np.exp(2j*np.pi*175.*time+.4j)
    z[7, N//3:] *= -1; z[8:] *= -1
    cfg = CaBitEdgeSearch(CaSearch(FS, (-250., 0., 250.)))
    result = acquire_ca_bit_edge(interleave(z), cfg)
    assert result.carrier_hz == 175.
    assert result.delay_samples == 37 and result.excluded_period == 7
    assert result.statistic > 18*N*.97


def test_constant_polarity_does_not_need_to_drop_any_period():
    local = _averaged_template(generate_ca_code(1), 2, -11.)
    cfg = CaBitEdgeSearch(CaSearch(FS, (0.,)), periods=4, residual_grid_hz=(0.,), shortlist_size=None)
    result = acquire_ca_bit_edge(interleave(np.array([local]*4)), cfg)
    assert result.excluded_period is None and result.delay_samples == -11
    assert result.statistic == pytest.approx(4*N)


def test_fft_score_matches_direct_time_domain_edge_hypotheses():
    z = np.random.default_rng(81).normal(size=(4,N))+1j*np.random.default_rng(91).normal(size=(4,N))
    local = _averaged_template(generate_ca_code(1), 2, 0.)
    projections = np.array([[np.vdot(np.roll(local, lag), row) for lag in range(N)] for row in z])
    best = -1.
    for edge in (None, 0, 1, 2, 3):
        weights = np.ones(4)
        if edge is not None:
            weights[edge] = 0.; weights[edge+1:] = -1.
        included = weights != 0
        scores = N*abs(weights@projections)**2/(float(local@local)*np.sum(abs(z[included])**2))
        best = max(best, float(scores.max()))
    cfg = CaBitEdgeSearch(CaSearch(FS, (0.,)), periods=4, residual_grid_hz=(0.,), shortlist_size=None)
    iq = interleave(z); saved = iq.copy()
    result = acquire_ca_bit_edge(iq,cfg)
    assert result.statistic == pytest.approx(best, rel=2e-14)
    np.testing.assert_array_equal(iq,saved)


def test_zero_and_invalid_input_and_configuration():
    cfg = CaBitEdgeSearch(CaSearch(FS, (0.,)))
    assert acquire_ca_bit_edge(np.zeros((20,2*N)),cfg) is None
    for iq in (np.zeros((19,2*N)), np.full((20,2*N),np.nan), np.zeros((20,2*N),dtype=complex)):
        with pytest.raises(ValueError): acquire_ca_bit_edge(iq,cfg)
    for kwargs in (dict(periods=2), dict(residual_grid_hz=(25.,)),dict(residual_grid_hz=(0.,0.))):
        with pytest.raises(ValueError):CaBitEdgeSearch(CaSearch(FS,(0.,)),**kwargs)


def test_shortlist_retains_strong_signal_and_calibration_has_strict_gate():
    local = _averaged_template(generate_ca_code(1), 2, 57.)
    iq = interleave(np.array([local]*4))
    cfg = CaBitEdgeSearch(CaSearch(FS,(0.,)), periods=4,residual_grid_hz=(0.,),shortlist_size=8)
    candidate = acquire_ca_bit_edge(iq,cfg)
    threshold = calibrate_ca_bit_edge({"white":[candidate.statistic]*499},cfg)
    result = detect_ca_bit_edge(iq,threshold)
    assert candidate.delay_samples == 57
    assert not result.accepted and result.candidate == candidate
    assert threshold.score == candidate.statistic
    with pytest.raises(ValueError):calibrate_ca_bit_edge({},cfg)
    with pytest.raises(ValueError):calibrate_ca_bit_edge({"bad":[np.nan]*499},cfg)
