from types import SimpleNamespace

import numpy as np

from gnss_gpu.carrier_afv_observation import build_carrier_afv_observation


def _measurement(
    *,
    carrier_phase=1.2e6,
    snr=35.0,
    elevation=0.5,
    sat_ecef=(20_000_000.0, 0.0, 0.0),
    corrected_pseudorange=18_000_000.0,
    weight=0.8,
):
    return SimpleNamespace(
        carrier_phase=carrier_phase,
        snr=snr,
        elevation=elevation,
        satellite_ecef=np.asarray(sat_ecef, dtype=np.float64),
        corrected_pseudorange=corrected_pseudorange,
        weight=weight,
    )


def test_carrier_afv_observation_collects_valid_rows_and_sigmas():
    measurements = [
        _measurement(sat_ecef=(20_000_000.0 + i, 0.0, 0.0), weight=0.5 + i)
        for i in range(4)
    ]

    obs = build_carrier_afv_observation(
        measurements,
        np.array([m.satellite_ecef for m in measurements]),
        np.array([m.corrected_pseudorange for m in measurements]),
        np.array([np.nan, 0.0, 0.0]),
        snr_min=25.0,
        elev_min=0.15,
        target_sigma_cycles=0.05,
    )

    assert obs is not None
    assert obs.n_sat == 4
    assert obs.sat_ecef.shape == (4, 3)
    np.testing.assert_allclose(obs.weights, [0.5, 1.5, 2.5, 3.5])
    assert obs.sigma_sequence_cycles == (2.0, 0.5, 0.05)


def test_carrier_afv_observation_filters_low_quality_rows():
    measurements = [
        _measurement(carrier_phase=0.0),
        _measurement(carrier_phase=100.0),
        _measurement(snr=10.0),
        _measurement(elevation=0.05),
        _measurement(),
        _measurement(sat_ecef=(20_000_001.0, 0.0, 0.0)),
        _measurement(sat_ecef=(20_000_002.0, 0.0, 0.0)),
        _measurement(sat_ecef=(20_000_003.0, 0.0, 0.0)),
    ]

    obs = build_carrier_afv_observation(
        measurements,
        np.array([m.satellite_ecef for m in measurements]),
        np.array([m.corrected_pseudorange for m in measurements]),
        np.array([np.nan, 0.0, 0.0]),
        snr_min=25.0,
        elev_min=0.15,
        target_sigma_cycles=0.05,
    )

    assert obs is not None
    assert obs.n_sat == 4


def test_carrier_afv_observation_returns_none_below_min_sats():
    measurements = [_measurement(), _measurement(), _measurement()]

    obs = build_carrier_afv_observation(
        measurements,
        np.array([m.satellite_ecef for m in measurements]),
        np.array([m.corrected_pseudorange for m in measurements]),
        np.array([np.nan, 0.0, 0.0]),
        snr_min=25.0,
        elev_min=0.15,
        target_sigma_cycles=0.05,
    )

    assert obs is None


def test_carrier_afv_observation_filters_large_pseudorange_residual():
    spp = np.array([2_000_000.0, 0.0, 0.0], dtype=np.float64)
    good = [
        _measurement(
            sat_ecef=(20_000_000.0 + i, 0.0, 0.0),
            corrected_pseudorange=18_000_000.0 + i + 10.0,
        )
        for i in range(4)
    ]
    outlier = _measurement(
        sat_ecef=(20_000_010.0, 0.0, 0.0),
        corrected_pseudorange=18_000_100.0,
    )
    measurements = [*good, outlier]

    obs = build_carrier_afv_observation(
        measurements,
        np.array([m.satellite_ecef for m in measurements]),
        np.array([m.corrected_pseudorange for m in measurements]),
        spp,
        snr_min=25.0,
        elev_min=0.15,
        target_sigma_cycles=0.05,
    )

    assert obs is not None
    assert obs.n_sat == 4
