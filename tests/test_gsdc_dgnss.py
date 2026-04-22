from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np

from gnss_gpu.gsdc_dgnss import (
    DDWLSConfig,
    cors_rinex_urls,
    cors_station_candidates,
    dd_pseudorange_position_update,
    gsdc_epoch_measurements,
)
from gnss_gpu.dd_pseudorange import DDPseudorangeResult


def test_cors_candidates_use_gsdc_area_tokens() -> None:
    assert cors_station_candidates(run_name="2023-05-09-21-32-us-ca-mtv-pe1")[:2] == [
        "SLAC",
        "P222",
    ]
    assert cors_station_candidates(run_name="2023-05-25-19-10-us-ca-sjc-be2")[:3] == [
        "MHC2",
        "MHCB",
        "P222",
    ]
    assert cors_station_candidates(run_name="2022-04-01-18-22-us-ca-lax-t")[:2] == [
        "TORP",
        "CRHS",
    ]


def test_cors_urls_match_noaa_daily_pattern() -> None:
    urls = cors_rinex_urls("SLAC", date(2023, 5, 23))

    assert urls[0].endswith("/rinex/2023/143/slac/slac1430.23d.gz")
    assert any(url.endswith("/rinex/2023/143/slac/slac1430.23o.gz") for url in urls)


def test_gsdc_epoch_measurements_maps_l1_e1_rows() -> None:
    class _Group:
        def itertuples(self, index=False):
            del index
            yield SimpleNamespace(
                SignalType="GPS_L1_CA",
                Svid=3,
                RawPseudorangeMeters=21_000_000.0,
                SvClockBiasMeters=100.0,
                IonosphericDelayMeters=np.nan,
                TroposphericDelayMeters=2.0,
                IsrbMeters=1.0,
                SvPositionXEcefMeters=1.0,
                SvPositionYEcefMeters=2.0,
                SvPositionZEcefMeters=3.0,
                SvElevationDegrees=30.0,
                Cn0DbHz=40.0,
            )
            yield SimpleNamespace(
                SignalType="GPS_L5_Q",
                Svid=3,
                RawPseudorangeMeters=22_000_000.0,
                SvClockBiasMeters=0.0,
                IonosphericDelayMeters=0.0,
                TroposphericDelayMeters=0.0,
                IsrbMeters=0.0,
                SvPositionXEcefMeters=1.0,
                SvPositionYEcefMeters=2.0,
                SvPositionZEcefMeters=3.0,
                SvElevationDegrees=30.0,
                Cn0DbHz=40.0,
            )

    measurements = gsdc_epoch_measurements(_Group())
    raw_measurements = gsdc_epoch_measurements(_Group(), apply_gsdc_corrections=False)

    assert len(measurements) == 1
    assert measurements[0].system_id == 0
    assert measurements[0].prn == 3
    assert measurements[0].corrected_pseudorange == 21_000_097.0
    assert raw_measurements[0].corrected_pseudorange == 21_000_000.0
    np.testing.assert_allclose(measurements[0].satellite_ecef, [1.0, 2.0, 3.0])


def test_dd_position_update_moves_seed_toward_true_position() -> None:
    base = np.array([1.1e6, -4.8e6, 4.0e6], dtype=np.float64)
    true = base + np.array([40.0, -25.0, 12.0], dtype=np.float64)
    seed = true + np.array([8.0, -5.0, 3.0], dtype=np.float64)
    directions = np.array(
        [
            [0.82, 0.25, 0.52],
            [-0.32, 0.74, 0.59],
            [0.18, -0.91, 0.37],
            [-0.76, -0.20, 0.62],
            [0.53, -0.38, 0.76],
        ],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sat_ecef = true + directions * 20_200_000.0
    base_ranges = np.linalg.norm(sat_ecef - base, axis=1)
    rover_ranges = np.linalg.norm(sat_ecef - true, axis=1)
    ref = 0
    dd_obs = (rover_ranges[1:] - rover_ranges[ref]) - (base_ranges[1:] - base_ranges[ref])
    dd = DDPseudorangeResult(
        dd_pseudorange_m=dd_obs,
        sat_ecef_k=sat_ecef[1:],
        sat_ecef_ref=np.repeat(sat_ecef[[ref]], len(dd_obs), axis=0),
        base_range_k=base_ranges[1:],
        base_range_ref=np.repeat(base_ranges[ref], len(dd_obs)),
        dd_weights=np.ones(len(dd_obs)),
        ref_sat_ids=tuple(["G01"] * len(dd_obs)),
        n_dd=len(dd_obs),
    )

    updated, stats = dd_pseudorange_position_update(
        seed,
        dd,
        DDWLSConfig(prior_sigma_m=100.0, dd_sigma_m=1.0, max_shift_m=50.0),
    )

    assert stats["accepted"] is True
    assert np.linalg.norm(updated - true) < np.linalg.norm(seed - true)
