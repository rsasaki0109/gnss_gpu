from __future__ import annotations

import math

import numpy as np

from gnss_gpu.validation import (
    ResidualSample,
    bin_by_elevation,
    bin_by_los,
    ks_statistic,
    percentiles,
    records_from_epoch,
    summarize,
    wasserstein1,
)


def test_percentiles_on_0_to_100() -> None:
    pct = percentiles(np.arange(0, 101))

    assert math.isclose(pct[50.0], 50.0)
    assert math.isclose(pct[68.0], 68.0)
    assert math.isclose(pct[95.0], 95.0)
    assert math.isclose(pct[99.0], 99.0)


def test_wasserstein_identity_and_constant_shift() -> None:
    x = np.array([-2.0, 0.0, 5.0])

    assert math.isclose(wasserstein1(x, x), 0.0)
    assert math.isclose(
        wasserstein1(np.zeros(16, dtype=np.float64), np.full(16, 3.0)),
        3.0,
    )


def test_ks_statistic_disjoint_distributions() -> None:
    assert math.isclose(
        ks_statistic(np.array([0.0, 1.0]), np.array([10.0, 11.0])),
        1.0,
    )


def test_summarize_signed_bias_and_rms() -> None:
    values = np.array([1.0, 2.0, 3.0])
    summary = summarize(values)

    assert summary["count"] == 3
    assert summary["mean"] > 0.0
    assert math.isclose(summary["rms"], math.sqrt((1.0 + 4.0 + 9.0) / 3.0))
    assert math.isclose(summary["mae"], 2.0)


def test_binning_by_elevation_and_los() -> None:
    samples = [
        ResidualSample(0, "G01", 0.0, math.radians(0.0), 0.0, None, True),
        ResidualSample(0, "G02", 0.0, math.radians(29.0), 0.0, None, False),
        ResidualSample(0, "G03", 0.0, math.radians(30.0), 0.0, None, True),
        ResidualSample(0, "G04", 0.0, math.radians(60.0), 0.0, None, False),
        ResidualSample(0, "G05", 0.0, math.radians(-1.0), 0.0, None, True),
    ]

    elev_bins = bin_by_elevation(samples, [0.0, 30.0, 60.0])
    assert [sample.prn for sample in elev_bins["[0,30)"]] == ["G01", "G02"]
    assert [sample.prn for sample in elev_bins["[30,60)"]] == ["G03"]

    los_bins = bin_by_los(samples)
    assert [sample.prn for sample in los_bins["los"]] == ["G01", "G03", "G05"]
    assert [sample.prn for sample in los_bins["nlos"]] == ["G02", "G04"]


def test_records_from_epoch_filters_invisible_and_formats_prn() -> None:
    records = records_from_epoch(
        epoch=7,
        prn_list=[1, "G02", 3],
        residual_m=np.array([1.5, 2.5, 3.5]),
        elevations=np.array([0.1, 0.2, 0.3]),
        azimuths=np.array([1.1, 1.2, 1.3]),
        is_los=np.array([True, False, True]),
        visible=np.array([True, False, True]),
        cn0_dbhz=np.array([45.0, 30.0, 35.0]),
    )

    assert len(records) == 2
    assert records[0].epoch == 7
    assert records[0].prn == "G01"
    assert records[1].prn == "G03"
    assert math.isclose(records[0].residual_m, 1.5)
    assert math.isclose(records[1].elevation_rad, 0.3)
    assert records[0].is_los is True
    assert math.isclose(records[0].cn0_dbhz, 45.0)


def test_empty_arrays_return_nan_without_exception() -> None:
    summary = summarize(np.array([], dtype=np.float64))
    assert summary["count"] == 0

    for key in (
        "mean",
        "rms",
        "mae",
        "p50",
        "p68",
        "p95",
        "p99",
        "abs_p50",
        "abs_p95",
    ):
        assert math.isnan(summary[key])

    pct = percentiles(np.array([], dtype=np.float64))
    assert set(pct.keys()) == {50.0, 68.0, 95.0, 99.0}
    assert all(math.isnan(value) for value in pct.values())
