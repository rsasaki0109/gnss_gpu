"""Receiver-only estimates and missing/error calibration denominators."""

import numpy as np

from experiments.run_cpu_position_uncertainty import estimate_channels, summarize
from experiments.run_cpu_rain_positioning import constellation
from gnss_gpu.range_uncertainty_cpu import RangeUncertaintyConfig


def test_estimation_ignores_truth_visibility_weather_and_detection_score():
    sat = constellation()
    channels = []
    for block in range(15):
        for prn, satellite in enumerate(sat, 1):
            channels.append(dict(block=block, prn=prn, used=True,
                                 delay_m=np.linalg.norm(satellite)-22e6+np.sin(block+prn)))
    other = [dict(row, truth_error_m=1e6, stage="wet", direct_visible=False, statistic=.001) for row in channels]
    first = estimate_channels(channels, sat, RangeUncertaintyConfig())
    second = estimate_channels(other, sat, RangeUncertaintyConfig())
    for a,b in zip(first, second):
        np.testing.assert_array_equal(a["sigma"], b["sigma"])
        for mode in ("equal", "weighted"):
            np.testing.assert_array_equal(a["solutions"][mode].solution.position_m, b["solutions"][mode].solution.position_m)
            np.testing.assert_array_equal(a["solutions"][mode].covariance_m2, b["solutions"][mode].covariance_m2)


def test_unavailable_epoch_has_no_error_or_successful_coverage_credit():
    summary = summarize([dict(available=False, position_error_m=None, covered_95=None),
                         dict(available=True, position_error_m=150., covered_95=False, position_nees=100.)])
    assert summary["blocks"] == 2 and summary["unavailable"] == 1
    assert summary["error_over_100_m"] == 1 and summary["covered_95"] == 0
    assert summary["position_rmse_m"] == 150. and summary["coverage_95_fraction"] == 0.
    missing = summarize([dict(available=False)])
    assert missing["position_rmse_m"] is None and missing["coverage_95_fraction"] is None
