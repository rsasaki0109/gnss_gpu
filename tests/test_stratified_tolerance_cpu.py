from dataclasses import replace

import numpy as np
import pytest

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentSearch
from gnss_gpu.stratified_tolerance_cpu import calibrate_ca_stratified_tolerance, detect_ca_stratified_tolerance


def test_maximum_gate_covers_all_strata_with_family_confidence_allocation():
    cfg = CaNoncoherentSearch(CaSearch(2046000., (0.,)), 1)
    threshold = calibrate_ca_stratified_tolerance({"white": np.linspace(1., 10., 999),
                                                  "other": np.linspace(2., 30., 999)}, cfg)
    assert threshold.score == max(value.score for _, value in threshold.strata)
    assert all(value.calibration_failure_probability == .05/4 for _, value in threshold.strata)
    assert all(value.binomial_failure_bound <= .05/4 for _, value in threshold.strata)
    found = detect_ca_stratified_tolerance(np.zeros((1, 4092)), threshold)
    assert not found.accepted and found.candidate is None
    with pytest.raises(ValueError, match="dominate"):
        replace(threshold, score=1.)
    with pytest.raises(ValueError, match="dominate"):
        replace(threshold, config=CaNoncoherentSearch(CaSearch(2046000., (500.,)), 1))


@pytest.mark.parametrize("scores,budget,methods", [({}, .05, 2), ({"a": [1.]*999}, 1., 2),
                                                 ({"": [1.]*999}, .05, 2), ({"a": [1.]*999}, .05, 0)])
def test_invalid_family_configuration(scores, budget, methods):
    with pytest.raises(ValueError):
        calibrate_ca_stratified_tolerance(scores, CaNoncoherentSearch(CaSearch(2046000., (0.,))),
                                          joint_failure_budget=budget, methods_in_family=methods)
