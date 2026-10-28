import numpy as np
import pytest

from experiments.compare_cpu_carrier_grids import calibrate_grids, paired_carrier_errors, WIDE_GRID
from gnss_gpu.acquisition_cpu import CaSearch


def test_each_complete_grid_gets_its_own_rank_threshold():
    narrow, wide = CaSearch(2046000., (-500., 0., 500.)), CaSearch(2046000., WIDE_GRID)
    noise = np.random.default_rng(719).normal(size=(99, 4092))
    (a, sa), (b, sb) = calibrate_grids(noise, (narrow, wide))
    assert a.search == narrow and b.search == wide
    assert np.all(np.asarray(sb) >= np.asarray(sa))
    assert a.calibration.score == max(sa)
    assert b.calibration.score == max(sb)
    assert b.calibration.score >= a.calibration.score


def test_common_used_errors_exclude_outage_and_unavailable_blocks():
    before = [dict(measurement_used=True, carrier_hz=500.) for _ in range(4)]
    after = [dict(measurement_used=True, carrier_hz=700.) for _ in range(4)]
    after[0] = dict(measurement_used=False, carrier_hz=None)
    result = paired_carrier_errors(before, after, (1, 3), 700.)
    assert result["common_used_indices"] == [3]
    assert result["common_carrier_mae_before_hz"] == 200.
    assert result["common_carrier_mae_after_hz"] == 0.


def test_empty_common_set_is_missing_instead_of_zero_error():
    result = paired_carrier_errors([dict(measurement_used=False)], [dict(measurement_used=False)], (0, 0), 700.)
    assert result["common_used_blocks"] == 0
    assert result["common_carrier_mae_before_hz"] is None
    assert result["common_carrier_mae_after_hz"] is None
    with pytest.raises(ValueError):
        paired_carrier_errors([], [{}], (0, 0), 700.)
