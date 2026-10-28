"""Real-data comparison must respect receiver clock and one-to-one epoch IDs."""

from datetime import datetime

import numpy as np
import pytest

from experiments.run_public_rinex_baseline import read_rtklib_positions, solution_gpst
from gnss_gpu.transmission_time import C_LIGHT


def test_positive_receiver_clock_is_removed_before_matching():
    observed = datetime(2005, 4, 2, 0, 59, 30, 5000)
    assert solution_gpst(observed, .005*C_LIGHT) == datetime(2005, 4, 2, 0, 59, 30)


def test_negative_clock_and_output_rounding_cross_day():
    observed = datetime(2005, 4, 2, 23, 59, 59, 999000)
    assert solution_gpst(observed, -.001*C_LIGHT) == datetime(2005, 4, 3)
    assert solution_gpst(datetime(2005, 4, 2, 0, 0, 0, 999800), 0.) == datetime(2005, 4, 2, 0, 0, 1)


def test_invalid_clock_is_not_matched():
    with pytest.raises(ValueError, match="finite receiver clock"):
        solution_gpst(datetime(2005, 4, 2), np.nan)


def test_rtklib_reader_rejects_duplicate_epoch_instead_of_overwriting(tmp_path):
    path = tmp_path/"result.pos"
    row = "2005/04/02 00:00:00.000 -3976219.0 3382373.0 3652513.0 5 7\n"
    path.write_text("% header\n"+row)
    positions = read_rtklib_positions(path)
    np.testing.assert_array_equal(positions[datetime(2005, 4, 2)], [-3976219., 3382373., 3652513.])
    path.write_text(row+row)
    with pytest.raises(ValueError, match="duplicate"):
        read_rtklib_positions(path)
