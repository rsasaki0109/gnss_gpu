"""CPU-side validation tests for the RAIM wrapper."""

import numpy as np
import pytest

from gnss_gpu.raim import raim_check, raim_fde


def _make_raim_inputs(n_sat=8):
    sat_ecef = np.array([
        [-14985000.0,  -3988000.0,  21474000.0],
        [ -9575000.0,  15498000.0,  19457000.0],
        [  7624000.0, -16218000.0,  19843000.0],
        [ 16305000.0,  12037000.0,  17183000.0],
        [-20889000.0,  13759000.0,   8291000.0],
        [  5463000.0,  24413000.0,   8934000.0],
        [ 22169000.0,   3975000.0,  13781000.0],
        [-11527000.0, -19421000.0,  13682000.0],
    ])[:n_sat]
    pseudoranges = np.full(n_sat, 2.2e7)
    weights = np.ones(n_sat)
    position = np.array([-3957199.0, 3310205.0, 3737911.0, 3000.0])
    return sat_ecef, pseudoranges, weights, position


def test_raim_check_rejects_bad_pseudorange_shape():
    sat_ecef, pseudoranges, weights, position = _make_raim_inputs()

    with pytest.raises(RuntimeError, match="pseudoranges must have shape"):
        raim_check(sat_ecef, pseudoranges.reshape(-1, 1), weights, position)


def test_raim_check_rejects_bad_p_fa():
    sat_ecef, pseudoranges, weights, position = _make_raim_inputs()

    with pytest.raises(RuntimeError, match="p_fa must be in"):
        raim_check(sat_ecef, pseudoranges, weights, position, p_fa=0.0)


def test_raim_check_rejects_too_few_satellites():
    sat_ecef, pseudoranges, weights, position = _make_raim_inputs(n_sat=3)

    with pytest.raises(RuntimeError, match="requires at least 4 satellites"):
        raim_check(sat_ecef, pseudoranges, weights, position)


def test_raim_fde_rejects_bad_pseudorange_shape():
    sat_ecef, pseudoranges, weights, position = _make_raim_inputs()

    with pytest.raises(RuntimeError, match="pseudoranges must have shape"):
        raim_fde(sat_ecef, pseudoranges.reshape(-1, 1), weights, position)
