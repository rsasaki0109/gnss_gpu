"""CPU-side validation tests for the AtmosphereCorrection wrapper."""

import numpy as np
import pytest

from gnss_gpu.atmosphere import AtmosphereCorrection


def test_init_rejects_invalid_iono_params():
    with pytest.raises(RuntimeError, match="iono_alpha must have shape"):
        AtmosphereCorrection(iono_alpha=[1e-8, 2e-8, 3e-8])

    with pytest.raises(RuntimeError, match="iono_beta must be finite"):
        AtmosphereCorrection(iono_beta=[1e5, 2e5, np.inf, 4e5])


def test_tropo_rejects_invalid_sat_el_shape():
    atm = AtmosphereCorrection()
    rx_lla = np.array([np.radians(45.0), np.radians(0.0), 0.0])
    sat_el = np.radians(np.array([30.0, 60.0]))

    with pytest.raises(RuntimeError, match="sat_el must have shape"):
        atm.tropo(rx_lla, sat_el.reshape(1, 2))
