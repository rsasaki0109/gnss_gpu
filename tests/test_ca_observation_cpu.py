from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gnss_gpu.acquisition_cpu import CaSearch, acquire_ca_signal
from gnss_gpu.code_delay_cpu import (
    _prepare_ca_observation, _estimate_prepared_delay, estimate_ca_delay,
)
from gnss_gpu.code_detection_cpu import ca_detection_statistic
from gnss_gpu.signal_sim_cpu import CHIP_RATE


@pytest.mark.parametrize("dtype,scale", [(np.float32, 1.), (np.float64, 1.),
                                           (np.float64, 1e-200), (np.float64, 1e200)])
def test_shared_search_matches_independently_normalized_frequency_trials(dtype, scale):
    # Strided input exercises both conversion and ownership, including scales
    # where unnormalized inner products would underflow or overflow.
    iq = (np.random.default_rng(219).normal(size=8184)*scale).astype(dtype)[::2]
    saved = iq.copy()
    search = CaSearch(2*CHIP_RATE, (-700., 0., 900.), prn=9)
    independent = [(frequency, estimate_ca_delay(iq, search.sampling_freq,
                    prn=search.prn, carrier_hz=frequency)) for frequency in search.carrier_grid_hz]
    frequency, fitted = max(independent, key=lambda item: ca_detection_statistic(item[1]))
    serial = acquire_ca_signal(iq, search)
    assert serial.carrier_hz == frequency
    assert serial.delay == fitted
    assert acquire_ca_signal(iq, search, workers=3) == serial
    np.testing.assert_array_equal(iq, saved)


def test_prepared_signal_owns_its_data_and_parallel_fits_do_not_mutate_it():
    iq = np.random.default_rng(71).normal(size=4092)
    prepared = _prepare_ca_observation(iq, 2*CHIP_RATE)
    expected = _estimate_prepared_delay(prepared, carrier_hz=250.)
    saved = prepared.signal.copy()
    iq[:] = 0.
    with pytest.raises(ValueError):
        prepared.signal[0] = 1.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _estimate_prepared_delay(prepared, carrier_hz=250.), range(16)))
    assert all(result == expected for result in results)
    np.testing.assert_array_equal(prepared.signal, saved)


@pytest.mark.parametrize("carrier", [np.nan, np.inf, 2*CHIP_RATE])
def test_zero_observation_still_validates_carrier(carrier):
    with pytest.raises(ValueError, match="carrier_hz"):
        estimate_ca_delay(np.zeros(4092), 2*CHIP_RATE, carrier_hz=carrier)
