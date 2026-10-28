from dataclasses import asdict

import numpy as np
import pytest

import gnss_gpu.code_delay_cpu as delay
from gnss_gpu.signal_sim_cpu import CHIP_RATE


@pytest.mark.parametrize("samples_per_chip", [2, 3, 4, 8, 20])
def test_cache_cold_warm_and_disabled_results_are_exact(samples_per_chip, monkeypatch):
    delay.clear_ca_delay_cache()
    rng = np.random.default_rng(731+samples_per_chip)
    iq = rng.normal(size=2*1023*samples_per_chip)
    cold = delay.estimate_ca_delay(iq, samples_per_chip*CHIP_RATE, prn=7, carrier_hz=133.25)
    warm = delay.estimate_ca_delay(iq, samples_per_chip*CHIP_RATE, prn=7, carrier_hz=133.25)
    monkeypatch.setattr(delay, "_CACHE_MAX_SAMPLES", 0)
    uncached = delay.estimate_ca_delay(iq, samples_per_chip*CHIP_RATE, prn=7, carrier_hz=133.25)
    assert asdict(cold) == asdict(warm) == asdict(uncached)


def test_reference_and_interval_arrays_cannot_be_mutated():
    delay.clear_ca_delay_cache()
    samples, spectrum = delay._cached_reference_data(1, 2)
    a, b, *_ = delay._cached_interval_data(1, 2, 0)
    for value in (samples, spectrum, a, b):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value[0] = 123


def test_cache_capacity_clear_and_eviction_preserve_results():
    delay.clear_ca_delay_cache()
    iq = np.random.default_rng(193).normal(size=4092)
    expected = delay.estimate_ca_delay(iq, 2*CHIP_RATE)
    for prn in range(1, 33):
        delay._cached_reference_data(prn, 2)
    for center in range(80):
        delay._cached_interval_data(2, 2, center)
    assert delay._cached_reference_data.cache_info().currsize == 16
    assert delay._cached_interval_data.cache_info().currsize == 64
    assert asdict(delay.estimate_ca_delay(iq, 2*CHIP_RATE)) == asdict(expected)
    delay.clear_ca_delay_cache()
    assert delay._cached_reference_data.cache_info().currsize == 0
    assert delay._cached_interval_data.cache_info().currsize == 0


def test_oversized_period_bypasses_retained_caches():
    delay.clear_ca_delay_cache()
    iq = np.random.default_rng(191).normal(size=2*1023*20)
    assert delay.estimate_ca_delay(iq, 20*CHIP_RATE) is not None
    assert delay._cached_reference_data.cache_info().currsize == 0
    assert delay._cached_interval_data.cache_info().currsize == 0
