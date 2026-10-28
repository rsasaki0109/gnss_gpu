"""Null-domain independence and spawned-process equivalence for the grid demo."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from experiments.run_cpu_carrier_search import (
    CALIBRATION_DOMAIN, HELDOUT_DOMAIN, SIGNAL_DOMAIN, _null_job, _rng,
)
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.signal_sim_cpu import CHIP_RATE


def test_domain_seeds_are_reproducible_and_disjoint():
    domains = (CALIBRATION_DOMAIN, HELDOUT_DOMAIN, SIGNAL_DOMAIN)
    draws = [_rng(42, domain, 1).normal(size=16) for domain in domains]
    for domain, draw in zip(domains, draws):
        np.testing.assert_array_equal(draw, _rng(42, domain, 1).normal(size=16))
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[0], draws[2])
    assert not np.array_equal(draws[1], draws[2])


def test_spawned_null_search_matches_serial_and_is_independent_of_count():
    search = CaSearch(2*CHIP_RATE, (-1000., 0., 1000.))
    jobs = [(42, CALIBRATION_DOMAIN, trial, search) for trial in range(4)]
    expected = [_null_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert list(pool.map(_null_job, jobs)) == expected
    assert [_null_job(job) for job in jobs[:2]] == expected[:2]
