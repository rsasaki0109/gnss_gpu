"""Weight invalidation and exact ESS replay across shared-buffer reuse."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_ess_cache_cases import COUNTS, DISTRIBUTIONS, MUTATORS, PRESERVERS, check

native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('kind', MUTATORS)
def test_every_weight_mutation_matches_forced_gpu_reduction(count, kind):
    check(native, count, kind)


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('kind', PRESERVERS)
def test_weight_preserving_work_keeps_exact_ess(count, kind):
    check(native, count, kind)


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
def test_repeat_and_result_buffer_reuse_preserve_nonfinite_contract(count, distribution):
    check(native, count, 'repeat', distribution)
