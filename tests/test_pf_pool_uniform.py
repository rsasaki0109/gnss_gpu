"""Direct FP64 CDF equivalence for uniform and almost-uniform weights."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from pf_pool_uniform_cases import COUNTS, FORMS, POLICIES, VARIANTS, check_cdf


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('form', FORMS)
def test_uniform_and_almost_uniform_CDF_bits_and_input_ownership(count, variant, form):
    check_cdf(count, variant, form)


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('form', ('contiguous', 'matrix'))
def test_uniform_CDF_error_policy_and_matrix_fallback(variant, policy, form):
    check_cdf(65536, variant, form, policy)
