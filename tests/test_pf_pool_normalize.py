"""Exact CDF reuse, caller ownership, and eager policy fallback."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_pool_normalize_cases import COUNTS,FORMS,POLICIES,check_cdf


@pytest.mark.parametrize('count',COUNTS)
@pytest.mark.parametrize('variant',range(8))
@pytest.mark.parametrize('form',FORMS)
def test_CDF_bits_and_readonly_strided_input_ownership(count,variant,form):
    check_cdf(count,variant,form)


@pytest.mark.parametrize('variant',range(8))
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('form',('contiguous','matrix'))
def test_warning_raise_callback_log_and_print_operation_order(variant,policy,form):
    check_cdf(65536,variant,form,policy)
