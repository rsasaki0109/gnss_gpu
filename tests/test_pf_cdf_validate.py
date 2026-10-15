"""Exact CDF acceptance, left search, and rejection ownership at SIMD boundaries."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_cdf_validate_cases import COUNTS, FORMS, VARIANTS, ValidationSession


@pytest.fixture(scope='module', params=COUNTS)
def session(request):
    native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    result = ValidationSession(native, ParticleFilterDevice, request.param)
    yield result
    result.close()


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('form', FORMS)
def test_CDF_acceptance_search_and_declined_state_ownership(session, variant, form):
    session.check(variant, form)
