"""Uniform CDF selection retains every selected particle state bit."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_uniform_inverse_cases import SOURCE_COUNTS, TARGET_COUNTS, VARIANTS, check_inverse


@pytest.fixture(scope='module')
def backend():
    native=pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    return native,ParticleFilterDevice


@pytest.mark.parametrize('count',SOURCE_COUNTS)
@pytest.mark.parametrize('target',TARGET_COUNTS)
@pytest.mark.parametrize('variant',VARIANTS)
def test_uniform_CDF_selection_without_diagnostic_output(backend,count,target,variant):
    check_inverse(*backend,count,target,variant)
