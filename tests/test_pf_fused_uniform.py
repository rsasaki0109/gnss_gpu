"""Exact coalesced covariance, full-state and zero-weight fused uniform gather."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_fused_uniform_cases import COUNTS,TARGETS,VARIANTS
from pf_uniform_inverse_cases import check_inverse
from pf_uniform_cdf_cases import check_uniform_queue

@pytest.fixture(scope='module')
def backend():
    native=pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    return native,ParticleFilterDevice

@pytest.mark.parametrize('count',COUNTS)
@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('variant',VARIANTS)
def test_fused_uniform_all16_zero_weights_and_block_tails(backend,count,target,variant):
    check_inverse(*backend,count,target,variant)

@pytest.mark.parametrize('count',COUNTS)
def test_fused_uniform_next_predict_KF_resample_and_scratch(backend,count):
    check_uniform_queue(*backend,count)

@pytest.mark.parametrize('count',COUNTS)
@pytest.mark.parametrize('variant',VARIANTS)
def test_fused_uniform_full_diagnostic_CDF_and_state_bits(backend,count,variant):
    from pf_uniform_cdf_cases import check_uniform
    check_uniform(*backend,count,257,variant)


def test_measurement_lifecycle_releases_session_without_cyclic_GC():
    import gc
    from types import SimpleNamespace
    import weakref
    import numpy as np
    from bench_pf_fused_uniform_compare import Session

    def make():
        session=Session.__new__(Session)
        session.native=SimpleNamespace(pf_device_destroy=lambda state:None)
        session.state=object()
        session.pf=SimpleNamespace(_state=session.state)
        session.peer=SimpleNamespace(_state=object())
        session.pool=SimpleNamespace(factory=lambda n:session.pf)
        session.saved=np.empty(1024*1024,dtype=np.uint8)
        return session

    enabled=gc.isenabled()
    gc.disable()
    try:
        session=make()
        owner,payload=weakref.ref(session),weakref.ref(session.saved)
        session.close()
        del session
        assert owner() is None and payload() is None
    finally:
        gc.collect()
        if enabled:
            gc.enable()
