"""Full GPU uniform-weight scan and exact old-policy migration fallback."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_uniform_scan_cases import COUNTS,VARIANTS,MODES,check_scan,check_dispatch,check_queue


@pytest.fixture(scope='module')
def backend():
    native=pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    return native,ParticleFilterDevice


@pytest.mark.parametrize('count',COUNTS)
@pytest.mark.parametrize('variant',VARIANTS)
def test_uniform_weight_scan_all_bits_finite_zero_NaN_Inf(backend,count,variant):
    check_scan(*backend,count,variant)


@pytest.mark.parametrize('count',(65536,100000,1000000))
@pytest.mark.parametrize('mode',MODES)
def test_uniform_weight_scan_getter_CDF_hook_policy_count_guards(backend,count,mode):
    check_dispatch(*backend,count,mode)


@pytest.mark.parametrize('count',(65536,100000,1000000))
def test_uniform_weight_scan_queued_producers_and_reduction_scratch(backend,count):
    check_queue(*backend,count)


@pytest.mark.parametrize('form',('null_source','null_target','destroyed_source','destroyed_target','inplace'))
def test_uniform_weight_scan_declines_invalid_or_inplace_handles(backend,form):
    from bench_pf_covariance_broadcast_compare import _new_pf
    native,klass=backend
    method=getattr(native,'pf_device_try_gather_uniform_weights',None)
    if method is None:
        assert not hasattr(native,'pf_device_try_gather_uniform_weights')
        return
    source,target=_new_pf(klass,65536),_new_pf(klass,32768)
    try:
        a,b=target._state,source._state
        if form=='null_source':
            b=None
        elif form=='null_target':
            a=None
        elif form.startswith('destroyed'):
            native.pf_device_destroy(source._state if form.endswith('source') else target._state)
        elif form=='inplace':
            a=b
        assert not method(a,b)
    finally:
        for pf in (source,target):
            if not (form.startswith('destroyed') and pf is (source if form.endswith('source') else target)):
                native.pf_device_destroy(pf._state)
            pf._state=None
