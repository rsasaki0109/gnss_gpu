"""Exact input conversion, unsupported fallbacks and full PF/RNG state."""

from collections import UserDict
from copy import deepcopy
import ctypes
import gc
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
from rbpf_dd_pack import DdPacker, PackedDdScorer, PackedDdStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case, make_oracle  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_move_key_memo import MoveKeyMemoStep  # noqa: E402


@pytest.fixture(scope='module')
def packer():
    path = os.environ.get('GNSS_GPU_RBPF_DD_PACK_DLL')
    if not path:
        pytest.skip('An explicit CPython DD pack DLL is required')
    return DdPacker(path)


@pytest.fixture(scope='module')
def scorers(packer):
    path = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not path:
        pytest.skip('An explicit standalone likelihood DLL is required')
    with BatchScorer(path) as old, PackedDdScorer(path, pack_dll=packer.dll._name) as new:
        yield old, new


def exact_state(a, b):
    check = comparison(a, b)
    assert check['discrete_equal'] and check['numeric_bits_equal'] and check['position_bits_equal'], check
    assert check['max_abs_error'] == 0., check


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('count,rows,groups', [(1,16,1), (21,64,4), (128,128,8), (512,128,8)])
def test_pack_bits_gauges_missing_order_and_fresh_buffers(packer, seed, count, rows, groups):
    oracle, problem, hypotheses = make_case(count, rows=rows, groups=groups, seed=seed)
    # Exact arbitrary-size subtraction before FP64 conversion, with a large
    # common gauge and intentionally noncanonical insertion order.
    for h in hypotheses:
        for group, values in list(h.items()):
            h[group] = {sat:z+2**180 for sat,z in reversed(list(values.items()))}
    before = deepcopy(hypotheses)
    expected = np.ascontiguousarray([oracle._pair_dd(h, problem) for h in hypotheses])
    calls = packer.native_calls
    actual = packer.batch(hypotheses, problem, oracle._pair_dd)
    # Count512 fixtures include duplicate requests, exceeding the native
    # capacity; the entire batch must fall back, without truncation.
    assert packer.native_calls-calls == int(len(hypotheses) <= 512)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert hypotheses == before
    actual[:] = 11.
    again = packer.batch(hypotheses, problem, oracle._pair_dd)
    assert again.tobytes() == expected.tobytes()
    assert not np.shares_memory(actual, again)


@pytest.mark.parametrize('values', [
    {}, {1:0}, {1:None,2:4}, {1:4,2:None}, {1:2**100+3,2:2**100},
    {1:2**54+1,2:0}, {1:-(2**54+1),2:0}, {1:2**100,2:-(2**100)},
    {1:2**1023,2:2**1023-3}, None])
def test_single_large_int_rounding_none_and_nan_payload(packer, values):
    oracle = make_oracle()
    problem = type('Problem', (), dict(pairs=[((1,0),1,2),((1,0),2,1),((2,0),3,4)]))()
    packer.source_dd = oracle._pair_dd
    hypothesis = {(1,0):values}
    expected = oracle._pair_dd(hypothesis, problem)
    assert packer.one(hypothesis, problem).tobytes() == expected.tobytes()


def test_integer_conversion_overflow_retains_exception_and_reference_counts(packer):
    oracle = make_oracle()
    problem = type('Problem', (), dict(pairs=[((1,0),1,2)]))()
    h = {(1,0):{1:2**2000,2:0}}
    for method in (oracle._pair_dd, lambda h,p:packer.batch([h],p,oracle._pair_dd)):
        with pytest.raises(OverflowError) as error:
            method(h,problem)
        assert str(error.value) == 'int too large to convert to float'
    values = {(1,0):{1:2**100+1,2:2**100}}
    before = [sys.getrefcount(o) for o in (values,values[(1,0)],values[(1,0)][1])]
    for _ in range(2000):
        packer.batch([values],problem,oracle._pair_dd)
    gc.collect()
    assert [sys.getrefcount(o) for o in (values,values[(1,0)],values[(1,0)][1])] == before


@pytest.mark.parametrize('kind', ['mapping','dict_subclass','numpy_int','bool','float','tuple_subclass','pair_tuple'])
def test_unsupported_types_delegate_exactly(packer, kind):
    class Dict(dict):
        pass
    class Tuple(tuple):
        pass
    oracle = make_oracle()
    problem = type('Problem', (), dict(pairs=[((1,0),1,2)]))()
    h = {(1,0):{1:4,2:2}}
    if kind == 'mapping':
        h = UserDict(h)
    elif kind == 'dict_subclass':
        h = Dict(h)
    elif kind == 'tuple_subclass':
        problem.pairs[0] = Tuple(problem.pairs[0])
    elif kind == 'pair_tuple':
        problem.pairs = tuple(problem.pairs)
    else:
        h[(1,0)][1] = dict(numpy_int=np.int64(4),bool=True,float=4.)[kind]
    calls = []
    def original(h,p):
        calls.append(h)
        return oracle._pair_dd(h,p)
    expected = np.ascontiguousarray([oracle._pair_dd(h,problem)])
    before = packer.fallback_calls
    actual = packer.batch([h],problem,original)
    assert actual.tobytes() == expected.tobytes()
    assert len(calls) == 1 and calls[0] is h
    assert packer.fallback_calls-before == 1


@pytest.mark.parametrize('kind', ['short_output','null_output','negative_size','too_many_pairs','too_many_hypotheses','invalid_late_hypothesis','malformed_pair'])
def test_native_bounds_and_preflight_do_not_write(packer, kind):
    pairs = [((1,0),1,2)]
    hypotheses = [{(1,0):{1:4,2:2}}]
    output = np.full(1024,123.25)
    capacity, pointer = output.size,output.ctypes.data
    if kind == 'short_output':
        capacity = 0
    elif kind == 'null_output':
        pointer = 0
    elif kind == 'negative_size':
        capacity = -1
    elif kind == 'too_many_pairs':
        pairs *= 129
    elif kind == 'too_many_hypotheses':
        hypotheses *= 513
    elif kind == 'invalid_late_hypothesis':
        hypotheses += [UserDict()]
    else:
        pairs += [((1,0),1)]
    before = output.tobytes()
    assert packer.fill(hypotheses,pairs,pointer,capacity,np.nan) == 0
    assert output.tobytes() == before


@pytest.mark.parametrize('count,pairs', [(0,0),(0,4),(4,0),(512,128)])
def test_supported_capacity_edges_and_canaries(packer, count, pairs):
    output = np.full(count*pairs+2,123.25)
    assert packer.fill([{}]*count,[((1,0),1,2)]*pairs,output[1:].ctypes.data,count*pairs,np.nan) == 1
    assert output[0] == output[-1] == 123.25
    assert np.isnan(output[1:-1]).all()


@pytest.mark.parametrize('seed', [20260710,20260711,20260712])
@pytest.mark.parametrize('rows,groups', [(16,1),(64,4),(128,8)])
@pytest.mark.parametrize('uniform,huber,cluster', [
    (True,True,False),(False,True,False),(True,False,False),
    (False,False,False),(True,True,True),(False,True,True)])
def test_all_scores_statuses_and_intermediate_bits(scorers, seed, rows, groups, uniform, huber, cluster):
    oracle,problem,hypotheses = make_case(21,rows=rows,groups=groups,seed=seed,
        marg_uniform=uniform,huber=huber,cluster=cluster)
    for backend in ('native_cpu','gpu'):
        expected,ed = scorers[0].score(hypotheses,problem,oracle,backend,diagnostics=True)
        actual,ad = scorers[1].score(hypotheses,problem,oracle,backend,diagnostics=True)
        assert actual.tobytes() == expected.tobytes()
        for name in ('diagnostic','statuses'):
            assert ad[name].tobytes() == ed[name].tobytes()
        assert ad['scalar_fallbacks'] == ed['scalar_fallbacks'] == 0


def test_custom_dd_hook_keeps_calls_and_values(scorers):
    oracle,problem,hypotheses = make_case(8)
    original = oracle._pair_dd
    calls = []
    def custom(h,p):
        calls.append(h)
        return original(h,p)+1.
    oracle._pair_dd = custom
    expected,ed = scorers[0].score(hypotheses,problem,oracle,'gpu',diagnostics=True)
    n = len(calls)
    actual,ad = scorers[1].score(hypotheses,problem,oracle,'gpu',diagnostics=True)
    assert len(calls) == n*2 == 16
    assert expected.tobytes() == actual.tobytes()
    assert ed['diagnostic'].tobytes() == ad['diagnostic'].tobytes()


@pytest.mark.parametrize('backend', ['native_cpu','gpu','scalar'])
@pytest.mark.parametrize('seed', [20260710,20260711,20260712])
@pytest.mark.parametrize('options', [{},dict(cluster=True),dict(shadow=True,spawn=True,feedback=True)])
def test_complete_pf_four_epoch_all_state_and_rng_bits(scorers, packer, backend, seed, options):
    a,ra,ca = make_step_case(seed=seed,**options)
    b,rb,cb = make_step_case(seed=seed,**options)
    old = MoveKeyMemoStep(a,scorers[0],backend)
    new = PackedDdStep(b,scorers[1],backend,pack_dll=packer.dll._name)
    a.step,b.step = old,new
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100.+ep
        for ctx in (ca,cb):
            ctx['problem'].rows_bmeas[32:] += .001*ep
            ctx['problem'].Qb *= 1.+ep*.001
        exact_state(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
        assert old.last['requested_moves'] == new.last['requested_moves']
        assert new.last['DD_pack']['native_calls'] > 0
        assert new.last['DD_pack']['fallback_calls'] == 0
        assert new._view is new._oracle is None
        assert '_pair_dd' not in vars(b)


@pytest.mark.parametrize('policy', ['raise','under_warn'])
def test_strict_policy_uses_original_step_and_partial_state(scorers, policy):
    a,ra,ca = make_step_case()
    b,rb,cb = make_step_case()
    original = a.step
    packed = PackedDdStep(b,scorers[1],'gpu')
    b.step = packed
    settings = dict(all='raise') if policy == 'raise' else dict(under='warn')
    outcomes = []
    for runtime,rtk,ctx,call in ((a,ra,ca,original),(b,rb,cb,packed)):
        with np.errstate(**settings):
            try:
                result = call(rtk,ctx)
                outcomes.append(('ok',state(runtime,rtk,result)))
            except FloatingPointError as error:
                outcomes.append((str(error),state(runtime,rtk,None)))
    assert outcomes[0][0] == outcomes[1][0]
    exact_state(outcomes[0][1],outcomes[1][1])
    assert packed.last == dict(original_step_fallback='numpy_error_policy')


def test_changed_method_guard_and_cleanup_on_exception(scorers):
    runtime,rtk,ctx = make_step_case()
    packed = PackedDdStep(runtime,scorers[1],'gpu')
    runtime.step = packed
    original = runtime._conditional_mean
    runtime._conditional_mean = lambda *a:None
    with pytest.raises(ValueError,match='method changed'):
        packed(rtk,ctx)
    runtime._conditional_mean = original
    def fail(*args,**kwargs):
        raise RuntimeError('fixture extraction failed')
    runtime._extract = fail
    with pytest.raises(RuntimeError,match='fixture extraction failed'):
        packed(rtk,ctx)
    assert packed._view is packed._oracle is None


def test_distinct_runtimes_keep_their_fallback_sources(scorers):
    calls = []
    for seed in (20260710,20260711):
        runtime,rtk,ctx = make_step_case(seed=seed)
        call = PackedDdStep(runtime,scorers[1],'gpu')
        calls.append((runtime,call))
    assert calls[0][1].packer is not calls[1][1].packer
    for runtime,call in calls:
        assert call.packer.source_dd.__self__ is runtime
