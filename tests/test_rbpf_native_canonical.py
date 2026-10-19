"""Exact arbitrary-integer keys, fallbacks, lifetimes and entire PF state."""

import builtins
from collections import UserDict
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys
from types import FunctionType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_oracle, SCALAR_SOURCE  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_move_key_memo import MoveKeyMemoStep  # noqa: E402
from rbpf_native_canonical import CanonicalEngine, NativeCanonicalStep  # noqa: E402


@pytest.fixture(scope='module')
def extension():
    path = os.environ.get('GNSS_GPU_RBPF_CANONICAL_EXT')
    if not path:
        pytest.skip('An explicit standalone canonical extension is required')
    return path


@pytest.fixture(scope='module')
def engine(extension):
    result = CanonicalEngine(extension,make_oracle().basin_id)
    assert result.literal
    return result


@pytest.fixture(scope='module')
def scorer():
    dll = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not dll:
        pytest.skip('An explicit unchanged likelihood DLL is required')
    with BatchScorer(dll) as result:
        yield result


def exact_state(a,b):
    check = comparison(a,b)
    assert check['discrete_equal'] and check['numeric_bits_equal'] and check['position_bits_equal'],check
    assert check['max_abs_error'] == 0.,check


@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('groups,satellites',[(1,1),(4,8),(8,32),(16,64)])
def test_random_keys_gauges_order_and_hashes(engine,seed,groups,satellites):
    rng = np.random.default_rng(seed+groups*17+satellites)
    original = engine.original
    before_native = engine.native_calls
    for sample in range(100):
        h = {}
        group_keys = [(g-5,g%3-1) for g in range(groups)]
        for gi in rng.permutation(groups):
            group = group_keys[gi]
            gauge = (1 << (80+sample*7)) * (-1 if sample%2 else 1)
            sat_keys = [-(1 << 100)+i*19 for i in range(satellites)]
            h[group] = {sat_keys[si]:gauge+int(rng.integers(-100,101)) for si in rng.permutation(satellites)}
        if sample%3 == 0:
            h[(100,0)] = {}
        saved = deepcopy(h)
        expected = original(h)
        key = engine.canonical(h)
        assert type(key) is tuple and key == expected and hash(key) == hash(expected)
        assert h == saved and list(h) == list(saved)
        shifted = {g:{s:z+(1 << 16000) for s,z in reversed(list(values.items()))}
            for g,values in reversed(list(h.items()))}
        assert engine.canonical(shifted) == expected
    assert engine.native_calls-before_native == 200


@pytest.mark.parametrize('hypothesis',[
    {},{(1,0):{}},{(1,0):{9:0}},
    {(1,0):{9:2**200+1,8:2**200}},
    {(1,0):{9:-(2**4096),8:2**4096}},
    {(2**100,-2**100):{2**101:5,-2**102:6}},
    {(1,0):{7:0,3:0,-1:0},(-1,-1):{},(0,1):{9:-7,0:2}}
])
def test_edge_keys_and_input_aliases(engine,hypothesis):
    actual = engine.canonical(hypothesis)
    expected = engine.original(hypothesis)
    assert actual == expected and hash(actual) == hash(expected)
    for group,entries in actual:
        original_group = next(g for g in hypothesis if g == group)
        assert group is original_group
        assert all(type(s) is type(z) is int for s,z in entries)


def test_large_distinct_integers_are_never_merged(engine):
    a = {(1,0):{1:0,2:2**200+1}}
    b = {(1,0):{1:0,2:2**200+2}}
    ka,kb = engine.canonical(a),engine.canonical(b)
    assert ka != kb
    assert len({ka:a,kb:b}) == 2
    assert ka == engine.original(a) and kb == engine.original(b)


@pytest.mark.parametrize('kind',['mapping','dict_subclass','numpy_int','bool','float','group_subclass','group_numpy','none_empty'])
def test_unsupported_types_keep_original_results(engine,kind):
    class Dict(dict):
        pass
    class Tuple(tuple):
        pass
    h = {(1,0):{1:4,2:2}}
    if kind == 'mapping':
        h = UserDict(h)
    elif kind == 'dict_subclass':
        h = Dict(h)
    elif kind == 'group_subclass':
        h = {Tuple((1,0)):{1:4,2:2}}
    elif kind == 'group_numpy':
        h = {(np.int64(1),0):{1:4,2:2}}
    elif kind == 'none_empty':
        h[(2,0)] = None
    else:
        h[(1,0)][1] = dict(numpy_int=np.int64(4),bool=True,float=4.)[kind]
    expected = engine.original(h)
    before = engine.fallback_calls
    assert engine.module.canonical(h) is None
    actual = engine.canonical(h)
    assert actual == expected and hash(actual) == hash(expected)
    assert engine.fallback_calls-before == 1


def test_preflight_preserves_subclass_callbacks_and_exceptions(engine):
    events = []
    class Integer(int):
        def __sub__(self,other):
            events.append((int(self),int(other)))
            raise ArithmeticError('original integer callback')
    h = {(1,0):{1:Integer(3),2:4}}
    assert engine.module.canonical(h) is None and events == []
    for call in (engine.original,engine.canonical):
        with pytest.raises(ArithmeticError,match='original integer callback'):
            call(h)
    assert events == [(3,3),(3,3)]
    for call in (engine.original,engine.canonical):
        with pytest.raises(AttributeError) as error:
            call(None)
        assert str(error.value) == "'NoneType' object has no attribute 'keys'"


@pytest.mark.parametrize('binding',['sorted','min','tuple'])
def test_changed_global_builtin_binding_is_respected(extension,binding):
    function = make_oracle().basin_id
    engine = CanonicalEngine(extension,function)
    h = {(2,0):{7:4,3:2},(1,0):{1:2,2:3}}
    calls = []
    literal = getattr(builtins,binding)
    def custom(*args,**kwargs):
        calls.append(binding)
        return literal(*args,**kwargs)
    function.__globals__[binding] = custom
    expected = function(h)
    events = list(calls)
    calls.clear()
    actual = engine.canonical(h)
    assert actual == expected and calls == events
    assert engine.native_calls == 0 and engine.fallback_calls == 1
    del function.__globals__[binding]
    assert engine.canonical(h) == expected and engine.native_calls == 1


def test_custom_function_and_changed_code_keep_python_calls(extension):
    calls = []
    def custom(h):
        calls.append(h)
        return ('custom',)
    engine = CanonicalEngine(extension,custom)
    assert not engine.literal
    h = {}
    assert engine.canonical(h) == ('custom',) and calls == [h]
    function = make_oracle().basin_id
    engine = CanonicalEngine(extension,function)
    assert engine.literal
    function.__code__ = (lambda h:('changed',)).__code__
    assert engine.canonical(h) == ('changed',)
    assert engine.native_calls == 0 and engine.fallback_calls == 1


def test_imported_module_and_isolated_canonical_code_forms(extension):
    module = compile(SCALAR_SOURCE.read_bytes(),str(SCALAR_SOURCE),'exec',dont_inherit=True)
    code = next(c for c in module.co_consts if getattr(c,'co_name',None) == '_canon_basin_id')
    original = make_oracle().basin_id
    imported = FunctionType(code,{})
    engine = CanonicalEngine(extension,imported)
    assert engine.literal
    for h in ({},{(1,0):{7:2**100,3:2**100+1}}):
        assert engine.canonical(h) == original(h)


def test_reference_counts_no_retained_inputs_and_fresh_nonempty_keys(engine):
    h = {(1,0):{1001:2**200+1,1002:2**200+2}}
    group = next(iter(h))
    integers = h[group]
    objects = [h,group,integers,*integers.keys(),*integers.values()]
    before = [sys.getrefcount(o) for o in objects]
    for _ in range(10000):
        engine.canonical(h)
    gc.collect()
    assert [sys.getrefcount(o) for o in objects] == before
    a,b = engine.canonical(h),engine.canonical(h)
    assert a == b and a is not b
    h[group][1002] += 1
    assert engine.canonical(h) != a


@pytest.mark.parametrize('backend',['scalar','native_cpu','gpu'])
@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('options',[{},dict(cluster=True),dict(shadow=True,spawn=True,feedback=True)])
def test_complete_pf_four_epoch_all_state_rng_and_move_bits(extension,scorer,backend,seed,options):
    a,ra,ca = make_step_case(seed=seed,**options)
    b,rb,cb = make_step_case(seed=seed,**options)
    old,new = MoveKeyMemoStep(a,scorer,backend),NativeCanonicalStep(b,scorer,backend,extension=extension)
    a.step,b.step = old,new
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100.+ep
        for ctx in (ca,cb):
            ctx['problem'].rows_bmeas[32:] += .001*ep
            ctx['problem'].Qb *= 1.+ep*.001
        exact_state(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
        for name in ('requested_moves','unique_moves','extend_calls','shrink_calls','move_reuse','move_key_memo'):
            assert old.last[name] == new.last[name]
        assert new.last['native_canonical']['native_calls'] > 0
        assert new.last['native_canonical']['fallback_calls'] == 0
        assert new._view is new._oracle is None
        assert '_pair_dd' not in vars(b)
        assert new._scope['_canon_basin_id'] is new.canonical
        assert all(clone.__code__ is new._source_functions[name].__code__ for name,clone in new._functions.items())


@pytest.mark.parametrize('policy',['raise','under_warn'])
def test_strict_policy_original_step_and_partial_rng(extension,scorer,policy):
    a,ra,ca = make_step_case()
    b,rb,cb = make_step_case()
    original = a.step
    native = NativeCanonicalStep(b,scorer,'gpu',extension=extension)
    b.step = native
    outcomes = []
    settings = dict(all='raise') if policy == 'raise' else dict(under='warn')
    for runtime,rtk,ctx,call in ((a,ra,ca,original),(b,rb,cb,native)):
        with np.errstate(**settings):
            try:
                result = call(rtk,ctx)
                outcomes.append(('ok',state(runtime,rtk,result)))
            except FloatingPointError as error:
                outcomes.append((str(error),state(runtime,rtk,None)))
    assert outcomes[0][0] == outcomes[1][0]
    exact_state(outcomes[0][1],outcomes[1][1])
    assert native.last == dict(original_step_fallback='numpy_error_policy')


def test_guards_and_cleanup_on_extraction_error(extension,scorer):
    runtime,rtk,ctx = make_step_case()
    native = NativeCanonicalStep(runtime,scorer,'gpu',extension=extension)
    runtime.step = native
    previous = native._source_globals['_canon_basin_id']
    native._source_globals['_canon_basin_id'] = lambda h:()
    with pytest.raises(ValueError,match='canonical function changed'):
        native(rtk,ctx)
    native._source_globals['_canon_basin_id'] = previous
    original = runtime._conditional_mean
    runtime._conditional_mean = lambda *args:None
    with pytest.raises(ValueError,match='method changed'):
        native(rtk,ctx)
    runtime._conditional_mean = original
    def fail(*args,**kwargs):
        raise RuntimeError('fixture extraction failed')
    runtime._extract = fail
    with pytest.raises(RuntimeError,match='fixture extraction failed'):
        native(rtk,ctx)
    assert native._view is native._oracle is None
    assert native._scope['_canon_basin_id'] is native.canonical
