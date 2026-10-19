"""Exact parent scores, original cluster decisions and complete PF state/RNG."""

import builtins
from collections import UserDict
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_oracle, SCALAR_SOURCE  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_native_canonical import NativeCanonicalStep  # noqa: E402
from rbpf_parent_score import NativeParentScoreStep, ParentScoreEngine  # noqa: E402


@pytest.fixture(scope='module')
def extensions():
    key = os.environ.get('GNSS_GPU_RBPF_CANONICAL_EXT')
    parent = os.environ.get('GNSS_GPU_RBPF_PARENT_EXT')
    if not key or not parent:
        pytest.skip('Explicit standalone canonical and parent-score extensions are required')
    return dict(extension=key,parent_extension=parent)


@pytest.fixture(scope='module')
def engine(extensions):
    result = ParentScoreEngine(extensions['parent_extension'],make_oracle()._basin_parent_score)
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
@pytest.mark.parametrize('groups,satellites',[(1,2),(4,8),(8,32),(16,64)])
def test_random_sparse_overlap_gauge_reference_and_order(engine,seed,groups,satellites):
    rng = np.random.default_rng(seed+groups*17+satellites)
    before = engine.native_calls
    for sample in range(100):
        child,parent = {},{}
        for gi in rng.permutation(groups):
            group = (int(gi)-5,int(gi)%3-1)
            satellites_here = [-(1 << 100)+i*19 for i in range(satellites)]
            entries = {s:int(rng.integers(-10,11)) for s in satellites_here}
            for h in (child,parent):
                if rng.random() < .15:
                    continue
                gauge = (1 << (80+sample*7)) * (-1 if h is child else 1)
                h[group] = {satellites_here[si]:gauge+entries[satellites_here[si]]+
                    (int(rng.integers(-1,2)) if h is parent and rng.random()<.2 else 0)
                    for si in rng.permutation(satellites) if rng.random()>.2}
        saved = deepcopy((child,parent))
        expected = engine.original(child,parent)
        actual = engine.score(child,parent)
        assert type(actual) is tuple and all(type(x) is int for x in actual)
        assert actual == expected and (child,parent) == saved
        # Different independent gauges and all insertion orders reversed.
        shifted = [{g:{s:z+(1 << 16000)*(i+1) for s,z in reversed(list(zs.items()))}
            for g,zs in reversed(list(h.items()))} for i,h in enumerate((child,parent))]
        assert engine.score(*shifted) == expected
        assert engine.score(parent,child) == expected
    assert engine.native_calls-before == 300


@pytest.mark.parametrize('child,parent,expected',[
    ({},{},(0,0)),
    ({(1,0):{}},{(1,0):{1:0}},(0,0)),
    ({(1,0):{1:0}},{(1,0):{1:100,2:101}},(0,0)),
    ({(1,0):{1:0,2:1}},{(2,0):{1:0,2:1}},(0,0)),
    ({(1,0):{1:100,3:5,5:7}},{(1,0):{2:100,3:50,5:52}},(1,1)),
    ({(1,0):{1:2**200,2:2**200+1}},{(1,0):{1:0,2:2}},(0,1)),
    ({(1,0):{1:-2**4096,2:2**4096}},{(1,0):{1:0,2:2**4097}},(1,1)),
    ({(-2**100,2**100):{-2**101:0,2**101:1,0:2}},
     {(-2**100,2**100):{-2**101:100,2**101:101,0:103}},(1,2)),
])
def test_edge_scores_and_minimum_common_reference(engine,child,parent,expected):
    assert engine.original(child,parent) == expected
    assert engine.score(child,parent) == expected


@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('kind',['mapping','dict_subclass','numpy_int','bool','float','group_subclass','group_numpy','sat_numpy','none_empty'])
def test_unsupported_types_keep_original_results(engine,side,kind):
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
    elif kind == 'sat_numpy':
        h = {(1,0):{np.int64(1):4,2:2}}
    elif kind == 'none_empty':
        h[(2,0)] = {}
        h[(3,0)] = None
    else:
        h[(1,0)][1] = dict(numpy_int=np.int64(4),bool=True,float=4.)[kind]
    pair = [{(1,0):{1:14,2:12}},{(1,0):{1:14,2:12}}]
    pair[side] = h
    expected = engine.original(*pair)
    before = engine.fallback_calls
    assert engine.module.score(*pair) is None
    assert engine.score(*pair) == expected
    assert engine.fallback_calls-before == 1


def test_preflight_preserves_custom_subtraction_callbacks_and_errors(engine):
    events = []
    class Integer(int):
        def __sub__(self,other):
            events.append((int(self),int(other)))
            raise ArithmeticError('original subtraction callback')
    child,parent = {(1,0):{1:Integer(3),2:4}},{(1,0):{1:0,2:1}}
    assert engine.module.score(child,parent) is None and events == []
    for call in (engine.original,engine.score):
        with pytest.raises(ArithmeticError,match='original subtraction callback'):
            call(child,parent)
    assert events == [(3,4),(3,4)]
    for call in (engine.original,engine.score):
        with pytest.raises(TypeError,match='not iterable'):
            call(None,parent)


@pytest.mark.parametrize('binding',['set','sorted','len'])
def test_changed_global_builtin_bindings(extensions,binding):
    function = make_oracle()._basin_parent_score
    engine = ParentScoreEngine(extensions['parent_extension'],function)
    pair = [{(1,0):{2:3,1:4}},{(1,0):{1:40,2:39}}]
    events = []
    literal = getattr(builtins,binding)
    def custom(*args,**kwargs):
        events.append(binding)
        return literal(*args,**kwargs)
    function.__globals__[binding] = custom
    expected = function(*pair)
    original_events = list(events)
    events.clear()
    assert engine.score(*pair) == expected and events == original_events
    assert engine.native_calls == 0 and engine.fallback_calls == 1
    del function.__globals__[binding]
    assert engine.score(*pair) == expected and engine.native_calls == 1


def test_custom_function_and_changed_code_keep_original_calls(extensions):
    events = []
    def custom(child,parent):
        events.append((child,parent))
        return (7,9)
    engine = ParentScoreEngine(extensions['parent_extension'],custom)
    assert not engine.literal and engine.score({}, {}) == (7,9) and events == [({}, {})]
    function = make_oracle()._basin_parent_score
    engine = ParentScoreEngine(extensions['parent_extension'],function)
    function.__code__ = (lambda child,parent:(8,10)).__code__
    assert engine.score({}, {}) == (8,10)
    assert engine.native_calls == 0 and engine.fallback_calls == 1


def test_imported_module_and_isolated_function_forms(extensions):
    module = compile(SCALAR_SOURCE.read_bytes(),str(SCALAR_SOURCE),'exec',dont_inherit=True)
    klass = next(c for c in module.co_consts if getattr(c,'co_name',None) == 'RbpfFgo')
    code = next(c for c in klass.co_consts if getattr(c,'co_name',None) == '_basin_parent_score')
    engine = ParentScoreEngine(extensions['parent_extension'],FunctionType(code,{}))
    assert engine.literal
    assert engine.score({(1,0):{1:0,2:1}},{(1,0):{1:9,2:10}}) == (1,1)


def test_no_retained_input_and_updates_between_calls(engine):
    child,parent = {(1,0):{1001:2**200,1002:2**200+1}},{(1,0):{1001:0,1002:1}}
    objects = [child,parent,*child.keys(),*child.values(),*parent.keys(),*parent.values(),
        *child[(1,0)].keys(),*child[(1,0)].values(),*parent[(1,0)].keys(),*parent[(1,0)].values()]
    before = [sys.getrefcount(o) for o in objects]
    for _ in range(10000):
        assert engine.score(child,parent) == (1,1)
    gc.collect()
    assert [sys.getrefcount(o) for o in objects] == before
    parent[(1,0)][1002] += 1
    assert engine.score(child,parent) == (0,1)
    child[(1,0)][1002] += 1
    assert engine.score(child,parent) == (1,1)


@pytest.mark.parametrize('minimum,fraction',[(1,0.),(2,.5),(3,.8),(3,1.),(4,1.)])
def test_cluster_identity_support_exact_key_ties_thresholds_and_updates(engine,minimum,fraction):
    old = make_oracle(cluster_relin=1,cluster_parent_min=minimum,cluster_parent_frac=fraction)
    new = make_oracle(cluster_relin=1,cluster_parent_min=minimum,cluster_parent_frac=fraction)
    new._basin_parent_score = engine.score
    child = {(1,0):{1:0,2:1,3:2,4:3}}
    a = dict(H={(1,0):{0:90,1:100,2:101,3:102,4:103}},pos=np.zeros(3),P=np.eye(3))
    b = dict(H={(1,0):{1:-100,2:-99,3:-98,4:-96}},pos=np.ones(3),P=np.eye(3))
    for parents in ([],[a],[b],[a,b],[b,a],[a,a]):
        states = {('parent',i):s for i,s in enumerate(parents)}
        old.cluster_states = new.cluster_states = states
        assert new._cluster_state(child) is old._cluster_state(child)
    states = {old.basin_id(child):b,('parent',0):a}
    old.cluster_states = new.cluster_states = states
    assert new._cluster_state(child) is old._cluster_state(child)
    states.pop(old.basin_id(child))
    a['H'][(1,0)][4] += 1
    assert new._cluster_state(child) is old._cluster_state(child)
    old.cfg.cluster_relin = new.cfg.cluster_relin = 0
    assert new._cluster_state(child) is old._cluster_state(child) is None


@pytest.mark.parametrize('backend',['scalar','native_cpu','gpu'])
@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('options',[dict(cluster=True),dict(cluster=True,shadow=True,spawn=True,feedback=True)])
def test_complete_pf_four_epoch_all_state_rng_and_move_bits(extensions,scorer,backend,seed,options):
    a,ra,ca = make_step_case(seed=seed,**options)
    b,rb,cb = make_step_case(seed=seed,**options)
    old = NativeCanonicalStep(a,scorer,backend,extension=extensions['extension'])
    new = NativeParentScoreStep(b,scorer,backend,**extensions)
    a.step,b.step = old,new
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100.+ep
        for ctx in (ca,cb):
            ctx['problem'].rows_bmeas[32:] += .001*ep
            ctx['problem'].Qb *= 1.+ep*.001
        exact_state(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
        for name in ('requested_moves','unique_moves','extend_calls','shrink_calls','move_reuse','move_key_memo','native_canonical'):
            assert old.last[name] == new.last[name]
        assert new.last['parent_score']['native_calls'] > 0
        assert new.last['parent_score']['fallback_calls'] == 0
        assert new._view is new._oracle is None
        assert '_basin_parent_score' not in vars(b)
        assert all(clone.__code__ is new._source_functions[name].__code__ for name,clone in new._functions.items()
            if name != '_basin_parent_score')


@pytest.mark.parametrize('policy',['raise','under_warn'])
def test_strict_policy_original_step_and_partial_rng(extensions,scorer,policy):
    a,ra,ca = make_step_case(cluster=True)
    b,rb,cb = make_step_case(cluster=True)
    original = a.step
    native = NativeParentScoreStep(b,scorer,'gpu',**extensions)
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


def test_source_guard_and_exception_cleanup(extensions,scorer):
    runtime,rtk,ctx = make_step_case(cluster=True)
    native = NativeParentScoreStep(runtime,scorer,'gpu',**extensions)
    runtime.step = native
    original = runtime._basin_parent_score
    runtime._basin_parent_score = lambda a,b:(0,0)
    with pytest.raises(ValueError,match='method changed'):
        native(rtk,ctx)
    runtime._basin_parent_score = original
    def fail(*args,**kwargs):
        raise RuntimeError('fixture extraction failed')
    runtime._extract = fail
    with pytest.raises(RuntimeError,match='fixture extraction failed'):
        native(rtk,ctx)
    assert native._view is native._oracle is None
    assert native._scope['_canon_basin_id'] is native.canonical
