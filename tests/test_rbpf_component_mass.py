"""Exact integer conversion, ordered sums/FIX boundaries and full PF state."""

import builtins
from collections import UserDict
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from rbpf_component_mass import NativeComponentMassStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_parent_score import NativeParentScoreStep  # noqa: E402


@pytest.fixture(scope='module')
def extensions():
    names = dict(extension='GNSS_GPU_RBPF_CANONICAL_EXT',parent_extension='GNSS_GPU_RBPF_PARENT_EXT',
        mass_extension='GNSS_GPU_RBPF_MASS_EXT')
    if any(not os.environ.get(v) for v in names.values()):
        pytest.skip('Explicit standalone key/parent/mass extensions required')
    return {k:os.environ[v] for k,v in names.items()}


@pytest.fixture(scope='module')
def scorer():
    dll = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not dll:
        pytest.skip('Explicit unchanged likelihood DLL required')
    with BatchScorer(dll) as result:
        yield result


def exact_state(a,b):
    check = comparison(a,b)
    assert check['discrete_equal'] and check['numeric_bits_equal'] and check['position_bits_equal'],check
    assert check['max_abs_error'] == 0.,check


def setup_engine(extensions,count=64,**options):
    runtime,rtk,ctx = make_step_case(count=count,**options)
    call = NativeComponentMassStep(runtime,**extensions)
    return runtime,rtk,ctx,call.mass_engine


def compare_mass(engine,runtime,problem,weights,target):
    a = engine.original(runtime,problem,weights,target)
    b = engine.mass(runtime,problem,weights,target)
    assert list(a) == list(b)
    assert all(type(k) is int and type(v) is float for k,v in b.items())
    assert all(np.float64(a[k]).tobytes() == np.float64(b[k]).tobytes() for k in a)
    return b


@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('count,rows,groups',[(0,64,4),(1,16,1),(64,64,4),(128,128,8),(257,128,8)])
def test_random_whole_matrix_and_ordered_sums(extensions,seed,count,rows,groups):
    runtime,_rtk,ctx,engine = setup_engine(extensions,count,rows=rows,groups=groups,seed=seed)
    problem = ctx['problem']
    rng = np.random.default_rng(seed+count)
    for _ in range(20):
        weights = rng.random(count)
        weights /= weights.sum() if count else 1.
        target = runtime._pair_dd(runtime.particles[-1].H,problem) if count else np.zeros(len(problem.pairs))
        compare_mass(engine,runtime,problem,weights,target)
    assert engine.native_calls == 20 and engine.fallback_calls == 0


@pytest.mark.parametrize('value',[0,1,-1,2**53+1,2**53+2,-2**53-1,2**100+3,-2**100+1,2**1023,-2**1023])
def test_arbitrary_integer_gauges_original_fp64_conversion(extensions,value):
    runtime,_rtk,ctx,engine = setup_engine(extensions,8)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    for i,p in enumerate(runtime.particles):
        p.H = {(1,0):{1:2**16000+i,2:2**16000+i-value}}
    weights = np.full(8,1./8)
    target = runtime._pair_dd(runtime.particles[0].H,problem)
    assert compare_mass(engine,runtime,problem,weights,target) == {0:1.}
    assert engine.native_calls == 1


@pytest.mark.parametrize('target',[0.,-0.,.99,-.99,1.99,-1.99,2**100,1e300,np.nan,np.inf,-np.inf])
def test_int_equality_means_truncation_not_double_equality(extensions,target):
    target = float(target)
    runtime,_rtk,ctx,engine = setup_engine(extensions,4)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    values = [0,1,-1,int(1e300)]
    for p,z in zip(runtime.particles,values):
        p.H = {(1,0):{1:0,2:-z}}
    result = compare_mass(engine,runtime,problem,np.array([.1,.2,.3,.4]),np.array([target],dtype=np.float64))
    if np.isfinite(target):
        expected = sum(w for z,w in zip(values,[.1,.2,.3,.4]) if int(float(z)) == int(target))
        assert result == {0:expected}
    else:
        assert result == {}


def test_object_target_preserves_original_isfinite_error(extensions):
    runtime,_rtk,ctx,engine = setup_engine(extensions,1)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    target = np.array([2**100],dtype=object)
    for call in (engine.original,engine.mass):
        with pytest.raises(TypeError,match="ufunc 'isfinite' not supported"):
            call(runtime,problem,np.ones(1),target)
    assert engine.fallback_calls == 1 and engine.native_calls == 0


def test_non_target_overflow_still_raises_in_original_expansion_order(extensions):
    runtime,_rtk,ctx,engine = setup_engine(extensions,2)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2),((1,0),1,3)]
    runtime.particles[0].H = {(1,0):{1:2**2000,2:2**2000,3:0}}
    runtime.particles[1].H = {}
    weights,target = np.array([.5,.5]),np.array([0.,np.nan])
    for call in (engine.original,engine.mass):
        with pytest.raises(OverflowError,match='int too large to convert to float'):
            call(runtime,problem,weights,target)
    assert engine.native_calls == engine.fallback_calls == 0


def test_sum_order_subnormals_and_adjacent_fix_thresholds(extensions):
    runtime,_rtk,ctx,engine = setup_engine(extensions,128)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    for p in runtime.particles:
        p.H = {(1,0):{1:0,2:-1}}
    weights = np.r_[.5,np.full(126,2.**-55),np.nextafter(0.,1.)]
    target = np.array([1.])
    result = compare_mass(engine,runtime,problem,weights,target)
    expected = 0.
    for w in weights:
        expected += float(w)
    assert np.float64(result[0]).tobytes() == np.float64(expected).tobytes()
    assert result[0] != float(np.sum(weights))  # A tree sum would change a FIX boundary.
    for threshold in (np.nextafter(expected,-np.inf),expected,np.nextafter(expected,np.inf)):
        original = [pi for pi,am in engine.original(runtime,problem,weights,target).items() if am>threshold]
        native = [pi for pi,am in engine.mass(runtime,problem,weights,target).items() if am>threshold]
        assert original == native


@pytest.mark.parametrize('kind',['mapping','dict_subclass','numpy_integer','bool','float','noncontiguous_weight','noncontiguous_target','float32_weight','negative_weight','nan_weight','infinite_weight'])
def test_unsupported_domain_executes_original(extensions,kind):
    runtime,_rtk,ctx,engine = setup_engine(extensions,4)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    for p in runtime.particles:
        p.H = {(1,0):{1:0,2:-1}}
    weights,target = np.full(4,.25),np.array([1.])
    class Dict(dict):
        pass
    h = runtime.particles[0].H
    if kind == 'mapping':
        runtime.particles[0].H = UserDict(h)
    elif kind == 'dict_subclass':
        runtime.particles[0].H = Dict(h)
    elif kind in ('numpy_integer','bool','float'):
        h[(1,0)][2] = dict(numpy_integer=np.int64(-1),bool=True,float=-1.)[kind]
    elif kind == 'noncontiguous_weight':
        weights = np.full(8,.25)[::2]
    elif kind == 'noncontiguous_target':
        problem.pairs.append(((1,0),1,2))
        target = np.ones(4)[::2]
    elif kind == 'float32_weight':
        weights = weights.astype(np.float32)
    else:
        weights[0] = dict(negative_weight=-.25,nan_weight=np.nan,infinite_weight=np.inf)[kind]
    compare_mass(engine,runtime,problem,weights,target)
    assert engine.native_calls == 0 and engine.fallback_calls == 1


def test_custom_integer_callback_and_error_preserved_once(extensions):
    runtime,_rtk,ctx,engine = setup_engine(extensions,1)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    calls = []
    class Integer(int):
        def __sub__(self,other):
            calls.append((int(self),int(other)))
            raise ArithmeticError('original subtraction callback')
    runtime.particles[0].H = {(1,0):{1:Integer(0),2:-1}}
    for call in (engine.original,engine.mass):
        with pytest.raises(ArithmeticError,match='original subtraction callback'):
            call(runtime,problem,np.ones(1),np.ones(1))
    assert calls == [(0,-1),(0,-1)]
    assert engine.fallback_calls == 1 and engine.native_calls == 0


@pytest.mark.parametrize('binding',['int','float','enumerate','len'])
def test_custom_step_builtin_bindings_keep_original_calls(extensions,binding):
    runtime,_rtk,ctx,engine = setup_engine(extensions,4)
    problem = ctx['problem']
    for p in runtime.particles:
        p.H = {(1,0):{1000:0,1001:0}}
    weights,target = np.full(4,.25),np.zeros(len(problem.pairs))
    events = []
    builtin = getattr(builtins,binding)
    def custom(*a,**kw):
        events.append(binding)
        return builtin(*a,**kw)
    engine.scope[binding] = custom
    expected = engine.original(runtime,problem,weights,target)
    original_events = list(events)
    events.clear()
    actual = engine.mass(runtime,problem,weights,target)
    assert actual == expected and events == original_events
    assert engine.native_calls == 0 and engine.fallback_calls == 1


def test_input_lifetimes_fresh_results_and_mutations(extensions):
    runtime,_rtk,ctx,engine = setup_engine(extensions,4)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    for p in runtime.particles:
        p.H = {(1,0):{1:0,2:-1}}
    weights,target = np.full(4,.25),np.array([1.])
    objects = [problem.pairs,target,weights,*[p.H for p in runtime.particles]]
    counts = [sys.getrefcount(o) for o in objects]
    for _ in range(10000):
        engine.mass(runtime,problem,weights,target)
    gc.collect()
    assert [sys.getrefcount(o) for o in objects] == counts
    a,b = engine.mass(runtime,problem,weights,target),engine.mass(runtime,problem,weights,target)
    assert a == b == {0:1.} and a is not b
    runtime.particles[0].H[(1,0)][2] = -2
    assert engine.mass(runtime,problem,weights,target) == {0:.75}
    target[0] = 2.
    assert engine.mass(runtime,problem,weights,target) == {0:.25}
    weights[0] = .125
    assert engine.mass(runtime,problem,weights,target) == {0:.125}
    assert engine.peak_matrix_bytes == 32


@pytest.mark.parametrize('backend',['scalar','native_cpu','gpu'])
@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('options',[{},dict(cluster=True),dict(shadow=True,spawn=True,feedback=True)])
def test_complete_pf_four_epoch_state_rng_and_decision_bits(extensions,scorer,backend,seed,options):
    a,ra,ca = make_step_case(seed=seed,**options)
    b,rb,cb = make_step_case(seed=seed,**options)
    old = NativeParentScoreStep(a,scorer,backend,extension=extensions['extension'],parent_extension=extensions['parent_extension'])
    new = NativeComponentMassStep(b,scorer,backend,**extensions)
    a.step,b.step = old,new
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100.+ep
        for ctx in (ca,cb):
            ctx['problem'].rows_bmeas[32:] += .001*ep
            ctx['problem'].Qb *= 1.+ep*.001
        exact_state(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
        for name in ('requested_moves','unique_moves','extend_calls','shrink_calls','move_reuse','move_key_memo','native_canonical','parent_score'):
            assert old.last[name] == new.last[name]
        assert new.last['component_mass']['native_calls'] == 1
        assert new.last['component_mass']['fallback_calls'] == 0
        assert new._view is new._oracle is None
        assert '_rbpf_component_mass' not in new._scope


@pytest.mark.parametrize('policy',['raise','under_warn'])
def test_strict_policy_original_partial_state_and_rng(extensions,scorer,policy):
    a,ra,ca = make_step_case()
    b,rb,cb = make_step_case()
    original = a.step
    new = NativeComponentMassStep(b,scorer,'gpu',**extensions)
    b.step = new
    outcomes = []
    settings = dict(all='raise') if policy == 'raise' else dict(under='warn')
    for runtime,rtk,ctx,call in ((a,ra,ca,original),(b,rb,cb,new)):
        with np.errstate(**settings):
            try:
                value = call(rtk,ctx)
                outcomes.append(('ok',state(runtime,rtk,value)))
            except FloatingPointError as error:
                outcomes.append((str(error),state(runtime,rtk,None)))
    assert outcomes[0][0] == outcomes[1][0]
    exact_state(outcomes[0][1],outcomes[1][1])
    assert new.last == dict(original_step_fallback='numpy_error_policy')


def test_changed_method_guard_and_extraction_error_cleanup(extensions,scorer):
    runtime,rtk,ctx = make_step_case()
    new = NativeComponentMassStep(runtime,scorer,'gpu',**extensions)
    runtime.step = new
    original = runtime._pair_dd
    runtime._pair_dd = lambda *args:None
    with pytest.raises(ValueError,match='method changed'):
        new(rtk,ctx)
    runtime._pair_dd = original
    def fail(*a,**kw):
        raise RuntimeError('fixture extraction failed')
    runtime._extract = fail
    with pytest.raises(RuntimeError,match='fixture extraction failed'):
        new(rtk,ctx)
    assert new._view is new._oracle is None
    assert '_rbpf_component_mass' not in new._scope
