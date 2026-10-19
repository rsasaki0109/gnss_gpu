"""Held-DD masses, release threshold/order, prelude mutations and PF state."""

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
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


@pytest.fixture(scope='module')
def extensions():
    names = dict(extension='GNSS_GPU_RBPF_CANONICAL_EXT',parent_extension='GNSS_GPU_RBPF_PARENT_EXT',mass_extension='GNSS_GPU_RBPF_MASS_EXT')
    if any(not os.environ.get(name) for name in names.values()):
        pytest.skip('Explicit unchanged key/parent/component-mass extensions required')
    return {k:os.environ[v] for k,v in names.items()}


@pytest.fixture(scope='module')
def scorer():
    dll = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not dll:
        pytest.skip('Explicit unchanged likelihood DLL required')
    with BatchScorer(dll) as result:
        yield result


def exact(a,b):
    result = comparison(a,b)
    assert result['discrete_equal'] and result['numeric_bits_equal'] and result['position_bits_equal'],result
    assert result['max_abs_error'] == 0.,result


def setup(extensions,count=64,**options):
    runtime,rtk,ctx = make_feedback_case(count,**options)
    call = NativeFeedbackMassStep(runtime,**extensions)
    return runtime,rtk,ctx,call


def check_statistics(engine,runtime,problem,weights,committed):
    a = engine.original(runtime,problem,weights,committed,runtime.cfg)
    b = engine.release_mass(runtime,problem,weights,committed,runtime.cfg)
    exact(a,b)
    return b


@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('count,rows,groups',[(0,64,4),(1,16,1),(64,64,4),(128,128,8)])
def test_random_held_mass_group_order_and_missing_values(extensions,seed,count,rows,groups):
    runtime,rtk,ctx,call = setup(extensions,count,rows=rows,groups=groups,seed=seed)
    rng = np.random.default_rng(seed+count)
    engine,problem = call.feedback_engine,ctx['problem']
    for _ in range(25):
        weights = rng.random(count)
        weights /= weights.sum() if count else 1.
        committed = {k:v for k,v in rtk._committed.items() if rng.random()>.2}
        for k in list(committed):
            if rng.random()<.1:
                committed[k] = None
        check_statistics(engine,runtime,problem,weights,committed)
    assert engine.native_calls == 25 and engine.fallback_calls == 0


@pytest.mark.parametrize('difference',[0.,-.0,.5,1.5,2.5,-.5,-1.5,-2.5,2.**53,2.**100])
def test_original_round_to_even_and_large_held_integer(extensions,difference):
    runtime,rtk,ctx,call = setup(extensions,4)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    value = int(np.rint(difference))
    for p in runtime.particles:
        p.H = {(1,0):{1:2**16000,2:2**16000-value}}
    result = check_statistics(call.feedback_engine,runtime,problem,np.full(4,.25),{(1,0):difference,(2,0):0.})
    assert result == ([((1,0),1,2,1.)],{(1,0):1},{})


def test_strict_release_threshold_ulp_and_original_sum_order(extensions):
    runtime,rtk,ctx,call = setup(extensions,128)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2)]
    for p in runtime.particles:
        p.H = {(1,0):{1:0,2:-1}}
    weights = np.r_[.5,np.full(126,2.**-55),np.nextafter(0.,1.)]
    committed = {(1,0):0.,(2,0):-1.}
    expected = 0.
    for w in weights:
        expected += float(w)
    assert expected != float(np.sum(weights))
    for threshold in (np.nextafter(expected,-np.inf),expected,np.nextafter(expected,np.inf)):
        runtime.cfg.fb_rel_mass = float(threshold)
        held,gh,gb = check_statistics(call.feedback_engine,runtime,problem,weights,committed)
        assert held == [((1,0),1,2,expected)] and gh == {(1,0):1}
        assert gb == ({(1,0):1} if expected<threshold else {})


@pytest.mark.parametrize('kind',['mapping','dict_subclass','integer_hold','numpy_hold','large_hold','unaligned_weight','noncontiguous_weight','negative_weight','nan_weight'])
def test_unsupported_domain_runs_original_block_once(extensions,kind):
    runtime,rtk,ctx,call = setup(extensions,4)
    committed,weights = deepcopy(rtk._committed),np.full(4,.25)
    class Dict(dict):
        pass
    if kind == 'mapping':
        committed = UserDict(committed)
    elif kind == 'dict_subclass':
        committed = Dict(committed)
    elif kind in ('integer_hold','numpy_hold','large_hold'):
        committed[next(iter(committed))] = dict(integer_hold=1,numpy_hold=np.float64(1.),large_hold=1e200)[kind]
    elif kind == 'unaligned_weight':
        weights = np.ndarray((4,),dtype=np.float64,buffer=bytearray(33),offset=1)
        weights[:] = .25
    elif kind == 'noncontiguous_weight':
        weights = np.full(8,.25)[::2]
    else:
        weights[0] = -.25 if kind == 'negative_weight' else np.nan
    check_statistics(call.feedback_engine,runtime,ctx['problem'],weights,committed)
    assert call.feedback_engine.native_calls == 0 and call.feedback_engine.fallback_calls == 1


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_bad_hold_and_non_target_overflow_keep_original_error_order(extensions,bad):
    runtime,rtk,ctx,call = setup(extensions,1)
    problem = ctx['problem']
    problem.pairs = [((1,0),1,2),((1,0),1,3)]
    runtime.particles[0].H = {(1,0):{1:2**2048,2:2**2048,3:0}}
    held = {(1,0):bad,(2,0):0.}
    for function in (call.feedback_engine.original,call.feedback_engine.release_mass):
        with pytest.raises(OverflowError,match='int too large to convert to float'):
            function(runtime,problem,np.ones(1),held,runtime.cfg)
    assert call.feedback_engine.fallback_calls == 1


def test_custom_mapping_get_callback_order(extensions):
    runtime,rtk,ctx,call = setup(extensions,4)
    events = []
    class Mapping(dict):
        def get(self,key,*args):
            events.append(key)
            return super().get(key,*args)
    held = Mapping(rtk._committed)
    engine = call.feedback_engine
    a = engine.original(runtime,ctx['problem'],np.full(4,.25),held,runtime.cfg)
    old_events = list(events)
    events.clear()
    b = engine.release_mass(runtime,ctx['problem'],np.full(4,.25),held,runtime.cfg)
    exact(a,b)
    assert events == old_events and engine.native_calls == 0 and engine.fallback_calls == 1


def test_custom_rounding_binding_keeps_original_calls(extensions,monkeypatch):
    runtime,rtk,ctx,call = setup(extensions,4)
    literal,events = np.rint,[]
    def custom(value):
        events.append(float(value))
        return literal(value)
    monkeypatch.setattr(np,'rint',custom)
    engine = call.feedback_engine
    a = engine.original(runtime,ctx['problem'],np.full(4,.25),rtk._committed,runtime.cfg)
    old_events = list(events)
    events.clear()
    b = engine.release_mass(runtime,ctx['problem'],np.full(4,.25),rtk._committed,runtime.cfg)
    exact(a,b)
    assert events == old_events and engine.native_calls == 0 and engine.fallback_calls == 1


def test_no_retained_inputs_fresh_groups_and_mutations(extensions):
    runtime,rtk,ctx,call = setup(extensions,4)
    problem,weights,held = ctx['problem'],np.full(4,.25),rtk._committed
    objects = [weights,held,problem.pairs,*[p.H for p in runtime.particles]]
    before = [sys.getrefcount(o) for o in objects]
    for _ in range(10000):
        call.feedback_engine.release_mass(runtime,problem,weights,held,runtime.cfg)
    gc.collect()
    assert [sys.getrefcount(o) for o in objects] == before
    a = call.feedback_engine.release_mass(runtime,problem,weights,held,runtime.cfg)
    b = call.feedback_engine.release_mass(runtime,problem,weights,held,runtime.cfg)
    exact(a,b)
    assert all(x is not y for x,y in zip(a,b))
    held.clear()
    assert call.feedback_engine.release_mass(runtime,problem,weights,held,runtime.cfg) == ([],{},{})


def invoke_feedback(call,runtime,rtk,problem,weights,dd,used,**kw):
    call._scope['_rbpf_feedback_mass'] = call.feedback_engine.release_mass
    try:
        return call._functions['_feedback'](runtime,rtk,problem,weights,dd,used,**kw)
    finally:
        del call._scope['_rbpf_feedback_mass']


@pytest.mark.parametrize('prelude',['support','ddpr'])
def test_prelude_runs_once_and_native_reads_changed_particles_after_release(extensions,prelude):
    a,ra,ca = make_feedback_case(count=8)
    b,rb,cb = make_feedback_case(count=8)
    new = NativeFeedbackMassStep(b,**extensions)
    for runtime,rtk in ((a,ra),(b,rb)):
        runtime.cfg.spawn_enable = 1
        if prelude == 'support':
            runtime.cfg.fb_support_min = 10000
            runtime.cfg.fb_support_m = 1
        else:
            runtime.cfg.fb_ddpr_rel_d = .1
            runtime.cfg.fb_ddpr_rel_m = 1
        old = rtk._release_ambiguity
        def purge(sat,freq,reason,*,rt=runtime,release=old):
            release(sat,freq,reason)
            for p in rt.particles:
                p.H.clear()
        rtk._release_ambiguity = purge
    dd = np.zeros(len(ca['problem'].pairs))
    # Exercise D235 first; D236 must not retain its now-stale DD matrix.
    new.mass_engine.mass(b,cb['problem'],np.full(8,.125),dd)
    x = a._feedback(ra,ca['problem'],np.full(8,.125),dd,[],rep_dpr=1.,ddpr_qual=.1)
    y = invoke_feedback(new,b,rb,cb['problem'],np.full(8,.125),dd,[],rep_dpr=1.,ddpr_qual=.1)
    exact(state(a,ra,x),state(b,rb,y))
    assert ra.events and ra.events == rb.events
    assert new.feedback_engine.native_calls == 1 and new.feedback_engine.held_targets == 0


@pytest.mark.parametrize('backend',['scalar','native_cpu','gpu'])
@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('options',[{},dict(cluster=True),dict(shadow=True,spawn=True,preload=False),dict(feedback=False,preload=False)])
def test_complete_pf_four_epoch_state_rng_and_graph_event_bits(extensions,scorer,backend,seed,options):
    a,ra,ca = make_feedback_case(seed=seed,**options)
    b,rb,cb = make_feedback_case(seed=seed,**options)
    old = NativeComponentMassStep(a,scorer,backend,**extensions)
    new = NativeFeedbackMassStep(b,scorer,backend,**extensions)
    a.step,b.step = old,new
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100.+ep
        for ctx in (ca,cb):
            ctx['problem'].rows_bmeas[32:] += .001*ep
            ctx['problem'].Qb *= 1.+ep*.001
        exact(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
        for key in ('requested_moves','unique_moves','extend_calls','shrink_calls','move_reuse','move_key_memo','native_canonical','parent_score','component_mass'):
            assert old.last[key] == new.last[key]
        assert new.last['feedback_mass']['native_calls'] == int(bool(b.cfg.fb_enable))
        assert new.last['feedback_mass']['fallback_calls'] == 0
        assert new._view is new._oracle is None
        assert '_rbpf_feedback_mass' not in new._scope and '_rbpf_component_mass' not in new._scope


@pytest.mark.parametrize('policy',['raise','under_warn'])
def test_strict_policy_original_partial_state_and_rng(extensions,scorer,policy):
    a,ra,ca = make_feedback_case()
    b,rb,cb = make_feedback_case()
    original = a.step
    new = NativeFeedbackMassStep(b,scorer,'gpu',**extensions)
    b.step = new
    outcomes = []
    settings = dict(all='raise') if policy == 'raise' else dict(under='warn')
    for runtime,rtk,ctx,call in ((a,ra,ca,original),(b,rb,cb,new)):
        with np.errstate(**settings):
            try:
                output = call(rtk,ctx)
                outcomes.append(('ok',state(runtime,rtk,output)))
            except FloatingPointError as error:
                outcomes.append((str(error),state(runtime,rtk,None)))
    assert outcomes[0][0] == outcomes[1][0]
    exact(outcomes[0][1],outcomes[1][1])
    assert new.last == dict(original_step_fallback='numpy_error_policy')


def test_method_guard_and_hook_cleanup(extensions,scorer):
    runtime,rtk,ctx = make_feedback_case()
    new = NativeFeedbackMassStep(runtime,scorer,'gpu',**extensions)
    runtime.step = new
    original = runtime._feedback
    runtime._feedback = lambda *a,**kw:None
    with pytest.raises(ValueError,match='method changed'):
        new(rtk,ctx)
    runtime._feedback = original
    def fail(*a,**kw):
        raise RuntimeError('fixture extraction failed')
    runtime._extract = fail
    with pytest.raises(RuntimeError,match='fixture extraction failed'):
        new(rtk,ctx)
    assert new._view is new._oracle is None
    assert '_rbpf_feedback_mass' not in new._scope and '_rbpf_component_mass' not in new._scope
