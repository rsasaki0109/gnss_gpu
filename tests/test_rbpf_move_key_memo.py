"""Bit-exact complete-step checks for the local move-key alternative."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_move_key_memo import MoveKeyMemoStep  # noqa: E402
from rbpf_move_reuse import MoveReuseStep  # noqa: E402


@pytest.fixture(scope='module')
def scorer():
    dll = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not dll:
        pytest.skip('Explicit experimental DLL is required')
    with BatchScorer(dll) as value:
        yield value


def exact(a,b):
    check = comparison(a,b)
    assert check['discrete_equal'] and check['numeric_bits_equal'] and check['max_abs_error'] == 0., check


@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('options',[{},dict(cluster=True),dict(shadow=True,spawn=True,feedback=True)])
@pytest.mark.parametrize('mode',['scalar','native_cpu','gpu'])
def test_full_state_and_original_bound_methods(scorer,seed,options,mode):
    a,ra,ca = make_step_case(seed=seed,**options)
    b,rb,cb = make_step_case(seed=seed,**options)
    old = a.step if mode == 'scalar' else MoveReuseStep(a,scorer,mode)
    new = MoveKeyMemoStep(b,scorer,mode)
    b.step = new
    bound = {name:getattr(b,name) for name in ('_loglik','_pair_dd','_conditional_mean','_feedback')}
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100. + ep
        for ctx in (ca,cb):
            ctx['problem'].rows_bmeas[32:] += .001 * ep
        exact(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
        assert new.last['move_key_memo']['canonical_source_calls'] < new.last['move_key_memo']['canonical_requests']
        assert new.last['move_key_memo']['DD_entries'] == 0
        assert all(getattr(b,name) == method for name,method in bound.items())


def test_unusual_integer_and_large_gauge_preserve_exact_states(scorer):
    a,ra,ca = make_step_case()
    b,rb,cb = make_step_case()
    full = a._adopt(ca['problem'],0)
    variants = [full,{g:{s:z+2**100 for s,z in zm.items()} for g,zm in full.items()},
        {g:{s:np.int64(z) for s,z in zm.items()} for g,zm in full.items()}]
    for runtime in (a,b):
        for i,p in enumerate(runtime.particles):
            p.H = deepcopy(variants[i % 3])
    old,new = MoveReuseStep(a,scorer,'gpu'),MoveKeyMemoStep(b,scorer,'gpu')
    exact(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
    assert new.last['move_key_memo']['canonical_bypass'] > 0


def test_strict_policy_uses_original_complete_step(scorer):
    a,ra,ca = make_step_case()
    b,rb,cb = make_step_case()
    new = MoveKeyMemoStep(b,scorer,'gpu')
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError):
            a.step(ra,ca)
        with pytest.raises(FloatingPointError):
            new(rb,cb)
    exact(state(a,ra,None),state(b,rb,None))
    assert new.last == dict(original_step_fallback='numpy_error_policy')
