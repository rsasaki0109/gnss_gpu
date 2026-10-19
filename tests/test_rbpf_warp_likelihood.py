"""Exact serial/warp GPU comparisons, including pivot failures and PF state."""

import ctypes
from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from rbpf_likelihood_batch import BatchScorer, MAX_D  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_move_key_memo import MoveKeyMemoStep  # noqa: E402
from rbpf_warp_batch import WarpBatchScorer  # noqa: E402


@pytest.fixture(scope='module')
def scorers():
    old = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    new = os.environ.get('GNSS_GPU_RBPF_WARP_DLL')
    if not old or not new:
        pytest.skip('Both explicit standalone DLL paths are required')
    with BatchScorer(old) as legacy, WarpBatchScorer(new,kernel='serial') as serial, WarpBatchScorer(new) as warp:
        yield legacy,serial,warp


def exact_state(a,b):
    result = comparison(a,b)
    assert result['discrete_equal'] and result['numeric_bits_equal'] and result['position_bits_equal'],result
    assert result['max_abs_error'] == 0.,result


@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('rows,groups',[(16,1),(64,4),(128,8)])
@pytest.mark.parametrize('uniform,huber,cluster',[
    (True,True,False),(False,True,False),(True,False,False),
    (False,False,False),(True,True,True),(False,True,True)])
def test_all_scores_intermediates_and_cpu_bits(scorers,seed,rows,groups,uniform,huber,cluster):
    oracle,problem,hypotheses = make_case(8,rows=rows,groups=groups,seed=seed,
        marg_uniform=uniform,huber=huber,cluster=cluster)
    before = deepcopy((problem,hypotheses,oracle.cluster_states))
    legacy,serial,warp = scorers
    for mode in ('native_cpu','gpu'):
        reference,details = legacy.score(hypotheses,problem,oracle,mode,diagnostics=True)
        for scorer in (serial,warp):
            values,actual = scorer.score(hypotheses,problem,oracle,mode,diagnostics=True)
            assert values.tobytes() == reference.tobytes()
            assert actual['statuses'].tobytes() == details['statuses'].tobytes()
            assert actual['diagnostic'].tobytes() == details['diagnostic'].tobytes()
            assert actual['scalar_fallbacks'] == details['scalar_fallbacks'] == 0
    a,b = before[0],problem
    for name,value in vars(a).items():
        other = getattr(b,name)
        assert value.tobytes() == other.tobytes() if isinstance(value,np.ndarray) else value == other
    assert before[1] == hypotheses
    for key,value in before[2].items():
        for name,component in value.items():
            other = oracle.cluster_states[key][name]
            assert component.tobytes() == other.tobytes() if isinstance(component,np.ndarray) else component == other


def lu_check(scorer,matrices,rhs,cooperative):
    function = scorer.dll.rbpf_lu_check
    doubles = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    integers = np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    function.argtypes = [doubles,doubles,ctypes.c_int,ctypes.c_int,ctypes.c_int,doubles,integers]
    function.restype = ctypes.c_int
    count,n,_ = matrices.shape
    output = np.empty((count,1+MAX_D+MAX_D*MAX_D))
    meta = np.empty((count,1+MAX_D),dtype=np.int32)
    assert function(matrices,rhs,count,n,int(cooperative),output,meta) == 0
    return output,meta


@pytest.mark.parametrize('dimension',list(range(1,12)))
def test_direct_lu_general_pivots_scales_and_failure_bits(scorers,dimension):
    _legacy,serial,_warp = scorers
    rng = np.random.default_rng(20261016+dimension)
    n = dimension
    matrices = rng.normal(size=(1040,n,n))
    spd = matrices[:1024] @ matrices[:1024].transpose(0,2,1) + .01*np.eye(n)
    matrices[:1024:3] = spd[::3]
    matrices[1:1024:3] *= np.ldexp(np.ones((341,n,1)),rng.integers(-200,201,size=(341,n,1)))
    rhs = rng.normal(size=(1040,n))
    edges = [np.zeros((n,n)),np.eye(n),-np.eye(n),np.eye(n)*np.finfo(float).tiny,
        np.eye(n)*np.nextafter(0.,1.),np.eye(n)*np.finfo(float).max]
    for i,value in enumerate(edges):
        matrices[1024+i] = value
    matrices[1030] = np.eye(n)
    matrices[1030,0,0] = np.nan
    matrices[1031] = np.eye(n)
    matrices[1031,0,0] = np.inf
    matrices[1032] = np.eye(n)
    rhs[1032,0] = np.inf
    matrices[1033] = np.eye(n)
    rhs[1033,0] = np.nan
    matrices[1034] = np.eye(n)
    matrices[1034,0,0] = -0.
    if n > 1:
        matrices[1035] = np.eye(n)[::-1]
        matrices[1036] = np.eye(n)
        matrices[1036,0,0] = 0.
        matrices[1036,1,0] = 2.
        matrices[1036,0,1] = 1.
    before = matrices.tobytes(),rhs.tobytes()
    a,ia = lu_check(serial,matrices,rhs,False)
    b,ib = lu_check(serial,matrices,rhs,True)
    assert ia.tobytes() == ib.tobytes()
    assert a.tobytes() == b.tobytes()
    assert (matrices.tobytes(),rhs.tobytes()) == before
    assert np.any(ia[:,0] == 0) and np.any(ia[:,0] != 0)
    if n > 1:
        assert np.any(ia[:,1:n+1] != np.arange(n))


@pytest.mark.parametrize('seed',[20260710,20260711,20260712])
@pytest.mark.parametrize('options',[{},dict(cluster=True),dict(shadow=True,spawn=True,feedback=True)])
def test_complete_pf_saved_state_and_rng_bits(scorers,seed,options):
    legacy,_serial,warp = scorers
    a,ra,ca = make_step_case(seed=seed,**options)
    b,rb,cb = make_step_case(seed=seed,**options)
    old,new = MoveKeyMemoStep(a,legacy,'gpu'),MoveKeyMemoStep(b,warp,'gpu')
    a.step,b.step = old,new
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100.+ep
        for ctx in (ca,cb):
            ctx['problem'].rows_bmeas[32:] += .001*ep
            ctx['problem'].Qb *= 1.+ep*.001
        exact_state(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
        assert old.last['requested_moves'] == new.last['requested_moves']


@pytest.mark.parametrize('rows,count',[(0,1),(1,3),(63,7),(127,21)])
def test_odd_alignment_and_empty_rows(scorers,rows,count):
    oracle,problem,hypotheses = make_case(count,rows=128,groups=8)
    for name,value in vars(problem).items():
        if name.startswith('rows_'):
            setattr(problem,name,value[:rows].copy())
    reference,detail = scorers[0].score(hypotheses,problem,oracle,'gpu',diagnostics=True)
    for scorer in scorers[1:]:
        actual,ad = scorer.score(hypotheses,problem,oracle,'gpu',diagnostics=True)
        assert reference.tobytes() == actual.tobytes()
        assert detail['diagnostic'].tobytes() == ad['diagnostic'].tobytes()


def test_singular_and_strict_fallbacks(scorers):
    oracle,problem,hypotheses = make_case(8)
    problem.Ppos[:] = 0.
    expected = BatchScorer.scalar(hypotheses,problem,oracle)
    for scorer in scorers:
        result,details = scorer.score(hypotheses,problem,oracle,'gpu')
        assert result.tobytes() == expected.tobytes()
        assert details['scalar_fallbacks'] == 8
    oracle,problem,hypotheses = make_case(8)
    for scorer in scorers:
        with np.errstate(all='raise'),pytest.raises(FloatingPointError):
            scorer.score(hypotheses,problem,oracle,'gpu')
