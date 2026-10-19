"""Differential checks for the explicitly built experimental likelihood DLL."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from rbpf_likelihood_batch import BatchScorer, MAX_D  # noqa: E402
from rbpf_likelihood_cases import capture, make_case  # noqa: E402


@pytest.fixture(scope='module')
def scorer():
    path = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not path:
        pytest.skip('Standalone RBPF prototype DLL is not explicitly selected')
    with BatchScorer(path, capacity=512) as instance:
        yield instance


def check_case(scorer, oracle, problem, hypotheses):
    before = deepcopy((problem, hypotheses, oracle.cluster_states))
    expected = BatchScorer.scalar(hypotheses, problem, oracle)
    for mode in ('native_cpu', 'gpu'):
        result, details = scorer.score(hypotheses, problem, oracle, mode, diagnostics=True)
        assert details['scalar_fallbacks'] == 0
        assert details['unique'] == len({oracle.basin_id(h) for h in hypotheses})
        # Diagnostic screen, not permission to change a positioning tolerance.
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-8)
        for index, hypothesis in enumerate(details['unique_hypotheses']):
            _value, original = capture(oracle, hypothesis, problem)
            diagnostic = details['diagnostic'][index]
            n = int(diagnostic[0])
            assert n == len(original['sol'])
            np.testing.assert_allclose(diagnostic[3:3+n], original['sol'], rtol=1e-12, atol=1e-9)
            normal = diagnostic[3+MAX_D:].reshape(MAX_D, MAX_D)[:n, :n]
            np.testing.assert_allclose(normal, original['Lam'], rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(diagnostic[1:3], [original['logdet_lam'], original['logdet_p']], rtol=1e-12, atol=1e-9)
    old_problem, old_hypotheses, old_states = before
    assert hypotheses == old_hypotheses
    for key, value in vars(problem).items():
        old = getattr(old_problem, key)
        if isinstance(value, np.ndarray):
            assert value.tobytes() == old.tobytes()
        else:
            assert value == old
    for key, state in oracle.cluster_states.items():
        assert state['H'] == old_states[key]['H']
        assert state['pos'].tobytes() == old_states[key]['pos'].tobytes()
        assert state['P'].tobytes() == old_states[key]['P'].tobytes()


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('rows,groups', [(16, 1), (64, 4), (128, 8)])
@pytest.mark.parametrize('marg_uniform,huber,cluster', [
    (True, True, False), (False, True, False), (True, False, False),
    (False, False, False), (True, True, True), (False, True, True)])
def test_original_numeric_methods(scorer, seed, rows, groups, marg_uniform, huber, cluster):
    oracle, problem, hypotheses = make_case(8, rows=rows, groups=groups, seed=seed,
        marg_uniform=marg_uniform, huber=huber, cluster=cluster)
    check_case(scorer, oracle, problem, hypotheses)


def test_interleaved_row_order_and_nuisance_first_appearance(scorer):
    oracle, problem, hypotheses = make_case(16, cluster=True)
    order = np.random.default_rng(15).permutation(len(problem.rows_v0))
    for key, value in vars(problem).items():
        if key.startswith('rows_'):
            setattr(problem, key, value[order].copy())
    check_case(scorer, oracle, problem, hypotheses)


def test_all_cp_uniform_and_no_remaining_measurements(scorer):
    oracle, problem, hypotheses = make_case(1)
    cp = problem.rows_iscp
    for key, value in vars(problem).items():
        if key.startswith('rows_'):
            setattr(problem, key, value[cp].copy())
    check_case(scorer, oracle, problem, hypotheses)


def test_general_covariance_is_not_silently_symmetrized(scorer):
    oracle, problem, hypotheses = make_case(12)
    problem.Ppos[0, 1] += 1e-8
    check_case(scorer, oracle, problem, hypotheses)


def test_nonfinite_shadow_centre_uses_original_prior(scorer):
    oracle, problem, hypotheses = make_case(8, cluster=True)
    for state in oracle.cluster_states.values():
        state['pos'][0] = np.nan
    check_case(scorer, oracle, problem, hypotheses)


@pytest.mark.parametrize('rows,groups,capacity', [(130, 4, 512), (72, 9, 512), (64, 4, 4)])
def test_supported_bounds_fall_back_without_dropping_candidates(scorer, rows, groups, capacity):
    oracle, problem, hypotheses = make_case(8, rows=rows, groups=groups)
    with BatchScorer(os.environ['GNSS_GPU_RBPF_PROBE_DLL'], capacity=capacity) as limited:
        expected = BatchScorer.scalar(hypotheses, problem, oracle)
        result, details = limited.score(hypotheses, problem, oracle)
        assert details['scalar_fallbacks'] == 8
        assert result.tobytes() == expected.tobytes()


def test_strict_numpy_policy_keeps_original_calls(scorer):
    oracle, problem, hypotheses = make_case(8)
    hypotheses = hypotheses[:2]
    with np.errstate(all='raise'):
        expected = BatchScorer.scalar(hypotheses, problem, oracle)
        result, details = scorer.score(hypotheses, problem, oracle)
    assert details['scalar_fallbacks'] == 2
    assert result.tobytes() == expected.tobytes()


def test_strict_numpy_exception_is_preserved(scorer):
    oracle, problem, hypotheses = make_case(8)
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError) as original:
            BatchScorer.scalar(hypotheses, problem, oracle)
        with pytest.raises(FloatingPointError) as batch:
            scorer.score(hypotheses, problem, oracle)
    assert str(batch.value) == str(original.value)


def test_singular_prior_uses_original_failure_result(scorer):
    oracle, problem, hypotheses = make_case(8)
    problem.Ppos[:] = 0.
    for mode in ('native_cpu', 'gpu'):
        expected = BatchScorer.scalar(hypotheses, problem, oracle)
        result, details = scorer.score(hypotheses, problem, oracle, mode)
        assert details['scalar_fallbacks'] == 8
        assert result.tobytes() == expected.tobytes()


def test_empty_request_has_no_scores(scorer):
    oracle, problem, _hypotheses = make_case(1)
    result, details = scorer.score([], problem, oracle)
    assert result.shape == (0,) and details['unique'] == 0
