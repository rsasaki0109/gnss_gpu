from types import SimpleNamespace
import numpy as np
import pytest
from gnss_gpu.bandlimited_multipath_v2_cpu import _select_equivalent_converged
from experiments.run_cpu_bandlimited_multipath import evaluate
import experiments.run_cpu_bandlimited_multipath as experiment
from gnss_gpu.bandlimited_multipath_v2_cpu import fit_bandlimited_paths


def candidate(error, x, success):
    return SimpleNamespace(fun=error, x=np.array(x), success=success)


def test_equivalent_converged_candidate_wins_numerical_tie():
    failed = candidate(10., [.1,.5], False)
    good = candidate(10.+1e-14, [.1+1e-9,.5], True)
    assert _select_equivalent_converged([failed,good]) is good


@pytest.mark.parametrize('error,x', [(10.001,[.1,.5]), (10.,[.2,.5]), (np.inf,[.1,.5]), (10.,[np.nan,.5])])
def test_worse_or_distinct_solution_cannot_hide_failure(error,x):
    failed = candidate(10., [.1,.5], False)
    assert _select_equivalent_converged([failed,candidate(error,x,True)]) is failed


def test_original_success_and_all_failures_are_preserved():
    good = candidate(1.,[.1,.5],True)
    failed = candidate(2.,[.1,.5],False)
    assert _select_equivalent_converged([good,failed]) is good
    assert _select_equivalent_converged([failed]) is failed


@pytest.mark.parametrize('replica',[10038,10049,10054,10067,10074,10098])
def test_development_nonconvergence_regression(monkeypatch, replica):
    old = evaluate((3.6,55.,.8,.5,np.pi/2,replica))
    monkeypatch.setattr(experiment,'fit_bandlimited_paths',fit_bandlimited_paths)
    new = evaluate((3.6,55.,.8,.5,np.pi/2,replica))
    assert old['fit']['reason'] == 'optimizer did not converge'
    assert new['fit']['selected_model'] == 'two_path'
    assert new['observation_sha256'] == old['observation_sha256']
    assert abs(new['fit']['first_delay_chips']-old['fit']['first_delay_chips']) < 1e-7
    assert abs(new['selected_error_m']) < 10
