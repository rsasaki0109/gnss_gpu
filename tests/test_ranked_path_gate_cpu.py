import numpy as np
import pytest
from scipy.stats import binom
from gnss_gpu.ranked_path_gate_cpu import permissible_rank,calibrate_ranked_gate


def test_fixed_rank_and_exact_bound():
    assert permissible_rank(1000)==4
    assert 3*binom.cdf(3,1000,.01)<.05
    assert 3*binom.cdf(4,1000,.01)>.05
    assert permissible_rank(299,confidence_failure=.05)==1


def test_order_statistic_input_preserved_and_largest_outlier_does_not_set_gate():
    scores=np.arange(1000,dtype=float)
    saved=scores.copy()
    gate=calibrate_ranked_gate(scores)
    assert gate.score==996 and gate.rank_from_largest==4
    np.testing.assert_array_equal(scores,saved)
    scores[-1]=1e9
    assert calibrate_ranked_gate(scores).score==gate.score
    assert calibrate_ranked_gate(scores[::-1]).score==gate.score


def test_tied_scores_remain_a_strict_gate():
    gate=calibrate_ranked_gate(np.ones(1000))
    assert gate.score==1
    assert not 1>gate.score


@pytest.mark.parametrize('scores',[[],[1.]*100,np.ones((1000,1)),np.full(1000,np.nan)])
def test_invalid_or_insufficient_scores(scores):
    with pytest.raises(ValueError):calibrate_ranked_gate(scores)


@pytest.mark.parametrize('alpha,delta',[(0,.05),(.01,1),(.01,np.nan)])
def test_invalid_budgets(alpha,delta):
    with pytest.raises(ValueError):permissible_rank(1000,alpha,delta)
