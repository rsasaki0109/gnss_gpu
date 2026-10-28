import pytest
from experiments.validate_cpu_bandlimited_multipath import paired_summary


def test_identical_methods_have_no_discordance():
    result = paired_summary([True, False], [True, False])
    assert result['paired_exact_p'] == 1
    assert result['difference'] == 0
    assert result['paired_bootstrap_pointwise_95'] == [0, 0]
    assert result['baseline_exact_95'][0] < .5 < result['baseline_exact_95'][1]


def test_paired_all_improvement_exact_probability_and_zero_success_uncertainty():
    result = paired_summary([False]*8, [True]*8)
    # Both extreme outcomes under eight fair discordance coin tosses.
    assert result['paired_exact_p'] == pytest.approx(2/256)
    assert result['gains'] == 8 and result['losses'] == 0
    assert result['baseline_exact_95'][1] > 0
    assert result['selected_exact_95'][0] < 1


def test_marginal_equality_does_not_erase_discordance():
    result = paired_summary([True, False]*4, [False, True]*4)
    assert result['gains'] == result['losses'] == 4
    assert result['paired_exact_p'] == 1
    assert result['paired_bootstrap_pointwise_95'][0] < 0
    assert result['paired_bootstrap_pointwise_95'][1] > 0


@pytest.mark.parametrize('a,b', [([], []), ([1], [1,0]), ([[1]], [[1]])])
def test_invalid_pairs(a, b):
    with pytest.raises(ValueError):
        paired_summary(a, b)
