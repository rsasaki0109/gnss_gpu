from types import SimpleNamespace
import numpy as np
import pytest
from gnss_gpu.drift_ambiguity_cpu import ambiguity_score,candidate_grid
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED,CHIP_RATE


def test_score_orders_ambiguity_and_is_scale_invariant():
    assert ambiguity_score(10,9,[],100)==0
    assert ambiguity_score(10,9,[9,11],100)==1
    assert ambiguity_score(10,9,[8],100)==1
    near=ambiguity_score(10,9,[9.01],100)
    far=ambiguity_score(10,9,[10],100)
    assert 0<far<near<1
    assert np.isclose(near,ambiguity_score(100,90,[90.1],100))
    with pytest.raises(ValueError):ambiguity_score(10,9,[float('nan')],100)


def test_grid_is_centered_on_chosen_model_and_excludes_near_points():
    single=SimpleNamespace(first_delay_chips=.1)
    two=SimpleNamespace(first_delay_chips=.2)
    for center in (.1,.2):
        grid=candidate_grid(center,single,two)
        assert np.all(np.abs(grid-center)*LIGHT_SPEED/CHIP_RATE>10)
        assert np.isclose(grid,center+.035).any()
    assert .1 in candidate_grid(.2,single,two)
    assert .2 in candidate_grid(.1,single,two)
