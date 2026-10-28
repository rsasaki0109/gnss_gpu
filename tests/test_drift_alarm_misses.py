import numpy as np
from experiments.diagnose_cpu_drift_alarm_misses import missed_cases,diagnostic_grid


def test_selection_keeps_boundary_errors_not_unavailable_or_warned():
    rows=[dict(error_m=e,warning=w) for e,w in [(None,False),(9.99,False),(10,False),(-10,False),(12,True)]]
    assert missed_cases(rows)==rows[2:4]


def test_grid_uses_selected_estimate_and_retains_previous_candidates():
    row=dict(selected=dict(delay_chips=.9),fits=dict(single=dict(first_delay_chips=.8),two=dict(first_delay_chips=.9)),
             profile_alarm=dict(points=[dict(delay_chips=.65)]),true_delay_chips=-.5)
    grid=diagnostic_grid(row)
    assert np.all(np.diff(grid)>0) and np.max(grid)<=1 and np.min(grid)>=-1
    assert .65 in grid and .8 in grid and .9 in grid
    row['true_delay_chips']=.123
    np.testing.assert_array_equal(grid,diagnostic_grid(row))
