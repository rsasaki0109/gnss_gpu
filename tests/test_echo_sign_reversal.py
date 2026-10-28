import numpy as np
from experiments.run_cpu_echo_sign_reversal import synthesize, tasklist
from experiments.run_cpu_accelerating_echo_pilot import synthesize as previous
from experiments.run_cpu_moving_echo_pilot import synthesize as moving


def test_positive_branch_exact_previous_and_negative_zero_exact():
    for rate in (0., -2., 2.):
        actual = synthesize((.7, rate, .5, -.3, 1370000))
        expected = previous((rate, .5, -.3, 1370000))
        for x, y in zip(actual, expected):
            np.testing.assert_array_equal(x, y)
    for x, y in zip(synthesize((-.7, 0., .5, -.3, 1370000)),
                    moving(('moving_code', -.7, .5, -.3, 1370000))):
        np.testing.assert_array_equal(x, y)


def test_all_six_cells_share_noise_and_nuisance():
    tasks = [t for t in tasklist() if t[-1]==1370000]
    assert len(tasks)==6 and len({t[-1] for t in tasklist()})==16
    b, s, truth = synthesize(tasks[0])
    for task in tasks[1:]:
        other, signal, other_truth = synthesize(task)
        assert other_truth==truth
        np.testing.assert_allclose(other-signal, b-s, rtol=1e-12, atol=1e-15)
        assert np.linalg.norm(signal-s)>0
