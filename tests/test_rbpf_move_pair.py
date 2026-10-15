"""Shared DDs remain read-only, local, and equivalent to scalar move rules."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_move_pair import paired_reference  # noqa: E402
from benchmark_move_selection import encode  # noqa: E402
from test_rbpf_move_selection import old_extend, old_shrink, problem  # noqa: E402


class Counting(type(paired_reference(HERE / "rbpf_fgo.py"))):
    def __init__(self):
        self.arrays = []

    def _pair_dd(self, H, prob):
        dd = super()._pair_dd(H, prob)
        dd.setflags(write=False)
        self.arrays.append(dd)
        return dd


@pytest.mark.parametrize("H", [{}, {(0, 0): {}}, {(0, 0): {1: 0}},
                               {(0, 0): {1: 0, 3: -2}},
                               {(0, 0): {1: 19, 3: 17, 4: 15}},
                               {(0, 0): {1: 0, 3: -2}, (1, 0): {6: 5, 7: 8}}])
@pytest.mark.parametrize("case", ["normal", "singular", "nonfinite", "no_cp"])
def test_one_readonly_dd_preserves_both_outputs_and_ownership(H, case):
    prob = problem()
    if case == "singular":
        prob.Qb[:] = 0
    elif case == "nonfinite":
        prob.rows_bmeas[1:3] = [np.nan, np.inf]
    elif case == "no_cp":
        prob.rows_iscp[:] = False
    before = deepcopy(H)
    arrays = {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    with np.errstate(invalid="ignore"):
        expected = old_extend(H, prob), old_shrink(H, prob)
        owner = Counting()
        observed = owner._move_pair(H, prob)
    assert len(owner.arrays) == 1
    assert [encode(v) for v in observed] == [encode(v) for v in expected]
    assert H == before
    assert arrays == {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    for result in observed:
        if result is not None:
            assert result is not H
            assert all(result.get(g) is not z for g, z in H.items())
    first, second = observed
    if first is not None and second is not None:
        assert all(first.get(g) is not z for g, z in second.items())


def test_prepared_dd_avoids_fallback_computation():
    owner = Counting()
    H = {(0, 0): {1: 0, 3: -2}}
    prob = problem()
    dd = owner._pair_dd(H, prob)
    expected = old_extend(H, prob), old_shrink(H, prob)
    assert (owner._extend(H, prob, dd), owner._shrink_worst(H, prob, dd)) == expected
    assert len(owner.arrays) == 1


def test_recompute_after_hypothesis_and_epoch_changes():
    owner = Counting()
    H = {(0, 0): {1: 0, 3: -2}}
    prob = problem()
    old_dds = []
    for epoch in range(4):
        H[(0, 0)][3] -= epoch
        prob.y[1] += epoch
        prob.rows_bmeas[2] += epoch
        if epoch == 2:
            prob.pairs[0] = ((0, 0), 3, 1)
        expected = old_extend(H, prob), old_shrink(H, prob)
        assert owner._move_pair(H, prob) == expected
        assert all(dd.tobytes() == saved for dd, saved in old_dds)
        old_dds.append((owner.arrays[-1], owner.arrays[-1].tobytes()))
    assert len(owner.arrays) == 4
    assert len({id(dd) for dd in owner.arrays}) == 4


def test_failed_extension_does_not_leave_reusable_state():
    owner = Counting()
    H = {(0, 0): {1: 0, 3: -2}}
    prob = problem()
    prob.y[1] = np.nan
    with pytest.raises(ValueError):
        owner._move_pair(H, prob)
    prob.y[1] = 4.5
    assert owner._move_pair(H, prob) == (old_extend(H, prob), old_shrink(H, prob))
    assert len(owner.arrays) == 2


def test_direct_legacy_helpers_still_compute_their_own_dds():
    owner = Counting()
    H = {(0, 0): {1: 0, 3: -2}}
    prob = problem()
    assert owner._extend(H, prob) == old_extend(H, prob)
    assert owner._shrink_worst(H, prob) == old_shrink(H, prob)
    assert len(owner.arrays) == 2
