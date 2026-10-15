"""Reference lookup reuse preserves row order, missing support and fresh calls."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_move_selection import reference  # noqa: E402
from benchmark_dd_layout import epoch_type  # noqa: E402

owner = reference(HERE / "rbpf_fgo.py")
EpochProblem = epoch_type(HERE / "rbpf_fgo.py")
G, OTHER = (0, 0), (1, 1)
PAIRS = [(G, 1, 2), (G, 1, 3), (G, 1, 4),
         (OTHER, 8, 9), (OTHER, 8, 10), (G, 2, 1), (G, 2, 3)]


def problem(pairs, prepared=False):
    prob = SimpleNamespace(pairs=pairs)
    if prepared:
        EpochProblem.set_pairs(prob, pairs)
    return prob


def original(H, prob):
    out = np.full(len(prob.pairs), np.nan)
    for pi, (g, ref, j) in enumerate(prob.pairs):
        zmap = H.get(g)
        if zmap is None:
            continue
        zr, zj = zmap.get(ref), zmap.get(j)
        if zr is not None and zj is not None:
            out[pi] = zr-zj
    return out


@pytest.mark.parametrize("H", [{}, {G: {}}, {G: None}, {G: {2: 5}},
                               {G: {1: 0, 2: -2, 3: 7, 4: 10}},
                               {G: {1: None, 2: 4}, OTHER: {8: 0, 9: 3}},
                               {G: {1: 0, 2: -2}, OTHER: {8: 20, 9: 15, 10: 7}}])
@pytest.mark.parametrize("pairs", [[], PAIRS, list(reversed(PAIRS)),
                                   PAIRS[::2]+PAIRS[1::2], PAIRS+PAIRS[:2],
                                   [(G, 1, 2), (G, 2, 1), (G, 1, 3), (G, 1, 1)]])
@pytest.mark.parametrize("prepared", [False, True])
def test_exact_bits_for_arbitrary_order_references_and_missing_support(H, pairs, prepared):
    prob = problem(pairs, prepared)
    saved_H, saved_pairs = deepcopy(H), deepcopy(pairs)
    result = owner._pair_dd(H, prob)
    assert result.dtype == np.float64 and result.shape == (len(pairs),)
    assert result.tobytes() == original(H, prob).tobytes()
    assert H == saved_H and pairs == saved_pairs


@pytest.mark.parametrize("values", [[0, 2**60+1, -(2**60)],
                                     [np.int64(0), np.int64(17), np.int64(-5)],
                                     [np.nan, 2, np.inf], [np.inf, np.inf, -np.inf]])
@pytest.mark.parametrize("prepared", [False, True])
def test_original_numeric_arithmetic_and_nonfinite_results(values, prepared):
    H = {G: dict(zip([1, 2, 3], values))}
    prob = problem([(G, 1, 2), (G, 1, 3), (G, 2, 3)], prepared)
    with np.errstate(invalid="ignore"):
        assert owner._pair_dd(H, prob).tobytes() == original(H, prob).tobytes()


@pytest.mark.parametrize("prepared", [False, True])
def test_same_objects_are_recomputed_after_hypothesis_and_pair_changes(prepared):
    H = {G: {1: 0, 2: -2, 3: 7}, OTHER: {8: 20, 9: 15}}
    prob = problem(PAIRS.copy(), prepared)
    saved = []
    for epoch in range(4):
        if epoch == 1:
            H[G][1] = 11
        elif epoch == 2:
            pairs = [(G, 2, 1), (OTHER, 9, 8), (G, 2, 3)]
            if prepared:
                EpochProblem.set_pairs(prob, pairs)
            else:
                prob.pairs[:] = pairs
        elif epoch == 3:
            del H[OTHER]
        result = owner._pair_dd(H, prob)
        assert result.tobytes() == original(H, prob).tobytes()
        assert all(old.tobytes() == bits for old, bits in saved)
        assert all(not np.shares_memory(result, old) for old, _ in saved)
        saved.append((result, result.tobytes()))


class Counted(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)


def test_group_and_reference_are_read_once_per_contiguous_run():
    zmap = Counted({1: 0, 2: -2, 3: 7, 4: 10})
    H = Counted({G: zmap})
    prob = problem([(G, 1, 2), (G, 1, 3), (G, 1, 4)], True)
    assert owner._pair_dd(H, prob).tolist() == [2, -7, -10]
    assert H.reads == [G] and zmap.reads == [1, 2, 3, 4]


def test_reference_changes_and_reappearing_groups_refresh_the_lookup():
    zmap, other = Counted({1: 0, 2: -2, 3: 7}), Counted({8: 20, 9: 15})
    H = Counted({G: zmap, OTHER: other})
    prob = problem([(G, 1, 2), (G, 2, 3), (OTHER, 8, 9), (G, 1, 3)], True)
    assert owner._pair_dd(H, prob).tolist() == [2, -9, 5, -7]
    assert H.reads == [G, G, OTHER, G]
    assert zmap.reads == [1, 2, 2, 3, 1, 3] and other.reads == [8, 9]


@pytest.mark.parametrize("prepared", [False, True])
def test_exception_leaves_no_state_for_the_next_call(prepared):
    H = {G: {1: 0, 2: "invalid"}}
    prob = problem([(G, 1, 2)], prepared)
    with pytest.raises(TypeError):
        owner._pair_dd(H, prob)
    H[G][2] = -2
    assert owner._pair_dd(H, prob).tolist() == [2]


def test_layout_freezes_mutable_input_rows_without_aliasing():
    pairs = [[G, 1, 2], [G, 1, 3]]
    prob = EpochProblem()
    prob.set_pairs(pairs)
    saved = prob._pair_runs
    pairs[0][1] = 3
    pairs.append([OTHER, 8, 9])
    assert prob.pairs == ((G, 1, 2), (G, 1, 3))
    assert prob._pair_runs is saved and saved[0] is prob.pairs
    with pytest.raises(TypeError):
        prob.pairs[0][1] = 3


def test_direct_replacement_uses_fallback_until_layout_is_prepared_again():
    prob = problem(PAIRS, True)
    saved_pairs = prob.pairs
    H = {G: {1: 0, 2: -2, 3: 7}, OTHER: {8: 20, 9: 15}}
    prob.pairs = [(OTHER, 9, 8), (G, 2, 1)]
    assert owner._pair_dd(H, prob).tobytes() == original(H, prob).tobytes()
    prob.pairs[0] = (G, 3, 1)
    assert owner._pair_dd(H, prob).tobytes() == original(H, prob).tobytes()
    prob.pairs = saved_pairs
    H[G][1] = 12
    assert owner._pair_dd(H, prob).tobytes() == original(H, prob).tobytes()
    EpochProblem.set_pairs(prob, list(reversed(PAIRS)))
    assert prob._pair_runs[0] is prob.pairs
    assert owner._pair_dd(H, prob).tobytes() == original(H, prob).tobytes()


def test_layout_contains_no_hypothesis_values_or_reusable_dd_arrays():
    prob = problem(PAIRS, True)
    saved = repr(prob._pair_runs)
    H = {G: {1: 0, 2: -2, 3: 7}}
    owner._pair_dd(H, prob)
    H[G][1] += 99
    assert owner._pair_dd(H, prob).tobytes() == original(H, prob).tobytes()
    assert repr(prob._pair_runs) == saved
