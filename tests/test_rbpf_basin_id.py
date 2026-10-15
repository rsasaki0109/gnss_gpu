"""Canonical identity retains gauge/order semantics without persistent state."""

from copy import deepcopy
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_basin_id import canonical  # noqa: E402

current = canonical(HERE / "basin_id_candidate.py")


def original(H):
    out = []
    for g in sorted(H.keys()):
        zmap = H[g]
        if not zmap:
            continue
        base = zmap[min(zmap.keys())]
        out.append((g, tuple(sorted((s, z-base) for s, z in zmap.items()))))
    return tuple(out)


@pytest.mark.parametrize("H", [{}, {(0, 0): {}}, {(0, 0): {7: 91}},
                               {(2, 1): {9: -14, 1: -2, 4: 8}, (0, 0): {}},
                               {(0, 1): {2: 10**80, 1: -10**80}},
                               {(np.int64(1), 0): {np.int64(3): np.int64(-2),
                                                  np.int64(1): np.int64(4)}}])
def test_original_identity_hash_types_and_immutable_input(H):
    before = deepcopy(H)
    observed, expected = current(H), original(H)
    assert observed == expected
    assert hash(observed) == hash(expected)
    assert repr(observed) == repr(expected)
    assert H == before


@pytest.mark.parametrize("group_order", list(permutations([(0, 0), (0, 1), (2, 0)])))
@pytest.mark.parametrize("sat_order", list(permutations([1, 3, 9])))
def test_group_sat_order_and_independent_gauge_shifts(group_order, sat_order):
    H = {g: {s: (s*7-20)+(g[0]*101+g[1]*31) for s in sat_order} for g in group_order}
    shifted = {g: {s: z+987654321*(i+1) for s, z in zmap.items()}
               for i, (g, zmap) in enumerate(H.items())}
    assert current(H) == original(H) == current(shifted)
    assert [g for g, _ in current(H)] == sorted(group_order)


def test_mutating_same_hypothesis_never_reuses_an_identity():
    H = {(0, 0): {9: 2, 1: 0, 4: -5}}
    first = current(H)
    H[(0, 0)][9] += 1
    second = current(H)
    assert first != second == original(H)
    H[(0, 0)].pop(1)
    assert current(H) == original(H)
    H[(1, 1)] = {8: -20}
    assert current(H) == original(H)
    H[(0, 0)].clear()
    assert current(H) == (((1, 1), ((8, 0),)),)
    assert first == (((0, 0), ((1, 0), (4, -5), (9, 2))),)
