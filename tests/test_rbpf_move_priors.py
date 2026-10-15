"""Available move classes share priors only within one proposal phase."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_rbpf_shared_moves import RUN, owner, scalar_moves


def configured(seed, layout):
    patterns = {"stay": [-1]*8, "survive_release": [0]*8, "extend": [1]*8,
                "shrink": [2]*8, "both": [3]*8, "mixed": [-1, 0, 1, 2, 3, 1, 2, 3]}
    hypotheses = [{} if k < 0 else {(0, 0): {1: k, 3: k-2}} for k in patterns[layout]]
    result = owner(seed, hypotheses)
    def structural(H, extend):
        k = H[(0, 0)][1]
        enabled = k in ([1, 3] if extend else [2, 3])
        if not enabled:
            return None
        copied = deepcopy(H)
        if extend:
            copied[(0, 0)][9] = 7
        else:
            copied[(0, 0)].pop(3, None)
        return copied
    result._extend = lambda H, prob: structural(H, True)
    result._shrink_worst = lambda H, prob: structural(H, False)
    result._move_pair = lambda H, prob: (structural(H, True), structural(H, False))
    return result


def config(kind):
    values = [0.5, 0.17, 0.03, 0.1, 0.1, 0.3, 0.2]
    if kind == "sparse":
        values = [0.5, 0., 0.3, 0.1, 0., 0., 0.]
    elif kind == "tiny":
        values = [1e-320, 3e-320, 2e-320, 4e-320, 1e-320, 2e-320, 1e-320]
    return SimpleNamespace(track=2, **dict(zip(
        ["p_survive", "p_extend", "p_shrink", "p_release", "p_adopt_cons", "p_adopt", "p_spawn"], values)))


def score(H):
    return -2.-sum(abs(z) for group in H.values() for z in group.values())/20


def compare(a, b, cfg, shared):
    base = {(0, 0): {1: 0, 3: -2}}
    consensus = base if shared else None
    candidates = [base, deepcopy(base), {(0, 0): {1: 1, 3: -1}}] if shared else []
    spawns = [{(0, 0): {1: 0, 3: -8}}, {(0, 0): {1: 0, 3: -9}}] if shared else []
    before = deepcopy((consensus, candidates, spawns))
    with np.errstate(divide="ignore"):
        scalar_moves(a, cfg, consensus, candidates, spawns, score)
        RUN(b, cfg, None, None, 17, consensus, candidates, spawns, score, -2., 0., -np.inf)
    assert [vars(p) for p in a.particles] == [vars(p) for p in b.particles]
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert (consensus, candidates, spawns) == before


@pytest.mark.parametrize("layout", ["stay", "survive_release", "extend", "shrink", "both", "mixed"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", ["normal", "sparse", "tiny"])
@pytest.mark.parametrize("seed", [0, 20260710])
def test_prior_availability_preserves_moves_weights_and_rng(layout, shared, kind, seed):
    compare(configured(seed, layout), configured(seed, layout), config(kind), shared)


def test_prior_configuration_changes_are_seen_next_epoch():
    a, b = configured(81, "mixed"), configured(81, "mixed")
    cfg = config("normal")
    compare(a, b, cfg, True)
    cfg.p_extend, cfg.p_shrink, cfg.p_adopt = .01, .8, .95
    compare(a, b, cfg, True)
