"""Nuisance groups retain first occurrence order and original scalar rounding."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_nuisance_columns import evaluator, exact  # noqa: E402
from tests.test_rbpf_scalar_rows import config, problem  # noqa: E402


def scalar_columns(prob, iscp, pi, sig):
    groups, ids = {}, []
    for row in range(len(sig)):
        if iscp[row]:
            group = prob.pairs[pi[row]][0]
            if group not in groups:
                groups[group] = len(groups)
            ids.append(groups[group])
        else:
            ids.append(-1)
    if not groups:
        return None, 0
    C = np.zeros((len(sig), len(groups)))
    for row, group in enumerate(ids):
        if group >= 0:
            C[row, group] = 1.0/sig[row]
    return C, len(groups)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("kind", ["all", "subset", "code", "empty", "strided"])
@pytest.mark.parametrize("reverse", [False, True])
def test_group_order_masks_and_scalar_promotion(dtype, kind, reverse):
    prob = problem(dtype, reverse)
    mask = np.ones(len(prob.rows_sig), dtype=bool)
    if kind == "subset":
        mask[[1, 4]] = False
    elif kind == "code":
        mask = ~prob.rows_iscp
    elif kind == "empty":
        mask[:] = False
    iscp, pi, sig = prob.rows_iscp[mask], prob.rows_pair[mask], prob.rows_sig[mask]
    if kind == "strided":
        iscp, pi, sig = iscp[::2], pi[::2], sig[::2]
    owner = evaluator(HERE/"rbpf_fgo.py")
    saved = [a.tobytes() for a in [iscp, pi, sig]]
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = owner._nuisance_columns(prob, iscp, pi, sig)
        assert exact(observed, scalar_columns(prob, iscp, pi, sig))
    assert saved == [a.tobytes() for a in [iscp, pi, sig]]
    if observed[0] is not None:
        assert observed[0].dtype == np.float64
        assert not np.shares_memory(observed[0], sig)
        with np.errstate(divide="ignore", invalid="ignore"):
            assert not np.shares_memory(observed[0], owner._nuisance_columns(prob, iscp, pi, sig)[0])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("value", [0., -0., np.inf, -np.inf, np.nan, 1e-30, -1e30])
def test_nonfinite_and_signed_scale_bytes(dtype, value):
    prob = problem(dtype)
    prob.rows_sig[prob.rows_iscp] = value
    owner = evaluator(HERE/"rbpf_fgo.py")
    with np.errstate(all="ignore"):
        assert exact(owner._nuisance_columns(prob, prob.rows_iscp, prob.rows_pair, prob.rows_sig),
                     scalar_columns(prob, prob.rows_iscp, prob.rows_pair, prob.rows_sig))


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("state_kind", ["none", "offset", "invalid"])
@pytest.mark.parametrize("H", [{}, {(0, 0): {1: 0, 2: 1}},
    {(0, 0): {1: 0, 2: 1, 3: -1}, (1, 0): {8: 0, 9: -3}}, {(0, 0): {2: 1, 3: -1}}])
def test_full_likelihood_cache_and_hypotheses(uniform, state_kind, H):
    old, new = evaluator(HERE/"rbpf_fgo.py"), evaluator(HERE/"rbpf_fgo.py")
    prob = problem()
    old.cfg = new.cfg = config(uniform)
    old._nuisance_columns = lambda p, cp, pi, sig: scalar_columns(p, cp, pi, sig)
    state = None if state_kind == "none" else dict(pos=prob.pos+np.array([.01, -.02, .03]), P=prob.Ppos*2)
    if state_kind == "invalid":
        state["pos"][0] = np.nan
    old.replay_state = new.replay_state = state
    a, b, before = {}, {}, deepcopy(H)
    observed, expected = new._loglik(H, prob, a), old._loglik(H, prob, b)
    assert exact(observed, expected) and a == b and H == before
    new._nuisance_columns = lambda *args: pytest.fail("cache hit rebuilt columns")
    assert exact(new._loglik(H, prob, a), observed)


def test_layout_is_local_to_each_call():
    owner = evaluator(HERE/"rbpf_fgo.py")
    prob = problem()
    first = owner._nuisance_columns(prob, prob.rows_iscp, prob.rows_pair, prob.rows_sig)
    prob.pairs[0] = ((4, 2), 1, 2)
    second = owner._nuisance_columns(prob, prob.rows_iscp, prob.rows_pair, prob.rows_sig)
    assert not exact(first, second)
    assert exact(second, scalar_columns(prob, prob.rows_iscp, prob.rows_pair, prob.rows_sig))
