"""Bulk held rows preserve scalar arithmetic and lattice accumulation order."""

from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_scalar_rows import evaluator, exact_rows  # noqa: E402


def config(uniform=True, beta=0.5):
    return SimpleNamespace(marg_uniform=uniform, beta=beta, huber_c=1.345,
                           sig_group=0.1, mix_eps=0.01, mix_sigma_cp=0.5, mix_sigma_pr=10.0)


def problem(dtype=np.float64, reverse=False):
    prob = SimpleNamespace(
        rows_v0=np.array([.2, -9, -8, .1, -7, -6, -.2], dtype=dtype),
        rows_sig=np.array([.5, .003, .004, .6, .007, .01, .7], dtype=dtype),
        rows_iscp=np.array([0, 1, 1, 0, 1, 1, 0], dtype=bool),
        rows_pair=np.array([-1, 2, 0, -1, 2, 1, -1]),
        rows_lam=np.array([0, .19, .24, 0, .19, .3, 0], dtype=dtype),
        rows_bmeas=np.array([0, 3.2, -1.1, 0, 3.3, 1.4, 0], dtype=dtype),
        Qb=np.diag([.4, -.1, .6]), y=np.array([-1., 1., 3.]),
        pairs=[((0, 0), 1, 2), ((0, 0), 1, 3), ((1, 0), 8, 9)],
        rows_J=np.array([[1., 0, 0], [0, 1, .2], [.1, .3, 1], [1, 1, 0],
                         [.2, 1, .1], [.1, -.3, .5], [0, 0, 1]]),
        pos=np.array([1e6, 2e6, 3e6]), Ppos=np.diag([.2, .3, .4]))
    if reverse:
        for name in vars(prob):
            if name.startswith("rows_"):
                setattr(prob, name, getattr(prob, name)[::-1].copy())
    return prob


def scalar_rows(prob, dd, cfg):
    m = len(prob.rows_v0)
    v, sig = prob.rows_v0.copy(), prob.rows_sig.copy()
    iscp, pi, lam = prob.rows_iscp, prob.rows_pair, prob.rows_lam
    qdiag, sb = np.diag(prob.Qb), math.sqrt(cfg.beta)
    uniform, const_ll = np.zeros(m, dtype=bool), 0.0
    for r in range(m):
        if not iscp[r]:
            continue
        p, z = pi[r], dd[pi[r]]
        if np.isfinite(z):
            v[r] = lam[r]*(prob.rows_bmeas[r]-z)
            sig[r] = sig[r]/sb
        elif cfg.marg_uniform:
            uniform[r] = True
            const_ll += -math.log(max(lam[r], 1e-3))
        else:
            v[r] = lam[r]*(prob.rows_bmeas[r]-prob.y[p])
            sig[r] = math.sqrt(sig[r]**2+lam[r]**2*(max(qdiag[p], 0.0)+1.0/12.0))/sb
    return v, sig, uniform, const_ll


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("beta", [.5, 1., 4.])
@pytest.mark.parametrize("dd", [[np.nan]*3, [-1., 0., 3.], [np.nan, 1., 3.], [np.inf, -np.inf, 0.]])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_rows_scales_mask_and_constant(dtype, uniform, beta, dd, reverse):
    prob, cfg = problem(dtype, reverse), config(uniform, beta)
    owner = evaluator(HERE/"rbpf_fgo.py")
    owner.cfg = cfg
    saved = {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    dd = np.array(dd)
    observed = owner._likelihood_rows(prob, dd)
    assert exact_rows(observed, scalar_rows(prob, dd, cfg))
    assert saved == {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    assert not np.shares_memory(observed[0], prob.rows_v0)
    assert not np.shares_memory(observed[1], prob.rows_sig)


@pytest.mark.parametrize("kind", ["empty", "code_only", "floor", "nonfinite"])
@pytest.mark.parametrize("uniform", [False, True])
def test_empty_code_only_and_nonfinite_rows(kind, uniform):
    prob, cfg = problem(), config(uniform)
    if kind == "empty":
        for name in vars(prob):
            if name.startswith("rows_"):
                setattr(prob, name, getattr(prob, name)[:0])
    elif kind == "code_only":
        prob.rows_iscp[:] = False
    elif kind == "floor":
        prob.rows_lam[[1, 2, 4, 5]] = [0, -1., 1e-10, 1e-3]
    else:
        prob.rows_bmeas[2] = np.nan
        prob.rows_lam[4] = np.nan
        prob.rows_sig[5] = np.inf
    owner = evaluator(HERE/"rbpf_fgo.py")
    owner.cfg = cfg
    dd = np.array([np.nan, 1., 3.])
    with np.errstate(invalid="ignore"):
        assert exact_rows(owner._likelihood_rows(prob, dd), scalar_rows(prob, dd, cfg))


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("state_kind", ["none", "offset", "invalid"])
@pytest.mark.parametrize("H", [{}, {(0, 0): {1: 0, 2: 1}},
                               {(0, 0): {1: 0, 2: 1, 3: -1}, (1, 0): {8: 0, 9: -3}},
                               {(0, 0): {2: 1, 3: -1}}])
def test_likelihood_and_cache_exact_against_scalar_row_oracle(uniform, state_kind, H):
    old, new = evaluator(HERE/"rbpf_fgo.py"), evaluator(HERE/"rbpf_fgo.py")
    prob, cfg = problem(), config(uniform)
    old.cfg = new.cfg = cfg
    old._likelihood_rows = lambda p, dd: scalar_rows(p, dd, cfg)
    state = None if state_kind == "none" else dict(pos=prob.pos+np.array([.01, -.02, .03]), P=prob.Ppos*2)
    if state_kind == "invalid":
        state["pos"][0] = np.nan
    old.replay_state = new.replay_state = state
    a, b = {}, {}
    before = deepcopy(H)
    observed, expected = new._loglik(H, prob, a), old._loglik(H, prob, b)
    assert np.float64(observed).tobytes() == np.float64(expected).tobytes()
    assert a == b and H == before
    new._likelihood_rows = lambda *args: pytest.fail("cache hit recomputed rows")
    assert new._loglik(H, prob, a) == observed


def test_invalid_beta_preserves_original_error():
    owner = evaluator(HERE/"rbpf_fgo.py")
    owner.cfg = config(beta=-1)
    with pytest.raises(ValueError):
        owner._likelihood_rows(problem(), np.zeros(3))
