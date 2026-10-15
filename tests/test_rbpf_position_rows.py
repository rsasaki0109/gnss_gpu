"""Position-row batching preserves dtype promotion, groups and solve outputs."""

from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_position_rows import evaluator, exact  # noqa: E402
from tests.test_rbpf_scalar_rows import problem  # noqa: E402


def scalar(prob, dd, sb):
    values, scales, jacobians, group_ids, groups = [], [], [], [], {}
    for r in range(len(prob.rows_v0)):
        if prob.rows_iscp[r]:
            z = dd[prob.rows_pair[r]]
            if not np.isfinite(z):
                continue
            values.append(prob.rows_lam[r]*(prob.rows_bmeas[r]-z))
            scales.append(prob.rows_sig[r]/sb)
            g = prob.pairs[prob.rows_pair[r]][0]
            group_ids.append(groups.setdefault(g, len(groups)))
        else:
            values.append(prob.rows_v0[r])
            scales.append(prob.rows_sig[r])
            group_ids.append(-1)
        jacobians.append(prob.rows_J[r])
    if not values:
        return None
    return np.asarray(values), np.asarray(scales), np.asarray(jacobians), np.asarray(group_ids), len(groups)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("dd", [[np.nan]*3, [-1., 0., 3.], [np.nan, 1., 3.], [np.inf, -np.inf, 0.]])
@pytest.mark.parametrize("beta", [.5, 1., 4.])
@pytest.mark.parametrize("reverse", [False, True])
def test_row_bytes_dtype_order_and_fresh_outputs(dtype, dd, beta, reverse):
    prob = problem(dtype, reverse)
    # The production fast path requires every numeric input to be FP64.
    prob.rows_J = prob.rows_J.astype(dtype)
    owner = evaluator(HERE/"rbpf_fgo.py")
    before = {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    result = owner._position_rows(prob, np.asarray(dd), np.sqrt(beta))
    assert exact(result, scalar(prob, np.asarray(dd), np.sqrt(beta)))
    assert before == {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    for array in result[:3]:
        assert all(not np.shares_memory(array, source) for source in [prob.rows_v0, prob.rows_sig, prob.rows_J])


@pytest.mark.parametrize("kind", ["empty", "code_only", "carrier_only", "no_held", "nonfinite", "mixed"])
def test_sparse_empty_and_nonfinite_rows(kind):
    prob = problem()
    dd = np.array([np.nan, 1., 3.])
    if kind == "empty":
        for name in vars(prob):
            if name.startswith("rows_"):
                setattr(prob, name, getattr(prob, name)[:0])
    elif kind == "code_only":
        prob.rows_iscp[:] = False
        dd = np.array([])
    elif kind in {"carrier_only", "no_held"}:
        prob.rows_iscp[:] = True
        prob.rows_pair[:] = 0
        if kind == "no_held":
            dd[:] = np.nan
    elif kind == "nonfinite":
        prob.rows_v0[0] = np.nan
        prob.rows_bmeas[1] = np.inf
        prob.rows_sig[4] = np.inf
    elif kind == "mixed":
        prob.rows_lam = prob.rows_lam.astype(np.float32)
    owner = evaluator(HERE/"rbpf_fgo.py")
    assert exact(owner._position_rows(prob, dd, .5), scalar(prob, dd, .5))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("state_kind", ["none", "offset", "invalid"])
@pytest.mark.parametrize("huber", [0., 1.345])
def test_full_solve_matches_original_rows(dtype, state_kind, huber):
    prob = problem(dtype)
    owner = evaluator(HERE/"rbpf_fgo.py")
    from types import SimpleNamespace
    owner.cfg = SimpleNamespace(beta=.5, sig_group=.1, huber_c=huber, cluster_sig_floor=.01)
    state = None if state_kind == "none" else dict(pos=prob.pos+np.array([.1, .2, -.1]), P=prob.Ppos)
    if state_kind == "invalid":
        state['pos'][0] = np.nan
    owner.replay_state = state
    dd = np.array([-1., 0., 3.])
    observed = owner._ls_offset({}, prob, dd)
    owner._position_rows = lambda prob, dd, sb: scalar(prob, dd, sb)
    expected = owner._ls_offset({}, prob, dd)
    assert exact(observed, expected)


def test_same_objects_are_fresh_after_dd_observation_and_order_changes():
    prob, dd = problem(), np.array([-1., 0., 3.])
    owner = evaluator(HERE/"rbpf_fgo.py")
    first = owner._position_rows(prob, dd, 1.)
    bits = first[0].tobytes()
    dd[0] = 17.
    prob.rows_bmeas[1] = -21.
    second = owner._position_rows(prob, dd, 1.)
    assert exact(second, scalar(prob, dd, 1.)) and second[0].tobytes() != bits
    assert first[0].tobytes() == bits
    for name in vars(prob):
        if name.startswith('rows_'):
            setattr(prob, name, getattr(prob, name)[::-1].copy())
    third = owner._position_rows(prob, dd, 1.)
    assert exact(third, scalar(prob, dd, 1.))
