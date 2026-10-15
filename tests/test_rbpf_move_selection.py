"""Selection preserves scalar tie/NaN rules and exact integer hypotheses."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_move_selection import reference  # noqa: E402

LIVE = reference(HERE / "rbpf_fgo.py")


def old_extend(H, prob):
    dd = LIVE._pair_dd(H, prob)
    idxH = np.where(np.isfinite(dd))[0]
    idxC = [pi for pi, (g, ref, j) in enumerate(prob.pairs)
            if not np.isfinite(dd[pi]) and H.get(g, {}).get(ref) is not None]
    if len(idxH) == 0 or len(idxC) == 0:
        return None
    try:
        QHH = prob.Qb[np.ix_(idxH, idxH)]
        QCH = prob.Qb[np.ix_(idxC, idxH)]
        r = np.linalg.solve(QHH, prob.y[idxH] - dd[idxH])
        b_cond = prob.y[idxC] - QCH @ r
    except np.linalg.LinAlgError:
        return None
    out = {g: dict(z) for g, z in H.items() if z}
    for k, pi in enumerate(idxC):
        g, ref, j = prob.pairs[pi]
        z = out.setdefault(g, {})
        zr = z.get(ref)
        if zr is not None:
            z[j] = int(zr - int(np.rint(b_cond[k])))
    return out


def old_shrink(H, prob):
    dd = LIVE._pair_dd(H, prob)
    worst_pi, worst_r = -1, -1.
    for r in range(len(prob.rows_v0)):
        if not prob.rows_iscp[r]:
            continue
        pi = prob.rows_pair[r]
        if not np.isfinite(dd[pi]):
            continue
        value = abs(prob.rows_lam[r] * (prob.rows_bmeas[r]-dd[pi])) / max(prob.rows_sig[r], 1e-6)
        if value > worst_r:
            worst_r, worst_pi = value, pi
    if worst_pi < 0:
        return None
    g, _, j = prob.pairs[worst_pi]
    out = {g: dict(z) for g, z in H.items() if z}
    z = out.get(g)
    if z is None or j not in z:
        return None
    z.pop(j, None)
    if len(z) < 2:
        out.pop(g, None)
    return out or None


def problem():
    return SimpleNamespace(
        pairs=[((0, 0), 1, 3), ((0, 0), 1, 4), ((1, 0), 6, 7)],
        y=np.array([2., 4.5, -.5]), Qb=np.array([[2., .2, .1], [.2, 3., .3], [.1, .3, 4.]]),
        rows_v0=np.zeros(5), rows_iscp=np.array([False, True, True, True, False]),
        rows_pair=np.array([999, 1, 0, 2, -999]), rows_lam=np.ones(5),
        rows_bmeas=np.array([0., 6., 4., 1., 0.]), rows_sig=np.ones(5))


def check(H, prob, method, oracle):
    original = deepcopy(H)
    arrays = {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    with np.errstate(invalid="ignore", over="ignore"):
        expected, observed = oracle(H, prob), method(H, prob)
    assert observed == expected
    if observed is not None:
        assert list(observed) == list(expected)
        for g in observed:
            assert list(observed[g].items()) == list(expected[g].items())
            assert observed[g] is not H.get(g)
    assert H == original
    assert arrays == {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    return observed


@pytest.mark.parametrize("H", [{}, {(0, 0): {}}, {(0, 0): {1: 0}},
                               {(0, 0): {1: 0, 3: -2}}, {(0, 0): {4: 5, 3: 7, 1: 9}},
                               {(0, 0): {1: 0, 3: -2, 4: -4}, (1, 0): {6: 0, 7: 1}}])
@pytest.mark.parametrize("singular", [False, True])
def test_extension_support_and_singular_covariance(H, singular):
    prob = problem()
    if singular:
        prob.Qb[:] = 0
    check(H, prob, LIVE._extend, old_extend)


@pytest.mark.parametrize("rounding", [-2.5, -1.5, -.5, .5, 1.5, 2.5])
def test_extension_keeps_ties_to_even_and_gauge(rounding):
    prob = problem()
    prob.Qb = np.eye(3)
    prob.y[1] = rounding
    result = check({(0, 0): {1: 29, 3: 27}}, prob, LIVE._extend, old_extend)
    assert result[(0, 0)][4] == 29 - int(np.rint(rounding))


@pytest.mark.parametrize("value", [0., -0., 1e-9, -2., np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["rows_sig", "rows_lam", "rows_bmeas"])
def test_release_nonfinite_rows_and_sigma_floor(field, value):
    prob = problem()
    getattr(prob, field)[1] = value
    check({(0, 0): {1: 0, 3: -2, 4: -4}, (1, 0): {6: 0, 7: 1}},
          prob, LIVE._shrink_worst, old_shrink)


@pytest.mark.parametrize("value", [2., np.inf, np.nan])
def test_release_first_equal_row_and_all_nan(value):
    prob = problem()
    H = {(0, 0): {1: 0, 3: -2, 4: -4}}
    prob.rows_bmeas[1:4] = [4 + value, 2 + value, value]
    result = check(H, prob, LIVE._shrink_worst, old_shrink)
    assert result is None if np.isnan(value) else result == {(0, 0): {1: 0, 3: -2}}


def test_release_floor_stays_double_precision_with_float32_sigmas():
    prob = problem()
    prob.rows_sig = np.array([1., 0., 2e-6, 1., 1.], dtype=np.float32)
    prob.rows_bmeas[1:3] = 6.
    H = {(0, 0): {1: 0, 3: -2, 4: -4}}
    result = check(H, prob, LIVE._shrink_worst, old_shrink)
    assert result == {(0, 0): {1: 0, 4: -4}}


@pytest.mark.parametrize("case", ["empty", "no_cp", "no_held", "drop_group", "empty_rows", "duplicate"])
def test_release_support_cases(case):
    prob = problem()
    H = {(0, 0): {1: 0, 3: -2}}
    if case == "empty":
        H = {}
    elif case == "no_cp":
        prob.rows_iscp[:] = False
    elif case == "no_held":
        H = {(0, 0): {1: 0}}
    elif case == "empty_rows":
        for name, val in list(vars(prob).items()):
            if name.startswith("rows_"):
                setattr(prob, name, val[:0])
    elif case == "duplicate":
        prob.rows_pair[1:4] = 0
    check(H, prob, LIVE._shrink_worst, old_shrink)


@pytest.mark.parametrize("seed", range(12))
def test_varied_covariance_and_partial_hypotheses(seed):
    rng = np.random.default_rng(seed)
    prob = problem()
    A = rng.normal(size=(3, 3))
    prob.Qb = A @ A.T + np.eye(3)
    prob.y = rng.normal(size=3) * 30
    prob.rows_sig = rng.uniform(.001, 2., 5)
    prob.rows_bmeas = rng.normal(size=5)
    H = {(0, 0): {1: 100, 3: int(rng.integers(-10, 10))}}
    if seed % 2:
        H[(1, 0)] = {6: -20, 7: 10}
    if seed % 3:
        H[(0, 0)][4] = int(rng.integers(-10, 10))
    check(H, prob, LIVE._extend, old_extend)
    check(H, prob, LIVE._shrink_worst, old_shrink)
