"""Bulk integer matching preserves scalar tests and ordered weight additions."""

import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_feedback_dd import outcome, pack  # noqa: E402
from tests.test_rbpf_feedback_dd import FIELDS, scalar_dd, setup  # noqa: E402


def live_mass():
    tree = ast.parse((HERE/"rbpf_fgo.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_feedback")
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "mass_dd")
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "bad_keys")
    wrapper = ast.parse("def run(self, prob, w, dd_all, committed, cfg):\n    pass").body[0]
    wrapper.body = method.body[start:end]+ast.parse("return held_pairs, group_held, group_bad").body
    namespace = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "mass", "exec"), namespace)
    return namespace["run"]


RUN = live_mass()


def original(owner, prob, w, dd_all, committed, cfg):
    held_pairs, group_held, group_bad = [], {}, {}
    for pi, (g, ref, j) in enumerate(prob.pairs):
        f = g[1]
        vr, vj = committed.get((int(ref), f)), committed.get((int(j), f))
        if vr is None or vj is None:
            continue
        z_held, mass = int(np.rint(vr-vj)), 0.0
        for i in range(len(owner.particles)):
            di = dd_all[i][pi]
            if np.isfinite(di) and int(di) == z_held:
                mass += float(w[i])
        held_pairs.append((g, ref, j, mass))
        group_held[g] = group_held.get(g, 0)+1
        if mass < cfg.fb_rel_mass:
            group_bad[g] = group_bad.get(g, 0)+1
    return held_pairs, group_held, group_bad


def inputs(dtype=np.float64):
    prob = SimpleNamespace(pairs=[((0, 0), 1, 2), ((0, 0), 1, 3)])
    dds = [np.array(v, dtype=dtype) for v in [[2.9, -4.9], [2., -4.], [np.nan, np.inf],
                                            [-np.inf, -4.], [-2., 4.], [0., -0.], [2.1, -4.1], [2., np.nan]]]
    return SimpleNamespace(particles=[None]*len(dds)), prob, np.ones(len(dds))/len(dds), dds, \
        {(1, 0): 0., (2, 0): -2., (3, 0): 4.}, SimpleNamespace(fb_rel_mass=.5)


def check(values):
    saved = pack((vars(values[1]), values[2], values[3], values[4]))
    assert pack(RUN(*values)) == pack(original(*values))
    assert saved == pack((vars(values[1]), values[2], values[3], values[4]))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("container", ["arrays", "lists", "extra", "ragged"])
@pytest.mark.parametrize("weights", ["normalized", "cancellation", "nonfinite", "negative_zero"])
def test_integer_matching_and_weight_order(dtype, container, weights):
    values = list(inputs(dtype))
    if weights == "cancellation":
        values[2] = np.array([2**53, 1., 0, -2**53, 0, 0, 1., 1e-300])
    elif weights == "nonfinite":
        values[2] = np.array([np.inf, -np.inf, np.nan, 0, 0, 0, 0, 0])
    elif weights == "negative_zero":
        values[2] = np.full(8, -0.)
    if container == "lists":
        values[3] = [a.tolist() for a in values[3]]
    elif container == "extra":
        values[3] = [np.r_[a, 5.] for a in values[3]]
    elif container == "ragged":
        values[3][0] = np.r_[values[3][0], 5.]
    check(values)


@pytest.mark.parametrize("kind", ["empty", "no_pairs", "uncommitted", "duplicate", "strided", "ndarray", "endian"])
def test_shapes_and_fallback_containers(kind):
    values = list(inputs())
    if kind == "empty":
        values[0].particles, values[2], values[3] = [], np.array([]), []
    elif kind == "no_pairs":
        values[1].pairs, values[3] = [], [a[:0] for a in values[3]]
    elif kind == "uncommitted":
        values[4] = {}
    elif kind == "duplicate":
        values[1].pairs[1] = values[1].pairs[0]
    elif kind == "strided":
        values[3] = [np.repeat(a, 2)[::2] for a in values[3]]
    elif kind == "ndarray":
        values[3] = np.array(values[3])
    else:
        values[3] = [a.astype(">f8") for a in values[3]]
    check(values)


@pytest.mark.parametrize("held", [-2**60, -2**53, -2**53+1, 0, 2**53-1, 2**53, 2**53+2, 2**60, .5, -1.5])
def test_exact_integer_boundary_and_rounding(held):
    values = list(inputs())
    values[4][(2, 0)] = -held
    for i, array in enumerate(values[3]):
        array[0] = float(held) if i % 2 else np.nextafter(float(held), np.inf)
    check(values)


def test_large_extended_precision_hold_uses_python_integer_comparison():
    values = list(inputs())
    values[4][(2, 0)] = -(np.longdouble(2)**60+1)
    for array in values[3]:
        array[0] = float(2**60)
    check(values)


@pytest.mark.parametrize("kind", ["normal", "mass", "carry", "unseen", "support", "ddpr",
    "dual", "witness", "commit_gate", "new_commit", "nonfinite_seed", "no_committed"])
@pytest.mark.parametrize("prepared", [False, True])
def test_complete_feedback_actions_against_scalar_fallback(kind, prepared):
    results = []
    for bulk in [False, True]:
        evaluation, rtk, prob, w, dd_map, kwargs, _ = setup(kind, prepared, False)
        owner = evaluation[0]
        dds = [scalar_dd(p.H, prob) for p in owner.particles]
        saved = pack(dds), pack([p.H for p in owner.particles])
        value = owner._feedback(rtk, prob, w, dd_map, [0, 1], **kwargs,
                               dd_all=dds if bulk else [a.tolist() for a in dds])
        results.append((outcome(owner, rtk, FIELDS, value), pack(rtk.events)))
        assert saved == (pack(dds), pack([p.H for p in owner.particles]))
    assert results[0] == results[1]


def test_each_call_sees_current_dd_and_weight_values():
    values = list(inputs())
    before = RUN(*values)
    values[3][0][0], values[2][0] = 99., .9
    after = RUN(*values)
    assert pack(before) != pack(after)
    check(deepcopy(values))
