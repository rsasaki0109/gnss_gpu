"""All staged moves/identities and unchanged numerical PF/RNG behavior."""

from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_move_plan_checked import MovePlanStep  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.fixture(scope="module")
def deps():
    return json.loads((ROOT / "dist/rbpf_copy_work_tests_20261018.plan.json").read_text(encoding="utf8"))["extensions"]


@pytest.fixture(scope="module")
def owned_scorer(deps):
    with RegisterLUScorer(ROOT / deps["DLL"]) as owned:
        yield owned


def exact(a, b):
    result = comparison(a, b)
    assert result["discrete_equal"] and result["numeric_bits_equal"] and result["max_abs_error"] == 0, result


def make(inputs, deps, scorer=None, backend="scalar", kind="plan"):
    runtime, rtk, _ = inputs
    _bind_fixture_callbacks(rtk)
    cls = NativeFeedbackMassStep if kind == "original" else MovePlanStep
    options = {} if kind == "original" else dict(planned=kind == "plan")
    call = cls(runtime, scorer, backend, extension=ROOT / deps["CANONICAL_EXT"],
        parent_extension=ROOT / deps["PARENT_EXT"], mass_extension=ROOT / deps["MASS_EXT"], **options)
    runtime.step = call
    return call


def check_metadata(original, actual, expect_plan=True):
    """Only three work diagnostics change; verify their accounting explicitly."""
    left, right = deepcopy(original), deepcopy(actual)
    plan = right.pop("move_plan")
    if not expect_plan:
        assert plan["fallback"] == 1 and plan["planned"] == 0
        exact(left, right)
        return plan
    assert plan["planned"] == 1 and plan["fallback"] == 0
    assert plan["particles"] == left["particles"] == right["particles"]
    assert plan["logical_registrations"] == left["requested_moves"] == right["requested_moves"]
    assert plan["executed_registrations"] + plan["avoided_registrations"] == plan["logical_registrations"]
    assert plan["prefix_plans"] + plan["prefix_replays"] == plan["particles"]
    lm, rm = left.pop("move_key_memo"), right.pop("move_key_memo")
    assert lm["canonical_requests"] == plan["logical_registrations"]
    assert rm["canonical_requests"] == plan["executed_registrations"]
    for m in (lm, rm):
        assert m["canonical_requests"] == m["canonical_source_calls"] + m["canonical_identity_hits"] + m["canonical_empty_hits"]
        assert m["identity_entries"] == m["canonical_source_calls"] and m["canonical_bypass"] == 0
        assert not m["DD_reuse_enabled"] and all(m[k] == 0 for k in m if k.startswith("DD_") and k != "DD_reuse_enabled")
    lc, rc = left.pop("native_canonical"), right.pop("native_canonical")
    assert lc["fallback_calls"] == rc["fallback_calls"] == 0
    assert lc["native_calls"] - rc["native_calls"] == lm["canonical_source_calls"] - rm["canonical_source_calls"] >= 0
    lr, rr = left.pop("move_reuse"), right.pop("move_reuse")
    for name in ("extend", "shrink"):
        assert lr["calls"][name] == rr["calls"][name]
        assert lr["requests"][name] == left["held_particles"]
        assert rr["requests"][name] == rr["calls"][name]
        assert lr["hits"][name] == lr["requests"][name] - lr["calls"][name]
        assert rr["hits"][name] == rr["uncacheable"][name] == lr["uncacheable"][name] == 0
    assert lr["extend_entries"] == rr["extend_entries"] and lr["shrink_entries"] == rr["shrink_entries"]
    assert lr["identity_entries"] >= rr["identity_entries"]
    exact(left, right)
    return plan


@pytest.mark.parametrize("backend", ("scalar", "native_cpu", "gpu"))
@pytest.mark.parametrize("seed", (20260710, 20260713))
@pytest.mark.parametrize("cluster", (False, True))
def test_four_full_epochs_exact(deps, owned_scorer, backend, seed, cluster):
    original = make_feedback_case(64, 6, seed=seed, cluster=cluster, shadow=True, spawn=True)
    inputs = [deepcopy(original) for _ in range(3)]
    calls = [make(x, deps, owned_scorer, backend, k) for x, k in zip(inputs, ("original", "off", "plan"))]
    enabled, threshold = gc.isenabled(), gc.get_threshold()
    for epoch in range(4):
        values = []
        for x, call in zip(inputs, calls):
            x[1].epoch2 = epoch
            out = call(x[1], x[2])
            values.append((state(*x[:2], out), deepcopy(call.last)))
        exact(values[0][0], values[1][0])
        exact(values[0][0], values[2][0])
        check_metadata(values[0][1], values[1][1], False)
        stats = check_metadata(values[0][1], values[2][1])
        assert stats["avoided_registrations"] > 0
    assert gc.isenabled() == enabled and gc.get_threshold() == threshold
    assert all(c._view is c._oracle is None for c in calls)


def _capture_stage(call, records):
    engine = call.staging_engine
    original = engine.stage
    def observed(*args):
        held = original(*args)
        context, runtime, cfg, prob, cons, cand, spawn, cache, keys, hypotheses, lookup, pending, staged = args
        source_particles = runtime.runtime._actual.particles
        for p, moves in zip(source_particles, staged):
            for _, _, h, tag in moves:
                if tag in ("survive", "stay"):
                    assert h is p.H
        release = [h for moves in staged for _, _, h, tag in moves if tag == "release"]
        assert len({id(h) for h in release}) == len(release)
        assert len({id(m) for m in staged}) == len(staged)
        snapshot = (deepcopy(keys), deepcopy(hypotheses), deepcopy(lookup), deepcopy(pending), deepcopy(staged), held)
        records.append(snapshot)
        return held
    engine.stage = observed


@pytest.mark.parametrize("variant", ("ordered", "empty_groups", "huge", "aliases"))
def test_first_key_pending_record_order_and_particle_ownership(deps, variant):
    original = make_feedback_case(16, 4)
    rt = original[0]
    if variant == "ordered":
        for p in rt.particles[::2]:
            p.H = dict(reversed([(g, dict(reversed(list(z.items())))) for g, z in p.H.items()]))
    elif variant == "empty_groups":
        for p in rt.particles[::2]:
            p.H[(99, 0)] = {}
    elif variant == "huge":
        for p in rt.particles:
            p.H = {g: {s: z + (1 << 200) for s, z in zs.items()} for g, zs in p.H.items()}
    else:
        rt.particles[1].H = rt.particles[0].H
    left, right = deepcopy(original), deepcopy(original)
    calls = [make(left, deps, kind="off"), make(right, deps)]
    records = [[], []]
    for call, saved, x in zip(calls, records, (left, right)):
        _capture_stage(call, saved)
        call(x[1], x[2])
    exact(records[0], records[1])
    assert calls[1].last["move_plan"]["planned"] == 1
    assert calls[1].last["move_plan"]["avoided_registrations"] > 0


@pytest.mark.parametrize("kind", ("zero_prior", "custom_h", "subclass_array", "numpy_hook", "canonical_hook", "builtin_hook", "profile"))
def test_unsupported_input_runs_literal_stage_once(deps, monkeypatch, kind):
    inputs = make_feedback_case(16, 4)
    left, right = deepcopy(inputs), deepcopy(inputs)
    calls = [make(left, deps, kind="off"), make(right, deps)]
    if kind == "zero_prior":
        for x in (left, right):
            x[0].cfg.p_shrink = 0.0
    elif kind == "custom_h":
        class Custom(dict):
            pass
        for x in (left, right):
            x[0].particles[0].H = Custom(x[0].particles[0].H)
    elif kind == "subclass_array":
        class Array(np.ndarray):
            pass
        for x in (left, right):
            x[2]["problem"].y = x[2]["problem"].y.view(Array)
    elif kind == "numpy_hook":
        original = np.rint
        monkeypatch.setattr(np, "rint", lambda *a, **kw: original(*a, **kw))
    elif kind == "canonical_hook":
        for call in calls:
            original = call.engine.module.canonical
            monkeypatch.setattr(call.engine.module, "canonical", lambda h, f=original: f(h))
    elif kind == "builtin_hook":
        for call in calls:
            call._source_globals["sorted"] = lambda *a, **kw: sorted(*a, **kw)
    def profile(frame, event, arg):
        pass
    saved_profile = sys.getprofile()
    if kind == "profile":
        sys.setprofile(profile)
    try:
        values = []
        for x, call in zip((left, right), calls):
            out = call(x[1], x[2])
            values.append(state(*x[:2], out))
        exact(*values)
        lm, rm = deepcopy(calls[0].last), deepcopy(calls[1].last)
        ls, rs = lm.pop("move_plan"), rm.pop("move_plan")
        exact(lm, rm)
        assert ls["fallback"] == rs["fallback"] == 1 and rs["planned"] == 0
        if kind == "profile":
            assert sys.getprofile() is profile
    finally:
        if kind == "profile":
            sys.setprofile(saved_profile)


class BrokenRng:
    def __init__(self, mutate=False):
        self.calls = 0
        self.mutate = mutate
    def choice(self, n, p):
        self.calls += 1
        if self.mutate:
            p[:] = 0.0
            p[0] = 1.0
            return 0
        if self.calls == 3:
            raise ValueError("original rng error")
        return 0
    def random(self):
        return 0.25


@pytest.mark.parametrize("mutate", (False, True))
def test_original_rng_mutation_and_exception_order(deps, mutate):
    initial = make_feedback_case(16, 4, resample=False)
    inputs = [deepcopy(initial), deepcopy(initial)]
    calls = [make(inputs[0], deps, kind="off"), make(inputs[1], deps)]
    values = []
    for x, call in zip(inputs, calls):
        x[0].rng = BrokenRng(mutate)
        if mutate:
            out = call(x[1], x[2])
            assert out is not None
        else:
            with pytest.raises(ValueError, match="original rng error"):
                call(x[1], x[2])
        values.append(([(p.H, p.logw, p.last_move) for p in x[0].particles], x[0].rng.calls))
        assert call._view is call._oracle is None
    exact(*values)


def test_strict_and_skipped_step_keep_original_behavior(deps):
    for strict in (False, True):
        initial = make_feedback_case(8, 4)
        inputs = [deepcopy(initial), deepcopy(initial)]
        calls = [make(inputs[0], deps, kind="original"), make(inputs[1], deps)]
        if not strict:
            for x in inputs:
                x[2]["problem"].ok = False
                x[2]["problem"].reason = "test_skip"
        with np.errstate(all="raise") if strict else np.errstate():
            values, errors = [], []
            for x, call in zip(inputs, calls):
                out = None
                try:
                    out = call(x[1], x[2])
                    errors.append(None)
                except FloatingPointError as error:
                    errors.append((type(error).__name__, str(error)))
                values.append((state(*x[:2], out), call.last))
        exact(*values)
        assert errors[0] == errors[1]
        assert bool(errors[0]) == strict
        assert calls[1].staging_engine.stats["calls"] == 0


@pytest.mark.parametrize("hook", ("copy", "ordered_key"))
def test_custom_pure_helper_callbacks_are_not_suppressed(deps, monkeypatch, hook):
    initial = make_feedback_case(16, 4)
    inputs = [deepcopy(initial), deepcopy(initial)]
    calls = [make(inputs[0], deps, kind="off"), make(inputs[1], deps)]
    if hook == "ordered_key":
        import rbpf_move_reuse
        original = rbpf_move_reuse._ordered_integer_key
        def wrapped(h):
            return original(h)
        monkeypatch.setattr(rbpf_move_reuse, "_ordered_integer_key", wrapped)
    else:
        for call in calls:
            original = call._source_globals["_copy_H"]
            call._source_globals["_copy_H"] = lambda h, f=original: f(h)
    values = []
    for x, call in zip(inputs, calls):
        out = call(x[1], x[2])
        values.append((state(*x[:2], out), deepcopy(call.last)))
    exact(values[0][0], values[1][0])
    a, b = values[0][1], values[1][1]
    sa, sb = a.pop("move_plan"), b.pop("move_plan")
    assert sa["fallback"] == sb["fallback"] == 1 and sb["planned"] == 0
    exact(a, b)
