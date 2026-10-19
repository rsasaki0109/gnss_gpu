"""Ordered worst CP selection, conservative fallback and complete PF state."""

from collections import UserDict
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_shrink import NativeShrinkStep  # noqa: E402


@pytest.fixture(scope="module")
def extensions():
    names = dict(
        extension="GNSS_GPU_RBPF_CANONICAL_EXT",
        parent_extension="GNSS_GPU_RBPF_PARENT_EXT",
        mass_extension="GNSS_GPU_RBPF_MASS_EXT",
        shrink_extension="GNSS_GPU_RBPF_SHRINK_EXT",
    )
    if any(not os.environ.get(n) for n in names.values()):
        pytest.skip("Explicit key/parent/mass/shrink extensions required")
    return {k: os.environ[n] for k, n in names.items()}


@pytest.fixture(scope="module")
def scorer():
    dll = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    if not dll:
        pytest.skip("Explicit unchanged numerical DLL required")
    with BatchScorer(dll) as result:
        yield result


def setup(extensions, **options):
    runtime, rtk, ctx = make_feedback_case(**options)
    return runtime, rtk, ctx, NativeShrinkStep(runtime, **extensions)


def exact(a, b):
    result = comparison(a, b)
    assert (
        result["discrete_equal"]
        and result["numeric_bits_equal"]
        and result["position_bits_equal"]
    ), result
    assert result["max_abs_error"] == 0.0, result


def check(call, runtime, h, prob):
    expected = call.shrink_engine.original(runtime, h, prob)
    actual = call.shrink_engine.index(runtime, h, prob)
    assert actual == expected
    return actual


@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
def test_random_missing_support_and_order(extensions, seed, rows, groups):
    runtime, rtk, ctx, call = setup(extensions, seed=seed, rows=rows, groups=groups)
    rng, prob = np.random.default_rng(seed), ctx["problem"]
    for _ in range(50):
        h = deepcopy(runtime.particles[int(rng.integers(len(runtime.particles)))].H)
        for zm in h.values():
            for sat in list(zm):
                if rng.random() < 0.12:
                    del zm[sat]
        prob.rows_bmeas += rng.normal(0.0, 0.3, rows)
        check(call, runtime, h, prob)
    assert (
        call.shrink_engine.native_calls == 50 and call.shrink_engine.fallback_calls == 0
    )


def tiny(prob, hypothesis=None):
    prob.pairs = [((1, 0), 1, 2), ((1, 0), 1, 3), ((1, 0), 1, 4)]
    prob.rows_v0 = np.zeros(3)
    prob.rows_iscp = np.ones(3, dtype=bool)
    prob.rows_pair = np.arange(3, dtype=int)
    prob.rows_lam = np.ones(3)
    prob.rows_sig = np.ones(3)
    prob.rows_bmeas = np.ones(3)
    return (
        {(1, 0): {1: 2**16000, 2: 2**16000, 3: 2**16000, 4: 2**16000}}
        if hypothesis is None
        else hypothesis
    )


@pytest.mark.parametrize(
    "sigma", [-1.0, -0.0, 0.0, 1e-7, 1e-6, np.nextafter(1e-6, np.inf), 1.0, 1e100]
)
def test_first_tie_then_one_ulp_strict_max_and_huge_gauge(extensions, sigma):
    runtime, rtk, ctx, call = setup(extensions)
    p = ctx["problem"]
    h = tiny(p)
    p.rows_sig[:] = sigma
    assert check(call, runtime, h, p) == 0
    p.rows_bmeas[1] = np.nextafter(1.0, np.inf)
    assert check(call, runtime, h, p) == 1
    p.rows_bmeas[0] = p.rows_bmeas[1]
    assert check(call, runtime, h, p) == 0
    p.rows_bmeas[:] = 0.0
    assert check(call, runtime, h, p) == 0
    assert call.shrink_engine.native_calls == 4


@pytest.mark.parametrize("difference", [2**53 + 1, 2**100, -(2**100), 10**99])
def test_py_long_difference_before_binary64_conversion(extensions, difference):
    runtime, rtk, ctx, call = setup(extensions)
    p = ctx["problem"]
    h = tiny(p)
    h[(1, 0)][3] -= difference
    assert check(call, runtime, h, p) == 1
    assert call.shrink_engine.native_calls == 1


@pytest.mark.parametrize(
    "kind",
    [
        "empty_h",
        "missing_ref",
        "missing_j",
        "no_cp",
        "empty_rows",
        "empty_pairs",
        "noncp_nonfinite",
    ],
)
def test_absent_support_and_noncp_values(extensions, kind):
    runtime, rtk, ctx, call = setup(extensions)
    p = ctx["problem"]
    h = tiny(p)
    if kind == "empty_h":
        h = {}
    elif kind == "missing_ref":
        del h[(1, 0)][1]
    elif kind == "missing_j":
        h = {(1, 0): {1: 0}}
    elif kind == "no_cp":
        p.rows_iscp[:] = False
    elif kind == "empty_pairs":
        p.pairs = []
        p.rows_iscp[:] = False
    elif kind == "empty_rows":
        for name in (
            "rows_v0",
            "rows_iscp",
            "rows_pair",
            "rows_lam",
            "rows_bmeas",
            "rows_sig",
        ):
            setattr(p, name, getattr(p, name)[:0])
    else:
        p.rows_iscp[1] = False
        p.rows_bmeas[1] = np.nan
        p.rows_sig[1] = np.inf
        p.rows_v0[:] = np.nan
    check(call, runtime, h, p)
    assert call.shrink_engine.native_calls == 1


@pytest.mark.parametrize(
    "kind",
    [
        "mapping",
        "subclass",
        "int_subclass",
        "numpy_integer",
        "negative_index",
        "strided",
        "unaligned",
        "float32",
        "nan",
        "inf",
        "huge_row",
        "huge_dd",
        "int32",
        "readonly",
    ],
)
def test_native_safety_domain_and_original_fallback(extensions, kind):
    runtime, rtk, ctx, call = setup(extensions)
    p = ctx["problem"]
    h = tiny(p)

    class Dict(dict):
        pass

    class Int(int):
        pass

    if kind == "mapping":
        h = UserDict(h)
    elif kind == "subclass":
        h = Dict(h)
    elif kind == "int_subclass":
        h[(1, 0)][2] = Int(2**16000)
    elif kind == "numpy_integer":
        h[(1, 0)] = {1: 0, 2: np.int64(0), 3: 0, 4: 0}
    elif kind == "negative_index":
        p.rows_pair[0] = -1
    elif kind == "strided":
        p.rows_lam = np.ones(6)[::2]
    elif kind == "unaligned":
        p.rows_lam = np.ndarray((3,), dtype="float64", buffer=bytearray(25), offset=1)
        p.rows_lam[:] = 1.0
    elif kind == "float32":
        p.rows_lam = p.rows_lam.astype("float32")
    elif kind == "nan":
        p.rows_bmeas[1] = np.nan
    elif kind == "inf":
        p.rows_bmeas[1] = np.inf
    elif kind == "huge_row":
        p.rows_lam[1] = 1e200
    elif kind == "huge_dd":
        h[(1, 0)][2] -= 2**400
    elif kind == "int32":
        p.rows_pair = p.rows_pair.astype("int32")
    else:
        for name in (
            "rows_v0",
            "rows_iscp",
            "rows_pair",
            "rows_lam",
            "rows_bmeas",
            "rows_sig",
        ):
            getattr(p, name).flags.writeable = False
    check(call, runtime, h, p)
    native = kind in ("int32", "readonly")
    assert call.shrink_engine.native_calls == int(native)
    assert call.shrink_engine.fallback_calls == int(not native)


@pytest.mark.parametrize("kind", ["int_subclass", "numpy_integer"])
def test_unsupported_integer_with_overflow_preserves_original_exception(
    extensions, kind
):
    runtime, rtk, ctx, call = setup(extensions)
    p = ctx["problem"]
    h = tiny(p)

    class Int(int):
        pass

    if kind == "int_subclass":
        h[(1, 0)][2] = Int(0)
    else:
        h[(1, 0)][2] = np.int64(0)
        h[(1, 0)][1] = 0
    for fn in (call.shrink_engine.original, call.shrink_engine.index):
        with pytest.raises(OverflowError, match="int too large to convert to float"):
            fn(runtime, h, p)
    assert (
        call.shrink_engine.native_calls == 0 and call.shrink_engine.fallback_calls == 1
    )


def test_non_target_overflow_precedes_rows_and_custom_mapping_callbacks(extensions):
    runtime, rtk, ctx, call = setup(extensions)
    p = ctx["problem"]
    h = tiny(p)
    h[(1, 0)][4] = 0
    p.rows_iscp[:] = False
    for fn in (call.shrink_engine.original, call.shrink_engine.index):
        with pytest.raises(OverflowError, match="int too large to convert to float"):
            fn(runtime, h, p)
    events = []

    class Mapping(dict):
        def get(self, *args):
            events.append(args[0])
            return super().get(*args)

    h = Mapping(tiny(p))
    call.shrink_engine.original(runtime, h, p)
    expected = events.copy()
    events.clear()
    call.shrink_engine.index(runtime, h, p)
    assert events == expected


def test_custom_abs_binding_and_strict_policy_keep_original_events(extensions):
    runtime, rtk, ctx, call = setup(extensions)
    p = ctx["problem"]
    h = tiny(p)
    engine = call.shrink_engine
    events = []

    def abs_callback(x):
        events.append(float(x))
        return abs(x)

    engine.scope["abs"] = abs_callback
    check(call, runtime, h, p)
    assert len(events) == 6 and engine.fallback_calls == 1
    del engine.scope["abs"]
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        check(call, runtime, h, p)
    assert engine.fallback_calls == 2


def test_reference_counts_fresh_inputs_and_mutation(extensions):
    runtime, rtk, ctx, call = setup(extensions)
    p = ctx["problem"]
    h = tiny(p)
    objects = [
        h,
        p.pairs,
        p.rows_iscp,
        p.rows_pair,
        p.rows_lam,
        p.rows_bmeas,
        p.rows_sig,
    ]
    gc.collect()
    before = [sys.getrefcount(o) for o in objects]
    for _ in range(10000):
        assert call.shrink_engine.index(runtime, h, p) == 0
    gc.collect()
    assert [sys.getrefcount(o) for o in objects] == before
    p.rows_bmeas[2] = 2.0
    assert check(call, runtime, h, p) == 2
    del h[(1, 0)][4]
    assert check(call, runtime, h, p) == 0
    assert call.shrink_engine.native_calls == 10002


@pytest.mark.parametrize("mode", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(cluster=True),
        dict(shadow=True, spawn=True),
        dict(feedback=False, preload=False, resample=False, uniform=False, huber=False),
    ],
)
def test_complete_step_all_state_rng_graph_events(
    extensions, scorer, mode, seed, options
):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    oldargs = {k: v for k, v in extensions.items() if k != "shrink_extension"}
    old, new = (
        NativeFeedbackMassStep(a, scorer, mode, **oldargs),
        NativeShrinkStep(b, scorer, mode, **extensions),
    )
    a.step, b.step = old, new
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = a.step(ra, deepcopy(ca)), b.step(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        prior = {k: v for k, v in new.last.items() if k != "shrink"}
        exact(old.last, prior)
        assert (
            new.last["shrink"]["native_calls"] > 0
            and new.last["shrink"]["fallback_calls"] == 0
        )
    assert "_rbpf_worst_index" not in new._scope


def test_strict_whole_step_and_method_guards(extensions):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    new = NativeShrinkStep(b, **extensions)
    with np.errstate(under="raise"):
        for fn, rtk, ctx in ((a.step, ra, ca), (new, rb, cb)):
            with pytest.raises(FloatingPointError):
                fn(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    assert new.last == dict(original_step_fallback="numpy_error_policy")
    assert new.shrink_engine.native_calls == 0 and "_rbpf_worst_index" not in new._scope
    b._shrink_worst = lambda h, p: None
    with pytest.raises(ValueError, match="method changed"):
        new(rb, cb)
