"""Identity score-index aliases preserve keys, callbacks, math and RNG."""

from copy import deepcopy
import gc
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_epoch_memo import _EpochMemo  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_score_index_memo import ScoreIndexMemo, ScoreIndexMemoStep  # noqa: E402
from rbpf_move_reuse import MoveReuseStep  # noqa: E402


@pytest.fixture(scope="module")
def extensions():
    names = dict(
        extension="GNSS_GPU_RBPF_CANONICAL_EXT",
        parent_extension="GNSS_GPU_RBPF_PARENT_EXT",
        mass_extension="GNSS_GPU_RBPF_MASS_EXT",
        index_extension="GNSS_GPU_RBPF_INDEX_EXT",
    )
    if any(not os.environ.get(n) for n in names.values()):
        pytest.skip(
            "Explicit unchanged extensions and new score-index extension required"
        )
    return {k: os.environ[v] for k, v in names.items()}


@pytest.fixture(scope="module")
def module(extensions):
    spec = importlib.util.spec_from_file_location(
        "_rbpf_score_index_probe", extensions["index_extension"]
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def scorer():
    dll = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    if not dll:
        pytest.skip("Explicit unchanged numerical DLL required")
    with BatchScorer(dll) as result:
        yield result


def exact(a, b):
    r = comparison(a, b)
    assert (
        r["discrete_equal"] and r["numeric_bits_equal"] and r["max_abs_error"] == 0.0
    ), r


def run_sequence(hypotheses, canonical, kind, module):
    memo = _EpochMemo(canonical, None, cache_dd=False)
    aliases = (
        None
        if kind == "original"
        else ScoreIndexMemo(memo.canonical)
        if kind == "python"
        else module.IndexMemo(memo.identities)
    )
    keys, unique, lookup = [], [], {}

    def original(key, h):
        if key not in lookup:
            lookup[key] = len(keys)
            keys.append(key)
            unique.append(h)
        return lookup[key]

    outputs = []
    for h in hypotheses:
        key = memo.canonical(h)
        outputs.append(
            original(key, h) if aliases is None else aliases.get(key, h, original)
        )
    return (
        outputs,
        keys,
        unique,
        memo.summary(),
        (None if aliases is None else aliases.summary()),
    )


@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
def test_random_first_seen_order_and_original_canonical_metadata(
    module, seed, rows, groups
):
    oracle, p, hs = make_case(16, rows=rows, groups=groups, seed=seed, duplicates=False)
    rng = np.random.default_rng(seed)
    values = [hs[int(i)] for i in rng.integers(0, len(hs), 1000)]
    a = run_sequence(values, oracle.basin_id, "original", module)
    b = run_sequence(values, oracle.basin_id, "python", module)
    c = run_sequence(values, oracle.basin_id, "native", module)
    exact(a[:4], b[:4])
    exact(a[:4], c[:4])
    exact(b[4], c[4])
    assert (
        c[4]["requests"] == 1000
        and c[4]["identity_hits"] > 900
        and not c[4]["probability_cache"]
    )


def test_gauge_equivalent_distinct_objects_reordered_and_huge_integers(module):
    oracle, p, hs = make_case(4, duplicates=False)
    h = {(1, 0): {3: 2**16000, 1: 2**16000 + 2**80, 2: 2**16000 - 1}}
    equiv = {(1, 0): {2: -1, 3: 0, 1: 2**80}}
    clone = deepcopy(h)
    values = [{}, h, equiv, clone, h, {}, equiv, clone] * 20
    a = run_sequence(values, oracle.basin_id, "original", module)
    b = run_sequence(values, oracle.basin_id, "native", module)
    exact(a[:4], b[:4])
    assert len(a[1]) == 2 and all(v in (0, 1) for v in a[0])


@pytest.mark.parametrize(
    "kind", ["object", "tuple_subclass", "nested_integer_subclass", "exception"]
)
def test_custom_hash_callback_and_error_order_match_original(module, kind):
    def trial(mode):
        events = []

        class Key:
            def __hash__(self):
                events.append("hash")
                if kind == "exception":
                    raise ValueError("original hash error")
                return 12

            def __eq__(self, other):
                events.append("eq")
                return self is other

        class Tuple(tuple):
            def __hash__(self):
                events.append("tuplehash")
                return super().__hash__()

        class Int(int):
            def __hash__(self):
                events.append("inthash")
                return super().__hash__()

        special = (
            Tuple((1,))
            if kind == "tuple_subclass"
            else (Int(1),)
            if kind == "nested_integer_subclass"
            else Key()
        )
        normal, custom = {(1, 0): {1: 0, 2: 1}}, {(2, 0): {1: 0, 2: 1}}

        def canonical(h):
            events.append(("canonical", next(iter(h))))
            return (1,) if h is normal else special

        memo = _EpochMemo(canonical, None, cache_dd=False)
        aliases = None if mode == "original" else module.IndexMemo(memo.identities)
        lookup = {}

        def source(key, h):
            if key not in lookup:
                lookup[key] = len(lookup)
            return lookup[key]

        values = []
        for h in (normal, normal, custom, custom, normal):
            key = memo.canonical(h)
            try:
                values.append(
                    source(key, h) if aliases is None else aliases.get(key, h, source)
                )
            except ValueError as e:
                values.append(str(e))
        return values, events, memo.summary()

    exact(trial("original"), trial("native"))


def test_unknown_canonical_and_unretained_inputs_do_not_alias(module):
    native = module.IndexMemo(None)
    events = []

    def source(k, h):
        events.append((k, h))
        return 5

    for _ in range(5):
        assert native.get((1,), {}, source) == 5
    assert len(events) == 5 and native.summary()["identity_hits"] == 0
    retained = {}
    native = module.IndexMemo(retained)
    h = {(1, 0): {1: 0, 2: 1}}
    key = (1,)
    for _ in range(3):
        assert native.get(key, h, source) == 5
    assert native.summary()["identity_hits"] == 0 and native.summary()["bypassed"] == 3


def test_retained_key_and_type_refcounts_no_cross_block_cache(module):
    key = ((1, 0), ((1, 2**100), (2, -3)))
    h = {(1, 0): {1: 0, 2: 1}}
    retained = {id(h): (h, key)}
    objects = [key, h, retained, module.IndexMemo]
    gc.collect()
    before = [sys.getrefcount(o) for o in objects]

    def source(k, h):
        return 3

    for _ in range(10000):
        memo = module.IndexMemo(retained)
        assert memo.get(key, h, source) == 3 and memo.get(key, h, source) == 3
        assert memo.summary()["identity_hits"] == 1
        del memo
    gc.collect()
    assert [sys.getrefcount(o) for o in objects] == before


def test_gc_cycle_and_reinitialize(module):
    done = []

    class Payload:
        def __del__(self):
            done.append("done")

    retained = {}
    memo = module.IndexMemo(retained)
    assert gc.is_tracked(memo)
    retained["cycle"] = memo
    retained["payload"] = Payload()
    del memo, retained
    gc.collect()
    assert done == ["done"]
    memo = module.IndexMemo({})
    assert memo.get((), {}, lambda k, h: 0) == 0
    memo.__init__(None)
    assert memo.summary() == dict(
        requests=0,
        identity_hits=0,
        source_index_calls=0,
        bypassed=0,
        entries=0,
        enabled=False,
        probability_cache=False,
    )


@pytest.mark.parametrize("mode", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(cluster=True),
        dict(shadow=True, spawn=True),
        dict(feedback=False, preload=False, resample=False),
    ],
)
def test_four_epoch_all_state_metadata_rng_and_graph_events(
    extensions, scorer, mode, seed, options
):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(
        a,
        scorer,
        mode,
        **{k: v for k, v in extensions.items() if k != "index_extension"},
    )
    call = ScoreIndexMemoStep(b, scorer, mode, **extensions)
    a.step, b.step = old, call
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), call(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, {k: v for k, v in call.last.items() if k != "score_index_memo"})
        s = call.last["score_index_memo"]
        assert (
            s["requests"]
            == call.last["requested_moves"]
            == s["identity_hits"] + s["source_index_calls"]
        )
        assert (
            s["identity_hits"] > 0
            and not s["probability_cache"]
            and s["entries"] <= call.last["move_key_memo"]["identity_entries"] + 1
        )


def test_strict_policy_and_helper_mutation_guard(extensions, monkeypatch):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    call = ScoreIndexMemoStep(b, mode="scalar", **extensions)
    with np.errstate(under="raise"):
        for fn, rtk, ctx in ((a.step, ra, ca), (call, rb, cb)):
            with pytest.raises(FloatingPointError):
                fn(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    assert call.last == dict(original_step_fallback="numpy_error_policy")
    monkeypatch.setattr(MoveReuseStep, "_moves", lambda *a: None)
    with pytest.raises(ValueError, match="helper changed"):
        call(rb, cb)
