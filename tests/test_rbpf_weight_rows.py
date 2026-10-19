"""Exact row arithmetic and complete frozen PF/graph/RNG comparisons."""

from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402
from rbpf_weight_rows import WeightRowsStep, prepare_rows, AST_PROOFS  # noqa: E402


def exact(a, b):
    c = comparison(a, b)
    assert (
        c["discrete_equal"] and c["numeric_bits_equal"] and c["max_abs_error"] == 0.0
    ), c


def literal_statistics(records, likelihood):
    moves = [(prior, likelihood[i], h, tag) for prior, i, h, tag in records]
    pri = np.array([mv[0] for mv in moves], dtype=float)
    pri /= pri.sum()
    lam = np.log(pri) + np.array([mv[1] for mv in moves])
    mx = float(np.max(lam))
    wsum = np.exp(lam - mx)
    tot = float(np.sum(wsum))
    return mx, tot, wsum / tot


@pytest.mark.parametrize(
    "width",
    [
        1,
        2,
        3,
        7,
        8,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
    ],
)
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
def test_fast_axis_reductions_and_transcendentals_are_bit_identical(width, seed):
    rng = np.random.default_rng(seed + width)
    likelihood = rng.normal(0, 40, 43).tolist()
    staged = [
        [
            (float(np.exp(rng.uniform(-20, 0))), int(rng.integers(43)), {}, "x")
            for _ in range(width)
        ]
        for _ in range(19)
    ]
    prepared, stats = prepare_rows(staged, likelihood)
    assert (
        stats["rows"] == 19
        and stats["cells"] == 19 * width
        and stats["row_batches"] == 1
    )
    for records, actual in zip(staged, prepared, strict=True):
        want = literal_statistics(records, likelihood)
        for x, y in zip(want, actual, strict=True):
            assert np.array_equal(
                np.asarray(x).view(np.uint64), np.asarray(y).view(np.uint64)
            )


def test_ragged_rows_keep_order_and_duplicate_candidates():
    rng = np.random.default_rng(81)
    likelihood = [-1200.0, -700.0, -10.0, 0.0, 0.0, 1e3]
    staged = [
        [(float(rng.uniform(0.01, 1)), int(rng.integers(6)), {}, "x") for _ in range(n)]
        for n in [3, 8, 3, 17, 8, 1, 17, 3, 3, 64]
    ]
    with np.errstate(under="ignore"):
        prepared, stats = prepare_rows(staged, likelihood)
        assert stats["row_batches"] == 5
        for records, actual in zip(staged, prepared, strict=True):
            exact(literal_statistics(records, likelihood), actual)


@pytest.mark.parametrize(
    "prior,score",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (5e-324, 1.0),
    ],
)
def test_unsafe_rows_defer_without_warnings_or_input_mutation(prior, score):
    staged = [
        [(0.5, 0, {}, "a"), (0.5, 0, {}, "b")],
        [(prior, 0, {}, "c"), (1.7e308, 0, {}, "d")],
    ]
    saved = deepcopy(staged)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        assert prepare_rows(staged, [score]) is None
    assert not records
    exact(saved, staged)


def test_nonstandard_values_and_capacity_defer_without_running_float_hooks():
    class Foreign:
        def __float__(self):
            raise AssertionError("No early custom conversion")

    assert prepare_rows([[(Foreign(), 0, {}, "x")]], [0.0]) is None
    assert prepare_rows([[]], [0.0]) is None
    assert prepare_rows([[(1.0, 2, {}, "x")]], [0.0]) is None
    assert prepare_rows([[(1.0, 0, {}, "x")] * 131073], [0.0]) is None
    assert len(AST_PROOFS) == 4 and all(p["restored_AST_equal"] for p in AST_PROOFS)


@pytest.fixture(scope="module")
def paths():
    names = ("DLL", "CANONICAL_EXT", "PARENT_EXT", "MASS_EXT")
    values = {n: os.environ.get("GNSS_GPU_RBPF_WEIGHT_" + n) for n in names}
    if not all(values.values()):
        pytest.skip("Explicit recorded numerical DLL and CPU extensions required")
    return values


@pytest.fixture(scope="module")
def scorer(paths):
    with RegisterLUScorer(paths["DLL"]) as owned:
        yield owned


def options(paths):
    return dict(
        extension=paths["CANONICAL_EXT"],
        parent_extension=paths["PARENT_EXT"],
        mass_extension=paths["MASS_EXT"],
    )


@pytest.mark.parametrize("backend", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711])
@pytest.mark.parametrize(
    "config", [(64, 6, 64, 4, False), (64, 6, 64, 4, True), (128, 32, 128, 8, True)]
)
def test_four_epochs_full_state_rng_graph_and_metadata(
    paths, scorer, backend, seed, config
):
    n, u, m, g, cluster = config
    saved, details = [], []
    for variant in ("original", "literal", "batched"):
        runtime, rtk, ctx = make_feedback_case(
            n,
            u,
            rows=m,
            groups=g,
            cluster=cluster,
            seed=seed,
            shadow=True,
            spawn=True,
            feedback=True,
            preload=True,
        )
        if variant == "original":
            call = NativeFeedbackMassStep(runtime, scorer, backend, **options(paths))
        else:
            call = WeightRowsStep(
                runtime, scorer, backend, enabled=variant == "batched", **options(paths)
            )
        runtime.step = call
        values, meta = [], []
        for _ in range(4):
            output = call(rtk, ctx)
            values.append(state(runtime, rtk, output))
            meta.append(
                deepcopy({k: v for k, v in call.last.items() if k != "weight_rows"})
            )
            if variant != "original":
                counts = call.last_weight_rows
                assert counts["used"] == (variant == "batched")
                assert (
                    counts["rows"] == n
                    and counts["cells"] == call.last["requested_moves"]
                )
        saved.append(values)
        details.append(meta)
    for value in saved[1:]:
        exact(saved[0], value)
    for value in details[1:]:
        exact(details[0], value)


@pytest.mark.parametrize(
    "kind", ["zero_prior", "nan_score", "custom_rng", "numpy_binding"]
)
def test_literal_selection_partial_errors_rng_and_binding_fallback(
    paths, scorer, kind, monkeypatch
):
    results = []
    for enabled in (False, True):
        runtime, _, _ = make_feedback_case(feedback=False)
        call = WeightRowsStep(
            runtime, scorer, "native_cpu", enabled=enabled, **options(paths)
        )
        call._row_clones[0].__globals__.update(_NP_bindings_only_for_test=True)
        particles = [call._particle({}, float(i)) for i in range(3)]
        runtime.particles = particles
        runtime.rng = np.random.default_rng(38)
        events = []
        if kind == "custom_rng":
            original = runtime.rng

            class Hook:
                def choice(self, n, p):
                    events.append((n, p.copy()))
                    return original.choice(n, p=p)

            runtime.rng = Hook()
        if kind == "numpy_binding":
            original_log = np.log

            def hooked(x, *a, **kw):
                events.append(("log", x.copy()))
                return original_log(x, *a, **kw)

            monkeypatch.setattr(np, "log", hooked)
        staged = [[(0.7, 0, {}, "a"), (0.3, 1, {}, "b")] for _ in particles]
        likelihood = [0.0, float("nan") if kind == "nan_score" else -2.0]
        if kind == "zero_prior":
            staged[1] = [(0.0, 0, {}, "c"), (0.0, 1, {}, "d")]
        error, selected = None, None
        with (
            np.errstate(under="ignore"),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            try:
                selected = call._choices(runtime, staged, likelihood, call._literal)
            except Exception as exc:
                error = type(exc).__name__, str(exc)
        rng = runtime.rng if kind != "custom_rng" else original
        results.append(
            (
                [p.logw for p in particles],
                selected,
                error,
                rng.bit_generator.state,
                events,
                [(str(w.message), w.category.__name__) for w in caught],
            )
        )
        assert not call.last_weight_rows["used"]
        if kind == "numpy_binding":
            monkeypatch.undo()
    exact(*results)
