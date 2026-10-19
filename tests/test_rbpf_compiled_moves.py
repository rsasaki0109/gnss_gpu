"""Literal native controller versus D236 state, RNG, feedback and metadata."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

if os.name != "nt" or not os.environ.get("GNSS_RBPF_COMPILED_MOVES_EXT"):
    pytest.skip(
        "Explicit Windows compiled-moves experiment required", allow_module_level=True
    )
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_compiled_moves import CompiledMovesStep  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_move_batch import MoveBatchStep  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


def exact(a, b):
    check = comparison(a, b)
    assert (
        check["discrete_equal"]
        and check["numeric_bits_equal"]
        and check["max_abs_error"] == 0.0
    ), check


@pytest.fixture(scope="module")
def paths():
    return dict(
        extension=ROOT / os.environ["GNSS_RBPF_CANONICAL_EXT"],
        parent_extension=ROOT / os.environ["GNSS_RBPF_PARENT_EXT"],
        mass_extension=ROOT / os.environ["GNSS_RBPF_MASS_EXT"],
    )


@pytest.fixture(scope="module")
def scorer():
    with RegisterLUScorer(ROOT / os.environ["GNSS_RBPF_BASELINE_DLL"]) as value:
        yield value


def controller(runtime, scorer, backend, variant, paths):
    if variant == "original":
        return NativeFeedbackMassStep(runtime, scorer, backend, **paths)
    return CompiledMovesStep(
        runtime,
        scorer,
        backend,
        compiled_extension=os.environ["GNSS_RBPF_COMPILED_MOVES_EXT"],
        enabled=variant == "on",
        **paths,
    )


@pytest.mark.parametrize("backend", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711])
@pytest.mark.parametrize(
    "n,u,rows,groups,cluster,feedback",
    [
        (64, 6, 64, 4, False, True),
        (64, 32, 64, 4, False, False),
        (64, 6, 128, 8, True, True),
        (128, 128, 128, 8, True, True),
    ],
)
def test_complete_filter_and_every_rng_draw_are_exact(
    paths, scorer, backend, seed, n, u, rows, groups, cluster, feedback
):
    saved, metadata = [], []
    for variant in ("original", "off", "on"):
        runtime, rtk, ctx = make_feedback_case(
            n, u, rows=rows, groups=groups, cluster=cluster, feedback=feedback
        )
        runtime.rng = np.random.default_rng(seed)
        call = controller(runtime, scorer, backend, variant, paths)
        runtime.step = call
        values, details = [], []
        for _ in range(4):
            output = call(rtk, ctx)
            values.append(state(runtime, rtk, output))
            details.append(deepcopy(call.last))
        if variant != "original":
            assert call.move_counts() == (
                dict(compiled=4, literal=0)
                if variant == "on"
                else dict(compiled=0, literal=4)
            )
        saved.append(values)
        metadata.append(details)
    for value in saved[1:]:
        exact(saved[0], value)
    for value in metadata[1:]:
        exact(metadata[0], value)


@pytest.mark.parametrize("kind", ["module", "source_global", "numpy", "method"])
def test_changed_entry_bindings_use_the_literal_controller(
    paths, scorer, monkeypatch, kind
):
    values = []
    for variant in ("off", "on"):
        runtime, rtk, ctx = make_feedback_case()
        call = controller(runtime, scorer, "native_cpu", variant, paths)
        runtime.step = call
        if kind == "module":
            monkeypatch.setattr(call.module, "np", object())
        elif kind == "source_global":
            # A new shadowing global was absent when the controller bound.
            monkeypatch.setitem(MoveBatchStep._moves.__globals__, "float", float)
        elif kind == "numpy":
            old = np.log
            monkeypatch.setattr(np, "log", lambda *a, **kw: old(*a, **kw))
        else:
            old = call._compiled_classes[0]._moves
            monkeypatch.setattr(
                call._compiled_classes[0], "_moves", lambda *a, **kw: old(*a, **kw)
            )
        output = call(rtk, ctx)
        assert call.move_counts() == dict(compiled=0, literal=1)
        values.append((state(runtime, rtk, output), deepcopy(call.last)))
        monkeypatch.undo()
    exact(*values)


def test_strict_numpy_fallback_stays_in_the_original_step(paths, scorer):
    saved = []
    for variant in ("original", "off", "on"):
        runtime, rtk, ctx = make_feedback_case()
        call = controller(runtime, scorer, "gpu", variant, paths)
        runtime.step = call
        output = error = None
        with np.errstate(all="raise"):
            try:
                output = call(rtk, ctx)
            except Exception as exc:
                error = (type(exc).__name__, str(exc))
        saved.append((state(runtime, rtk, output), deepcopy(call.last), error))
        if variant != "original":
            assert call.move_counts() == dict(compiled=0, literal=0)
    exact(saved[0], saved[1])
    exact(saved[0], saved[2])
