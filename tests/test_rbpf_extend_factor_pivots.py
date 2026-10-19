"""Additional pivot, conditioning and interrupted-epoch coverage."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_extend_factor import ExtendFactorCache, ExtendFactorStep  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


@pytest.mark.parametrize("size", [2, 3, 7, 11, 32, 64, 128])
def test_factor_reuse_with_actual_pivot_exchanges(size):
    rng = np.random.default_rng(20261018 + size)
    matrix = np.roll(np.eye(size), 1, axis=0) + 0.001 * rng.normal(size=(size, size))
    cache = ExtendFactorCache()
    cache.begin()
    for _ in range(8):
        rhs = rng.normal(size=size)
        assert cache.solve(matrix, rhs).tobytes() == np.linalg.solve(matrix, rhs).tobytes()
    _, pivots = next(iter(cache.cache.values()))
    assert np.any(pivots != np.arange(1, size + 1))
    assert cache.counts["factorizations"] == 1 and cache.counts["factor_hits"] == 7
    cache.end()


@pytest.mark.parametrize("size", [3, 7, 11, 32])
def test_ill_conditioned_hilbert_solutions(size):
    indices = np.arange(size, dtype=float)
    matrix = 1.0 / (indices[:, None] + indices[None, :] + 1.0)
    cache = ExtendFactorCache()
    cache.begin()
    for scale in [1.0, 2.0**-500, 2.0**400]:
        rhs = np.arange(1, size + 1, dtype=float) * scale
        try:
            expected = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            with pytest.raises(np.linalg.LinAlgError):
                cache.solve(matrix, rhs)
        else:
            assert cache.solve(matrix, rhs).tobytes() == expected.tobytes()
    cache.end()


def test_failure_after_active_factor_reuse_clears_cache_and_preserves_state():
    dependencies = json.loads((ROOT / "dist/rbpf_common_staging_warm_tests_20261018.plan.json").read_text(encoding="utf8"))["extensions"]
    original = make_feedback_case(64, 32)
    runtime, _, ctx = original
    template = None
    for particle in runtime.particles:
        dd = runtime._pair_dd(particle.H, ctx["problem"])
        if np.isfinite(dd).any() and any(not np.isfinite(dd[pi]) and particle.H.get(g, {}).get(ref) is not None for pi, (g, ref, _) in enumerate(ctx["problem"].pairs)):
            template = deepcopy(particle.H)
            break
    assert template is not None
    for index, particle in enumerate(runtime.particles):
        particle.H = {g: {sat: z + 31 * index for sat, z in integers.items()} for g, integers in template.items()}
    modified = deepcopy(original)
    results = []
    class InterruptedRng:
        def __init__(self):
            self.source = np.random.default_rng(20261018)
            self.bit_generator = self.source.bit_generator
        def __getattr__(self, name):
            return getattr(self.source, name)
        def choice(self, *args, **kwargs):
            self.source.choice(*args, **kwargs)
            raise RuntimeError("interrupted original particle choice")
    for inputs, cls in [(original, NativeFeedbackMassStep), (modified, ExtendFactorStep)]:
        runtime, rtk, ctx = inputs
        _bind_fixture_callbacks(rtk)
        runtime.rng = InterruptedRng()
        call = cls(runtime, None, "scalar", extension=ROOT / dependencies["CANONICAL_EXT"], parent_extension=ROOT / dependencies["PARENT_EXT"], mass_extension=ROOT / dependencies["MASS_EXT"])
        runtime.step = call
        with pytest.raises(RuntimeError, match="interrupted original particle choice"):
            runtime.step(rtk, ctx)
        results.append(state(runtime, rtk, None))
        if cls is ExtendFactorStep:
            assert call.extend_factor.counts["factor_hits"] > 0
            assert call.extend_factor.cache is None
            assert "_rbpf_extend_solve" not in call._scope
    c = comparison(*results)
    assert c["discrete_equal"] and c["numeric_bits_equal"] and c["max_abs_error"] == 0, c
