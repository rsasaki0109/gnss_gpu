"""Regression for static builtin type namespaces and owned API references."""

import gc
import sys
from types import SimpleNamespace

import pytest

from test_rbpf_versioned_staging import native, paths  # noqa: F401
from rbpf_epoch_memo import _EpochMemo
from rbpf_fused_staging import _MEMO, _CODE


@pytest.mark.parametrize("count", [0, 1, 64])
@pytest.mark.parametrize("common", [False, True])
def test_static_namespace_cold_empty_and_repeated_identity(native, count, common):
    a = {(1, 0): {1: 0, 2: 2}}
    memo = _EpochMemo(
        lambda h: tuple((g, tuple(z.items())) for g, z in h.items()),
        None,
        cache_dd=False,
    )
    context = SimpleNamespace(canonical=memo.canonical)
    runtime = SimpleNamespace(particles=[SimpleNamespace(H={}) for _ in range(count)])
    cfg = SimpleNamespace(
        p_survive=0.62,
        p_extend=0.12,
        p_shrink=0.04,
        p_release=0.02,
        p_adopt_cons=0.12,
        p_adopt=0.08,
        p_spawn=0.1,
    )
    args = (
        context,
        runtime,
        cfg,
        None,
        None,
        [a, a] if common else [],
        [],
        {},
        [],
        [],
        {},
        [],
        [],
    )
    held, stats = native.stage_versioned(*args, _MEMO, _CODE, _EpochMemo)
    assert held == 0 and stats["cache_deoptimizations"] == stats["invalidations"] == 0
    assert stats["attribute_cache_reads"] == count * (3 if common else 1)
    assert memo.counts["canonical_requests"] == stats["attribute_cache_reads"]
    if count and common:
        assert stats["identity_cache_hits"] > 0


def test_static_and_heap_type_dictionary_reference_counts(native):
    watched = gc.get_referents(vars(SimpleNamespace)) + gc.get_referents(
        vars(_EpochMemo)
    )
    assert len(watched) == 2 and all(type(d) is dict for d in watched)
    gc.collect()
    before = [sys.getrefcount(d) for d in watched]
    for _ in range(1000):
        test_static_namespace_cold_empty_and_repeated_identity(native, 1, True)
    gc.collect()
    assert before == [sys.getrefcount(d) for d in watched]
