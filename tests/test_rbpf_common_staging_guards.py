"""Mutations after actual suffix replay must deoptimize with exact state."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("threadpoolctl")
from test_rbpf_common_staging_checked import paths, native  # noqa: E402,F401
from rbpf_epoch_memo import _EpochMemo  # noqa: E402
from rbpf_fused_staging import _MEMO, _CODE  # noqa: E402
from test_rbpf_native_staging import exact  # noqa: E402


def trial(native, fast, change):
    class Plain:
        pass

    h = {(1, 0): {1: 0, 2: 3}}
    a = {(1, 0): {1: 0, 2: 5}}
    b = {(1, 0): {1: 0, 2: 7}}
    candidates, spawn = [a, b, deepcopy(a)], [b]
    cfg = Plain()
    vars(cfg).update(p_survive=.6, p_extend=.1, p_shrink=.1, p_release=.2,
                     p_adopt_cons=.12, p_adopt=.08, p_spawn=.1)
    context = Plain()
    events = []

    def canonical(value):
        events.append("source")
        return tuple((g, tuple(z.items())) for g, z in value.items())

    memo = _EpochMemo(canonical, None, cache_dd=False)
    context.canonical = memo.canonical
    calls = 0

    class Count(int):
        def __add__(self, value):
            events.append("counter")
            return int(self) + value

    def extend(value, problem):
        nonlocal calls
        calls += 1
        events.append("extend")
        if calls == 4:
            if change == "nested_h":
                a[(1, 0)][2] += 1
            elif change == "root_h":
                a[(2, 0)] = {4: 0, 5: 2}
            elif change == "candidate_list":
                candidates.append(h)
            elif change == "prior":
                cfg.p_adopt = .16
            elif change == "cfg_dict":
                cfg.__dict__ = dict(vars(cfg), p_adopt=.16)
            elif change == "context_dict":
                context.__dict__ = dict(vars(context))
            elif change == "counts_value":
                memo.counts["canonical_requests"] = Count(memo.counts["canonical_requests"])
            elif change == "exception":
                raise ValueError("original callback")
        return None

    runtime = SimpleNamespace(particles=[SimpleNamespace(H=h) for _ in range(7)],
                              _extend=extend, _shrink_worst=lambda value, problem: None)
    keys, hs, lookup, pending, staged = [], [], {}, [], []
    args = (context, runtime, cfg, None, a, candidates, spawn, {}, keys, hs, lookup, pending, staged)
    error = None
    stats = None
    try:
        if fast:
            held, stats = native.stage_common(*args, _MEMO, _CODE, _EpochMemo)
        else:
            held = native.stage(*args)
    except ValueError as failure:
        held, error = None, str(failure)
    return dict(held=held, error=error, keys=keys, hs=hs, pending=pending, staged=staged,
                counts=memo.summary(), events=events), stats


@pytest.mark.parametrize("change", ["none", "nested_h", "root_h", "candidate_list", "prior", "cfg_dict", "context_dict", "counts_value", "exception"])
def test_after_replay_mutation_and_exception_state(native, change):
    original, _ = trial(native, False, change)
    candidate, stats = trial(native, True, change)
    exact(original, candidate)
    if stats is not None:
        assert stats["replayed_particles"] > 0
        if change not in ("none", "counts_value"):
            assert stats["replayed_particles"] == 1
