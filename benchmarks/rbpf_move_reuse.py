"""Explicit frozen-model experiment: reuse identical pure move construction.

The existing MoveBatchStep weight/choice code is reused without rewriting its
math. A per-call view delegates original bound methods except extend/shrink;
those two remember results for exactly equal ordered integer dictionaries.
No runtime method is replaced temporarily, and nothing survives the move block.
"""

from types import SimpleNamespace

from rbpf_move_batch import MoveBatchStep


def _ordered_integer_key(hypothesis):
    """Keep gauge, group/satellite insertion order and original integer types.

    Unusual mappings or numeric types are evaluated normally, without reuse.
    In particular this is deliberately stricter than a canonical basin ID.
    """
    if type(hypothesis) is not dict:
        return None
    parts = []
    for group, integers in hypothesis.items():
        if (type(group) is not tuple or len(group) != 2
                or any(type(value) is not int for value in group)
                or type(integers) is not dict
                or any(type(sat) is not int or type(z) is not int for sat, z in integers.items())):
            return None
        parts.append((group, tuple(integers.items())))
    return tuple(parts)


class _MoveView:
    """Original-state view; only deterministic candidate methods are cached."""

    def __init__(self, runtime, problem):
        self.runtime, self.problem = runtime, problem
        self.identities, self.extended, self.shrunk = {}, {}, {}
        self.requests = dict(extend=0, shrink=0)
        self.calls = dict(extend=0, shrink=0)
        self.uncacheable = dict(extend=0, shrink=0)

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def _key(self, h):
        identity = id(h)
        if identity not in self.identities:
            # Retain H as well as its key: no id reuse, and the frozen pure
            # move block does not change any H before the later commit loop.
            self.identities[identity] = (h, _ordered_integer_key(h))
        return self.identities[identity][1]

    def _evaluate(self, kind, method, memo, hypothesis, problem):
        if problem is not self.problem:
            raise ValueError('Move reuse cannot cross epoch problems')
        self.requests[kind] += 1
        key = self._key(hypothesis)
        if key is None:
            self.calls[kind] += 1
            self.uncacheable[kind] += 1
            return method(hypothesis, problem)
        # None is also a result: absence of an available move is reusable.
        if key not in memo:
            self.calls[kind] += 1
            memo[key] = method(hypothesis, problem)
        return memo[key]

    def _extend(self, hypothesis, problem):
        return self._evaluate('extend', self.runtime._extend, self.extended, hypothesis, problem)

    def _shrink_worst(self, hypothesis, problem):
        return self._evaluate('shrink', self.runtime._shrink_worst, self.shrunk, hypothesis, problem)


class MoveReuseStep(MoveBatchStep):
    """Opt-in per-epoch exact move reuse, with the original backend choices.

    All particles, requested moves, priors, scoring order and RNG draws are
    retained. Shared generated dictionaries are read only until the original
    commit loop copies each selected dictionary separately for each particle.
    The inherited source/method guards and strict-policy fallback still apply.
    Concurrent calls on one filter and malformed-input side effects are outside
    the same experimental domain as the original MoveBatchStep.
    """

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        view = _MoveView(self.runtime, prob)
        context = SimpleNamespace(runtime=view, oracle=self.oracle, canonical=self.canonical,
            scorer=self.scorer, mode=self.mode, last=None)
        result = MoveBatchStep._moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        self.last.update(extend_calls=view.calls['extend'], shrink_calls=view.calls['shrink'],
            move_reuse=dict(requests=view.requests, calls=view.calls, uncacheable=view.uncacheable,
                hits={key: view.requests[key] - view.calls[key] for key in view.requests},
                identity_entries=len(view.identities), extend_entries=len(view.extended), shrink_entries=len(view.shrunk)))
        return result
