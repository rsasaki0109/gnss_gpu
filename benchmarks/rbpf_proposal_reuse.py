"""Explicit frozen RBPF proposal templates; every particle still draws once.

The original staging/weight expressions are compiled from the pinned D227
caller. Common candidates are staged lazily after the first particle's moves,
preserving first-score order. Equal ordered prior/index sequences may reuse
their exact normalizer and probabilities within this move block only.
"""

import ast
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rbpf_epoch_memo import _EpochMemo
from rbpf_move_batch import MoveBatchStep
from rbpf_move_key_memo import MoveKeyMemoStep
from rbpf_move_reuse import _MoveView, _ordered_integer_key

MOVES_SHA256 = '69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea'
MOVES_SOURCE = Path(__file__).with_name('rbpf_move_batch.py')


class _ProposalMemo:
    def __init__(self, likelihood, common, rng):
        self.likelihood, self.common = likelihood, (() if common is None else common)
        self.enabled = type(rng) is np.random.Generator
        self.common_safe = all(self._positive(prior) for prior, _i, _h, _tag in self.common)
        self.cache = {}
        self.requests = self.evaluations = self.hits = self.bypassed = 0

    @staticmethod
    def _positive(prior):
        return type(prior) is float and math.isfinite(prior) and prior > 0.

    def statistics(self, records):
        self.requests += 1
        prefix = records[:-len(self.common)] if self.common else records
        safe = self.enabled and self.common_safe and all(self._positive(r[0]) for r in prefix)
        key = tuple((prior, i) for prior, i, _h, _tag in prefix) if safe else None
        if safe and key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.evaluations += 1
        mass, probabilities, normalized = _raw_statistics(records, self.likelihood)
        # Do not suppress per-particle warnings for zero/overflowed normalized
        # priors or nonfinite totals. Strict policies use the original step.
        if (safe and math.isfinite(mass) and np.all(normalized > 0.)
                and np.isfinite(normalized).all() and np.isfinite(probabilities).all()):
            self.cache[key] = (mass, probabilities)
        else:
            self.bypassed += 1
        return mass, probabilities

    def summary(self):
        return dict(requests=self.requests, evaluations=self.evaluations,
            hits=self.hits, bypassed=self.bypassed, entries=len(self.cache),
            probability_payload_bytes=sum(p.nbytes for _mass, p in self.cache.values()),
            common_records=len(self.common), standard_generator=self.enabled)


def _compile_original():
    source = MOVES_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != MOVES_SHA256:
        raise ValueError('Frozen move caller changed')
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoveBatchStep')
    original = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '_moves')
    fn = deepcopy(original)
    staging = next(n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'runtime.particles')
    marker = next(i for i,n in enumerate(staging.body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'H_cons is not None')
    common = deepcopy(staging.body[marker:marker+3])
    assert [ast.unparse(n.iter) for n in common[1:]] == ['cand_H', 'spawn_H']
    assert len(staging.body[marker:]) == 4 and ast.unparse(staging.body[-1]) == 'staged.append(moves)'
    begin = ast.parse('_common_begin = len(moves)').body[0]
    end = ast.parse('_common = tuple(moves[_common_begin:])').body[0]
    conditional = ast.If(test=ast.parse('_common is None', mode='eval').body,
        body=[begin]+common+[end], orelse=ast.parse('moves.extend(_common)').body)
    staging.body[marker:marker+3] = [conditional]
    fn.body.insert(fn.body.index(staging), ast.parse('_common = None').body[0])

    selection = next(n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'zip(runtime.particles, staged)')
    assert len(selection.body) == 10
    assert ast.unparse(selection.body[7]) == 'particle.logw += mx + math.log(tot)'
    raw = ast.FunctionDef(name='_raw_statistics', args=ast.parse('def f(records, likelihood): pass').body[0].args,
        body=deepcopy(selection.body[:7])+ast.parse('return mx + math.log(tot), wsum / tot, pri').body,
        decorator_list=[])
    selection.body = ast.parse('''
mass, probabilities = _proposal.statistics(records)
particle.logw += mass
pick = int(runtime.rng.choice(len(records), p=probabilities))
new_states.append((records[pick][2], records[pick][3]))
''').body
    fn.body.insert(fn.body.index(selection), ast.parse(
        '_proposal = _ProposalMemo(likelihood, _common, runtime.rng)').body[0])
    fn.body.insert(-1, ast.parse("self.last['proposal_reuse'] = _proposal.summary()").body[0])
    fn.name = '_proposal_moves'
    scope = dict(MoveBatchStep._moves.__globals__, _ProposalMemo=_ProposalMemo)
    module = ast.fix_missing_locations(ast.Module(body=[raw, fn], type_ignores=[]))
    exec(compile(module, str(Path(__file__)), 'exec'), scope)
    return scope['_raw_statistics'], scope['_proposal_moves']


_raw_statistics, _proposal_moves = _compile_original()


class ProposalReuseStep(MoveKeyMemoStep):
    """Opt-in lazy common staging and exact proposal-distribution reuse.

    Shared records/probabilities are read only. Selected H/tag still come from
    each particle's own ordered records; the original commit copies each H.
    Cache keys omit only the common suffix shared by this whole call, never
    a prior, score index or their order within a particle-specific prefix.
    """

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        cfg = self.runtime.cfg
        priors = ('p_survive', 'p_extend', 'p_shrink', 'p_release', 'p_adopt_cons', 'p_adopt', 'p_spawn')
        standard = (type(cand_H) is list and type(spawn_H) is list
            and all(type(getattr(cfg, name)) is float and math.isfinite(getattr(cfg, name)) for name in priors))
        if standard:
            common = ([] if H_cons is None else [H_cons]) + cand_H + spawn_H
            standard = all(_ordered_integer_key(h) is not None for h in common)
        if not standard:
            result = super()._moves(prob, H_cons, cand_H, spawn_H, cache, ll_burst)
            self.last['proposal_reuse'] = dict(original_moves_fallback='nonstandard_common_input')
            return result
        memo = _EpochMemo(self.canonical, self.runtime._pair_dd, cache_dd=False)
        view = _MoveView(self.runtime, prob)
        context = SimpleNamespace(runtime=view, oracle=self.oracle, canonical=memo.canonical,
            scorer=self.scorer, mode=self.mode, last=None)
        result = _proposal_moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        self.last.update(extend_calls=view.calls['extend'], shrink_calls=view.calls['shrink'],
            move_reuse=dict(requests=view.requests, calls=view.calls, uncacheable=view.uncacheable,
                hits={key: view.requests[key]-view.calls[key] for key in view.requests},
                identity_entries=len(view.identities), extend_entries=len(view.extended), shrink_entries=len(view.shrunk)),
            move_key_memo=memo.summary())
        return result
