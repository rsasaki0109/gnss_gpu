"""Explicit experiment: stage frozen RBPF moves once, then batch their scores.

Construct a MoveBatchStep for one runtime and explicitly assign it to
``runtime.step``. The shipped source and default dispatch are never patched.
Only the fully adapted move loop is replaced; extraction, maintenance, shadow
observations, conditional positions, feedback and resampling remain literal.
"""

import ast
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np

from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256
from rbpf_likelihood_runtime_adapter import runtime_reference


def _function(method):
    return getattr(method, '__func__', method)


class MoveBatchStep:
    """Per-runtime, opt-in caller experiment; no candidate/RNG approximation.

    Scalar mode isolates staging from native numerical differences. Native
    modes need an explicit BatchScorer with a caller-owned lifetime. Valid
    finite inputs are the experimental domain; strict NumPy exception policies
    retain the original step and its original partial side effects on error.
    Source/method guards prevent silently applying this model to a newer one.
    ``last`` describes the move block only, outside the runtime's own state.
    """

    _METHODS = ('step', '_extend', '_shrink_worst', '_pair_dd', '_loglik',
                '_cluster_state', '_basin_parent_score', '_adopt', '_consensus_subset')

    def __init__(self, runtime, scorer=None, mode='scalar'):
        if mode not in ('scalar', 'native_cpu', 'gpu'):
            raise ValueError('Unknown explicit move scoring mode')
        if mode != 'scalar' and scorer is None:
            raise ValueError('Native modes require an explicit owning scorer')
        source = SCALAR_SOURCE.read_bytes()
        if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
            raise ValueError('Frozen RBPF source changed')
        self.runtime, self.scorer, self.mode = runtime, scorer, mode
        self.oracle = runtime_reference(runtime)
        self.original = runtime.step
        self.methods = {name: _function(getattr(runtime, name)) for name in self._METHODS}
        for name, function in self.methods.items():
            path = Path(function.__code__.co_filename)
            if (function.__name__ != name or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != SCALAR_SHA256):
                raise ValueError(f'Unsupported RBPF method: {name}')
        tree = ast.parse(source)
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RbpfFgo')
        step = deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'step'))
        # Exactly two top-level statements are replaced. Keep the following
        # original commit loop, and every other step statement, untouched.
        indices = [i for i, node in enumerate(step.body) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'new_states']
        if len(indices) != 1:
            raise ValueError('Move block boundary changed')
        i = indices[0]
        loop = step.body[i + 1]
        if not (isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)
                and loop.target.id == 'p' and ast.unparse(loop.iter) == 'self.particles'):
            raise ValueError('Move block loop changed')
        replacement = ast.parse(
            'new_states = _rbpf_batch_moves(prob, H_cons, cand_H, spawn_H, cache, ll_burst)'
        ).body[0]
        ast.copy_location(replacement, step.body[i])
        step.body[i:i + 2] = [replacement]
        step.name = '_explicit_batched_step'
        # The original module globals retain the exact conditional/backbone
        # helpers. The added callback is local to this compiled function.
        scope = dict(self.methods['step'].__globals__)
        scope['_rbpf_batch_moves'] = self._moves
        module = ast.fix_missing_locations(ast.Module(body=[step], type_ignores=[]))
        exec(compile(module, str(Path(__file__)), 'exec'), scope)
        self.compiled = scope[step.name]
        self.canonical = scope['_canon_basin_id']
        self.last = None

    def __call__(self, rtk, ctx):
        for name, function in self.methods.items():
            current = getattr(self.runtime, name)
            if name == 'step' and current is self:
                continue
            if _function(current) is not function:
                raise ValueError(f'RBPF method changed after binding: {name}')
        self.last = None
        policy = np.geterr()
        if any(v not in ('ignore', 'warn') for v in policy.values()) or policy['under'] != 'ignore':
            self.last = dict(original_step_fallback='numpy_error_policy')
            return self.original(rtk, ctx)
        return self.compiled(self.runtime, rtk, ctx)

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        runtime, cfg = self.runtime, self.runtime.cfg
        self.oracle.cfg = cfg
        keys, hypotheses, lookup, pending, staged = [], [], {}, [], []

        def index(hypothesis):
            key = self.canonical(hypothesis)
            if key not in lookup:
                lookup[key] = len(keys)
                keys.append(key)
                hypotheses.append(hypothesis)
                if cache.get(key) is None:
                    pending.append(len(keys) - 1)
            return lookup[key]

        held = 0
        for particle in runtime.particles:
            h = particle.H
            moves = []

            def add(prior, hypothesis, tag):
                moves.append((prior, index(hypothesis), hypothesis, tag))

            if h:
                held += 1
                add(cfg.p_survive, h, 'survive')
                extended = runtime._extend(h, prob)
                if extended is not None:
                    add(cfg.p_extend, extended, 'extend')
                shrunk = runtime._shrink_worst(h, prob)
                if shrunk is not None:
                    add(cfg.p_shrink, shrunk, 'shrink')
                add(cfg.p_release, {}, 'release')
            else:
                add(cfg.p_survive + cfg.p_extend + cfg.p_shrink + cfg.p_release, h, 'stay')
            if H_cons is not None:
                add(cfg.p_adopt_cons, H_cons, 'adopt_cons')
            for h in cand_H:
                add(cfg.p_adopt / len(cand_H), h, 'adopt')
            for h in spawn_H:
                add(cfg.p_spawn / len(spawn_H), h, 'spawn')
            staged.append(moves)

        requested = [hypotheses[i] for i in pending]
        if self.mode == 'scalar':
            values = [runtime._loglik(h, prob, cache) for h in requested]
            details = dict(unique=len(pending), scalar_fallbacks=0)
        else:
            values, details = self.scorer.score(requested, prob, self.oracle, mode=self.mode)
        for i, value in zip(pending, values):
            cache[keys[i]] = float(value)
        lp_nb = math.log(max(1.0 - cfg.p_burst, 1e-12))
        lp_b = math.log(max(cfg.p_burst, 1e-12)) + ll_burst
        likelihood = [float(np.logaddexp(lp_nb + cache[key], lp_b)) for key in keys]

        new_states = []
        for particle, records in zip(runtime.particles, staged):
            moves = [(prior, likelihood[i], h, tag) for prior, i, h, tag in records]
            # Literal original weight/choice expressions, in particle order.
            pri = np.array([mv[0] for mv in moves], dtype=float)
            pri /= pri.sum()
            lam_arr = np.log(pri) + np.array([mv[1] for mv in moves])
            mx = float(np.max(lam_arr))
            wsum = np.exp(lam_arr - mx)
            tot = float(np.sum(wsum))
            particle.logw += mx + math.log(tot)
            pick = int(runtime.rng.choice(len(moves), p=wsum / tot))
            new_states.append((moves[pick][2], moves[pick][3]))
        self.last = dict(mode=self.mode, particles=len(staged), held_particles=held,
            extend_calls=held, shrink_calls=held, requested_moves=sum(map(len, staged)),
            unique_moves=len(keys), existing_cache_hits=len(keys) - len(pending), batch=details)
        return new_states
