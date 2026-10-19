# cython: language_level=3, infer_types=False, binding=True, cdivision=False, boundscheck=True, wraparound=True, annotation_typing=False, profile=False, linetrace=False
# Generated from unchanged method ASTs; do not edit after numerical execution.

import math
import numpy as np
from types import SimpleNamespace
from rbpf_epoch_memo import _EpochMemo
from rbpf_move_reuse import _MoveView

class MoveBatchStep:

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        runtime, cfg = (self.runtime, self.runtime.cfg)
        self.oracle.cfg = cfg
        keys, hypotheses, lookup, pending, staged = ([], [], {}, [], [])

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
            pri = np.array([mv[0] for mv in moves], dtype=float)
            pri /= pri.sum()
            lam_arr = np.log(pri) + np.array([mv[1] for mv in moves])
            mx = float(np.max(lam_arr))
            wsum = np.exp(lam_arr - mx)
            tot = float(np.sum(wsum))
            particle.logw += mx + math.log(tot)
            pick = int(runtime.rng.choice(len(moves), p=wsum / tot))
            new_states.append((moves[pick][2], moves[pick][3]))
        self.last = dict(mode=self.mode, particles=len(staged), held_particles=held, extend_calls=held, shrink_calls=held, requested_moves=sum(map(len, staged)), unique_moves=len(keys), existing_cache_hits=len(keys) - len(pending), batch=details)
        return new_states

class MoveReuseStep:

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        view = _MoveView(self.runtime, prob)
        context = SimpleNamespace(runtime=view, oracle=self.oracle, canonical=self.canonical, scorer=self.scorer, mode=self.mode, last=None)
        result = MoveBatchStep._moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        self.last.update(extend_calls=view.calls['extend'], shrink_calls=view.calls['shrink'], move_reuse=dict(requests=view.requests, calls=view.calls, uncacheable=view.uncacheable, hits={key: view.requests[key] - view.calls[key] for key in view.requests}, identity_entries=len(view.identities), extend_entries=len(view.extended), shrink_entries=len(view.shrunk)))
        return result

class MoveKeyMemoStep:

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        memo = _EpochMemo(self.canonical, self.runtime._pair_dd, cache_dd=False)
        context = SimpleNamespace(runtime=self.runtime, oracle=self.oracle, canonical=memo.canonical, scorer=self.scorer, mode=self.mode, last=None)
        result = MoveReuseStep._moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        self.last['move_key_memo'] = memo.summary()
        return result

class NativeCanonicalStep:

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        context = SimpleNamespace(runtime=self._view, oracle=self._oracle, canonical=self.engine.canonical, scorer=self.scorer, mode=self.mode, last=None)
        result = MoveKeyMemoStep._moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        return result
SOURCE_METHOD_SHA256 = {'rbpf_move_batch.py': '69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea', 'rbpf_move_reuse.py': '9d1c35239afcbac4c4ed17494c5072cb2fe793a24e8dd3548d5c85b2933b4612', 'rbpf_move_key_memo.py': 'ddcc139b41212bfe813d552f6921b2914b6452571be410e2783a65f0d9cf9632', 'rbpf_native_canonical.py': 'f177e3247617cca70e4aa0c4369f1856dba979703f478d62c1f05cd814c63bd2'}
