"""Seeded move/whole-step fixtures; graph extraction is a declared boundary.

The complete frozen RBPF class is compiled without GTSAM/cssrlib imports.
Synthetic epoch problems, satellite system IDs, times and integer-search
results replace external inputs only. All RBPF numerical/decision methods,
maintenance, spawning and shadow bookkeeping execute their literal source.
This is not a graph solve or a measured trajectory from receiver data.
"""

import ast
from collections import OrderedDict
from copy import deepcopy
import hashlib
import math
import os
import time
from types import SimpleNamespace

import numpy as np

from rbpf_likelihood_cases import make_case, SCALAR_SOURCE, SCALAR_SHA256


def frozen_classes():
    source = SCALAR_SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SCALAR_SHA256
    tree = ast.parse(source)
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in
              ('_env_f', '_env_i', '_canon_basin_id', '_copy_H')) or
             (isinstance(node, ast.ClassDef) and node.name in
              ('RbpfConfig', 'Particle', 'EpochProblem', 'RbpfFgo'))]
    future = ast.parse('from __future__ import annotations').body[0]
    module = ast.fix_missing_locations(ast.Module(body=[future] + deepcopy(nodes), type_ignores=[]))
    scope = dict(np=np, math=math, os=os, time=time, OrderedDict=OrderedDict,
                 _LOG2PI=math.log(2. * math.pi), __file__=str(SCALAR_SOURCE))
    exec(compile(module, str(SCALAR_SOURCE), 'exec'), scope)
    return SimpleNamespace(**scope)


def make_step_case(count=64, unique=8, *, rows=64, groups=4, seed=20260710,
                   cluster=False, uniform=True, huber=True, shadow=False,
                   spawn=False, resample=True, feedback=False):
    classes = frozen_classes()
    oracle, source, hypotheses = make_case(unique, rows=rows, groups=groups,
        seed=seed, cluster=cluster, marg_uniform=uniform, huber=huber, duplicates=False)
    cfg = classes.RbpfConfig()
    # Explicit values override ambient RBPF_* switches in this synthetic job.
    for name, value in vars(oracle.cfg).items():
        setattr(cfg, name, value)
    cfg.n_particles, cfg.seed, cfg.use_cuda = count, seed, 0
    cfg.fb_enable, cfg.spawn_enable = int(feedback), int(spawn)
    cfg.fb_m, cfg.fb_rel_m = 1, 1
    cfg.fb_dual_enable, cfg.fb_witness_pos, cfg.fb_support_min = 0, 0, 0
    cfg.fb_ddpr_rel_d, cfg.fb_commit_max_dpr = 0., 0.
    cfg.shadow_epochs, cfg.shadow_margin = (2 if shadow else 0), -1e4
    cfg.delayed_enable, cfg.delayed_rejuv_epochs = 0, 0
    cfg.gamma_fix, cfg.min_fix_nb = .99, 5
    cfg.p_survive, cfg.p_extend, cfg.p_shrink, cfg.p_release = .62, .12, .04, .02
    cfg.p_adopt_cons, cfg.p_adopt, cfg.p_spawn, cfg.p_burst = .12, .08, .12, .1
    cfg.ess_frac = 1.01 if resample else 0.
    cfg.spawn_after, cfg.spawn_min_dpos, cfg.spawn_min_cp = 2, 0., 3
    cfg.fix_max_dpr, cfg.fix_vote_dd, cfg.fix_min_info, cfg.loo_enable = 5., 0., 0., 0
    runtime = classes.RbpfFgo(cfg)
    problem = classes.EpochProblem()
    for name, value in vars(source).items():
        setattr(problem, name, value)
    problem.ok = True
    problem.Qab3 = np.zeros((3, len(problem.pairs)))
    # Deliberate agreement subset plus one nonconsensus pair, including
    # duplicate candidate columns; the original caller keeps their priors.
    base = np.rint(problem.y).astype(np.int64)
    problem.cands = np.column_stack((base, base, base.copy()))
    problem.cands[-1, -1] += 1
    problem.cand_sqnorm = np.arange(3, dtype=float)
    runtime.particles = [classes.Particle(deepcopy(hypotheses[i % unique]),
        float(-i % 7)) for i in range(count)]
    runtime.cluster_states = deepcopy(oracle.cluster_states)
    if shadow:
        runtime.shadow_challengers = [dict(H=deepcopy(hypotheses[-1]), score=0., age=0)]
    # Extraction receives the fixture via ctx. LAMBDA receives an explicit
    # synthetic candidate matrix; its search cost is not in this benchmark.
    runtime._extract = lambda _rtk, ctx: ctx['problem']
    runtime._lambda_topk = lambda y, _q, _k: (problem.cands[:len(y)].copy(), np.arange(3, dtype=float))
    systems = {int(sat): int(group[0]) for group, ref, j in problem.pairs for sat in (ref, j)}
    classes.sat2prn = lambda sat: (systems[int(sat)], int(sat))
    classes.time2gpst = lambda value: (0, float(value))
    # Functions execute against the compiled scope, not the namespace copy.
    scope = runtime.step.__func__.__globals__
    scope['sat2prn'], scope['time2gpst'] = classes.sat2prn, classes.time2gpst
    rtk = SimpleNamespace(epoch2=0, _amb_init_epoch={}, _rbpf_released=set(), _committed={})
    if feedback:
        indices = {key: i for i, key in enumerate(sorted({(sat, group[1])
            for group, ref, j in problem.pairs for sat in (ref, j)}))}
        rtk.amb_keys, rtk.events = set(indices), []
        rtk.nav = SimpleNamespace(x=np.zeros(len(indices)), na=0)
        rtk.IB = lambda sat, freq, _na: indices[(sat, freq)]

        def inject(values, keys):
            rtk.events.append(('inject', tuple(keys), values.copy()))
            rtk.nav.x = values.copy()

        def release(sat, freq, reason):
            rtk.events.append(('release', sat, freq, reason))
            rtk._committed.pop((sat, freq), None)
            rtk._rbpf_released.add((sat, freq))

        rtk._inject_hold, rtk._release_ambiguity = inject, release
    ctx = dict(problem=problem, obs=SimpleNamespace(t=100.))
    return runtime, rtk, ctx


def state(runtime, rtk, output):
    """All persistent RBPF fields except wall time and external input hooks."""
    skip = {'step_seconds', 'cfg', 'rng', 'particles', '_extract', '_lambda_topk', 'step'}
    data = {k: deepcopy(v) for k, v in vars(runtime).items() if k not in skip}
    data.update(particles=[dict(H=deepcopy(p.H), logw=p.logw, last_move=p.last_move)
                           for p in runtime.particles],
                rng=deepcopy(runtime.rng.bit_generator.state), output=deepcopy(output),
                rtk={k: deepcopy(v) for k, v in vars(rtk).items() if not callable(v)})
    return data


def comparison(left, right):
    """Record exact discrete/array bits and maximum numerical difference."""
    result = dict(discrete_equal=True, numeric_bits_equal=True, max_abs_error=0.,
                  position_bits_equal=True, position_max_abs_error=0.)

    def visit(a, b, path):
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
                result['discrete_equal'] = False
                return
            if np.issubdtype(a.dtype, np.floating):
                exact = a.tobytes() == b.tobytes()
                result['numeric_bits_equal'] &= exact
                finite = np.isfinite(a) & np.isfinite(b)
                error = float(np.max(np.abs(a[finite] - b[finite]))) if np.any(finite) else 0.
                if not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.array_equal(np.isnan(a), np.isnan(b)):
                    error = float('inf')
                result['max_abs_error'] = max(result['max_abs_error'], error)
                if path.endswith('/pos') or path.endswith('/P') or path.endswith('/Ppos'):
                    result['position_bits_equal'] &= exact
                    result['position_max_abs_error'] = max(result['position_max_abs_error'], error)
            else:
                result['discrete_equal'] &= a.tobytes() == b.tobytes()
        elif isinstance(a, SimpleNamespace):
            if not isinstance(b, SimpleNamespace):
                result['discrete_equal'] = False
                return
            visit(vars(a), vars(b), path)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or list(a) != list(b):
                result['discrete_equal'] = False
                return
            for k in a:
                visit(a[k], b[k], path + '/' + str(k))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                result['discrete_equal'] = False
                return
            for i, (av, bv) in enumerate(zip(a, b)):
                visit(av, bv, path + '/' + str(i))
        elif isinstance(a, (float, np.floating)):
            result['numeric_bits_equal'] &= np.float64(a).tobytes() == np.float64(b).tobytes()
            if np.isfinite(a) and np.isfinite(b):
                result['max_abs_error'] = max(result['max_abs_error'], abs(float(a) - float(b)))
            elif not ((np.isnan(a) and np.isnan(b)) or a == b):
                result['max_abs_error'] = float('inf')
        else:
            result['discrete_equal'] &= a == b

    visit(left, right, '')
    return result
