"""Exact held-DD feedback mass using the unchanged D235 CPU extension."""

import ast
import builtins
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from types import FunctionType

import numpy as np

from rbpf_component_mass import NativeComponentMassStep
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256


def _compile_codes():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError('Frozen RBPF source changed')
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'RbpfFgo')
    method = deepcopy(next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name == '_feedback'))
    baseline = deepcopy(method)
    begin, = [i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'dd_all']
    end, = [i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'bad_keys']
    original = deepcopy(method.body[begin:end])
    assert len(original) == 5
    assert ast.unparse(original[0]) == 'dd_all = [self._pair_dd(p.H, prob) for p in self.particles]'
    assert [ast.unparse(n) for n in original[1:4]] == ['group_held = {}','group_bad = {}','held_pairs = []']
    loop = original[4]
    assert ast.unparse(loop.iter) == 'enumerate(prob.pairs)' and len(loop.body) == 10
    assert ast.unparse(loop.body[4]) == 'z_held = int(np.rint(vr - vj))'
    assert ast.unparse(loop.body[5]) == 'mass = 0.0'
    assert ast.unparse(loop.body[6].iter) == 'range(len(self.particles))'
    raw = ast.parse('def _original_release_mass(self, prob, w, committed, cfg): pass').body[0]
    raw.body = deepcopy(original)+ast.parse('return held_pairs, group_held, group_bad').body
    collect = ast.parse('def _held_targets(self, prob, w, committed, cfg): pass').body[0]
    prefix = deepcopy(loop)
    prefix.body = deepcopy(loop.body[:5])+ast.parse('_target[pi] = z_held\n_records.append((pi,g,ref,j))').body
    collect.body = ast.parse('_target = np.full(len(prob.pairs),np.nan)\n_records = []').body+[prefix]+ast.parse('return _records, _target').body
    finish = ast.parse('def _release_groups(records, masses, cfg): pass').body[0]
    grouped = ast.parse('for pi,g,ref,j in records: pass').body[0]
    grouped.body = ast.parse('mass = masses[pi]').body+deepcopy(loop.body[7:])
    finish.body = deepcopy(original[1:4])+[grouped]+ast.parse('return held_pairs, group_held, group_bad').body
    method.body[begin:end] = ast.parse('held_pairs, group_held, group_bad = _rbpf_feedback_mass(self, prob, w, committed, cfg)').body
    restored = deepcopy(method)
    restored.body[begin:begin+1] = deepcopy(original)
    assert ast.dump(restored,include_attributes=False) == ast.dump(baseline,include_attributes=False)
    module = compile(ast.fix_missing_locations(ast.Module(body=[raw,collect,finish,method],type_ignores=[])),
        str(Path(__file__)), 'exec',dont_inherit=True)
    return {c.co_name:c for c in module.co_consts if hasattr(c,'co_name')}


_CODES = _compile_codes()
_RINT, _RANGE = np.rint,builtins.range
_ARRAY, _F64 = np.ndarray,np.dtype('float64')


class FeedbackMassEngine:
    """Original hold rounding/group decisions with D235 ordered masses.

    No new extension or matrix cache. Prelude releases run exactly once before
    this block, then current H/committed values are read. Only plain stable
    source objects, finite modest builtin-float holds and ordinary bindings use
    target collection; unsupported cases run the original block once, after
    the same prelude. All DDs still expand, including non-target pairs/errors.
    """

    def __init__(self, component):
        self.component = component
        self.scope = component.scope
        self.original = FunctionType(_CODES['_original_release_mass'],self.scope)
        self.collect = FunctionType(_CODES['_held_targets'],self.scope)
        self.finish = FunctionType(_CODES['_release_groups'],self.scope)
        self.cfg_type = self.scope['RbpfConfig']
        self.native_calls = self.fallback_calls = self.native_particles = self.native_pairs = self.held_targets = 0

    def _eligible(self, runtime, problem, weights, committed, cfg):
        component = self.component
        if (not component._ordinary() or np.rint is not _RINT
                or self.scope.get('range',self.original.__builtins__.get('range')) is not _RANGE
                or type(problem) is not component.problem_type
                or component.problem_type.__getattribute__ is not object.__getattribute__
                or type(cfg) is not self.cfg_type or self.cfg_type.__getattribute__ is not object.__getattribute__
                or type(committed) is not dict or type(weights) is not _ARRAY or weights.dtype != _F64
                or weights.ndim != 1 or not weights.flags.c_contiguous):
            return False
        if (component.problem_type.__dict__.get('pairs') is not None or 'pairs' not in vars(problem)
                or self.cfg_type.__dict__.get('fb_rel_mass') is not None or 'fb_rel_mass' not in vars(cfg)):
            return False
        particles,pairs = runtime.particles,problem.pairs
        if (type(particles) is not list or weights.size != len(particles)
                or any(type(p) is not component.particle_type for p in particles)
                or type(pairs) is not list or type(cfg.fb_rel_mass) is not float):
            return False
        for key,value in committed.items():
            if (type(key) is not tuple or len(key) != 2 or any(type(x) is not int for x in key)
                    or (value is not None and (type(value) is not float or not math.isfinite(value) or abs(value)>1e100))):
                return False
        for pair in pairs:
            if (type(pair) is not tuple or len(pair) != 3 or type(pair[0]) is not tuple or len(pair[0]) != 2
                    or any(type(x) is not int for x in (*pair[0],pair[1],pair[2]))):
                return False
        return True

    def release_mass(self, runtime, problem, weights, committed, cfg):
        if self._eligible(runtime,problem,weights,committed,cfg):
            records,target = self.collect(runtime,problem,weights,committed,cfg)
            # Collecting finite plain holds cannot raise or mutate state. The
            # unchanged native call still expands EVERY DD in original order.
            masses = self.component.module.component_mass([p.H for p in runtime.particles],problem.pairs,target,weights)
            if masses is not None:
                self.native_calls += 1
                self.native_particles += len(runtime.particles)
                self.native_pairs += len(problem.pairs)
                self.held_targets += len(records)
                return self.finish(records,masses,cfg)
        self.fallback_calls += 1
        return self.original(runtime,problem,weights,committed,cfg)

    def counts(self):
        return dict(native_calls=self.native_calls,fallback_calls=self.fallback_calls,
            native_particles=self.native_particles,native_pairs=self.native_pairs,held_targets=self.held_targets)


class NativeFeedbackMassStep(NativeComponentMassStep):
    """D235 plus one feedback statistics block; graph mutation code unchanged.

    The original support/DDPR-release prelude, release/commit ordering,
    strict '< fb_rel_mass', streak counts, graph callbacks and RNG remain.
    No cross-phase DD reuse: prelude callbacks may alter particles/holds.
    """

    def __init__(self, runtime, scorer=None, mode='scalar', **extensions):
        super().__init__(runtime,scorer,mode,**extensions)
        self.feedback_engine = FeedbackMassEngine(self.mass_engine)
        original = self._source_functions['_feedback']
        replacement = FunctionType(_CODES['_feedback'],self._scope,original.__name__,
            original.__defaults__,original.__closure__)
        replacement.__kwdefaults__ = original.__kwdefaults__
        self._functions['_feedback'] = replacement
        base = self.compiled

        def call(view,rtk,ctx):
            self._scope['_rbpf_feedback_mass'] = self.feedback_engine.release_mass
            try:
                return base(view,rtk,ctx)
            finally:
                del self._scope['_rbpf_feedback_mass']

        self.compiled = call

    def __call__(self, rtk, ctx):
        before,previous = self.feedback_engine.counts(),self.last
        try:
            return super().__call__(rtk,ctx)
        finally:
            if self.last is not previous and self.last is not None and 'batch' in self.last:
                self.last['feedback_mass'] = {k:v-before[k] for k,v in self.feedback_engine.counts().items()}
