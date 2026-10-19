"""Exact fused DD expansion/ordered FIX-component mass on the D234 caller."""

import ast
import builtins
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import FunctionType, MemberDescriptorType

import numpy as np

from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256, make_oracle
from rbpf_native_canonical import _code_key
from rbpf_parent_score import NativeParentScoreStep


def _compile_codes():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError('Frozen RBPF source changed')
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'RbpfFgo')
    step = deepcopy(next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name == 'step'))
    # Same literal D227 move adaptation; do not change its commit/remainder.
    i, = [i for i,n in enumerate(step.body) if isinstance(n,ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'new_states']
    assert ast.unparse(step.body[i+1].iter) == 'self.particles'
    step.body[i:i+2] = ast.parse('new_states = _rbpf_batch_moves(prob, H_cons, cand_H, spawn_H, cache, ll_burst)').body
    baseline = deepcopy(step)
    block, = [n for n in step.body if isinstance(n,ast.If) and ast.unparse(n.test) == 'dd_map is not None']
    assert len(block.body) == 5
    original = deepcopy(block.body[:3])
    assert ast.unparse(original[0]) == 'dd_all = [self._pair_dd(p.H, prob) for p in self.particles]'
    assert ast.unparse(original[1]) == 'pair_mass = {}'
    assert ast.unparse(original[2].iter) == 'np.where(np.isfinite(dd_map))[0]'
    raw = ast.parse('def _original_component_mass(self, prob, w, dd_map): pass').body[0]
    raw.body = original + ast.parse('return pair_mass').body
    block.body[:3] = ast.parse('pair_mass = _rbpf_component_mass(self, prob, w, dd_map)').body
    step.name = '_component_mass_step'
    module = compile(ast.fix_missing_locations(ast.Module(body=[raw,step],type_ignores=[])),
        str(Path(__file__)), 'exec', dont_inherit=True)
    codes = {c.co_name:c for c in module.co_consts if hasattr(c,'co_name')}
    # Structural proof: undo only the allowed substitution and recover D227.
    restored = deepcopy(step)
    restored.name = baseline.name
    restore_block, = [n for n in restored.body if isinstance(n,ast.If) and ast.unparse(n.test) == 'dd_map is not None']
    restore_block.body[:1] = deepcopy(original)
    assert ast.dump(restored,include_attributes=False) == ast.dump(baseline,include_attributes=False)
    imported = compile(source,str(SCALAR_SOURCE),'exec',dont_inherit=True)
    cls = next(c for c in imported.co_consts if getattr(c,'co_name',None) == 'RbpfFgo')
    pair = next(c for c in cls.co_consts if getattr(c,'co_name',None) == '_pair_dd')
    return codes['_original_component_mass'], codes['_component_mass_step'], (
        _code_key(pair),_code_key(make_oracle()._pair_dd.__func__.__code__))


_RAW_CODE, _STEP_CODE, _PAIR_CODES = _compile_codes()
_BINDINGS = {name:getattr(builtins,name) for name in ('len','enumerate','int','float')}
_FULL, _FINITE, _WHERE, _NAN = np.full, np.isfinite, np.where, np.nan
_ARRAY, _F64 = np.ndarray, np.dtype('float64')


class ComponentMassEngine:
    """No cache: exact particle/pair expansion followed by ordered FP64 sums.

    A finite binary64 value truncated toward zero is itself an exactly
    representable binary64 integer, even beyond int64. Comparing these values
    gives the same result as Python's unlimited int(di)==int(target). Integer
    hypotheses are subtracted as PyLong before the original FP64 conversion.
    Unsupported types/bindings/layouts execute the literal Python block once.
    """

    def __init__(self, extension, pair_dd, scope):
        if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3,12):
            raise RuntimeError('The explicit component-mass probe requires CPython 3.12')
        spec = importlib.util.spec_from_file_location('_rbpf_component_mass_probe',Path(extension).resolve())
        if spec is None or spec.loader is None:
            raise ValueError('An explicit component-mass extension is required')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.pair = pair_dd.__func__
        self.code = self.pair.__code__
        self.literal = (Path(self.code.co_filename).resolve() == SCALAR_SOURCE.resolve()
            and _code_key(self.code) in _PAIR_CODES)
        self.scope = scope
        self.original = FunctionType(_RAW_CODE,scope)
        self.particle_type, self.problem_type = scope['Particle'],scope['EpochProblem']
        self.slot = self.particle_type.__dict__['H']
        self.native_calls = self.fallback_calls = self.native_particles = self.native_pairs = self.peak_matrix_bytes = 0

    def _ordinary(self):
        if not self.literal or self.pair.__code__ is not self.code:
            return False
        for scope,base in ((self.scope,self.original.__builtins__),
                (self.pair.__globals__,self.pair.__builtins__)):
            if scope.get('np') is not np or any(scope.get(k,base.get(k)) is not v for k,v in _BINDINGS.items()):
                return False
        return (np.full is _FULL and np.isfinite is _FINITE and np.where is _WHERE and np.nan is _NAN
            and type(self.slot) is MemberDescriptorType
            and self.particle_type.__dict__.get('H') is self.slot
            and self.particle_type.__getattribute__ is object.__getattribute__)

    def mass(self, runtime, problem, weights, target):
        particles = runtime.particles
        if (self._ordinary() and type(particles) is list and type(problem) is self.problem_type
                and all(type(p) is self.particle_type for p in particles)
                and type(weights) is _ARRAY and weights.dtype == _F64 and weights.ndim == 1
                and type(target) is _ARRAY and target.dtype == _F64 and target.ndim == 1
                and weights.flags.c_contiguous and target.flags.c_contiguous
                and weights.size == len(particles) and target.size == len(problem.pairs)):
            result = self.module.component_mass([p.H for p in particles],problem.pairs,target,weights)
            if result is not None:
                self.native_calls += 1
                self.native_particles += len(particles)
                self.native_pairs += len(problem.pairs)
                self.peak_matrix_bytes = max(self.peak_matrix_bytes,8*len(particles)*len(problem.pairs))
                return result
        self.fallback_calls += 1
        return self.original(runtime,problem,weights,target)

    def counts(self):
        return dict(native_calls=self.native_calls,fallback_calls=self.fallback_calls,
            native_particles=self.native_particles,native_pairs=self.native_pairs)


class NativeComponentMassStep(NativeParentScoreStep):
    """D234 plus one literal-block substitution; original thresholds remain.

    DDs for every pair are expanded in original particle/pair order BEFORE any
    component sum, preserving conversion exceptions even for non-target pairs.
    Each sum starts at +0.0 and adds w[i] in original particle order. No tree
    reduction, epsilon, threshold/particle/candidate/RNG change or H/DD cache.
    """

    def __init__(self, runtime, scorer=None, mode='scalar', *, extension, parent_extension, mass_extension):
        super().__init__(runtime,scorer,mode,extension=extension,parent_extension=parent_extension)
        self.mass_engine = ComponentMassEngine(mass_extension,runtime._pair_dd,self._scope)
        if not self.mass_engine.literal:
            raise ValueError('Native component mass requires the frozen literal pair DD')
        compiled = FunctionType(_STEP_CODE,self._scope)

        def call(view,rtk,ctx):
            self._scope['_rbpf_component_mass'] = self.mass_engine.mass
            try:
                return compiled(view,rtk,ctx)
            finally:
                del self._scope['_rbpf_component_mass']

        self.compiled = call

    def __call__(self, rtk, ctx):
        before,previous = self.mass_engine.counts(),self.last
        try:
            return super().__call__(rtk,ctx)
        finally:
            if self.last is not previous and self.last is not None and 'batch' in self.last:
                self.last['component_mass'] = {k:v-before[k] for k,v in self.mass_engine.counts().items()}
