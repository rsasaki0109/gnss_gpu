"""Exact native release-subset row selection on the explicit D236 caller."""

import ast
import builtins
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

import numpy as np

from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256


def _compile_codes():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError('Frozen RBPF source changed')
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'RbpfFgo')
    method = deepcopy(next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name == '_shrink_worst'))
    baseline = deepcopy(method)
    original = deepcopy(method.body[1:4])
    assert ast.unparse(original[0]) == 'dd = self._pair_dd(H, prob)'
    assert ast.unparse(original[1]) == 'worst_pi, worst_r = (-1, -1.0)'
    assert ast.unparse(original[2].iter) == 'range(len(prob.rows_v0))'
    assert ast.unparse(original[2].body[-1].test) == 'wr > worst_r'
    raw = ast.parse('def _original_worst(self, H, prob): pass').body[0]
    raw.body = deepcopy(original)+ast.parse('return worst_pi').body
    method.body[1:4] = ast.parse('worst_pi = _rbpf_worst_index(self, H, prob)').body
    restored = deepcopy(method)
    restored.body[1:2] = deepcopy(original)
    assert ast.dump(restored,include_attributes=False) == ast.dump(baseline,include_attributes=False)
    module = compile(ast.fix_missing_locations(ast.Module(body=[raw,method],type_ignores=[])),
        str(Path(__file__)),'exec',dont_inherit=True)
    return {c.co_name:c for c in module.co_consts if hasattr(c,'co_name')}


_CODES = _compile_codes()
_BINDINGS = {n:getattr(builtins,n) for n in ('range','len','abs','max')}
_ARRAY, _F64 = np.ndarray,np.dtype('float64')
_FIELDS = ('pairs','rows_v0','rows_iscp','rows_pair','rows_lam','rows_bmeas','rows_sig')


class ShrinkEngine:
    """Select the same first strict maximum, with original Python fallback.

    Only ordinary source objects, exact arrays/plain integer H and finite
    bounded CP arithmetic use C++. The bounds prevent overflow/invalid NumPy
    warnings; outside them the original block runs once. Underflow remains
    ignored as in the existing explicit caller. No H/problem cache is kept.
    """

    def __init__(self, extension, component):
        self.component,self.scope = component,component.scope
        spec = importlib.util.spec_from_file_location('_rbpf_shrink_probe',Path(extension).resolve())
        if spec is None or spec.loader is None:
            raise ValueError('An explicit shrink extension is required')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.original = FunctionType(_CODES['_original_worst'],self.scope)
        self.native_calls = self.fallback_calls = self.native_rows = self.native_pairs = 0

    def index(self, runtime, hypothesis, problem):
        component = self.component
        ordinary = (component._ordinary() and type(problem) is component.problem_type
            and component.problem_type.__getattribute__ is object.__getattribute__
            and all(component.problem_type.__dict__.get(n) is None and n in vars(problem) for n in _FIELDS)
            and all(self.scope.get(k,self.original.__builtins__.get(k)) is v for k,v in _BINDINGS.items()))
        policy = np.geterr()
        ordinary = ordinary and policy['under'] == 'ignore' and all(v in ('ignore','warn') for v in policy.values())
        if ordinary:
            values = [getattr(problem,n) for n in _FIELDS[1:]]
            ordinary = all(type(v) is _ARRAY and v.ndim == 1 and v.flags.c_contiguous for v in values)
            ordinary = ordinary and all(values[i].dtype == _F64 for i in (0,3,4,5))
            ordinary = ordinary and values[1].dtype == np.dtype('bool') and values[2].dtype in (np.dtype('int32'),np.dtype('int64'))
            if ordinary:
                result = self.module.worst_index(hypothesis,problem.pairs,*values)
                if result is not None:
                    self.native_calls += 1
                    self.native_rows += len(problem.rows_v0)
                    self.native_pairs += len(problem.pairs)
                    return result
        self.fallback_calls += 1
        return self.original(runtime,hypothesis,problem)

    def counts(self):
        return dict(native_calls=self.native_calls,fallback_calls=self.fallback_calls,
            native_rows=self.native_rows,native_pairs=self.native_pairs)


class NativeShrinkStep(NativeFeedbackMassStep):
    """D236 plus exact DD/CP worst-row selection; copy/drop tail unchanged."""

    def __init__(self, runtime, scorer=None, mode='scalar', *, shrink_extension, **extensions):
        super().__init__(runtime,scorer,mode,**extensions)
        self.shrink_engine = ShrinkEngine(shrink_extension,self.mass_engine)
        original = self._source_functions['_shrink_worst']
        replacement = FunctionType(_CODES['_shrink_worst'],self._scope,original.__name__,
            original.__defaults__,original.__closure__)
        replacement.__kwdefaults__ = original.__kwdefaults__
        self._functions['_shrink_worst'] = replacement
        base = self.compiled

        def call(view,rtk,ctx):
            self._scope['_rbpf_worst_index'] = self.shrink_engine.index
            try:
                return base(view,rtk,ctx)
            finally:
                del self._scope['_rbpf_worst_index']

        self.compiled = call

    def __call__(self, rtk, ctx):
        before,previous = self.shrink_engine.counts(),self.last
        try:
            return super().__call__(rtk,ctx)
        finally:
            if self.last is not previous and self.last is not None and 'batch' in self.last:
                self.last['shrink'] = {k:v-before[k] for k,v in self.shrink_engine.counts().items()}
