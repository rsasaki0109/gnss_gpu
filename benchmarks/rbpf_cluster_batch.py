"""Explicit batched exact cluster selection with an unchanged numerical DLL."""

import ast
import builtins
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import BuiltinFunctionType, FunctionType, MethodType, SimpleNamespace

import numpy as np

import rbpf_likelihood_batch as original
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256, make_oracle
from rbpf_native_canonical import CanonicalEngine, _CanonicalView, _code_key
from rbpf_parent_score import ParentScoreEngine
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]
BATCH_SHA256 = '7f35ae5b466145f07aa28a3de5b18f95121606b3a27a7bc85b2f9810c88bd663'
_PREPARE = '__rbpf_cluster_prepare_20261017'
_STATES = '__rbpf_cluster_states_20261017'
_BUILTINS = {name: getattr(builtins, name) for name in ('sum', 'max', 'len', 'set', 'sorted', 'min', 'tuple', 'enumerate', 'float')}
_NUMPY = {name: getattr(np, name) for name in ('empty', 'zeros', 'asarray', 'all', 'isfinite', 'abs', 'geterr', 'ndarray')}
_F64 = np.dtype('float64')
_CANONICAL_CALL, _PARENT_CALL = CanonicalEngine.canonical, ParentScoreEngine.score
_CANONICAL_CALL_CODE, _PARENT_CALL_CODE = _CANONICAL_CALL.__code__, _PARENT_CALL.__code__


def _codes(code):
    yield code
    for child in code.co_consts:
        if hasattr(child, 'co_code'):
            yield from _codes(child)


def _literal_codes():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError('Frozen scalar source changed')
    compiled = compile(source, str(SCALAR_SOURCE), 'exec', dont_inherit=True)
    raw = make_oracle()
    accepted = {name: {_code_key(c) for c in _codes(compiled) if c.co_name == name}
        for name in ('_cluster_state', '_canon_basin_id', '_basin_parent_score')}
    accepted['_cluster_state'].add(_code_key(raw._cluster_state.__func__.__code__))
    accepted['_canon_basin_id'].add(_code_key(raw.basin_id.__code__))
    accepted['_basin_parent_score'].add(_code_key(raw._basin_parent_score.__code__))
    wrappers = set()
    for name, digest in (
        ('rbpf_irls_replay.py', 'fc4f178520685f9ed3aee74b611c9021331b8d93f24da2634e89cd8594e11d02'),
        ('rbpf_parent_score.py', 'e888a4244519242e7dd88cf047843a0fcb6cb11f07c4f5aaea9948b005b8ae9d')):
        path = ROOT / 'benchmarks' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen parent wrapper changed')
        wrappers.update(_code_key(c) for c in _codes(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True))
            if c.co_name == 'parent_score')
    return accepted, wrappers


_LITERAL_CODES, _WRAPPER_CODES = _literal_codes()


def _source():
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != BATCH_SHA256:
        raise ValueError('Frozen score source changed')
    tree = ast.parse(Path(original.__file__).read_bytes())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchScorer')
    method = deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'score'))
    start, = [i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parameters']
    block = deepcopy(method.body[start:start+2])
    assert isinstance(block[1], ast.For) and ast.unparse(block[1].iter) == 'enumerate(unique)'
    return method, start, block


def _compiled():
    method, start, block = _source()
    replacement = ast.parse(f'parameters = {_PREPARE}(unique, problem, oracle)').body
    method.body[start:start+2] = replacement
    restored = deepcopy(method)
    restored.body[start:start+1] = deepcopy(block)
    assert ast.dump(restored, include_attributes=False) == ast.dump(_source()[0], include_attributes=False)
    factory = ast.parse(f'def _factory({_PREPARE}): pass').body[0]
    factory.body = [method, ast.Return(ast.Name('score', ast.Load()))]
    plain = ast.parse('def _plain(unique, problem, oracle): pass').body[0]
    plain.body = deepcopy(block) + ast.parse('return parameters').body
    packed = ast.parse(f'def _packed(unique, problem, oracle, {_STATES}): pass').body[0]
    packed.body = deepcopy(plain.body)
    assignments = [n for n in ast.walk(packed) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'state']
    assert len(assignments) == 1 and ast.unparse(assignments[0].value) == 'oracle._cluster_state(hypothesis)'
    assignments[0].value = ast.parse(f'{_STATES}[i]', mode='eval').body
    undo = deepcopy(packed)
    undo.name, undo.args = plain.name, deepcopy(plain.args)
    next(n for n in ast.walk(undo) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'state').value = deepcopy(
        next(n for n in ast.walk(plain) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'state').value)
    assert ast.dump(undo, include_attributes=False) == ast.dump(plain, include_attributes=False)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory, plain, packed], type_ignores=[])),
        str(Path(__file__)), 'exec'), namespace)
    scope = original.BatchScorer.score.__globals__
    return (FunctionType(namespace['_factory'].__code__, scope),
        FunctionType(namespace['_plain'].__code__, scope), FunctionType(namespace['_packed'].__code__, scope))


_FACTORY, _PLAIN, _PACKED = _compiled()


def restored_source_ast():
    method, start, block = _source()
    altered = deepcopy(method)
    altered.body[start:start+2] = ast.parse(f'parameters = {_PREPARE}(unique, problem, oracle)').body
    altered.body[start:start+1] = deepcopy(block)
    return ast.dump(method, include_attributes=False), ast.dump(altered, include_attributes=False)


def _bindings(function, names):
    return all(function.__globals__.get(name, function.__builtins__.get(name)) is _BUILTINS[name] for name in names)


def _literal(function, name):
    return (type(function) is FunctionType and function.__defaults__ is None
        and Path(function.__code__.co_filename).resolve() == SCALAR_SOURCE.resolve()
        and _code_key(function.__code__) in _LITERAL_CODES[name])


def _ordinary_fields(value, names):
    klass = type(value)
    if klass is not SimpleNamespace and (klass.__getattribute__ is not object.__getattribute__
            or klass.__setattr__ is not object.__setattr__):
        return False
    if any(name in vars(base) for base in klass.__mro__ for name in names):
        return False
    try:
        fields = vars(value)
    except TypeError:
        return False
    return type(fields) is dict and all(name in fields for name in names)


def _native_export(module, name):
    function = getattr(module, name, None)
    return (type(function) is BuiltinFunctionType and function.__self__ is module
        and function.__name__ == name)


def _array(value, shape):
    return type(value) is _NUMPY['ndarray'] and value.dtype == _F64 and value.shape == shape


def _bounded_position(value):
    return _array(value, (3,)) and np.all(np.isfinite(value)) and np.all(np.abs(value) <= 1e100)


class ClusterBatchEngine:
    """Batch guarded ordinary lookups; retain arbitrary ints and Python division.

    Custom bindings, callbacks, objects, policies or unsafe centre arithmetic
    run the original per-hypothesis block once. No retained H/state/key cache.
    Existing canonical/parent counters count logical requests; batch_calls
    separately counts new C API invocations. State objects keep their identity.
    """

    def __init__(self, extension):
        if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12):
            raise RuntimeError('Explicit CPython 3.12 cluster probe required')
        path = Path(extension).resolve()
        spec = importlib.util.spec_from_file_location('_rbpf_cluster_batch_probe', path)
        if spec is None or spec.loader is None:
            raise ValueError('An explicit cluster extension is required')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.select = self.module.select_states
        self.batch_calls = self.fallback_calls = self.hypotheses = 0
        self.canonical_requests = self.parent_requests = self.exact_hits = self.selected_parents = self.ties = 0

    def _eligible(self, problem, oracle):
        if type(oracle) is not SimpleNamespace or not _ordinary_fields(oracle, ('cfg', '_cluster_state')):
            return None
        cfg = oracle.cfg
        names = ('cluster_relin', 'cluster_parent_min', 'cluster_parent_frac')
        if not _ordinary_fields(cfg, names) or type(cfg.cluster_relin) not in (int, bool) or not cfg.cluster_relin:
            return None
        if type(cfg.cluster_parent_min) is not int or cfg.cluster_parent_min <= 0 or type(cfg.cluster_parent_frac) is not float:
            return None
        method = oracle._cluster_state
        if type(method) is not MethodType or type(method.__self__) is not _CanonicalView:
            return None
        view, function = method.__self__, method.__func__
        if not _literal(function, '_cluster_state') or not _bindings(function, ('sum', 'max', 'len')):
            return None
        actual = view._actual
        if not _ordinary_fields(actual, ('cfg', 'cluster_states')) or actual.cfg is not cfg or type(actual.cluster_states) is not dict:
            return None
        canonical = function.__globals__.get('_canon_basin_id')
        if (type(canonical) is not MethodType or type(canonical.__self__) is not CanonicalEngine
                or canonical.__func__ is not _CANONICAL_CALL or _CANONICAL_CALL.__code__ is not _CANONICAL_CALL_CODE
                or CanonicalEngine.canonical is not _CANONICAL_CALL):
            return None
        ce = canonical.__self__
        if (not _ordinary_fields(ce, ('literal', 'original', 'code', 'module', 'native_calls', 'fallback_calls'))
                or not ce.literal or not _literal(ce.original, '_canon_basin_id') or ce.original.__code__ is not ce.code
                or not _bindings(ce.original, ('sorted', 'min', 'tuple')) or not _native_export(ce.module, 'canonical')):
            return None
        wrapper = view._functions.get('_basin_parent_score')
        if (type(wrapper) is not FunctionType or _code_key(wrapper.__code__) not in _WRAPPER_CODES
                or wrapper.__closure__ is None or len(wrapper.__closure__) != 1):
            return None
        pe = wrapper.__closure__[0].cell_contents
        bound = view._bound.get('_basin_parent_score')
        if bound is not None and (type(bound) is not MethodType or bound.__self__ is not view or bound.__func__ is not wrapper):
            return None
        if (type(pe) is not ParentScoreEngine
                or not _ordinary_fields(pe, ('literal', 'original', 'code', 'module', 'native_calls', 'fallback_calls'))
                or 'score' in vars(pe) or ParentScoreEngine.score is not _PARENT_CALL
                or _PARENT_CALL.__code__ is not _PARENT_CALL_CODE or not pe.literal
                or not _literal(pe.original, '_basin_parent_score') or pe.original.__code__ is not pe.code
                or not _bindings(pe.original, ('set', 'sorted', 'len')) or not _native_export(pe.module, 'score')):
            return None
        if any(type(getattr(engine, name)) is not int for engine in (ce, pe) for name in ('native_calls', 'fallback_calls')):
            return None
        if (any(getattr(np, name) is not value for name, value in _NUMPY.items())
                or _PLAIN.__globals__.get('np') is not np or not _bindings(_PLAIN, ('len', 'enumerate', 'float'))
                or any(v not in ('ignore', 'warn') for v in np.geterr().values()) or np.geterr()['under'] != 'ignore'
                or not _ordinary_fields(problem, ('pos', 'Ppos')) or not _bounded_position(problem.pos)
                or not _array(problem.Ppos, (3, 3)) or self.module.select_states is not self.select):
            return None
        for state in actual.cluster_states.values():
            if (type(state) is not dict or any(type(k) is not str for k in state)
                    or 'pos' not in state or 'P' not in state or not _bounded_position(state['pos'])
                    or not _array(state['P'], (3, 3))):
                return None
        return actual.cluster_states, cfg.cluster_parent_min, cfg.cluster_parent_frac, ce, pe

    def prepare(self, unique, problem, oracle):
        eligible = self._eligible(problem, oracle)
        if eligible is not None:
            states, minimum, fraction, ce, pe = eligible
            result = self.select(unique, states, minimum, fraction)
            if result is not None:
                selected, canonical, parents, exact, chosen, ties = result
                # Preserve existing diagnostic request counts without claiming
                # they are individual C API calls in the batched implementation.
                ce.native_calls += canonical
                pe.native_calls += parents
                self.batch_calls += 1
                self.hypotheses += len(unique)
                self.canonical_requests += canonical
                self.parent_requests += parents
                self.exact_hits += exact
                self.selected_parents += chosen
                self.ties += ties
                return _PACKED(unique, problem, oracle, selected)
        self.fallback_calls += 1
        return _PLAIN(unique, problem, oracle)

    def counts(self):
        return {name: getattr(self, name) for name in ('batch_calls', 'fallback_calls', 'hypotheses',
            'canonical_requests', 'parent_requests', 'exact_hits', 'selected_parents', 'ties')}


class ClusterBatchScorer(RegisterLUScorer):
    """Replace only parameter-state lookups; retain all numerical score code."""

    def __init__(self, dll, extension, capacity=512, *, enabled=True):
        if type(enabled) is not bool:
            raise TypeError('An explicit bool control is required')
        engine = ClusterBatchEngine(extension)
        super().__init__(dll, capacity, kernel='register')
        self.cluster_engine, self.enabled = engine, enabled
        self.score = MethodType(_FACTORY(engine.prepare) if enabled else original.BatchScorer.score, self)
