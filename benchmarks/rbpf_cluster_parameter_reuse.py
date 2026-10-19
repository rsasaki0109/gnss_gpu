"""Exact call-local parameter-row reuse after guarded batched state selection."""

import ast
import builtins
from copy import deepcopy
import hashlib
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, MethodType

import rbpf_cluster_batch as initial
import rbpf_cluster_fast_guard as base
from rbpf_register_lu import RegisterLUScorer

BASE_SHA256 = 'de3601976146125b0c0705016afd72474085f7a5ad3f7ffcd5e7ef4351cb738f'
if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA256:
    raise ValueError('Frozen fast-guard wrapper changed')

# This is a new implementation primitive, not an original score global.
# Capture the genuine intrinsic once; state references stay alive in selected.
_identity = builtins.id
if (type(_identity) is not BuiltinFunctionType or _identity.__self__ is not builtins
        or _identity.__name__ != 'id'):
    raise ValueError('The original identity intrinsic is required')


def _packing_source():
    _, _, block = initial._source()
    packed = ast.parse(f'def _packed(unique, problem, oracle, {initial._STATES}): pass').body[0]
    packed.body = deepcopy(block) + ast.parse('return parameters').body
    loop = packed.body[1]
    state, = [n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'state']
    assert ast.unparse(state.value) == 'oracle._cluster_state(hypothesis)'
    state.value = ast.parse(f'{initial._STATES}[i]', mode='eval').body
    expected = deepcopy(packed)
    original_body = deepcopy(loop.body)
    # Misses execute every original arithmetic/assignment expression verbatim.
    # Hits only copy an already completed float64 row. No value-based merging.
    prefix = ast.parse('''key = _identity(state)
row = rows.get(key)
if row is not None:
    parameters[i] = parameters[row]
    continue
''').body
    miss = [n for n in loop.body if n is not state]
    loop.body = [deepcopy(state), *prefix, *miss, *ast.parse('rows[key] = i').body]
    packed.body.insert(1, ast.parse('rows = {}').body[0])
    undo = deepcopy(packed)
    del undo.body[1]
    undo.body[1].body = original_body
    assert ast.dump(undo, include_attributes=False) == ast.dump(expected, include_attributes=False)
    return packed, expected, undo


def _build_packing():
    packed, _, _ = _packing_source()
    factory = ast.parse('def _factory(_identity): pass').body[0]
    factory.body = [packed, ast.Return(ast.Name('_packed', ast.Load()))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), str(Path(__file__)), 'exec'), namespace)
    make = FunctionType(namespace['_factory'].__code__, initial._PACKED.__globals__)
    return make(_identity)


_PACKED = _build_packing()


def _prepare_source():
    tree = ast.parse(Path(initial.__file__).read_bytes())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ClusterBatchEngine')
    return deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare'))


def _build_prepare():
    factory = ast.parse('def _factory(_PACKED): pass').body[0]
    factory.body = [_prepare_source(), ast.Return(ast.Name('prepare', ast.Load()))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), str(Path(__file__)), 'exec'), namespace)
    make = FunctionType(namespace['_factory'].__code__, initial.ClusterBatchEngine.prepare.__globals__)
    return make(_PACKED)


class ClusterBatchEngine(base.ClusterBatchEngine):
    """Keep the entire eligibility/selection/counter/fallback protocol unchanged.

    The row dictionary exists only during packing and contains integer identities
    and first row indices. The owned selected list keeps state objects alive.
    No hypothesis/state/row survives the call through this helper.
    """

    prepare = _build_prepare()


class ClusterBatchScorer(RegisterLUScorer):
    def __init__(self, dll, extension, capacity=512, *, enabled=True):
        if type(enabled) is not bool:
            raise TypeError('An explicit bool control is required')
        engine = ClusterBatchEngine(extension)
        super().__init__(dll, capacity, kernel='register')
        self.cluster_engine, self.enabled = engine, enabled
        self.score = MethodType(initial._FACTORY(engine.prepare) if enabled else initial.original.BatchScorer.score, self)


def packing_AST_exact():
    _, expected, restored = _packing_source()
    return ast.dump(expected, include_attributes=False) == ast.dump(restored, include_attributes=False)


# Forward the unchanged fallback and guard proof for regression coverage.
_PLAIN = initial._PLAIN
_literal = base._literal
guard_AST_exact = base.guard_AST_exact
restored_source_ast = initial.restored_source_ast
