"""Exact batched cluster lookup with filesystem-free hot guard checks."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType, MethodType

import rbpf_cluster_batch as base
from rbpf_native_canonical import _code_key
from rbpf_register_lu import RegisterLUScorer

BASE_SHA256 = 'c0382b5f6bf05a4ec43d2252dd62d585b1ca92f39221b53cfd65cf59373848e4'
_FILENAMES = frozenset((str(base.SCALAR_SOURCE), base.SCALAR_SOURCE.as_posix()))


def _literal(function, name):
    # The frozen source SHA/code families were checked when base imported.
    # Unknown spellings fall back,including aliases formerly accepted by
    # Path.resolve. No model value,result or mutable binding is cached.
    return (type(function) is FunctionType and function.__defaults__ is None
        and function.__code__.co_filename in _FILENAMES
        and _code_key(function.__code__) in base._LITERAL_CODES[name])


def _guard_source():
    source = Path(base.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise ValueError('Frozen initial cluster wrapper changed')
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ClusterBatchEngine')
    return deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '_eligible'))


def _build():
    method = _guard_source()
    factory = ast.parse('def _factory(_literal): pass').body[0]
    factory.body = [method, ast.Return(ast.Name('_eligible', ast.Load()))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), str(Path(__file__)), 'exec'), namespace)
    make = FunctionType(namespace['_factory'].__code__, base.ClusterBatchEngine._eligible.__globals__)
    return make(_literal)


class ClusterBatchEngine(base.ClusterBatchEngine):
    """The entire guard AST stays literal;only _literal is a closure helper.

    All remaining names resolve in the same live original globals. All code,
    builtin/global/native-export/type/buffer/policy checks are retained. The
    supported filename spelling set is narrower;unknown forms run the original
    parameter block. Native selector,logical counts and NumPy packing stay
    unchanged from the preserved rejected initial implementation.
    """

    _eligible = _build()


class ClusterBatchScorer(RegisterLUScorer):
    def __init__(self, dll, extension, capacity=512, *, enabled=True):
        if type(enabled) is not bool:
            raise TypeError('An explicit bool control is required')
        engine = ClusterBatchEngine(extension)
        super().__init__(dll, capacity, kernel='register')
        self.cluster_engine, self.enabled = engine, enabled
        self.score = MethodType(base._FACTORY(engine.prepare) if enabled else base.original.BatchScorer.score, self)


def guard_AST_exact():
    method = _guard_source()
    tree = ast.parse(Path(base.__file__).read_bytes())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ClusterBatchEngine')
    expected = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '_eligible')
    return ast.dump(method, include_attributes=False) == ast.dump(expected, include_attributes=False)


# Exact reused parameter arithmetic and source undo for forwarded old tests.
_PLAIN = base._PLAIN
restored_source_ast = base.restored_source_ast
