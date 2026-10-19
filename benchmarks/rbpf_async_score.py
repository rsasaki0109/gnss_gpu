"""Owning begin/finish GPU transport, retaining the original score packing/math."""

import ast
import builtins
from copy import deepcopy
import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rbpf_likelihood_batch as original
from rbpf_register_lu import RegisterLUScorer

_BATCH_SHA = "7f35ae5b466145f07aa28a3de5b18f95121606b3a27a7bc85b2f9810c88bd663"
_NUMPY = {n: getattr(np, n) for n in ("ascontiguousarray", "column_stack", "empty", "diag",
    "asarray", "zeros", "all", "isfinite", "any", "abs", "array", "flatnonzero", "geterr", "ndarray", "float64", "int32")}
_BUILTINS = {n: getattr(builtins, n) for n in ("len", "type", "any", "getattr", "range", "int", "dict", "enumerate", "float")}


def _compile():
    path = Path(original.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != _BATCH_SHA:
        raise ValueError("Frozen score source changed")
    tree = ast.parse(path.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BatchScorer")
    base = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "score"))
    index = next(i for i, n in enumerate(base.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "code")
    native = deepcopy(base.body[index])
    assert ast.unparse(native.value.func) == "self.dll.rbpf_score"
    names = "hypotheses,problem,oracle,unique,scatter,diagnostics,output,status,diagnostic"
    pack = deepcopy(base)
    pack.name = "_packed_begin"
    pack.body = pack.body[:index] + [ast.Return(ast.Call(
        func=ast.parse("self._launch_bundle", mode="eval").body,
        args=[ast.Tuple(elts=deepcopy(native.value.args), ctx=ast.Load()),
              ast.parse("(" + names + ")", mode="eval").body], keywords=[]))]
    tail = ast.parse("def _finish_score(self," + names + ",code): pass").body[0]
    tail.body = deepcopy(base.body[index + 1:])
    restored = deepcopy(pack)
    restored.name = base.name
    restored.body = restored.body[:-1] + [native] + deepcopy(tail.body)
    assert ast.dump(restored, include_attributes=False) == ast.dump(base, include_attributes=False)
    scope = dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pack, tail], type_ignores=[])),
        __file__, "exec", dont_inherit=True), scope)
    return scope[pack.name], scope[tail.name], dict(original_score_AST_restored_exact=True,
        original_source_sha256=_BATCH_SHA, only_native_call_scheduling_split=True)


_PACK, _FINISH, AST_PROOF = _compile()


class AsyncScorer(RegisterLUScorer):
    """One exclusive workspace, no thread, no numerical/FP64/kernel change.

    Begin copies packed inputs to owned pinned staging before returning. A job
    retains every NumPy target until finish/drain. Finish runs the unchanged
    scalar-fallback/diagnostic/scatter tail on the caller thread. Blocked uses
    the same transport but waits in begin, providing a scheduling control.
    Foreign concurrent mutation/ABI misuse and allocation/CUDA failure
    injection are outside this private prototype's tested domain.
    """

    def __init__(self, dll, capacity=512, *, blocked=False):
        if type(blocked) is not bool:
            raise TypeError("blocked must be bool")
        self.async_handle = self.active = None
        self.blocked, self.poisoned = blocked, False
        self.counts = dict(begins=0, completions=0, drains=0, immediate=0, errors=0)
        super().__init__(dll, capacity)
        try:
            for name in ("create", "destroy", "finish", "drain"):
                function = getattr(self.dll, "rbpf_async_" + name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_void_p if name == "create" else (None if name == "destroy" else ctypes.c_int)
            self.dll.rbpf_async_begin.argtypes = list(self.dll.rbpf_score.argtypes)
            self.dll.rbpf_async_begin.restype = ctypes.c_int
            self.async_handle = self.dll.rbpf_async_create(self.handle)
            if not self.async_handle:
                raise RuntimeError("Async stream/pinned staging creation failed")
        except BaseException:
            self.close()
            raise

    def safe(self):
        return (type(self) is AsyncScorer and self.handle and self.async_handle and not self.poisoned
            and self.active is None and all(n not in vars(self) and getattr(type(self), n) is f
                and f.__code__ is code for n, (f, code) in _METHODS.items())
            and all(getattr(np, n) is f for n, f in _NUMPY.items())
            and all(_PACK.__globals__.get(n, _PACK.__builtins__.get(n)) is f for n, f in _BUILTINS.items()))

    def score(self, *args, **kwargs):
        if self.active is not None:
            raise RuntimeError("Finish/drain the owned GPU job before reusing its workspace")
        return super().score(*args, **kwargs)

    def begin(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False):
        if not self.handle or not self.async_handle or self.poisoned:
            raise RuntimeError("Closed or poisoned async scorer")
        if self.active is not None:
            raise RuntimeError("Only one owned GPU job may be outstanding")
        if mode != "gpu":
            value = self.score(hypotheses, problem, oracle, mode, diagnostics)
        else:
            value = _PACK(self, hypotheses, problem, oracle, mode, diagnostics)
        if type(value) is tuple:
            self.counts["immediate"] += 1
            return SimpleNamespace(owner=self, immediate=value, closed=False)
        return value

    def _launch_bundle(self, arguments, bundle):
        code = self.dll.rbpf_async_begin(self.async_handle, *arguments[1:])
        self.counts["begins"] += 1
        self.counts["errors"] += int(code != 0)
        job = self.active = SimpleNamespace(owner=self, arguments=arguments, bundle=bundle,
            code=None if code == 0 else code, closed=False, immediate=None)
        if self.blocked:
            self._wait(job)
        return job

    def _wait(self, job):
        if job.code is None:
            job.code = self.dll.rbpf_async_finish(self.async_handle)
            self.counts["completions"] += 1
            self.counts["errors"] += int(job.code != 0)
        return job.code

    def finish(self, job):
        if job.owner is not self or job.closed:
            raise ValueError("Foreign or completed GPU job")
        if job.immediate is not None:
            job.closed = True
            return job.immediate
        if job is not self.active:
            raise ValueError("GPU job is not active")
        try:
            code = self._wait(job)
            return _FINISH(self, *job.bundle, code)
        finally:
            job.closed = True
            job.arguments = job.bundle = None
            self.active = None

    def cancel(self, job=None):
        job = self.active if job is None else job
        if job is None or (job.owner is self and job.closed):
            return 0
        if job.owner is self and job.immediate is not None:
            job.closed = True
            return 0
        if job.owner is not self or job is not self.active:
            raise ValueError("Foreign GPU job")
        code = 0
        if job.code is None:
            code = self.dll.rbpf_async_drain(self.async_handle)
            self.counts["drains"] += 1
            self.counts["errors"] += int(code != 0)
            self.poisoned = bool(code)
        job.closed = True
        job.arguments = job.bundle = None
        self.active = None
        return code

    def close(self):
        try:
            self.cancel()
            if self.async_handle:
                self.dll.rbpf_async_destroy(self.async_handle)
                self.async_handle = None
        finally:
            super().close()


_METHODS = {n: (getattr(AsyncScorer, n), getattr(AsyncScorer, n).__code__)
    for n in ("score", "begin", "_launch_bundle", "_wait", "finish", "cancel", "safe")}
