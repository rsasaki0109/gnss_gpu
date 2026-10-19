"""Memoize stable score indices after the unchanged canonical request.

All key requests, priors, particle probabilities and RNG draws still execute.
Only repeated hashing/lookup of an existing immutable key object is skipped.
"""

import ast
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

from rbpf_epoch_memo import _EpochMemo
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_phase_profile import _CODES, _METHODS, _ORIGINALS, _ORIGINAL_CODES


def _compile_raw():
    path = Path(__file__).with_name("rbpf_move_batch.py")
    source = path.read_bytes()
    if (
        hashlib.sha256(source).hexdigest()
        != "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea"
    ):
        raise ValueError("Frozen move caller changed")
    tree = ast.parse(source)
    klass = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MoveBatchStep"
    )
    fn = deepcopy(
        next(
            n
            for n in klass.body
            if isinstance(n, ast.FunctionDef) and n.name == "_moves"
        )
    )
    baseline = deepcopy(fn)
    (pos,) = [
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.FunctionDef) and n.name == "index"
    ]
    index = fn.body[pos]
    oldbody = deepcopy(index.body)
    assert ast.unparse(oldbody[0]) == "key = self.canonical(hypothesis)"
    assert ast.unparse(oldbody[-1]) == "return lookup[key]"
    by_key = ast.parse("def _original_key_index(key,hypothesis): pass").body[0]
    by_key.body = deepcopy(oldbody[1:])
    index.body = (
        index.body[:1]
        + ast.parse(
            "return _score_indices.get(key,hypothesis,_original_key_index)"
        ).body
    )
    fn.body[pos : pos + 1] = ast.parse(
        "_score_indices = ScoreIndexMemo(self.canonical)"
    ).body + [by_key, index]
    fn.body.insert(
        -1,
        ast.parse("self.last['score_index_memo'] = _score_indices.summary()").body[0],
    )
    restored = deepcopy(fn)
    restored.body = [
        n
        for n in restored.body
        if not (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and ast.unparse(n.targets[0])
            in ("_score_indices", "self.last['score_index_memo']")
        )
    ]
    restored.body = [
        n
        for n in restored.body
        if not (isinstance(n, ast.FunctionDef) and n.name == "_original_key_index")
    ]
    original_index = next(
        n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == "index"
    )
    original_index.body = deepcopy(oldbody)
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )
    module = compile(
        ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
        str(Path(__file__)),
        "exec",
        dont_inherit=True,
    )
    return next(c for c in module.co_consts if getattr(c, "co_name", None) == "_moves")


_RAW_CODE = _compile_raw()


def _plain_key(key):
    """Only immutable builtin tuples/integers: no custom hash/equality calls."""
    if type(key) is not tuple:
        return False
    stack = [key]
    while stack:
        for value in stack.pop():
            if type(value) is tuple:
                stack.append(value)
            elif type(value) is not int:
                return False
    return True


class ScoreIndexMemo:
    """One move block only; the original canonical callback still runs.

    Cache only keys already retained by the existing _EpochMemo or builtin
    empty key. Distinct equal keys still execute the original value lookup on
    first encounter. Retain each key object so ids cannot be recycled. Once
    any custom hash/equality key appears, disable aliases for the rest of the
    block and execute all original lookup operations. No probability cache.
    """

    def __init__(self, canonical):
        memo = getattr(canonical, "__self__", None)
        self.memo = (
            memo
            if type(memo) is _EpochMemo
            and _EpochMemo.__getattribute__ is object.__getattribute__
            and getattr(canonical, "__func__", None) is _EpochMemo.canonical
            else None
        )
        self.aliases = {}
        self.enabled = self.memo is not None
        self.requests = self.hits = self.source_calls = self.bypassed = 0

    def get(self, key, hypothesis, original):
        self.requests += 1
        ident = id(key)
        hit = self.aliases.get(ident)
        if self.enabled and hit is not None and hit[0] is key:
            self.hits += 1
            return hit[1]
        self.source_calls += 1
        try:
            result = original(key, hypothesis)
        except BaseException:
            self.enabled = False
            self.aliases.clear()
            raise
        # The original contains/get/set and any custom callbacks run before
        # cache eligibility checks on a new key. Preserve their error order.
        if not self.enabled or self.memo is None:
            self.bypassed += 1
            return result
        if not _plain_key(key):
            self.enabled = False
            self.aliases.clear()
            self.bypassed += 1
            return result
        memo = self.memo
        if type(memo.identities) is not dict:
            self.enabled = False
            self.aliases.clear()
            self.bypassed += 1
            return result
        entry = memo.identities.get(id(hypothesis))
        retained = (
            type(entry) is tuple
            and len(entry) == 2
            and entry[0] is hypothesis
            and entry[1] is key
        )
        if retained or (type(hypothesis) is dict and not hypothesis and key == ()):
            self.aliases[ident] = (key, result)
        else:
            self.bypassed += 1
        return result

    def summary(self):
        assert self.requests == self.hits + self.source_calls
        return dict(
            requests=self.requests,
            identity_hits=self.hits,
            source_index_calls=self.source_calls,
            bypassed=self.bypassed,
            entries=len(self.aliases),
            enabled=self.enabled,
            probability_cache=False,
        )


class ScoreIndexMemoStep(NativeFeedbackMassStep):
    """D236 plus index aliases; literal weights/choice and all keys stay.

    Reuse the three verified D239 adapter redirects without their raw/scorer
    timing code. No profiler is inherited and no span/timer is called. The
    original raw nested index tail executes for every miss; canonical is
    requested for every original move, keeping existing memo metadata exact.
    """

    def __init__(
        self, runtime, scorer=None, mode="native_cpu", *, index_extension, **extensions
    ):
        super().__init__(runtime, scorer, mode, **extensions)
        spec = importlib.util.spec_from_file_location(
            "_rbpf_score_index_probe", Path(index_extension).resolve()
        )
        if spec is None or spec.loader is None:
            raise ValueError("An explicit score-index extension is required")
        self.index_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.index_module)
        codes = (_RAW_CODE, *_CODES[1:4])
        self._index_clones = []
        for code, original in zip(codes, _ORIGINALS[:4], strict=True):
            clone = FunctionType(
                code,
                dict(original.__globals__),
                original.__name__,
                original.__defaults__,
                original.__closure__,
            )
            clone.__kwdefaults__ = original.__kwdefaults__
            self._index_clones.append(clone)

    def __call__(self, rtk, ctx):
        for (cls, name), original, code in zip(
            _METHODS[:4], _ORIGINALS[:4], _ORIGINAL_CODES[:4], strict=True
        ):
            if getattr(cls, name) is not original or original.__code__ is not code:
                raise ValueError(
                    "Score-index helper changed after binding: "
                    + cls.__name__
                    + "."
                    + name
                )
        additions = dict(
            ScoreIndexMemo=self._new_index_memo,
            _profile_raw_moves=self._index_clones[0],
            _profile_reuse_moves=self._index_clones[1],
            _profile_key_moves=self._index_clones[2],
        )
        for clone, original in zip(self._index_clones, _ORIGINALS[:4], strict=True):
            clone.__globals__.clear()
            clone.__globals__.update(original.__globals__)
            clone.__globals__.update(additions)
        return super().__call__(rtk, ctx)

    def _moves(self, *args, **kwargs):
        return self._index_clones[3](self, *args, **kwargs)

    def _new_index_memo(self, canonical):
        memo = getattr(canonical, "__self__", None)
        supported = (
            type(memo) is _EpochMemo
            and _EpochMemo.__getattribute__ is object.__getattribute__
            and getattr(canonical, "__func__", None) is _EpochMemo.canonical
            and type(memo.identities) is dict
        )
        return self.index_module.IndexMemo(memo.identities if supported else None)
