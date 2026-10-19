"""Field-domain guards over the unchanged compiled QC numerical loop.

Accept exact small integer thresholds and the fork's plain string-format
cache on signal objects. Both are representation checks only; the original
objects, metadata-cache builder and numerical body remain in use.
"""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import rbpf_qcedit_batch as numeric
import rbpf_qcedit_batch_list as lists
from rbpf_qcedit_batch import (  # noqa: F401
    capture_input,
    capture_output,
    exact,
    restore,
)

BASE_SHA256 = "a2da097b60127b56983a48bad1a5f137ae08a13483ae8f28bb06cc08c498c4ca"


def _guard():
    path = Path(numeric.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen numerical implementation changed")
    cls = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.ClassDef) and n.name == "QcEngine"
    )
    node = deepcopy(
        next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "eligible"
        )
    )
    before = deepcopy(node)
    changes = [
        (
            "type(v) in (float, np.float64)",
            "type(v) in (float, np.float64) or (type(v) is int and v.bit_length() <= 53)",
        ),
        (
            "set(vars(s)) == {'sys', 'typ', 'sig'}",
            "set(vars(s)) in ({'sys', 'typ', 'sig'}, {'sys', 'typ', 'sig', '_str_cache'}) and ('_str_cache' not in vars(s) or type(s._str_cache) is str)",
        ),
    ]

    def replace(tree, old, new):
        target = ast.dump(ast.parse(old, mode="eval").body, include_attributes=False)
        value = ast.parse(new, mode="eval").body

        class Edit(ast.NodeTransformer):
            count = 0

            def visit(self, item):
                if ast.dump(item, include_attributes=False) == target:
                    self.count += 1
                    return ast.copy_location(deepcopy(value), item)
                return super().visit(item)

        edit = Edit()
        result = edit.visit(tree)
        assert edit.count == 1
        return result

    for old, new in changes:
        node = replace(node, old, new)
    restored = deepcopy(node)
    for old, new in reversed(changes):
        restored = replace(restored, new, old)
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        before, include_attributes=False
    )
    scope = dict(vars(numeric))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            __file__,
            "exec",
            dont_inherit=True,
        ),
        scope,
    )
    return scope["eligible"], dict(
        undo_AST_equal=True,
        explicit_guard_expression_changes=2,
        numerical_body_changed=False,
        original_source_sha256=BASE_SHA256,
    )


_eligible, GUARD_PROOF = _guard()


class QcEngine(lists.QcEngine):
    eligible = _eligible


def make_type(original):
    engine = QcEngine(original)

    class BatchedFieldQc(original):
        def qcedit(self, obs, rs, dts, svh, rr=None):
            return engine(self, obs, rs, dts, svh, rr)

    return BatchedFieldQc, engine
