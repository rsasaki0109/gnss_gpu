"""Complete RTK process comparison with the frozen original PF calculation."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import bench_rbpf_ambiguity_initx_route as base
from rbpf_qcedit_batch_field_schema import make_type


def build_main():
    path = Path(base.__file__)
    node = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    before = deepcopy(node)

    class Labels(ast.NodeTransformer):
        def __init__(self, undo=False):
            self.undo, self.count = undo, 0

        def visit_Constant(self, item):
            changes = {"bulk": "batch", "initx_counts": "qc_counts"}
            if self.undo:
                changes = {v: k for k, v in changes.items()}
            if type(item.value) is str and item.value in changes:
                self.count += 1
                return ast.copy_location(ast.Constant(changes[item.value]), item)
            return item

    edit = Labels()
    node = edit.visit(node)
    restored = Labels(undo=True).visit(deepcopy(node))
    assert edit.count == 3
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        before, include_attributes=False
    )
    scope = dict(base.main.__globals__, make_type=make_type)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            __file__,
            "exec",
            dont_inherit=True,
        ),
        scope,
    )
    return scope["main"], dict(
        undo_main_AST_equal=True,
        label_substitutions=3,
        parent_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        only_factory_binding_changed="field-domain QC factory",
    )


def main():
    run, _proof = build_main()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
