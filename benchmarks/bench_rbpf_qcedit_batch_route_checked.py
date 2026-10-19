"""Preserve keyword dictionary labels when adapting the frozen RTK runner."""

import ast
from copy import deepcopy
from pathlib import Path

import bench_rbpf_qcedit_batch_route as rejected


def build_main():
    path = Path(rejected.__file__)
    node = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "build_main"
    )
    before = deepcopy(node)
    labels = next(
        n for n in node.body if isinstance(n, ast.ClassDef) and n.name == "Labels"
    )
    labels.body.append(
        ast.parse("""
def visit_keyword(self, item):
    old, new = ('qc_counts', 'initx_counts') if self.undo else ('initx_counts', 'qc_counts')
    if item.arg == old:
        self.count += 1
        return ast.copy_location(ast.keyword(arg=new, value=self.visit(item.value)), item)
    return self.generic_visit(item)
""").body[0]
    )
    restored = deepcopy(node)
    old_labels = next(
        n for n in restored.body if isinstance(n, ast.ClassDef) and n.name == "Labels"
    )
    old_labels.body.pop()
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        before, include_attributes=False
    )
    scope = dict(vars(rejected), __file__=__file__)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            __file__,
            "exec",
            dont_inherit=True,
        ),
        scope,
    )
    return scope["build_main"]()


def main():
    run, _proof = build_main()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
