"""Split actual observation/IMU/motion work using coarse external spans."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import bench_rbpf_process_profile_field as base
from rbpf_rtk_work_profile import RtkWorkMonitor

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = "bef18a2bbe1003ca6f3213ae5b022faf01d3c9f5883a5600f53c5953d3023bd0"


def build_main():
    path = Path(base.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen coarse field helper changed")
    node = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    original = deepcopy(node)
    names = next(
        n.value
        for n in node.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "names"
    )
    extra = [
        "benchmarks/rbpf_rtk_work_profile.py",
        "benchmarks/bench_rbpf_rtk_work_profile_field.py",
    ]
    names.elts.extend(ast.Constant(n) for n in extra)
    restored = deepcopy(node)
    old_names = next(
        n.value
        for n in restored.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "names"
    )
    del old_names.elts[-len(extra) :]
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )
    scope = dict(base.main.__globals__, ProcessMonitor=RtkWorkMonitor)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
            dont_inherit=True,
        ),
        scope,
    )
    return scope["main"], dict(
        restored_parent_main_AST_equal=True,
        parent_main_source_sha256=BASE_SHA256,
        only_added_source_hashes=extra,
        monitor_binding="RtkWorkMonitor in cloned globals only",
    )


def main():
    run, _proof = build_main()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
