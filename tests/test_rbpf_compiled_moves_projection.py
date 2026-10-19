"""The compiler input must retain each original numerical method AST."""

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks/probes/rbpf_compiled_moves/generate.py"
SPEC = importlib.util.spec_from_file_location("compiled_moves_projection", PATH)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.mark.parametrize("klass,filename,digest", generator.SPECS)
def test_every_original_statement_and_expression_is_preserved(klass, filename, digest):
    source, proof = generator.project()
    projected = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == klass
    )
    old = next(
        n
        for n in ast.parse((ROOT / "benchmarks" / filename).read_bytes()).body
        if isinstance(n, ast.ClassDef) and n.name == klass
    )
    original = next(
        n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "_moves"
    )
    assert ast.dump(projected.body[0], include_attributes=False) == ast.dump(
        original, include_attributes=False
    )
    assert (
        proof["method_AST_exact"]
        and proof["changed_numeric_expressions"] == proof["new_numeric_caches"] == 0
    )
    assert proof["source_sha256"][filename] == digest
    assert "infer_types=False" in source and "cdivision=False" in source
