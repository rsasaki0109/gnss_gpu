"""Guard against reintroducing silent broad exception suppression."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python" / "gnss_gpu"


def _exception_name(handler: ast.ExceptHandler) -> str | None:
    if handler.type is None:
        return None
    if isinstance(handler.type, ast.Name):
        return handler.type.id
    if isinstance(handler.type, ast.Attribute):
        return handler.type.attr
    return ""


def test_no_bare_or_silent_broad_exception_handlers():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            name = _exception_name(node)
            is_silent_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
            if name is None or (name in {"Exception", "BaseException"} and is_silent_pass):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == [], "silent broad handlers: " + ", ".join(violations)
