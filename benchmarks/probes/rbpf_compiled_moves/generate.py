"""Project four literal D236 move methods into a Cython compilation unit."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPECS = (
    (
        "MoveBatchStep",
        "rbpf_move_batch.py",
        "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea",
    ),
    (
        "MoveReuseStep",
        "rbpf_move_reuse.py",
        "9d1c35239afcbac4c4ed17494c5072cb2fe793a24e8dd3548d5c85b2933b4612",
    ),
    (
        "MoveKeyMemoStep",
        "rbpf_move_key_memo.py",
        "ddcc139b41212bfe813d552f6921b2914b6452571be410e2783a65f0d9cf9632",
    ),
    (
        "NativeCanonicalStep",
        "rbpf_native_canonical.py",
        "f177e3247617cca70e4aa0c4369f1856dba979703f478d62c1f05cd814c63bd2",
    ),
)
DIRECTIVES = "language_level=3, infer_types=False, binding=True, cdivision=False, boundscheck=True, wraparound=True, annotation_typing=False, profile=False, linetrace=False"


def project(root=ROOT):
    tree = ast.parse(
        "import math\nimport numpy as np\nfrom types import SimpleNamespace\nfrom rbpf_epoch_memo import _EpochMemo\nfrom rbpf_move_reuse import _MoveView\n"
    )
    methods, hashes = {}, {}
    for klass, filename, expected in SPECS:
        source = (root / "benchmarks" / filename).read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + filename)
        oldclass = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == klass
        )
        method = deepcopy(
            next(
                n
                for n in oldclass.body
                if isinstance(n, ast.FunctionDef) and n.name == "_moves"
            )
        )
        methods[klass] = ast.dump(method, include_attributes=False)
        # These classes are namespaces only: original methods are called with
        # the original controller/context, never constructed from these types.
        newclass = ast.ClassDef(
            name=klass,
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
            type_params=[],
        )
        tree.body.append(newclass)
        hashes[filename] = expected
    tree.body += ast.parse("SOURCE_METHOD_SHA256 = " + repr(hashes)).body
    result = (
        "# cython: "
        + DIRECTIVES
        + "\n# Generated from unchanged method ASTs; do not edit after numerical execution.\n\n"
        + ast.unparse(ast.fix_missing_locations(tree))
        + "\n"
    )
    generated = ast.parse(result)
    for node in generated.body:
        if isinstance(node, ast.ClassDef):
            assert (
                ast.dump(node.body[0], include_attributes=False) == methods[node.name]
            )
    return result, dict(
        source_sha256=hashes,
        method_AST_exact=True,
        changed_numeric_expressions=0,
        new_numeric_caches=0,
        directives=DIRECTIVES,
    )


if __name__ == "__main__":
    output = Path(__file__).with_name("_rbpf_compiled_moves.pyx")
    if output.exists():
        raise FileExistsError(output)
    text, proof = project()
    output.write_bytes(text.replace("\n", "\r\n").encode("utf8"))
    import json

    print(json.dumps(proof, indent=2))
