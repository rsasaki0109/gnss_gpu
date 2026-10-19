"""Frozen complete-RTK timing body; bulk variant now selects D247 combined.

The D246 helper/globals remain unchanged. Outer plans pin this adapter too.
"""

import json
from pathlib import Path
import sys
from types import FunctionType

import bench_rbpf_ambiguity_initx_route as base
from rbpf_ambiguity_bookkeeping import make_type


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    engines = []

    def selected(original):
        chosen, engine = make_type(original)
        engines.append(engine)
        return chosen, engine

    run = FunctionType(base.main.__code__, dict(vars(base), make_type=selected),
        base.main.__name__, base.main.__defaults__, base.main.__closure__)
    code = run()
    (output / "bookkeeping.json").write_text(json.dumps(dict(
        variant_bulk_selects="D247 batch integer bookkeeping plus D246 checked initx",
        engines=[dict(counts=e.scan_counts(), AST_proof=e.proof) for e in engines]),
        indent=2) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
