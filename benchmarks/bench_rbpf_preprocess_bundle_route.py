"""Whole RTK timing body, with the composed preprocessing factory selected."""

import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType

import bench_rbpf_qcedit_batch_route_checked as base
from rbpf_preprocess_bundle import make_type

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    "benchmarks/bench_rbpf_qcedit_batch_route_checked.py": "e9c3932c3fb6d364225790e909924773a145e2578bd2aac29024955bd9541005",
    "benchmarks/bench_rbpf_qcedit_batch_route.py": "ee873047ce8ceef5dd0a03974eb56c4cad569601f0b25e829cdd969c27114d65",
    "benchmarks/bench_rbpf_ambiguity_initx_route.py": "42de7482471d86f51ad93a090da23d96ccc4bcd5fce56f8c2a46628147f7c2df",
}


def main():
    for name, digest in FROZEN.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen process timing helper changed: " + name)
    engines = []

    def selected(original):
        cls, engine = make_type(original)
        engines.append(engine)
        return cls, engine

    original, proof = base.build_main()
    run = FunctionType(
        original.__code__,
        dict(original.__globals__, make_type=selected),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    code = run()
    assert code == 0
    for engine in engines:
        assert engine.qc.fast_calls == 800 and engine.qc.fallback_calls == 0
        book = engine.bookkeeping
        assert book.updates > 0 and book.original_updates == 0
        assert book.fast_scans == book.scans and book.literal_scans == 0
        assert (
            book.initx_engine.fast_calls > 0 and book.initx_engine.fallback_calls == 0
        )
    (output / "bundle.json").write_text(
        json.dumps(
            dict(
                variant_batch_selects="D269: frozen D247 ambiguity and D268 QC composition",
                engines=[e.details() for e in engines],
                unchanged_process_timing_body=True,
                parent_main_AST_proof=proof,
                only_factory_binding_changed=True,
                source_checks=FROZEN,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
