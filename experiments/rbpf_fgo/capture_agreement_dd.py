"""Observe actual component-agreement phases in sampled WP16 steps.

The original phase statements run unchanged, with input/output callbacks.
Observer elapsed time is not speed evidence. Use fresh output directories.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from benchmark_agreement_dd import phase_bounds, result_digest
from benchmark_move_selection import encode, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["output", "source", "tc-src"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--every", type=int, default=16)
    parser.add_argument("--max-captures", type=int, default=24)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output.exists() or args.every < 1 or args.max_captures < 1:
        parser.error("new output and positive sampling settings required")
    if not args.runner_args or args.runner_args[0] != "--":
        parser.error("runner arguments must follow --")
    args.output.mkdir(parents=True)
    sys.dont_write_bytecode = True
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(args.tc_src.resolve())]
    import wp16_run_rbpf as runner
    paths = [args.source, *sorted(here.glob("*.py"))]
    sources = [fingerprint(p) for p in paths]
    records, active = [], {}
    original_setup = runner._set_shared_filter_env

    def observe(rb, values, before):
        epoch = int(values["ep"])
        if before:
            if len(records) >= args.max_captures or epoch % args.every:
                return
            prob = values["prob"]
            path = args.output/f"{len(records):04d}-agreement.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, weights=values["w"])
            active[id(rb)] = dict(file=path.name, fingerprint=fingerprint(path), epoch=epoch,
                particles=len(rb.particles), hypotheses=[encode(p.H) for p in rb.particles],
                ranked=values["ranked"], representatives=[[bid, encode(H)] for bid, H in values["rep"].items()],
                mapH=encode(values["map_H"]), config=dict(gamma_fix=rb.cfg.gamma_fix),
                pairs=[[list(g), int(ref), int(sat)] for g, ref, sat in prob.pairs])
        elif id(rb) in active:
            row = active.pop(id(rb))
            row["resultDigest"] = result_digest(values)
            records.append(row)

    def setup(options):
        original_setup(options)
        for item in options.env:
            key, value = item.split("=", 1)
            os.environ[key] = value
        if "rbpf_fgo" in sys.modules:
            raise RuntimeError("runtime imported before configuration")
        spec = importlib.util.spec_from_file_location("rbpf_fgo", args.source.resolve())
        module = importlib.util.module_from_spec(spec)
        sys.modules["rbpf_fgo"] = module
        spec.loader.exec_module(module)
        tree = ast.parse(args.source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
        start, end = phase_bounds(step)
        step.body.insert(end, ast.parse("_observe_agreement(self, locals(), False)").body[0])
        step.body.insert(start, ast.parse("_observe_agreement(self, locals(), True)").body[0])
        module.__dict__["_observe_agreement"] = observe
        compiled = ast.fix_missing_locations(ast.Module(body=[step], type_ignores=[]))
        namespace = dict(module.__dict__)
        exec(compile(compiled, str(args.source), "exec"), namespace)
        module.RbpfFgo.step = namespace["step"]

    runner._set_shared_filter_env = setup
    sys.argv = [str(here/"wp16_run_rbpf.py"), *args.runner_args[1:]]
    result = runner.main()
    if result:
        return result
    from wp13a_run_standalone_rtk import _DATA_ROOT
    import cssrlib.gnss
    import gtsam
    run = int(args.runner_args[1])
    inputs = [fingerprint(_DATA_ROOT/f"run{run}"/name) for name in
              ["rover.obs", "base.obs", "base.nav", "imu.csv", "reference.csv"]]
    libraries = [fingerprint(cssrlib.gnss.__file__), fingerprint(Path(gtsam.__file__).parent/"gtsam.pyd")]
    if sources != [fingerprint(p) for p in paths] or active:
        raise RuntimeError("source changed or phase capture incomplete")
    output = dict(schemaVersion=1, records=records, runtime=sources[0], sourceFiles=sources,
                  inputs=inputs, libraries=libraries, numpy=np.__version__, argv=sys.argv,
                  captureEvery=args.every,
                  note="Actual posterior component-agreement inputs, original order/count. Original statements unchanged with before/after callbacks; output digests include DD/mass/diagnostic bytes. No synthetic/expanded hypotheses. Observer elapsed time is not speed evidence.")
    with (args.output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(f"Captured {len(records)} actual component-agreement phases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
