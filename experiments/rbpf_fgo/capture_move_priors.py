"""Observe actual scalar move distributions inside sampled WP16 steps.

The callback follows all original arithmetic. Observer time is not speed evidence.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

from benchmark_feedback_dd import pack
from benchmark_move_selection import fingerprint


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
    selected = 0
    original_setup = runner._set_shared_filter_env

    def observe(rb, particle, moves, probabilities, values, before):
        state = active.get(id(rb))
        if state is not None:
            H = particle.H
            state["calls"].append(pack(dict(H=H, before=before,
                extended=values["H_ext"] if H else None, shrunk=values["H_shr"] if H else None,
                moves=[tuple(mv[:2]) for mv in moves], tags=[mv[3] for mv in moves],
                expected=(particle.logw, probabilities))))
        return probabilities

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
        original_step = module.RbpfFgo.step
        tree = ast.parse(args.source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
        loop = next(n for n in method.body if isinstance(n, ast.For) and ast.unparse(n.target) == "p"
                    and "cfg.track != 1" in ast.unparse(n.iter))
        start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "pri")
        loop.body[start:start] = ast.parse("_move_before = p.logw").body
        choices = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.rng.choice"]
        assert len(choices) == 1
        probability = next(k for k in choices[0].keywords if k.arg == "p")
        probability.value = ast.Call(func=ast.Name(id="_observe_move", ctx=ast.Load()),
            args=[ast.Name(id=name, ctx=ast.Load()) for name in ["self", "p", "moves"]]
                 + [probability.value, ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                    ast.Name(id="_move_before", ctx=ast.Load())], keywords=[])
        namespace = dict(module.__dict__, _observe_move=observe)
        observed = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        exec(compile(observed, str(args.source), "exec"), namespace)
        original_step = namespace["step"]

        def step(rb, rtk, context):
            nonlocal selected
            take = selected < args.max_captures and int(rtk.epoch2) % args.every == 0
            if take:
                selected += 1
                active[id(rb)] = dict(calls=[])
            try:
                result = original_step(rb, rtk, context)
                state = active.get(id(rb))
                if state is not None and state["calls"]:
                    path = args.output/f"{len(records):04d}-moves.json"
                    with path.open("x", encoding="utf-8") as stream:
                        json.dump(["list", state["calls"]], stream, separators=(",", ":"), allow_nan=False)
                    records.append(dict(file=path.name, fingerprint=fingerprint(path), epoch=int(rtk.epoch2),
                                        particles=len(rb.particles), calls=len(state["calls"])))
                return result
            finally:
                active.pop(id(rb), None)

        module.RbpfFgo.step = step

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
    if sources != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during capture")
    output = dict(schemaVersion=1, records=records, runtime=sources[0], sourceFiles=sources,
        inputs=inputs, libraries=libraries, argv=sys.argv, captureEvery=args.every,
        note="Actual scalar move inputs and original RNG probability arrays in original order/count. A callback wraps the existing probability argument and returns the same array unchanged; no added likelihood, structural move, probability division or RNG call. Observer time is not speed evidence. No synthetic/expanded inputs.")
    with (args.output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(f"Captured {sum(r['calls'] for r in records)} actual move distributions at {len(records)} epochs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
