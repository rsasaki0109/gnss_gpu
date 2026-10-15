"""Observe retained nuisance inputs at actual likelihood misses in sampled WP16 steps.

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

import numpy as np

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
    selected = 0
    original_setup = runner._set_shared_filter_env

    def observe(rb, H, prob, values):
        state = active.get(id(rb))
        if state is None:
            return
        if "prob" not in state:
            state["prob"] = prob
            state["arrays"] = {k: np.array(getattr(prob, k), copy=True) for k in
                ["rows_J", "rows_v0", "rows_sig", "rows_iscp", "rows_pair",
                 "rows_lam", "rows_bmeas", "Qb", "y", "pos", "Ppos"]}
            state["config"] = {k: getattr(rb.cfg, k) for k in
                ["marg_uniform", "beta", "huber_c", "sig_group", "mix_eps", "mix_sigma_cp", "mix_sigma_pr"]}
        if state["prob"] is not prob:
            raise ValueError("multiple likelihood problems in one sampled step")
        i, arrays = len(state["calls"]), state["arrays"]
        cluster = values["state"]
        call = dict(H=encode(H), hasState=cluster is not None, groups=int(values["G"]))
        for name, key in [("iscp", "iscp"), ("pi", "pi_kept"), ("sig", "sig"), ("ll", "ll")]:
            arrays[f"capture_{name}_{i}"] = np.array(values[key], copy=True)
        if call["groups"]:
            arrays[f"capture_C_{i}"] = np.array(values["C"], copy=True)
        if cluster is not None:
            arrays[f"capture_pos_{i}"] = np.array(cluster["pos"], copy=True)
            arrays[f"capture_prior_{i}"] = np.array(cluster["P"], copy=True)
        state["calls"].append(call)

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
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_loglik")
        index = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == "cache[key]")
        method.body[index:index] = ast.parse("_observe_nuisance(self, H, prob, locals())").body
        namespace = dict(module.__dict__, _observe_nuisance=observe)
        observed = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        exec(compile(observed, str(args.source), "exec"), namespace)
        module.RbpfFgo._loglik = namespace["_loglik"]

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
                    path = args.output/f"{len(records):04d}-nuisance.npz"
                    with path.open("xb") as stream:
                        np.savez_compressed(stream, **state["arrays"])
                    records.append(dict(file=path.name, fingerprint=fingerprint(path), epoch=int(rtk.epoch2),
                        particles=len(rb.particles), calls=state["calls"], config=state["config"],
                        pairs=[[list(g), int(ref), int(sat)] for g, ref, sat in state["prob"].pairs]))
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
        inputs=inputs, libraries=libraries, numpy=np.__version__, argv=sys.argv, captureEvery=args.every,
        note="Actual cache misses, retained nuisance inputs/columns and resolved cluster states in original order/count. A callback immediately before cache storage snapshots original locals after all original arithmetic; no extra DD or cluster calls. Original return unchanged. No synthetic/expanded calls. Observer time is not speed evidence.")
    with (args.output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(f"Captured {sum(len(r['calls']) for r in records)} actual misses at {len(records)} epochs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
