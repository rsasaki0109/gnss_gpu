"""Record original extension/release inputs and outputs from a real WP16 run.

Example: python capture_move_inputs.py --output NEW --source BASELINE.py
         --tc-src tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]
The observer returns original results and its run time is not speed evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from benchmark_move_selection import encode, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tc-src", type=Path, required=True)
    parser.add_argument("--every", type=int, default=16)
    parser.add_argument("--max-captures", type=int, default=24)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output.exists() or args.every < 1 or args.max_captures < 1:
        parser.error("new output and positive sampling settings required")
    if not args.runner_args or args.runner_args[0] != "--":
        parser.error("supply runner arguments after --")
    args.output.mkdir(parents=True)
    sys.dont_write_bytecode = True
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(args.tc_src.resolve())]
    import wp16_run_rbpf as runner
    paths = [args.source, *sorted(here.glob("*.py"))]
    sources = [fingerprint(p) for p in paths]
    records, active = [], {}
    original_setup = runner._set_shared_filter_env
    selected = 0

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

        def observe(name, original):
            def call(rb, H, prob):
                state = active.get(id(rb))
                if state is not None and "arrays" not in state:
                    state["arrays"] = {key: np.array(getattr(prob, key), copy=True) for key in
                                       ["Qb", "y", "rows_v0", "rows_iscp", "rows_pair",
                                        "rows_lam", "rows_bmeas", "rows_sig"]}
                    state["pairs"] = [[list(g), int(ref), int(sat)] for g, ref, sat in prob.pairs]
                saved = encode(H) if state is not None else None
                result = original(rb, H, prob)
                if state is not None:
                    state["calls"].append(dict(method=name, H=saved, expected=encode(result)))
                return result
            return call

        for name in ["_extend", "_shrink_worst"]:
            setattr(module.RbpfFgo, name, observe(name, getattr(module.RbpfFgo, name)))

        def step(rb, rtk, context):
            nonlocal selected
            take = selected < args.max_captures and int(rtk.epoch2) % args.every == 0
            if take:
                selected += 1
                active[id(rb)] = dict(epoch=int(rtk.epoch2), calls=[], particles=len(rb.particles))
            try:
                result = original_step(rb, rtk, context)
                state = active.get(id(rb))
                if state is not None and state["calls"]:
                    path = args.output / f"{len(records):04d}-problem.npz"
                    arrays = state.pop("arrays")
                    with path.open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    records.append(dict(file=path.name, fingerprint=fingerprint(path), **state))
                return result
            finally:
                active.pop(id(rb), None)

        module.RbpfFgo.step = step

    runner._set_shared_filter_env = setup
    sys.argv = [str(here / "wp16_run_rbpf.py"), *args.runner_args[1:]]
    result = runner.main()
    if result:
        return result
    from wp13a_run_standalone_rtk import _DATA_ROOT
    import cssrlib.gnss
    import gtsam
    run = int(args.runner_args[1])
    inputs = [fingerprint(_DATA_ROOT / f"run{run}" / name) for name in
              ["rover.obs", "base.obs", "base.nav", "imu.csv", "reference.csv"]]
    libraries = [fingerprint(cssrlib.gnss.__file__)]
    native = Path(gtsam.__file__).parent / "gtsam.pyd"
    if native.is_file():
        libraries.append(fingerprint(native))
    if sources != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during capture")
    manifest = dict(schemaVersion=1, records=records, runtime=sources[0], sourceFiles=sources,
                    inputs=inputs, libraries=libraries, numpy=np.__version__, argv=sys.argv,
                    captureEvery=args.every,
                    note="Original helper outputs returned unchanged. All actual calls retained at sampled epochs, including naturally repeated particle hypotheses. Observer timing is not speed evidence.")
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(f"Captured {sum(len(r['calls']) for r in records)} actual move calls at {len(records)} epochs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
