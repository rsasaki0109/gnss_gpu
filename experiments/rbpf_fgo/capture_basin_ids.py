"""Observe original canonical-ID calls within sampled real WP16 filter steps.

The observer returns the original identity unchanged. Its timing is not speed
evidence. Pass the usual WP56 runner arguments after -- and use new directories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from benchmark_basin_id import json_id
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
    records, active = [], []
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
        original_id, original_step = module._canon_basin_id, module.RbpfFgo.step

        def canonical(H):
            saved = encode(H) if active else None
            result = original_id(H)
            if active:
                active[-1].append(dict(H=saved, expected=json_id(result)))
            return result

        def step(rb, rtk, context):
            nonlocal selected
            take = selected < args.max_captures and int(rtk.epoch2) % args.every == 0
            if not take:
                return original_step(rb, rtk, context)
            selected += 1
            calls = []
            active.append(calls)
            try:
                result = original_step(rb, rtk, context)
            finally:
                active.pop()
            if calls:
                path = args.output / f"{len(records):04d}-calls.json"
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(calls, stream, separators=(",", ":"), allow_nan=False)
                records.append(dict(file=path.name, fingerprint=fingerprint(path),
                                    epoch=int(rtk.epoch2), particles=len(rb.particles), calls=len(calls)))
            return result

        module._canon_basin_id, module.RbpfFgo.step = canonical, step

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
                    note="All original canonical-ID inputs and outputs inside sampled steps, including natural repetition. Original results returned unchanged. No timings used as speed evidence.")
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(f"Captured {sum(r['calls'] for r in records)} canonical calls at {len(records)} epochs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
