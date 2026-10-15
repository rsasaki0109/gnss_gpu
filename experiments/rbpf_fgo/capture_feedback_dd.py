"""Capture actual feedback calls, existing DDs and graph transitions.

Original graph methods and feedback run unchanged and return original values.
Observer elapsed time is not speed evidence. Use fresh output directories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from benchmark_feedback_dd import feedback_fields, graph_state, outcome, owner_state, pack
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
    fields, cfg_fields = feedback_fields(args.source)
    records, active = [], {}
    original_setup = runner._set_shared_filter_env

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
        original_feedback, original_dd = module.RbpfFgo._feedback, module.RbpfFgo._pair_dd

        def dd(rb, H, prob):
            value = original_dd(rb, H, prob)
            if id(rb) in active:
                active[id(rb)].append((pack(H), pack(value)))
            return value

        def feedback(rb, rtk, prob, weights, dd_map, used_pairs, **kwargs):
            take = len(records) < args.max_captures and int(rtk.epoch2) % args.every == 0
            if not take:
                return original_feedback(rb, rtk, prob, weights, dd_map, used_pairs, **kwargs)
            caller = sys._getframe(1)
            if caller.f_code.co_name != "step" or caller.f_locals.get("self") is not rb:
                raise RuntimeError("feedback must be observed at the production step call site")
            provided = caller.f_locals["dd_all"]
            hypotheses = [p.H for p in rb.particles]
            saved_hypotheses = pack(hypotheses)
            ib = {tuple(k): int(rtk.IB(int(k[0]), int(k[1]), rtk.nav.na))
                  for k in dict.fromkeys([*rtk.amb_keys, *rtk._committed])}
            inputs = pack(dict(owner=owner_state(rb, fields), config={k: getattr(rb.cfg, k) for k in cfg_fields},
                hypotheses=hypotheses, problem=dict(pairs=prob.pairs, cands=prob.cands, witness_cands=prob.witness_cands),
                weights=weights, ddMap=dd_map, usedPairs=used_pairs, kwargs=kwargs, providedDd=provided,
                graph=graph_state(rtk), ib=ib))
            events, previous = [], {}
            active[id(rb)] = []

            def hook(name, original, call_args):
                before = pack(graph_state(rtk))
                saved_args = pack(call_args)
                value = original(*call_args[:2], **({"reason": call_args[2]} if name == "release" else {}))
                events.append(dict(name=name, args=saved_args, before=before,
                                   after=pack(graph_state(rtk)), **{"return": pack(value)}))
                return value

            for name, event_name in [("_release_ambiguity", "release"), ("_inject_hold", "inject")]:
                previous[name] = (name in vars(rtk), vars(rtk).get(name))
                original = getattr(rtk, name)
                if event_name == "release":
                    def release(sat, frequency, reason="", original=original):
                        return hook("release", original, (sat, frequency, reason))
                    setattr(rtk, name, release)
                else:
                    def inject(xa, newly, original=original):
                        return hook("inject", original, (xa, newly))
                    setattr(rtk, name, inject)
            try:
                value = original_feedback(rb, rtk, prob, weights, dd_map, used_pairs, **kwargs)
                actual = active[id(rb)]
                assert [H for H, _ in actual] == [pack(H) for H in hypotheses]
                if provided is not None:
                    assert [dd for _, dd in actual] == [pack(dd) for dd in provided]
                assert saved_hypotheses == pack([p.H for p in rb.particles])
                row = dict(inputs=inputs, events=events, expected=outcome(rb, rtk, fields, value),
                           observedDdCalls=len(actual))
            finally:
                active.pop(id(rb), None)
                for name, (had, old) in previous.items():
                    if had:
                        setattr(rtk, name, old)
                    else:
                        delattr(rtk, name)
            path = args.output/f"{len(records):04d}-feedback.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump(row, stream, separators=(",", ":"), allow_nan=False)
            records.append(dict(file=path.name, fingerprint=fingerprint(path), epoch=int(rtk.epoch2),
                                particles=len(rb.particles), providedDd=provided is not None, graphEvents=len(events)))
            return value

        module.RbpfFgo._feedback, module.RbpfFgo._pair_dd = feedback, dd

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
        raise RuntimeError("source changed or feedback capture incomplete")
    output = dict(schemaVersion=1, runtime=sources[0], sourceFiles=sources, records=records,
                  inputs=inputs, libraries=libraries, argv=sys.argv, captureEvery=args.every,
                  note="Actual feedback calls and projected graph transitions; original graph methods/feedback/results unchanged. Provided step DDs checked byte-for-byte against actual feedback DD recomputation; H unchanged. No synthetic/expanded hypotheses. Observer elapsed time is not speed evidence.")
    with (args.output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(f"Captured {len(records)} actual feedback calls", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
