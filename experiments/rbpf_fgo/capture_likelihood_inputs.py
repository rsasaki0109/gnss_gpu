"""Record actual scalar-weight and tracked-conditional inputs without replacing results.

Example: python capture_likelihood_inputs.py --output NEW --tc-src TC/src --
         1 --max-ep 600 --out-dir NEW_RUN [the normal run_ship.sh flags]
Capture overhead is excluded from all subsequent evaluator timing claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tc-src", type=Path, required=True)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--max-captures", type=int, default=32)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output.exists() or args.every < 1 or args.max_captures < 0:
        parser.error("new output, every >= 1 and max-captures >= 0 required")
    if not args.runner_args or args.runner_args[0] != "--":
        parser.error("supply runner arguments after --")
    args.output.mkdir(parents=True)
    sys.dont_write_bytecode = True
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(args.tc_src.resolve())]
    import wp16_run_rbpf as runner

    records = []
    source_files = [fingerprint(path) for path in sorted(here.glob("*.py"))]
    original_setup = runner._set_shared_filter_env
    selected = 0

    def setup(options):
        original_setup(options)
        for item in options.env:
            key, value = item.split("=", 1)
            os.environ[key] = value
        if args.max_captures == 0:
            return
        import rbpf_fgo as module
        original_step = module.RbpfFgo.step
        original_scalar = module.RbpfFgo._loglik
        original_batch = module.RbpfFgo._loglik_batch
        active = {}

        def save(rb, prob, kind, dds, centres, priors, expected):
            A0, _ = rb._batch_rows(prob)
            path = args.output / f"{len(records):04d}-{kind}.npz"
            config = {name: getattr(rb.cfg, name) for name in
                      ["beta", "huber_c", "sig_group", "mix_eps", "mix_sigma_cp", "mix_sigma_pr"]}
            arrays = {name: np.asarray(getattr(prob, name)) for name in
                      ["rows_J", "rows_v0", "rows_sig", "rows_iscp", "rows_pair", "rows_lam", "rows_bmeas"]}
            arrays.update(A0=A0, dds=np.asarray(dds), centres=np.asarray(centres), priors=np.asarray(priors))
            arrays.update({"expected_" + key: value for key, value in expected.items()})
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            records.append({"file": path.name, "fingerprint": fingerprint(path), "kind": kind,
                            "epoch": int(rb._capture_epoch), "hypotheses": len(dds),
                            "rows": len(prob.rows_J), "stateSize": A0.shape[1], "config": config})

        def scalar(rb, H, prob, cache):
            state = active.get(id(rb))
            miss = module._canon_basin_id(H) not in cache
            result = original_scalar(rb, H, prob, cache)
            if state is not None and miss:
                if not rb.cfg.marg_uniform:
                    raise ValueError("capture requires the shipped uniform-lattice model")
                centre, prior = np.zeros(3), prob.Ppos
                cluster = rb._cluster_state(H)
                if cluster is not None:
                    centre = np.asarray(cluster["pos"]) - prob.pos
                    prior = np.asarray(cluster["P"])
                    if not np.all(np.isfinite(centre)):
                        centre, prior = np.zeros(3), prob.Ppos
                state["prob"] = prob
                state["dds"].append(rb._pair_dd(H, prob))
                state["centres"].append(np.array(centre, copy=True))
                state["priors"].append(np.array(prior, copy=True))
                state["ll"].append(result)
            return result

        def batch(rb, prob, dds, centres, priors):
            result = original_batch(rb, prob, dds, centres, priors)
            if id(rb) in active:
                save(rb, prob, "conditional", dds, centres, priors,
                     dict(ll=result[0], sol=result[1], covariance=result[2]))
            return result

        def step(rb, rtk, context):
            nonlocal selected
            capture = selected < args.max_captures and int(rtk.epoch2) % args.every == 0
            if capture:
                selected += 1
                rb._capture_epoch = int(rtk.epoch2)
                active[id(rb)] = dict(dds=[], centres=[], priors=[], ll=[])
            try:
                result = original_step(rb, rtk, context)
                state = active.get(id(rb))
                if state is not None and state["dds"]:
                    save(rb, state["prob"], "weight", state["dds"], state["centres"], state["priors"],
                         dict(ll=np.asarray(state["ll"])))
                return result
            finally:
                active.pop(id(rb), None)

        module.RbpfFgo._loglik = scalar
        module.RbpfFgo._loglik_batch = batch
        module.RbpfFgo.step = step

    runner._set_shared_filter_env = setup
    sys.argv = [str(here / "wp16_run_rbpf.py"), *args.runner_args[1:]]
    result = runner.main()
    if result:
        return result
    from wp13a_run_standalone_rtk import _DATA_ROOT
    run = int(args.runner_args[1])
    inputs = [fingerprint(_DATA_ROOT / f"run{run}" / name)
              for name in ["rover.obs", "base.obs", "base.nav", "imu.csv", "reference.csv"]]
    for item in source_files:
        if fingerprint(item["path"]) != item:
            raise ValueError("capture source changed")
    import gtsam
    import cssrlib.gnss
    native = Path(gtsam.__file__).parent / "gtsam.pyd"
    libraries = [fingerprint(cssrlib.gnss.__file__)]
    if native.is_file():
        libraries.append(fingerprint(native))
    manifest = dict(schemaVersion=1, records=records, sourceFiles=source_files, inputs=inputs, libraries=libraries,
                    numpy=np.__version__, argv=sys.argv, captureEvery=args.every,
                    note="Original CPU results are returned unchanged. Inputs and expected values are copied from "
                         "actual calls; scalar captures include cache misses only. No generated/duplicated hypotheses. "
                         "Capture timings include observation overhead and are not speed evidence.")
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(f"Captured {len(records)} real likelihood batches", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
