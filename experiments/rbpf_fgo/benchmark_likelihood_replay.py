"""Compare the original NumPy batch evaluator with fused CUDA on recorded inputs.

The live filter, graph, proposal loop, resampling and FIX decisions are not run
by this benchmark. CUDA clocks include every input upload and result readback.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from functools import partial
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

from capture_likelihood_inputs import fingerprint
from likelihood_cuda import CudaLikelihood, prepare


def cpu_reference(source):
    """Compile the two original method bodies without importing GTSAM/cssrlib.

    This keeps replay available in CPU-only test environments. No arithmetic
    or error branch is rewritten. The original memoized design is supplied.
    """
    parsed = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "RbpfFgo")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and
               node.name in {"_batch_rows", "_loglik_batch"}]
    if len(methods) != 2:
        raise ValueError("original batch methods not found")
    reduced = ast.ClassDef(name="Reference", bases=[], keywords=[], body=methods, decorator_list=[])
    reduced.type_params = []
    module = ast.fix_missing_locations(ast.Module(body=[reduced], type_ignores=[]))
    namespace = {"np": np, "math": math, "_LOG2PI": math.log(2.0 * math.pi)}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["Reference"]


def reference_call(reference, packed, config):
    evaluator = reference()
    evaluator.cfg = SimpleNamespace(**config)
    problem = SimpleNamespace(**{key: value for key, value in packed.items() if key.startswith("rows_")})
    problem.rows_iscp = packed["rows_iscp"].astype(bool)
    problem._batch_rows = (packed["A0"], packed["A0"].shape[1] - 3)
    return lambda: evaluator._loglik_batch(problem, packed["dds"], packed["centres"], packed["priors"])


def check(observed, expected):
    errors = {}
    for name, value in zip(["ll", "sol", "covariance"], observed):
        if "expected_" + name not in expected:
            continue
        target = expected["expected_" + name]
        np.testing.assert_allclose(value, target, rtol=1e-9, atol=1e-9)
        errors[name] = dict(maxAbs=float(np.max(np.abs(value-target))),
                           maxRelative=float(np.max(np.abs(value-target)/np.maximum(1.0, np.abs(target)))))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--start-policy", choices=["cpu", "cuda"], default="cpu")
    options = [value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    if len(options) != len(set(options)):
        parser.error("duplicate options")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.repeats <= 100:
        parser.error("new output and repeats 1..100 required")
    sys.dont_write_bytecode = True
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest["records"]:
        parser.error("recorded inputs required")
    here = Path(__file__).resolve().parent
    source = here / "rbpf_fgo.py"
    pinned = next(item for item in manifest["sourceFiles"] if Path(item["path"]).name == "rbpf_fgo.py")
    if fingerprint(source)["sha256"] != pinned["sha256"]:
        parser.error("original evaluator source differs")
    sources = [fingerprint(here / name) for name in
               ["benchmark_likelihood_replay.py", "likelihood_cuda.py", "capture_likelihood_inputs.py", "rbpf_fgo.py"]]
    reference = cpu_reference(source)
    engine = CudaLikelihood()
    batches = []
    for record in manifest["records"]:
        path = args.manifest.parent / record["file"]
        if fingerprint(path)["sha256"] != record["fingerprint"]["sha256"]:
            raise ValueError("recorded input differs")
        with np.load(path, allow_pickle=False) as archive:
            packed, config = prepare(archive, record["config"])
            expected = {key: archive[key].copy() for key in archive.files if key.startswith("expected_")}
        cpu = reference_call(reference, packed, config)
        cuda = partial(engine.evaluate, packed, config)
        a, b = cpu(), cuda()
        cpu_errors, cuda_errors = check(a, expected), check(b, expected)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-9)
        batches.append(dict(record=record, calls=dict(cpu=cpu, cuda=cuda),
                            parity=dict(cpu=cpu_errors, cuda=cuda_errors)))
    print(json.dumps(dict(stage="parity", batches=len(batches))), flush=True)
    order = [args.start_policy, "cuda" if args.start_policy == "cpu" else "cpu"]

    def trial(index):
        names = order if index % 2 == 0 else list(reversed(order))
        observations = []
        for batch in batches:
            times = {}
            for policy in names:
                start = time.perf_counter_ns()
                result = batch["calls"][policy]()
                times[policy] = (time.perf_counter_ns()-start) / 1e6
                if not all(np.all(np.isfinite(value)) for value in result):
                    raise ValueError("nonfinite replay output")
            observations.append(times)
        return dict(order=names, observations=observations)

    warmups = [trial(0), trial(1)]
    trials = [trial(index) for index in range(args.repeats)]
    groups = defaultdict(list)
    for index, batch in enumerate(batches):
        groups[batch["record"]["kind"]].append(index)
    summaries = {}
    for kind, indices in groups.items():
        cpu = [sum(item["observations"][index]["cpu"] for index in indices) for item in trials]
        cuda = [sum(item["observations"][index]["cuda"] for index in indices) for item in trials]
        summaries[kind] = dict(cpuTotalMedianMillis=statistics.median(cpu), cudaTotalMedianMillis=statistics.median(cuda),
                               medianPairedShorteningPercent=statistics.median(100*(1-b/a) for a,b in zip(cpu,cuda)),
                               fasterPairs=sum(a>b for a,b in zip(cpu,cuda)), batches=len(indices))
    for item in sources:
        if fingerprint(item["path"]) != item:
            raise ValueError("replay source changed")
    import numba
    device = engine.cuda.get_current_device()
    result = dict(schemaVersion=1, comparison="rbpf-likelihood-cuda-replay", manifest=fingerprint(args.manifest),
                  captureProvenance={key: value for key, value in manifest.items() if key != "records"},
                  sources=sources, environment=dict(python=sys.version.split()[0], numpy=np.__version__, numba=numba.__version__,
                  computeCapability=list(device.compute_capability), ompNumThreads=os.environ.get("OMP_NUM_THREADS"),
                  openblasNumThreads=os.environ.get("OPENBLAS_NUM_THREADS")),
                  startPolicy=args.start_policy, repeats=args.repeats,
                  batches=[dict(record=item["record"], parity=item["parity"]) for item in batches],
                  warmups=warmups, trials=trials, summary=summaries,
                  note="Recorded calls from a real CPU filter prefix, without hypothesis replication. Original NumPy "
                       "_loglik_batch method bodies are compiled unchanged from the pinned source; cached design and "
                       "packed arrays are prepared before clocks. Scalar-weight captures validate against original "
                       "scalar outputs but CPU timings use the existing batched method, not the scalar proposal loop. "
                       "CUDA durations include all uploads, fused FP64 evaluation, all output readback and stream "
                       "synchronization. Imports/JIT, packing, parity, output checks and serialization are excluded. "
                       "Shared warm workspace; alternating policies; no live graph, proposal generation, RNG, "
                       "resampling, FIX decisions, ensemble coordination or full positioning speedup is measured.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    engine.close()
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
