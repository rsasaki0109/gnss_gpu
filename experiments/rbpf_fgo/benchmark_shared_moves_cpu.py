"""Record process CPU time and wall time for two real-input RB-FGO-PF snapshots.

Requires the WP16 custom GTSAM/cssrlib environment and original PPC files.
No observations or hypotheses are generated, repeated, or batched differently.
Each worker imports a selected source after the runner applies its environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import statistics
import subprocess
import sys
import time


def fingerprint(path):
    path = Path(path)
    return dict(name=path.name, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def worker(argv):
    source, runner, result = map(Path, argv[:3])
    instances = []

    class Loader(importlib.machinery.SourceFileLoader):
        def exec_module(self, module):
            super().exec_module(module)
            original = module.RbpfFgo.__init__

            def retain(obj, *args, **kwargs):
                original(obj, *args, **kwargs)
                instances.append(obj)

            module.RbpfFgo.__init__ = retain

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "rbpf_fgo":
                return importlib.util.spec_from_loader(fullname, Loader(fullname, str(source)))
            return None

    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, Finder())
    sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), *argv[3:]]
    start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        runpy.run_path(str(runner), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu_start
    if len(instances) != 1:
        raise RuntimeError("expected one live RB-FGO-PF instance")
    pf = instances[0]
    import numpy as np
    import gtsam
    import cssrlib.gnss
    native = Path(gtsam.__file__).parent / "gtsam.pyd"
    libraries = [fingerprint(cssrlib.gnss.__file__)]
    if native.is_file():
        libraries.append(fingerprint(native))
    write_new(result, dict(source=fingerprint(source), wallSeconds=wall, cpuSeconds=cpu,
                           pfSeconds=pf.step_seconds, steps=pf.n_steps,
                           measuredSteps=pf.n_meas_updates, skippedSteps=pf.n_skipped,
                           resamples=pf.n_resamples, fixes=pf.n_fix,
                           python=platform.python_version(), numpy=np.__version__,
                           config={key: repr(value) for key, value in vars(pf.cfg).items()},
                           libraries=libraries))


def exact_outputs(first, second):
    import numpy as np
    if (first / "run1.pos").read_bytes() != (second / "run1.pos").read_bytes():
        raise AssertionError("position output differs")
    checked = {}
    for name in ["run1.npz", "run1_rows.npz"]:
        with np.load(first / name, allow_pickle=False) as a, np.load(second / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise AssertionError(f"{name}: array keys differ")
            for key in a.files:
                x, y = a[key], b[key]
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    raise AssertionError(f"{name}: {key} differs")
            checked[name] = len(a.files)
    return dict(position=fingerprint(first / "run1.pos"), exactArrays=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).with_name("rbpf_fgo.py"))
    parser.add_argument("--tc-src", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="PPC tokyo directory")
    parser.add_argument("--output", type=Path, required=True, help="must not exist")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.output.exists() or args.epochs < 201 or not 1 <= args.pairs <= 20:
        parser.error("new output, >=201 epochs, and 1..20 pairs required")
    here = args.experiment_dir.resolve()
    runner = here / "wp16_run_rbpf.py"
    data = args.data_root.resolve() / "run1"
    inputs = [fingerprint(data / name) for name in
              ["rover.obs", "base.obs", "base.nav", "imu.csv", "reference.csv"]]
    args.output.mkdir(parents=True)
    work = args.output.resolve()
    sources = {}
    for label, source in [("baseline", args.baseline), ("candidate", args.candidate)]:
        target = work / f"rbpf_fgo_{label}.py"
        shutil.copyfile(source, target)
        sources[label] = target
    overrides = dict(SEED_N_CYCLES=1, COND_HOLD=1, RBPF_FB=1, RBPF_SPAWN=1,
                     RBPF_FIX_VOTE_DD=6.0, RBPF_FIX_VOTE_DPR=3.5, RBPF_FB_COMMIT_MAX_DPR=1.5,
                     RBPF_TRACK=2, RBPF_TRACK_EMPTY=1, RBPF_TRACK_OUT=2, RBPF_KEEP=1,
                     QC_ANY_BAND=1, QC_ANY_BAND_CODE_ONLY=1, QC_ANY_BAND_PR_SCALE=10,
                     QC_BASE_MASK=1, ELMIN_DEG=10.0, HUBER_PR=1.345, HUBER_CP=2.0,
                     PR_CNR_ERR=0.5, DOP_V2=1, DOPPLER_LEVER_COMP=1, DOP_COV_SCALE=1.5,
                     RBPF_CARRY=1, RBPF_DUMP_ROWS=1)
    flags = ["1", "--max-ep", str(args.epochs), "--nf", "2", "--band-priority", "C:2,6;R:-",
             "--dr-fill", "1", "--gap-fill", "1", "--fix-rule", "5,15,5,15,0.1,5,1.0",
             "--fix-hyst", "5,2,2,6,2,6,0.1,5", "--carry", "1", "--af-select", "6",
             "--fix-flt-max", "0.5", "--checkpoint-every", "0", "--log-every", "100",
             "--rb-seed", "20260710"]
    for key, value in overrides.items():
        flags += ["--env", f"{key}={value}"]
    env = os.environ.copy()
    # Do not inherit experimental RBPF toggles from the calling shell.
    for key in list(env):
        if key.startswith("RBPF_"):
            del env[key]
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", RBPF_CUDA="0",
               NUMBA_CACHE_DIR=str(work / "numba-cache"), MPLCONFIGDIR=str(work / "matplotlib"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONPATH=str(args.tc_src.resolve()), PPC_DATA_ROOT=str(args.data_root.resolve()))
    dependencies = [Path(__file__), *sources.values(),
                    *sorted(here.glob("*.py")),
                    args.tc_src.resolve() / "gnss_fgo/utils/sig_autodetect.py"]
    frozen = [fingerprint(path) for path in dependencies]
    write_new(work / "freeze.json", dict(sources=frozen, inputs=inputs, flags=flags,
                                        cpuThreads=1, warmupsPerSource=1, epochs=args.epochs))
    records, parity = [], []
    reference = None
    # One untimed full run per source fills the shared Numba disk cache.
    for trial in range(-1, args.pairs):
        order = ["baseline", "candidate"] if trial % 2 == 0 or trial == -1 else ["candidate", "baseline"]
        for label in order:
            tag = f"warmup-{label}" if trial < 0 else f"pair-{trial:02d}-{label}"
            out = work / tag
            meta = work / f"{tag}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", str(sources[label]),
                   str(runner), str(meta), *flags, "--out-dir", str(out)]
            with (work / f"{tag}.log").open("x", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=here, env=env, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f"{tag} failed: see its log")
            row = dict(trial=trial, policy=label, **json.loads(meta.read_text(encoding="utf-8")))
            records.append(row)
            if reference is None:
                reference = out
            parity.append(dict(run=tag, **exact_outputs(reference, out)))
            print(json.dumps(row), flush=True)
    # Verify the source snapshots and all dependent sources stayed fixed.
    current = [fingerprint(path) for path in dependencies]
    if current != frozen:
        raise RuntimeError("source changed during measurement")
    if inputs != [fingerprint(data / name) for name in
                  ["rover.obs", "base.obs", "base.nav", "imu.csv", "reference.csv"]]:
        raise RuntimeError("input changed during measurement")
    paired = []
    for trial in range(args.pairs):
        a = next(r for r in records if r["trial"] == trial and r["policy"] == "baseline")
        b = next(r for r in records if r["trial"] == trial and r["policy"] == "candidate")
        paired.append(dict(trial=trial, cpuReductionPercent=100 * (1-b["cpuSeconds"]/a["cpuSeconds"]), pfReductionPercent=100 * (1-b["pfSeconds"]/a["pfSeconds"]),
                           wallReductionPercent=100 * (1-b["wallSeconds"]/a["wallSeconds"])))
    write_new(work / "results.json", dict(schemaVersion=1, sources=frozen, inputs=inputs,
              flags=flags, cpuThreads=1, records=records, parity=parity, paired=paired,
              medianCpuReductionPercent=statistics.median(r["cpuReductionPercent"] for r in paired),
              medianPfReductionPercent=statistics.median(r["pfReductionPercent"] for r in paired),
              medianWallReductionPercent=statistics.median(r["wallReductionPercent"] for r in paired),
              timingScope="CPU: process_time across the full runner; PF: existing uninstrumented step accumulator; wall: runner imports, input load, full runner and output writes; excludes parent startup and parity. One warmup per source excluded. Tokyo run1 prefix only; no full-route/ensemble accuracy claim."))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker(sys.argv[2:])
    else:
        main()
