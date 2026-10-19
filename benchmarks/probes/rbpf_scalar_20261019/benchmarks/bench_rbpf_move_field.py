"""Run the frozen shipped RBPF with an explicit experimental move caller.

Use an environment containing the pinned GTSAM/cssrlib dependencies. Supply
the external tc/src and Tokyo dataset directories; neither is modified.
Each epoch records complete RBPF state and float navigation x/P for a paired
replay audit. Setup and serialization are outside the recorded step timer.
This runner is a diagnostic, not a new default or a route accuracy claim.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from rbpf_likelihood_batch import BatchScorer
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256
from rbpf_move_batch import MoveBatchStep
from rbpf_move_cases import state

ROOT = Path(__file__).resolve().parents[1]


def encode(value):
    """Lossless tagged JSON, retaining dict order and NumPy dtype/array bits."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object arrays are not accepted')
        return dict(array=value.tobytes().hex(), dtype=value.dtype.str, shape=value.shape)
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, dict):
        return dict(mapping=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, tuple):
        return dict(tuple=[encode(v) for v in value])
    if isinstance(value, set):
        return dict(set=[encode(v) for v in sorted(value, key=repr)])
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, float):
        return dict(float64=value.hex())
    if isinstance(value, SimpleNamespace):
        return dict(namespace=encode(vars(value)))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(type(value))


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if 'array' in value:
        return np.frombuffer(bytes.fromhex(value['array']), dtype=value['dtype']).reshape(value['shape']).copy()
    if 'mapping' in value:
        return {decode(k): decode(v) for k, v in value['mapping']}
    if 'tuple' in value:
        return tuple(decode(v) for v in value['tuple'])
    if 'set' in value:
        return set(decode(v) for v in value['set'])
    if 'float64' in value:
        return float.fromhex(value['float64'])
    if 'namespace' in value:
        return SimpleNamespace(**decode(value['namespace']))
    raise ValueError('Unknown snapshot tag')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('original', 'scalar', 'native_cpu', 'gpu'), required=True)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--tc-src', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20260710)
    args = parser.parse_args()
    assert sha(SCALAR_SOURCE) == SCALAR_SHA256
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ['PPC_DATA_ROOT'] = str(args.data_root.resolve())
    sys.path.insert(0, str(ROOT/'experiments/rbpf_fgo'))
    sys.path.append(str(args.tc_src.resolve()))
    # The runner's own env/CLI ordering is unchanged. Imports use the existing
    # custom dependency build; no GTSAM mock or extraction fixture is involved.
    import rbpf_fgo
    import wp16_run_rbpf

    names = ('benchmarks/bench_rbpf_move_field.py', 'benchmarks/rbpf_move_batch.py',
        'benchmarks/rbpf_move_cases.py', 'benchmarks/rbpf_likelihood_batch.py',
        'benchmarks/rbpf_likelihood_cases.py', 'benchmarks/rbpf_likelihood_runtime_adapter.py',
        'experiments/rbpf_fgo/rbpf_fgo.py', 'experiments/rbpf_fgo/wp16_run_rbpf.py',
        'experiments/rbpf_fgo/gtsam_rtk_standalone.py', 'experiments/rbpf_fgo/wp13a_run_standalone_rtk.py')
    sources = {name: sha(ROOT/name) for name in names}
    plan = dict(mode=args.mode, run=args.run, epochs=args.epochs, seed=args.seed,
        source_sha256=sources, DLL_sha256=sha(args.dll), numpy=np.__version__,
        input_sha256={name: sha(args.data_root/f'run{args.run}'/name)
                      for name in ('rover.obs', 'base.obs', 'base.nav', 'imu.csv', 'reference.csv')},
        scope='Actual frozen shipped graph+RBPF prefix. One seed/route prefix is diagnostic only. No complete-route, multi-seed accuracy/performance acceptance.',
        timing='Recorded call seconds include controller entry and complete RBPF step; hash/snapshot serialization and workspace setup excluded. One unpinned serial run, normal priority, nonexclusive host/GPU. Not a paired statistical speed gate.')
    planpath = args.output_dir/'plan.json'
    planpath.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf8')
    original_init, original_step = rbpf_fgo.RbpfFgo.__init__, rbpf_fgo.RbpfFgo.step
    scorer = BatchScorer(args.dll) if args.mode in ('native_cpu', 'gpu') else None
    calls, metadata = [], []
    snapshots = args.output_dir/'states.jsonl.gz'
    stream = gzip.open(snapshots, 'wt', encoding='utf8', newline='\n')

    def initialize(runtime, cfg):
        original_init(runtime, cfg)
        # Instance assignment is explicit; class.step stays the frozen method
        # for source checks and for any witness instances constructed later.
        selected = (original_step.__get__(runtime) if args.mode == 'original'
                    else MoveBatchStep(runtime, scorer, args.mode))

        def recorded(rtk, ctx):
            start = time.perf_counter_ns()
            output = selected(rtk, ctx)
            elapsed = (time.perf_counter_ns() - start) / 1e9
            calls.append(elapsed)
            boundary = SimpleNamespace(_committed=rtk._committed,
                _rbpf_released=getattr(rtk, '_rbpf_released', set()),
                nav=SimpleNamespace(x=rtk.nav.x, P=rtk.nav.P))
            snapshot = state(runtime, boundary, output)
            stream.write(json.dumps(encode(snapshot), separators=(',', ':'))+'\n')
            stream.flush()
            if args.mode != 'original':
                metadata.append(encode(selected.last))
            return output

        # Guard checks the exposed step identity; call the explicit controller
        # through a temporary instance attribute, then restore the recorder.
        def guarded_recorded(rtk, ctx):
            runtime.step = selected
            try:
                return recorded(rtk, ctx)
            finally:
                runtime.step = guarded_recorded

        runtime.step = guarded_recorded

    rbpf_fgo.RbpfFgo.__init__ = initialize
    env = ['SEED_N_CYCLES=1', 'COND_HOLD=1', 'RBPF_FB=1', 'RBPF_SPAWN=1',
           'RBPF_CUDA=0', 'RBPF_FIX_VOTE_DD=6.0', 'RBPF_FIX_VOTE_DPR=3.5',
           'RBPF_FB_COMMIT_MAX_DPR=1.5']
    if args.run == 1:
        env += ['SANITY_PR_ONLY=1', 'RECOV_CP_HOLD_EPOCHS=5', 'RECOV_CP_RELEASE_THRESH=2.0',
                'RECOV_CP_RELEASE_COUNT=5', 'SANITY_MAX_MEDIAN_RATIO=5.0']
    sys.argv = [str(ROOT/'experiments/rbpf_fgo/wp16_run_rbpf.py'), str(args.run),
        '--max-ep', str(args.epochs), '--out-dir', str(args.output_dir/'trajectory'),
        '--rb-seed', str(args.seed), '--checkpoint-every', '0', '--log-every', '100']
    if args.run == 1:
        sys.argv += ['--sanity-enable', '1']
    for value in env:
        sys.argv += ['--env', value]
    start = time.perf_counter()
    try:
        exit_code = wp16_run_rbpf.main()
    finally:
        stream.close()
        if scorer is not None:
            scorer.close()
    assert all(sha(ROOT/name) == digest for name, digest in sources.items())
    assert sha(args.dll) == plan['DLL_sha256']
    (args.output_dir/'record.json').write_text(json.dumps(dict(exit_code=exit_code,
        plan_sha256=sha(planpath), snapshots_sha256=sha(snapshots), calls=len(calls),
        call_seconds=calls, caller_metadata=metadata, elapsed_seconds=time.perf_counter()-start,
        source_and_DLL_unchanged=True), indent=2)+'\n', encoding='utf8')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
