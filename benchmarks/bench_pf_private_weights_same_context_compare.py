"""Compare Python-only PF migration changes inside one exact native context."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tracemalloc

import numpy as np

from bench_pf_private_weights_compare import ROOT, Session, _load_extension, _setup, _pin, _sha
from bench_pf_covariance_broadcast_compare import _load_python


def _session(native, count, stage, module):
    session = Session(native, count, stage)
    if stage.startswith('public_pool_'):
        session.pool = module.PersistentParticleRuntimePool(
            lambda n: session.peer if n == count//2 else session.pf, (count//2, count))
        session.pool.runtimes = {count: session.pf, count//2: session.peer}
        session.pool.active = session.pf
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-dir', 'source-record', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000000])
    parser.add_argument('--stages', nargs='+', default=['public_pool_posterior', 'public_pool_roundtrip', 'PR_control'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--memory', action='store_true')
    args = parser.parse_args()
    assert min([*args.sizes, args.rounds, args.repeats]) > 0
    record = json.loads(args.source_record.read_text(encoding='utf8'))
    for label, sources in record['source_sha256'].items():
        for name, digest in sources.items():
            path = args.baseline_dir/Path(name).name if label == 'baseline' else ROOT/name
            assert _sha(path) == digest, (label, name)
    binary = ROOT/'build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd'
    assert _sha(binary) == record['extension_sha256']['candidate'] == record['extension_sha256']['baseline']
    _pin()
    sys.path.insert(0, str(ROOT/'python'))
    native = _load_extension(binary)
    _setup(native)
    before = _load_python(args.baseline_dir/'realtime_runtime.py', 'pf_private_weights_before')
    after = _load_python(ROOT/'python/gnss_gpu/realtime_runtime.py', 'pf_private_weights_after')
    records, memory = {}, {}
    for count in args.sizes:
        for stage in args.stages:
            sessions = {}
            try:
                sessions['baseline'] = _session(native, count, stage, before)
                sessions['candidate'] = _session(native, count, stage, after)
                rows, medians = [], {label: [] for label in sessions}
                for round_index in range(args.rounds):
                    for session in sessions.values():
                        session.measure(1, snapshot=True, warmups=0)
                    samples = {label: [] for label in sessions}
                    for repetition in range(args.repeats):
                        order = ('baseline', 'candidate') if (round_index+repetition) % 2 == 0 else ('candidate', 'baseline')
                        pair = {label: sessions[label].measure(1, snapshot=True, warmups=0) for label in order}
                        for field in ('input_sha256', 'output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                            assert pair['baseline'][field] == pair['candidate'][field], (count, stage, field)
                        rows.append(dict(round=round_index, repetition=repetition, order=order, runs=pair))
                        for label in sessions:
                            samples[label].append(pair[label]['samples_ms'][0])
                    for label in sessions:
                        medians[label].append(float(np.median(samples[label])))
                a, b = (float(np.median(medians[label])) for label in sessions)
                key = f'N{count}/{stage}'
                records[key] = dict(baseline_ms=a, candidate_ms=b, reduction_percent=100*(1-b/a),
                    improving_rounds=sum(y<x for x, y in zip(medians['baseline'], medians['candidate'], strict=True)),
                    round_medians_ms=medians, pairs=rows, all_timed_full_state_weight_return_snapshots_identical=True)
                if args.memory:
                    memory[key] = {}
                    for label, session in sessions.items():
                        session.prepare()
                        native.pf_device_sync(session.state)
                        tracemalloc.start()
                        session.last_returned = session.operation()
                        native.pf_device_sync(session.state)
                        live, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        memory[key][label] = dict(live_bytes=live, peak_bytes=peak, snapshot=session.snapshot())
                    assert memory[key]['baseline']['snapshot'] == memory[key]['candidate']['snapshot']
                print(key, a, '->', b, flush=True)
            finally:
                for session in sessions.values():
                    session.close()
    helpers = {str(Path(module.__file__).resolve().relative_to(ROOT)).replace('\\', '/'): _sha(Path(module.__file__))
        for module in list(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).suffix == '.py' and Path(module.__file__).resolve().is_relative_to(ROOT)}
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_record_sha256=_sha(args.source_record), source_sha256=record['source_sha256'], extension_sha256=record['extension_sha256'],
        benchmark_sha256=helpers, measurements=records, tracemalloc_peak_measurements=memory,
        rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, stages=args.stages,
        method='One owned process pinned to last allowed CPU, normal priority. Same actual native module/context and unchanged PF runtime class; independent baseline/candidate pool classes and source/target states. Serial alternating complete synchronized operations; every timed full16-field source/target state, weight and return snapshot checked outside timing; nonexclusive host.',
        scope='Public pool N->N/2->N after PR posterior or uniform initialization, or complete PR control. Initialization/restoration, input setup and hashing excluded; readout, FP64 normalization/sequential CDF, transfers/native validation/gather/reset/synchronization/estimate included.')
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
