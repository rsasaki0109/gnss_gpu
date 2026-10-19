"""Compare complete PF capacity switches using exact private host readouts."""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from bench_pf_fused_uniform_compare import Session, _load_extension, _setup, validate

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    handle = kernel.GetCurrentProcess()
    assert kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system))
    assert kernel.SetProcessAffinityMask(handle, 1 << (allowed.value.bit_length()-1))


def worker(extension, python_dir):
    _pin()
    sys.path.insert(0, str(ROOT/'python'))
    native = _load_extension(extension.resolve())
    klass = _setup(native, python_dir)
    session, key = None, None
    print(json.dumps(dict(ready=True)), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request['mode'] == 'close':
                return
            try:
                if request['mode'] == 'validate':
                    result = validate(native, klass)
                else:
                    current = request['size'], request['stage']
                    if current != key:
                        if session:
                            session.close()
                        session, key = Session(native, *current), current
                    if request['mode'] == 'memory':
                        session.prepare()
                        native.pf_device_sync(session.state)
                        tracemalloc.start()
                        session.last_returned = session.operation()
                        native.pf_device_sync(session.state)
                        live, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        result = dict(live_bytes=live, peak_bytes=peak, snapshot=session.snapshot())
                    else:
                        result = session.measure(1, snapshot=True, warmups=0)
                print(json.dumps(result), flush=True)
            except Exception as error:
                import traceback
                print(json.dumps(dict(error=repr(error), traceback=traceback.format_exc())), flush=True)
    finally:
        if session:
            session.close()


class Worker:
    def __init__(self, extension, folder, output, label):
        argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--extension', str(extension.resolve())]
        if folder:
            argv += ['--python-dir', str(folder.resolve())]
        self.logpath = ROOT/f'dist/{output.stem}_{label}_worker.log'
        self.log = self.logpath.open('wb')
        self.process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, encoding='utf8', env=dict(os.environ))
        (ROOT/f'dist/{output.stem}_{label}_worker_pid.json').write_text(json.dumps(dict(pid=self.process.pid)), encoding='utf8')
        assert json.loads(self.process.stdout.readline()) == dict(ready=True)

    def request(self, **params):
        self.process.stdin.write(json.dumps(params)+'\n')
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        if 'error' in result:
            raise RuntimeError(result)
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps(dict(mode='close'))+'\n')
            self.process.stdin.flush()
        self.process.communicate(timeout=30)
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('extension', 'python-dir', 'baseline-dir', 'source-record', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, default=[100000, 1000000])
    parser.add_argument('--stages', nargs='+', default=['public_pool_posterior', 'public_pool_roundtrip', 'PR_control'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    if args.extension:
        worker(args.extension, args.python_dir)
        return
    assert args.baseline_dir and args.source_record and args.output
    assert min([*args.sizes, args.rounds, args.repeats]) > 0
    edit = json.loads(args.source_record.read_text(encoding='utf8'))
    for label, sources in edit['source_sha256'].items():
        for name, digest in sources.items():
            path = args.baseline_dir/Path(name).name if label == 'baseline' else ROOT/name
            assert _sha(path) == digest, (label, name)
    binary = '_gnss_gpu_pf_device.cp312-win_amd64.pyd'
    assert _sha(args.baseline_dir/binary) == _sha(ROOT/'build/diag/Release'/binary) == edit['extension_sha256']['candidate']
    workers, records, memory, validation = {}, {}, {}, {}
    try:
        # Use one exact DLL path for both processes; create candidate first to
        # check that prior worker creation order did not determine the result.
        candidate = Worker(ROOT/'build/diag/Release'/binary, None, args.output, 'candidate')
        workers['candidate'] = candidate
        original = Worker(ROOT/'build/diag/Release'/binary, args.baseline_dir, args.output, 'baseline')
        workers = dict(baseline=original, candidate=candidate)
        if args.validation_only:
            values = {label: owner.request(mode='validate') for label, owner in workers.items()}
            assert values['baseline'] == values['candidate']
            validation = dict(cases=len(values['baseline']), all_outputs_bit_identical=True, records=values['baseline'])
        else:
            for count in args.sizes:
                for stage in args.stages:
                    rows, medians = [], {label: [] for label in workers}
                    for round_index in range(args.rounds):
                        for owner in workers.values():
                            owner.request(mode='measure', size=count, stage=stage)
                        samples = {label: [] for label in workers}
                        for repeat in range(args.repeats):
                            order = ('baseline', 'candidate') if (round_index+repeat) % 2 == 0 else ('candidate', 'baseline')
                            pair = {label: workers[label].request(mode='measure', size=count, stage=stage) for label in order}
                            for field in ('input_sha256', 'output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                                assert pair['baseline'][field] == pair['candidate'][field], (count, stage, field)
                            rows.append(dict(round=round_index, repetition=repeat, order=order, runs=pair))
                            for label in workers:
                                samples[label].append(pair[label]['samples_ms'][0])
                        for label in workers:
                            medians[label].append(float(np.median(samples[label])))
                    a, b = (float(np.median(medians[label])) for label in workers)
                    key = f'N{count}/{stage}'
                    records[key] = dict(baseline_ms=a, candidate_ms=b, reduction_percent=100*(1-b/a),
                        improving_rounds=sum(y<x for x, y in zip(medians['baseline'], medians['candidate'], strict=True)),
                        round_medians_ms=medians, pairs=rows, all_timed_full_state_weight_return_snapshots_identical=True)
                    if args.memory:
                        values = {label: owner.request(mode='memory', size=count, stage=stage) for label, owner in workers.items()}
                        assert values['baseline']['snapshot'] == values['candidate']['snapshot']
                        memory[key] = values
                    print(key, a, '->', b, flush=True)
    finally:
        for owner in workers.values():
            owner.close()
    helpers = {str(Path(module.__file__).resolve().relative_to(ROOT)).replace('\\', '/'): _sha(Path(module.__file__))
        for module in list(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).suffix == '.py' and Path(module.__file__).resolve().is_relative_to(ROOT)}
    helpers['benchmarks/bench_pf_private_weights_compare.py'] = _sha(Path(__file__))
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_record_sha256=_sha(args.source_record), source_sha256=edit['source_sha256'], extension_sha256=edit['extension_sha256'],
        benchmark_sha256=helpers, measurements=records, tracemalloc_peak_measurements=memory, validation=validation,
        rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, stages=args.stages,
        worker_creation_order=['candidate', 'baseline'],
        method='Separate persistent original/candidate Python workers, same exact native PE and DLL path. Each owned worker pinned to same last allowed CPU, normal priority. Alternating serial complete synchronized calls; full source/target states, weights and returns compared after every timed call outside timing; nonexclusive host.',
        scope='Public pool N->N/2->N after PR posterior or uniform initialization, or complete PR control. Initialization/restoration, input setup, IPC and hashes excluded; readout, CPU FP64 normalization/sequential CDF, transfers/native validation/gather/weight reset/synchronization/estimate included.',
        worker_logs_sha256={label: _sha(owner.logpath) for label, owner in workers.items()})
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
