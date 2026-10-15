"""Compare owner-local Thrust storage with the exact packed-satellite PF build.

Actual production workers run sequentially. We retain the same inclusive scan,
record its inherited exact-boundary variation, and check timed states/ancestors.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from bench_pf_dd_reference_compare import _digest, _initialize, _load_extension, _scenario, _snapshot
from bench_pf_packed_satellite_compare import _validate as _preceding_validate
from bench_pf_systematic_compare import (
    _distributions, _scan_repeatability, _validate as _systematic_validate,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_probe(path):
    spec = importlib.util.spec_from_file_location('_pf_scan_cache_probe', path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return probe


def _ownership(native, probe):
    records = {}
    for count in (1, 257, 2048, 10000, 100000, 1000000):
        owners = [native.pf_device_create(count), native.pf_device_create(count)]
        try:
            original = [dict(probe.cache_metadata(owner)) for owner in owners]
            assert all(v['pointer'] == v['capacity_bytes'] == 0 and not v['in_use'] for v in original)
            addresses = [None, None]
            history = []
            for index in (0, 1, 0, 1, 0):
                owner = owners[index]
                _initialize(native, owner)
                native.pf_device_resample_systematic(owner, 42)
                meta = dict(probe.cache_metadata(owner))
                assert meta['pointer'] and meta['capacity_bytes'] and not meta['in_use']
                if addresses[index] is not None:
                    assert addresses[index] == (meta['pointer'], meta['capacity_bytes'])
                addresses[index] = meta['pointer'], meta['capacity_bytes']
                history.append(meta)
            assert addresses[0][0] != addresses[1][0]
            records[f'N{count}'] = dict(initial=original, history=history)
        finally:
            for owner in owners:
                native.pf_device_destroy(owner)
        final = [dict(probe.cache_metadata(owner)) for owner in owners]
        assert all(v['pointer'] == v['capacity_bytes'] == 0 and not v['in_use'] and not v['allocated'] for v in final)
        records[f'N{count}']['closed'] = final
    return records


def _validate(native, probe, candidate):
    records = _preceding_validate(native)
    records.update({'systematic/'+k:v for k,v in _systematic_validate(native, probe).items()})
    return dict(records=records, scan_repeatability=_scan_repeatability(native, probe),
                ownership=_ownership(native, probe) if candidate else {})


def _stages():
    return ['resample_uniform', 'resample_normal', 'resample_dominant',
            'public_systematic_step', 'public_systematic_MUPF_step',
            'public_no_resample_control', 'megopolis_control_step', 'cold_systematic_step']


def _profile(native, count, stage, repeats):
    sys.modules['gnss_gpu._gnss_gpu_pf_device'] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    position, sat, pr, ws = _scenario()
    cp = pr/.190293673+np.arange(12)*7.
    if stage == 'cold_systematic_step':
        samples, hashes = [], set()
        for repetition in range(repeats+3):
            start = time.perf_counter_ns()
            state = native.pf_device_create(count)
            try:
                _initialize(native, state)
                native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1)
                native.pf_device_weight(state, sat, pr, ws, 12, 5.)
                ess = native.pf_device_ess(state)
                resampled = ess < .5*count
                if resampled:
                    native.pf_device_resample_systematic(state, 42)
                value = np.array([ess, resampled, *native.pf_device_estimate(state)])
            finally:
                native.pf_device_destroy(state)
            elapsed = (time.perf_counter_ns()-start)/1e6
            hashes.add(_digest(value))
            if repetition >= 3:
                samples.append(elapsed)
        assert len(hashes) == 1
        digest = hashes.pop()
        return dict(median_ms=float(np.median(samples)), samples_ms=samples,
                    output_sha256=digest, returned_output_sha256=digest,
                    log_weights_sha256=None, ancestry_sha256=None, all_timed_outputs_identical=True,
                    output_scope='ESS/decision/estimate; entire lifecycle timed; no diagnostic full-state export')
    pf = None
    if stage.startswith('resample_'):
        state = native.pf_device_create(count)
        lw = _distributions(count)[stage.removeprefix('resample_')]
        def prepare():
            _initialize(native, state)
            native.pf_device_set_log_weights(state, lw)
        def operation():
            native.pf_device_resample_systematic(state, 42)
            return np.array([True])
    else:
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5.,
                                  resampling='megopolis' if stage == 'megopolis_control_step' else 'systematic', seed=42)
        state = pf._state
        def prepare():
            pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.)
        def operation():
            pf.predict([1., 2., 3.], dt=.1)
            pf.update(sat, pr, weights=ws, resample=False)
            flags = [False]
            if stage != 'public_no_resample_control':
                flags = [pf.resample_if_needed()]
            if stage == 'public_systematic_MUPF_step':
                pf.update_carrier_afv(sat, cp, weights=ws, resample=False)
                flags.append(pf.resample_if_needed())
            return np.array([*flags, *pf.estimate()])
    samples, weights_hashes, returns, ancestors_hashes = [], set(), set(), set()
    log_weights = np.empty(count)
    ancestors = np.empty(count, dtype=np.int32)
    try:
        for repetition in range(repeats+3):
            prepare()
            native.pf_device_sync(state)
            start = time.perf_counter_ns()
            value = operation()
            native.pf_device_sync(state)
            elapsed = (time.perf_counter_ns()-start)/1e6
            native.pf_device_get_log_weights(state, log_weights)
            weights_hashes.add(_digest(log_weights))
            returns.add(_digest(value))
            flag_count = 2 if stage == 'public_systematic_MUPF_step' else 1
            capture = (stage.startswith('resample_') or stage.startswith('public_systematic')) and bool(np.any(value[:flag_count]))
            if capture:
                native.pf_device_get_resample_ancestors(state, ancestors)
                ancestors_hashes.add(_digest(ancestors))
            if repetition >= 3:
                samples.append(elapsed)
        assert len(weights_hashes) == len(returns) == 1 and len(ancestors_hashes) <= 1
        actual, weights = _snapshot(native, state)
        return dict(median_ms=float(np.median(samples)), samples_ms=samples,
                    output_sha256=_digest(actual, weights, value), log_weights_sha256=weights_hashes.pop(),
                    returned_output_sha256=returns.pop(), ancestry_sha256=next(iter(ancestors_hashes), None),
                    all_timed_outputs_identical=True, final_returned_values=value.tolist(),
                    output_scope='every timed weights/returned estimate/decision/last-systematic ancestry; final full16-state snapshot')
    finally:
        native.pf_device_destroy(state)
        if pf is not None:
            pf._state = None


def _worker(args):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'python'))
    native = _load_extension(args.extension.resolve())
    probe = _load_probe(args.probe.resolve())
    print(json.dumps(dict(ready=True)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request['mode'] == 'close':
            return
        try:
            result = _validate(native, probe, args.cache_owner) if request['mode'] == 'validate' else _profile(
                native, request['size'], request['stage'], request['repeats'])
            print(json.dumps(result), flush=True)
        except Exception as error:
            print(json.dumps(dict(error=repr(error))), flush=True)


class _Worker:
    def __init__(self, extension, probe, candidate):
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--extension', str(extension.resolve()),
                   '--probe', str(probe.resolve())]
        if candidate:
            command.append('--cache-owner')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding='utf8')
        assert json.loads(self.process.stdout.readline()) == dict(ready=True)

    def request(self, **args):
        self.process.stdin.write(json.dumps(args)+'\n')
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps(dict(mode='close'))+'\n')
            self.process.stdin.flush()
        self.process.communicate(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','baseline-dir','extension','probe','output'):
        parser.add_argument('--'+key,type=Path)
    parser.add_argument('--cache-owner',action='store_true')
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--stages',nargs='+',choices=_stages())
    parser.add_argument('--sizes',type=int,nargs='+',default=[2048,10000,100000,1000000])
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=20)
    args = parser.parse_args()
    if args.extension:
        _worker(args)
        return
    root = Path(__file__).resolve().parents[1]
    reference_path = 'benchmarks/pf_packed_satellite_gpu_20261013.json'
    ref = json.loads((root/reference_path).read_text(encoding='utf8'))
    sources = dict(baseline={p:_sha(args.baseline_dir/Path(p).name) for p in ref['source_sha256']['candidate']},
                   candidate={p:_sha(root/p) for p in ref['source_sha256']['candidate']})
    extensions = dict(baseline=_sha(args.baseline),candidate=_sha(args.candidate))
    assert sources['baseline'] == ref['source_sha256']['candidate']
    assert extensions['baseline'] == ref['extension_sha256']['candidate']
    assert all(sources['baseline'][p] == sources['candidate'][p] for p in list(sources['candidate'])[2:])
    workers = {}
    runs, summary = {}, {}
    try:
        for label,path in (('baseline',args.baseline),('candidate',args.candidate)):
            workers[label] = _Worker(path,args.probe,label=='candidate')
        validations = {label:worker.request(mode='validate') for label,worker in workers.items()}
        before,after = [validations[label]['records'] for label in ('baseline','candidate')]
        assert before.keys() == after.keys()
        state_exceptions, cdf_variations = [], []
        for key,previous in before.items():
            current = after[key]
            for field in ('input_sha256','normalized_weights_sha256','max_and_sum_sha256'):
                if field in previous:
                    assert previous[field] == current[field], (key,field)
            if 'cdf_sha256' in previous and previous['cdf_sha256'] != current['cdf_sha256']:
                cdf_variations.append(key)
            if previous.get('output_sha256') != current.get('output_sha256'):
                assert key.startswith('systematic/') and key in cdf_variations, key
                assert any('/'+family+'/' in key for family in ('uniform','sparse')), key
                assert key.endswith(('/seed0','/seed1000000')), key
                state_exceptions.append(key)
            if not key.startswith('systematic/'):
                assert previous == current, key
        print(f'{len(before)-len(state_exceptions)}/{len(before)} matching outputs; {len(state_exceptions)} inherited scan boundary exceptions',flush=True)
        if not args.validation_only:
            for count in args.sizes:
                for stage in args.stages or _stages():
                    key = f'N{count}/{stage}'
                    data = {label:[] for label in workers}
                    for round_index in range(args.rounds):
                        order = ('baseline','candidate') if round_index%2==0 else ('candidate','baseline')
                        for label in order:
                            data[label].append(workers[label].request(mode='measure',size=count,stage=stage,repeats=args.repeats))
                    for field in ('output_sha256','log_weights_sha256','returned_output_sha256','ancestry_sha256'):
                        assert len({v[field] for values in data.values() for v in values}) == 1, (key,field)
                    medians = {label:[v['median_ms'] for v in values] for label,values in data.items()}
                    old,new = [float(np.median(medians[label])) for label in ('baseline','candidate')]
                    summary[key] = dict(baseline_ms=old,candidate_ms=new,speedup=old/new,round_medians_ms=medians)
                    runs[key] = data
                    print(f'{key}: {old:.4f}->{new:.4f}ms ({old/new:.3f}x)',flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = dict(ref['benchmark_sha256'])
    for p in ('benchmarks/bench_pf_scan_cache_compare.py','benchmarks/bench_pf_systematic_compare.py',
              'benchmarks/probes/pf_scan_cache_probe.cpp'):
        helpers[p] = _sha(root/p)
    assert all(_sha(root/p)==v for p,v in helpers.items())
    report = dict(schema_version=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration='CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math',
        baseline_reference=dict(path=reference_path,sha256=_sha(root/reference_path)),
        method='separate persistent production workers; sequential stage batches; alternating order each round; nonexclusive host',
        scope='synchronized whole systematic resample and public predict/update/conditional resample/estimate; initialization/IPC/diagnostic hash exports excluded; cold entire lifecycle included; control Megopolis and no-resample flows unchanged',
        rounds=args.rounds,repeats=args.repeats,sizes=args.sizes,source_sha256=sources,extension_sha256=extensions,
        benchmark_sha256=helpers,probe_sha256=_sha(args.probe),
        validation=dict(cases=len(before),matching_state_output_cases=len(before)-len(state_exceptions),
            all_state_outputs_bit_identical=not state_exceptions,normalized_weights_and_sums_bit_identical=True,
            inherited_parallel_scan_state_exceptions=state_exceptions,internal_cdf_variation_cases=cdf_variations,
            baseline_by_case=before,candidate_by_case=after,
            scan_repeatability={label:v['scan_repeatability'] for label,v in validations.items()},
            candidate_owner_checks=validations['candidate']['ownership']),summary=summary,runs=runs)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
