"""Reverse worker creation and alternate every complete scan-cache PF call."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bench_pf_scan_cache_compare import _Worker


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','baseline-dir','probe','reference','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = json.loads(args.reference.read_text(encoding='utf8'))
    for label in ('baseline','candidate'):
        for name,digest in reference['source_sha256'][label].items():
            path = args.baseline_dir/Path(name).name if label == 'baseline' else root/name
            assert sha(path) == digest, name
    assert sha(args.baseline) == reference['extension_sha256']['baseline']
    assert sha(args.candidate) == reference['extension_sha256']['candidate']
    assert sha(args.probe) == reference['probe_sha256']
    workers = {}
    summary,runs = {},{}
    try:
        for label,path in (('candidate',args.candidate),('baseline',args.baseline)):
            workers[label] = _Worker(path,args.probe,label=='candidate')
        for stage in ('resample_uniform','public_systematic_step','public_systematic_MUPF_step',
                      'public_no_resample_control','megopolis_control_step'):
            data = {label:[] for label in workers}
            for round_index in range(5):
                batch = {label:[] for label in workers}
                for repetition in range(20):
                    order = ('candidate','baseline') if (round_index+repetition)%2==0 else ('baseline','candidate')
                    for label in order:
                        batch[label].append(workers[label].request(mode='measure',size=2048,stage=stage,repeats=1))
                for label in workers:
                    data[label].append(batch[label])
            for field in ('output_sha256','log_weights_sha256','returned_output_sha256','ancestry_sha256'):
                assert len({v[field] for rounds in data.values() for batch in rounds for v in batch}) == 1, (stage,field)
            medians = {label:[float(np.median([v['median_ms'] for v in batch])) for batch in rounds]
                       for label,rounds in data.items()}
            old,new = [float(np.median(medians[label])) for label in ('baseline','candidate')]
            summary[stage] = dict(baseline_ms=old,candidate_ms=new,speedup=old/new,round_medians_ms=medians)
            runs[stage] = data
            print(f'{stage}: {old:.4f}->{new:.4f}ms ({old/new:.3f}x)',flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = dict(reference['benchmark_sha256'])
    assert all(sha(root/name)==digest for name,digest in helpers.items())
    helpers['benchmarks/bench_pf_scan_cache_recheck.py'] = sha(Path(__file__).resolve())
    result = dict(schema_version=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),
        reference=dict(path=str(args.reference.relative_to(root)) if args.reference.is_absolute() else str(args.reference),
                       sha256=sha(args.reference)),
        method='candidate worker created first; three warmup calls then one timed call per RPC; alternate every whole profile/round; nonexclusive host; initialization/export/destroy/IPC excluded',
        rounds=5,repeats=20,size=2048,source_sha256=reference['source_sha256'],extension_sha256=reference['extension_sha256'],
        benchmark_sha256=helpers,probe_sha256=reference['probe_sha256'],summary=summary,runs=runs)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
