"""Check complete traces at the large-parallel ray limit using frozen operations."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from bench_bvh_build_compare import SOURCE_PATHS, _activate, _sha
from bench_bvh_large_parallel_compare import _timing
from bench_bvh_multipath_compare import _load_extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'candidate', 'baseline-dir', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'python'))
    anchor_path = 'benchmarks/bvh_serial_cache_gpu_20261013.json'
    anchor = json.loads((root/anchor_path).read_text(encoding='utf8'))
    sources = dict(baseline={p:_sha(args.baseline_dir/Path(p).name) for p in SOURCE_PATHS},
                   candidate={p:_sha(root/p) for p in SOURCE_PATHS})
    extensions = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert sources['baseline'] == anchor['source_sha256']['candidate']
    assert extensions['baseline'] == anchor['extension_sha256']['candidate']
    assert all(sources['baseline'][p] == sources['candidate'][p] for p in SOURCE_PATHS[1:])
    before = _load_extension(args.baseline.resolve(), 'baseline._bvh')
    after = _load_extension(args.candidate.resolve(), 'candidate._bvh')
    _activate(after)
    measurements = _timing(before, after, 3, 5, [12000], [(512, 32), (513, 32)])
    helpers = ('benchmarks/bench_bvh_large_parallel_upper_check.py',
               'benchmarks/bench_bvh_large_parallel_compare.py', 'benchmarks/bench_bvh_build_compare.py',
               'benchmarks/bench_bvh_combined_compare.py', 'benchmarks/bench_bvh_multipath_compare.py',
               'benchmarks/bench_raytrace.py')
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        baseline_reference=dict(path=anchor_path, sha256=_sha(root/anchor_path)),
        method='same process; sequential whole calls; alternating order per call/round; nonexclusive host',
        scope='native reflection/public LOS+reflection/native owner creation+first trace+close/unchanged LOS at 16384 and 16416 rays; all timed arrays checked bitwise; preceding full validator separately covers 16384/16385 boundaries',
        rounds=3, repeats=5, source_sha256=sources, extension_sha256=extensions,
        benchmark_sha256={p:_sha(root/p) for p in helpers}, measurements=measurements)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
