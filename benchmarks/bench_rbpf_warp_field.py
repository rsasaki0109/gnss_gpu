"""Actual frozen RBPF with D229 staging and an explicit serial/warp scorer."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_move_key_memo import MoveKeyMemoStep
from rbpf_warp_batch import WarpBatchScorer

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--kernel',choices=('serial','warp'),default='warp')
    parser.add_argument('--output-dir',type=Path,required=True)
    own,remaining = parser.parse_known_args()
    names = ['benchmarks/bench_rbpf_warp_field.py','benchmarks/rbpf_warp_batch.py',
        'benchmarks/bench_rbpf_move_field.py','benchmarks/rbpf_move_key_memo.py',
        'benchmarks/rbpf_epoch_memo.py','benchmarks/rbpf_move_reuse.py','benchmarks/rbpf_move_batch.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes/rbpf_likelihood_warp').iterdir()
        if p.suffix in {'.cu','.cuh','.txt'}]
    sources = {name:sha(ROOT/name) for name in names}
    planpath = own.output_dir.with_suffix('.implementation.json')
    assert not planpath.exists()
    planpath.write_text(json.dumps(dict(kernel=own.kernel,source_sha256=sources,
        selection='Explicit standalone kernel and D229 caller; frozen runtime/runner unchanged.'),indent=2)+'\n',encoding='utf8')
    prior_step,prior_scorer = field.MoveBatchStep,field.BatchScorer
    field.MoveBatchStep = MoveKeyMemoStep
    field.BatchScorer = lambda dll: WarpBatchScorer(dll,kernel=own.kernel)
    sys.argv = [sys.argv[0],*remaining,'--output-dir',str(own.output_dir)]
    try:
        code = field.main()
    finally:
        field.MoveBatchStep,field.BatchScorer = prior_step,prior_scorer
    assert all(sha(ROOT/name) == digest for name,digest in sources.items())
    (own.output_dir/'implementation_check.json').write_text(json.dumps(dict(
        implementation_plan_sha256=sha(planpath),source_unchanged=True,exit_code=code),indent=2)+'\n',encoding='utf8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
