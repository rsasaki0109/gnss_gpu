"""Actual frozen RBPF with D229 staging and exact native DD expansion."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_move_key_memo import MoveKeyMemoStep
from rbpf_dd_pack_runtime import PackedDdScorer, PackedDdStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pack-dll',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    own,remaining = parser.parse_known_args()
    names = ['benchmarks/bench_rbpf_dd_pack_runtime_field.py','benchmarks/rbpf_dd_pack_runtime.py','benchmarks/rbpf_dd_pack.py',
        'benchmarks/bench_rbpf_move_field.py','benchmarks/rbpf_move_key_memo.py',
        'benchmarks/rbpf_epoch_memo.py','benchmarks/rbpf_move_reuse.py','benchmarks/rbpf_move_batch.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes/rbpf_dd_pack').iterdir()
        if p.suffix in {'.cpp','.txt'}]
    sources = {name:sha(ROOT/name) for name in names}
    planpath = own.output_dir.with_suffix('.implementation.json')
    assert not planpath.exists()
    planpath.write_text(json.dumps(dict(pack_DLL_sha256=sha(own.pack_dll),source_sha256=sources,
        selection='Explicit native DD expansion and D229 caller; frozen runtime/runner unchanged.'),indent=2)+'\n',encoding='utf8')
    prior_step,prior_scorer = field.MoveBatchStep,field.BatchScorer
    field.MoveBatchStep = PackedDdStep
    field.BatchScorer = lambda dll: PackedDdScorer(dll,pack_dll=own.pack_dll)
    sys.argv = [sys.argv[0],*remaining,'--output-dir',str(own.output_dir)]
    try:
        code = field.main()
    finally:
        field.MoveBatchStep,field.BatchScorer = prior_step,prior_scorer
    assert all(sha(ROOT/name) == digest for name,digest in sources.items())
    assert sha(own.pack_dll) == json.loads(planpath.read_text(encoding='utf8'))['pack_DLL_sha256']
    (own.output_dir/'implementation_check.json').write_text(json.dumps(dict(
        implementation_plan_sha256=sha(planpath),source_unchanged=True,exit_code=code),indent=2)+'\n',encoding='utf8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
