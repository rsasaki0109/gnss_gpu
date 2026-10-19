"""Actual frozen RBPF with exact native parent scores on the unchanged D233 caller."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_parent_score import NativeParentScoreStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--extension',type=Path,required=True)
    parser.add_argument('--parent-extension',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    own,remaining = parser.parse_known_args()
    names = ['benchmarks/bench_rbpf_parent_score_field.py','benchmarks/rbpf_native_canonical.py','benchmarks/rbpf_parent_score.py',
        'benchmarks/bench_rbpf_move_field.py','benchmarks/rbpf_move_key_memo.py',
        'benchmarks/rbpf_epoch_memo.py','benchmarks/rbpf_move_reuse.py','benchmarks/rbpf_move_batch.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes/rbpf_canonical').iterdir()
        if p.suffix in {'.cpp','.txt'}]
    sources = {name:sha(ROOT/name) for name in names}
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes/rbpf_parent_score').iterdir() if p.suffix in {'.cpp','.txt'}]
    sources = {name:sha(ROOT/name) for name in names}
    extension_hash = sha(own.extension)
    parent_hash = sha(own.parent_extension)
    planpath = own.output_dir.with_suffix('.implementation.json')
    assert not planpath.exists()
    planpath.write_text(json.dumps(dict(extension_sha256=extension_hash,parent_extension_sha256=parent_hash,source_sha256=sources,
        selection='Exact native parent overlap on frozen D233; original cluster decisions/numerical/graph/caller behavior.'),indent=2)+'\n',encoding='utf8')
    prior = field.MoveBatchStep
    field.MoveBatchStep = lambda runtime,scorer,mode: NativeParentScoreStep(runtime,scorer,mode,extension=own.extension,parent_extension=own.parent_extension)
    sys.argv = [sys.argv[0],*remaining,'--output-dir',str(own.output_dir)]
    try:
        code = field.main()
    finally:
        field.MoveBatchStep = prior
    assert all(sha(ROOT/name) == digest for name,digest in sources.items())
    assert sha(own.extension) == extension_hash and sha(own.parent_extension) == parent_hash
    (own.output_dir/'implementation_check.json').write_text(json.dumps(dict(
        implementation_plan_sha256=sha(planpath),source_unchanged=True,exit_code=code),indent=2)+'\n',encoding='utf8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
