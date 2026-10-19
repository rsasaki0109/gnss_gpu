"""Accuracy-only actual prefix exercising the existing cluster_relin option."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_native_canonical import NativeCanonicalStep
from rbpf_parent_score import NativeParentScoreStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant',choices=('key','parent'),required=True)
    parser.add_argument('--extension',type=Path,required=True)
    parser.add_argument('--parent-extension',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    own,remaining = parser.parse_known_args()
    names = ['benchmarks/bench_rbpf_parent_cluster_field.py','benchmarks/rbpf_parent_score.py',
        'benchmarks/rbpf_native_canonical.py','benchmarks/bench_rbpf_move_field.py',
        'benchmarks/rbpf_move_key_memo.py','benchmarks/rbpf_epoch_memo.py',
        'benchmarks/rbpf_move_reuse.py','benchmarks/rbpf_move_batch.py']
    for folder in ('rbpf_canonical','rbpf_parent_score'):
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes'/folder).iterdir()
            if p.suffix in {'.cpp','.txt'}]
    sources = {name:sha(ROOT/name) for name in names}
    binaries = dict(canonical=sha(own.extension),parent=sha(own.parent_extension))
    planpath = own.output_dir.with_suffix('.implementation.json')
    assert not planpath.exists()
    planpath.write_text(json.dumps(dict(variant=own.variant,source_sha256=sources,binary_sha256=binaries,
        explicit_configuration=dict(RBPF_CLUSTER_RELIN='1'),
        scope='Accuracy-only same-backend comparison with original optional cluster_relin enabled for BOTH variants. All other original cluster/model settings unchanged. Not a speed or full-route gate.'),indent=2)+'\n',encoding='utf8')
    os.environ['RBPF_CLUSTER_RELIN'] = '1'
    configs = []
    prior = field.MoveBatchStep

    def select(runtime,scorer,mode):
        assert runtime.cfg.cluster_relin == 1
        configs.append(vars(runtime.cfg).copy())
        if own.variant == 'key':
            return NativeCanonicalStep(runtime,scorer,mode,extension=own.extension)
        return NativeParentScoreStep(runtime,scorer,mode,extension=own.extension,parent_extension=own.parent_extension)

    field.MoveBatchStep = select
    sys.argv = [sys.argv[0],*remaining,'--output-dir',str(own.output_dir)]
    try:
        code = field.main()
    finally:
        field.MoveBatchStep = prior
    assert configs and all(c['cluster_relin'] == 1 for c in configs)
    assert all(sha(ROOT/name) == value for name,value in sources.items())
    assert sha(own.extension) == binaries['canonical'] and sha(own.parent_extension) == binaries['parent']
    (own.output_dir/'implementation_check.json').write_text(json.dumps(dict(
        implementation_plan_sha256=sha(planpath),source_unchanged=True,exit_code=code,configs=configs),indent=2)+'\n',encoding='utf8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
