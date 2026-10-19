"""Explicit proposal-template selection around the unchanged field runner."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_move_key_memo import MoveKeyMemoStep
from rbpf_proposal_reuse import ProposalReuseStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reuse-proposals', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    own, _ = parser.parse_known_args()
    names = ('benchmarks/bench_rbpf_proposal_field.py', 'benchmarks/rbpf_proposal_reuse.py',
        'benchmarks/bench_rbpf_move_field.py', 'benchmarks/rbpf_move_key_memo.py',
        'benchmarks/rbpf_epoch_memo.py', 'benchmarks/rbpf_move_reuse.py', 'benchmarks/rbpf_move_batch.py')
    sources = {name:sha(ROOT/name) for name in names}
    planpath = own.output_dir.with_suffix('.implementation.json')
    assert not planpath.exists()
    planpath.write_text(json.dumps(dict(reuse_proposals=own.reuse_proposals, source_sha256=sources,
        selection='Explicit process-local caller selection. Default is the D229 staging-key caller.'), indent=2)+'\n', encoding='utf8')

    def selected(runtime, scorer, mode):
        cls = ProposalReuseStep if own.reuse_proposals else MoveKeyMemoStep
        return cls(runtime, scorer, mode)

    previous = field.MoveBatchStep
    field.MoveBatchStep = selected
    if '--reuse-proposals' in sys.argv:
        sys.argv.remove('--reuse-proposals')
    try:
        code = field.main()
    finally:
        field.MoveBatchStep = previous
    assert all(sha(ROOT/name) == digest for name,digest in sources.items())
    (own.output_dir/'implementation_check.json').write_text(json.dumps(dict(
        implementation_plan_sha256=sha(planpath), source_unchanged=True, exit_code=code), indent=2)+'\n', encoding='utf8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
