"""Explicit move-reuse selection around the frozen field diagnostic runner.

Pass --reuse-moves to select MoveReuseStep. The original runner and its graph
configuration remain frozen. A separate implementation plan hashes the new
caller before and after the run, in addition to the original runner's plan.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_move_reuse import MoveReuseStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reuse-moves', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    own, _remaining = parser.parse_known_args()
    names = ('benchmarks/bench_rbpf_move_reuse_field.py', 'benchmarks/rbpf_move_reuse.py',
             'benchmarks/bench_rbpf_move_field.py', 'benchmarks/rbpf_move_batch.py')
    sources = {name: sha(ROOT/name) for name in names}
    planpath = own.output_dir.with_suffix('.implementation.json')
    assert not planpath.exists()
    planpath.write_text(json.dumps(dict(reuse_moves=own.reuse_moves, source_sha256=sources,
        selection='Explicit process-local caller class selection; frozen field runtime/runner files untouched.'), indent=2)+'\n', encoding='utf8')
    previous = field.MoveBatchStep
    if own.reuse_moves:
        field.MoveBatchStep = MoveReuseStep
        sys.argv.remove('--reuse-moves')
    try:
        code = field.main()
    finally:
        field.MoveBatchStep = previous
    assert all(sha(ROOT/name) == digest for name, digest in sources.items())
    (own.output_dir/'implementation_check.json').write_text(json.dumps(dict(
        implementation_plan_sha256=sha(planpath), source_unchanged=True, exit_code=code), indent=2)+'\n', encoding='utf8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
