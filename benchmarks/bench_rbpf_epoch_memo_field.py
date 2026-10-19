"""Explicit epoch memo selection around the unchanged frozen field runner."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_epoch_memo import EpochMemoStep
from rbpf_move_reuse import MoveReuseStep
from rbpf_move_key_memo import MoveKeyMemoStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epoch-memo', action='store_true')
    parser.add_argument('--canonical-only', action='store_true')
    parser.add_argument('--move-key-memo', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    own, _ = parser.parse_known_args()
    if own.canonical_only and not own.epoch_memo:
        raise ValueError('--canonical-only requires explicit --epoch-memo')
    if own.move_key_memo and own.epoch_memo:
        raise ValueError('Choose one explicit memo scope')
    names = ('benchmarks/bench_rbpf_epoch_memo_field.py', 'benchmarks/rbpf_epoch_memo.py',
        'benchmarks/bench_rbpf_move_field.py', 'benchmarks/rbpf_move_reuse.py', 'benchmarks/rbpf_move_batch.py',
        'benchmarks/rbpf_move_key_memo.py')
    sources = {name: sha(ROOT/name) for name in names}
    planpath = own.output_dir.with_suffix('.implementation.json')
    assert not planpath.exists()
    planpath.write_text(json.dumps(dict(epoch_memo=own.epoch_memo, canonical_only=own.canonical_only, move_key_memo=own.move_key_memo,
        source_sha256=sources, selection='Explicit process-local caller selection; frozen runtime and runner untouched.'), indent=2)+'\n', encoding='utf8')

    def selected(runtime, scorer, mode):
        if own.move_key_memo:
            return MoveKeyMemoStep(runtime, scorer, mode)
        if own.epoch_memo:
            return EpochMemoStep(runtime, scorer, mode, cache_dd=not own.canonical_only)
        return MoveReuseStep(runtime, scorer, mode)

    previous = field.MoveBatchStep
    field.MoveBatchStep = selected
    for flag in ('--epoch-memo', '--canonical-only', '--move-key-memo'):
        if flag in sys.argv:
            sys.argv.remove(flag)
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
