"""Replay every frozen actual likelihood input for CUDA timeline attribution.

This driver changes no numerical code. Use an external process-scoped profiler;
the JSON order maps one original likelihood kernel to each checked score call.
Profiler overhead must be calibrated separately before attributing wall time.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from bench_pf_private_weights_compare import _pin
from bench_rbpf_irls_replay import digest, prepare
from rbpf_irls_replay import load_corpus, oracle_from_snapshot
from rbpf_likelihood_batch import BatchScorer

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "extension", "parent-extension", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-sha", required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-sha", action="append", required=True)
    parser.add_argument("--limit", type=int, default=394)
    args = parser.parse_args()
    assert 1 <= args.limit <= 394
    inputs = list(zip(args.corpus, args.corpus_sha, strict=True))
    assert len(inputs) == 2 and all(sha(p) == d for p, d in inputs)
    assert sha(args.reference) == args.reference_sha
    assert all(os.environ.get(k) == "1" for k in
        ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"))
    _pin()
    source_paths = [Path(__file__), ROOT / "benchmarks/bench_rbpf_irls_replay.py",
        ROOT / "benchmarks/bench_rbpf_move_field.py", ROOT / "benchmarks/bench_pf_private_weights_compare.py",
        ROOT / "benchmarks/rbpf_irls_replay.py", ROOT / "benchmarks/rbpf_likelihood_batch.py",
        ROOT / "benchmarks/rbpf_native_canonical.py", ROOT / "benchmarks/rbpf_parent_score.py",
        ROOT / "experiments/rbpf_fgo/rbpf_fgo.py", args.dll, args.extension, args.parent_extension,
        args.reference, *args.corpus]
    frozen = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    plan_path = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not plan_path.exists()
    plan = dict(source_and_input_sha256=frozen, cases=args.limit,
        order="All warm diagnostic calls in cluster/index order,then all timed calls in the same order.",
        timer="Whole inherited score:dedup,DD,cluster,pack,guards,all copies,kernel,sync,scatter,fallback.",
        outside_timer="Setup,restore/copy,warm,result hashing,input hashing,serialization.",
        scope="Diagnostic replay only;no candidate/default/numerical changes or PF/FGO speed claim.")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    reference = json.loads(args.reference.read_text(encoding="utf8"))
    expected = {(r["cluster"], r["index"]): r for r in reference["warm"] if r["mode"] == "old_gpu"}
    assert len(expected) == 394
    corpora = [load_corpus(p) for p, _ in inputs]
    assert all(len(c) == 197 and c[0]["inputs"]["cfg"]["cluster_relin"] == i
        for i, c in enumerate(corpora))
    cases = [(ci, entry) for ci, corpus in enumerate(corpora) for entry in corpus][:args.limit]
    rows = []
    with BatchScorer(args.dll) as scorer:
        oracles = {i: oracle_from_snapshot(c[0]["inputs"], args.extension, args.parent_extension)
            for i, c in enumerate(corpora)}
        for phase in ("warm", "timed"):
            for ci, entry in cases:
                oracle = oracles[ci]
                hs, problem = prepare(oracle, entry["inputs"])
                before = digest((hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states))
                start = time.perf_counter_ns()
                values, details = scorer.score(hs, problem, oracle, "gpu", diagnostics=phase == "warm")
                elapsed = time.perf_counter_ns() - start
                if phase == "warm":
                    assert digest((values, details)) == expected[ci, entry["index"]]["diagnostic_sha256"]
                    details = {k: v for k, v in details.items() if k not in ("diagnostic", "unique_hypotheses")}
                result_sha = digest((values, details))
                assert result_sha == expected[ci, entry["index"]]["result_sha256"]
                assert details["scalar_fallbacks"] == 0 and details["unique"] > 0
                assert before == digest((hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states))
                rows.append(dict(ordinal=len(rows), phase=phase, cluster=ci, index=entry["index"],
                    hypotheses=len(hs), unique=int(details["unique"]), rows=len(problem.rows_v0),
                    pairs=len(problem.pairs), ns=elapsed, result_sha256=result_sha,
                    inputs_unchanged=True, scalar_fallbacks=0))
            print(json.dumps(dict(phase_complete=phase, cases=len(cases))), flush=True)
    assert all(sha(ROOT / name) == d for name, d in frozen.items())
    result = dict(plan_sha256=sha(plan_path), records=rows, sources_inputs_binaries_unchanged=True,
        full_diagnostics_equal_to_frozen_gpu_reference=True, all_result_and_input_bits_equal=True,
        timed_corpus_ns={str(ci): sum(r["ns"] for r in rows if r["phase"] == "timed" and r["cluster"] == ci)
            for ci in (0, 1)}, default_changed=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
