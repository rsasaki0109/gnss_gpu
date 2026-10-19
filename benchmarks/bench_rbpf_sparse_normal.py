"""Controlled whole-score GPU structural-zero comparison on every actual saved input."""

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from bench_pf_private_weights_compare import _pin
from bench_rbpf_irls_replay import digest, prepare
from rbpf_irls_replay import load_corpus, oracle_from_snapshot
from rbpf_sparse_normal import SparseNormalScorer
from rbpf_likelihood_batch import BatchScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("cpu", "old_gpu", "original_gpu", "dense_gpu", "sparse_gpu", "gpu_AA", "cpu_AA", "sparse_AA")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(records):
    results = {}
    for ci in (0, 1):
        totals = {(rd, g, mode): sum(r["ns"] for r in records if
            (r["cluster"], r["round"], r["group"], r["mode"]) == (ci, rd, g, mode))
            for rd in range(5) for g in range(3) for mode in MODES}
        contrasts = {}
        for name, a, b in (("sparse_vs_old_gpu", "old_gpu", "sparse_gpu"),
            ("sparse_vs_original_gpu", "original_gpu", "sparse_gpu"),
            ("sparse_vs_dense_gpu", "dense_gpu", "sparse_gpu"),
            ("sparse_vs_cpu", "cpu", "sparse_gpu"), ("gpu_AA", "old_gpu", "gpu_AA"),
            ("cpu_AA", "cpu", "cpu_AA"), ("sparse_AA", "sparse_gpu", "sparse_AA"),
            ("original_control", "old_gpu", "original_gpu"), ("dense_control", "old_gpu", "dense_gpu")):
            values = [statistics.median(100 * (1 - totals[rd, g, b] / totals[rd, g, a])
                for g in range(3)) for rd in range(5)]
            contrasts[name] = dict(round_reduction_percent=values,
                paired_reduction_percent=statistics.median(values), improving_rounds=sum(x > 0 for x in values))
        controls = all(abs(contrasts[k]["paired_reduction_percent"]) < 5
            for k in ("gpu_AA", "cpu_AA", "sparse_AA", "original_control"))
        speed = all(contrasts[k]["paired_reduction_percent"] >= 10 and contrasts[k]["improving_rounds"] >= 4
            for k in ("sparse_vs_old_gpu", "sparse_vs_original_gpu", "sparse_vs_dense_gpu", "sparse_vs_cpu"))
        results[str(ci)] = dict(contrasts=contrasts, controls_passed=controls,
            GPU_joint_component_passed=controls and speed,
            median_corpus_ns={m: statistics.median(totals[r, g, m] for r in range(5) for g in range(3)) for m in MODES})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-dll", "dll", "extension", "parent-extension", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-sha", action="append", required=True)
    args = parser.parse_args()
    inputs = list(zip(args.corpus, args.corpus_sha, strict=True))
    assert len(inputs) == 2 and all(sha(p) == d for p, d in inputs)
    assert all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"))
    _pin()
    sources = [Path(__file__), ROOT / "benchmarks/rbpf_sparse_normal.py",
        ROOT / "benchmarks/bench_rbpf_irls_replay.py", ROOT / "benchmarks/rbpf_irls_replay.py",
        ROOT / "benchmarks/rbpf_likelihood_batch.py", ROOT / "benchmarks/rbpf_native_canonical.py",
        ROOT / "benchmarks/rbpf_parent_score.py", ROOT / "experiments/rbpf_fgo/rbpf_fgo.py",
        *[p for p in (ROOT / "benchmarks/probes/rbpf_sparse_normal").iterdir() if p.is_file()],
        args.old_dll, args.dll, args.extension, args.parent_extension, *args.corpus]
    frozen = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    plan = dict(source_and_input_sha256=frozen, modes=MODES, rounds=5, groups=3, cases_per_cluster=197,
        gate="All197batches per group,integer ns sums;within-round median3 paired percentages then median5. Sparse must beat old GPU,same-binary original GPU,new dense-control GPU,and optimized CPU by>=10percent with>=4of5improving;GPU AA,CPU AA,sparse AA,original control each absmedian<5percent. New dense kernel has different code/resources;its old-GPU contrast is diagnostic,not an equivalence control. Every case/raw/failure retained,no retries. Component only.",
        scope="Whole inherited score including dedup/DD/cluster/packing/upload/kernel/completion/readback/scatter/fallback. Restore/copy/setup/warm/hash outside timing. One allowed CPU/BLAS1/normal priority/nonexclusive. No events in timed calls. Group-index/finite guards inside timer;separate post-timing operation counts;not PF/FGO speed or GPU utilization.")
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    corpora = [load_corpus(p) for p, _ in inputs]
    assert all(len(c) == 197 and c[0]["inputs"]["cfg"]["cluster_relin"] == i for i, c in enumerate(corpora))
    warm, records = [], []
    with ExitStack() as stack:
        cpu = stack.enter_context(BatchScorer(args.old_dll))
        old = stack.enter_context(BatchScorer(args.old_dll))
        original = stack.enter_context(SparseNormalScorer(args.dll, kernel="original"))
        dense = stack.enter_context(SparseNormalScorer(args.dll, kernel="dense"))
        sparse = stack.enter_context(SparseNormalScorer(args.dll, kernel="sparse"))
        scorers = dict(cpu=cpu, old_gpu=old, original_gpu=original, dense_gpu=dense, sparse_gpu=sparse,
            gpu_AA=old, cpu_AA=cpu, sparse_AA=sparse)
        oracles = {(i, m): oracle_from_snapshot(c[0]["inputs"], args.extension, args.parent_extension)
            for i, c in enumerate(corpora) for m in MODES}
        expected = {}
        for ci, corpus in enumerate(corpora):
            for entry in corpus:
                diagnostics = {}
                for mode in MODES:
                    oracle = oracles[ci, mode]
                    hs, problem = prepare(oracle, entry["inputs"])
                    before = digest((hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states))
                    backend = "native_cpu" if mode.startswith("cpu") else "gpu"
                    values, details = scorers[mode].score(hs, problem, oracle, backend, diagnostics=True)
                    dsha = digest((values, details))
                    assert diagnostics.setdefault(backend, dsha) == dsha
                    details = {k: v for k, v in details.items() if k not in ("diagnostic", "unique_hypotheses")}
                    result_sha = digest((values, details))
                    if backend == "native_cpu":
                        assert result_sha == digest(entry["result"])
                    assert before == digest((hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states))
                    expected[ci, entry["index"], mode] = result_sha
                    warm.append(dict(cluster=ci, index=entry["index"], mode=mode, result_sha256=result_sha,
                        diagnostic_sha256=dsha, inputs_unchanged=True))
            print(json.dumps(dict(warm_complete=ci)), flush=True)
        output = stack.enter_context(gzip.open(args.output.with_suffix(".calls.jsonl.gz"), "wt", encoding="utf8", newline="\n"))
        for rd in range(5):
            for group in range(3):
                for ci, corpus in enumerate(corpora):
                    for entry in corpus:
                        shift = (rd * 3 + group + ci + entry["index"]) % len(MODES)
                        order = MODES[shift:] + MODES[:shift]
                        if (rd + group + ci) % 2:
                            order = order[::-1]
                        for mode in order:
                            oracle = oracles[ci, mode]
                            hs, problem = prepare(oracle, entry["inputs"])
                            start = time.perf_counter_ns()
                            result = scorers[mode].score(hs, problem, oracle,
                                "native_cpu" if mode.startswith("cpu") else "gpu")
                            ns = time.perf_counter_ns() - start
                            actual = digest(result)
                            assert actual == expected[ci, entry["index"], mode]
                            row = dict(round=rd, group=group, cluster=ci, index=entry["index"], mode=mode,
                                order=order, ns=ns, result_sha256=actual)
                            records.append(row)
                            output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
            print(json.dumps(dict(round_complete=rd, timed_calls=len(records))), flush=True)
        counts = []
        for ci, corpus in enumerate(corpora):
            for entry in corpus:
                pair = {}
                for mode, scorer in (("dense_gpu", dense), ("sparse_gpu", sparse)):
                    oracle = oracles[ci, mode]
                    hs, problem = prepare(oracle, entry["inputs"])
                    values, details, count = scorer.count_score(hs, problem, oracle)
                    assert digest((values, details)) == expected[ci, entry["index"], mode]
                    assert count is not None
                    counts.append(dict(cluster=ci, index=entry["index"], mode=mode,
                        result_sha256=digest((values, details)), **count))
                    pair[mode] = count
                a, b = pair["dense_gpu"], pair["sparse_gpu"]
                assert a["sparse_stages"] == a["skipped_products"] == a["unsafe_stages"] == 0
                assert a["dense_stages"] == b["dense_stages"] + b["sparse_stages"]
    assert len(records) == 47280 and len(warm) == 3152 and len(counts) == 788
    attribution = {str(ci): dict(sparse_stages=sum(r["sparse_stages"] for r in counts if r["cluster"] == ci),
        skipped_products=sum(r["skipped_products"] for r in counts if r["cluster"] == ci),
        unsafe_stages=sum(r["unsafe_stages"] for r in counts if r["cluster"] == ci)) for ci in (0, 1)}
    assert all(r["sparse_stages"] > 0 and r["skipped_products"] > 0 for r in attribution.values())
    assert all(sha(ROOT / n) == v for n, v in frozen.items())
    result = dict(plan_sha256=sha(planpath), calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")),
        warm=warm, timed_calls=len(records), operation_counts=counts, attribution=attribution, conditions=summaries(records),
        all_same_backend_score_diagnostic_status_input_bits_equal=True, sources_inputs_binaries_unchanged=True,
        timed_calls_have_no_diagnostic_counters=True, PF_or_FGO_speed_established=False, default_changed=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(conditions=result["conditions"])), flush=True)


if __name__ == "__main__":
    main()
