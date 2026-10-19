"""Calibrate original/copy-control/stage-marker whole-score costs on all actual inputs."""

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_rbpf_irls_replay import digest, prepare
from rbpf_irls_replay import load_corpus, oracle_from_snapshot
from rbpf_likelihood_batch import BatchScorer
from rbpf_gpu_stages import StageScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("old_gpu", "original_gpu", "copy_gpu", "trace_gpu", "gpu_AA", "trace_AA")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_cycles(trace, huber):
    """Per-block elapsed differences only; no cross-SM timestamp subtraction."""
    assert trace.dtype == np.uint64 and trace.ndim == 2 and trace.shape[1] == 16
    pairs = dict(initialize=[(0, 1)], prepare=[(1, 2)], normal=[(2, 3)], solve=[(3, 4)])
    if huber:
        pairs["reweight"] = [(4, 5), (7, 8)]
        pairs["normal"] += [(5, 6), (8, 9)]
        pairs["solve"] += [(6, 7), (9, 10)]
    else:
        pairs["reweight"] = []
    pairs.update(likelihood=[(10 if huber else 4, 11)], finish=[(11, 12)], write_result=[(12, 13)])
    visited = list(range(14)) if huber else [0, 1, 2, 3, 4, 11, 12, 13]
    expected = set(visited)
    rows = []
    sentinel = np.iinfo(np.uint64).max
    for r in trace:
        assert int(r[15] >> np.uint64(32)) == 0
        assert {i for i, v in enumerate(r[:14]) if v != sentinel} == expected
        assert all(int(r[b]) >= int(r[a]) for a, b in zip(visited, visited[1:]))
        total = int(r[13]) - int(r[0])
        assert total > 0
        row = {key: sum(int(r[b]) - int(r[a]) for a, b in intervals) for key, intervals in pairs.items()}
        assert sum(row.values()) == total
        rows.append(dict(total=total, **row))
    return rows


def summaries(records):
    result = {}
    for ci in (0, 1):
        totals = {(r, g, m): sum(x["ns"] for x in records
            if (x["cluster"], x["round"], x["group"], x["mode"]) == (ci, r, g, m))
            for r in range(5) for g in range(3) for m in MODES}
        contrasts = {}
        for name, a, b in (("total_profile_overhead", "original_gpu", "trace_gpu"),
            ("markers_vs_copy_control", "copy_gpu", "trace_gpu"),
            ("copy_control_overhead", "original_gpu", "copy_gpu"),
            ("original_control", "old_gpu", "original_gpu"),
            ("gpu_AA", "old_gpu", "gpu_AA"), ("trace_AA", "trace_gpu", "trace_AA")):
            rd = [statistics.median(100 * (totals[r, g, b] / totals[r, g, a] - 1)
                for g in range(3)) for r in range(5)]
            contrasts[name] = dict(round_overhead_percent=rd, paired_overhead_percent=statistics.median(rd))
        calibrated = all(abs(c["paired_overhead_percent"]) < 5 for c in contrasts.values())
        result[str(ci)] = dict(contrasts=contrasts, calibration_passed=calibrated,
            median_corpus_ns={m: statistics.median(totals[r, g, m] for r in range(5) for g in range(3))
                for m in MODES})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-dll", "dll", "extension", "parent-extension", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-sha", required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-sha", action="append", required=True)
    args = parser.parse_args()
    inputs = list(zip(args.corpus, args.corpus_sha, strict=True))
    assert len(inputs) == 2 and all(sha(p) == s for p, s in inputs)
    assert sha(args.reference) == args.reference_sha
    assert all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"))
    _pin()
    paths = [Path(__file__), ROOT / "benchmarks/rbpf_gpu_stages.py",
        ROOT / "benchmarks/bench_rbpf_irls_replay.py", ROOT / "benchmarks/rbpf_irls_replay.py",
        ROOT / "benchmarks/rbpf_likelihood_batch.py", ROOT / "benchmarks/rbpf_native_canonical.py",
        ROOT / "benchmarks/rbpf_parent_score.py", ROOT / "experiments/rbpf_fgo/rbpf_fgo.py",
        *[p for p in (ROOT / "benchmarks/probes/rbpf_gpu_stages").iterdir() if p.is_file()],
        args.old_dll, args.dll, args.extension, args.parent_extension, args.reference, *args.corpus]
    frozen = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    plan = dict(source_and_input_sha256=frozen, modes=MODES, rounds=5, groups=3, cases_per_cluster=197,
        calibration="Whole-score integer-ns sum of197 cases;median3 paired group overhead per round,then median5. Absmedian<5percent for all6 contrasts including AA and same-DLL original. Otherwise uncalibrated diagnostic only;no retiming or optimization adoption claim.",
        scope="Per-block elapsed SM cycles include stalls/time-slicing;not kernel wall time,exclusive operation cost,GPU utilization or whole PF GPU share. No extra math/barriers. Fixed FP64/count/order/prior/RNG.",
        timing="Whole inherited score plus optional profile selection/readback/query/reset. Restore/copy/setup/warm/hash/trace serialization outside. One allowed CPU/BLAS1/normal priority/nonexclusive. Trace data retained for every profile call.")
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    ref = json.loads(args.reference.read_text(encoding="utf8"))
    expected = {(r["cluster"], r["index"]): r for r in ref["warm"] if r["mode"] == "old_gpu"}
    assert len(expected) == 394
    corpora = [load_corpus(p) for p, _ in inputs]
    assert all(len(c) == 197 and c[0]["inputs"]["cfg"]["cluster_relin"] == i for i, c in enumerate(corpora))
    records, warm, batches = [], [], []
    with ExitStack() as stack:
        old = stack.enter_context(BatchScorer(args.old_dll))
        original = stack.enter_context(StageScorer(args.dll))
        copy = stack.enter_context(StageScorer(args.dll))
        trace = stack.enter_context(StageScorer(args.dll))
        scorers = dict(old_gpu=old, original_gpu=original, copy_gpu=copy, trace_gpu=trace,
            gpu_AA=old, trace_AA=trace)
        oracles = {(ci, mode): oracle_from_snapshot(c[0]["inputs"], args.extension, args.parent_extension)
            for ci, c in enumerate(corpora) for mode in MODES}
        raw = stack.enter_context(gzip.open(args.output.with_suffix(".calls.jsonl.gz"), "wt", encoding="utf8", newline="\n"))

        def call(ci, entry, mode, diagnostics):
            oracle = oracles[ci, mode]
            hs, problem = prepare(oracle, entry["inputs"])
            before = digest((hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states)) if diagnostics else None
            start = time.perf_counter_ns()
            if mode in ("copy_gpu", "trace_gpu", "trace_AA"):
                values, detail, stamps = scorers[mode].profiled_score(hs, problem, oracle,
                    diagnostics=diagnostics, control=mode == "copy_gpu")
            else:
                values, detail = scorers[mode].score(hs, problem, oracle, "gpu", diagnostics=diagnostics)
                stamps = None
            ns = time.perf_counter_ns() - start
            if diagnostics:
                assert digest((values, detail)) == expected[ci, entry["index"]]["diagnostic_sha256"]
                assert before == digest((hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states))
                detail = {k: v for k, v in detail.items() if k not in ("diagnostic", "unique_hypotheses")}
            result_sha = digest((values, detail))
            assert result_sha == expected[ci, entry["index"]]["result_sha256"]
            assert detail["scalar_fallbacks"] == 0
            if stamps is not None:
                assert stamps.shape == (detail["unique"], 16)
                blocks = stage_cycles(stamps, oracle.cfg.huber_c > 0)
                fractions = {key: statistics.median(100 * r[key] / r["total"] for r in blocks)
                    for key in blocks[0] if key != "total"}
            else:
                blocks, fractions = [], None
            return dict(ns=ns, result_sha256=result_sha, unique=detail["unique"], rows=len(problem.rows_v0),
                block_count=len(blocks), stage_median_block_percent=fractions), stamps

        for ci, corpus in enumerate(corpora):
            for entry in corpus:
                for mode in MODES:
                    row, stamps = call(ci, entry, mode, True)
                    row.update(phase="warm", cluster=ci, index=entry["index"], mode=mode)
                    warm.append(row)
                    raw.write(json.dumps(dict(**row, trace=None if stamps is None else stamps.tolist()), separators=(",", ":")) + "\n")
            print(json.dumps(dict(warm_complete=ci)), flush=True)
        for rd in range(5):
            for group in range(3):
                for ci, corpus in enumerate(corpora):
                    for entry in corpus:
                        shift = (rd * 3 + group + ci + entry["index"]) % len(MODES)
                        order = MODES[shift:] + MODES[:shift]
                        if (rd + group + ci) % 2:
                            order = order[::-1]
                        for mode in order:
                            row, stamps = call(ci, entry, mode, False)
                            row.update(phase="timed", round=rd, group=group, cluster=ci,
                                index=entry["index"], mode=mode, order=order)
                            records.append(row)
                            raw.write(json.dumps(dict(**row, trace=None if stamps is None else stamps.tolist()), separators=(",", ":")) + "\n")
                raw.flush()
            print(json.dumps(dict(round_complete=rd, timed_calls=len(records))), flush=True)
        for ci in (0, 1):
            selected = [r for r in records if r["cluster"] == ci and r["mode"] in ("trace_gpu", "trace_AA")]
            names = selected[0]["stage_median_block_percent"]
            batches.append(dict(cluster=ci, sampled_calls=len(selected), blocks=sum(r["block_count"] for r in selected),
                median_batch_of_median_block_percent={k: statistics.median(r["stage_median_block_percent"][k] for r in selected)
                    for k in names}))
    assert len(warm) == 2364 and len(records) == 35460
    assert all(sha(ROOT / n) == d for n, d in frozen.items())
    args.output.write_text(json.dumps(dict(plan_sha256=sha(planpath), calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")),
        warm=warm, timed_calls=len(records), conditions=summaries(records), stage_summaries=batches,
        all_same_backend_diagnostics_score_and_warm_input_bits_equal=True, sources_inputs_binaries_unchanged=True,
        exclusive_stage_costs_or_kernel_wall_time_established=False, whole_PF_GPU_percentage_established=False,
        default_changed=False), indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
