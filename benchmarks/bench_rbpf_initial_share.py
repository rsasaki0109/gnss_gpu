"""Complete actual-input score screen for exact initial GPU matrix/LU sharing."""

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_rbpf_irls_replay import prepare, digest
from bench_rbpf_move_field import encode
from rbpf_initial_share import InitialShareScorer
from rbpf_irls_replay import load_corpus, oracle_from_snapshot
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("cpu", "old_gpu", "off_gpu", "on_gpu", "gpu_AA", "cpu_AA", "on_AA")
CONTRASTS = (
    ("gpu_vs_old", "old_gpu", "on_gpu"),
    ("gpu_vs_off", "off_gpu", "on_gpu"),
    ("gpu_vs_cpu", "cpu", "on_gpu"),
    ("gpu_AA", "old_gpu", "gpu_AA"),
    ("cpu_AA", "cpu", "cpu_AA"),
    ("on_AA", "on_gpu", "on_AA"),
    ("off_control", "old_gpu", "off_gpu"),
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summaries(records):
    results = {}
    for ci in (0, 1):
        totals = {}
        for r in records:
            if r["cluster"] == ci:
                key = r["round"], r["group"], r["mode"]
                totals[key] = totals.get(key, 0) + r["ns"]
        contrasts = {}
        for name, a, b in CONTRASTS:
            rounds = [
                statistics.median(
                    100 * (1 - totals[rd, g, b] / totals[rd, g, a]) for g in range(3)
                )
                for rd in range(5)
            ]
            median = statistics.median(rounds)
            positive = sum(v > 0 for v in rounds)
            contrasts[name] = dict(
                reference=a,
                candidate=b,
                round_paired_percent=rounds,
                median_paired_percent=median,
                improving_rounds=positive,
                numeric_screen=median >= 10 and positive >= 4,
            )
        controls = all(
            abs(contrasts[k]["median_paired_percent"]) < 5
            for k in ("gpu_AA", "cpu_AA", "on_AA", "off_control")
        )
        gpu = controls and all(
            contrasts[k]["numeric_screen"] for k in ("gpu_vs_old", "gpu_vs_off")
        )
        results[str(ci)] = dict(
            contrasts=contrasts,
            controls_passed=controls,
            GPU_only_component_passed=gpu,
            GPU_joint_component_passed=gpu
            and contrasts["gpu_vs_cpu"]["numeric_screen"],
            p50_corpus_ns={
                mode: statistics.median(
                    totals[rd, g, mode] for rd in range(5) for g in range(3)
                )
                for mode in MODES
            },
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-dll", "dll", "extension", "parent-extension", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-sha", action="append", required=True)
    args = parser.parse_args()
    assert all(
        os.environ.get(k) == "1"
        for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    )
    inputs = list(zip(args.corpus, args.corpus_sha, strict=True))
    assert len(inputs) == 2 and all(sha(p) == d for p, d in inputs)
    _pin()
    parent = json.loads(
        (ROOT / "dist/rbpf_initial_share_checked_tests_20261017.plan.json").read_text(
            encoding="utf8"
        )
    )
    paths = set(parent["source_sha256"]) | {
        "benchmarks/bench_rbpf_initial_share.py",
        "benchmarks/bench_rbpf_irls_replay.py",
        "benchmarks/bench_rbpf_move_field.py",
        "benchmarks/bench_pf_private_weights_compare.py",
        "dist/rbpf_sparse_normal_measurements_20261017.json",
    }
    paths.update(
        p.relative_to(ROOT).as_posix()
        for p in (
            args.old_dll,
            args.dll,
            args.extension,
            args.parent_extension,
            *args.corpus,
        )
    )
    frozen = {n: sha(ROOT / n) for n in paths}
    reference = ROOT / "dist/rbpf_sparse_normal_measurements_20261017.json"
    assert (
        sha(reference)
        == "690e28b8a43e79aa29022554803a1d4178dcb334178ccf4d696dda0f2a7c6eac"
    )
    gpu_reference = {
        (r["cluster"], r["index"]): r
        for r in json.loads(reference.read_text(encoding="utf8"))["warm"]
        if r["mode"] == "old_gpu"
    }
    assert len(gpu_reference) == 394
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    plan = dict(
        source_and_binary_sha256=frozen,
        modes=MODES,
        contrasts=CONTRASTS,
        rounds=5,
        groups=3,
        cases_per_cluster=197,
        numpy_version=np.__version__,
        python_version=sys.version,
        scorer_contexts=3,
        single_allowed_CPU=True,
        BLAS_threads=1,
        exclusive_host_or_GPU=False,
        gate="All197batches per group,sum integer ns,median3 paired group percentages then median5 rounds. >=10percent and>=4positive rounds versus corrected old GPU AND same-binary off GPU. All four controls GPUAA/CPUAA/onAA/off-vs-old absmedian<5percent. Joint additionally onGPU vs old CPU. Retain all negative/noisy/failing results,no completed retiming.",
        scope="Entire inherited score:dedup/canonical/DD/cluster/packing,host signature/owner grouping and extra owner bytes,upload,template+likelihood launches,completion/readback/scatter/fallback. No usage query/events/extra output in timed calls. Three owning contexts,CPU and GPU serial,normal priority/nonexclusive. Restore/copy/setup/hash/serialization outside. Component only;PF/field trial only after a passing GPU-only component condition,no default adoption or full-FGO/GPUshare inference.",
    )
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    corpora = [load_corpus(p) for p, _ in inputs]
    assert all(
        len(c) == 197 and c[0]["inputs"]["cfg"]["cluster_relin"] == i
        for i, c in enumerate(corpora)
    )
    records, warm, expected = [], [], {}
    with ExitStack() as stack:
        old = stack.enter_context(RegisterLUScorer(args.old_dll))
        off = stack.enter_context(InitialShareScorer(args.dll, enabled=False))
        on = stack.enter_context(InitialShareScorer(args.dll, enabled=True))
        scorers = dict(
            cpu=old,
            old_gpu=old,
            off_gpu=off,
            on_gpu=on,
            gpu_AA=old,
            cpu_AA=old,
            on_AA=on,
        )
        oracles = {
            (ci, mode): oracle_from_snapshot(
                c[0]["inputs"], args.extension, args.parent_extension
            )
            for ci, c in enumerate(corpora)
            for mode in MODES
        }
        snapshots = stack.enter_context(
            gzip.open(
                args.output.with_suffix(".states.jsonl.gz"),
                "wt",
                encoding="utf8",
                newline="\n",
            )
        )
        for ci, corpus in enumerate(corpora):
            for entry in corpus:
                same_backend = {}
                for mode in MODES:
                    oracle = oracles[ci, mode]
                    hs, p = prepare(oracle, entry["inputs"])
                    before = digest(
                        (
                            hs,
                            p,
                            oracle.cfg,
                            oracle._cluster_state.__self__.cluster_states,
                        )
                    )
                    backend = "native_cpu" if mode.startswith("cpu") else "gpu"
                    result = scorers[mode].score(
                        hs, p, oracle, backend, diagnostics=True
                    )
                    full_hash = digest(result)
                    assert same_backend.setdefault(backend, full_hash) == full_hash
                    if backend == "gpu":
                        assert (
                            full_hash
                            == gpu_reference[ci, entry["index"]]["diagnostic_sha256"]
                        )
                    values, details = result
                    compact = (
                        values,
                        {
                            k: v
                            for k, v in details.items()
                            if k not in ("diagnostic", "unique_hypotheses")
                        },
                    )
                    compact_hash = digest(compact)
                    if backend == "native_cpu":
                        assert compact_hash == digest(entry["result"])
                    assert before == digest(
                        (
                            hs,
                            p,
                            oracle.cfg,
                            oracle._cluster_state.__self__.cluster_states,
                        )
                    )
                    usage = (
                        scorers[mode].usage()
                        if mode in ("off_gpu", "on_gpu", "on_AA")
                        else None
                    )
                    if usage is not None:
                        assert usage["count"] == details["unique"]
                        assert (
                            not np.any(usage["used"])
                            if mode == "off_gpu"
                            else np.all((usage["used"] == 0) | (usage["used"] == 1))
                        )
                    expected[ci, entry["index"], mode] = compact_hash
                    row = dict(
                        cluster=ci,
                        index=entry["index"],
                        mode=mode,
                        diagnostic_sha256=full_hash,
                        result_sha256=compact_hash,
                        inputs_unchanged=True,
                        usage=encode(usage),
                    )
                    warm.append(row)
                    snapshots.write(
                        json.dumps(
                            dict(
                                cluster=ci,
                                index=entry["index"],
                                mode=mode,
                                result=encode(result),
                                usage=encode(usage),
                            ),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            snapshots.flush()
            print(json.dumps(dict(warm_complete=ci)), flush=True)
        output = stack.enter_context(
            gzip.open(
                args.output.with_suffix(".calls.jsonl.gz"),
                "wt",
                encoding="utf8",
                newline="\n",
            )
        )
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
                            hs, p = prepare(oracle, entry["inputs"])
                            start = time.perf_counter_ns()
                            result = scorers[mode].score(
                                hs,
                                p,
                                oracle,
                                "native_cpu" if mode.startswith("cpu") else "gpu",
                            )
                            ns = time.perf_counter_ns() - start
                            actual = digest(result)
                            assert actual == expected[ci, entry["index"], mode]
                            row = dict(
                                round=rd,
                                group=group,
                                cluster=ci,
                                index=entry["index"],
                                mode=mode,
                                order=order,
                                ns=ns,
                                result_sha256=actual,
                            )
                            records.append(row)
                            output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
                print(
                    json.dumps(
                        dict(round=rd, group_complete=group, timed_calls=len(records))
                    ),
                    flush=True,
                )
    assert len(records) == 41370 and len(warm) == 2758
    assert all(sha(ROOT / n) == d for n, d in frozen.items())
    result = dict(
        plan_sha256=sha(planpath),
        calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")),
        states_sha256=sha(args.output.with_suffix(".states.jsonl.gz")),
        warm=warm,
        timed_calls=len(records),
        warm_calls=len(warm),
        conditions=summaries(records),
        all_same_backend_score_diagnostic_status_input_bits_equal=True,
        sources_inputs_binaries_unchanged=True,
        timed_calls_have_no_diagnostic_counters=True,
        PF_or_FGO_speed_established=False,
        default_changed=False,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(conditions=result["conditions"])), flush=True)


if __name__ == "__main__":
    main()
