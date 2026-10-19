"""Paired whole-score timings on every captured actual batch; component scope."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_rbpf_move_field import encode
from rbpf_irls_replay import ControlledIdentityScorer, load_corpus, oracle_from_snapshot
from rbpf_likelihood_batch import BatchScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("old_cpu", "off_cpu", "on_cpu", "old_gpu", "off_gpu", "on_gpu", "old_cpu_aa")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(encode(value), separators=(",", ":")).encode()
    ).hexdigest()


def prepare(oracle, saved):
    # Restore all mutable numerical inputs outside each score timer.
    view = oracle._cluster_state.__self__
    view.cfg = deepcopy(oracle.cfg)
    vars(view.cfg).clear()
    vars(view.cfg).update(deepcopy(saved["cfg"]))
    oracle.cfg = view.cfg
    view.cluster_states = deepcopy(saved["cluster_states"])
    return deepcopy(saved["hypotheses"]), deepcopy(saved["problem"])


def statistics_for(records):
    result = {}
    for cluster in (0, 1):
        subset = [r for r in records if r["cluster_relin"] == cluster]
        totals = {
            (rd, group, mode): sum(
                r["seconds"]
                for r in subset
                if (r["round"], r["group"], r["mode"]) == (rd, group, mode)
            )
            for rd in range(5)
            for group in range(3)
            for mode in MODES
        }

        def contrast(a, b):
            rounds = [
                statistics.median(
                    100 * (1 - totals[rd, g, b] / totals[rd, g, a]) for g in range(3)
                )
                for rd in range(5)
            ]
            return dict(
                reference=a,
                candidate=b,
                round_paired_percent=rounds,
                median_paired_percent=statistics.median(rounds),
                improving_rounds=sum(x > 0 for x in rounds),
            )

        contrasts = {
            name: contrast(a, b)
            for name, a, b in (
                ("cpu_control", "off_cpu", "on_cpu"),
                ("cpu_prior", "old_cpu", "on_cpu"),
                ("gpu_control", "off_gpu", "on_gpu"),
                ("gpu_prior", "old_gpu", "on_gpu"),
                ("gpu_vs_cpu", "on_cpu", "on_gpu"),
                ("cpu_aa", "old_cpu", "old_cpu_aa"),
            )
        }
        aa_valid = abs(contrasts["cpu_aa"]["median_paired_percent"]) < 5
        for v in contrasts.values():
            v["numeric_screen"] = (
                v["median_paired_percent"] >= 10 and v["improving_rounds"] >= 4
            )
        result[str(cluster)] = dict(
            contrasts=contrasts,
            aa_valid=aa_valid,
            cpu_component_pass=aa_valid
            and all(
                contrasts[k]["numeric_screen"] for k in ("cpu_control", "cpu_prior")
            ),
            gpu_component_pass=aa_valid
            and all(
                contrasts[k]["numeric_screen"]
                for k in ("gpu_control", "gpu_prior", "gpu_vs_cpu")
            ),
            median_corpus_seconds={
                mode: statistics.median(
                    totals[rd, g, mode] for rd in range(5) for g in range(3)
                )
                for mode in MODES
            },
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dll", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--parent-extension", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.corpus) == 2 and not args.output.exists()
    _pin()
    assert all(
        os.environ.get(k) == "1"
        for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    )
    paths = [
        Path(__file__),
        ROOT / "benchmarks/rbpf_irls_replay.py",
        ROOT / "benchmarks/rbpf_irls_identity.py",
        ROOT / "benchmarks/rbpf_likelihood_batch.py",
        ROOT / "benchmarks/rbpf_likelihood_cases.py",
        ROOT / "benchmarks/bench_rbpf_move_field.py",
        ROOT / "benchmarks/rbpf_native_canonical.py",
        ROOT / "benchmarks/rbpf_parent_score.py",
        ROOT / "experiments/rbpf_fgo/rbpf_fgo.py",
        ROOT / "benchmarks/bench_pf_private_weights_compare.py",
        *sorted((ROOT / "benchmarks/probes/rbpf_irls_replay").iterdir()),
        args.prior_dll,
        args.dll,
        args.extension,
        args.parent_extension,
        *args.corpus,
    ]
    frozen = {
        p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in paths if p.is_file()
    }
    corpora = [load_corpus(p) for p in args.corpus]
    assert [r[0]["inputs"]["cfg"]["cluster_relin"] for r in corpora] == [0, 1]
    assert all(len(r) == 197 for r in corpora)
    plan = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_input_binary_sha256=frozen,
        batches_per_corpus=[len(r) for r in corpora],
        modes=MODES,
        rounds=5,
        groups=3,
        selection="All captured batches in order for both original cluster settings; no hit/speed selection.",
        timing="perf_counter_ns around complete score only; includes dedup/packing/native compute/transfers/readback/fallback and on/off completed-counter query/allocation. Input copies/restoration, diagnostics/warmup, equality/hash/counter serialization and setup/control are outside. Synchronous completed results, normal priority, own last allowed CPU, BLAS1, nonexclusive host/GPU; serial calls, no overlap.",
        gate="Per config: median over 5 round medians of 3 paired corpus reductions >=10%, >=4/5 positive, absolute CPU A/A <5%. CPU on must beat same-DLL off and prior. GPU on must beat same-DLL off, priorGPU and onCPU. All exact checks mandatory. Component only; no PF/graph/default adoption.",
        counter_rule="Full per-hypothesis skipped vector saved for EVERY warm and timed on/off call; off must be zero. Active-on counts must equal warm snapshot.",
    )
    planpath = args.output.with_suffix(".plan.json")
    assert not planpath.exists()
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    records, warm = [], []
    with ExitStack() as stack:
        prior = stack.enter_context(BatchScorer(args.prior_dll))
        off = stack.enter_context(ControlledIdentityScorer(args.dll, enabled=False))
        on = stack.enter_context(ControlledIdentityScorer(args.dll, enabled=True))
        scorers = {
            m: prior if m.startswith("old") else on if m.startswith("on") else off
            for m in MODES
        }
        oracles = {
            (ci, m): oracle_from_snapshot(
                c[0]["inputs"], args.extension, args.parent_extension
            )
            for ci, c in enumerate(corpora)
            for m in MODES
        }
        expected = {}
        for ci, corpus in enumerate(corpora):
            for entry in corpus:
                diagnostics = {}
                for mode in MODES:
                    scorer, oracle = scorers[mode], oracles[ci, mode]
                    hs, problem = prepare(oracle, entry["inputs"])
                    before = digest(
                        (
                            hs,
                            problem,
                            oracle.cfg,
                            oracle._cluster_state.__self__.cluster_states,
                        )
                    )
                    value, details = scorer.score(
                        hs,
                        problem,
                        oracle,
                        "gpu" if "gpu" in mode else "native_cpu",
                        diagnostics=True,
                    )
                    diagnostic_sha = digest((value, details))
                    backend = "gpu" if "gpu" in mode else "cpu"
                    if backend in diagnostics:
                        assert diagnostics[backend] == diagnostic_sha
                    else:
                        diagnostics[backend] = diagnostic_sha
                    details = {
                        k: v
                        for k, v in details.items()
                        if k not in ("diagnostic", "unique_hypotheses")
                    }
                    result_sha = digest((value, details))
                    if backend == "cpu":
                        assert result_sha == digest(entry["result"])
                    assert before == digest(
                        (
                            hs,
                            problem,
                            oracle.cfg,
                            oracle._cluster_state.__self__.cluster_states,
                        )
                    )
                    skipped = (
                        [] if mode.startswith("old") else scorer.last_skipped.tolist()
                    )
                    if mode.startswith("off"):
                        assert not any(skipped)
                    expected[ci, entry["index"], mode] = result_sha, skipped
                    warm.append(
                        dict(
                            cluster_relin=ci,
                            index=entry["index"],
                            mode=mode,
                            result_sha256=result_sha,
                            diagnostic_sha256=diagnostic_sha,
                            skipped=skipped,
                        )
                    )
            print(json.dumps(dict(warm_complete=ci, batches=len(corpus))), flush=True)
        stream = stack.enter_context(
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
                            scorer, oracle = scorers[mode], oracles[ci, mode]
                            hs, problem = prepare(oracle, entry["inputs"])
                            start = time.perf_counter_ns()
                            result = scorer.score(
                                hs,
                                problem,
                                oracle,
                                "gpu" if "gpu" in mode else "native_cpu",
                            )
                            elapsed = (time.perf_counter_ns() - start) / 1e9
                            result_sha = digest(result)
                            skipped = (
                                []
                                if mode.startswith("old")
                                else scorer.last_skipped.tolist()
                            )
                            assert (result_sha, skipped) == expected[
                                ci, entry["index"], mode
                            ]
                            record = dict(
                                round=rd,
                                group=group,
                                cluster_relin=ci,
                                index=entry["index"],
                                mode=mode,
                                order=order,
                                seconds=elapsed,
                                result_sha256=result_sha,
                                skipped=skipped,
                            )
                            stream.write(
                                json.dumps(record, separators=(",", ":")) + "\n"
                            )
                            records.append(record)
                stream.flush()
            print(
                json.dumps(dict(round_complete=rd, timed_calls=len(records))),
                flush=True,
            )
    assert all(sha(ROOT / n) == v for n, v in frozen.items())
    args.output.write_text(
        json.dumps(
            dict(
                plan_sha256=sha(planpath),
                calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")),
                warm=warm,
                timed_calls=len(records),
                conditions=statistics_for(records),
                all_same_backend_bits_equal=True,
                every_timed_counter_matches_warm=True,
                source_inputs_binaries_unchanged=True,
                default_changed=False,
                PF_speed_established=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
