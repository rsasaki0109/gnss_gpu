"""Calibrate host statement spans on all saved actual GPU score inputs."""

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
from rbpf_register_lu import RegisterLUScorer
from rbpf_score_host_phases_v2 import HostPhaseScorer, PHASES, spans

ROOT = Path(__file__).resolve().parents[1]
MODES = ("original", "clone", "profile", "original_AA", "profile_AA")
CONTRASTS = (("profile_vs_original", "original", "profile"),
    ("profile_vs_clone", "clone", "profile"), ("clone_control", "original", "clone"),
    ("original_AA", "original", "original_AA"), ("profile_AA", "profile", "profile_AA"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(records):
    result = {}
    for ci in (0, 1):
        totals = {(rd, g, m): sum(r["ns"] for r in records
            if (r["cluster"], r["round"], r["group"], r["mode"]) == (ci, rd, g, m))
            for rd in range(5) for g in range(3) for m in MODES}
        contrasts = {}
        for key, a, b in CONTRASTS:
            rounds = [statistics.median(100*(totals[rd,g,b]/totals[rd,g,a]-1) for g in range(3)) for rd in range(5)]
            contrasts[key] = dict(round_overhead_percent=rounds, paired_overhead_percent=statistics.median(rounds))
        result[str(ci)] = dict(contrasts=contrasts,
            calibrated=all(abs(r["paired_overhead_percent"]) < 5 for r in contrasts.values()),
            median_corpus_ns={m:statistics.median(totals[rd,g,m] for rd in range(5) for g in range(3)) for m in MODES})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "extension", "parent-extension", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-sha", action="append", required=True)
    args = parser.parse_args()
    inputs = list(zip(args.corpus, args.corpus_sha, strict=True))
    assert len(inputs) == 2 and all(sha(p) == d for p,d in inputs)
    assert all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"))
    _pin()
    reference = ROOT / "dist/rbpf_sparse_normal_measurements_20261017.json"
    assert sha(reference) == "690e28b8a43e79aa29022554803a1d4178dcb334178ccf4d696dda0f2a7c6eac"
    expected = {(r["cluster"],r["index"]):r for r in json.loads(reference.read_text(encoding="utf8"))["warm"] if r["mode"] == "old_gpu"}
    assert len(expected) == 394
    paths = [Path(__file__), ROOT/"benchmarks/rbpf_score_host_phases_v2.py", ROOT/"benchmarks/rbpf_likelihood_batch.py",
        ROOT/"benchmarks/rbpf_register_lu.py", ROOT/"benchmarks/bench_rbpf_irls_replay.py", ROOT/"benchmarks/rbpf_irls_replay.py",
        ROOT/"benchmarks/rbpf_native_canonical.py", ROOT/"benchmarks/rbpf_parent_score.py", ROOT/"experiments/rbpf_fgo/rbpf_fgo.py",
        args.dll,args.extension,args.parent_extension,reference,*args.corpus]
    frozen = {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    plan = dict(source_and_input_sha256=frozen, modes=MODES, phases=PHASES, rounds=5, groups=3, cases_per_cluster=197,
        calibration="All197case integer-ns sums;median3 paired group overheads per round then median5. Every one of5 contrasts absmedian<5percent,otherwise condition uncalibrated. Same code-object clone,original/profile AA. No completed timing retry or optimization adoption.",
        scope="Host wall spans include callbacks,ctypes validation,native transfers/compute/waits,scheduling and marker overhead. Native span is not pure kernel time,CPU cycles,GPU utilization or PF share. No new CUDA code or numeric changes. Per-call fractions then medians across all profiled calls;all raw events retained.",
        timing="Complete score call including trace publication. Input restore/setup/warm/hash/span validation/serialization outside. Allowed CPU1/BLAS1/normal priority/nonexclusive.")
    planpath.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8")
    corpora = [load_corpus(p) for p,_ in inputs]
    assert all(len(c) == 197 and c[0]["inputs"]["cfg"]["cluster_relin"] == ci for ci,c in enumerate(corpora))
    warm, records = [], []
    with ExitStack() as stack:
        original = stack.enter_context(RegisterLUScorer(args.dll))
        clone = stack.enter_context(HostPhaseScorer(args.dll, measure=False))
        profile = stack.enter_context(HostPhaseScorer(args.dll))
        scorers = dict(original=original, clone=clone, profile=profile, original_AA=original, profile_AA=profile)
        oracles = {(ci,m):oracle_from_snapshot(c[0]["inputs"],args.extension,args.parent_extension) for ci,c in enumerate(corpora) for m in MODES}
        raw = stack.enter_context(gzip.open(args.output.with_suffix(".calls.jsonl.gz"),"wt",encoding="utf8",newline="\n"))

        def call(ci, entry, mode, diagnostics):
            oracle = oracles[ci,mode]
            hs, problem = prepare(oracle,entry["inputs"])
            before = digest((hs,problem,oracle.cfg,oracle._cluster_state.__self__.cluster_states)) if diagnostics else None
            start = time.perf_counter_ns()
            values, details = scorers[mode].score(hs,problem,oracle,"gpu",diagnostics=diagnostics)
            elapsed = time.perf_counter_ns()-start
            dsha = digest((values,details)) if diagnostics else None
            if diagnostics:
                assert dsha == expected[ci,entry["index"]]["diagnostic_sha256"]
                assert before == digest((hs,problem,oracle.cfg,oracle._cluster_state.__self__.cluster_states))
                details = {k:v for k,v in details.items() if k not in ("diagnostic","unique_hypotheses")}
            result_sha = digest((values,details))
            assert result_sha == expected[ci,entry["index"]]["result_sha256"] and details["scalar_fallbacks"] == 0
            marked = mode in ("profile","profile_AA")
            events = profile.last_profile if marked else None
            partition = spans(events) if marked else None
            if partition is not None:
                assert tuple(n for n,_ in events[:-1]) == PHASES
                assert 0 < partition["total"] <= elapsed
            return dict(ns=elapsed, result_sha256=result_sha, diagnostic_sha256=dsha,
                warm_inputs_equal=diagnostics, unique=details["unique"], input_rows=len(problem.rows_v0),
                partition=partition, trace=events)

        for ci, corpus in enumerate(corpora):
            for entry in corpus:
                for mode in MODES:
                    row = dict(phase="warm",cluster=ci,index=entry["index"],mode=mode,**call(ci,entry,mode,True))
                    warm.append(row)
                    raw.write(json.dumps(row,separators=(",",":"))+"\n")
            print(json.dumps(dict(warm_complete=ci)),flush=True)
        for rd in range(5):
            for group in range(3):
                for ci, corpus in enumerate(corpora):
                    for entry in corpus:
                        shift = (rd*3+group+ci+entry["index"]) % len(MODES)
                        order = MODES[shift:]+MODES[:shift]
                        if (rd+group+ci) % 2:
                            order = order[::-1]
                        for mode in order:
                            row = dict(phase="timed",round=rd,group=group,cluster=ci,index=entry["index"],mode=mode,
                                order=order,**call(ci,entry,mode,False))
                            records.append(row)
                            raw.write(json.dumps(row,separators=(",",":"))+"\n")
                raw.flush()
            print(json.dumps(dict(round_complete=rd,timed_calls=len(records))),flush=True)
    phase_results = []
    for ci in (0,1):
        selected = [r for r in records if r["cluster"] == ci and r["partition"] is not None]
        phase_results.append(dict(cluster=ci,calls=len(selected),median_call_fraction_percent={k:statistics.median(
            100*r["partition"][k]/r["partition"]["total"] for r in selected) for k in PHASES},
            median_recorded_span_ns=statistics.median(r["partition"]["total"] for r in selected)))
    assert len(warm) == 1970 and len(records) == 29550
    assert all(sha(ROOT/n) == d for n,d in frozen.items())
    args.output.write_text(json.dumps(dict(plan_sha256=sha(planpath),calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")),
        warm=warm,timed_calls=len(records),conditions=summaries(records),phase_summaries=phase_results,
        all_warm_diagnostics_input_bits_and_all_timed_results_exact=True,source_inputs_binaries_frozen=True,
        native_span_is_pure_GPU_time=False,whole_PF_share_or_speed_established=False,default_changed=False),indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
