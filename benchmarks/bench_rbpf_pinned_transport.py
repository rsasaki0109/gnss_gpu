"""Frozen full-score steady/15s-idle transport screens,all outcomes retained."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_rbpf_irls_replay import digest, prepare
from rbpf_irls_replay import load_corpus, oracle_from_snapshot
from rbpf_native_call_trace import NativeCallScorer
from rbpf_pair_residuals import ScoreObserver
from rbpf_pinned_transport import PinnedTransportScorer, TracedPinnedTransportScorer
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("old", "pageable", "pinned", "old_AA")


def ordered(round_, group=0, case=0):
    offset = (round_ + group + case) % len(MODES)
    order = MODES[offset:] + MODES[:offset]
    return order if (round_ + group) % 2 == 0 else order[::-1]


def contrasts(totals, groups):
    result = {}
    for name, a, b in (("pinned_vs_old", "old", "pinned"),
        ("pinned_vs_pageable", "pageable", "pinned"), ("old_AA", "old", "old_AA"),
        ("pageable_control", "old", "pageable")):
        rounds = [median(100.0*(1-totals[rd,g,b]/totals[rd,g,a]) for g in range(groups)) for rd in range(5)]
        result[name] = dict(round_reduction_percent=rounds, paired_reduction_percent=median(rounds),
            improving_rounds=sum(v>0 for v in rounds), rounds_abs_under5=sum(abs(v)<5 for v in rounds))
    controls = all(abs(result[name]["paired_reduction_percent"])<5 and result[name]["rounds_abs_under5"]>=4
        for name in ("old_AA", "pageable_control"))
    improvement = all(result[name]["paired_reduction_percent"]>=10 and result[name]["improving_rounds"]>=4
        for name in ("pinned_vs_old", "pinned_vs_pageable"))
    return dict(contrasts=result, controls_passed=controls, component_screen_passed=controls and improvement)


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    _pin()
    out = ROOT / plan["screen_output"]
    assert not out.exists()
    corpora = [load_corpus(ROOT / name) for name in plan["corpora"]]
    assert all(len(c)==197 and c[0]["inputs"]["cfg"]["cluster_relin"]==i for i,c in enumerate(corpora))
    records, warm, expected = [], [], {}
    with ExitStack() as stack:
        scorers = {mode:stack.enter_context(RegisterLUScorer(ROOT / plan["old_baseline_DLL"]["path"]))
            if mode in ("old","old_AA") else stack.enter_context(PinnedTransportScorer(ROOT / plan["binaries"]["DLL"]["path"],transport=mode))
            for mode in MODES}
        oracles = {(ci,mode):oracle_from_snapshot(corpus[0]["inputs"], ROOT / plan["binaries"]["CANONICAL_EXT"]["path"],
            ROOT / plan["binaries"]["PARENT_EXT"]["path"]) for ci,corpus in enumerate(corpora) for mode in MODES}
        for ci,corpus in enumerate(corpora):
            for entry in corpus:
                for mode in MODES:
                    oracle = oracles[ci,mode]
                    hs,problem = prepare(oracle,entry["inputs"])
                    before = digest((hs,problem,oracle.cfg,oracle._cluster_state.__self__.cluster_states))
                    result = scorers[mode].score(hs,problem,oracle,"gpu")
                    assert result[1]["scalar_fallbacks"]==0
                    result_digest = digest(result)
                    if mode=="old":
                        expected[ci,entry["index"]]=result_digest
                    assert expected[ci,entry["index"]]==result_digest
                    assert before==digest((hs,problem,oracle.cfg,oracle._cluster_state.__self__.cluster_states))
                    warm.append(dict(cluster=ci,index=entry["index"],mode=mode,result_sha256=result_digest,input_unchanged=True))
        for rd in range(5):
            for group in range(3):
                for ci,corpus in enumerate(corpora):
                    for entry in corpus:
                        for mode in ordered(rd,group,entry["index"]+ci):
                            oracle = oracles[ci,mode]
                            hs,problem = prepare(oracle,entry["inputs"])
                            score = scorers[mode].score
                            start = time.perf_counter_ns()
                            result = score(hs,problem,oracle,"gpu")
                            ns = time.perf_counter_ns()-start
                            result_digest = digest(result)
                            assert result_digest==expected[ci,entry["index"]] and result[1]["scalar_fallbacks"]==0
                            records.append(dict(cluster=ci,round=rd,group=group,index=entry["index"],mode=mode,ns=ns,result_sha256=result_digest))
            print(json.dumps(dict(stage="steady",round=rd,completed_calls=len(records))),flush=True)
        steady = {}
        for ci in (0,1):
            totals = {(rd,g,m):sum(r["ns"] for r in records if (r["cluster"],r["round"],r["group"],r["mode"])==(ci,rd,g,m))
                for rd in range(5) for g in range(3) for m in MODES}
            steady[str(ci)] = dict(contrasts(totals,3),median_corpus_ns={m:median(totals[rd,g,m] for rd in range(5) for g in range(3)) for m in MODES})
        save(out,dict(verified=True,stage="steady",records=records,warm=warm,summary=steady,
            environment=dict(Python=sys.version,NumPy=np.__version__,executable=sys.executable),
            scope="Whole original unmarked score,all197inputs/cluster,5rounds3groups4modes,all raw. Setup/restore/hash outside timers. New D287 selected host allocation only;no profiler/trace/events/keepalive/power changes. Component screen,not RTK or GPU utilization.",
            timed_calls=len(records),warm_calls=len(warm),all_bits_exact=True))

        cold, coldwarm = [], []
        for mode in MODES:
            oracle = oracles[0,mode]
            hs,problem = prepare(oracle,corpora[0][0]["inputs"])
            result = scorers[mode].score(hs,problem,oracle,"gpu")
            assert digest(result)==expected[0,0]
            coldwarm.append(dict(mode=mode,result_sha256=digest(result)))
        for rd in range(5):
            for mode in ordered(rd):
                oracle = oracles[0,mode]
                hs,problem = prepare(oracle,corpora[0][0]["inputs"])
                before = digest((hs,problem,oracle.cfg,oracle._cluster_state.__self__.cluster_states))
                score = scorers[mode].score
                idle_start = time.perf_counter_ns()
                time.sleep(15)
                idle_ns = time.perf_counter_ns()-idle_start
                cpu_start = time.thread_time_ns()
                start = time.perf_counter_ns()
                result = score(hs,problem,oracle,"gpu")
                ns = time.perf_counter_ns()-start
                cpu_ns = time.thread_time_ns()-cpu_start
                result_digest = digest(result)
                assert idle_ns>=15*10**9 and result_digest==expected[0,0] and result[1]["scalar_fallbacks"]==0
                assert before==digest((hs,problem,oracle.cfg,oracle._cluster_state.__self__.cluster_states))
                row = dict(round=rd,mode=mode,ns=ns,thread_cpu_ns=cpu_ns,idle_ns=idle_ns,
                    result_sha256=result_digest,input_unchanged=True)
                cold.append(row)
                print(json.dumps(dict(stage="idle",**row)),flush=True)
        totals = {(r["round"],0,r["mode"]):r["ns"] for r in cold}
        summary = contrasts(totals,1)
        save(ROOT / plan["cold_output"],dict(verified=True,records=cold,warm=coldwarm,summary=summary,
            timed_calls=20,all_bits_exact=True,environment=dict(Python=sys.version,NumPy=np.__version__,executable=sys.executable),
            scope="All20unmarked completed scores of one frozen cluster0input;each own15sidle,rotated/reversed order,5rounds4modes. Two independent old controls. Host/driver nonexclusive. Exact same gates as declared before execution. No removed outliers/negative rounds/retries. Component latency,not sustained RTK speed."))

    traces = []
    for transport in ("pageable","pinned"):
        with TracedPinnedTransportScorer(ROOT / plan["binaries"]["DLL"]["path"],transport=transport) as scorer:
            oracle = oracle_from_snapshot(corpora[0][0]["inputs"],ROOT / plan["binaries"]["CANONICAL_EXT"]["path"],ROOT / plan["binaries"]["PARENT_EXT"]["path"])
            observer = ScoreObserver(lambda:("A",203))
            score = observer.bind(scorer)
            for index,label in enumerate(("initial","after_idle15","immediate")):
                hs,problem = prepare(oracle,corpora[0][0]["inputs"])
                if index==1:
                    time.sleep(15)
                result = score(hs,problem,oracle,"gpu")
                assert digest(result)==expected[0,0]
                native = scorer.native_records[-1]
                assert native["probe_error"] is None and native["trace"]["full_GPU_path"]
                traces.append(dict(transport=transport,label=label,clocks=observer.records[-1],native=native,result_sha256=digest(result)))
    save(ROOT / plan["transport_trace_output"],dict(verified=True,records=traces,score_calls=6,all_bits_exact=True,
        scope="Separate post-screen host-clock diagnosis only,not a speed gate. For each new transport3calls:initial,after15sidle,immediate. No external profiler/no additional GPU work."))
    print(json.dumps(dict(complete=True,steady=steady,cold=summary)),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
