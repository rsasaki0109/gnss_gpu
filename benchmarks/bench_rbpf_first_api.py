"""One fresh-process first-operation diagnostic,one workspace,one idle gap."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_rbpf_irls_replay import digest, prepare
from rbpf_first_api import FirstApiScorer, OPERATIONS, parse_probe
from rbpf_irls_replay import load_corpus, oracle_from_snapshot
from rbpf_pair_residuals import ScoreObserver

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    trial = plan["trials"][args.trial]
    assert trial["trial"] == args.trial and trial["operation"] in OPERATIONS
    output = ROOT / trial["output"]
    assert not output.exists()
    _pin()
    entry = load_corpus(ROOT / plan["corpora"][0])[0]
    assert entry["index"] == 0 and entry["inputs"]["cfg"]["cluster_relin"] == 0
    oracle = oracle_from_snapshot(entry["inputs"], ROOT / plan["binaries"]["CANONICAL_EXT"]["path"],
        ROOT / plan["binaries"]["PARENT_EXT"]["path"])
    reference = json.loads((ROOT / plan["reference"]).read_text(encoding="utf8"))
    expected = next(r["result_sha256"] for r in reference["warm"] if r["mode"] == "old_gpu" and r["cluster"] == r["index"] == 0)
    hs, problem = prepare(oracle, entry["inputs"])
    data = (hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states)
    input_before = digest(data)
    rows = []
    with FirstApiScorer(ROOT / plan["binaries"]["DLL"]["path"]) as scorer:
        observer = ScoreObserver(lambda:("A",203))
        score = observer.bind(scorer)

        def keep(label, result):
            assert digest(result) == expected and result[1]["scalar_fallbacks"] == 0
            clocks, native = observer.records[-1], scorer.native_records[-1]
            assert clocks["completed"] and clocks["probe_error"] is native["probe_error"] is None
            assert native["trace"]["full_GPU_path"]
            assert native["trace"]["totals"]["wall_ticks"]*10**9 <= clocks["partition"]["native"]*native["trace"]["frequency"]
            rows.append(dict(label=label,clocks=clocks,native=native,result_sha256=expected))

        keep("warm",score(hs,problem,oracle,"gpu"))
        scorer.prepare_probe()
        idle_start = time.perf_counter_ns()
        time.sleep(15)
        idle_ns = time.perf_counter_ns()-idle_start
        combined_start = time.perf_counter_ns()
        raw = scorer.probe_raw(trial["operation"])
        probe_end = time.perf_counter_ns()
        result = score(hs,problem,oracle,"gpu")
        combined_end = time.perf_counter_ns()
        probe = parse_probe(raw)
        keep("after_probe",result)
        keep("immediate",score(hs,problem,oracle,"gpu"))
        assert digest(data) == input_before and idle_ns >= 15*10**9
        assert probe["total"]["wall_ticks"]*10**9 <= (probe_end-combined_start)*probe["frequency"]
        assert rows[1]["clocks"]["wall_ns"] <= combined_end-probe_end
    result = dict(verified=True,trial=trial,records=rows,score_calls=3,probe=probe,
        timeline=dict(combined_start_ns=combined_start,probe_end_ns=probe_end,combined_end_ns=combined_end,
            probe_wall_ns=probe_end-combined_start,following_score_interval_ns=combined_end-probe_end,
            combined_wall_ns=combined_end-combined_start,idle_ns=idle_ns),
        all_input_and_result_bits_exact=True,input_sha256=input_before,
        environment=dict(pid=os.getpid(),parent_pid=os.getppid(),Python=sys.version,executable=sys.executable,NumPy=np.__version__),
        new_process_one_workspace=True,no_external_profiler=True,optimization_speed_gain_established=False,
        scope="New first-operation diagnosis,not RTK speed. All CUDA work/completion is inside probe;the original following score fresh-packs/uploads all active inputs. Combined time retains probe plus completed score,so moving a wait is not a speedup. Initial warm and immediate calls also retained. One declared15sidle per process,no retries/power/keepalive/default/model change.")
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(dict(verified=True,trial=args.trial,operation=trial["operation"],
        probe_ms=result["timeline"]["probe_wall_ns"]/1e6,combined_ms=result["timeline"]["combined_wall_ns"]/1e6)),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
