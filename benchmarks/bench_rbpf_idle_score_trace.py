"""Four frozen actual-batch scores,one declared idle gap;diagnostic only."""

import argparse
import json
from pathlib import Path
import sys
import time

from bench_pf_private_weights_compare import _pin
from bench_rbpf_irls_replay import digest, prepare
from rbpf_irls_replay import load_corpus, oracle_from_snapshot
from rbpf_pair_residuals import ScoreObserver
from rbpf_score_host_phases_v2 import HostPhaseScorer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    _pin()
    entry = load_corpus(ROOT / plan["corpus"])[0]
    assert entry["index"] == 0 and entry["inputs"]["cfg"]["cluster_relin"] == 0
    oracle = oracle_from_snapshot(entry["inputs"], ROOT / plan["canonical"], ROOT / plan["parent"])
    reference = json.loads((ROOT / plan["reference"]).read_text(encoding="utf8"))
    expected = next(r["result_sha256"] for r in reference["warm"]
        if r["cluster"] == 0 and r["index"] == 0 and r["mode"] == "old_gpu")
    records = []
    with HostPhaseScorer(ROOT / plan["dll"]) as scorer:
        observer = ScoreObserver(lambda: ("A", 203))
        score = observer.bind(scorer)
        for index, label in enumerate(("initial", "immediate", "after_idle15", "immediate_after_idle")):
            hypotheses, problem = prepare(oracle, entry["inputs"])
            before = digest((hypotheses, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states))
            if index == 2:
                start = time.perf_counter_ns()
                time.sleep(15)
                idle_ns = time.perf_counter_ns() - start
            else:
                idle_ns = 0
            values, details = score(hypotheses, problem, oracle, "gpu", diagnostics=False)
            result = digest((values, details))
            assert result == expected and details["scalar_fallbacks"] == 0
            assert before == digest((hypotheses, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states))
            row = observer.records[-1]
            assert row["completed"] and row["probe_error"] is None
            records.append(dict(index=index, label=label, actual_idle_ns=idle_ns, clocks=row,
                result_sha256=result, input_unchanged=True, scalar_fallbacks=0))
    output = ROOT / plan["output"]
    assert not output.exists()
    output.write_text(json.dumps(dict(verified=True, records=records, score_calls=4,
        all_result_bits_exact_to_frozen_GPU_reference=True,
        scope="One unchanged actual score input,one workspace,four sequential calls,one explicit15second idle. Nsight CUDA API tracing perturbs execution. Not RTK throughput,not unobserved idle latency,not a calibrated optimization or cause proof.",
        optimization_speed_gain_established=False, original_failed_gates_preserved=True), indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(verified=True, score_calls=4, all_result_bits_exact=True)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
