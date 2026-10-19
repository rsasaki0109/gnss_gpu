"""Actual JointMarginal copy-loop screen; no graph/whole-route speed claim."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

from bench_pf_private_weights_compare import _pin
from rbpf_joint_covariance import make_type
from rbpf_joint_covariance_cases import make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    own = parser.parse_args()
    own.output_dir.mkdir(parents=True, exist_ok=False)
    _pin()
    root = Path(__file__).resolve().parents[1]
    names = [
        "benchmarks/rbpf_joint_covariance.py",
        "benchmarks/rbpf_joint_covariance_cases.py",
        "benchmarks/bench_rbpf_joint_covariance.py",
    ]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sources = {n: sha(root / n) for n in names}
    plan = dict(
        source_sha256=sources,
        dimensions=[3, 6],
        ambiguities=[4, 16, 48],
        rounds=5,
        groups=3,
        repetitions=16,
        warm_per_mode=16,
        gate="Per case median5>=10percent/4of5positive;originalAA AND candidateAA absolute medians<5percent. Continue only>=4of6cases including BOTH48ambiguity cases. All P bytes exact. Retain all raw,no retry. Component screen only;wrapper/global refresh,graph solve,continuousRTK and forced-wait effects are outside this component.",
    )
    (own.output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf8"
    )
    results, passed = [], 0
    for dim in (3, 6):
        for count in (4, 16, 48):
            cls, source, _, key_x, _, jm, active = make_case(dim, count)
            chosen, engine = make_type(cls)
            method = "_write_back" if dim == 3 else "_write_back_tc"
            _, _, _, _, scope, literal = engine.methods[method]
            literal(source, active, jm)
            expected = source.nav.P.tobytes()
            modes = ["reference", "bulk", "AA", "bulkAA"]
            targets, calls = {}, {}
            for mode in modes:
                target = (chosen if mode.startswith("bulk") else cls)()
                target.nav = deepcopy(source.nav)
                targets[mode] = target
                calls[mode] = (
                    (
                        lambda target=target: engine.copy_pairs(
                            target, active, jm, key_x, dim, scope, literal
                        )
                    )
                    if mode.startswith("bulk")
                    else (lambda target=target: literal(target, active, jm))
                )
            raw = []
            for warm, ri, gi in [(True, -1, -1)] + [
                (False, r, g) for r in range(5) for g in range(3)
            ]:
                for rep in range(16):
                    offset = (max(ri, 0) + max(gi, 0) + rep) % 4
                    for mode in modes[offset:] + modes[:offset]:
                        start = time.perf_counter_ns()
                        calls[mode]()
                        duration = time.perf_counter_ns() - start
                        raw.append(
                            dict(
                                warm=warm,
                                round=ri,
                                group=gi,
                                repetition=rep,
                                mode=mode,
                                ns=duration,
                            )
                        )
                assert all(
                    target.nav.P.tobytes() == expected for target in targets.values()
                )
            summary = {}
            for name, left, right in (
                ("bulk", "reference", "bulk"),
                ("AA", "reference", "AA"),
                ("bulkAA", "bulk", "bulkAA"),
            ):
                rounds = []
                for ri in range(5):
                    groups = []
                    for gi in range(3):
                        sums = [
                            sum(
                                r["ns"]
                                for r in raw
                                if not r["warm"]
                                and r["round"] == ri
                                and r["group"] == gi
                                and r["mode"] == mode
                            )
                            for mode in (left, right)
                        ]
                        groups.append((sums[0] - sums[1]) / sums[0] * 100)
                    rounds.append(statistics.median(groups))
                summary[name] = dict(
                    round_reduction_percent=rounds,
                    paired_reduction_percent=statistics.median(rounds),
                    improving_rounds=sum(v > 0 for v in rounds),
                )
            good = (
                summary["bulk"]["paired_reduction_percent"] >= 10
                and summary["bulk"]["improving_rounds"] >= 4
                and all(
                    abs(summary[k]["paired_reduction_percent"]) < 5
                    for k in ("AA", "bulkAA")
                )
            )
            passed += good
            assert (
                len(raw) == 1024
                and engine.fallback_calls == 0
                and engine.fast_calls == 512
            )
            record = dict(
                position_dim=dim,
                ambiguities=count,
                raw=raw,
                summary=summary,
                component_gate_passed=good,
                all_final_P_bytes_equal=True,
                engine_counts=engine.counts(),
            )
            results.append(record)
            (own.output_dir / f"d{dim}_n{count}.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf8"
            )
            print(
                json.dumps(
                    dict(
                        position_dim=dim,
                        ambiguities=count,
                        summary=summary,
                        component_gate_passed=good,
                    )
                ),
                flush=True,
            )
    assert all(sha(root / n) == d for n, d in sources.items())
    proceed = passed >= 4 and all(
        r["component_gate_passed"] for r in results if r["ambiguities"] == 48
    )
    (own.output_dir / "summary.json").write_text(
        json.dumps(
            dict(
                component_passes=passed,
                cases=6,
                timed_calls=5760,
                warm_calls=384,
                continue_to_field=proceed,
                sources_unchanged=True,
                all_P_bytes_equal=True,
                results=[{k: v for k, v in r.items() if k != "raw"} for r in results],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
