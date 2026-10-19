"""Independently reconstruct every frozen native layout screen observation."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("original", "off", "on", "original_AA", "off_AA")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(encoded):
    return hashlib.sha256(
        json.dumps(encoded, separators=(",", ":")).encode()
    ).hexdigest()


def reconstruct(rows, primary):
    result = {}
    for phase in ("timed", "cold"):
        contrasts = {}
        for label, before, after in (
            ("original_on", "original", "on"),
            ("off_on", "off", "on"),
            ("original_AA", "original", "original_AA"),
            ("off_AA", "off", "off_AA"),
        ):
            values = []
            for rd in range(5):
                gains = []
                for group in range(3):
                    times = []
                    for mode in (before, after):
                        matched = [
                            r["ns"]
                            for r in rows
                            if r["phase"] == phase
                            and r["mode"] == mode
                            and r["round"] == rd
                            and r.get("group", 0) == group
                        ]
                        assert len(matched) == (2)
                        times.append(sum(matched))
                    gains.append(100 * (1 - times[1] / times[0]))
                values.append(median(gains))
            contrasts[label] = dict(
                reference=before,
                candidate=after,
                round_percent=values,
                median_percent=median(values),
                positive_rounds=len([v for v in values if v > 0]),
                absolute_median_percent=median([abs(v) for v in values]),
                absolute_rounds_under5=len([v for v in values if abs(v) < 5]),
            )
        valid = all(
            contrasts[label]["absolute_median_percent"] < 5
            and contrasts[label]["absolute_rounds_under5"] >= 4
            for label in ("original_AA", "off_AA")
        )
        threshold = 10 if primary and phase == "timed" else -5
        numeric = True
        for label in ("original_on", "off_on"):
            values = contrasts[label]["round_percent"]
            count = (
                sum(v > 0 for v in values)
                if threshold == 10
                else sum(v >= -5 for v in values)
            )
            numeric &= median(values) >= threshold and count >= 4
        result[phase] = dict(
            contrasts=contrasts,
            AA_valid=valid,
            numeric_gate=bool(numeric),
            passed=valid and numeric,
            required_median_percent=threshold,
        )
    result["passed"] = result["timed"]["passed"] and result["cold"]["passed"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    frozen, result = read(args.frozen_plan), read(args.result)
    assert sha(Path(__file__)) == frozen["pre_execution_independent_auditor_sha256"]
    for section in (
        "source_sha256",
        "external_python_sha256",
        "input_sha256",
        "gold_sha256",
        "saved_pair_sha256",
        "new_gate_evidence_sha256",
    ):
        assert all(sha(ROOT / n) == d for n, d in frozen[section].items()), section
    assert all(
        sha(ROOT / b["path"]) == b["sha256"] for b in frozen["binaries"].values()
    )
    assert result["frozen_plan_sha256"] == sha(args.frozen_plan)
    measurement_path = args.result.with_suffix(".measurement.plan.json")
    measurement = read(measurement_path)
    assert result["measurement_plan_sha256"] == sha(measurement_path)
    assert measurement["frozen_plan_sha256"] == sha(args.frozen_plan)
    conditions = [
        dict(triangles=12000, epochs=128, satellites=12, primary=True),
        dict(triangles=120000, epochs=32, satellites=32, primary=True),
        dict(triangles=12000, epochs=1, satellites=12, primary=False),
        dict(triangles=4096, epochs=16, satellites=8, primary=False),
    ]
    assert measurement["conditions"] == conditions and measurement["modes"] == list(
        MODES
    )
    assert measurement["contexts_per_condition"] == 2
    assert (
        measurement["original_AA_same_scene"]
        and measurement["off_on_and_off_AA_same_scene"]
    )
    environment = measurement["environment"]
    assert environment["GC_enabled"] and len(environment["affinity"]) == 1
    assert environment["priority"] == 32
    assert all(p["num_threads"] == 1 for p in environment["thread_pools"])
    input_hash = {}
    for ci, condition in enumerate(conditions):
        proof = measurement["input_proof"][str(ci)]
        path = ROOT / proof["path"]
        assert sha(path) == proof["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays["mesh"].shape == (condition["triangles"], 3, 3)
            assert arrays["rx"].shape == (condition["epochs"], 3)
            assert arrays["sat"].shape == (
                condition["epochs"],
                condition["satellites"],
                3,
            )
            encoded = [
                dict(
                    dtype=arrays[n].dtype.str,
                    shape=list(arrays[n].shape),
                    hex=arrays[n].tobytes().hex(),
                )
                for n in ("nodes", "mesh", "rx", "sat")
            ]
        assert digest(encoded) == proof["arrays_sha256"]
        input_hash[ci] = proof["arrays_sha256"]
    calls_path, states_path = (
        args.result.with_suffix(".calls.jsonl.gz"),
        args.result.with_suffix(".states.jsonl.gz"),
    )
    assert (
        sha(calls_path) == result["calls_sha256"]
        and sha(states_path) == result["states_sha256"]
    )
    with gzip.open(calls_path, "rt", encoding="utf8") as stream:
        calls = [json.loads(line) for line in stream]
    with gzip.open(states_path, "rt", encoding="utf8") as stream:
        states = [json.loads(line) for line in stream]
    assert len(calls) == 1272 and len(states) == 60
    assert (
        dict(Counter(r["phase"] for r in calls))
        == result["counts"]
        == dict(setup=12, warm=60, timed=600, cold=600)
    )
    expected = []
    for ci in range(4):
        expected.extend(("setup", ci, m, None, None, None, None) for m in MODES[:3])
    for ci in range(4):
        expected.extend(
            ("warm", ci, m, None, None, rep, None) for m in MODES for rep in range(3)
        )
        for rd in range(5):
            for group in range(3):
                shift = (ci + rd * 3 + group) % 5
                order = list(MODES[shift:] + MODES[:shift])
                if (ci + rd + group) % 2:
                    order.reverse()
                expected.extend(
                    ("timed", ci, m, rd, group, rep, order)
                    for m in order
                    for rep in range(2)
                )
        for rd in range(5):
            for group in range(3):
                shift = (ci + rd * 3 + group) % 5
                order = list(MODES[shift:] + MODES[:shift])
                if (ci + rd + group) % 2:
                    order.reverse()
                expected.extend(("cold", ci, m, rd, group, rep, order)
                                for m in order for rep in range(2))
    assert len(expected) == len(calls)
    refs = {}
    stable = {}
    for row, wanted in zip(calls, expected, strict=True):
        actual = tuple(
            row.get(k)
            for k in (
                "phase",
                "condition",
                "mode",
                "round",
                "group",
                "repetition",
                "order",
            )
        )
        assert actual == wanted
        ci, phase, mode = row["condition"], row["phase"], row["mode"]
        assert type(row["ns"]) is int and row["ns"] > 0
        assert row["input_sha256"] == input_hash[ci]
        if ci not in refs:
            assert phase == "setup" and mode == "original"
            refs[ci] = row["output_sha256"]
        assert refs[ci] == row["output_sha256"]
        if phase != "cold":
            assert row["group_active"] == (mode == "on" and conditions[ci]["primary"])
            assert row["allocated_bytes"] > 0
        if phase == "setup":
            assert not row["speed_evidence"]
            if mode == "on" and conditions[ci]["primary"]:
                assert 0 < row["group_count"] * 10 <= conditions[ci]["triangles"] * 7
            else:
                assert row["group_count"] == 0
            assert (row["metadata_bytes"] > 0) == row["group_active"]
        if phase == "timed":
            owner = (ci, mode.startswith("original"))
            footprint = (row["allocated_bytes"], row["metadata_bytes"])
            stable.setdefault(owner, footprint)
            assert stable[owner] == footprint
            if not conditions[ci]["primary"]:
                assert row["metadata_bytes"] == 0
        if phase == "cold":
            assert row["closed_bytes"] == 0
    warm = [r for r in calls if r["phase"] == "warm"]
    for snapshot, row in zip(states, warm, strict=True):
        for name in ("condition", "mode", "repetition"):
            assert snapshot[name] == row[name]
        assert digest(snapshot["arrays"]) == row["output_sha256"]
        for encoded in snapshot["arrays"]:
            data = bytes.fromhex(encoded["hex"])
            assert np.prod(encoded["shape"]) * np.dtype(
                encoded["dtype"]
            ).itemsize == len(data)
    computed = {
        str(ci): reconstruct([r for r in calls if r["condition"] == ci], c["primary"])
        for ci, c in enumerate(conditions)
    }
    assert computed == result["results"]
    passed = all(c["passed"] for c in computed.values())
    assert passed == result["native_screen_pass"]
    assert (
        not result["public_or_urban_speed_established"]
        and not result["default_changed"]
    )
    args.output.write_text(
        json.dumps(
            dict(
                verified=True,
                frozen_before_speed=True,
                numerical_reexecution=False,
                frozen_plan_sha256=sha(args.frozen_plan),
                auditor_sha256=sha(Path(__file__)),
                result_sha256=sha(args.result),
                calls_sha256=sha(calls_path),
                states_sha256=sha(states_path),
                raw_calls=1272,
                full_warm_states=60,
                timed_calls=600,
                cold_calls=600,
                all_output_dtype_shape_bits_equal=True,
                all_grid_order_and_input_hashes_exact=True,
                stable_owned_allocation=True,
                results=computed,
                native_screen_pass=passed,
                public_or_urban_speed_established=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
