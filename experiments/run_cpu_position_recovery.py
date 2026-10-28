"""Compare raw, legacy gate and local-cluster recovery on new noise seeds.

PYTHONPATH=python:. python experiments/run_cpu_position_recovery.py --workers 4
Previously inspected seeds are development data; new holdout settings freeze
before generation. No reference position or weather flags enter either gate.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_cpu_multipath_tracking import _worker_init
from experiments.run_cpu_position_quality import decode_thresholds, evaluate_case
from experiments.run_cpu_wet_antenna import RECEIVER, _case
from experiments.run_cpu_rain_positioning import constellation
from gnss_gpu.input_validation import positive_int
from gnss_gpu.position_quality_cpu import PositionGateConfig, PositionRecoveryConfig

HOLDOUT_SEEDS = (24687, 24688)
SCENARIOS = ("dry", "loss_6db", "loss_30db_stress", "combined_stress")


def read_cases(directory, channel_file, position_file):
    directory = Path(directory)
    channels, positions = defaultdict(list), defaultdict(list)
    for name, target in ((channel_file, channels), (position_file, positions)):
        with (directory/name).open() as stream:
            for row in csv.DictReader(stream):
                target[int(row["seed"]), row["environment"], row["scenario"]].append(row)
    if channels.keys() != positions.keys():
        raise ValueError("saved channel/position cases differ")
    return [dict(channels=values, positions=positions[key]) for key, values in channels.items()]


def compare_case(case, dataset, config, recovery, satellites, truth) -> dict[str, Any]:
    legacy = evaluate_case(case, dataset, config, satellites, truth)
    current = evaluate_case(case, dataset, config, satellites, truth, recovery_config=recovery)
    result = {key:[] for key in ("positions", "summary", "stages", "reasons")}
    for source, mapping in ((legacy, {"raw":"raw", "quality_gate":"legacy_gate"}),
                            (current, {"quality_gate":"recovery_gate"})):
        for key in result:
            for row in source[key]:
                if row["mode"] in mapping:
                    result[key].append(dict(row, mode=mapping[row["mode"]]))
    for row in result["summary"]:
        row["reanchor_events"] = sum(record["reanchored"] for record in result["positions"] if record["mode"] == row["mode"])
    result["max_raw_replay_delta_m"] = max(legacy["max_raw_replay_delta_m"], current["max_raw_replay_delta_m"])
    return result


def run_experiment(*, workers=1, wet_dir="output/cpu-wet-antenna", previous_dir="output/cpu-position-quality",
                   policy_file="docs/position_recovery_policy.json", output_dir="output/cpu-position-recovery") -> dict[str, Any]:
    workers = positive_int("workers", workers)
    policy = Path(policy_file).read_bytes()
    parsed = json.loads(policy)
    config, recovery = PositionGateConfig(**parsed["gate"]), PositionRecoveryConfig(**parsed["recovery"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/"frozen_policy.json").write_bytes(policy)
    wet_metadata = json.loads((Path(wet_dir)/"metadata.json").read_text())
    old_metadata = json.loads((Path(previous_dir)/"metadata.json").read_text())
    if asdict(config) != old_metadata["policy"]:
        raise ValueError("ordinary policy must match the previous frozen baseline")
    development_seeds = sorted(set(wet_metadata["seeds"]) | set(old_metadata["holdout_seeds"]))
    if set(development_seeds) & set(HOLDOUT_SEEDS):
        raise ValueError("new holdout must be disjoint from all previously inspected seeds")
    gates = decode_thresholds(wet_metadata)
    satellites, truth = np.asarray(wet_metadata["constellation_m"]), np.asarray(wet_metadata["receiver_m"])
    if not np.array_equal(satellites, constellation()) or not np.array_equal(truth, RECEIVER):
        raise ValueError("development geometry differs from generator")
    cases = read_cases(wet_dir, "channels.csv", "positions.csv")
    cases += read_cases(previous_dir, "holdout_channels.csv", "holdout_raw_positions.csv")
    results = [compare_case(case, "development", config, recovery, satellites, truth) for case in cases]
    print("Development complete. Recovery policy frozen; generating fresh holdout IQ.", flush=True)
    jobs = [(seed, environment, scenario, gates) for seed in HOLDOUT_SEEDS
            for environment in ("open", "wall") for scenario in SCENARIOS]
    if workers == 1:
        fresh = list(map(_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as pool:
            fresh = list(pool.map(_case, jobs))
    results.extend(compare_case(case, "holdout", config, recovery, satellites, truth) for case in fresh)
    if Path(policy_file).read_bytes() != policy:
        raise RuntimeError("policy changed during evaluation; no holdout retuning permitted")
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_cpu_position_recovery.py", "experiments/run_cpu_position_quality.py",
               "python/gnss_gpu/position_quality_cpu.py", "python/gnss_gpu/positioning_cpu.py",
               "experiments/run_cpu_wet_antenna.py", "experiments/run_cpu_rain_positioning.py",
               "experiments/run_cpu_multipath_tracking.py", "python/gnss_gpu/wall_signal_sim.py",
               "python/gnss_gpu/wet_antenna_cpu.py", "python/gnss_gpu/wet_delay_cpu.py", "python/gnss_gpu/rain_cpu.py",
               "python/gnss_gpu/fresnel.py", "python/gnss_gpu/tracking_cpu.py", "python/gnss_gpu/acquisition_cpu.py",
               "python/gnss_gpu/code_delay_cpu.py", "python/gnss_gpu/code_detection_cpu.py", "python/gnss_gpu/signal_sim_cpu.py"]
    input_paths = [Path(wet_dir)/name for name in ("metadata.json", "positions.csv", "channels.csv")]
    input_paths += [Path(previous_dir)/name for name in ("metadata.json", "holdout_channels.csv", "holdout_raw_positions.csv")]
    metadata = dict(policy=parsed, policy_sha256=hashlib.sha256(policy).hexdigest(), workers=workers,
                    development_seeds=development_seeds, holdout_seeds=list(HOLDOUT_SEEDS), holdout_scenarios=list(SCENARIOS),
                    source_sha256={file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in sources},
                    input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths},
                    receiver_metadata=wet_metadata,
                    max_raw_replay_delta_m=max(result["max_raw_replay_delta_m"] for result in results),
                    design="Ordinary policy unchanged. Optional median published reference and 25-epoch stronger-quality recovery cluster fixed on prior development cases; no tuning after fresh holdout. Outputs remain current raw candidates, not median positions.",
                    truth_boundary="Neither gate receives true position, stage/weather/visibility. Those are used only after publication decisions for evaluation.",
                    limitations="Engineering stability checks, not a protection level or independent-evidence count. Coherent wrong clusters/slow bias can pass. Two fresh noise seeds, one static geometry per environment, paired cases and correlated receiver states; no real RF or general integrity guarantee.")
    return dict(**{key:[row for result in results for row in result[key]] for key in ("positions", "summary", "stages", "reasons")},
                metadata=metadata, holdout_channels=[row for case in fresh for row in case["channels"]],
                holdout_raw_positions=[row for case in fresh for row in case["positions"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wet-dir", default="output/cpu-wet-antenna")
    parser.add_argument("--previous-dir", default="output/cpu-position-quality")
    parser.add_argument("--policy-file", default="docs/position_recovery_policy.json")
    parser.add_argument("--output-dir", default="output/cpu-position-recovery")
    args = parser.parse_args()
    result = run_experiment(workers=args.workers, wet_dir=args.wet_dir, previous_dir=args.previous_dir,
                            policy_file=args.policy_file, output_dir=args.output_dir)
    output = Path(args.output_dir)
    for key in ("positions", "summary", "stages", "reasons", "holdout_channels", "holdout_raw_positions"):
        with (output/f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    (output/"metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for i, environment in enumerate(("open", "wall")):
        for j, seed in enumerate(HOLDOUT_SEEDS):
            axis = axes[i, j]
            for mode, offset in (("raw", -.25), ("legacy_gate", 0.), ("recovery_gate", .25)):
                rows = {row["scenario"]:row for row in result["summary"] if row["dataset"] == "holdout" and row["seed"] == seed
                        and row["environment"] == environment and row["mode"] == mode}
                axis.bar(np.arange(len(SCENARIOS))+offset, [rows[scenario]["available"] for scenario in SCENARIOS], .25, label=mode)
            axis.set(xticks=np.arange(len(SCENARIOS)), xticklabels=SCENARIOS, ylabel="published / 500 blocks",
                     ylim=(0, 520), title=f"{environment}, fresh seed {seed}")
            axis.tick_params(axis="x", labelrotation=20)
            axis.legend()
            axis.grid(axis="y", alpha=.2)
    figure.suptitle("Local-cluster recovery: fresh-noise holdout availability; errors retained in CSV")
    figure.savefig(output/"position_recovery.png", dpi=150)
    plt.close(figure)
    for row in result["summary"]:
        if row["dataset"] == "holdout":
            print(row)
    print(f"Saved three-policy decisions, reanchor events, holdout inputs and frozen settings to {output}")


if __name__ == "__main__":
    main()
