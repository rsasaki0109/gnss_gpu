"""Frozen position-publication policy: development replay and fresh-noise holdout.

PYTHONPATH=python:. python experiments/run_cpu_position_quality.py --workers 4
The gate sees only receiver pseudoranges, source coordinates and time. Truth,
wetness, visibility, scenario/stage labels are evaluation-only.
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
from experiments.run_cpu_wet_antenna import RECEIVER, _case, recovery_metrics
from experiments.run_cpu_rain_positioning import REFERENCE, constellation
from gnss_gpu.code_detection_cpu import CaDetectionThreshold
from gnss_gpu.input_validation import positive_int
from gnss_gpu.position_quality_cpu import PositionGateConfig, PositionPublicationGate

HOLDOUT_SEEDS = (24685, 24686)
HOLDOUT_SCENARIOS = ("dry", "loss_6db", "loss_30db_stress", "combined_stress")


def _boolean(value):
    if isinstance(value, bool):
        return value
    if value not in ("True", "False"):
        raise ValueError("boolean receiver fields must be True or False")
    return value == "True"


def replay_channels(channels, config, satellites, *, recovery_config=None):
    """Replay receiver fields ONLY; error/truth/visibility fields are ignored."""
    grouped = defaultdict(list)
    for channel in channels:
        grouped[int(channel["block"])].append(channel)
    gate = PositionPublicationGate(config, recovery_config=recovery_config)
    decisions = []
    for block, values in sorted(grouped.items()):
        used = [row for row in values if _boolean(row["used"])]
        prns = [int(row["prn"]) for row in used]
        if len(prns) != len(set(prns)) or any(prn < 1 or prn > len(satellites) for prn in prns):
            raise ValueError("receiver epoch has duplicate/invalid PRNs")
        sat = np.asarray([satellites[prn-1] for prn in prns], dtype=float).reshape(-1, 3)
        ranges = [REFERENCE+float(row["delay_m"]) for row in used]
        decision = gate.update(sat, ranges, time_s=(block+.5)*.001)
        decisions.append((block, decision))
    return decisions


def error_summary(rows):
    available = [row for row in rows if row["available"]]
    return dict(blocks=len(rows), available=len(available), unavailable=len(rows)-len(available),
                available_error_over_100_m=sum(row["position_error_m"] > 100 for row in available),
                horizontal_p95_m_conditional=float(np.quantile([row["horizontal_error_m"] for row in available], .95)) if available else None,
                position_p95_m_conditional=float(np.quantile([row["position_error_m"] for row in available], .95)) if available else None,
                position_max_m_conditional=max((row["position_error_m"] for row in available), default=None))


def evaluate_case(case, dataset, config, satellites, truth, *, recovery_config=None) -> dict[str, Any]:
    channels = case["channels"]
    original = {int(row["block"]):row for row in case["positions"]}
    first = channels[0]
    labels = dict(dataset=dataset, seed=int(first["seed"]), environment=first["environment"], scenario=first["scenario"])
    output = []
    max_replay_delta = 0.
    for block, decision in replay_channels(channels, config, satellites, recovery_config=recovery_config):
        expected = original[block]
        candidate = decision.candidate
        if (candidate is not None) != _boolean(expected["available"]):
            raise RuntimeError("replayed raw availability differs from input receiver")
        if candidate is not None:
            previous = np.array([float(expected[key]) for key in ("east_m", "north_m", "up_m")])
            max_replay_delta = max(max_replay_delta, float(np.linalg.norm(candidate.position_m-previous)))
        quality = decision.quality
        for mode in ("raw", "quality_gate"):
            position = candidate.position_m if mode == "raw" and candidate is not None else decision.position_m if mode == "quality_gate" else None
            error = position-truth if position is not None else None
            output.append(dict(**labels, mode=mode, block=block, time_s=(block+.5)*.001, stage=expected["stage"],
                               available=position is not None, fresh_channels=int(expected["fresh_channels"]),
                               reason=decision.reason if mode == "quality_gate" else ("raw_solution" if position is not None else "no_raw_solution"),
                               confirmation_count=decision.confirmation_count if mode == "quality_gate" else 0,
                               pdop=quality.pdop if quality else None,
                               residual_rms_m=quality.residual_rms_m if quality else None,
                               residual_dof=quality.degrees_of_freedom if quality else None,
                               anchor_jump_m=decision.anchor_jump_m, anchor_age_s=decision.anchor_age_s,
                               recovery_count=decision.recovery_count, recovery_radius_m=decision.recovery_radius_m,
                               reanchored=decision.reanchored, anchor_source=decision.anchor_source,
                               east_m=float(position[0]) if position is not None else None,
                               north_m=float(position[1]) if position is not None else None,
                               up_m=float(position[2]) if position is not None else None,
                               horizontal_error_m=float(np.linalg.norm(error[:2])) if error is not None else None,
                               position_error_m=float(np.linalg.norm(error)) if error is not None else None))
    if max_replay_delta > 1e-6:
        raise RuntimeError(f"raw replay changed candidate position by {max_replay_delta} m")
    summaries, stages, reasons = [], [], []
    for mode in ("raw", "quality_gate"):
        rows = [row for row in output if row["mode"] == mode]
        summaries.append(dict(**labels, mode=mode, **error_summary(rows),
                              **{key:value for key,value in recovery_metrics(rows).items()
                                 if key not in ("blocks", "position_available", "position_unavailable")}))
        for stage in ("initial_dry", "wetting", "wet", "drying", "dry"):
            stages.append(dict(**labels, mode=mode, stage=stage,
                               **error_summary([row for row in rows if row["stage"] == stage])))
        for reason in sorted({row["reason"] for row in rows}):
            reasons.append(dict(**labels, mode=mode, reason=reason, blocks=sum(row["reason"] == reason for row in rows)))
    return dict(positions=output, summary=summaries, stages=stages, reasons=reasons,
                max_raw_replay_delta_m=max_replay_delta)


def load_development(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text())
    channels, positions = defaultdict(list), defaultdict(list)
    for filename, target in (("channels.csv", channels), ("positions.csv", positions)):
        with (directory/filename).open() as stream:
            for row in csv.DictReader(stream):
                target[int(row["seed"]), row["environment"], row["scenario"]].append(row)
    if channels.keys() != positions.keys():
        raise ValueError("development positions/channels do not match")
    return metadata, [dict(channels=value, positions=positions[key]) for key, value in channels.items()]


def decode_thresholds(metadata):
    """Read the saved per-PRN single-carrier gate schema without recalibration."""
    gates = {int(prn):CaDetectionThreshold(**values) for prn, values in metadata["thresholds"].items()}
    if any(prn != gate.prn or gate.sampling_freq != metadata["sampling_freq_hz"] or gate.carrier_hz != 0.
           for prn, gate in gates.items()):
        raise ValueError("saved thresholds do not match the receiver PRN/rate/carrier configuration")
    return gates


def run_experiment(*, development_dir="output/cpu-wet-antenna", policy_file="docs/position_quality_policy.json",
                   output_dir="output/cpu-position-quality", workers=1) -> dict[str, Any]:
    workers = positive_int("workers", workers)
    policy_bytes = Path(policy_file).read_bytes()
    config = PositionGateConfig(**json.loads(policy_bytes))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Concrete immutable policy is saved BEFORE replay or holdout generation.
    (output/"frozen_policy.json").write_bytes(policy_bytes)
    development, cases = load_development(development_dir)
    if set(HOLDOUT_SEEDS) & set(development["seeds"]):
        raise ValueError("held-out seeds must be disjoint from development")
    gates = decode_thresholds(development)
    satellites, truth = np.asarray(development["constellation_m"]), np.asarray(development["receiver_m"])
    if not np.array_equal(satellites, constellation()) or not np.array_equal(truth, RECEIVER):
        raise ValueError("development geometry does not match this generator")
    results = [evaluate_case(case, "development", config, satellites, truth) for case in cases]
    print("Development replay complete; frozen policy unchanged. Generating held-out IQ.", flush=True)
    jobs = [(seed, environment, scenario, gates) for seed in HOLDOUT_SEEDS
            for environment in ("open", "wall") for scenario in HOLDOUT_SCENARIOS]
    if workers == 1:
        fresh = list(map(_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as pool:
            fresh = list(pool.map(_case, jobs))
    results.extend(evaluate_case(case, "holdout", config, satellites, truth) for case in fresh)
    if Path(policy_file).read_bytes() != policy_bytes:
        raise RuntimeError("policy file changed during evaluation; no silent retuning permitted")
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_cpu_position_quality.py", "python/gnss_gpu/position_quality_cpu.py",
               "python/gnss_gpu/positioning_cpu.py", "experiments/run_cpu_wet_antenna.py",
               "experiments/run_cpu_rain_positioning.py", "experiments/run_cpu_multipath_tracking.py",
               "python/gnss_gpu/wall_signal_sim.py", "python/gnss_gpu/wet_antenna_cpu.py",
               "python/gnss_gpu/rain_cpu.py", "python/gnss_gpu/wet_delay_cpu.py", "python/gnss_gpu/fresnel.py",
               "python/gnss_gpu/tracking_cpu.py", "python/gnss_gpu/acquisition_cpu.py",
               "python/gnss_gpu/code_delay_cpu.py", "python/gnss_gpu/code_detection_cpu.py", "python/gnss_gpu/signal_sim_cpu.py"]
    metadata = dict(policy=asdict(config), policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
                    development_seeds=development["seeds"], holdout_seeds=list(HOLDOUT_SEEDS),
                    holdout_scenarios=list(HOLDOUT_SCENARIOS), workers=workers,
                    development_input_sha256={name:hashlib.sha256((Path(development_dir)/name).read_bytes()).hexdigest()
                                              for name in ("metadata.json", "channels.csv", "positions.csv")},
                    receiver_metadata=development, max_raw_replay_delta_m=max(result["max_raw_replay_delta_m"] for result in results),
                    design="Engineering policy fixed after receiver-only development geometry inspection, not statistically calibrated protection levels. No threshold search or retuning on holdout. All inputs and rejection reasons retained.",
                    truth_boundary="Gate accepts only satellites, pseudoranges, timestamp; visibility/weather/stage/reference truth are not passed. Truth is used after decisions for error evaluation only.",
                    timing="2 ms channel confirmation followed by 5 good PVT candidates. Report first publication and a common retrospective five-position recovery metric for BOTH modes; do not confuse those two delays.",
                    limitations="One static synthetic geometry per environment, two fresh noise seeds, paired scenarios and serial states; no certified integrity or false-fix guarantee. Coherent/slow bias can pass; stale anchors expire after 2 s. Motion bound is an assumed operating envelope and may withhold legitimate faster motion. No real RF validation.",
                    source_sha256={file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in sources})
    return dict(positions=[row for result in results for row in result["positions"]],
                summary=[row for result in results for row in result["summary"]],
                stages=[row for result in results for row in result["stages"]],
                reasons=[row for result in results for row in result["reasons"]], metadata=metadata,
                holdout_channels=[row for case in fresh for row in case["channels"]],
                holdout_raw_positions=[row for case in fresh for row in case["positions"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--development-dir", default="output/cpu-wet-antenna")
    parser.add_argument("--policy-file", default="docs/position_quality_policy.json")
    parser.add_argument("--output-dir", default="output/cpu-position-quality")
    args = parser.parse_args()
    result = run_experiment(development_dir=args.development_dir, policy_file=args.policy_file,
                            output_dir=args.output_dir, workers=args.workers)
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
        for j, mode in enumerate(("raw", "quality_gate")):
            axis = axes[i, j]
            for seed in HOLDOUT_SEEDS:
                rows = [row for row in result["positions"] if row["dataset"] == "holdout" and row["seed"] == seed
                        and row["environment"] == environment and row["scenario"] == "loss_30db_stress" and row["mode"] == mode]
                axis.plot([row["time_s"] for row in rows], [row["position_error_m"] if row["available"] else np.nan for row in rows], label=f"seed {seed}")
            axis.axhline(100., linestyle="--", color="black", label="100 m diagnostic threshold")
            axis.axvspan(.05, .2, color="blue", alpha=.06)
            axis.axvspan(.2, .4, color="orange", alpha=.08)
            axis.set_yscale("symlog", linthresh=20.)
            axis.set(xlabel="time [s]", ylabel="3D error [m], symlog", title=f"{environment}, {mode}; gaps = not published")
            axis.legend()
            axis.grid(alpha=.2)
    figure.suptitle("Frozen position policy on fresh-noise holdout: errors AND withheld positions")
    figure.savefig(output/"position_quality.png", dpi=150)
    plt.close(figure)
    for row in result["summary"]:
        if row["dataset"] == "holdout":
            print(row)
    print(f"Saved all raw/gated candidates, rejections, holdout inputs and frozen policy to {output}")


if __name__ == "__main__":
    main()
