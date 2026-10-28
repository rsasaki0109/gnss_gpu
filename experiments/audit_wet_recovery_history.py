"""Post-hoc diagnostic: distinguish receiver recovery from publication history.

Uses existing development recordings; generates no new noise holdout. After
400 ms the intervention feeds exactly the dry recording's channel dictionaries
to the gate, retaining the wet prefix. This isolates gate state from tracking.
Run with PYTHONPATH=python:. python experiments/audit_wet_recovery_history.py
"""

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.run_cpu_position_quality import replay_channels
from experiments.run_cpu_rain_positioning import COUNT, FS, REFERENCE, constellation
from experiments.run_cpu_wet_antenna import RECEIVER, RESPONSES, noise_iq
from gnss_gpu.position_quality_cpu import PositionGateConfig
from gnss_gpu.signal_sim_cpu import CpuSignalSimulator
from gnss_gpu.wall_signal_sim import VerticalWall, simulate_wall_signal


DATASETS = (
    ("cpu-wet-antenna", "positions.csv"),
    ("cpu-position-quality", "holdout_raw_positions.csv"),
    ("cpu-position-recovery", "holdout_raw_positions.csv"),
    ("cpu-position-uncertainty", "holdout_raw_positions.csv"),
)
SCENARIOS = ("loss_6db", "loss_30db_stress", "combined_stress")


def read_groups(path):
    groups = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            groups[int(row["seed"]), row["environment"], row["scenario"]].append(row)
    return groups


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output, raw, interventions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for index, environment in enumerate(("open", "wall")):
        values = [r["max_paired_raw_difference_m"]*100 for r in raw if r["environment"] == environment]
        axes[0].scatter(index+np.linspace(-.15, .15, len(values)), values, alpha=.7)
    axes[0].set(xticks=[0, 1], xticklabels=["Open", "Wall"],
                ylabel="Maximum paired position difference per episode (cm)",
                title="Native receiver tails: 48 paired episodes")
    y = np.arange(len(interventions))
    axes[1].barh(y-.18, [r["dry_prefix_published"] for r in interventions], height=.36, label="Dry prefix")
    axes[1].barh(y+.18, [r["wet_prefix_shared_dry_suffix_published"] for r in interventions], height=.36, label="Wet prefix")
    axes[1].set(yticks=y, yticklabels=[f"{r['seed']} {r['environment']} {r['scenario']}" for r in interventions],
                xlabel="Published positions / 100 dry-tail epochs",
                title="Identical channel suffix; different gate history", xlim=(0, 105))
    axes[1].legend(loc="lower right")
    figure.suptitle("Post-hoc synthetic diagnostic; uncalibrated wetting, engineering gate")
    figure.savefig(output/"recovery_history.png", dpi=160)
    plt.close(figure)


def check_iq_suffix():
    """Spot-check the dry suffix claim at sample level, including paired noise."""
    checks = []
    simulator = CpuSignalSimulator(FS, noise_floor_db=-10000)
    satellites = constellation()
    for environment in ("open", "wall"):
        wall = VerticalWall(0., -100., 100., 0., 100., antenna_lhcp_response=.1j)
        for prn in (1, 12):
            for block in (400, 450, 499):
                signals = {}
                for scenario in ("dry", *SCENARIOS):
                    signal = simulate_wall_signal(
                        simulator, wall, RECEIVER, [0., 0., 0.], satellites[prn-1],
                        COUNT, prn=prn, start_sample=block*COUNT,
                        reference_range_m=REFERENCE, nav_bit=-1 if (block//20) % 2 else 1,
                        include_reflection=environment == "wall", integrate_samples=True,
                        antenna_response=RESPONSES[scenario])
                    signals[scenario] = signal.iq + noise_iq(
                        24686, prn, block, signal.direct_antenna_phase_rad)
                for scenario in SCENARIOS:
                    equal = np.array_equal(signals["dry"], signals[scenario])
                    checks.append(dict(environment=environment, prn=prn, block=block,
                                       scenario=scenario, seed=24686, iq_exactly_equal=equal))
                    if not equal:
                        raise RuntimeError("dry suffix IQ spot-check failed")
    return checks


def run(root=Path(".")):
    root = Path(root)
    output = root/"output/research-recovery-history"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [root/"output"/directory/filename for directory, filename in DATASETS]
    inputs += [root/"output/cpu-position-quality/holdout_channels.csv",
               root/"docs/position_quality_policy.json"]
    provenance = dict(
        study_type="post_hoc_development_diagnostic_not_confirmatory",
        suffix_start_block=400, block_duration_s=.001,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limitations=["All noise seeds have been inspected before this audit.",
                     "One static synthetic constellation; isolated PRN channels; no carrier tracking.",
                     "Compressed 0.5 s wetting schedule and antenna responses are not weather calibrated.",
                     "Frozen gate is an engineering baseline, not a published state-of-the-art method.",
                     "Sample-level IQ equality is spot-checked, not exhaustively checked."])
    raw = []
    for directory, filename in DATASETS:
        groups = read_groups(root/"output"/directory/filename)
        for (seed, environment, scenario), rows in sorted(groups.items()):
            if scenario not in SCENARIOS:
                continue
            wet = {int(r["block"]): r for r in rows if int(r["block"]) >= 400}
            dry = {int(r["block"]): r for r in groups[seed, environment, "dry"] if int(r["block"]) >= 400}
            if set(wet) != set(range(400, 500)) or wet.keys() != dry.keys():
                raise ValueError("incomplete/unaligned tail")
            differences = [np.linalg.norm([float(wet[b][k])-float(dry[b][k])
                                           for k in ("east_m", "north_m", "up_m")])
                           for b in wet if wet[b]["available"] == dry[b]["available"] == "True"]
            raw.append(dict(dataset=directory, seed=seed, environment=environment, scenario=scenario,
                            tail_epochs=100, paired_available_epochs=len(differences),
                            max_paired_raw_difference_m=max(differences) if differences else None))
    write_csv(output/"raw_tail_pairs.csv", raw)

    channels = read_groups(inputs[-2])
    config = PositionGateConfig(**json.loads(inputs[-1].read_text()))
    provenance["publication_policy"] = asdict(config)
    satellites = constellation()
    interventions = []
    for (seed, environment, scenario), wet in sorted(channels.items()):
        if scenario not in SCENARIOS:
            continue
        dry = channels[seed, environment, "dry"]
        # The same saved channel rows are used, not reconstructed dry measurements.
        shared = [r for r in wet if int(r["block"]) < 400] + [r for r in dry if int(r["block"]) >= 400]
        traces = {name: replay_channels(values, config, satellites)
                  for name, values in (("dry", dry), ("native", wet), ("shared", shared))}
        tail = {name: {b: d for b, d in values if b >= 400} for name, values in traces.items()}
        raw_equal = all(np.array_equal(tail["dry"][b].candidate.position_m,
                                      tail["shared"][b].candidate.position_m)
                        if tail["dry"][b].candidate is not None and tail["shared"][b].candidate is not None
                        else tail["dry"][b].candidate is tail["shared"][b].candidate
                        for b in tail["dry"])
        if not raw_equal:
            raise RuntimeError("gate changed raw solution under identical channel suffix")
        interventions.append(dict(
            seed=seed, environment=environment, scenario=scenario, tail_epochs=100,
            dry_prefix_published=sum(d.position_m is not None for d in tail["dry"].values()),
            wet_prefix_native_published=sum(d.position_m is not None for d in tail["native"].values()),
            wet_prefix_shared_dry_suffix_published=sum(d.position_m is not None for d in tail["shared"].values()),
            identical_suffix_raw_candidates=raw_equal))
    write_csv(output/"gate_state_intervention.csv", interventions)
    checks = check_iq_suffix()
    write_csv(output/"iq_suffix_spot_checks.csv", checks)
    plot_results(output, raw, interventions)
    provenance["results"] = dict(
        raw_paired_episodes=len(raw), paired_epochs=sum(r["paired_available_epochs"] for r in raw),
        max_raw_tail_difference_m=max(r["max_paired_raw_difference_m"] for r in raw),
        intervention_episodes=len(interventions),
        episodes_with_different_publication_count_under_same_suffix=sum(
            r["dry_prefix_published"] != r["wet_prefix_shared_dry_suffix_published"] for r in interventions),
        iq_equality_spot_checks=len(checks), all_iq_spot_checks_equal=all(r["iq_exactly_equal"] for r in checks))
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance["results"], indent=2))
    return provenance


if __name__ == "__main__":
    run()
