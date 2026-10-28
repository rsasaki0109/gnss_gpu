"""Show all held-out availability costs, including the dry-tail rejection case.

PYTHONPATH=python:. python experiments/plot_cpu_position_quality_availability.py
Reads saved decisions; no gate retuning or IQ regeneration.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/cpu-position-quality")
    args = parser.parse_args()
    output = Path(args.output_dir)
    with (output/"summary.csv").open() as stream:
        summary = [row for row in csv.DictReader(stream) if row["dataset"] == "holdout"]
    with (output/"positions.csv").open() as stream:
        positions = [row for row in csv.DictReader(stream) if row["dataset"] == "holdout"]
    metadata = json.loads((output/"metadata.json").read_text())
    seeds, scenarios = metadata["holdout_seeds"], metadata["holdout_scenarios"]
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    for axis, environment in zip(axes[:2], ("open", "wall")):
        keys = [(seed, scenario) for seed in seeds for scenario in scenarios]
        for mode, offset in (("raw", -.2), ("quality_gate", .2)):
            lookup = {(int(row["seed"]), row["scenario"]):int(row["available"])
                      for row in summary if row["environment"] == environment and row["mode"] == mode}
            axis.bar(np.arange(len(keys))+offset, [lookup[key] for key in keys], .4, label=mode)
        axis.set(xticks=np.arange(len(keys)), xticklabels=[f"{seed}/{scenario}" for seed, scenario in keys],
                 ylabel="published / 500 input blocks", ylim=(0, 520), title=f"{environment}: all held-out episodes")
        axis.tick_params(axis="x", labelrotation=20)
        axis.legend()
        axis.grid(axis="y", alpha=.2)
    # Explicit diagnostic, selected after evaluation and labeled as such.
    for mode, level in (("raw", 1.), ("quality_gate", 0.)):
        rows = [row for row in positions if int(row["seed"]) == 24686 and row["environment"] == "wall"
                and row["scenario"] == "loss_6db" and row["mode"] == mode]
        axes[2].scatter([float(row["time_s"]) for row in rows],
                        [level if row["available"] == "True" else np.nan for row in rows], s=6, label=mode)
    axes[2].axvspan(.4, .5, color="red", alpha=.12, label="fully dry: raw 100/100, gated 0/100")
    axes[2].set(xlabel="time [s]", yticks=[0, 1], yticklabels=["quality gate", "raw"],
                 ylim=(-.5, 1.5), title="Post-evaluation diagnostic: seed 24686, wall, 6 dB loss; dots = published")
    axes[2].legend()
    axes[2].grid(alpha=.2)
    figure.suptitle("Frozen position policy: error reduction has an availability cost; no holdout retuning")
    figure.savefig(output/"position_quality_availability.png", dpi=150)
    plt.close(figure)
    provenance = dict(inputs_sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                                     for name in ("summary.csv", "positions.csv", "metadata.json")},
                      renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      diagnostic_selection="All heldout episode counts plus explicitly post-evaluation worst 6 dB wall availability example; no parameter changes")
    (output/"availability_plot_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Saved complete availability costs and dry-tail failure to {output/'position_quality_availability.png'}")


if __name__ == "__main__":
    main()
