"""Render all available 3D errors, including failures, from an existing replay.

PYTHONPATH=python:. python experiments/plot_cpu_wet_antenna_errors.py
No IQ regeneration, threshold fitting or selection of only recovered positions.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/cpu-wet-antenna")
    args = parser.parse_args()
    output = Path(args.output_dir)
    with (output/"positions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads((output/"metadata.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    seeds = metadata["seeds"]
    figure, axes = plt.subplots(2, len(seeds), figsize=(14, 10), squeeze=False, constrained_layout=True)
    for i, environment in enumerate(("open", "wall")):
        for j, seed in enumerate(seeds):
            axis = axes[i, j]
            for mode in ("dry", "loss_6db", "loss_30db_stress", "combined_stress"):
                selected = [row for row in rows if int(row["seed"]) == seed and row["environment"] == environment and row["scenario"] == mode]
                errors = [float(row["position_error_m"]) if row["position_error_m"] else np.nan for row in selected]
                axis.plot([float(row["time_s"]) for row in selected], errors, label=mode)
            axis.axhline(100., color="black", linestyle="--", linewidth=.7, label="100 m diagnostic threshold")
            axis.axvspan(.05, .2, color="blue", alpha=.06)
            axis.axvspan(.2, .4, color="orange", alpha=.08)
            axis.set_yscale("symlog", linthresh=20.)
            axis.set(xlabel="time [s]", ylabel="3D error [m], symlog", title=f"{environment}, seed {seed}")
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
    figure.suptitle("Wet-antenna stress: available position does not prove accuracy; gaps = unavailable")
    figure.savefig(output/"wet_antenna_errors.png", dpi=150)
    plt.close(figure)
    provenance = dict(input_positions_sha256=hashlib.sha256((output/"positions.csv").read_bytes()).hexdigest(),
                      input_metadata_sha256=hashlib.sha256((output/"metadata.json").read_bytes()).hexdigest(),
                      renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      selection="All positions in four labeled scenarios, both seeds/environments; no error clipping or recovered-only filtering; missing solutions stay gaps")
    (output/"error_plot_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Saved failed-position timelines to {output/'wet_antenna_errors.png'}")


if __name__ == "__main__":
    main()
