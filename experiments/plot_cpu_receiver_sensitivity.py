"""Export exploratory replica-event rates with pointwise Wilson intervals."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((directory/"aggregate.json").read_text())
    metadata = json.loads((directory/"metadata.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for axis, fraction, interval, title in (
            (axes[0], "acquisition_fraction", "acquisition_wilson_95", "Internal acquisition by 21 ms"),
            (axes[1], "carrier_criterion_fraction", "carrier_criterion_wilson_95", "Acquisition + first carrier error ≤100 Hz")):
        for doppler in metadata["doppler_hz"]:
            selected = sorted((row for row in rows if row["primary_doppler_hz"] == doppler),
                              key=lambda row: row["nominal_primary_cn0_db_hz"])
            x = [row["nominal_primary_cn0_db_hz"] for row in selected]
            y = np.array([row[fraction] for row in selected])
            lower = np.array([row[interval][0] for row in selected])
            upper = np.array([row[interval][1] for row in selected])
            axis.errorbar(x, y, yerr=np.maximum(0., [y-lower, upper-y]), marker="o", capsize=3,
                          label=f"{doppler:g} Hz")
        axis.set(xlabel="Nominal primary C/N0 (dB-Hz)", title=title, ylim=(-.03, 1.03),
                 xticks=metadata["nominal_cn0_db_hz"])
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Replica event fraction (pointwise 95% Wilson interval)")
    axes[0].legend()
    figure.suptitle(f"Frozen five-bin gate; {metadata['replicas']} replicas per condition; synthetic 1-ms receiver")
    figure.savefig(directory/"sensitivity.png", dpi=160)
    figure.savefig(directory/"sensitivity.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/cpu-receiver-sensitivity"))
    plot(parser.parse_args().directory)
