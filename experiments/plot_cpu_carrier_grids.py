"""Plot paired carrier errors on common fresh-measurement blocks."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((directory/"summary.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for axis, cn0 in zip(axes, (45., 55.)):
        for name, offset in (("before", -.18), ("after", .18)):
            means = []
            for frequency in (0., 170., 700.):
                selected = [row for row in rows if row["nominal_primary_cn0_db_hz"] == cn0
                            and row["primary_doppler_hz"] == frequency and row["common_used_blocks"]]
                count = sum(row["common_used_blocks"] for row in selected)
                means.append(sum(row["common_used_blocks"]*row[f"common_carrier_mae_{name}_hz"] for row in selected)/count
                             if count else np.nan)
            axis.bar(np.arange(3)+offset, means, width=.36, label="±500 Hz" if name == "before" else "±1000 Hz, recalibrated")
        axis.set(xticks=range(3), xticklabels=(0, 170, 700), xlabel="True primary Doppler (Hz)",
                 title=f"Nominal C/N0 = {cn0:g} dB-Hz")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Carrier MAE on common fresh blocks (Hz)")
    axes[0].legend()
    figure.suptitle("Paired synthetic grid comparison; selected blocks, not an integrity bound")
    figure.savefig(directory/"paired_carrier_error.png", dpi=160)
    figure.savefig(directory/"paired_carrier_error.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/cpu-carrier-grid-comparison"))
    plot(parser.parse_args().directory)
