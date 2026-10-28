"""Export fresh-signal bit-timing results with calibration-budget context."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((directory/"aggregate.json").read_text())
    protocol = json.loads((directory/"protocol.json").read_text())
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained", sharex=True, sharey=True)
    for axis_row, mode in zip(axes, ("constant", "flip")):
        for axis, doppler in zip(axis_row, (170., 700.)):
            for k in (1, 4):
                selected = sorted((row for row in rows if row["bit_mode"] == mode
                    and row["primary_doppler_hz"] == doppler and row["integration_ms"] == k),
                    key=lambda row: row["nominal_primary_cn0_db_hz"])
                x = [row["nominal_primary_cn0_db_hz"] for row in selected]
                y = np.array([row["criterion_fraction"] for row in selected])
                lower = np.array([row["criterion_wilson_95"][0] for row in selected])
                upper = np.array([row["criterion_wilson_95"][1] for row in selected])
                axis.errorbar(x, y, yerr=np.maximum(0., [y-lower, upper-y]), marker="o", capsize=3,
                              label=f"{k} ms accumulation")
            axis.set(title=f"{doppler:g} Hz; {'constant polarity' if mode == 'constant' else 'forced within-period reversal'}",
                     ylim=(-.03, 1.03), xticks=protocol["signal_levels_db_hz"])
            axis.grid(alpha=.25)
    for axis in axes[-1]:
        axis.set_xlabel("Nominal primary C/N0 (dB-Hz)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accepted + coarse carrier/phase criterion\nPointwise 95% Wilson interval")
    axes[0, 0].legend()
    figure.suptitle(f"Fresh paired signals: {protocol['replicas']} replicas per condition, fixed 20-ms decisions\n"
                   "Common 1% white-noise false-alarm upper budget; true false-alarm rates need not be equal")
    figure.savefig(directory/"bit_validation.png", dpi=160)
    figure.savefig(directory/"bit_validation.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/cpu-noncoherent-bit-validation"))
    plot(parser.parse_args().directory)
