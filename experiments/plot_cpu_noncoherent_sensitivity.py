"""Plot exploratory integration gains alongside observed null-pass counts."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((directory/"aggregate.json").read_text())
    report = json.loads((directory/"report.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained", sharey=True)
    for axis, doppler in zip(axes, (170., 700.)):
        for k in (1, 4):
            selected = sorted((row for row in rows if row["integration_ms"] == k and row["primary_doppler_hz"] == doppler),
                              key=lambda row: row["nominal_primary_cn0_db_hz"])
            x = [row["nominal_primary_cn0_db_hz"] for row in selected]
            y = np.array([row["criterion_fraction"] for row in selected])
            lower = np.array([row["criterion_wilson_95"][0] for row in selected])
            upper = np.array([row["criterion_wilson_95"][1] for row in selected])
            axis.errorbar(x, y, yerr=np.maximum(0., [y-lower, upper-y]), marker="o", capsize=3,
                          label=f"{k} ms power accumulation, {20//k} windows")
        axis.set(xlabel="Nominal primary C/N0 (dB-Hz)", title=f"True carrier {doppler:g} Hz", ylim=(-.03, 1.03))
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Accepted + coarse carrier/phase criterion\nReplica fraction, pointwise 95% Wilson interval")
    axes[0].legend(fontsize=9)
    figure.suptitle("Matched 20-ms records; separately calibrated integer-delay detectors\n"
        f"Observed noise passes: 1 ms = {report['null_accepted']['1']}/{report['null_records']}, "
        f"4 ms = {report['null_accepted']['4']}/{report['null_records']}; empirical false alarms are not matched")
    figure.savefig(directory/"noncoherent_sensitivity.png", dpi=160)
    figure.savefig(directory/"noncoherent_sensitivity.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/cpu-noncoherent-sensitivity"))
    plot(parser.parse_args().directory)
