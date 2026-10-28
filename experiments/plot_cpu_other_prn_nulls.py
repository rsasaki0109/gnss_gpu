"""Plot the false-alarm/sensitivity cost of a fixed multi-stratum threshold."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((directory/"report.json").read_text())
    rows = json.loads((directory/"aggregate.json").read_text())
    profiles = ("white", "other_prns", "near_far_20db")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    top_max = 0.
    for column, k in enumerate(("1", "4")):
        axis = axes[0, column]
        for gate, offset, label in (("white_calibrated", -.18, "White-noise calibration"),
                                    ("stratified", .18, "Fixed stratified maximum")):
            selected = [report["null_results"][gate][profile][k] for profile in profiles]
            y = np.array([row["fraction"] for row in selected])*100
            bounds = np.array([row["pointwise_wilson_95"] for row in selected])*100
            top_max = max(top_max, float(bounds[:, 1].max()))
            axis.bar(np.arange(3)+offset, y, width=.36, label=label,
                     yerr=np.maximum(0., [y-bounds[:, 0], bounds[:, 1]-y]), capsize=3)
        axis.set(title=f"{k} ms accumulation: PRN 1 absent", xticks=range(3),
                 xticklabels=("White only", "11 other PRNs", "Other PRNs\n+20 dB power"), ylabel="Noise/nuisance passage rate (%)")
        axis.axhline(1., color="gray", linestyle="--")
        axis.grid(axis="y", alpha=.2)
        axis = axes[1, column]
        for profile, color in (("other_prns", "tab:blue"), ("near_far_20db", "tab:red")):
            for gate, style in (("white_calibrated", "--"), ("stratified", "-")):
                selected = sorted((row for row in rows if row["profile"] == profile and row["gate"] == gate
                    and row["integration_ms"] == int(k) and row["primary_doppler_hz"] == 170.),
                    key=lambda row: row["nominal_primary_cn0_db_hz"])
                x = [row["nominal_primary_cn0_db_hz"] for row in selected]
                y = [row["criterion_fraction"] for row in selected]
                axis.plot(x, y, style, marker="o", color=color, label=f"{profile}, {'white' if gate == 'white_calibrated' else 'stratified'} gate")
        axis.set(title=f"{k} ms: PRN 1 present at 170 Hz, forced bit reversal", xlabel="Nominal primary C/N0 (dB-Hz)",
                 ylabel="Detection + coarse carrier/phase criterion", xticks=(37, 41, 45), ylim=(-.03, 1.03))
        axis.grid(alpha=.2)
    for axis in axes[0]:
        axis.set_ylim(0., top_max*1.12)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=7)
    figure.suptitle("Target-absent cross-correlation control versus sensitivity cost\n"
                   "Null intervals: pointwise Wilson; signal curves descriptive; fixed finite nuisance family")
    figure.savefig(directory/"other_prn_tradeoff.png", dpi=160)
    figure.savefig(directory/"other_prn_tradeoff.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/cpu-other-prn-nulls"))
    plot(parser.parse_args().directory)
