"""Plot blind nuisance-cancellation detector outcomes, including null controls."""

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
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    max_null = 0.
    for column, k in enumerate((1, 4)):
        axis = axes[0, column]
        for mode, offset in (("raw", -.18), ("cancelled", .18)):
            selected = [report["null_results"][profile][f"{mode}_{k}"] for profile in profiles]
            y = np.array([row["fraction"] for row in selected])*100
            bounds = np.array([row["pointwise_wilson_95"] for row in selected])*100
            max_null = max(max_null, float(bounds[:, 1].max()))
            axis.bar(np.arange(3)+offset, y, width=.36, label=mode,
                     yerr=np.maximum(0., [y-bounds[:, 0], bounds[:, 1]-y]), capsize=3)
        axis.set(title=f"{k} ms: target absent, separately calibrated gates", xticks=range(3),
                 xticklabels=("White", "11 other PRNs", "Other PRNs +20 dB"), ylabel="Passage rate (%)")
        axis.axhline(1., color="gray", linestyle="--")
        axis.grid(axis="y", alpha=.2)
        axis = axes[1, column]
        for doppler, color in ((170., "tab:blue"), (700., "tab:red")):
            for mode, style in (("raw", "--"), ("cancelled", "-")):
                selected = sorted((row for row in rows if row["profile"] == "near_far_20db"
                    and row["integration_ms"] == k and row["primary_doppler_hz"] == doppler and row["mode"] == mode),
                    key=lambda row: row["nominal_primary_cn0_db_hz"])
                axis.plot([row["nominal_primary_cn0_db_hz"] for row in selected],
                          [row["criterion_fraction"] for row in selected], style, marker="o", color=color,
                          label=f"{mode}, {doppler:g} Hz")
        axis.set(title=f"{k} ms: target present with strong other PRNs", xlabel="Nominal target C/N0 (dB-Hz)",
                 ylabel="Detection + coarse carrier/phase criterion", ylim=(-.03, 1.03), xticks=(37, 41, 45))
        axis.grid(alpha=.2)
    for axis in axes[0]:
        axis.set_ylim(0., max(1.2, max_null*1.12))
    axes[0, 0].legend()
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Blind other-satellite projection: full-pipeline null calibration and sensitivity\n"
                 "Finite IID strata; null intervals pointwise; signal curves descriptive")
    fig.savefig(directory/"cancellation_results.png", dpi=160)
    fig.savefig(directory/"cancellation_results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/cpu-ca-cancellation"))
    plot(parser.parse_args().directory)
