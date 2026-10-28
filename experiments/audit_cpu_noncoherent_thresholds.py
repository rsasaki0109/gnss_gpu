"""Compare old/new thresholds on the SAME fresh held-out candidate scores."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.run_cpu_receiver_sensitivity import wilson_interval


def compare_same_stream(records, old_thresholds, new_thresholds):
    if not records:
        raise ValueError("evaluation records required")
    result = {}
    for key in ("1", "4"):
        old = [row["candidates"][key]["statistic"] > old_thresholds[key] for row in records]
        new = [row["candidates"][key]["statistic"] > new_thresholds[key] for row in records]
        if new != [row["accepted"][key] for row in records]:
            raise ValueError("saved acceptance does not match frozen new threshold")
        result[key] = dict(records=len(records), old_accepted=sum(old), new_accepted=sum(new),
            old_fraction=sum(old)/len(old), new_fraction=sum(new)/len(new),
            old_wilson_95=wilson_interval(sum(old), len(old)), new_wilson_95=wilson_interval(sum(new), len(new)),
            only_old=sum(a and not b for a, b in zip(old, new)),
            only_new=sum(b and not a for a, b in zip(old, new)))
    return result


def audit(directory, prior_dir):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_path, report_path = directory/"protocol.json", directory/"report.json"
    protocol, report = json.loads(protocol_path.read_text()), json.loads(report_path.read_text())
    prior_path = prior_dir/"report.json"
    if digest(prior_path) != protocol["prior_input_sha256"]["report.json"] or digest(protocol_path) != report["protocol_sha256"]:
        raise ValueError("input hashes differ from validation record")
    if digest(directory/"frozen_thresholds.json") != report["frozen_thresholds_sha256"]:
        raise ValueError("frozen threshold file changed")
    records = [json.loads(line) for line in (directory/"evaluation.jsonl").read_text().splitlines()]
    if len(records) != protocol["evaluation_records"]:
        raise ValueError("incomplete evaluation stream")
    result = compare_same_stream(records, report["prior_thresholds"], report["new_thresholds"])
    (directory/"same_stream_threshold_comparison.json").write_text(json.dumps(dict(per_detector=result,
        evaluation_sha256=digest(directory/"evaluation.jsonl"), audit_sha256=digest(Path(__file__))), indent=2)+"\n")
    plot(directory, json.loads(prior_path.read_text()), result)
    return result


def plot(directory, prior, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained", sharey=True)
    max_upper = 0.
    for axis, key in zip(axes, ("1", "4")):
        row = result[key]
        counts = [prior["null_accepted"][key], row["old_accepted"], row["new_accepted"]]
        sizes = [prior["null_records"], row["records"], row["records"]]
        y = np.array([k/n for k, n in zip(counts, sizes)])*100
        intervals = np.array([wilson_interval(k, n) for k, n in zip(counts, sizes)])*100
        max_upper = max(max_upper, float(intervals[:, 1].max()))
        axis.errorbar(range(3), y, yerr=np.maximum(0., [y-intervals[:, 0], intervals[:, 1]-y]), fmt="o", capsize=4)
        axis.axhline(1., linestyle="--", color="gray", label="1% marginal calibration target")
        axis.set(xticks=range(3), xticklabels=("Original data\nold threshold", "Fresh data\nold threshold", "Same fresh data\nnew threshold"),
                 title=f"{key} ms noncoherent windows")
        for x, count, n in zip(range(3), counts, sizes):
            axis.annotate(f"{count}/{n}", (x, y[x]), xytext=(0, 8), textcoords="offset points", ha="center")
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Noise passage rate (%) with pointwise 95% Wilson interval")
    axes[0].set_ylim(0., max_upper*1.12)
    axes[0].legend(fontsize=8)
    figure.suptitle("Noise validation: distinguish evaluation-sample changes from threshold changes")
    figure.savefig(directory/"null_validation.png", dpi=160)
    figure.savefig(directory/"null_validation.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/cpu-noncoherent-calibration-validation"))
    parser.add_argument("--prior-dir", type=Path, default=Path("output/cpu-noncoherent-sensitivity"))
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.prior_dir), indent=2))
