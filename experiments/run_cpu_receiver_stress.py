"""Exploratory receiver stress matrix with independent calibration and nulls.

Conditions share noise within each replica for paired comparisons. Neither
truth nor condition labels enter the receiver. This is not a rain experiment.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.run_cpu_batch_receiver import FS, COUNT, scene, render, receive, json_default
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, acquire_ca_signal, calibrate_ca_search
from gnss_gpu.input_validation import positive_int

SEED = 20261024
CALIBRATION, EVALUATION, SIGNAL = 0x53544341, 0x53544E55, 0x53545349
NOISE_DB = float(10*np.log10(FS/2)-55.)


def seed_for(domain, replica, block):
    return int(np.random.SeedSequence([SEED, domain, replica, block]).generate_state(1, dtype=np.uint64)[0])


def make_scene(cn0, doppler, outage_ms, replica, blocks=64):
    rows, starts, _, _ = scene(blocks)
    outage_ms = positive_int("outage_ms", outage_ms)
    begin = blocks//3
    end = begin+outage_ms
    if end > blocks-2:
        raise ValueError("outage must leave two post-return blocks")
    phase = np.random.default_rng(seed_for(SIGNAL, replica, blocks)).uniform(0., 2*np.pi, 12)
    for index, row in enumerate(rows):
        for channel in row:
            prn = channel["prn"]
            channel["amplitude"] = 0. if begin <= index < end else (10**((cn0-55.)/20) if prn == 1 else .4/prn)
            channel["carrier_phase"] = float(phase[prn-1])
            if prn == 1:
                channel["doppler_hz"] = float(doppler)
    seeds = [seed_for(SIGNAL, replica, index) for index in range(blocks)]
    return rows, starts, seeds, (begin, end)


def summarize(trace, outage, doppler):
    begin, end = outage
    pre_acquired = any(row.measurement_used for row in trace[:begin])
    lost = pre_acquired and any(row.status == "lost" for row in trace[begin:end])
    initial = next((i+1 for i, row in enumerate(trace[:begin]) if row.status == "acquired"), None)
    recovery = next((i+1 for i, row in enumerate(trace[end:]) if row.measurement_used), None) if pre_acquired else None
    reacquisition = next((i+1 for i, row in enumerate(trace[end:]) if row.status == "acquired"), None) if lost else None
    present = [row for i, row in enumerate(trace) if not begin <= i < end]
    errors = [abs(row.carrier_hz-doppler) for row in present if row.measurement_used]
    return dict(present_blocks=len(present), used_present_blocks=sum(row.measurement_used for row in present),
                absent_blocks=end-begin, used_absent_blocks=sum(row.measurement_used for row in trace[begin:end]),
                accepted_absent_blocks=sum(row.detection_accepted for row in trace[begin:end]),
                initial_acquisition_before_outage_ms=initial, acquired_before_outage=pre_acquired,
                lost_during_outage=lost, first_fresh_after_return_ms=recovery,
                reacquisition_after_loss_ms=reacquisition,
                conditional_carrier_mae_hz=float(np.mean(errors)) if errors else None,
                states=dict(Counter(row.status for row in trace)))


def run(output, *, replicas=3, blocks=64, calibration_trials=199, null_trials=199, baseline_dir=None):
    replicas = positive_int("replicas", replicas)
    blocks = positive_int("blocks", blocks)
    calibration_trials = positive_int("calibration_trials", calibration_trials)
    null_trials = positive_int("null_trials", null_trials)
    search = CaSearch(FS, (-500., 0., 500.), prn=1)
    simulator = CpuSignalSimulator(FS, noise_floor_db=NOISE_DB)
    def null_iq(domain, index):
        simulator.noise_seed = seed_for(domain, 0, index)
        return simulator.generate_epoch([], COUNT)
    scores = [acquire_ca_signal(null_iq(CALIBRATION, i), search).statistic for i in range(calibration_trials)]
    threshold = calibrate_ca_search(scores, search)
    null_blocks = np.stack([null_iq(EVALUATION, i) for i in range(null_trials)])
    null_trace = receive(null_blocks, threshold, [i*COUNT for i in range(null_trials)])
    baseline = None
    if baseline_dir is not None:
        from benchmarks.bench_cpu_receiver_preparation import load_baseline
        baseline = load_baseline(Path(baseline_dir))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, iq_hashes = [], {}
    with (output/"traces.jsonl").open("w") as handle:
        for condition, (cn0, doppler, outage_ms, replica) in enumerate(itertools.product(
                (35., 45., 55.), (0., 170., 700.), (2, 8), range(replicas))):
            rows, starts, seeds, outage = make_scene(cn0, doppler, outage_ms, replica, blocks)
            iq = render("batch", rows, starts, seeds, NOISE_DB)
            np.testing.assert_array_equal(iq, render("individual", rows, starts, seeds, NOISE_DB))
            trace = receive(iq, threshold, starts)
            if baseline is not None:
                tracker = baseline["tracking_cpu"].CpuCaTracker(threshold, confirmation_blocks=2, coast_blocks=3, workers=1)
                previous = [tracker.update(row, start_sample=start) for row, start in zip(iq, starts)]
                assert [asdict(x) for x in previous] == [asdict(x) for x in trace]
            identity = dict(condition=condition, nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler,
                            outage_ms=outage_ms, replica=replica)
            summaries.append(dict(**identity, **summarize(trace, outage, doppler)))
            iq_hashes[str(condition)] = hashlib.sha256(iq.tobytes()).hexdigest()
            handle.write(json.dumps(dict(**identity, outage_blocks=outage,
                                        trace=[asdict(x) for x in trace]), default=json_default)+"\n")
    with (output/"summary.csv").open("w", newline="") as handle:
        fields = [key for key in summaries[0] if key != "states"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"experiments/run_cpu_batch_receiver.py"]
    paths += [root/"python/gnss_gpu"/(name+".py") for name in
              ("signal_sim_cpu", "code_delay_cpu", "acquisition_cpu", "code_detection_cpu", "tracking_cpu")]
    metadata = dict(seed=SEED, replicas=replicas, blocks=blocks, sampling_freq_hz=FS, noise_floor_db=NOISE_DB,
                    threshold=asdict(threshold), calibration_scores=scores,
                    noise_domains=dict(calibration=CALIBRATION, held_out_null=EVALUATION, signal=SIGNAL),
                    iq_sha256=iq_hashes, batch_individual_iq_equal=True,
                    baseline_receiver_results_equal=True if baseline is not None else None,
                    baseline_source_sha256={name: hashlib.sha256((Path(baseline_dir)/(name+".py")).read_bytes()).hexdigest()
                                            for name in baseline} if baseline is not None else None,
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    numpy=np.__version__, held_out_null=dict(blocks=null_trials,
                        accepted_blocks=sum(row.detection_accepted for row in null_trace),
                        used_blocks=sum(row.measurement_used for row in null_trace),
                        states=dict(Counter(row.status for row in null_trace))),
                    limitations=["Exploratory 3-replica matrix, no population performance or rare-event claim.",
                      "Conditions share noise and phases within replicas; they are paired, not independent trials.",
                      "Nominal C/N0 is based on primary amplitude and per-I/Q white noise, not estimated C/N0.",
                      "All 12 signal paths are muted during outage; other PRNs remain fixed-power outside it.",
                      "Point-sampled synthesis versus aperture delay fit; no absolute ranging validation.",
                      "700 Hz is beyond the fixed carrier grid; local refinement cannot recover out-of-grid frequency.",
                      "Per-block noise calibration is not a track-level guarantee; no rain, foliage, PVT or RF validation."])
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    plot(summaries, output)
    return metadata, summaries


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, outage in zip(axes, (2, 8)):
        matrix = [[np.mean([r["used_present_blocks"]/r["present_blocks"] for r in rows
                   if r["outage_ms"] == outage and r["nominal_primary_cn0_db_hz"] == cn0
                   and r["primary_doppler_hz"] == doppler]) for doppler in (0., 170., 700.)] for cn0 in (35., 45., 55.)]
        image = axis.imshow(matrix, vmin=0., vmax=1., cmap="viridis")
        for i, j in itertools.product(range(3), repeat=2):
            axis.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", color="white" if matrix[i][j]<.5 else "black")
        axis.set(xticks=range(3), xticklabels=(0, 170, 700), yticks=range(3), yticklabels=(35, 45, 55),
                 xlabel="True primary Doppler (Hz)", ylabel="Nominal primary C/N0 (dB-Hz)", title=f"{outage} ms all-signal outage")
    fig.colorbar(image, ax=axes, label="Fresh measurements / signal-present blocks")
    fig.suptitle("Synthetic receiver stress: exploratory paired replicas")
    fig.savefig(output/"availability.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-receiver-stress"))
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    metadata, rows = run(args.output, replicas=args.replicas, blocks=args.blocks, baseline_dir=args.baseline_dir)
    print(json.dumps(dict(conditions=len(rows), held_out_null=metadata["held_out_null"],
                          baseline_receiver_results_equal=metadata["baseline_receiver_results_equal"]), indent=2))
