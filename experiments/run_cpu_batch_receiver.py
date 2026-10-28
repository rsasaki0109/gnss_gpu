"""Continuous batch IQ -> calibrated acquisition/code following equivalence.

One synthetic composite-signal regression, not positioning or real RF validation.
Calibration uses separate noise seeds and the complete fixed carrier grid.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numba
import numpy as np

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, acquire_ca_signal, calibrate_ca_search
from gnss_gpu.input_validation import positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.tracking_cpu import CpuCaTracker


FS, COUNT, SEED = 2*CHIP_RATE, 2046, 424242
CALIBRATION_DOMAIN, SIGNAL_DOMAIN = 0x42415443, 0x42415453


def noise_seed(domain, index):
    return int(np.random.SeedSequence([SEED, domain, index]).generate_state(1, dtype=np.uint64)[0])


def scene(blocks):
    if blocks < 16:
        raise ValueError("at least 16 blocks required for outage and confirmation")
    outage = (3*blocks//8, blocks//2)
    bits = [1 if index % 2 == 0 else -1 for index in range((blocks+19)//20+1)]
    rows = []
    for block in range(blocks):
        rows.append([dict(prn=prn, code_phase=-18.5 if prn == 1 else prn*3.7,
                          carrier_phase=.2+prn*.3, doppler_hz=170. if prn == 1 else (-1.)**prn*prn*150.,
                          amplitude=0. if outage[0] <= block < outage[1] else (1. if prn == 1 else .4/prn),
                          nav_bit=1, nav_bits=bits) for prn in range(1, 13)])
    return rows, [index*COUNT for index in range(blocks)], [noise_seed(SIGNAL_DOMAIN, i) for i in range(blocks)], outage


def render(mode, rows, starts, seeds, noise_db):
    simulator = CpuSignalSimulator(FS, noise_floor_db=noise_db)
    if mode == "batch":
        return simulator.generate_epochs(rows, COUNT, noise_seeds=seeds, start_samples=starts)
    output = []
    for row, start, seed in zip(rows, starts, seeds):
        simulator.noise_seed = seed
        output.append(simulator.generate_epoch(row, COUNT, start_sample=start))
    return np.stack(output)


def receive(iq, threshold, starts):
    tracker = CpuCaTracker(threshold, confirmation_blocks=2, coast_blocks=3, workers=1)
    return [tracker.update(row, start_sample=start) for row, start in zip(iq, starts)]


def run_experiment(*, blocks=128, repeats=5, calibration_trials=199):
    blocks, repeats = positive_int("blocks", blocks), positive_int("repeats", repeats)
    calibration_trials = positive_int("calibration_trials", calibration_trials)
    rows, starts, seeds, outage = scene(blocks)
    noise_db = 10*np.log10(FS/2)-55.
    search = CaSearch(FS, (-500., 0., 500.), prn=1)
    original_threads = numba.get_num_threads()
    numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
    try:
        calibrator = CpuSignalSimulator(FS, noise_floor_db=noise_db)
        scores = []
        for index in range(calibration_trials):
            calibrator.noise_seed = noise_seed(CALIBRATION_DOMAIN, index)
            candidate = acquire_ca_signal(calibrator.generate_epoch([], COUNT), search)
            scores.append(candidate.statistic)
        threshold = calibrate_ca_search(scores, search)
        # Freeze once before comparisons. Truth/outage labels never enter the receiver.
        metadata = dict(blocks=blocks, samples_per_block=COUNT, sampling_freq_hz=FS, signal_duration_s=blocks*.001,
                        repeats=repeats, threads=numba.get_num_threads(), seed=SEED,
                        software=dict(numpy=np.__version__, numba=numba.__version__),
                        signal_domain=SIGNAL_DOMAIN, calibration_domain=CALIBRATION_DOMAIN,
                        per_block_noise_seeds=seeds, nominal_primary_cn0_db_hz=55., noise_floor_db=float(noise_db),
                        outage_blocks=list(outage), outage_end_exclusive=True,
                        threshold=asdict(threshold), calibration_scores=scores,
                        tracker=dict(prn=1, confirmation_blocks=2, coast_blocks=3, workers=1),
                        timing_scope="generation then receiver construction/updates; JIT/calibration/scoring/serialization excluded")
        reference_iq = render("individual", rows, starts, seeds, noise_db)
        reference_trace = receive(reference_iq, threshold, starts)
        batch_iq = render("batch", rows, starts, seeds, noise_db)
        np.testing.assert_array_equal(batch_iq, reference_iq)
        assert receive(batch_iq, threshold, starts) == reference_trace
        timings = []
        for repeat in range(repeats):
            for mode in (("individual", "batch") if repeat % 2 == 0 else ("batch", "individual")):
                begin = time.perf_counter()
                iq = render(mode, rows, starts, seeds, noise_db)
                generated = time.perf_counter()
                trace = receive(iq, threshold, starts)
                received = time.perf_counter()
                timings.append(dict(repeat=repeat, mode=mode, generation_s=generated-begin,
                                    receiver_s=received-generated, total_s=received-begin))
                np.testing.assert_array_equal(iq, reference_iq)
                assert trace == reference_trace
        medians = {mode:{key:statistics.median(row[key] for row in timings if row["mode"] == mode)
                         for key in ("generation_s", "receiver_s", "total_s")} for mode in ("individual", "batch")}
        acquired = [i+1 for i, row in enumerate(reference_trace) if row.status == "acquired"]
        lost_during_outage = any(row.status == "lost" for row in reference_trace[outage[0]:outage[1]])
        reacquired = next((i+1-outage[1] for i, row in enumerate(reference_trace)
                          if i >= outage[1] and row.status == "acquired"), None) if lost_during_outage else None
        summary = dict(iq_bitwise_equal=True, receiver_results_equal=True, input_blocks=blocks,
                       signal_present_blocks=blocks-(outage[1]-outage[0]), signal_absent_blocks=outage[1]-outage[0],
                       used_signal_present_blocks=sum(row.measurement_used for i,row in enumerate(reference_trace)
                                                      if not outage[0] <= i < outage[1]),
                       used_signal_absent_blocks=sum(row.measurement_used for row in reference_trace[outage[0]:outage[1]]),
                       states=dict(Counter(row.status for row in reference_trace)),
                       initial_acquisition_decision_ms=acquired[0] if acquired else None,
                       lost_observed_during_outage=lost_during_outage, reacquisition_after_outage_ms=reacquired,
                       median_seconds=medians, generation_speedup=medians["individual"]["generation_s"]/medians["batch"]["generation_s"],
                       pipeline_speedup=medians["individual"]["total_s"]/medians["batch"]["total_s"])
    finally:
        numba.set_num_threads(original_threads)
    root = Path(__file__).resolve().parents[1]
    paths = ["experiments/run_cpu_batch_receiver.py", "python/gnss_gpu/signal_sim_cpu.py",
             "python/gnss_gpu/acquisition_cpu.py", "python/gnss_gpu/code_delay_cpu.py",
             "python/gnss_gpu/code_detection_cpu.py", "python/gnss_gpu/tracking_cpu.py"]
    metadata["source_sha256"] = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    metadata["iq_sha256"] = hashlib.sha256(reference_iq.tobytes()).hexdigest()
    metadata["limitations"] = ["One synthetic scenario; timing repeats are not independent GNSS trials.",
                                "Point-sampled rectangular C/A signals; receiver delay model assumes centered aperture averaging.",
                                "The mismatch is shared by both paths; equality is not an absolute-error validation.",
                                "Only PRN 1 is followed within the 12-PRN composite; no PLL/FLL/DLL, bit synchronization or PVT.",
                                "Per-block noise-only gate calibration is not a track-level false-alarm or integrity guarantee."]
    return dict(metadata=metadata, summary=summary, timings=timings, trace=reference_trace)


def json_default(value):
    if isinstance(value, complex):
        return dict(real=float(value.real), imag=float(value.imag))
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def plot_trace(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace, threshold = result["trace"], result["metadata"]["threshold"]["calibration"]["score"]
    time_ms = np.arange(1, len(trace)+1)
    states = ("searching", "confirming", "acquired", "tracking", "coasting", "lost")
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), layout="constrained")
    axes[0].plot(time_ms, [x.detection_statistic for x in trace], label="Full-grid statistic")
    axes[0].axhline(threshold, color="red", linestyle="--", label="Frozen noise calibration threshold")
    axes[0].set_ylabel("Detection statistic"); axes[0].legend()
    axes[1].step(time_ms, [states.index(x.status) for x in trace], where="post")
    axes[1].set(yticks=range(len(states)), yticklabels=states, xlabel="Observation-block end time (ms)", ylabel="Receiver state")
    for axis in axes:
        axis.axvspan(*result["metadata"]["outage_blocks"], color="red", alpha=.12)
        axis.grid(alpha=.2)
    figure.suptitle("Individual/batch IQ and all receiver results identical; one synthetic PRN-1 regression")
    figure.savefig(Path(output)/"receiver_equivalence.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-batch-receiver"))
    args = parser.parse_args()
    result = run_experiment(blocks=args.blocks, repeats=args.repeats)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("metadata", "summary"):
        (args.output/(name+".json")).write_text(json.dumps(result[name], indent=2, default=json_default)+"\n")
    (args.output/"receiver_trace.json").write_text(json.dumps([asdict(x) for x in result["trace"]], indent=2, default=json_default)+"\n")
    with (args.output/"timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["timings"][0])); writer.writeheader(); writer.writerows(result["timings"])
    plot_trace(result, args.output)
    print(json.dumps(result["summary"], indent=2, default=json_default))
