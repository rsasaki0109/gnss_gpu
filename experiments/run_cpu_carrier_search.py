"""CPU carrier/delay acquisition on paper-linked moving-source IQ.

PYTHONPATH=python:. python experiments/run_cpu_carrier_search.py --workers 4
Parallelizes independent null observations with spawned CPU processes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_cpu_wall_detection import wilson_interval
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import (
    CaSearch, acquire_ca_signal, calibrate_ca_search, detect_ca_search,
)
from gnss_gpu.input_validation import positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _noise_seed
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal

CALIBRATION_DOMAIN = 0x43415243
HELDOUT_DOMAIN = 0x43415248
SIGNAL_DOMAIN = 0x43415253


def _rng(seed, domain, trial):
    return np.random.default_rng(np.random.SeedSequence([seed, domain, trial]))


def _null_job(job):
    seed, domain, trial, search = job
    iq = _rng(seed, domain, trial).normal(size=2*round(search.sampling_freq*.001))
    candidate = acquire_ca_signal(iq, search)
    if candidate is None:
        raise RuntimeError("nonzero Gaussian null returned no acquisition candidate")
    return dict(domain=domain, trial=trial, statistic=candidate.statistic)


def run_experiment(*, calibration_trials=199, heldout_trials=400, workers=1, seed=20261024) -> dict[str, Any]:
    calibration_trials = positive_int("calibration_trials", calibration_trials)
    heldout_trials = positive_int("heldout_trials", heldout_trials)
    workers = positive_int("workers", workers)
    seed = _noise_seed(seed)
    search = CaSearch(2*CHIP_RATE, tuple(np.arange(-6000., 6001., 500.)))
    # Fail unsupported rank resolution before doing the expensive search.
    calibrate_ca_search(np.ones(calibration_trials), search)
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        def nulls(domain, count):
            jobs = [(seed, domain, trial, search) for trial in range(count)]
            return list(pool.map(_null_job, jobs, chunksize=4)) if pool else list(map(_null_job, jobs))

        calibration = nulls(CALIBRATION_DOMAIN, calibration_trials)
        gate = calibrate_ca_search([row["statistic"] for row in calibration], search)
        # Freeze before seeing any held-out score; never retune on evaluation.
        heldout = nulls(HELDOUT_DOMAIN, heldout_trials)
    finally:
        if pool:
            pool.shutdown()
    for row in heldout:
        row["accepted"] = row["statistic"] > gate.calibration.score

    fs = search.sampling_freq
    sim = CpuSignalSimulator(fs, noise_floor_db=-10000)
    wall = VerticalWall(0., -100., 100., 0., 100., antenna_lhcp_response=.1)
    trials = []
    for time_index, time_s in enumerate(np.linspace(0., .2, 11)):
        start = round(time_s*fs)
        for path_index, path in enumerate(("direct", "reflection")):
            block = simulate_wall_signal(
                sim, wall, [-20., 0., 2.], [5., 0., 0.], [-20000020., 0., 2.], 2046,
                start_sample=start, reference_range_m=20000000.,
                transmitter_velocity_mps=[1000., 0., 0.],
                rx_acceleration_mps2=[200., 0., 0.],
                transmitter_acceleration_mps2=[2., 0., 0.],
                include_direct=path == "direct", include_reflection=path == "reflection",
                integrate_samples=True)
            ranges = block.direct_range_m if path == "direct" else block.reflected_range_m
            doppler = block.direct_doppler_hz if path == "direct" else block.reflected_doppler_hz
            true_carrier = float(np.median(doppler))
            true_delay = float(((np.median(ranges)-20000000.)/C_LIGHT*fs-start+1023) % 2046-1023)
            for cn0 in (35., 45., 55.):
                for repeat in range(4):
                    trial_key = (time_index*2+path_index)*4+repeat
                    # Paired standardized noise across C/N0, independent domains
                    # from calibration and held-out nulls. C refers to the unit
                    # direct path before aperture, NOT reflected-path C/N0.
                    noise = _rng(seed, SIGNAL_DOMAIN, trial_key).normal(size=4092)
                    noisy = block.iq+noise*np.sqrt(fs/(2*10**(cn0/10)))
                    result = detect_ca_search(noisy, gate)
                    candidate = result.candidate
                    if candidate is None:
                        raise RuntimeError("nonzero noisy signal returned no acquisition candidate")
                    delay_error = (candidate.delay.delay_samples-true_delay+1023) % 2046-1023
                    trials.append(dict(path=path, time_s=float(time_s), repeat=repeat, cn0_reference_db_hz=cn0,
                                       accepted=result.accepted, statistic=result.statistic,
                                       true_carrier_hz=true_carrier, carrier_hz=candidate.carrier_hz,
                                       carrier_error_hz=candidate.carrier_hz-true_carrier,
                                       delay_error_m=delay_error*C_LIGHT/fs))
    false_alarms = sum(row["accepted"] for row in heldout)
    summary = []
    for path in ("direct", "reflection"):
        for cn0 in (35., 45., 55.):
            rows = [row for row in trials if row["path"] == path and row["cn0_reference_db_hz"] == cn0]
            accepted = [row for row in rows if row["accepted"]]
            summary.append(dict(path=path, cn0_reference_db_hz=cn0, trials=len(rows),
                                accepted=len(accepted), rejected=len(rows)-len(accepted),
                                accepted_carrier_error_over_250_hz=sum(abs(row["carrier_error_hz"]) > 250 for row in accepted),
                                accepted_delay_error_over_30_m=sum(abs(row["delay_error_m"]) > 30 for row in accepted)))
    metadata = dict(seed=seed, threshold=asdict(gate), workers=workers,
                    heldout_false_alarms=false_alarms, heldout_trials=heldout_trials,
                    heldout_false_alarm_wilson95=wilson_interval(false_alarms, heldout_trials),
                    domains=dict(calibration=CALIBRATION_DOMAIN, heldout=HELDOUT_DOMAIN, signal=SIGNAL_DOMAIN),
                    paper="https://arxiv.org/abs/2605.01553v1", equation="Eq. (8), relative LOS Doppler",
                    trajectories=dict(receiver_m=[-20., 0., 2.], receiver_velocity_mps=[5., 0., 0.],
                                      receiver_acceleration_mps2=[200., 0., 0.],
                                      transmitter_m=[-20000020., 0., 2.], transmitter_velocity_mps=[1000., 0., 0.],
                                      transmitter_acceleration_mps2=[2., 0., 0.]),
                    assumptions="Illustrative prescribed trajectories; isolated direct/reflected paths; 1 ms aperture IQ; constant nav bit; nominal direct pre-aperture C/N0; paired noise across C/N0",
                    limitations="Grid frequency estimate; no carrier refinement, code-Doppler compensation, tracking, orbit/clock/atmosphere, RF validation or PVT. Acceptance does not prove range accuracy. Signal trials at different C/N0 share standardized noise.",
                    wall_bounds_m=[0., -100., 100., 0., 100.], antenna_lhcp_voltage_response=.1)
    root = Path(__file__).resolve().parents[1]
    sources = ["python/gnss_gpu/acquisition_cpu.py", "python/gnss_gpu/code_delay_cpu.py",
               "python/gnss_gpu/code_detection_cpu.py", "python/gnss_gpu/wall_signal_sim.py",
               "python/gnss_gpu/fresnel.py", "python/gnss_gpu/signal_sim_cpu.py",
               "experiments/run_cpu_carrier_search.py", "experiments/run_cpu_wall_detection.py",
               "docs/gnss_signal_simulation_sources.json"]
    metadata["source_sha256"] = {path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources}
    return dict(metadata=metadata, calibration=calibration, heldout_noise=heldout, trials=trials, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calibration-trials", type=int, default=199)
    parser.add_argument("--heldout-trials", type=int, default=400)
    parser.add_argument("--output-dir", default="output/cpu-carrier-search")
    args = parser.parse_args()
    result = run_experiment(calibration_trials=args.calibration_trials,
                            heldout_trials=args.heldout_trials, workers=args.workers)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("calibration", "heldout_noise", "trials", "summary"):
        with (output/f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[name][0]))
            writer.writeheader()
            writer.writerows(result[name])
    metadata = result["metadata"]
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].hist([row["statistic"] for row in result["calibration"]], bins=25,
                 density=True, alpha=.5, label="calibration (199 blocks by default)")
    axes[0].hist([row["statistic"] for row in result["heldout_noise"]], bins=25,
                 density=True, alpha=.5, label="independent held-out nulls")
    axes[0].axvline(metadata["threshold"]["calibration"]["score"], color="black", label="frozen threshold")
    axes[0].set(xlabel="maximum fitted score over 25 carrier bins and code delay", ylabel="density")
    axes[0].legend()
    for axis, path in zip(axes[1:], ("direct", "reflection")):
        rows = [row for row in result["trials"] if row["path"] == path and row["cn0_reference_db_hz"] == 55.]
        axis.plot([row["time_s"] for row in rows], [row["true_carrier_hz"] for row in rows],
                  color="black", label="geometric Doppler (evaluation only)")
        for accepted, marker in ((True, "o"), (False, "x")):
            selected = [row for row in rows if row["accepted"] == accepted]
            axis.scatter([row["time_s"] for row in selected], [row["carrier_hz"] for row in selected],
                         marker=marker, label="accepted" if accepted else "rejected candidate", alpha=.6)
        axis.set(xlabel="block start time [s]", ylabel="received carrier [Hz]",
                 title=f"{path}: nominal direct pre-aperture C/N0 = 55 dB-Hz")
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(output/"carrier_search.png", dpi=150)
    plt.close(figure)
    print(f"Held-out false alarms: {metadata['heldout_false_alarms']}/{metadata['heldout_trials']}")
    print(f"95% Wilson interval: {metadata['heldout_false_alarm_wilson95']}; target is marginal over calibration sets, not guaranteed for this frozen threshold")
    for row in result["summary"]:
        print(row)
    print(f"Saved complete candidates, rejections, frozen threshold and source hashes to {output}")


if __name__ == "__main__":
    main()
