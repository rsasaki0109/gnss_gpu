"""Follow paper-linked accelerating IQ, with an explicit signal outage.

PYTHONPATH=python:. python experiments/run_cpu_tracking.py --workers 4
No true parameters are passed to the receiver; truth is evaluation-only.
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

from experiments.run_cpu_carrier_search import _null_job
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, calibrate_ca_search
from gnss_gpu.input_validation import positive_int, positive_float
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _sample_index
from gnss_gpu.tracking_cpu import CpuCaTracker
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal

SEED = 20261025
CALIBRATION_DOMAIN = 0x54524343
SIGNAL_DOMAIN = 0x54524353


def run_experiment(*, workers=1, seed=SEED, cn0_db_hz=55., confirmation_blocks=1,
                   calibration=None) -> dict[str, Any]:
    workers = positive_int("workers", workers)
    seed = _sample_index("seed", seed)
    cn0_db_hz = positive_float("cn0_db_hz", cn0_db_hz)
    fs, count = 2*CHIP_RATE, 2046
    search = CaSearch(fs, tuple(np.arange(-6000., 6001., 500.)))
    jobs = [(SEED, CALIBRATION_DOMAIN, index, search) for index in range(199)]
    if calibration is None:
        if workers == 1:
            calibration = list(map(_null_job, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                calibration = list(pool.map(_null_job, jobs, chunksize=4))
    threshold = calibrate_ca_search([row["statistic"] for row in calibration], search)
    tracker = CpuCaTracker(threshold, gain=.25, coast_blocks=3,
                           max_frequency_step_hz=500., max_delay_step_samples=1., confirmation_blocks=confirmation_blocks)
    simulator = CpuSignalSimulator(fs, noise_floor_db=-10000)
    wall = VerticalWall(0., -100., 100., 0., 100.)
    rows = []
    for index in range(200):
        start = index*count
        block = simulate_wall_signal(
            simulator, wall, [-20., 0., 2.], [5., 0., 0.], [-20000020., 0., 2.], count,
            start_sample=start, reference_range_m=20000000.,
            transmitter_velocity_mps=[1000., 0., 0.], rx_acceleration_mps2=[200., 0., 0.],
            transmitter_acceleration_mps2=[2., 0., 0.], include_reflection=False,
            integrate_samples=True)
        outage = 80 <= index < 100
        rng = np.random.default_rng(np.random.SeedSequence([seed, SIGNAL_DOMAIN, index]))
        noise = rng.normal(0., np.sqrt(fs/(2*10**(cn0_db_hz/10))), 2*count)
        # 50 bit/s illustrative alternating data bits, aligned to block edges.
        iq = noise if outage else block.iq*(-1 if (index//20) % 2 else 1)+noise
        result = tracker.update(iq, start_sample=start)
        truth_carrier = float(np.median(block.direct_doppler_hz))
        truth_delay = float(((np.median(block.direct_range_m)-20000000.)/C_LIGHT*fs-start+count/2) % count-count/2)
        refined = result.refined_candidate
        coarse = result.coarse_candidate
        delay_error = None if result.delay_samples is None else (
            (result.delay_samples-truth_delay+count/2) % count-count/2)*C_LIGHT/fs
        rows.append(dict(block=index, start_sample=start, time_s=start/fs, signal_outage=outage,
                         status=result.status, detection_accepted=result.detection_accepted,
                         measurement_used=result.measurement_used, missed_blocks=result.missed_blocks,
                         confirmation_count=result.confirmation_count,
                         detection_statistic=result.detection_statistic,
                         coarse_carrier_hz=coarse.carrier_hz if coarse else None,
                         refined_carrier_hz=refined.carrier_hz if refined else None,
                         tracked_carrier_hz=result.carrier_hz,
                         tracked_delay_samples=result.delay_samples,
                         true_carrier_hz=truth_carrier, true_delay_samples=truth_delay,
                         coarse_error_hz=coarse.carrier_hz-truth_carrier if coarse else None,
                         refined_error_hz=refined.carrier_hz-truth_carrier if refined else None,
                         tracked_error_hz=result.carrier_hz-truth_carrier if result.carrier_hz is not None else None,
                         tracked_delay_error_m=delay_error))
    used = [row for row in rows if row["measurement_used"] and not row["signal_outage"]]
    def rmse(column):
        return float(np.sqrt(np.mean([row[column]**2 for row in used]))) if used else None

    summary = dict(signal_present_blocks=180, signal_absent_blocks=20,
                   used_signal_present_blocks=len(used),
                   used_signal_absent_blocks=sum(row["measurement_used"] for row in rows if row["signal_outage"]),
                   coarse_rmse_hz_conditional_used=rmse("coarse_error_hz"),
                   refined_rmse_hz_conditional_used=rmse("refined_error_hz"),
                   tracked_rmse_hz_conditional_used=rmse("tracked_error_hz"),
                   tracked_delay_rmse_m_conditional_used=rmse("tracked_delay_error_m"),
                   states={state:sum(row["status"] == state for row in rows)
                           for state in ("acquired", "tracking", "coasting", "lost", "searching", "confirming")})
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_cpu_tracking.py", "experiments/run_cpu_carrier_search.py",
               "python/gnss_gpu/tracking_cpu.py", "python/gnss_gpu/acquisition_cpu.py",
               "python/gnss_gpu/code_delay_cpu.py", "python/gnss_gpu/code_detection_cpu.py",
               "python/gnss_gpu/signal_sim_cpu.py", "python/gnss_gpu/wall_signal_sim.py",
               "python/gnss_gpu/fresnel.py", "docs/gnss_signal_simulation_sources.json"]
    metadata = dict(seed=seed, calibration_seed=SEED, calibration_domain=CALIBRATION_DOMAIN, signal_domain=SIGNAL_DOMAIN,
                    threshold=asdict(threshold), cn0_reference_db_hz=cn0_db_hz, sampling_freq_hz=fs,
                    blocks=200, outage_blocks=[80, 100], outage_end_exclusive=True,
                    receiver_m=[-20., 0., 2.], receiver_velocity_mps=[5., 0., 0.],
                    receiver_acceleration_mps2=[200., 0., 0.], transmitter_m=[-20000020., 0., 2.],
                    transmitter_velocity_mps=[1000., 0., 0.], transmitter_acceleration_mps2=[2., 0., 0.],
                    reference_range_m=20000000., wall_bounds_m=[0., -100., 100., 0., 100.],
                    include_reflection=False, integrate_samples=True,
                    tracker=dict(gain=.25, coast_blocks=3, max_frequency_step_hz=500., max_delay_step_samples=1., confirmation_blocks=confirmation_blocks),
                    refinement=dict(tolerance_hz=.1, max_iterations=32),
                    paper="https://arxiv.org/abs/2605.01553v1", equation="Eq. (8), relative LOS Doppler",
                    assumptions="Illustrative constant-acceleration trajectories, direct-only IQ, nominal pre-aperture C/N0, block-aligned 50 bit/s alternating nav bits; stationary independent complex Gaussian noise",
                    limitations="Single seeded trajectory; conditional errors on used signal-present blocks; full grid gate per block, not a track false-alarm bound; no phase loop/bit synchronization/RF/PVT validation. Outage boundaries and truth are hidden from tracker.",
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    return dict(rows=rows, calibration=calibration, metadata=metadata, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confirmation-blocks", type=int, default=1,
                        help="1 preserves the original demo; the tracker API defaults to 2")
    parser.add_argument("--output-dir", default="output/cpu-tracking")
    args = parser.parse_args()
    result = run_experiment(workers=args.workers, confirmation_blocks=args.confirmation_blocks)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("rows", "calibration"):
        with (output/f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    for key in ("metadata", "summary"):
        (output/f"{key}.json").write_text(json.dumps(result[key], indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = result["rows"]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    t = [row["time_s"] for row in rows]
    axes[0].axhline(0., color="black", label="geometric truth (evaluation only)")
    for key, label in (("coarse_error_hz", "coarse (gate accepted)"),
                       ("refined_error_hz", "refined (measurement used)"),
                       ("tracked_error_hz", "smoothed state, including coasting")):
        values = [row[key] if (row["detection_accepted"] if key == "coarse_error_hz" else
                               row["measurement_used"] if key == "refined_error_hz" else True) else None for row in rows]
        axes[0].plot(t, values, ".-", markersize=2, label=label)
    axes[0].set_yscale("symlog", linthresh=50.)
    axes[0].set(ylabel="carrier error [Hz], symlog", title="CPU blockwise following: accelerating direct path, 55 dB-Hz")
    axes[0].legend()
    axes[1].plot(t, [row["tracked_delay_error_m"] for row in rows], label="state delay error, including predictions")
    axes[1].set_yscale("symlog", linthresh=10.)
    axes[1].set(ylabel="delay error [m], symlog")
    axes[1].legend()
    states = {state:index for index, state in enumerate(("searching", "confirming", "acquired", "tracking", "coasting", "lost"))}
    axes[2].step(t, [states[row["status"]] for row in rows], where="post")
    axes[2].set(yticks=list(states.values()), yticklabels=list(states), ylabel="receiver state")
    for axis in axes:
        axis.axvspan(.08, .1, alpha=.15, color="red", label="signal absent; noise remains")
        axis.grid(alpha=.2)
        axis.set_xlabel("block start time [s]")
    figure.savefig(output/"tracking.png", dpi=150)
    plt.close(figure)
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved all blocks, calibration, source hashes and plot to {output}")


if __name__ == "__main__":
    main()
