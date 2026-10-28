"""Paired continuous CPU tracking beside a finite reflecting wall.

PYTHONPATH=python:. python experiments/run_cpu_multipath_tracking.py --workers 4
Truth, reflection visibility and path-isolation switches are evaluation-only.
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
from gnss_gpu.input_validation import positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.tracking_cpu import CpuCaTracker
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal

FS, COUNT, BLOCKS = 2*CHIP_RATE, 2046, 400
CALIBRATION_SEED, CALIBRATION_DOMAIN = 20261030, 0x4D504341
EVALUATION_SEEDS, SIGNAL_DOMAIN = (20261028, 20261029), 0x4D505349
RECEIVER = np.array([-20., -20., 1.5])
VELOCITY = np.array([1., 20., 0.])
TRANSMITTER = np.array([-1e7, 1e7, 1e7])
REFERENCE_RANGE = float(np.linalg.norm(TRANSMITTER-RECEIVER))
MODES = ("direct", "ideal_rhcp", "lhcp_leakage_90deg")


def _worker_init():
    from numba import set_num_threads
    from threadpoolctl import threadpool_limits
    set_num_threads(1)
    threadpool_limits(limits=1)


def wall_observations(start_sample):
    """Identical time/geometry/nav data, with diagnostic propagation variants."""
    simulator = CpuSignalSimulator(FS, noise_floor_db=-10000)
    nav_bit = -1 if (start_sample//(20*COUNT)) % 2 else 1
    result = {}
    for mode in MODES:
        response = .1j if mode == "lhcp_leakage_90deg" else 0j
        wall = VerticalWall(0., -30., 5., 0., 100., antenna_lhcp_response=response)
        result[mode] = simulate_wall_signal(
            simulator, wall, RECEIVER, VELOCITY, TRANSMITTER, COUNT,
            start_sample=start_sample, reference_range_m=REFERENCE_RANGE,
            nav_bit=nav_bit, include_reflection=mode != "direct", integrate_samples=True)
    return result


def paired_delay_bias(direct, composite):
    """Wrapped composite-minus-direct state delay only when BOTH use a measurement.

    None explicitly preserves rejection/coasting/confirmation in the denominator
    without pretending these blocks are paired measured-bias observations.
    """
    if not direct.measurement_used or not composite.measurement_used:
        return None
    if direct.delay_samples is None or composite.delay_samples is None:
        return None
    return float(((composite.delay_samples-direct.delay_samples+COUNT/2) % COUNT-COUNT/2)*C_LIGHT/FS)


def summarize_rows(rows):
    used = [row for row in rows if row["measurement_used"]]
    paired = [row for row in rows if row["paired_incremental_bias_m"] is not None]
    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else None

    return dict(blocks=len(rows), measurement_used_blocks=len(used),
                unused_blocks=len(rows)-len(used),
                states={status:sum(row["status"] == status for row in rows)
                        for status in ("searching", "confirming", "acquired", "tracking", "coasting", "lost")},
                delay_rmse_m_conditional_used=rmse([row["delay_error_m"] for row in used]),
                used_delay_error_over_30_m=sum(abs(row["delay_error_m"]) > 30 for row in used),
                paired_measurement_blocks=len(paired),
                paired_bias_rmse_m_conditional=rmse([row["paired_incremental_bias_m"] for row in paired]),
                paired_bias_max_abs_m_conditional=max((abs(row["paired_incremental_bias_m"]) for row in paired), default=None))


def _case(job):
    seed, cn0, threshold = job
    trackers = {mode:CpuCaTracker(threshold) for mode in MODES}
    rows = []
    for index in range(BLOCKS):
        start = index*COUNT
        signals = wall_observations(start)
        if cn0 is None:
            noise = np.zeros(2*COUNT)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, SIGNAL_DOMAIN, index]))
            noise = rng.normal(0., np.sqrt(FS/(2*10**(cn0/10))), 2*COUNT)
        # Identical additive noise across modes, paired across C/N0 per seed.
        states = {mode:trackers[mode].update(signals[mode].iq+noise, start_sample=start) for mode in MODES}
        direct = signals["direct"]
        true_delay = float(((np.median(direct.direct_range_m)-REFERENCE_RANGE)/C_LIGHT*FS-start+COUNT/2) % COUNT-COUNT/2)
        for mode in MODES:
            state, block = states[mode], signals[mode]
            error = None if state.delay_samples is None else float(
                ((state.delay_samples-true_delay+COUNT/2) % COUNT-COUNT/2)*C_LIGHT/FS)
            refined = state.refined_candidate
            rows.append(dict(seed=seed, cn0_reference_db_hz=cn0, mode=mode, block=index,
                             start_sample=start, time_s=start/FS,
                             status=state.status, measurement_used=state.measurement_used,
                             detection_accepted=state.detection_accepted,
                             detection_statistic=state.detection_statistic,
                             confirmation_count=state.confirmation_count, missed_blocks=state.missed_blocks,
                             carrier_hz=state.carrier_hz, delay_samples=state.delay_samples,
                             coarse_carrier_hz=state.coarse_candidate.carrier_hz if state.coarse_candidate else None,
                             refined_carrier_hz=refined.carrier_hz if refined else None,
                             refined_delay_samples=refined.delay.delay_samples if refined else None,
                             fit_residual_fraction=refined.delay.fit_residual_fraction if refined else None,
                             geometric_direct_delay_samples=true_delay, delay_error_m=error,
                             direct_geometric_doppler_hz=float(np.median(block.direct_doppler_hz)),
                             reflected_geometric_doppler_hz=float(np.median(block.reflected_doppler_hz)),
                             direct_visible_fraction=float(np.mean(block.direct_visible)),
                             reflection_visible_fraction=float(np.mean(block.reflection_visible)),
                             reflected_voltage_abs=float(abs(block.reflection_coefficient[COUNT//2])),
                             reflected_excess_m=float(np.median(block.reflected_range_m-block.direct_range_m)),
                             paired_incremental_bias_m=paired_delay_bias(states["direct"], state)))
    summaries = [dict(seed=seed, cn0_reference_db_hz=cn0, mode=mode,
                      **summarize_rows([row for row in rows if row["mode"] == mode])) for mode in MODES]
    return dict(rows=rows, summary=summaries)


def run_experiment(*, workers=1) -> dict[str, Any]:
    workers = positive_int("workers", workers)
    # Fixed near-zero-IF prior suitable for this slow, stationary-transmitter
    # experiment; no exact geometric Doppler is passed to the receiver.
    search = CaSearch(FS, (-1000., -500., 0., 500., 1000.))
    jobs = [(CALIBRATION_SEED, CALIBRATION_DOMAIN, index, search) for index in range(199)]
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_worker_init)
    try:
        calibration = list(pool.map(_null_job, jobs, chunksize=4)) if pool else list(map(_null_job, jobs))
        threshold = calibrate_ca_search([row["statistic"] for row in calibration], search)
        case_jobs = [(0, None, threshold)]+[(seed, cn0, threshold)
                                          for seed in EVALUATION_SEEDS for cn0 in (55., 65.)]
        cases = list(pool.map(_case, case_jobs)) if pool else list(map(_case, case_jobs))
    finally:
        if pool:
            pool.shutdown()
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_cpu_multipath_tracking.py", "experiments/run_cpu_carrier_search.py",
               "python/gnss_gpu/tracking_cpu.py", "python/gnss_gpu/acquisition_cpu.py",
               "python/gnss_gpu/code_delay_cpu.py", "python/gnss_gpu/code_detection_cpu.py",
               "python/gnss_gpu/wall_signal_sim.py", "python/gnss_gpu/signal_sim_cpu.py",
               "python/gnss_gpu/fresnel.py", "docs/gnss_signal_simulation_sources.json"]
    metadata = dict(calibration_seed=CALIBRATION_SEED, calibration_domain=CALIBRATION_DOMAIN,
                    evaluation_seeds=list(EVALUATION_SEEDS), signal_domain=SIGNAL_DOMAIN,
                    threshold=asdict(threshold), workers=workers, sampling_freq_hz=FS,
                    block_samples=COUNT, blocks_per_track=BLOCKS, modes=list(MODES),
                    cn0_reference_db_hz=[None, 55., 65.], noiseless_seed_label=0,
                    geometry=dict(receiver_m=RECEIVER.tolist(), receiver_velocity_mps=VELOCITY.tolist(),
                                  transmitter_m=TRANSMITTER.tolist(), reference_range_m=REFERENCE_RANGE,
                                  wall_bounds_m=[0., -30., 5., 0., 100.], material="concrete"),
                    tracker=dict(confirmation_blocks=2, gain=.25, coast_blocks=3,
                                 max_frequency_step_hz=500., max_delay_step_samples=1.),
                    antenna=dict(ideal_rhcp_lhcp_voltage=[0., 0.], leakage_lhcp_voltage=[0., .1],
                                 interpretation="Illustrative -20 dB LHCP receive power response at +90 degrees; no measured antenna calibration"),
                    papers=["10.1007/s10291-013-0331-y", "10.1007/s10291-014-0370-z", "https://arxiv.org/abs/2605.01553v1"],
                    reviewed_scope="2014 author supplementary polarization code, not paywalled full text; 2026 preprint Eq. (8) geometric Doppler. Receiver following is our engineering implementation.",
                    assumptions="Finite opaque specular wall, stationary transmitter, constant receiver velocity, centered sample apertures, block-aligned alternating 50 bit/s nav data; independent white complex Gaussian noise",
                    pairing="Identical noise across path/antenna modes and standardized noise reused across C/N0 within each seed. Noiseless is one deterministic control, not an independent random trial.",
                    limitations="Two seeds and one short idealized geometry, no orbit/clocks/atmosphere/diffraction/diffuse scattering or RF/PVT validation; hard wall-edge mask at aperture centers; single-path receiver in multipath, no mitigation or ranging-integrity guarantee. Composite carrier has no single geometric truth. Visibility and truth are evaluation-only.",
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    return dict(rows=[row for case in cases for row in case["rows"]],
                summary=[row for case in cases for row in case["summary"]], calibration=calibration, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="output/cpu-multipath-tracking")
    args = parser.parse_args()
    result = run_experiment(workers=args.workers)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("rows", "calibration"):
        with (output/f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    for key in ("summary", "metadata"):
        (output/f"{key}.json").write_text(json.dumps(result[key], indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    for mode in MODES[1:]:
        clean = [row for row in result["rows"] if row["mode"] == mode and row["cn0_reference_db_hz"] is None]
        noisy = [row for row in result["rows"] if row["mode"] == mode and row["cn0_reference_db_hz"] == 65. and row["seed"] == EVALUATION_SEEDS[0]]
        axes[0].plot([row["time_s"] for row in clean], [row["reflected_voltage_abs"] for row in clean], label=mode)
        axes[1].plot([row["time_s"] for row in clean], [row["paired_incremental_bias_m"] for row in clean], label=mode)
        axes[2].plot([row["time_s"] for row in noisy], [row["paired_incremental_bias_m"] for row in noisy], label=mode)
    axes[0].set(ylabel="reflected voltage magnitude", title="Finite wall: reflection exits the wall during continuous tracking")
    axes[1].set(ylabel="paired state-delay bias [m]", title="Noiseless: composite minus direct-only, both measurements used")
    axes[2].set(ylabel="paired state-delay bias [m]", title="65 dB-Hz nominal direct C/N0, seed 20261028; same additive noise across modes")
    for axis in axes:
        axis.set_xlabel("block start time [s]")
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(output/"multipath_tracking.png", dpi=150)
    plt.close(figure)
    for row in result["summary"]:
        print(row)
    print(f"Saved full per-block states, paired biases, calibration, parameters and plot to {output}")


if __name__ == "__main__":
    main()
