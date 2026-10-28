"""Paired C/N0 Monte Carlo for CPU wall IQ and fractional code acquisition.

PYTHONPATH=python:. python experiments/run_cpu_wall_noise.py --workers 4
Uses spawn workers, retaining false peaks and a noise-only negative control.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
from pathlib import Path
import platform
from typing import Any, TypedDict

import numpy as np

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.input_validation import finite_float, positive_float, positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE, L1_HZ, _noise_seed
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal

CONDITIONS = ("direct", "multipath", "noise_only")


class ExperimentResult(TypedDict):
    metadata: dict[str, Any]
    trials: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    per_pose_summary: list[dict[str, Any]]


def noise_sigma(sampling_freq, cn0_dbhz):
    """Per-I/Q sigma: unit unfiltered direct power, E|n|²=fs/(C/N0).

    C/N0 is referenced to the direct carrier before sample-aperture chip-edge
    attenuation. Reflections do not change the receiver noise or reference.
    """
    fs = positive_float("sampling_freq", sampling_freq)
    cn0 = finite_float("cn0_dbhz", cn0_dbhz)
    try:
        sigma = math.sqrt(fs/2)*10**(-cn0/20)
    except OverflowError as exc:
        raise ValueError("C/N0 gives unsupported noise power") from exc
    if not math.isfinite(sigma) or sigma == 0:
        raise ValueError("C/N0 gives unsupported noise power")
    return sigma


def wrapped_error_m(estimate_m, expected_m):
    period_m = C_LIGHT*.001
    return float((estimate_m-expected_m+period_m/2) % period_m-period_m/2)


def _worker_init():
    # CI's scientific requirements include threadpoolctl. Limit BLAS inside
    # each spawned worker; the caller's parent/thread settings are preserved.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _estimate_trial(job):
    pose, trial, root_seed, cn0_values, fs, clean, expected, threshold = job
    # Reused across C/N0 levels and conditions, independent across pose/trial.
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, pose, trial]))
    standard = rng.normal(size=clean[0].shape)
    rows = []
    for cn0 in cn0_values:
        sigma = noise_sigma(fs, cn0)
        noise = standard*sigma
        for condition in CONDITIONS:
            iq = noise if condition == "noise_only" else clean[CONDITIONS.index(condition)]+noise
            estimate = estimate_ca_delay(iq, fs)
            delay = None if estimate is None else estimate.delay_m
            error = None if delay is None or condition == "noise_only" else wrapped_error_m(delay, expected)
            rows.append(dict(pose=pose, trial=trial, root_seed=root_seed, cn0_dbhz=cn0,
                             condition=condition, noise_sigma=sigma,
                             expected_delay_m=None if condition == "noise_only" else expected,
                             estimated_delay_m=delay, error_m=error,
                             fit_residual_fraction=(None if estimate is None else estimate.fit_residual_fraction),
                             status="no_estimate" if estimate is None else "peak_returned",
                             bad_or_missing=(None if condition == "noise_only" else
                                             error is None or abs(error) > threshold)))
    return rows


def summarize_trials(rows, threshold_m):
    """All trials remain in denominators; error stats include wrong code peaks.

    Missing estimates have separate counts and count as bad/missing. Noise-only
    controls have no ranging error or bad/missing classification.
    """
    summary = []
    groups = sorted({(row["cn0_dbhz"], row["condition"]) for row in rows})
    for cn0, condition in groups:
        group = [row for row in rows if row["cn0_dbhz"] == cn0 and row["condition"] == condition]
        errors = np.array([row["error_m"] for row in group if row["error_m"] is not None])
        count = len(group)
        peaks = sum(row["status"] == "peak_returned" for row in group)
        stats = dict(cn0_dbhz=cn0, condition=condition, trials=count, peaks_returned=peaks,
                     no_estimate=count-peaks, peak_return_fraction=peaks/count,
                     error_count=len(errors), outlier_threshold_m=threshold_m,
                     signed_bias_m=None, rmse_m=None, median_abs_error_m=None,
                     p95_abs_error_m=None, max_abs_error_m=None, bad_or_missing_fraction=None)
        if condition != "noise_only":
            stats["bad_or_missing_fraction"] = sum(
                row["error_m"] is None or abs(row["error_m"]) > threshold_m for row in group)/count
        if len(errors):
            stats.update(signed_bias_m=float(np.mean(errors)), rmse_m=float(np.sqrt(np.mean(errors**2))),
                         median_abs_error_m=float(np.median(np.abs(errors))),
                         p95_abs_error_m=float(np.quantile(np.abs(errors), .95, method="higher")),
                         max_abs_error_m=float(np.max(np.abs(errors))))
        summary.append(stats)
    return summary


def run_experiment(*, cn0_values=(25., 35., 45., 55.), trials=32, root_seed=20261024,
                   workers=1, samples_per_chip=20, threshold_m=30.) -> ExperimentResult:
    trials, workers = positive_int("trials", trials), positive_int("workers", workers)
    spc = positive_int("samples_per_chip", samples_per_chip)
    if spc < 2:
        raise ValueError("samples_per_chip must be >=2")
    threshold = positive_float("threshold_m", threshold_m)
    if threshold >= C_LIGHT*.0005:
        raise ValueError("threshold_m must be below half a code period")
    seed = _noise_seed(root_seed)
    if seed is None:
        raise ValueError("root_seed must be an explicit uint64 integer")
    cn0s = tuple(finite_float("cn0_dbhz", value) for value in cn0_values)
    if not cn0s or len(set(cn0s)) != len(cn0s):
        raise ValueError("cn0_values must be nonempty and distinct")
    fs, count = spc*CHIP_RATE, spc*1023
    for cn0 in cn0s:
        noise_sigma(fs, cn0)
    simulator = CpuSignalSimulator(fs, noise_floor_db=-10000)
    wall = VerticalWall(0., -30., 5., 0., 100., material="concrete", polarization="rhcp")
    transmitter = np.array([-1e7, 1e7, 1e7])
    poses = [np.array([-20.+offset, -20., 1.5]) for offset in (0., .05, .1)]
    jobs, scenes = [], []
    for index, rx in enumerate(poses):
        # The clock alignment deliberately gives a non-integer sample delay;
        # acquisition receives only IQ, fs and PRN, never this true value.
        reference = float(np.linalg.norm(transmitter-rx)-4.5)
        direct = simulate_wall_signal(simulator, wall, rx, [0., 0., 0.], transmitter,
                                      count, include_reflection=False,
                                      reference_range_m=reference, integrate_samples=True)
        full = simulate_wall_signal(simulator, wall, rx, [0., 0., 0.], transmitter, count,
                                    reference_range_m=reference, integrate_samples=True)
        expected = float(direct.direct_range_m[0]-reference)
        scenes.append(dict(pose=index, receiver_m=rx.tolist(), reference_range_m=reference,
                           expected_delay_m=expected,
                           excess_range_m=float(full.reflected_range_m[0]-full.direct_range_m[0]),
                           reflection_magnitude=float(abs(full.reflection_coefficient[0])),
                           fresnel_phase_rad=float(np.angle(full.reflection_coefficient[0])),
                           relative_carrier_phase_rad=float(np.angle(
                               full.reflection_coefficient[0]*np.exp(-2j*np.pi*L1_HZ*
                               (full.reflected_range_m[0]-full.direct_range_m[0])/C_LIGHT)))))
        for trial in range(trials):
            jobs.append((index, trial, seed, cn0s, fs, (direct.iq, full.iq), expected, threshold))
    if workers == 1:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            rows = [row for job in jobs for row in _estimate_trial(job)]
    else:
        # Spawn avoids forking the Numba/OpenMP pool used to synthesize scenes.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init) as pool:
            rows = [row for result in pool.map(_estimate_trial, jobs) for row in result]
    root = Path(__file__).resolve().parents[1]
    source_files = ["experiments/run_cpu_wall_noise.py", "python/gnss_gpu/code_delay_cpu.py",
                    "python/gnss_gpu/wall_signal_sim.py", "python/gnss_gpu/signal_sim_cpu.py",
                    "python/gnss_gpu/fresnel.py", "python/gnss_gpu/validation/tracking_residuals.py",
                    "python/gnss_gpu/input_validation.py"]
    metadata = dict(schema_version=1, sampling_freq_hz=fs, observation_s=.001,
                    samples_per_chip=spc, cn0_dbhz=cn0s, trials_per_pose=trials,
                    root_seed=seed, workers=workers, outlier_threshold_m=threshold,
                    poses=scenes, transmitter_m=transmitter.tolist(),
                    wall=dict(x_m=wall.x_m, y_min_m=wall.y_min_m, y_max_m=wall.y_max_m,
                              z_min_m=wall.z_min_m, z_max_m=wall.z_max_m,
                              material=wall.material, polarization=wall.polarization),
                    noise="Independent white I/Q Gaussian; sigma²=fs/(2*10**(C/N0/10)); pre-aperture unit direct reference",
                    pairing="Same standard normal array across C/N0 and conditions; independent pose/trial SeedSequence keys",
                    estimator="Single-path aperture-template fit, no detection threshold or truth delay input",
                    limitations="Static poses, constant nav bit, ideal wall/aperture; no dynamics, RF bandpass, diffraction or navigation decoding",
                    python=platform.python_version(), numpy=np.__version__, numba=version("numba"),
                    machine=platform.machine(), blas_threads_per_task=1,
                    error_statistics="All returned wrong peaks included; higher-order P95; missing estimates counted separately and as bad/missing",
                    source_sha256={path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in source_files})
    per_pose = [dict(pose=pose, **stats) for pose in range(len(poses))
                for stats in summarize_trials([row for row in rows if row["pose"] == pose], threshold)]
    return {"metadata": metadata, "trials": rows, "summary": summarize_trials(rows, threshold),
            "per_pose_summary": per_pose}


def save_results(result: ExperimentResult, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("trials", "summary", "per_pose_summary"):
        rows = result[name]
        with (output/f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output/"metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    trials_per_group = result["metadata"]["trials_per_pose"]*len(result["metadata"]["poses"])
    figure.suptitle(f"Paired wall/noise experiment: {trials_per_group} trials per condition and C/N0")
    for condition in ("direct", "multipath"):
        rows = [row for row in result["summary"] if row["condition"] == condition]
        cn0 = [row["cn0_dbhz"] for row in rows]
        axes[0].semilogy(cn0, [row["median_abs_error_m"] for row in rows], "o-", label=f"{condition}: median")
        axes[0].semilogy(cn0, [row["p95_abs_error_m"] for row in rows], "o--", label=f"{condition}: P95")
        axes[1].plot(cn0, [row["bad_or_missing_fraction"] for row in rows], "o-", label=condition)
    axes[0].set(ylabel="absolute code-delay error [m]", title="1 ms acquisition; all returned peaks retained")
    axes[0].legend()
    threshold = result["metadata"]["outlier_threshold_m"]
    axes[1].set(ylabel="fraction of all trials", ylim=(-.03, 1.03),
                title=f"|error| > {threshold:g} m or missing estimate")
    axes[1].legend()
    controls = [row for row in result["summary"] if row["condition"] == "noise_only"]
    axes[2].plot([row["cn0_dbhz"] for row in controls], [row["peak_return_fraction"] for row in controls], "o-")
    axes[2].set(ylabel="peak return fraction", ylim=(-.03, 1.03),
                title="Noise-only control: returning a peak is not detection")
    for axis in axes:
        axis.set_xlabel("direct-reference C/N0 [dB-Hz] (nominal noise scaling for noise-only)")
        axis.grid(alpha=.2)
    figure.savefig(output/"noise_comparison.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=32, help="trials per receiver pose")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--root-seed", type=int, default=20261024)
    parser.add_argument("--cn0", type=float, nargs="+", default=[25., 35., 45., 55.])
    parser.add_argument("--threshold-m", type=float, default=30.)
    parser.add_argument("--output-dir", default="output/cpu-wall-noise")
    args = parser.parse_args()
    result = run_experiment(cn0_values=args.cn0, trials=args.trials, workers=args.workers,
                            root_seed=args.root_seed, threshold_m=args.threshold_m)
    save_results(result, args.output_dir)
    for row in result["summary"]:
        if row["condition"] != "noise_only":
            print(f"{row['cn0_dbhz']:g} dB-Hz {row['condition']:9s}: median {row['median_abs_error_m']:.2f} m, "
                  f"P95 {row['p95_abs_error_m']:.2f} m, bad/missing {row['bad_or_missing_fraction']:.1%}")
    print(f"Saved {len(result['trials'])} trial rows to {args.output_dir}; noise-only peaks are not detections.")


if __name__ == "__main__":
    main()
