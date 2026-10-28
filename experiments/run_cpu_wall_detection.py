"""Freeze a noise-calibrated acquisition gate, then evaluate held-out trials.

PYTHONPATH=python:. python experiments/run_cpu_wall_detection.py --workers 4
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
from statistics import NormalDist
from typing import Any, TypedDict

import numpy as np

from experiments.run_cpu_wall_noise import _worker_init, run_experiment
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.code_detection_cpu import ca_detection_statistic, calibrate_ca_threshold
from gnss_gpu.input_validation import positive_int, positive_float
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _noise_seed

CALIBRATION_DOMAIN = 0x43414C
EVALUATION_DOMAIN = 0x4556414C


class DetectionExperimentResult(TypedDict):
    metadata: dict[str, Any]
    calibration: list[dict[str, Any]]
    heldout_noise: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    summary: list[dict[str, Any]]


def wilson_interval(successes, trials):
    """95% Wilson binomial interval for measured counts (not a guarantee)."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("valid binomial counts are required")
    z = NormalDist().inv_cdf(.975)
    p = successes/trials
    center = (p+z*z/(2*trials))/(1+z*z/trials)
    half = z*math.sqrt(p*(1-p)/trials+z*z/(4*trials**2))/(1+z*z/trials)
    return max(0., center-half), min(1., center+half)


def _null_score(job) -> dict[str, Any]:
    root_seed, domain, trial, fs = job
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, domain, trial]))
    iq = rng.normal(size=2*round(fs*.001))
    score = ca_detection_statistic(estimate_ca_delay(iq, fs))
    if score <= 0:
        raise RuntimeError("nonzero Gaussian calibration/evaluation block returned no peak")
    return dict(domain=domain, trial=trial, statistic=score)


def _null_trials(seed, domain, trials, fs, workers):
    jobs = [(seed, domain, trial, fs) for trial in range(trials)]
    if workers == 1:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return [_null_score(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_worker_init) as pool:
        return list(pool.map(_null_score, jobs))


def run_detection_experiment(*, calibration_trials=1000, evaluation_trials=1000,
                             false_alarm_probability=.01, signal_trials=32,
                             root_seed=20261024, workers=1, samples_per_chip=20) -> DetectionExperimentResult:
    calibration_trials = positive_int("calibration_trials", calibration_trials)
    evaluation_trials = positive_int("evaluation_trials", evaluation_trials)
    signal_trials = positive_int("signal_trials", signal_trials)
    workers = positive_int("workers", workers)
    spc = positive_int("samples_per_chip", samples_per_chip)
    if spc < 2:
        raise ValueError("samples_per_chip must be >=2")
    alpha = positive_float("false_alarm_probability", false_alarm_probability)
    if alpha >= 1 or math.ceil((calibration_trials+1)*(1-alpha)) > calibration_trials:
        raise ValueError("calibration count cannot support this false-alarm target")
    seed = _noise_seed(root_seed)
    if seed is None:
        raise ValueError("root_seed must be an explicit uint64 integer")
    fs = spc*CHIP_RATE
    # Domain keys cannot overlap the signal experiment's pose indices 0,1,2.
    calibration = _null_trials(seed, CALIBRATION_DOMAIN, calibration_trials, fs, workers)
    threshold = calibrate_ca_threshold((row["statistic"] for row in calibration), fs,
                                       false_alarm_probability=alpha)
    # This threshold is frozen before independent null/signal evaluation starts.
    evaluation = _null_trials(seed, EVALUATION_DOMAIN, evaluation_trials, fs, workers)
    for row in evaluation:
        row["accepted"] = row["statistic"] > threshold.score
    false_alarms = sum(row["accepted"] for row in evaluation)
    lower, upper = wilson_interval(false_alarms, evaluation_trials)
    signals = run_experiment(trials=signal_trials, root_seed=seed, workers=workers,
                              samples_per_chip=spc)
    for row in signals["trials"]:
        residual = row["fit_residual_fraction"]
        row["detection_statistic"] = 0. if residual is None else fs*.001*(1-residual)
        row["accepted"] = row["status"] == "peak_returned" and row["detection_statistic"] > threshold.score
    summary = []
    for baseline in signals["summary"]:
        group = [row for row in signals["trials"] if row["cn0_dbhz"] == baseline["cn0_dbhz"]
                 and row["condition"] == baseline["condition"]]
        accepted = [row for row in group if row["accepted"]]
        good = sum(row["error_m"] is not None and not row["bad_or_missing"] for row in accepted)
        wrong = sum(row["error_m"] is not None and row["bad_or_missing"] for row in accepted)
        errors = [abs(row["error_m"]) for row in accepted if row["error_m"] is not None]
        count = len(group)
        summary.append(dict(cn0_dbhz=baseline["cn0_dbhz"], condition=baseline["condition"],
                            trials=count, accepted=len(accepted), rejected=count-len(accepted),
                            acceptance_fraction=len(accepted)/count,
                            accepted_good_fraction_all=None if baseline["condition"] == "noise_only" else good/count,
                            accepted_wrong_fraction_all=None if baseline["condition"] == "noise_only" else wrong/count,
                            conditional_error_count=len(errors),
                            conditional_median_abs_error_m=float(np.median(errors)) if errors else None,
                            ungated_median_abs_error_m=baseline["median_abs_error_m"]))
    metadata = dict(schema_version=1, threshold=asdict(threshold),
                    marginal_rank_bound=threshold.marginal_false_alarm_bound,
                    root_seed=seed, calibration_domain=CALIBRATION_DOMAIN,
                    evaluation_domain=EVALUATION_DOMAIN,
                    heldout_noise_trials=evaluation_trials, heldout_false_alarms=false_alarms,
                    measured_false_alarm_fraction=false_alarms/evaluation_trials,
                    false_alarm_wilson_95=[lower, upper],
                    calibration_assumptions="Independent exchangeable white Gaussian null scores; one PRN, one carrier setting and full code-phase search",
                    split="Calibration frozen first; held-out null domain differs from calibration and signal pose keys; signal noise reused across C/N0, not independent replicated null evidence",
                    caveat="Acceptance is signal-presence evidence under calibrated null, not validation of ranging or a guarantee for this threshold/real RF noise",
                    signal_experiment=signals["metadata"])
    root = Path(__file__).resolve().parents[1]
    metadata["source_sha256"] = dict(signals["metadata"]["source_sha256"])
    for path in ("experiments/run_cpu_wall_detection.py", "python/gnss_gpu/code_detection_cpu.py"):
        metadata["source_sha256"][path] = hashlib.sha256((root/path).read_bytes()).hexdigest()
    return {"metadata": metadata, "calibration": calibration, "heldout_noise": evaluation,
            "trials": signals["trials"], "summary": summary}


def save_results(result: DetectionExperimentResult, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("calibration", "heldout_noise", "trials", "summary"):
        rows = result[name]
        with (output/f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    meta = result["metadata"]
    (output/"metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    for condition in ("direct", "multipath"):
        rows = [row for row in result["summary"] if row["condition"] == condition]
        cn0 = [row["cn0_dbhz"] for row in rows]
        axes[0].plot(cn0, [row["acceptance_fraction"] for row in rows], "o-", label=f"{condition}: accepted")
        axes[0].plot(cn0, [row["accepted_good_fraction_all"] for row in rows], "o--", label=f"{condition}: accepted, error <=30 m")
        axes[1].plot(cn0, [row["accepted_wrong_fraction_all"] for row in rows], "o-", label=condition)
    axes[0].set(ylabel="fraction of all signal trials", ylim=(-.03, 1.03), title="Frozen gate; rejected trials stay in denominators")
    axes[0].legend()
    max_wrong = max(row["accepted_wrong_fraction_all"] for row in result["summary"] if row["condition"] != "noise_only")
    axes[1].set(ylabel="accepted, |error| >30 m / all trials", ylim=(-.005, min(1.03, max(.05, max_wrong*1.2))), title="Acceptance does not validate ranging accuracy")
    axes[1].legend()
    for axis in axes[:2]:
        axis.set_xlabel("direct-reference C/N0 [dB-Hz]")
    rate = meta["measured_false_alarm_fraction"]
    lower, upper = meta["false_alarm_wilson_95"]
    axes[2].errorbar([0], [rate], yerr=[[rate-lower], [upper-rate]], fmt="o", capsize=6, label="held-out white noise; Wilson 95%")
    axes[2].axhline(meta["threshold"]["target_false_alarm_probability"], linestyle="--", color="gray", label="calibration target")
    axes[2].set(xticks=[0], xticklabels=[f"{meta['heldout_noise_trials']} independent null trials"],
                ylabel="false-alarm fraction", ylim=(0, max(.03, upper*1.2)),
                title=f"{meta['heldout_false_alarms']} held-out false alarms; no threshold retuning")
    axes[2].legend()
    figure.savefig(output/"detection_comparison.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--calibration-trials", type=int, default=1000)
    parser.add_argument("--evaluation-trials", type=int, default=1000)
    parser.add_argument("--signal-trials", type=int, default=32)
    parser.add_argument("--false-alarm-probability", type=float, default=.01)
    parser.add_argument("--root-seed", type=int, default=20261024)
    parser.add_argument("--output-dir", default="output/cpu-wall-detection")
    args = parser.parse_args()
    result = run_detection_experiment(workers=args.workers, calibration_trials=args.calibration_trials,
                                      evaluation_trials=args.evaluation_trials, signal_trials=args.signal_trials,
                                      false_alarm_probability=args.false_alarm_probability, root_seed=args.root_seed)
    save_results(result, args.output_dir)
    meta = result["metadata"]
    print(f"Frozen score threshold: {meta['threshold']['score']:.4f}")
    print(f"Held-out noise: {meta['heldout_false_alarms']}/{meta['heldout_noise_trials']} accepted "
          f"({meta['measured_false_alarm_fraction']:.2%}); Wilson 95% {meta['false_alarm_wilson_95']}")
    for row in result["summary"]:
        if row["condition"] != "noise_only":
            print(f"{row['cn0_dbhz']:g} dB-Hz {row['condition']:9s}: accepted {row['acceptance_fraction']:.1%}, "
                  f"accepted with >30 m error / all trials {row['accepted_wrong_fraction_all']:.1%}")
    print(f"Saved calibration, held-out controls and every signal trial to {args.output_dir}")


if __name__ == "__main__":
    main()
