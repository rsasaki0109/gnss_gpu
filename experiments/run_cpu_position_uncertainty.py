"""Validate causal range scatter, weighted position and model-only ellipsoids.

No new position-publication gate: this tests whether the uncertainty model is
credible before using it to decide which fixes to publish.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_cpu_multipath_tracking import _worker_init
from experiments.run_cpu_position_quality import _boolean, decode_thresholds
from experiments.run_cpu_position_recovery import read_cases
from experiments.run_cpu_rain_positioning import constellation
from experiments.run_cpu_wet_antenna import RECEIVER, _case
from gnss_gpu.range_uncertainty_cpu import CausalRangeUncertainty, RangeUncertaintyConfig
from gnss_gpu.position_uncertainty_cpu import solve_weighted_pseudorange_position

HOLDOUT_SEEDS = (24689, 24690)
SCENARIOS = ("dry", "loss_6db", "loss_30db_stress", "combined_stress")
CHI2_3_95 = 7.814727903251179


def estimate_channels(channels, satellites, config):
    """Receiver-only causal estimation; deliberately has no truth argument."""
    grouped = defaultdict(list)
    for row in channels:
        grouped[int(row["block"])].append(row)
    model = CausalRangeUncertainty(config)
    estimates = []
    for block, values in sorted(grouped.items()):
        used = [row for row in values if _boolean(row["used"])]
        prns = [int(row["prn"]) for row in used]
        if len(prns) != len(set(prns)) or any(prn < 1 or prn > len(satellites) for prn in prns):
            raise ValueError("duplicate or invalid fresh PRN")
        ranges = np.asarray([22e6+float(row["delay_m"]) for row in used])
        priors = [model.update(prn, distance, time_s=(block+.5)*.001) for prn, distance in zip(prns, ranges)]
        sigma = np.asarray([prior.sigma_m for prior in priors])
        sat = np.asarray([satellites[prn-1] for prn in prns]).reshape(-1, 3)
        common = np.full(len(sigma), np.sqrt(np.mean(sigma**2))) if len(sigma) else sigma
        solutions = {mode:solve_weighted_pseudorange_position(sat, ranges, scales, initial_position_m=(-25., 5., 0.))
                     for mode, scales in (("equal", common), ("weighted", sigma))}
        estimates.append(dict(block=block, prns=prns, ranges=ranges, sigma=sigma, priors=priors, solutions=solutions))
    return estimates


def evaluate_case(case, dataset, satellites, truth, config) -> dict[str, Any]:
    first = case["channels"][0]
    labels = dict(dataset=dataset, seed=int(first["seed"]), environment=first["environment"], scenario=first["scenario"])
    original = {int(row["block"]):row for row in case["positions"]}
    positions, ranges_output, max_delta = [], [], 0.
    for estimate in estimate_channels(case["channels"], satellites, config):
        block = estimate["block"]
        expected = original[block]
        equal = estimate["solutions"]["equal"]
        if (equal is not None) != _boolean(expected["available"]):
            raise RuntimeError("equal-weight availability differs from saved raw receiver")
        if equal is not None:
            previous = np.asarray([float(expected[key]) for key in ("east_m", "north_m", "up_m")])
            max_delta = max(max_delta, float(np.linalg.norm(equal.solution.position_m-previous)))
        for prn, distance, prior in zip(estimate["prns"], estimate["ranges"], estimate["priors"]):
            error = float(distance-np.linalg.norm(satellites[prn-1]-truth))
            ranges_output.append(dict(**labels, block=block, prn=prn, stage=expected["stage"], sigma_m=prior.sigma_m,
                                      prior_differences=prior.prior_differences, ready=prior.ready, error_m=error,
                                      covered_95=abs(error) <= 1.959963984540054*prior.sigma_m))
        for mode, result in estimate["solutions"].items():
            error = result.solution.position_m-truth if result is not None else None
            covariance = result.covariance_m2[:3, :3] if result is not None else None
            if error is not None and covariance is not None:
                nees = float(error@np.linalg.solve(covariance, error))
                position_error = float(np.linalg.norm(error))
                horizontal_error = float(np.linalg.norm(error[:2]))
                sigma_rss = float(np.sqrt(np.trace(covariance)))
                max_axis = float(np.sqrt(CHI2_3_95*np.linalg.eigvalsh(covariance)[-1]))
            else:
                nees = position_error = horizontal_error = sigma_rss = max_axis = None
            positions.append(dict(**labels, mode=mode, block=block, stage=expected["stage"],
                                  time_s=(block+.5)*.001, observations=len(estimate["prns"]),
                                  ready_ranges=sum(prior.ready for prior in estimate["priors"]), available=result is not None,
                                  east_m=float(result.solution.position_m[0]) if result else None,
                                  north_m=float(result.solution.position_m[1]) if result else None,
                                  up_m=float(result.solution.position_m[2]) if result else None,
                                  clock_bias_m=result.solution.clock_bias_m if result else None,
                                  position_error_m=position_error,
                                  horizontal_error_m=horizontal_error,
                                  normalized_residual_energy=result.normalized_residual_energy if result else None,
                                  residual_dof=result.degrees_of_freedom if result else None,
                                  position_sigma_rss_m=sigma_rss,
                                  ellipsoid_95_max_semiaxis_m=max_axis,
                                  position_nees=nees, covered_95=nees <= CHI2_3_95 if nees is not None else None))
    if max_delta > 1e-6:
        raise RuntimeError(f"equal-weight replay changed raw position by {max_delta} m")
    summaries, stages = [], []
    for mode in ("equal", "weighted"):
        selected = [row for row in positions if row["mode"] == mode]
        summaries.append(dict(**labels, mode=mode, **summarize(selected)))
        for stage in ("initial_dry", "wetting", "wet", "drying", "dry"):
            stages.append(dict(**labels, mode=mode, stage=stage, **summarize([row for row in selected if row["stage"] == stage])))
    return dict(positions=positions, ranges=ranges_output, summary=summaries, stages=stages, max_raw_replay_delta_m=max_delta)


def summarize(rows):
    available = [row for row in rows if row["available"]]
    errors = [row["position_error_m"] for row in available]
    return dict(blocks=len(rows), available=len(available), unavailable=len(rows)-len(available),
                error_over_100_m=sum(error > 100 for error in errors),
                position_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                position_p95_m=float(np.quantile(errors, .95)) if errors else None,
                position_max_m=max(errors, default=None), covered_95=sum(row["covered_95"] for row in available),
                coverage_95_fraction=float(np.mean([row["covered_95"] for row in available])) if available else None,
                mean_nees=float(np.mean([row["position_nees"] for row in available])) if available else None)


def run_experiment(workers=4, output_dir="output/cpu-position-uncertainty"):
    root, output = Path(__file__).resolve().parents[1], Path(output_dir)
    policy_path = root/"docs/position_uncertainty_policy.json"
    policy = policy_path.read_bytes()
    config = RangeUncertaintyConfig(**json.loads(policy))
    output.mkdir(parents=True, exist_ok=True)
    (output/"frozen_policy.json").write_bytes(policy)
    wet = Path("output/cpu-wet-antenna")
    previous = Path("output/cpu-position-quality")
    recovery = Path("output/cpu-position-recovery")
    metadata = json.loads((wet/"metadata.json").read_text())
    prior = json.loads((recovery/"metadata.json").read_text())
    seeds = sorted(set(prior["development_seeds"]) | set(prior["holdout_seeds"]))
    if set(seeds) & set(HOLDOUT_SEEDS):
        raise ValueError("fresh seeds overlap inspected data")
    satellites, truth = np.asarray(metadata["constellation_m"]), np.asarray(metadata["receiver_m"])
    if not np.array_equal(satellites, constellation()) or not np.array_equal(truth, RECEIVER):
        raise ValueError("source geometry differs from saved receiver")
    sources = sorted(set(prior["source_sha256"]) | {"experiments/run_cpu_position_uncertainty.py",
                     "python/gnss_gpu/position_uncertainty_cpu.py", "python/gnss_gpu/range_uncertainty_cpu.py",
                     "python/gnss_gpu/input_validation.py"})
    source_hashes = {path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources}
    inputs = {wet:("metadata.json", "channels.csv", "positions.csv"),
              previous:("metadata.json", "holdout_channels.csv", "holdout_raw_positions.csv"),
              recovery:("metadata.json", "holdout_channels.csv", "holdout_raw_positions.csv")}
    input_hashes = {str(directory/name):hashlib.sha256((directory/name).read_bytes()).hexdigest()
                    for directory, names in inputs.items() for name in names}
    cases = read_cases(wet, "channels.csv", "positions.csv")
    for directory in (previous, recovery):
        cases += read_cases(directory, "holdout_channels.csv", "holdout_raw_positions.csv")
    results = [evaluate_case(case, "development", satellites, truth, config) for case in cases]
    print("Development replay complete; frozen prior-only scatter model. Generating new IQ.", flush=True)
    gates = decode_thresholds(metadata)
    jobs = [(seed, environment, scenario, gates) for seed in HOLDOUT_SEEDS
            for environment in ("open", "wall") for scenario in SCENARIOS]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as pool:
        fresh = list(pool.map(_case, jobs))
    results.extend(evaluate_case(case, "holdout", satellites, truth, config) for case in fresh)
    if policy_path.read_bytes() != policy or any(hashlib.sha256((root/p).read_bytes()).hexdigest() != h for p,h in source_hashes.items()):
        raise RuntimeError("policy/source changed during evaluation")
    saved_metadata = dict(policy=asdict(config), policy_sha256=hashlib.sha256(policy).hexdigest(), workers=workers,
                          development_seeds=seeds, holdout_seeds=list(HOLDOUT_SEEDS), scenarios=list(SCENARIOS),
                          source_sha256=source_hashes, input_sha256=input_hashes, receiver_metadata=metadata,
                          max_raw_replay_delta_m=max(result["max_raw_replay_delta_m"] for result in results),
                          design="Prior-only MAD of range differences, stationary exponential-smoothing correction. Engineering defaults frozen before new IQ; no truth fitting or holdout retuning. Equal and weighted estimators share measurements; equal covariance uses RMS of declared range sigmas.",
                          truth_boundary="Estimation accepts only used flag, PRN, measured delay and block timestamp. Truth/stage/weather/visibility only score estimates afterward.",
                          limitations="Scatter is not bias uncertainty; independent unbiased Gaussian covariance assumptions can fail under reflection, tracking, ambiguity and transients. Nominal 95% ellipsoids are not protection levels. Static geometry, paired noise and correlated epochs, two fresh seeds; no confidence intervals or real-rain validation. No publication-policy change.")
    for key in ("positions", "ranges", "summary", "stages"):
        rows = [row for result in results for row in result[key]]
        with (output/f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for key in ("channels", "positions"):
        rows = [row for case in fresh for row in case[key]]
        with (output/f"holdout_{'raw_positions' if key == 'positions' else key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output/"metadata.json").write_text(json.dumps(saved_metadata, indent=2, allow_nan=False)+"\n")
    plot_results(output)
    print(f"Saved paired estimation and uncertainty calibration results to {output}", flush=True)


def plot_results(output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with (output/"positions.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["dataset"] == "holdout" and row["available"] == "True"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for index, environment in enumerate(("open", "wall")):
        for mode, offset in (("equal", -.18), ("weighted", .18)):
            groups = [[row for row in rows if row["mode"] == mode and row["environment"] == environment and row["scenario"] == scenario] for scenario in SCENARIOS]
            axes[index, 0].bar(np.arange(4)+offset, [np.sqrt(np.mean([float(row["position_error_m"])**2 for row in group])) for group in groups], .36, label=mode)
            axes[index, 1].bar(np.arange(4)+offset, [np.mean([row["covered_95"] == "True" for row in group])*100 for group in groups], .36, label=mode)
        for axis in axes[index]:
            axis.set(xticks=np.arange(4), xticklabels=SCENARIOS, title=environment)
            axis.tick_params(axis="x", labelrotation=18)
            axis.legend()
            axis.grid(axis="y", alpha=.2)
        axes[index, 0].set_ylabel("3D RMSE [m], available estimates")
        axes[index, 1].set(ylabel="Nominal 95% ellipsoid coverage [%]", ylim=(0, 101))
        axes[index, 1].axhline(95, color="black", linestyle="--", linewidth=1)
    figure.suptitle("Fresh-noise uncertainty validation: no publication filtering")
    figure.savefig(output/"position_uncertainty.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="output/cpu-position-uncertainty")
    args = parser.parse_args()
    run_experiment(workers=args.workers, output_dir=args.output_dir)
