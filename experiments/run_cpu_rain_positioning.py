"""Rain attenuation versus independent wet-wall sensitivity, IQ to position.

PYTHONPATH=python:. python experiments/run_cpu_rain_positioning.py --workers 4
Static per-channel IQ snapshots; no truth-dependent receiver gating or weights.
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

from experiments.run_cpu_multipath_tracking import _worker_init
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.code_detection_cpu import ca_detection_statistic, calibrate_ca_threshold
from gnss_gpu.acquisition_cpu import CaSearch, CaSearchThreshold
from gnss_gpu.input_validation import positive_int
from gnss_gpu.positioning_cpu import solve_pseudorange_position
from gnss_gpu.rain_cpu import RainLayer
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.tracking_cpu import CpuCaTracker
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal

FS, COUNT, REFERENCE = 2*CHIP_RATE, 2046, 22e6
CALIBRATION_SEED = 13579
CALIBRATION_DOMAIN, SIGNAL_DOMAIN = 0x52414341, 0x52414E53
SCENARIOS = (("dry", 0., 0.), ("rain_25", 25., 0.), ("rain_100", 100., 0.),
             ("film_0.1mm", 0., .0001), ("film_1mm", 0., .001), ("rain_100_film_1mm", 100., .001))


def constellation():
    """Fixed synthetic local ENU source geometry, not satellite ephemerides."""
    azimuth = np.deg2rad(np.arange(15., 360., 30.))
    elevation = np.deg2rad([20, 40, 65, 30, 50, 75, 25, 45, 70, 35, 55, 80])
    unit = np.column_stack([np.cos(elevation)*np.sin(azimuth),
                            np.cos(elevation)*np.cos(azimuth), np.sin(elevation)])
    return np.array([-20., 0., 1.5])+REFERENCE*unit


def _calibration(prn):
    scores = []
    for trial in range(199):
        rng = np.random.default_rng(np.random.SeedSequence([CALIBRATION_SEED, CALIBRATION_DOMAIN, prn, trial]))
        scores.append(ca_detection_statistic(estimate_ca_delay(rng.normal(size=2*COUNT), FS, prn=prn)))
    return prn, calibrate_ca_threshold(scores, FS, prn=prn), scores


def _pose_job(job):
    pose, repeats, gates, confirmation_blocks, signal_seed = job
    receiver = np.array([-20.+.03*pose, -1.+.13*pose, 1.5])
    satellites = constellation()
    simulator = CpuSignalSimulator(FS, noise_floor_db=-10000)
    observations, channels = [], []
    solutions = {}
    for environment in ("open", "wall"):
        scenarios = SCENARIOS[:3] if environment == "open" else SCENARIOS
        for label, rate, film in scenarios:
            # The distant y bounds remove all wall intersections in the open control.
            bounds = (-100., 100.) if environment == "wall" else (1e9, 1e9+100.)
            wall = VerticalWall(0., *bounds, 0., 100., antenna_lhcp_response=.1j,
                                water_film_thickness_m=film)
            clean = [simulate_wall_signal(simulator, wall, receiver, [0., 0., 0.], satellite, COUNT,
                                          prn=prn, reference_range_m=REFERENCE,
                                          include_reflection=environment == "wall", integrate_samples=True,
                                          rain_layer=RainLayer(rate, 3000.))
                     for prn, satellite in enumerate(satellites, 1)]
            for trial in range(repeats):
                accepted_satellites, pseudoranges = [], []
                for prn, block in enumerate(clean, 1):
                    # Nominal unit direct pre-aperture C/N0=55 dB-Hz; noise is
                    # identical across ALL weather and environment comparisons.
                    # The geometry is static and the signal repeats each 1 ms;
                    # each confirmation block gets fresh independent noise.
                    tracker = CpuCaTracker(CaSearchThreshold(CaSearch(FS, (0.,), prn), gates[prn]),
                                           confirmation_blocks=confirmation_blocks)
                    for confirm in range(confirmation_blocks):
                        # confirm=0 exactly retains the original raw seed key.
                        key = [signal_seed, SIGNAL_DOMAIN, pose, trial, prn]
                        if confirm:
                            key.append(confirm)
                        rng = np.random.default_rng(np.random.SeedSequence(key))
                        iq = block.iq+rng.normal(0., np.sqrt(FS/(2*10**(55/10))), 2*COUNT)
                        state = tracker.update(iq, start_sample=confirm*COUNT)
                    used = state.measurement_used and state.delay_samples is not None
                    if state.measurement_used and state.delay_samples is not None:
                        accepted_satellites.append(satellites[prn-1])
                        pseudoranges.append(REFERENCE+state.delay_samples*C_LIGHT/FS)
                    channels.append(dict(pose=pose, trial=trial, environment=environment, scenario=label,
                                         prn=prn, accepted=used, statistic=state.detection_statistic,
                                         detection_accepted=state.detection_accepted,
                                         receiver_state=state.status, confirmation_count=state.confirmation_count,
                                         delay_m=state.delay_samples*C_LIGHT/FS if state.delay_samples is not None else None,
                                         direct_visible=bool(block.direct_visible[0]),
                                         reflection_visible=bool(block.reflection_visible[0]),
                                         direct_rain_loss_db=float(block.direct_rain_loss_db[0]),
                                         reflected_rain_loss_db=float(block.reflected_rain_loss_db[0]),
                                         reflection_voltage_abs=float(abs(block.reflection_coefficient[0]))))
                enough = len(pseudoranges) >= 4
                solution = solve_pseudorange_position(np.asarray(accepted_satellites), pseudoranges,
                                                      initial_position_m=[-25., 5., 0.]) if enough else None
                error = solution.position_m-receiver if solution is not None else None
                key = (environment, trial)
                if label == "dry":
                    solutions[key] = solution
                baseline = solutions.get(key)
                paired_shift = (float(np.linalg.norm(solution.position_m-baseline.position_m))
                                if solution is not None and baseline is not None else None)
                observations.append(dict(pose=pose, trial=trial, environment=environment, scenario=label,
                                         rain_mm_h=rate, film_mm=1000*film, available=solution is not None,
                                         accepted_channels=len(pseudoranges),
                                         failure_reason=None if solution is not None else (
                                             "fewer_than_four_channels" if not enough else "rank_or_convergence_failure"),
                                         error_east_m=float(error[0]) if error is not None else None,
                                         error_north_m=float(error[1]) if error is not None else None,
                                         error_up_m=float(error[2]) if error is not None else None,
                                         horizontal_error_m=float(np.linalg.norm(error[:2])) if error is not None else None,
                                         position_error_m=float(np.linalg.norm(error)) if error is not None else None,
                                         clock_bias_m=solution.clock_bias_m if solution is not None else None,
                                         paired_position_shift_m=paired_shift))
    return dict(positions=observations, channels=channels)


def summarize_positions(rows):
    available = [row for row in rows if row["available"]]
    paired = [row["paired_position_shift_m"] for row in rows if row["paired_position_shift_m"] is not None]
    def quantile(key, probability):
        return float(np.quantile([row[key] for row in available], probability)) if available else None
    return dict(trials=len(rows), available=len(available), unavailable=len(rows)-len(available),
                horizontal_p50_m_conditional=quantile("horizontal_error_m", .5),
                horizontal_p95_m_conditional=quantile("horizontal_error_m", .95),
                position_p50_m_conditional=quantile("position_error_m", .5),
                position_p95_m_conditional=quantile("position_error_m", .95),
                position_max_m_conditional=quantile("position_error_m", 1.),
                available_error_over_100_m=sum(row["position_error_m"] > 100 for row in available),
                paired_trials=len(paired),
                paired_shift_p50_m=float(np.median(paired)) if paired else None,
                paired_shift_p95_m=float(np.quantile(paired, .95)) if paired else None)


def run_experiment(*, workers=1, repeats=8, confirmation_blocks=2, signal_seed=24681) -> dict[str, Any]:
    workers = positive_int("workers", workers)
    repeats = positive_int("repeats", repeats)
    confirmation_blocks = positive_int("confirmation_blocks", confirmation_blocks)
    signal_seed = positive_int("signal_seed", signal_seed)
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
    try:
        calibration = list(pool.map(_calibration, range(1, 13))) if pool else list(map(_calibration, range(1, 13)))
        gates = {prn:gate for prn, gate, _ in calibration}
        # Freeze all per-PRN single-carrier gates before any weather evaluation.
        jobs = [(pose, repeats, gates, confirmation_blocks, signal_seed) for pose in range(16)]
        cases = list(pool.map(_pose_job, jobs)) if pool else list(map(_pose_job, jobs))
    finally:
        if pool:
            pool.shutdown()
    positions = [row for case in cases for row in case["positions"]]
    channels = [row for case in cases for row in case["channels"]]
    summary = [dict(environment=environment, scenario=label,
                    **summarize_positions([row for row in positions if row["environment"] == environment and row["scenario"] == label]))
               for environment in ("open", "wall") for label, _, _ in (SCENARIOS[:3] if environment == "open" else SCENARIOS)]
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_cpu_rain_positioning.py", "experiments/run_cpu_multipath_tracking.py",
               "python/gnss_gpu/rain_cpu.py", "python/gnss_gpu/positioning_cpu.py",
               "python/gnss_gpu/wall_signal_sim.py", "python/gnss_gpu/fresnel.py",
               "python/gnss_gpu/code_delay_cpu.py", "python/gnss_gpu/code_detection_cpu.py",
               "python/gnss_gpu/signal_sim_cpu.py", "docs/rain_sources.json"]
    sources.extend(["python/gnss_gpu/tracking_cpu.py", "python/gnss_gpu/acquisition_cpu.py"])
    metadata = dict(sampling_freq_hz=FS, observation_s=.001, calibration_seed=CALIBRATION_SEED,
                    signal_seed=signal_seed, calibration_domain=CALIBRATION_DOMAIN, signal_domain=SIGNAL_DOMAIN,
                    confirmation_blocks=confirmation_blocks, minimum_observation_s=.001*confirmation_blocks,
                    poses=16, noise_repeats_per_pose=repeats, cn0_reference_db_hz=55., workers=workers,
                    scenarios=SCENARIOS, gates={str(prn):asdict(gate) for prn, gate in gates.items()},
                    constellation_m=constellation().tolist(), reference_range_m=REFERENCE,
                    pose_rule_m="[-20+0.03*pose, -1+0.13*pose, 1.5] for pose=0..15",
                    rain_layer_bounds_m=[0., 3000.], wall_bounds_m=[0., -100., 100., 0., 100.],
                    water_permittivity=[80., -8.], antenna_lhcp_voltage=[0., .1],
                    positioning="Existing CpuCaTracker consecutive confirmation, then NumPy nonlinear equal-weight least squares, xyz plus receiver clock in metres; initial [-25,5,0]; no truth-based channel rejection or weights",
                    ambiguity_assumption="Known common 22 Mm range reference selects the integer code period; no ambiguity resolution is claimed",
                    pairing="Identical noise per pose/trial/PRN across environments/weather; scenarios are paired, not independent trials. 16 prescribed poses with 8 noise repeats by default, not 128 independent city geometries.",
                    limitations="Static synthetic per-channel isolated IQ; known zero carrier and PRN, no multiaccess interference, orbit/transmit-time/Sagnac/clock/ionosphere or tropospheric residuals, rain phase/group delay, wet antenna, droplets, roughness or diffuse scattering. Film depth/permittivity are illustrative and independent of rainfall rate. No real-weather performance claim.",
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    return dict(positions=positions, channels=channels, summary=summary, metadata=metadata,
                calibration=[dict(prn=prn, trial=trial, statistic=score) for prn, _, scores in calibration for trial, score in enumerate(scores)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--confirmation-blocks", type=int, default=2)
    parser.add_argument("--signal-seed", type=int, default=24681)
    parser.add_argument("--output-dir", default="output/cpu-rain-positioning-confirmed")
    args = parser.parse_args()
    result = run_experiment(workers=args.workers, repeats=args.repeats,
                            confirmation_blocks=args.confirmation_blocks, signal_seed=args.signal_seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("positions", "channels", "summary", "calibration"):
        with (output/f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    (output/"metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    for axis, environment in zip(axes[:2], ("open", "wall")):
        selected = [row for row in result["summary"] if row["environment"] == environment]
        x = np.arange(len(selected))
        axis.bar(x-.2, [row["horizontal_p50_m_conditional"] for row in selected], .4, label="horizontal P50")
        axis.bar(x+.2, [row["horizontal_p95_m_conditional"] for row in selected], .4, label="horizontal P95")
        axis.set(xticks=x, xticklabels=[row["scenario"] for row in selected], ylabel="horizontal position error [m]",
                 title=f"{environment}: isolated-channel IQ → pseudorange → position + clock; conditional on a solution")
        axis.legend()
        axis.grid(axis="y", alpha=.2)
    effects = [row for row in result["summary"] if row["scenario"] != "dry"]
    x = np.arange(len(effects))
    axes[2].bar(x-.2, [row["paired_shift_p50_m"] for row in effects], .4, label="paired shift P50")
    axes[2].bar(x+.2, [row["paired_shift_p95_m"] for row in effects], .4, label="paired shift P95")
    axes[2].set_yscale("log")
    axes[2].set(xticks=x, xticklabels=[f"{row['environment']}/{row['scenario']}" for row in effects],
                ylabel="3D position shift relative to paired dry trial [m]",
                title="Effect size, NOT error increase; rain rate and film thickness are independent controls")
    axes[2].tick_params(axis="x", labelrotation=20)
    axes[2].legend()
    axes[2].grid(axis="y", alpha=.2)
    figure.savefig(output/"rain_positioning.png", dpi=150)
    plt.close(figure)
    for row in result["summary"]:
        print(row)
    print(f"Saved all channel decisions, unavailable positions, paired trials and model provenance to {output}")


if __name__ == "__main__":
    main()
