"""Wet-delay residual sensitivity, separated from rain attenuation and water film.

PYTHONPATH=python:. python experiments/run_cpu_wet_delay_positioning.py --workers 4
Same fixed constellation/receiver as the rain study; fresh noise seed and no
truth-based receiver gating. A separate noiseless geometric control is labeled.
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
from experiments.run_cpu_rain_positioning import FS, COUNT, REFERENCE, _calibration, constellation
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, CaSearchThreshold
from gnss_gpu.input_validation import positive_int
from gnss_gpu.positioning_cpu import solve_pseudorange_position
from gnss_gpu.rain_cpu import RainLayer
from gnss_gpu.signal_sim_cpu import L1_HZ
from gnss_gpu.tracking_cpu import CpuCaTracker
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal
from gnss_gpu.wet_delay_cpu import WetDelayResidualLayer

SIGNAL_SEED, SIGNAL_DOMAIN = 24682, 0x57455453
# Label, signed zenith residual [m], rain [mm/h], independent film depth [m].
SCENARIOS = (("reference", 0., 0., 0.), ("zwd_minus_0.3", -.3, 0., 0.),
             ("zwd_minus_0.1", -.1, 0., 0.), ("zwd_plus_0.1", .1, 0., 0.),
             ("zwd_plus_0.3", .3, 0., 0.), ("rain_100", 0., 100., 0.),
             ("rain_100_zwd_plus_0.3", .3, 100., 0.),
             ("rain_100_film_1mm", 0., 100., .001),
             ("rain_100_film_1mm_zwd_plus_0.3", .3, 100., .001))


def paired_noise(pose, trial, prn, confirm, wet_delay_m):
    """Common random numbers in the direct carrier frame; unchanged marginal N0.

    A deterministic unit-modulus rotation of circular Gaussian noise preserves
    its distribution. This cross-scenario variance-reduction coupling uses
    simulated delay ONLY in the noise generator, never in the receiver. It is
    not a claim that physical receiver noise tracks atmospheric phase.
    """
    rng = np.random.default_rng(np.random.SeedSequence([SIGNAL_SEED, SIGNAL_DOMAIN, pose, trial, prn, confirm]))
    noise = rng.normal(0., np.sqrt(FS/(2*10**(55/10))), (COUNT, 2))
    z = (noise[:, 0]+1j*noise[:, 1])*np.exp(-2j*np.pi*L1_HZ*wet_delay_m/C_LIGHT)
    return np.column_stack([z.real, z.imag]).ravel()


def geometric_control(receiver, satellites, zenith_residual):
    """Noiseless ALL-LOS range-domain control, not a receiver measurement."""
    layer = WetDelayResidualLayer(zenith_residual)
    excess, _ = layer.excess_range_and_rate(receiver, satellites)
    ranges = np.linalg.norm(satellites-receiver, axis=1)+excess
    return solve_pseudorange_position(satellites, ranges, initial_position_m=[-25., 5., 0.])


def _pose_job(job):
    pose, repeats, gates = job
    receiver = np.array([-20.+.03*pose, -1.+.13*pose, 1.5])
    satellites = constellation()
    simulator = CpuSignalSimulator(FS, noise_floor_db=-10000)
    positions, channels, controls = [], [], []
    for residual in (-.3, -.1, 0., .1, .3):
        solution = geometric_control(receiver, satellites, residual)
        if solution is None:
            raise RuntimeError("full-LOS noiseless geometric control failed")
        controls.append(dict(pose=pose, zenith_residual_m=residual,
                             error_east_m=float(solution.position_m[0]-receiver[0]),
                             error_north_m=float(solution.position_m[1]-receiver[1]),
                             error_up_m=float(solution.position_m[2]-receiver[2]),
                             clock_bias_m=solution.clock_bias_m))
    baselines = {}
    for environment in ("open", "wall"):
        for label, residual, rain, film in (SCENARIOS[:7] if environment == "open" else SCENARIOS):
            bounds = (-100., 100.) if environment == "wall" else (1e9, 1e9+100.)
            wall = VerticalWall(0., *bounds, 0., 100., antenna_lhcp_response=.1j, water_film_thickness_m=film)
            clean = [simulate_wall_signal(
                simulator, wall, receiver, [0., 0., 0.], satellite, COUNT, prn=prn,
                reference_range_m=REFERENCE, include_reflection=environment == "wall", integrate_samples=True,
                rain_layer=RainLayer(rain), wet_delay_layer=WetDelayResidualLayer(residual))
                     for prn, satellite in enumerate(satellites, 1)]
            for trial in range(repeats):
                accepted, ranges = [], []
                for prn, block in enumerate(clean, 1):
                    tracker = CpuCaTracker(CaSearchThreshold(CaSearch(FS, (0.,), prn), gates[prn]))
                    for confirm in range(2):
                        iq = block.iq+paired_noise(pose, trial, prn, confirm, float(block.direct_wet_delay_m[0]))
                        state = tracker.update(iq, start_sample=confirm*COUNT)
                    used = state.measurement_used and state.delay_samples is not None
                    if state.measurement_used and state.delay_samples is not None:
                        accepted.append(satellites[prn-1])
                        ranges.append(REFERENCE+state.delay_samples*C_LIGHT/FS)
                    channels.append(dict(pose=pose, trial=trial, environment=environment, scenario=label,
                                         prn=prn, used=used, detection_accepted=state.detection_accepted,
                                         status=state.status, confirmation_count=state.confirmation_count,
                                         statistic=state.detection_statistic,
                                         delay_m=state.delay_samples*C_LIGHT/FS if state.delay_samples is not None else None,
                                         direct_visible=bool(block.direct_visible[0]), reflection_visible=bool(block.reflection_visible[0]),
                                         direct_wet_delay_m=float(block.direct_wet_delay_m[0]),
                                         reflected_wet_delay_m=float(block.reflected_wet_delay_m[0]),
                                         direct_rain_loss_db=float(block.direct_rain_loss_db[0])))
                solution = solve_pseudorange_position(accepted, ranges, initial_position_m=[-25., 5., 0.]) if len(ranges) >= 4 else None
                if label == "reference":
                    baselines[environment, trial] = solution
                baseline = baselines.get((environment, trial))
                error = solution.position_m-receiver if solution is not None else None
                shift = (solution.position_m-baseline.position_m
                         if solution is not None and baseline is not None else None)
                clock_shift = (solution.clock_bias_m-baseline.clock_bias_m
                               if solution is not None and baseline is not None else None)
                positions.append(dict(pose=pose, trial=trial, environment=environment, scenario=label,
                                      zenith_residual_m=residual, rain_mm_h=rain, film_mm=1000*film,
                                      available=solution is not None, accepted_channels=len(ranges),
                                      failure_reason=None if solution is not None else (
                                          "fewer_than_four_channels" if len(ranges) < 4 else "rank_or_convergence_failure"),
                                      error_east_m=float(error[0]) if error is not None else None,
                                      error_north_m=float(error[1]) if error is not None else None,
                                      error_up_m=float(error[2]) if error is not None else None,
                                      horizontal_error_m=float(np.linalg.norm(error[:2])) if error is not None else None,
                                      position_error_m=float(np.linalg.norm(error)) if error is not None else None,
                                      clock_bias_m=solution.clock_bias_m if solution is not None else None,
                                      paired_clock_shift_m=clock_shift,
                                      paired_up_shift_m=float(shift[2]) if shift is not None else None,
                                      paired_position_shift_m=float(np.linalg.norm(shift)) if shift is not None else None))
    return dict(positions=positions, channels=channels, controls=controls)


def summarize_positions(rows):
    from experiments.run_cpu_rain_positioning import summarize_positions as rain_summary
    result = rain_summary(rows)
    available = [row for row in rows if row["available"]]
    up_shift = [row["paired_up_shift_m"] for row in rows if row["paired_up_shift_m"] is not None]
    clock_shift = [row["paired_clock_shift_m"] for row in rows if row["paired_clock_shift_m"] is not None]
    result.update(vertical_abs_p95_m_conditional=float(np.quantile([abs(row["error_up_m"]) for row in available], .95)) if available else None,
                  paired_up_shift_p50_m=float(np.median(up_shift)) if up_shift else None,
                  paired_up_shift_p05_m=float(np.quantile(up_shift, .05)) if up_shift else None,
                  paired_up_shift_p95_m=float(np.quantile(up_shift, .95)) if up_shift else None,
                  paired_clock_shift_p50_m=float(np.median(clock_shift)) if clock_shift else None)
    return result


def run_experiment(*, workers=1, repeats=8) -> dict[str, Any]:
    workers, repeats = positive_int("workers", workers), positive_int("repeats", repeats)
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
    try:
        calibration = list(pool.map(_calibration, range(1, 13))) if pool else list(map(_calibration, range(1, 13)))
        gates = {prn:gate for prn, gate, _ in calibration}
        jobs = [(pose, repeats, gates) for pose in range(16)]
        cases = list(pool.map(_pose_job, jobs)) if pool else list(map(_pose_job, jobs))
    finally:
        if pool:
            pool.shutdown()
    positions = [row for case in cases for row in case["positions"]]
    summary = [dict(environment=environment, scenario=label,
                    **summarize_positions([row for row in positions if row["environment"] == environment and row["scenario"] == label]))
               for environment in ("open", "wall") for label, *_ in (SCENARIOS[:7] if environment == "open" else SCENARIOS)]
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_cpu_wet_delay_positioning.py", "experiments/run_cpu_rain_positioning.py",
               "experiments/run_cpu_multipath_tracking.py", "python/gnss_gpu/wet_delay_cpu.py",
               "python/gnss_gpu/wall_signal_sim.py", "python/gnss_gpu/rain_cpu.py", "python/gnss_gpu/fresnel.py",
               "python/gnss_gpu/positioning_cpu.py", "python/gnss_gpu/code_delay_cpu.py",
               "python/gnss_gpu/code_detection_cpu.py", "python/gnss_gpu/acquisition_cpu.py",
               "python/gnss_gpu/tracking_cpu.py", "python/gnss_gpu/signal_sim_cpu.py", "docs/rain_sources.json"]
    metadata = dict(signal_seed=SIGNAL_SEED, signal_domain=SIGNAL_DOMAIN, calibration_seed=13579,
                    calibration_domain=0x52414341, thresholds={str(prn):asdict(gate) for prn, gate in gates.items()},
                    poses=16, noise_repeats_per_pose=repeats, workers=workers, sampling_freq_hz=FS,
                    confirmation_blocks=2, cn0_reference_db_hz=55., scenarios=SCENARIOS,
                    wet_layer_bounds_m=[0., 3000.], mapping="Uniform residual refractivity slab; finite straight-ray lengths; csc(elevation) for full crossings, not Niell/GMF",
                    residual_definition="Actual-minus-applied zenith wet delay over full slab. Signed correction error, not negative water vapour or rainfall-to-ZWD relation.",
                    geometry=dict(constellation_m=constellation().tolist(), reference_range_m=REFERENCE,
                                  pose_rule_m="[-20+0.03*pose,-1+0.13*pose,1.5]", wall_bounds_m=[0.,-100.,100.,0.,100.]),
                    antenna_lhcp_voltage=[0., .1], water_permittivity=[80., -8.],
                    noise_pairing="Common random numbers in direct carrier frame across weather scenarios; unit-modulus phase rotation preserves circular-Gaussian marginal N0, but is an artificial variance-reduction coupling, not physical noise tracking. Delay is used only in generation, never receiver.",
                    control="Noiseless range-domain ALL-LOS geometric solution is separate from IQ receiver results; no truth-based selection in receiver.",
                    assumptions="Static per-channel isolated IQ snapshots, known zero carrier/PRN, 22 Mm coarse code-period reference, two-frame confirmation and equal-weight position/clock LS; fresh noise seed, frozen gates",
                    limitations="Flat slab, constant signed residual, no bending/curved Earth, realistic wet mapping coefficients, humidity retrieval, weather dynamics/gradients, wet antenna/radome or RF/PVT validation. Same 16 poses, not independent city routes. Local results do not establish false-fix or weather performance bounds.",
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    return dict(positions=positions, channels=[row for case in cases for row in case["channels"]],
                controls=[row for case in cases for row in case["controls"]], summary=summary, metadata=metadata,
                calibration=[dict(prn=prn, trial=i, statistic=score) for prn, _, scores in calibration for i, score in enumerate(scores)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output-dir", default="output/cpu-wet-delay-positioning")
    args = parser.parse_args()
    result = run_experiment(workers=args.workers, repeats=args.repeats)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("positions", "channels", "controls", "summary", "calibration"):
        with (output/f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    (output/"metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    control = [row for row in result["controls"] if row["pose"] == 0]
    for key, label in (("error_up_m", "height shift"), ("clock_bias_m", "receiver clock [m]")):
        axes[0, 0].plot([row["zenith_residual_m"] for row in control], [row[key] for row in control], "o-", label=label)
    axes[0, 0].set(xlabel="signed zenith residual [m]", ylabel="geometric noiseless control [m]", title="ALL-LOS range-domain control, separate from receiver")
    for axis, environment in ((axes[0, 1], "open"), (axes[1, 0], "wall")):
        selected = [row for row in result["summary"] if row["environment"] == environment and row["scenario"] in {s[0] for s in SCENARIOS[:5]}]
        lookup = {s[0]:s[1] for s in SCENARIOS[:5]}
        selected.sort(key=lambda row:lookup[row["scenario"]])
        x = [lookup[row["scenario"]] for row in selected]
        axis.plot(x, [row["paired_up_shift_p50_m"] for row in selected], "o-", label="paired height-shift P50")
        axis.fill_between(x, [row["paired_up_shift_p05_m"] for row in selected],
                          [row["paired_up_shift_p95_m"] for row in selected], alpha=.2, label="paired distribution P05–P95")
        axis.set(xlabel="signed zenith residual [m]", ylabel="height shift vs matched reference [m]",
                 title=f"{environment}: noisy IQ → receiver → position (effect size)")
    labels = [SCENARIOS[i][0] for i in (0, 5, 6)]
    for offset, environment in ((-.2, "open"), (.2, "wall")):
        rows = {row["scenario"]:row for row in result["summary"] if row["environment"] == environment}
        axes[1, 1].bar(np.arange(len(labels))+offset, [rows[label]["vertical_abs_p95_m_conditional"] for label in labels], .4, label=environment)
    axes[1, 1].set(xticks=np.arange(len(labels)), xticklabels=["reference", "100 mm/h", "100 mm/h\n+ ZWD residual 0.3 m"], ylabel="absolute height error P95 [m]",
                   title="Rain attenuation and wet-delay residual are separate controls")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend()
    figure.savefig(output/"wet_delay_positioning.png", dpi=150)
    plt.close(figure)
    for row in result["summary"]:
        print(row)
    print(f"Saved all decisions, paired shifts, geometric controls and provenance to {output}")


if __name__ == "__main__":
    main()
