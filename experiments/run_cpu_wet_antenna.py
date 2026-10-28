"""Prescribed antenna wetting/drying: continuous code following and positioning.

PYTHONPATH=python:. python experiments/run_cpu_wet_antenna.py --workers 4
Response amplitudes/times are illustrative sensitivity/stress inputs, not
rainfall-derived or experimentally calibrated antenna/radome behaviour.
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
from experiments.run_cpu_rain_positioning import COUNT, FS, REFERENCE, _calibration, constellation
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, CaSearchThreshold
from gnss_gpu.input_validation import positive_int
from gnss_gpu.positioning_cpu import solve_pseudorange_position
from gnss_gpu.tracking_cpu import CpuCaTracker
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal
from gnss_gpu.wet_antenna_cpu import WetAntennaResponse

BLOCKS, SEEDS, NOISE_DOMAIN = 500, (24683, 24684), 0x57414E54
RECEIVER = np.array([-20., -1., 1.5])
RESPONSES = {
    "dry": WetAntennaResponse(),
    "loss_6db": WetAntennaResponse(loss_db=6.),
    "loss_30db_stress": WetAntennaResponse(loss_db=30.),
    "common_code_0.3m": WetAntennaResponse(code_bias_m=.3),
    "directional_code_0.3m": WetAntennaResponse(elevation_code_bias_m=.3),
    "phase_pattern": WetAntennaResponse(phase_bias_rad=1., elevation_phase_bias_rad=.5),
    "combined_stress": WetAntennaResponse(loss_db=30., code_bias_m=.3,
                                         elevation_code_bias_m=.3, phase_bias_rad=1., elevation_phase_bias_rad=.5),
}


def stage_at(time_s):
    if time_s < .05:
        return "initial_dry"
    if time_s < .1:
        return "wetting"
    if time_s < .2:
        return "wet"
    if time_s < .4:
        return "drying"
    return "dry"


def noise_iq(seed, prn, block, phase_rad):
    """Phase-paired circular noise, held AFTER antenna signal attenuation."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, NOISE_DOMAIN, prn, block]))
    noise = rng.normal(0., np.sqrt(FS/(2*10**(55/10))), (COUNT, 2))
    z = (noise[:, 0]+1j*noise[:, 1])*np.exp(1j*phase_rad)
    return np.column_stack([z.real, z.imag]).ravel()


def recovery_metrics(rows, run_blocks=5):
    """All-frame availability plus retrospectively confirmed recovery timing.

    Recovery means five consecutive available positions after drying begins,
    and the decision time is the END of the fifth block. It is only reported
    if at least one full-wet plateau block lacked a position. No-loss and
    not-recovered cases have distinct reasons, with None rather than zero time.
    """
    run_blocks = positive_int("run_blocks", run_blocks)
    first = next((row["block"]+1 for row in rows if row["available"]), None)
    wet_loss = any(not row["available"] for row in rows if row["stage"] == "wet")
    end = None
    if wet_loss:
        count = 0
        previous = None
        for row in rows:
            if row["block"] < 200:
                continue
            count = count+1 if row["available"] and (previous is None or row["block"] == previous+1) else (1 if row["available"] else 0)
            previous = row["block"]
            if count >= run_blocks:
                end = row["block"]+1
                break
    return dict(blocks=len(rows), position_available=sum(row["available"] for row in rows),
                position_unavailable=sum(not row["available"] for row in rows),
                first_position_latency_ms=first, wet_plateau_had_loss=wet_loss,
                recovery_after_drying_start_ms=None if end is None else end-200,
                recovery_decision_time_ms=end,
                recovery_reason="no_plateau_loss" if not wet_loss else ("not_recovered" if end is None else "five_consecutive_positions"))


def _case(job) -> dict[str, Any]:
    seed, environment, name, gates = job
    response = RESPONSES[name]
    y_min, y_max = (-100., 100.) if environment == "wall" else (1e9, 1e9+100.)
    wall = VerticalWall(0., y_min, y_max, 0., 100., antenna_lhcp_response=.1j)
    satellites = constellation()
    simulator = CpuSignalSimulator(FS, noise_floor_db=-10000)
    trackers = {prn:CpuCaTracker(CaSearchThreshold(CaSearch(FS, (0.,), prn), gates[prn])) for prn in range(1, 13)}
    channels, positions = [], []
    for index in range(BLOCKS):
        start = index*COUNT
        time_s = (index+.5)*.001
        stage = stage_at(time_s)
        accepted, ranges = [], []
        for prn, satellite in enumerate(satellites, 1):
            signal = simulate_wall_signal(
                simulator, wall, RECEIVER, [0., 0., 0.], satellite, COUNT, prn=prn,
                start_sample=start, reference_range_m=REFERENCE,
                nav_bit=-1 if (index//20) % 2 else 1,
                include_reflection=environment == "wall", integrate_samples=True, antenna_response=response)
            iq = signal.iq+noise_iq(seed, prn, index, signal.direct_antenna_phase_rad)
            state = trackers[prn].update(iq, start_sample=start)
            if state.measurement_used and state.delay_samples is not None:
                accepted.append(satellite)
                ranges.append(REFERENCE+state.delay_samples*C_LIGHT/FS)
            channels.append(dict(seed=seed, environment=environment, scenario=name, block=index, time_s=time_s,
                                 stage=stage, prn=prn, status=state.status, used=state.measurement_used,
                                 gate_passed=state.detection_accepted, statistic=state.detection_statistic,
                                 confirmation_count=state.confirmation_count, missed_blocks=state.missed_blocks,
                                 delay_m=state.delay_samples*C_LIGHT/FS if state.delay_samples is not None else None,
                                 loss_db=float(np.median(signal.direct_antenna_loss_db)),
                                 code_bias_m=float(np.median(signal.direct_antenna_code_bias_m)),
                                 phase_rad=float(np.median(signal.direct_antenna_phase_rad)),
                                 antenna_doppler_hz=float(np.median(signal.direct_doppler_hz)),
                                 direct_visible=bool(signal.direct_visible[0]), reflected_visible=bool(signal.reflection_visible[0])))
        solution = solve_pseudorange_position(accepted, ranges, initial_position_m=[-25., 5., 0.]) if len(ranges) >= 4 else None
        error = solution.position_m-RECEIVER if solution is not None else None
        positions.append(dict(seed=seed, environment=environment, scenario=name, block=index, time_s=time_s,
                              stage=stage, wetness=float(response.wetness(time_s)[0]),
                              available=solution is not None, fresh_channels=len(ranges),
                              failure_reason=None if solution is not None else ("fewer_than_four_fresh_channels" if len(ranges) < 4 else "rank_or_convergence_failure"),
                              east_m=float(solution.position_m[0]) if solution is not None else None,
                              north_m=float(solution.position_m[1]) if solution is not None else None,
                              up_m=float(solution.position_m[2]) if solution is not None else None,
                              clock_bias_m=solution.clock_bias_m if solution is not None else None,
                              horizontal_error_m=float(np.linalg.norm(error[:2])) if error is not None else None,
                              position_error_m=float(np.linalg.norm(error)) if error is not None else None,
                              vertical_error_m=float(error[2]) if error is not None else None))
    summary = dict(seed=seed, environment=environment, scenario=name, **recovery_metrics(positions),
                   channel_lost_events=sum(row["status"] == "lost" for row in channels),
                   channel_acquired_events=sum(row["status"] == "acquired" for row in channels),
                   channel_measurements_used=sum(row["used"] for row in channels),
                   channel_observations=len(channels))
    return dict(positions=positions, channels=channels, summary=summary)


def stage_summary(rows):
    used = [row for row in rows if row["available"]]
    paired = [row for row in rows if row["paired_up_shift_m"] is not None]
    return dict(blocks=len(rows), available=len(used), unavailable=len(rows)-len(used),
                horizontal_p95_m_conditional=float(np.quantile([row["horizontal_error_m"] for row in used], .95)) if used else None,
                position_max_m_conditional=max((row["position_error_m"] for row in used), default=None),
                available_error_over_100_m=sum(row["position_error_m"] > 100 for row in used),
                paired_blocks=len(paired),
                paired_up_p50_m=float(np.median([row["paired_up_shift_m"] for row in paired])) if paired else None,
                paired_clock_p50_m=float(np.median([row["paired_clock_shift_m"] for row in paired])) if paired else None)


def run_experiment(*, workers=1) -> dict[str, Any]:
    workers = positive_int("workers", workers)
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
    try:
        calibration = list(pool.map(_calibration, range(1, 13))) if pool else list(map(_calibration, range(1, 13)))
        gates = {prn:gate for prn, gate, _ in calibration}
        jobs = [(seed, environment, name, gates) for seed in SEEDS for environment in ("open", "wall") for name in RESPONSES]
        cases = list(pool.map(_case, jobs)) if pool else list(map(_case, jobs))
    finally:
        if pool:
            pool.shutdown()
    positions = [row for case in cases for row in case["positions"]]
    baseline = {(row["seed"], row["environment"], row["block"]):row for row in positions if row["scenario"] == "dry"}
    for row in positions:
        dry = baseline[row["seed"], row["environment"], row["block"]]
        pair = row["available"] and dry["available"]
        row["paired_up_shift_m"] = row["up_m"]-dry["up_m"] if pair else None
        row["paired_clock_shift_m"] = row["clock_bias_m"]-dry["clock_bias_m"] if pair else None
    stages = []
    for case in cases:
        for stage in ("initial_dry", "wetting", "wet", "drying", "dry"):
            rows = [row for row in case["positions"] if row["stage"] == stage]
            stages.append(dict(seed=case["summary"]["seed"], environment=case["summary"]["environment"],
                               scenario=case["summary"]["scenario"], stage=stage, **stage_summary(rows)))
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_cpu_wet_antenna.py", "experiments/run_cpu_rain_positioning.py",
               "experiments/run_cpu_multipath_tracking.py", "python/gnss_gpu/wet_antenna_cpu.py",
               "python/gnss_gpu/wall_signal_sim.py", "python/gnss_gpu/wet_delay_cpu.py", "python/gnss_gpu/rain_cpu.py",
               "python/gnss_gpu/fresnel.py", "python/gnss_gpu/positioning_cpu.py", "python/gnss_gpu/tracking_cpu.py",
               "python/gnss_gpu/acquisition_cpu.py", "python/gnss_gpu/code_delay_cpu.py", "python/gnss_gpu/code_detection_cpu.py",
               "python/gnss_gpu/signal_sim_cpu.py"]
    metadata = dict(seeds=list(SEEDS), noise_domain=NOISE_DOMAIN, blocks=BLOCKS, sampling_freq_hz=FS,
                    block_samples=COUNT, workers=workers, nominal_cn0_db_hz=55.,
                    responses={name:asdict(response) for name, response in RESPONSES.items()},
                    thresholds={str(prn):asdict(gate) for prn, gate in gates.items()},
                    calibration_seed=13579, calibration_trials_per_prn=199,
                    constellation_m=constellation().tolist(), receiver_m=RECEIVER.tolist(), reference_range_m=REFERENCE,
                    wall_bounds_m=[0., -100., 100., 0., 100.], wall_antenna_lhcp_voltage=[0., .1],
                    response_basis="loss/code bias/phase intercept independently prescribed; q=1-clipped_up_component, no rainfall law or EM fit",
                    time_interpretation="0.5 s compressed diagnostic replay; wet 50-100 ms, plateau 100-200 ms, dry 200-400 ms. NOT measured wetting/drying times.",
                    receiver="Known nominal zero carrier; fixed single-bin PRN-specific gate, default two-block code confirmation, gain .25/coast 3; no PLL/carrier phase navigation. Antenna phase transients induce a few Hz not searched by this experiment.",
                    pvt_policy="Only fresh measurement_used channel states; predictions during coasting are excluded; equal-weight position/clock LS, initial [-25,5,0], known 22 Mm code-period reference; no truth-dependent rejection",
                    pairing="Same standardized circular Gaussian noise per seed/PRN/block, rotated by direct antenna phase at each sample for variance reduction; noise power remains fixed AFTER loss. Not physical noise tracking or receiver truth aiding.",
                    recovery="First five consecutive available positions after drying begins, decision at end of fifth block; only if wet plateau contained a loss; no-loss/not-recovered remain distinct",
                    limitations="Two seeds, one static synthetic geometry per environment, isolated PRN channels, no empirical wet antenna/radome calibration, thermal-noise change, rainfall-to-response relation, full EM filter, orbit or real RF validation. 6/30 dB and .3 m/radian levels are independent sensitivity/stress controls. Serial blocks are correlated through receiver state.",
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    return dict(positions=positions, channels=[row for case in cases for row in case["channels"]],
                summary=[case["summary"] for case in cases], stages=stages, metadata=metadata,
                calibration=[dict(prn=prn, trial=i, statistic=score) for prn, _, scores in calibration for i, score in enumerate(scores)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="output/cpu-wet-antenna")
    args = parser.parse_args()
    result = run_experiment(workers=args.workers)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("positions", "channels", "summary", "stages", "calibration"):
        with (output/f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    (output/"metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for mode in ("dry", "loss_6db", "loss_30db_stress", "combined_stress"):
        rows = [row for row in result["positions"] if row["seed"] == SEEDS[0] and row["environment"] == "open" and row["scenario"] == mode]
        t = [row["time_s"] for row in rows]
        axes[0].plot(t, [row["fresh_channels"] for row in rows], label=mode)
    axes[0].axhline(4, color="black", linestyle="--", label="minimum four fresh channels")
    axes[0].set(ylabel="fresh channels", title="Wetting/drying replay, open space, seed 24683; nominal 55 dB-Hz")
    for mode in ("common_code_0.3m", "directional_code_0.3m", "phase_pattern"):
        rows = [row for row in result["positions"] if row["seed"] == SEEDS[0] and row["environment"] == "open" and row["scenario"] == mode]
        axes[1].plot([row["time_s"] for row in rows], [row["paired_up_shift_m"] for row in rows], label=mode)
        axes[2].plot([row["time_s"] for row in rows], [row["paired_clock_shift_m"] for row in rows], label=mode)
    axes[1].set(ylabel="paired height shift [m]", title="Effect size versus dry receiver, common carrier-frame noise")
    axes[2].set(ylabel="paired receiver-clock shift [m]", title="Common delay primarily enters receiver clock; no carrier-phase PVT")
    for axis in axes:
        axis.axvspan(.05, .2, color="blue", alpha=.08)
        axis.axvspan(.2, .4, color="orange", alpha=.08)
        axis.set_xlabel("time [s]; blue=wetting/plateau, orange=drying")
        axis.grid(alpha=.2)
        axis.legend()
    figure.savefig(output/"wet_antenna.png", dpi=150)
    plt.close(figure)
    for row in result["summary"]:
        print(row)
    print(f"Saved complete trajectories, losses/recovery, unavailable positions and settings to {output}")


if __name__ == "__main__":
    main()
