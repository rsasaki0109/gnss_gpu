"""Compare single-/two-path delay fits without providing geometry to the receiver.

PYTHONPATH=python:. python experiments/run_cpu_two_path_fit.py --workers 4
Samples 20 short blocks on the previous moving-wall route, at two sample rates.
This is per-block estimation, not insertion of a corrected delay into tracking.
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
from experiments.run_cpu_multipath_tracking import (
    RECEIVER, VELOCITY, TRANSMITTER, REFERENCE_RANGE, _worker_init,
)
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, calibrate_ca_search, detect_ca_search
from gnss_gpu.input_validation import positive_int
from gnss_gpu.multipath_fit_cpu import fit_ca_two_paths
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.tracking_cpu import refine_ca_carrier
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal

CALIBRATION_SEED, CALIBRATION_DOMAIN = 20261031, 0x32504341
SEEDS, SIGNAL_DOMAIN = (3101, 3102), 0x32505349
BLOCK_INDICES = tuple(range(0, 400, 20))


def summarize_rows(rows):
    detected = [row for row in rows if row["detection_accepted"]]
    baseline = [row for row in detected if row["single_error_m"] is not None]
    selected = [row for row in detected if row["selected_error_m"] is not None]
    corrected = [row for row in selected if row["selected_model"] == "two_path"]
    def rmse(items, key):
        return float(np.sqrt(np.mean([row[key]**2 for row in items]))) if items else None

    return dict(blocks=len(rows), detected_blocks=len(detected), rejected_blocks=len(rows)-len(detected),
                selected_delay_blocks=len(selected), missing_delay_blocks=len(rows)-len(selected),
                two_path_blocks=len(corrected),
                single_path_blocks=sum(row["selected_model"] == "single_path" for row in rows),
                ambiguous_blocks=sum(row["selected_model"] == "ambiguous" for row in rows),
                baseline_delay_blocks=len(baseline),
                single_rmse_m_conditional_baseline=rmse(baseline, "single_error_m"),
                single_rmse_m_same_selected_blocks=rmse(selected, "single_error_m"),
                selected_rmse_m_conditional=rmse(selected, "selected_error_m"),
                single_rmse_m_same_corrected_blocks=rmse(corrected, "single_error_m"),
                two_path_rmse_m_conditional=rmse(corrected, "selected_error_m"),
                selected_delay_error_over_30_m=sum(abs(row["selected_error_m"]) > 30 for row in selected))


def _case(job):
    spc, seed, cn0, gate = job
    fs, count = spc*CHIP_RATE, 1023*spc
    simulator = CpuSignalSimulator(fs, noise_floor_db=-10000)
    wall = VerticalWall(0., -30., 5., 0., 100.)
    rows = []
    for block_index in BLOCK_INDICES:
        start = block_index*count
        if cn0 is None:
            noise = np.zeros(2*count)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, SIGNAL_DOMAIN, spc, block_index]))
            noise = rng.normal(0., np.sqrt(fs/(2*10**(cn0/10))), 2*count)
        for mode in ("direct", "multipath"):
            block = simulate_wall_signal(
                simulator, wall, RECEIVER, VELOCITY, TRANSMITTER, count,
                start_sample=start, reference_range_m=REFERENCE_RANGE,
                nav_bit=-1 if (block_index//20) % 2 else 1,
                include_reflection=mode == "multipath", integrate_samples=True)
            iq = block.iq+noise
            detection = detect_ca_search(iq, gate)
            fitted = None
            carrier = None
            if detection.accepted and detection.candidate is not None:
                carrier = refine_ca_carrier(iq, gate.search, detection.candidate)
                if carrier is not None:
                    fitted = fit_ca_two_paths(iq, fs, carrier_hz=carrier.carrier_hz,
                                              max_delay_std_samples=5.*fs/C_LIGHT)
            truth = float(((np.median(block.direct_range_m)-REFERENCE_RANGE)/C_LIGHT*fs-start+count/2) % count-count/2)
            def error(delay):
                return None if delay is None else float(((delay-truth+count/2) % count-count/2)*C_LIGHT/fs)
            rows.append(dict(samples_per_chip=spc, seed=seed, cn0_reference_db_hz=cn0, mode=mode,
                             block=block_index, time_s=start/fs, detection_accepted=detection.accepted,
                             detection_statistic=detection.statistic,
                             acquired_carrier_hz=carrier.carrier_hz if carrier else None,
                             selected_model=fitted.selected_model if fitted else "no_fit",
                             reason=fitted.reason if fitted else "coarse detection rejected or no fit",
                             single_error_m=error(carrier.delay.delay_samples) if carrier else None,
                             selected_error_m=error(fitted.delay_samples) if fitted else None,
                             single_delay_samples=carrier.delay.delay_samples if carrier else None,
                             first_delay_samples=fitted.first_delay_samples if fitted else None,
                             second_delay_samples=fitted.second_delay_samples if fitted else None,
                             bic_improvement=fitted.bic_improvement if fitted else None,
                             condition_number=fitted.delay_condition_number if fitted else None,
                             local_delay_std_m=fitted.first_delay_std_samples*C_LIGHT/fs
                             if fitted and fitted.first_delay_std_samples is not None else None,
                             single_residual_fraction=fitted.single.fit_residual_fraction if fitted else None,
                             two_path_residual_fraction=fitted.fit_residual_fraction if fitted else None,
                             geometric_direct_delay_samples=truth,
                             geometric_excess_m=float(np.median(block.reflected_range_m-block.direct_range_m)),
                             reflection_visible_fraction=float(np.mean(block.reflection_visible))))
    summary = [dict(samples_per_chip=spc, seed=seed, cn0_reference_db_hz=cn0, mode=mode,
                    **summarize_rows([row for row in rows if row["mode"] == mode])) for mode in ("direct", "multipath")]
    return dict(rows=rows, summary=summary)


def run_experiment(*, workers=1) -> dict[str, Any]:
    workers = positive_int("workers", workers)
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
    gates, calibration = {}, []
    try:
        for spc in (2, 20):
            search = CaSearch(spc*CHIP_RATE, (-1000., -500., 0., 500., 1000.))
            jobs = [(CALIBRATION_SEED+spc, CALIBRATION_DOMAIN, index, search) for index in range(199)]
            scores = list(pool.map(_null_job, jobs, chunksize=4)) if pool else list(map(_null_job, jobs))
            gates[spc] = calibrate_ca_search([row["statistic"] for row in scores], search)
            calibration.extend([dict(samples_per_chip=spc, **row) for row in scores])
        # All gates and fit settings are fixed before evaluation.
        jobs = [(spc, seed, cn0, gates[spc]) for spc in (2, 20)
                for seed, cn0 in [(0, None)]+[(seed, cn0) for seed in SEEDS for cn0 in (55., 65., 75.)]]
        cases = list(pool.map(_case, jobs)) if pool else list(map(_case, jobs))
    finally:
        if pool:
            pool.shutdown()
    root = Path(__file__).resolve().parents[1]
    sources = ["experiments/run_cpu_two_path_fit.py", "experiments/run_cpu_multipath_tracking.py",
               "experiments/run_cpu_carrier_search.py", "python/gnss_gpu/multipath_fit_cpu.py",
               "python/gnss_gpu/code_delay_cpu.py", "python/gnss_gpu/acquisition_cpu.py",
               "python/gnss_gpu/code_detection_cpu.py", "python/gnss_gpu/tracking_cpu.py",
               "python/gnss_gpu/wall_signal_sim.py", "python/gnss_gpu/signal_sim_cpu.py",
               "python/gnss_gpu/fresnel.py", "docs/gnss_signal_simulation_sources.json"]
    metadata = dict(evaluation_seeds=list(SEEDS), calibration_seed_base=CALIBRATION_SEED,
                    calibration_seed_rule="base + samples_per_chip", calibration_domain=CALIBRATION_DOMAIN,
                    signal_domain=SIGNAL_DOMAIN, thresholds={str(spc):asdict(gate) for spc, gate in gates.items()},
                    workers=workers, block_indices=list(BLOCK_INDICES), samples_per_chip=[2, 20],
                    cn0_reference_db_hz=[None, 55., 65., 75.],
                    receiver_m=RECEIVER.tolist(), velocity_mps=VELOCITY.tolist(), transmitter_m=TRANSMITTER.tolist(),
                    reference_range_m=REFERENCE_RANGE, wall_bounds_m=[0., -30., 5., 0., 100.], material="concrete",
                    antenna_lhcp_response=[0., 0.], fit_settings=dict(search_radius_chips=.5, max_excess_chips=1.,
                                  starts=8, max_iterations=20, max_local_delay_std_m=5.),
                    pairing="Identical IQ for one-/two-path estimators; same noise for direct/multipath and standardized noise across C/N0 per seed/rate. Different sample rates have independent noise. One deterministic noiseless control per rate.",
                    assumptions="Local LOS-dominant two-path prior, common acquired carrier, constant complex amplitudes and code delays during 1 ms; synthetic centered-aperture IQ; ideal specular wall and block-aligned 50 bit/s data",
                    limitations="Sparse 20-block snapshot diagnostic; no corrected delay inserted into tracker. Common-carrier approximation ignores differential path Doppler and code drift. BIC, local rank/variance are diagnostics, not detection calibration, global identifiability or integrity guarantees. Missing results and selected single-path fallbacks remain in counts. Two seeds and one route, no RF/PVT validation.",
                    paper_scope="Geometry/polarization from previously reviewed preprint Eq. (8) and 2014 author supplements. This variable-projection receiver is our engineering implementation, not a reproduced paper algorithm.",
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    return dict(rows=[row for case in cases for row in case["rows"]],
                summary=[row for case in cases for row in case["summary"]], calibration=calibration, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="output/cpu-two-path-fit")
    args = parser.parse_args()
    result = run_experiment(workers=args.workers)
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

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    panels = [(2, None, 0), (20, None, 0), (20, 65., SEEDS[0]),
              (20, 75., SEEDS[0]), (2, 75., SEEDS[0]), (20, 75., SEEDS[1])]
    for axis, (spc, cn0, seed) in zip(axes.flat, panels):
        rows = [row for row in result["rows"] if row["samples_per_chip"] == spc and row["mode"] == "multipath"
                and row["cn0_reference_db_hz"] == cn0 and row["seed"] == seed]
        t = [row["time_s"] for row in rows]
        axis.plot(t, [row["single_error_m"] for row in rows], "o-", label="single-path estimate")
        axis.plot(t, [row["selected_error_m"] for row in rows], "s-", label="selected delay (gaps = ambiguous)")
        axis.axhline(0., color="black", linewidth=.5)
        axis.set(xlabel="block start time [s]", ylabel="delay error [m]",
                 title=f"{spc*CHIP_RATE/1e6:g} MHz, {'noiseless' if cn0 is None else f'{cn0:g} dB-Hz, seed {seed}'}")
        axis.legend()
        axis.grid(alpha=.2)
    figure.suptitle("Experimental two-path fits: improvements AND failures; per-block estimates, no tracking correction")
    figure.savefig(output/"two_path_fit.png", dpi=150)
    plt.close(figure)
    for row in result["summary"]:
        print(row)
    print(f"Saved all fit decisions, missing estimates, fixed settings and plot to {output}")


if __name__ == "__main__":
    main()
