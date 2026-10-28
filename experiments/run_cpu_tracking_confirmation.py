"""Paired comparison of one- and two-block acquisition confirmation.

PYTHONPATH=python:. python experiments/run_cpu_tracking_confirmation.py --workers 4
The old seed is a regression case; two new seeds are evaluation cases. All
configurations are fixed before evaluation; no outcome-dependent threshold fit.
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
from experiments.run_cpu_tracking import CALIBRATION_DOMAIN, SEED, run_experiment
from gnss_gpu.acquisition_cpu import CaSearch, calibrate_ca_search
from gnss_gpu.input_validation import positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE

EVALUATION_SEEDS = (20261026, 20261027)


def summarize_confirmation(rows):
    """All-block counts and decision latency, including confirmation waiting.

    Latency includes the acquisition block itself: acquisition on block zero
    takes 1 ms, on block one takes 2 ms. No acquisition gives None, never zero.
    """
    initial = next((row["block"] for row in rows
                    if row["block"] < 80 and row["status"] == "acquired"), None)
    returned = next((row["block"] for row in rows
                     if row["block"] >= 100 and row["status"] == "acquired"), None)
    present = [row for row in rows if not row["signal_outage"]]
    absent = [row for row in rows if row["signal_outage"]]
    used = [row for row in present if row["measurement_used"]]
    return dict(signal_present_blocks=len(present), signal_absent_blocks=len(absent),
                used_signal_present_blocks=len(used),
                unused_signal_present_blocks=len(present)-len(used),
                used_signal_absent_blocks=sum(row["measurement_used"] for row in absent),
                false_acquisition_events=sum(row["status"] == "acquired" for row in absent),
                pending_signal_absent_blocks=sum(row["status"] == "confirming" for row in absent),
                coarse_gate_passed_signal_absent_blocks=sum(row["detection_accepted"] for row in absent),
                initial_acquisition_latency_ms=None if initial is None else initial+1,
                reacquisition_latency_ms=None if returned is None else returned-100+1,
                used_signal_present_delay_error_over_30_m=sum(abs(row["tracked_delay_error_m"]) > 30 for row in used),
                used_signal_present_carrier_error_over_500_hz=sum(abs(row["tracked_error_hz"]) > 500 for row in used))


def _case(job):
    seed, cn0, calibration = job
    cases = []
    for confirmation in (1, 2):
        result = run_experiment(seed=seed, cn0_db_hz=cn0, confirmation_blocks=confirmation,
                                calibration=calibration)
        summary = dict(seed=seed, seed_role="regression" if seed == SEED else "evaluation",
                       cn0_db_hz=cn0, confirmation_blocks=confirmation,
                       **summarize_confirmation(result["rows"]))
        for row in result["rows"]:
            row.update(seed=seed, cn0_db_hz=cn0, confirmation_blocks=confirmation)
        cases.append(dict(rows=result["rows"], summary=summary, metadata=result["metadata"]))
    return cases


def run_comparison(*, workers=1) -> dict[str, Any]:
    workers = positive_int("workers", workers)
    search = CaSearch(2*CHIP_RATE, tuple(np.arange(-6000., 6001., 500.)))
    calibration_jobs = [(SEED, CALIBRATION_DOMAIN, trial, search) for trial in range(199)]
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        calibration = list(pool.map(_null_job, calibration_jobs, chunksize=4)) if pool else list(map(_null_job, calibration_jobs))
        # Freeze before computing ANY signal or outage decision.
        threshold = calibrate_ca_search([row["statistic"] for row in calibration], search)
        jobs = [(seed, cn0, calibration) for seed in (SEED, *EVALUATION_SEEDS) for cn0 in (45., 55.)]
        paired = list(pool.map(_case, jobs)) if pool else list(map(_case, jobs))
    finally:
        if pool:
            pool.shutdown()
    cases = [case for pair in paired for case in pair]
    rows = [row for case in cases for row in case["rows"]]
    root = Path(__file__).resolve().parents[1]
    metadata: dict[str, Any] = dict(calibration_seed=SEED, regression_seed=SEED, evaluation_seeds=list(EVALUATION_SEEDS),
                    cn0_db_hz=[45., 55.], confirmation_blocks=[1, 2],
                    threshold=cases[0]["metadata"]["threshold"], workers=workers,
                    pairing="Identical IQ/noise and same frozen coarse threshold for both modes; C/N0 settings reuse standardized noise. Seed 20261025 informed development and is regression-only.",
                    limitations="Two fresh random seeds, short outages, one geometry and Gaussian noise; no false-track probability guarantee. One-block detections are correlated through the search and state; no alpha-squared claim. Confirmation parameters are engineering choices, not a reproduction of the cited paper.",
                    case_metadata=[case["metadata"] for case in cases],
                    source_sha256=dict(cases[0]["metadata"]["source_sha256"]))
    if any(case["metadata"]["threshold"] != asdict(threshold) for case in cases):
        raise RuntimeError("evaluation did not use the frozen calibration")
    metadata["source_sha256"]["experiments/run_cpu_tracking_confirmation.py"] = hashlib.sha256(
        (root/"experiments/run_cpu_tracking_confirmation.py").read_bytes()).hexdigest()
    return dict(rows=rows, summary=[case["summary"] for case in cases],
                calibration=calibration, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="output/cpu-tracking-confirmation")
    args = parser.parse_args()
    result = run_comparison(workers=args.workers)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("rows", "summary", "calibration"):
        with (output/f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    (output/"metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    summary = result["summary"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    labels = [f"{seed} ({'regression' if seed == SEED else 'evaluation'}), {cn0:g} dB-Hz"
              for seed in (SEED, *EVALUATION_SEEDS) for cn0 in (45., 55.)]
    for mode, offset in ((1, -.18), (2, .18)):
        selected = [row for row in summary if row["confirmation_blocks"] == mode]
        x = np.arange(len(selected))+offset
        axes[0].bar(x, [row["false_acquisition_events"] for row in selected], width=.36,
                    label=f"{mode}-block confirmation")
        axes[1].bar(x, [row["unused_signal_present_blocks"] for row in selected], width=.36,
                    label=f"{mode}-block confirmation")
        # NaN denotes no reacquisition and leaves a gap rather than a zero bar.
        axes[2].bar(x, [np.nan if row["reacquisition_latency_ms"] is None else row["reacquisition_latency_ms"]
                        for row in selected], width=.36, label=f"{mode}-block confirmation")
    for axis in axes:
        axis.set(xticks=np.arange(len(labels)), xticklabels=labels)
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=.2)
        axis.legend()
    axes[0].set(ylabel="false acquisitions / 20 absent blocks", title="Paired fixed-threshold comparison; no track-level false-alarm guarantee")
    axes[1].set(ylabel="unused / 180 signal-present blocks")
    axes[2].set(ylabel="reacquisition decision latency [ms]")
    figure.savefig(output/"confirmation.png", dpi=150)
    plt.close(figure)
    for row in summary:
        print(row)
    print(f"Saved paired blocks, threshold, parameters and plot to {output}")


if __name__ == "__main__":
    main()
