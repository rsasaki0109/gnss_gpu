"""Frozen-IQ full-receiver comparison against saved pre-cache delay fitting.

Only the fitter is switched; generation, threshold, receiver settings and IQ
remain fixed. Timing runs are sequential and alternate version order.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from experiments.run_cpu_batch_receiver import (scene, render, receive, json_default,
                                                noise_seed, CALIBRATION_DOMAIN, FS, COUNT)
from gnss_gpu import CpuSignalSimulator
import gnss_gpu.acquisition_cpu as acquisition
import gnss_gpu.tracking_cpu as tracking
import gnss_gpu.code_delay_cpu as current
from gnss_gpu.code_detection_cpu import CaDetectionThreshold


@contextmanager
def selected_fitter(fitter):
    # Keep this fitter-only benchmark usable after observation preparation was
    # hoisted into the search. Both paths here deliberately prepare every fit.
    modules = (acquisition, tracking)
    saved = [(module._prepare_ca_observation, module._estimate_prepared_delay)
             for module in modules]
    def prepare(iq, fs, *, prn):
        return iq, fs, prn
    def fit(prepared, *, carrier_hz):
        iq, fs, prn = prepared
        return fitter(iq, fs, prn=prn, carrier_hz=carrier_hz)
    for module in modules:
        module._prepare_ca_observation, module._estimate_prepared_delay = prepare, fit
    try:
        yield
    finally:
        for module, (prepare, fit) in zip(modules, saved):
            module._prepare_ca_observation, module._estimate_prepared_delay = prepare, fit


def trace_values(trace):
    return [asdict(row) for row in trace]


def run(baseline_source, frozen_dir, repeats=7):
    source, frozen_dir = Path(baseline_source), Path(frozen_dir)
    spec = importlib.util.spec_from_file_location("cpu_delay_cache_baseline", source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    metadata = json.loads((frozen_dir/"metadata.json").read_text())
    rows, starts, seeds, outage = scene(metadata["blocks"])
    if seeds != metadata["per_block_noise_seeds"]:
        raise ValueError("scene seeds differ from frozen recording")
    threshold = acquisition.CaSearchThreshold(acquisition.CaSearch(**metadata["threshold"]["search"]),
                                             CaDetectionThreshold(**metadata["threshold"]["calibration"]))
    noise_db = metadata["noise_floor_db"]
    iq = render("batch", rows, starts, seeds, noise_db)
    if hashlib.sha256(iq.tobytes()).hexdigest() != metadata["iq_sha256"]:
        raise ValueError("IQ differs from frozen integration recording")
    # Confirm frozen noise scores exactly, not merely threshold decisions.
    calibrator = CpuSignalSimulator(FS, noise_floor_db=noise_db)
    scores = []
    for index in range(len(metadata["calibration_scores"])):
        calibrator.noise_seed = noise_seed(CALIBRATION_DOMAIN, index)
        scores.append(acquisition.acquire_ca_signal(calibrator.generate_epoch([], COUNT), threshold.search).statistic)
    np.testing.assert_array_equal(scores, metadata["calibration_scores"])
    current.clear_ca_delay_cache()
    begin = time.perf_counter()
    candidate_trace = trace_values(receive(iq, threshold, starts))
    cold_seconds = time.perf_counter()-begin
    with selected_fitter(baseline.estimate_ca_delay):
        baseline_trace = trace_values(receive(iq, threshold, starts))
    assert baseline_trace == candidate_trace
    assert baseline_trace == json.loads((frozen_dir/"receiver_trace.json").read_text(),
                                        object_hook=lambda d: complex(d["real"], d["imag"]) if set(d) == {"real", "imag"} else d)
    elapsed = []
    for repeat in range(repeats):
        for version in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
            fitter = baseline.estimate_ca_delay if version == "before" else current.estimate_ca_delay
            with selected_fitter(fitter):
                start = time.perf_counter()
                generated_iq = render("batch", rows, starts, seeds, noise_db)
                generated = time.perf_counter()
                trace = receive(generated_iq, threshold, starts)
                finished = time.perf_counter()
            elapsed.append(dict(repeat=repeat, version=version, generation_s=generated-start,
                                receiver_s=finished-generated, total_s=finished-start))
            np.testing.assert_array_equal(generated_iq, iq)
            assert trace_values(trace) == baseline_trace
    medians = {version:{key:statistics.median(r[key] for r in elapsed if r["version"] == version)
                        for key in ("generation_s", "receiver_s", "total_s")} for version in ("before", "after")}
    return dict(baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                frozen_metadata_sha256=hashlib.sha256((frozen_dir/"metadata.json").read_bytes()).hexdigest(),
                iq_sha256=metadata["iq_sha256"], numpy=np.__version__, blocks=len(rows), repeats=repeats,
                full_receiver_results_equal=True, frozen_calibration_scores_equal=True,
                calibration_trials=len(scores), cold_candidate_receiver_s=cold_seconds,
                seconds=elapsed, median_seconds=medians,
                receiver_speedup=medians["before"]["receiver_s"]/medians["after"]["receiver_s"],
                pipeline_speedup=medians["before"]["total_s"]/medians["after"]["total_s"],
                cache_info=dict(reference=current._cached_reference_data.cache_info()._asdict(),
                                interval=current._cached_interval_data.cache_info()._asdict()),
                limitations=["One frozen synthetic recording; timing repetitions are not independent GNSS trials.",
                             "Warm-cache speed depends on reuse; oversized periods bypass caches.",
                             "Results establish computational equivalence, not real RF accuracy or integrity."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, default=Path("output/cpu-batch-receiver"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-receiver-speed/comparison.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    report = run(args.baseline_source, args.frozen_dir, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=json_default)+"\n")
    print(json.dumps({k:report[k] for k in ("full_receiver_results_equal", "frozen_calibration_scores_equal", "median_seconds", "receiver_speedup", "pipeline_speedup", "cache_info")}, indent=2))
