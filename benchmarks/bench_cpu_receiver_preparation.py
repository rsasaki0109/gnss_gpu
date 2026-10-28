"""Compare complete saved and current receiver modules on frozen IQ.

Baseline modules are loaded with their original dependency bindings, then the
normal import registry is restored. Only the benchmark's receiver constructor
is selected between trials; production functions are not monkeypatched.
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

import experiments.run_cpu_batch_receiver as experiment
import gnss_gpu.acquisition_cpu as acquisition
import gnss_gpu.code_delay_cpu as delay
import gnss_gpu.tracking_cpu as tracking
from gnss_gpu.code_detection_cpu import CaDetectionThreshold
from gnss_gpu import CpuSignalSimulator


def load_baseline(directory):
    originals = {"gnss_gpu."+name: sys.modules["gnss_gpu."+name]
                 for name in ("code_delay_cpu", "acquisition_cpu", "tracking_cpu")}
    loaded = {}
    try:
        for index, qualified in enumerate(originals):
            name = qualified.rsplit(".", 1)[1]
            spec = importlib.util.spec_from_file_location("baseline_receiver_"+name, directory/(name+".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # dataclass introspection needs its defining module
            spec.loader.exec_module(module)
            sys.modules[qualified] = module
            loaded[name] = module
    finally:
        sys.modules.update(originals)
    return loaded


@contextmanager
def selected_tracker(tracker):
    previous = experiment.CpuCaTracker
    experiment.CpuCaTracker = tracker
    try:
        yield
    finally:
        experiment.CpuCaTracker = previous


def run(directory, frozen_dir, repeats):
    baseline = load_baseline(directory)
    metadata = json.loads((frozen_dir/"metadata.json").read_text())
    rows, starts, seeds, _ = experiment.scene(metadata["blocks"])
    assert seeds == metadata["per_block_noise_seeds"]
    threshold = acquisition.CaSearchThreshold(acquisition.CaSearch(**metadata["threshold"]["search"]),
                     CaDetectionThreshold(**metadata["threshold"]["calibration"]))
    iq = experiment.render("batch", rows, starts, seeds, metadata["noise_floor_db"])
    assert hashlib.sha256(iq.tobytes()).hexdigest() == metadata["iq_sha256"]
    frozen = json.loads((frozen_dir/"receiver_trace.json").read_text(),
                object_hook=lambda d: complex(d["real"], d["imag"]) if set(d) == {"real", "imag"} else d)
    simulator = CpuSignalSimulator(experiment.FS, noise_floor_db=metadata["noise_floor_db"])
    for index, expected in enumerate(metadata["calibration_scores"]):
        simulator.noise_seed = experiment.noise_seed(experiment.CALIBRATION_DOMAIN, index)
        noise = simulator.generate_epoch([], experiment.COUNT)
        for module in (baseline["acquisition_cpu"], acquisition):
            assert module.acquire_ca_signal(noise, threshold.search).statistic == expected
    # Warm both paths, verifying every result before recording any timings.
    trackers = {"before": baseline["tracking_cpu"].CpuCaTracker, "after": tracking.CpuCaTracker}
    for tracker in trackers.values():
        with selected_tracker(tracker):
            assert [asdict(x) for x in experiment.receive(iq, threshold, starts)] == frozen
    elapsed = []
    for repeat in range(repeats):
        for version in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
            with selected_tracker(trackers[version]):
                begin = time.perf_counter()
                generated_iq = experiment.render("batch", rows, starts, seeds, metadata["noise_floor_db"])
                generated = time.perf_counter()
                trace = experiment.receive(generated_iq, threshold, starts)
                finished = time.perf_counter()
            elapsed.append(dict(repeat=repeat, version=version, generation_s=generated-begin,
                                receiver_s=finished-generated, total_s=finished-begin))
            np.testing.assert_array_equal(generated_iq, iq)
            assert [asdict(x) for x in trace] == frozen
    medians = {v: {key: statistics.median(row[key] for row in elapsed if row["version"] == v)
                  for key in ("generation_s", "receiver_s", "total_s")} for v in trackers}
    def hashes(paths):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(baseline_sha256=hashes(directory/(name+".py") for name in baseline),
                candidate_sha256=hashes(Path(m.__file__) for m in (delay, acquisition, tracking)),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                frozen_metadata_sha256=hashlib.sha256((frozen_dir/"metadata.json").read_bytes()).hexdigest(),
                iq_sha256=metadata["iq_sha256"], numpy=np.__version__, blocks=len(rows), repeats=repeats,
                calibration_trials=len(metadata["calibration_scores"]), frozen_calibration_scores_equal=True,
                full_receiver_results_equal=True, seconds=elapsed, median_seconds=medians,
                receiver_speedup=medians["before"]["receiver_s"]/medians["after"]["receiver_s"],
                pipeline_speedup=medians["before"]["total_s"]/medians["after"]["total_s"],
                limitations=["One synthetic recording, warm repeated measurements, not independent GNSS trials.",
                             "Computational equivalence does not establish RF accuracy or real-time operation."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, default=Path("output/cpu-batch-receiver"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-receiver-preparation-speed/comparison.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    report = run(args.baseline_dir, args.frozen_dir, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("full_receiver_results_equal", "frozen_calibration_scores_equal",
                                                 "median_seconds", "receiver_speedup", "pipeline_speedup")}, indent=2))
