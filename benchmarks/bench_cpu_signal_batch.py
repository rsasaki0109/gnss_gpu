"""Compare independent-epoch batching against a saved CPU implementation."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numba
import numpy as np

from gnss_gpu import CpuSignalSimulator
import gnss_gpu.signal_sim_cpu as current


def run(baseline_source, repeats=9):
    source = Path(baseline_source)
    spec = importlib.util.spec_from_file_location("cpu_signal_batch_baseline", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results, original_threads = [], numba.get_num_threads()
    try:
        for name, epochs, samples, noise in (("short", 128, 2600, False),
                                             ("short_seeded_noise", 128, 2600, True),
                                             ("ragged", 128, 2600, False),
                                             ("long", 8, 26000, False),
                                             ("one_epoch", 1, 2600, False)):
            rows = []
            for epoch in range(epochs):
                count = (0, 1, 4, 12, 32)[epoch % 5] if name == "ragged" else 12
                rows.append([dict(prn=(ch+epoch) % 32+1, code_phase=epoch*.01-.3,
                                  carrier_phase=.7+ch, doppler_hz=(-1.)**ch*(ch+1)*127.,
                                  amplitude=1/(1+ch), nav_bit=(-1.)**epoch,
                                  **({"nav_bits":[1, -1]*4} if ch % 3 == 1 else {})) for ch in range(count)])
            seeds = [31+epoch for epoch in range(epochs)]
            simulators = {"before":module.CpuSignalSimulator(noise_floor_db=-20 if noise else -10000),
                          "after":CpuSignalSimulator(noise_floor_db=-20 if noise else -10000)}
            serial = None
            for threads in dict.fromkeys((1, min(4, numba.config.NUMBA_NUM_THREADS))):
                numba.set_num_threads(threads)
                for simulator in simulators.values():
                    simulator.generate_epochs(rows[:2], 64, noise_seeds=seeds[:2])
                before = simulators["before"].generate_epochs(rows, samples, noise_seeds=seeds)
                after = simulators["after"].generate_epochs(rows, samples, noise_seeds=seeds)
                np.testing.assert_array_equal(before, after)
                if serial is not None:
                    np.testing.assert_array_equal(after, serial)
                serial = after
                elapsed = {key:[] for key in simulators}
                for repeat in range(repeats):
                    for key in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
                        start = time.perf_counter()
                        simulators[key].generate_epochs(rows, samples, noise_seeds=seeds)
                        elapsed[key].append(time.perf_counter()-start)
                medians = {key:statistics.median(values) for key, values in elapsed.items()}
                results.append(dict(case=name, epochs=epochs, samples_per_epoch=samples, threads=threads,
                                    noise=noise, seconds=elapsed, median_seconds=medians,
                                    speedup=medians["before"]/medians["after"], outputs_bitwise_equal=True))
    finally:
        numba.set_num_threads(original_threads)
    return dict(baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                numba=numba.__version__, numpy=np.__version__, repeats=repeats,
                timing_scope="validation, packing, synthesis, per-epoch noise and output assembly; JIT excluded",
                results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-batch-speed/comparison.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    report = run(args.baseline_source, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps([{k:r[k] for k in ("case", "threads", "median_seconds", "speedup", "outputs_bitwise_equal")}
                      for r in report["results"]], indent=2))
