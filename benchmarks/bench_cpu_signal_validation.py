"""Compare CPU short-block overhead with a saved pre-change source file.

Run with PYTHONPATH=python:. and --baseline-source pointing to a saved copy of
signal_sim_cpu.py. Both versions run in one process with alternating order;
first-call JIT and waveform equality checks are outside timed intervals.
"""

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


def run(baseline_source, repeats=9, threads=4):
    path = Path(baseline_source)
    spec = importlib.util.spec_from_file_location("cpu_signal_validation_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    simulators = {"before":module.CpuSignalSimulator(noise_floor_db=-10000),
                  "after":CpuSignalSimulator(noise_floor_db=-10000)}
    cases = [("one_channel_1ms", 1, 2600, 100, False),
             ("twelve_channels_1ms", 12, 2600, 100, False),
             ("twelve_channels_1ms_nav", 12, 2600, 100, True),
             ("twelve_channels_100ms", 12, 260000, 1, False)]
    results = []
    original_threads = numba.get_num_threads()
    try:
        for name, channels, samples, blocks, navigation in cases:
            parameters = [dict(prn=p, code_phase=.3, carrier_phase=.4, doppler_hz=p*231.,
                               amplitude=1/p, nav_bit=1, **({"nav_bits":[1, -1]*4} if navigation else {}))
                          for p in range(1, channels+1)]

            def render(simulator):
                return np.concatenate([simulator.generate_epoch(parameters, samples, start_sample=i*samples)
                                       for i in range(blocks)])

            serial = None
            for workers in dict.fromkeys([1, threads]):
                numba.set_num_threads(workers)
                for sim in simulators.values():
                    sim.generate_epoch(parameters, 100)  # JIT + worker startup
                before, after = render(simulators["before"]), render(simulators["after"])
                np.testing.assert_array_equal(before, after)
                if serial is None:
                    serial = after
                else:
                    np.testing.assert_array_equal(after, serial)
                timings = {key:[] for key in simulators}
                for repeat in range(repeats):
                    order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
                    for key in order:
                        start = time.perf_counter()
                        render(simulators[key])
                        timings[key].append(time.perf_counter()-start)
                medians = {key:statistics.median(values) for key, values in timings.items()}
                results.append(dict(case=name, channels=channels, samples_per_call=samples,
                                    calls_per_trial=blocks, threads=workers, samples_per_trial=samples*blocks,
                                    signal_duration_s=samples*blocks/2.6e6,
                                    seconds=timings, median_seconds=medians,
                                    speedup=medians["before"]/medians["after"], outputs_bitwise_equal=True))
    finally:
        numba.set_num_threads(original_threads)
    report = dict(baseline_source=str(path), baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  numba=numba.__version__, numpy=np.__version__, repeats=repeats, noise="disabled",
                  timing_scope="channel validation/packing, synthesis, output checks and concatenation; JIT excluded",
                  results=results)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--threads", type=int, default=min(4, numba.config.NUMBA_NUM_THREADS))
    parser.add_argument("--output", type=Path, default=Path("output/cpu-signal-speed/comparison.json"))
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.threads <= numba.config.NUMBA_NUM_THREADS:
        parser.error("positive repeats and configured thread count required")
    result = run(args.baseline_source, args.repeats, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps([{k:r[k] for k in ("case", "threads", "median_seconds", "speedup", "outputs_bitwise_equal")}
                      for r in result["results"]], indent=2))
