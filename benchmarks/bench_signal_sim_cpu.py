"""Repeatable CPU IQ benchmark; timings exclude first-call JIT compilation.

Run from the repository root:
    PYTHONPATH=python python benchmarks/bench_signal_sim_cpu.py
"""

import argparse
import json
import statistics
import time

import numba
import numpy as np

from gnss_gpu import CpuSignalSimulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=260000)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--threads", type=int, default=min(4, numba.config.NUMBA_NUM_THREADS))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (args.samples < 1 or not 1 <= args.channels <= 32 or args.repeats < 1
            or not 1 <= args.threads <= numba.config.NUMBA_NUM_THREADS):
        parser.error("positive samples/repeats, 1..32 channels and available threads required")
    sim = CpuSignalSimulator(noise_floor_db=-10000)
    channels = [dict(prn=prn, code_phase=.3, carrier_phase=.4,
                     doppler_hz=prn*231., amplitude=1/prn, nav_bit=1)
                for prn in range(1, args.channels+1)]
    original_threads = numba.get_num_threads()
    reference = None
    timings = {}
    try:
        for threads in dict.fromkeys([1, args.threads]):
            numba.set_num_threads(threads)
            sim.generate_epoch(channels, 100)  # compile and warm worker pool
            elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                iq = sim.generate_epoch(channels, args.samples)
                elapsed.append(time.perf_counter()-start)
            if reference is None:
                reference = iq
            else:
                np.testing.assert_array_equal(iq, reference)
            timings[threads] = statistics.median(elapsed)
    finally:
        numba.set_num_threads(original_threads)
    print(json.dumps(dict(samples=args.samples, channels=args.channels,
                          signal_duration_s=args.samples/sim.sampling_freq,
                          median_seconds=timings,
                          speedup=timings[1]/timings[args.threads],
                          outputs_equal=True, numba=numba.__version__), indent=2))


if __name__ == "__main__":
    main()
