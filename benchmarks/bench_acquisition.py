"""Benchmark GPU-accelerated GNSS signal acquisition."""

import time
import numpy as np


def benchmark_acquisition(n_prn_list=None, duration_list=None, n_iter=5):
    """Benchmark acquisition varying PRN count and signal duration.

    Parameters
    ----------
    n_prn_list : list of int
        Number of PRNs to search.
    duration_list : list of float
        Signal durations in seconds (e.g., 1e-3 for 1 ms).
    n_iter : int
        Number of iterations for timing.
    """
    if n_prn_list is None:
        n_prn_list = [1, 8, 16, 32]
    if duration_list is None:
        duration_list = [1e-3, 10e-3, 100e-3]

    try:
        from gnss_gpu import Acquisition
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] Acquisition not available: {e}")
        return {}

    sampling_freq = 4.092e6  # 4x chip rate
    results = {}

    print("=" * 90)
    print("Signal Acquisition Benchmark")
    print("=" * 90)
    header = (f"{'Duration':>10} | {'N Samples':>10} | {'N PRNs':>8} | "
              f"{'Mean (ms)':>12} | {'Std (ms)':>10} | {'Throughput':>16}")
    print(header)
    print("-" * 90)

    # Generate a test signal once for warm-up
    acq = Acquisition(sampling_freq=sampling_freq, doppler_range=5000, doppler_step=500)
    warmup_signal = Acquisition.generate_test_signal(
        prn=1, code_phase=100, doppler=1500.0, snr_db=15.0,
        sampling_freq=sampling_freq, duration_s=1e-3)
    acq.acquire(warmup_signal, prn_list=[1])

    for dur in duration_list:
        n_samples = int(sampling_freq * dur)
        dur_label = f"{dur * 1e3:.0f} ms"

        # Generate composite test signal with multiple PRNs embedded
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(n_samples).astype(np.float32)
        # Embed a few PRNs so there is real work
        for prn in [1, 5, 10, 20]:
            try:
                s = Acquisition.generate_test_signal(
                    prn=prn, code_phase=rng.integers(0, 4092),
                    doppler=rng.uniform(-4000, 4000), snr_db=12.0,
                    sampling_freq=sampling_freq, duration_s=dur)
                signal[:len(s)] += s[:len(signal)]
            except Exception:
                pass

        for n_prn in n_prn_list:
            prn_list = list(range(1, n_prn + 1))

            # Warm-up
            try:
                acq.acquire(signal, prn_list=prn_list)
            except Exception:
                pass

            times = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                acq.acquire(signal, prn_list=prn_list)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            mean_t = np.mean(times)
            std_t = np.std(times)
            throughput = n_prn / mean_t

            if throughput >= 1e3:
                tp_str = f"{throughput / 1e3:.2f} K PRN/s"
            else:
                tp_str = f"{throughput:.1f} PRN/s"

            print(f"{dur_label:>10} | {n_samples:>10,} | {n_prn:>8} | "
                  f"{mean_t * 1e3:>12.3f} | {std_t * 1e3:>10.3f} | {tp_str:>16}")

            results[(dur_label, n_prn)] = {
                "mean_s": mean_t,
                "std_s": std_t,
                "n_samples": n_samples,
                "throughput_prn_s": throughput,
            }

        print("-" * 90)

    return results


if __name__ == "__main__":
    benchmark_acquisition()
