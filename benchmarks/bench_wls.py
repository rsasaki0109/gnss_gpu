"""Benchmark WLS batch positioning: GPU batch vs CPU single-epoch loop."""

import time
import numpy as np


def _make_test_scenario():
    """Create Tokyo Station 8-satellite scenario."""
    true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
    true_cb = 3000.0

    sat_ecef = np.array([
        [-14985000.0, -3988000.0, 21474000.0],
        [-9575000.0, 15498000.0, 19457000.0],
        [7624000.0, -16218000.0, 19843000.0],
        [16305000.0, 12037000.0, 17183000.0],
        [-20889000.0, 13759000.0, 8291000.0],
        [5463000.0, 24413000.0, 8934000.0],
        [22169000.0, 3975000.0, 13781000.0],
        [-11527000.0, -19421000.0, 13682000.0],
    ])

    ranges = np.sqrt(np.sum((sat_ecef - true_pos) ** 2, axis=1))
    pseudoranges = ranges + true_cb
    weights = np.ones(len(sat_ecef))

    return sat_ecef, pseudoranges, weights, true_pos, true_cb


def benchmark_wls(n_epochs_list=None, n_iter=5):
    """Benchmark WLS batch vs single-epoch loop.

    Parameters
    ----------
    n_epochs_list : list of int
        Number of epochs to batch-process.
    n_iter : int
        Number of iterations for timing.
    """
    if n_epochs_list is None:
        n_epochs_list = [100, 1000, 10000, 100000]

    try:
        from gnss_gpu._gnss_gpu import wls_position, wls_batch
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] WLS module not available: {e}")
        return {}

    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()
    rng = np.random.default_rng(42)
    results = {}

    print("=" * 90)
    print("WLS Batch Positioning Benchmark")
    print("=" * 90)
    header = (f"{'N Epochs':>10} | {'GPU Batch (ms)':>16} | {'CPU Loop (ms)':>16} | "
              f"{'Speedup':>8} | {'Throughput':>14}")
    print(header)
    print("-" * 90)

    # Warm-up
    sat_b = np.tile(sat_ecef, (10, 1, 1))
    pr_b = np.tile(pseudoranges, (10, 1)) + rng.normal(0, 3.0, (10, len(pseudoranges)))
    w_b = np.tile(weights, (10, 1))
    wls_batch(sat_b, pr_b, w_b)
    wls_position(sat_ecef.flatten(), pseudoranges, weights)

    for n_epoch in n_epochs_list:
        sat_batch = np.tile(sat_ecef, (n_epoch, 1, 1))
        pr_batch = np.tile(pseudoranges, (n_epoch, 1))
        pr_batch = pr_batch + rng.normal(0, 3.0, pr_batch.shape)
        w_batch = np.tile(weights, (n_epoch, 1))

        # --- GPU batch ---
        gpu_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            wls_batch(sat_batch, pr_batch, w_batch)
            t1 = time.perf_counter()
            gpu_times.append(t1 - t0)
        gpu_mean = np.mean(gpu_times)
        gpu_std = np.std(gpu_times)

        # --- CPU single-epoch loop ---
        # Only run CPU loop for smaller sizes to avoid excessive wait
        if n_epoch <= 10000:
            cpu_times = []
            cpu_iter = max(1, n_iter // 2) if n_epoch >= 10000 else n_iter
            for _ in range(cpu_iter):
                t0 = time.perf_counter()
                for i in range(n_epoch):
                    wls_position(sat_batch[i].flatten(), pr_batch[i], w_batch[i])
                t1 = time.perf_counter()
                cpu_times.append(t1 - t0)
            cpu_mean = np.mean(cpu_times)
            cpu_std = np.std(cpu_times)
            speedup = cpu_mean / gpu_mean if gpu_mean > 0 else float('inf')
        else:
            # Estimate from 1000-epoch timing
            cpu_mean = float('nan')
            cpu_std = float('nan')
            speedup = float('nan')

        throughput = n_epoch / gpu_mean

        gpu_str = f"{gpu_mean * 1e3:>8.2f} +/- {gpu_std * 1e3:>5.2f}"
        if np.isnan(cpu_mean):
            cpu_str = f"{'(est. too slow)':>16}"
            speedup_str = f"{'N/A':>8}"
        else:
            cpu_str = f"{cpu_mean * 1e3:>8.2f} +/- {cpu_std * 1e3:>5.2f}"
            speedup_str = f"{speedup:>7.1f}x"

        if throughput >= 1e6:
            tp_str = f"{throughput / 1e6:.2f} M epoch/s"
        elif throughput >= 1e3:
            tp_str = f"{throughput / 1e3:.2f} K epoch/s"
        else:
            tp_str = f"{throughput:.1f} epoch/s"

        print(f"{n_epoch:>10,} | {gpu_str} | {cpu_str} | {speedup_str} | {tp_str:>14}")

        results[n_epoch] = {
            "gpu_mean_s": gpu_mean,
            "gpu_std_s": gpu_std,
            "cpu_mean_s": cpu_mean,
            "cpu_std_s": cpu_std,
            "speedup": speedup,
            "throughput_epoch_s": throughput,
        }

    print("-" * 90)
    return results


if __name__ == "__main__":
    benchmark_wls()
