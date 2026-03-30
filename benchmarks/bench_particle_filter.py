"""Benchmark particle filter performance at different particle counts."""

import sys
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

    return sat_ecef, pseudoranges, true_pos, true_cb


def _fmt_time(seconds):
    """Format time in appropriate units."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    else:
        return f"{seconds:8.3f}  s"


def _fmt_throughput(count, seconds):
    """Format throughput with SI prefix."""
    rate = count / seconds
    if rate >= 1e9:
        return f"{rate / 1e9:.2f} G/s"
    elif rate >= 1e6:
        return f"{rate / 1e6:.2f} M/s"
    elif rate >= 1e3:
        return f"{rate / 1e3:.2f} K/s"
    else:
        return f"{rate:.2f} /s"


def benchmark_pf(n_particles_list=None):
    """Benchmark initialize, predict, weight, resample, estimate."""
    if n_particles_list is None:
        n_particles_list = [1000, 10000, 100000, 1000000]

    try:
        from gnss_gpu import ParticleFilter
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] ParticleFilter not available: {e}")
        return {}

    sat_ecef, pseudoranges, true_pos, true_cb = _make_test_scenario()
    n_iter = 10
    results = {}

    print("=" * 80)
    print("Particle Filter Stage Benchmarks")
    print("=" * 80)
    header = f"{'N Particles':>12} | {'Stage':>12} | {'Mean':>12} | {'Std':>12} | {'Throughput':>14}"
    print(header)
    print("-" * 80)

    for n_p in n_particles_list:
        stage_results = {}

        # --- Warm-up ---
        pf = ParticleFilter(n_particles=n_p, seed=42)
        pf.initialize(true_pos, clock_bias=true_cb, spread_pos=100.0, spread_cb=1000.0)
        pf.predict()
        pf.update(sat_ecef, pseudoranges)
        pf.estimate()

        # --- Initialize ---
        times = []
        for _ in range(n_iter):
            pf = ParticleFilter(n_particles=n_p, seed=42)
            t0 = time.perf_counter()
            pf.initialize(true_pos, clock_bias=true_cb, spread_pos=100.0, spread_cb=1000.0)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        mean_t, std_t = np.mean(times), np.std(times)
        stage_results["initialize"] = (mean_t, std_t)
        print(f"{n_p:>12,} | {'initialize':>12} | {_fmt_time(mean_t)} | {_fmt_time(std_t)} | {_fmt_throughput(n_p, mean_t):>14}")

        # --- Predict ---
        pf = ParticleFilter(n_particles=n_p, seed=42)
        pf.initialize(true_pos, clock_bias=true_cb, spread_pos=100.0, spread_cb=1000.0)
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            pf.predict()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        mean_t, std_t = np.mean(times), np.std(times)
        stage_results["predict"] = (mean_t, std_t)
        print(f"{'':>12} | {'predict':>12} | {_fmt_time(mean_t)} | {_fmt_time(std_t)} | {_fmt_throughput(n_p, mean_t):>14}")

        # --- Weight (update without resampling check) ---
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            pf.update(sat_ecef, pseudoranges)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        mean_t, std_t = np.mean(times), np.std(times)
        stage_results["update"] = (mean_t, std_t)
        print(f"{'':>12} | {'update':>12} | {_fmt_time(mean_t)} | {_fmt_time(std_t)} | {_fmt_throughput(n_p, mean_t):>14}")

        # --- Estimate ---
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            pf.estimate()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        mean_t, std_t = np.mean(times), np.std(times)
        stage_results["estimate"] = (mean_t, std_t)
        print(f"{'':>12} | {'estimate':>12} | {_fmt_time(mean_t)} | {_fmt_time(std_t)} | {_fmt_throughput(n_p, mean_t):>14}")

        # --- Total pipeline (init + predict + update + estimate) ---
        times = []
        for _ in range(n_iter):
            pf2 = ParticleFilter(n_particles=n_p, seed=42)
            t0 = time.perf_counter()
            pf2.initialize(true_pos, clock_bias=true_cb, spread_pos=100.0, spread_cb=1000.0)
            pf2.predict()
            pf2.update(sat_ecef, pseudoranges)
            pf2.estimate()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        mean_t, std_t = np.mean(times), np.std(times)
        stage_results["total_pipeline"] = (mean_t, std_t)
        print(f"{'':>12} | {'TOTAL':>12} | {_fmt_time(mean_t)} | {_fmt_time(std_t)} | {_fmt_throughput(n_p, mean_t):>14}")
        print("-" * 80)

        results[n_p] = stage_results

    return results


def benchmark_convergence(n_particles_list=None):
    """Measure positioning accuracy vs particle count."""
    if n_particles_list is None:
        n_particles_list = [1000, 10000, 100000]

    try:
        from gnss_gpu import ParticleFilter
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] ParticleFilter not available: {e}")
        return {}

    sat_ecef, pseudoranges, true_pos, true_cb = _make_test_scenario()
    n_epochs = 10
    rng = np.random.default_rng(123)
    results = {}

    print()
    print("=" * 80)
    print("Particle Filter Convergence (accuracy vs particle count)")
    print("=" * 80)
    print(f"{'N Particles':>12} | {'Final Err (m)':>14} | {'Clock Err (m)':>14} | {'Time (s)':>10}")
    print("-" * 60)

    for n_p in n_particles_list:
        pf = ParticleFilter(n_particles=n_p, sigma_pos=1.0, sigma_cb=300.0,
                            sigma_pr=5.0, seed=42)

        # Add noise to initial position
        init_pos = true_pos + rng.normal(0, 50, size=3)

        t0 = time.perf_counter()
        pf.initialize(init_pos, clock_bias=true_cb + rng.normal(0, 500),
                      spread_pos=100.0, spread_cb=1000.0)

        for epoch in range(n_epochs):
            # Add measurement noise
            pr_noisy = pseudoranges + rng.normal(0, 5.0, size=len(pseudoranges))
            pf.predict()
            pf.update(sat_ecef, pr_noisy)

        est = pf.estimate()
        t1 = time.perf_counter()

        pos_err = np.linalg.norm(est[:3] - true_pos)
        cb_err = abs(est[3] - true_cb)
        elapsed = t1 - t0

        results[n_p] = {"pos_error_m": pos_err, "cb_error_m": cb_err, "time_s": elapsed}
        print(f"{n_p:>12,} | {pos_err:>14.3f} | {cb_err:>14.3f} | {elapsed:>10.4f}")

    return results


if __name__ == "__main__":
    benchmark_pf()
    benchmark_convergence()
