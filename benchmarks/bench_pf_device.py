"""Benchmark: ParticleFilterDevice vs ParticleFilter

Compare persistent GPU memory approach (ParticleFilterDevice) vs
per-call allocation approach (ParticleFilter / cudaMalloc per call).
"""

import sys
import time
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_test_scenario():
    """Create Tokyo Station 8-satellite scenario."""
    true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
    true_cb = 3000.0

    sat_ecef = np.array([
        [-14985000.0, -3988000.0, 21474000.0],
        [-9575000.0,  15498000.0, 19457000.0],
        [ 7624000.0, -16218000.0, 19843000.0],
        [16305000.0,  12037000.0, 17183000.0],
        [-20889000.0, 13759000.0,  8291000.0],
        [ 5463000.0,  24413000.0,  8934000.0],
        [22169000.0,   3975000.0, 13781000.0],
        [-11527000.0,-19421000.0, 13682000.0],
    ])

    ranges = np.sqrt(np.sum((sat_ecef - true_pos) ** 2, axis=1))
    pseudoranges = ranges + true_cb
    return sat_ecef, pseudoranges, true_pos, true_cb


def _make_sat_scenario(n_sat):
    """Return a satellite scenario with the requested satellite count."""
    true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
    true_cb = 3000.0
    rng = np.random.default_rng(7)

    elevations = rng.uniform(np.radians(15), np.radians(80), n_sat)
    azimuths   = rng.uniform(0, 2 * np.pi, n_sat)
    dist = 26_560_000.0

    sat_ecef = np.column_stack([
        dist * np.cos(elevations) * np.cos(azimuths),
        dist * np.cos(elevations) * np.sin(azimuths),
        dist * np.sin(elevations),
    ])
    ranges = np.sqrt(np.sum((sat_ecef - true_pos) ** 2, axis=1))
    pseudoranges = ranges + true_cb
    return sat_ecef, pseudoranges, true_pos, true_cb


def _fmt_time(seconds):
    """Format a duration in human-readable units."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:9.2f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:9.3f} ms"
    else:
        return f"{seconds:9.4f}  s"


def _fmt_throughput(count, seconds):
    """Format throughput with SI prefix."""
    rate = count / seconds
    if rate >= 1e9:
        return f"{rate / 1e9:.3f} G/s"
    elif rate >= 1e6:
        return f"{rate / 1e6:.3f} M/s"
    elif rate >= 1e3:
        return f"{rate / 1e3:.3f} K/s"
    else:
        return f"{rate:.2f} /s"


def _measure(fn, n_iter=50):
    """Run *fn* n_iter times and return (mean_s, std_s)."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# 1. Device vs Standard comparison
# ---------------------------------------------------------------------------

def benchmark_device_vs_standard(n_particles_list=None):
    """For each particle count compare per-step time of ParticleFilterDevice
    (persistent GPU memory) against ParticleFilter (cudaMalloc per call).

    Each measurement covers one predict + update cycle (n_iter=100 cycles).
    """
    if n_particles_list is None:
        n_particles_list = [10_000, 100_000, 1_000_000]

    pf_ok = pfd_ok = True

    try:
        from gnss_gpu import ParticleFilter
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] ParticleFilter not available: {e}")
        pf_ok = False

    try:
        from gnss_gpu import ParticleFilterDevice
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] ParticleFilterDevice not available: {e}")
        pfd_ok = False

    if not pf_ok and not pfd_ok:
        return {}

    sat_ecef, pseudoranges, true_pos, true_cb = _make_test_scenario()
    n_iter  = 100
    n_warmup = 5

    col_n  = 12
    col_impl = 16
    col_mean = 13
    col_std  = 13
    col_tp   = 16
    col_su   = 10
    total_w  = col_n + col_impl + col_mean + col_std + col_tp + col_su + 17

    print()
    print("=" * total_w)
    print("1. ParticleFilterDevice vs ParticleFilter  (predict + update cycle)")
    print("=" * total_w)
    print(
        f"{'N Particles':>{col_n}} | {'Impl':>{col_impl}} | "
        f"{'Mean/step':>{col_mean}} | {'Std/step':>{col_std}} | "
        f"{'Throughput':>{col_tp}} | {'Speedup':>{col_su}}"
    )
    print("-" * total_w)

    results = {}

    for n_p in n_particles_list:
        row = {}

        # --- Standard PF ---
        mean_std = std_std = None
        if pf_ok:
            try:
                pf = ParticleFilter(n_particles=n_p, seed=42)
                pf.initialize(true_pos, clock_bias=true_cb)
                for _ in range(n_warmup):
                    pf.predict()
                    pf.update(sat_ecef, pseudoranges)

                mean_std, std_std = _measure(
                    lambda: (pf.predict(), pf.update(sat_ecef, pseudoranges)),
                    n_iter=n_iter)
                row["standard"] = (mean_std, std_std)
                print(
                    f"{n_p:>{col_n},} | {'ParticleFilter':>{col_impl}} | "
                    f"{_fmt_time(mean_std):>{col_mean}} | "
                    f"{_fmt_time(std_std):>{col_std}} | "
                    f"{_fmt_throughput(n_p, mean_std):>{col_tp}} | "
                    f"{'1.00x':>{col_su}}"
                )
            except Exception as e:
                print(f"{n_p:>{col_n},} | {'ParticleFilter':>{col_impl}} | [ERROR: {e}]")

        # --- Device PF ---
        if pfd_ok:
            try:
                pfd = ParticleFilterDevice(n_particles=n_p, seed=42)
                pfd.initialize(true_pos, clock_bias=true_cb)
                for _ in range(n_warmup):
                    pfd.predict()
                    pfd.update(sat_ecef, pseudoranges)

                mean_dev, std_dev = _measure(
                    lambda: (pfd.predict(), pfd.update(sat_ecef, pseudoranges)),
                    n_iter=n_iter)
                row["device"] = (mean_dev, std_dev)

                speedup = (mean_std / mean_dev) if mean_std else float("nan")
                print(
                    f"{'':>{col_n}} | {'PFDevice':>{col_impl}} | "
                    f"{_fmt_time(mean_dev):>{col_mean}} | "
                    f"{_fmt_time(std_dev):>{col_std}} | "
                    f"{_fmt_throughput(n_p, mean_dev):>{col_tp}} | "
                    f"{speedup:>{col_su-1}.2f}x"
                )
            except Exception as e:
                print(f"{'':>{col_n}} | {'PFDevice':>{col_impl}} | [ERROR: {e}]")

        print("-" * total_w)
        results[n_p] = row

    return results


# ---------------------------------------------------------------------------
# 2. Detailed PF Device operation timing
# ---------------------------------------------------------------------------

def benchmark_pf_device_detailed(n_particles=1_000_000):
    """Detailed timing of individual ParticleFilterDevice operations.

    Covers:
    - initialize
    - predict
    - weight/update (n_sat = 4, 8, 16, 32)
    - resample systematic
    - resample megopolis
    - estimate
    - get_particles  (D2H transfer overhead)
    """
    try:
        from gnss_gpu import ParticleFilterDevice
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] ParticleFilterDevice not available: {e}")
        return {}

    _, _, true_pos, true_cb = _make_test_scenario()
    n_iter   = 50
    n_warmup = 3

    col_op  = 30
    col_val = 16
    col_std = 16
    col_tp  = 16
    total_w = col_op + col_val + col_std + col_tp + 9

    print()
    print("=" * total_w)
    print(f"2. ParticleFilterDevice Detailed Stage Timing  (N={n_particles:,})")
    print("=" * total_w)
    print(
        f"{'Operation':>{col_op}} | {'Mean':>{col_val}} | "
        f"{'Std':>{col_std}} | {'Throughput':>{col_tp}}"
    )
    print("-" * total_w)

    results = {}

    def _row(label, mean_t, std_t, count=None):
        tp = _fmt_throughput(count, mean_t) if count else "---"
        print(
            f"{label:>{col_op}} | {_fmt_time(mean_t):>{col_val}} | "
            f"{_fmt_time(std_t):>{col_std}} | {tp:>{col_tp}}"
        )
        results[label] = (mean_t, std_t)

    # --- initialize ---
    pfd = ParticleFilterDevice(n_particles=n_particles, seed=42)
    pfd.initialize(true_pos, clock_bias=true_cb)  # warm-up construction

    for _ in range(n_warmup):
        pfd.initialize(true_pos, clock_bias=true_cb)

    mean_t, std_t = _measure(lambda: pfd.initialize(true_pos, clock_bias=true_cb), n_iter)
    _row("initialize", mean_t, std_t, n_particles)

    # --- predict ---
    pfd.initialize(true_pos, clock_bias=true_cb)
    for _ in range(n_warmup):
        pfd.predict()
    mean_t, std_t = _measure(pfd.predict, n_iter)
    _row("predict", mean_t, std_t, n_particles)

    # --- weight/update with varying n_sat ---
    for n_sat in [4, 8, 16, 32]:
        sat_ecef, pseudoranges, _, _ = _make_sat_scenario(n_sat)
        # Reset uniform weights by re-initializing
        pfd.initialize(true_pos, clock_bias=true_cb)
        for _ in range(n_warmup):
            # Use private weight method directly to avoid resample side-effects
            pfd._pf_device_weight(
                pfd._state, sat_ecef.ravel(), pseudoranges,
                np.ones(n_sat, dtype=np.float64), n_sat, pfd.sigma_pr)

        mean_t, std_t = _measure(
            lambda sat=sat_ecef, pr=pseudoranges, ns=n_sat: pfd._pf_device_weight(
                pfd._state, sat.ravel(), pr,
                np.ones(ns, dtype=np.float64), ns, pfd.sigma_pr),
            n_iter)
        _row(f"weight (n_sat={n_sat})", mean_t, std_t, n_particles * n_sat)

    print("-" * total_w)

    # --- resample systematic ---
    pfd.initialize(true_pos, clock_bias=true_cb)
    for _ in range(n_warmup):
        pfd._pf_device_resample_systematic(pfd._state, 42)
    mean_t, std_t = _measure(lambda: pfd._pf_device_resample_systematic(pfd._state, 42), n_iter)
    _row("resample systematic", mean_t, std_t, n_particles)

    # --- resample megopolis ---
    pfd.initialize(true_pos, clock_bias=true_cb)
    for _ in range(n_warmup):
        pfd._pf_device_resample_megopolis(pfd._state, 15, 42)
    mean_t, std_t = _measure(lambda: pfd._pf_device_resample_megopolis(pfd._state, 15, 42), n_iter)
    _row("resample megopolis", mean_t, std_t, n_particles)

    print("-" * total_w)

    # --- estimate ---
    pfd.initialize(true_pos, clock_bias=true_cb)
    for _ in range(n_warmup):
        pfd.estimate()
    mean_t, std_t = _measure(pfd.estimate, n_iter)
    _row("estimate", mean_t, std_t, n_particles)

    # --- get_particles (D2H transfer) ---
    for _ in range(n_warmup):
        pfd.get_particles()
    mean_t, std_t = _measure(pfd.get_particles, n_iter=10)
    n_bytes = n_particles * 4 * 8  # 4 floats * 8 bytes each
    bw_gbs  = n_bytes / mean_t / 1e9
    print(
        f"{'get_particles (D2H)':>{col_op}} | {_fmt_time(mean_t):>{col_val}} | "
        f"{_fmt_time(std_t):>{col_std}} | "
        f"{bw_gbs:>{col_tp-5}.2f} GB/s"
    )
    results["get_particles"] = (mean_t, std_t)

    print("-" * total_w)
    return results


# ---------------------------------------------------------------------------
# 3. CUDA Streams async overlap benefit
# ---------------------------------------------------------------------------

def benchmark_async_overlap():
    """Measure the benefit of CUDA Streams asynchronous execution.

    Compares two timing strategies over a predict-update-estimate pipeline:
    - sync_always:  cudaDeviceSynchronize (via pfd.sync()) after every step
    - sync_lazy:    sync only at estimate() (kernel overlap allowed)

    The difference reveals the overlap benefit delivered by persistent
    CUDA streams inside ParticleFilterDevice.
    """
    try:
        from gnss_gpu import ParticleFilterDevice
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] ParticleFilterDevice not available: {e}")
        return {}

    sat_ecef, pseudoranges, true_pos, true_cb = _make_test_scenario()
    n_particles = 1_000_000
    n_iter   = 100
    n_warmup = 5

    pfd = ParticleFilterDevice(n_particles=n_particles, seed=42)
    pfd.initialize(true_pos, clock_bias=true_cb)

    # warm-up both modes
    for _ in range(n_warmup):
        pfd.predict(); pfd.sync()
        pfd.update(sat_ecef, pseudoranges); pfd.sync()
        pfd.estimate()

    col_mode = 28
    col_val  = 14
    col_std  = 14
    col_tp   = 16
    total_w  = col_mode + col_val + col_std + col_tp + 9

    print()
    print("=" * total_w)
    print(f"3. CUDA Streams Async Overlap Benefit  (N={n_particles:,})")
    print("=" * total_w)
    print(
        f"{'Mode':>{col_mode}} | {'Mean/step':>{col_val}} | "
        f"{'Std/step':>{col_std}} | {'Throughput':>{col_tp}}"
    )
    print("-" * total_w)

    results = {}

    # --- sync after every step ---
    def _sync_always():
        pfd.predict()
        pfd.sync()
        pfd.update(sat_ecef, pseudoranges)
        pfd.sync()
        pfd.estimate()

    mean_sync, std_sync = _measure(_sync_always, n_iter)
    results["sync_always"] = (mean_sync, std_sync)
    print(
        f"{'sync after each step':>{col_mode}} | "
        f"{_fmt_time(mean_sync):>{col_val}} | "
        f"{_fmt_time(std_sync):>{col_std}} | "
        f"{_fmt_throughput(n_particles, mean_sync):>{col_tp}}"
    )

    # --- sync only at estimate ---
    def _sync_lazy():
        pfd.predict()
        pfd.update(sat_ecef, pseudoranges)
        pfd.estimate()   # internal sync before D2H transfer

    mean_lazy, std_lazy = _measure(_sync_lazy, n_iter)
    results["sync_lazy"] = (mean_lazy, std_lazy)
    print(
        f"{'sync only at estimate':>{col_mode}} | "
        f"{_fmt_time(mean_lazy):>{col_val}} | "
        f"{_fmt_time(std_lazy):>{col_std}} | "
        f"{_fmt_throughput(n_particles, mean_lazy):>{col_tp}}"
    )

    print("-" * total_w)
    overlap_pct = max(0.0, (mean_sync - mean_lazy) / mean_sync * 100)
    speedup = mean_sync / mean_lazy if mean_lazy > 0 else float("nan")
    print(
        f"  Overlap benefit: {overlap_pct:.1f}%  "
        f"({speedup:.2f}x speedup with async execution)"
    )
    print()

    return results


# ---------------------------------------------------------------------------
# Summary helper (used by bench_all.py)
# ---------------------------------------------------------------------------

def run_device_vs_standard_summary(n_particles=1_000_000):
    """Run device vs standard benchmark at a single particle count and return
    key metrics for inclusion in the bench_all summary table.

    Returns a dict with keys 'standard' and 'device', each containing a
    sub-dict with 'time_ms', 'throughput', and 'label'.
    Returns {} on failure.
    """
    results = {}

    sat_ecef, pseudoranges, true_pos, true_cb = _make_test_scenario()
    n_iter  = 20
    n_warmup = 3

    # Standard PF
    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(n_particles=n_particles, seed=42)
        pf.initialize(true_pos, clock_bias=true_cb)
        for _ in range(n_warmup):
            pf.predict(); pf.update(sat_ecef, pseudoranges)

        mean_std, _ = _measure(
            lambda: (pf.predict(), pf.update(sat_ecef, pseudoranges)),
            n_iter=n_iter)
        results["standard"] = {
            "time_ms":    mean_std * 1e3,
            "throughput": n_particles / mean_std,
            "label":      f"1M particles (PF)",
        }
    except (ImportError, RuntimeError):
        pass

    # Device PF
    try:
        from gnss_gpu import ParticleFilterDevice
        pfd = ParticleFilterDevice(n_particles=n_particles, seed=42)
        pfd.initialize(true_pos, clock_bias=true_cb)
        for _ in range(n_warmup):
            pfd.predict(); pfd.update(sat_ecef, pseudoranges)

        mean_dev, _ = _measure(
            lambda: (pfd.predict(), pfd.update(sat_ecef, pseudoranges)),
            n_iter=n_iter)
        results["device"] = {
            "time_ms":    mean_dev * 1e3,
            "throughput": n_particles / mean_dev,
            "label":      f"1M particles (PFD)",
        }
    except (ImportError, RuntimeError):
        pass

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    benchmark_device_vs_standard()
    benchmark_pf_device_detailed()
    benchmark_async_overlap()
