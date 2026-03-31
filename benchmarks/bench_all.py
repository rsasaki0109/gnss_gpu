"""Master benchmark script for gnss_gpu.

Runs all module benchmarks and produces a unified summary table.
"""

import sys
import time
import numpy as np


def _separator(char="=", width=72):
    print(char * width)


def _run_particle_filter_summary():
    """Run particle filter benchmark at 1M particles and return key metric."""
    try:
        from gnss_gpu import ParticleFilter
    except (ImportError, RuntimeError):
        return None

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

    n_p = 1_000_000
    n_iter = 5

    # Warm-up
    pf = ParticleFilter(n_particles=n_p, seed=42)
    pf.initialize(true_pos, clock_bias=true_cb)
    pf.predict()
    pf.update(sat_ecef, pseudoranges)
    pf.estimate()

    times = []
    for _ in range(n_iter):
        pf2 = ParticleFilter(n_particles=n_p, seed=42)
        t0 = time.perf_counter()
        pf2.initialize(true_pos, clock_bias=true_cb)
        pf2.predict()
        pf2.update(sat_ecef, pseudoranges)
        pf2.estimate()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_t = np.mean(times)
    throughput = n_p / mean_t
    return {"time_ms": mean_t * 1e3, "throughput": throughput, "label": "1M particles"}


def _run_wls_summary():
    """Run WLS batch benchmark at 10000 epochs and return key metric."""
    try:
        from gnss_gpu._gnss_gpu import wls_batch
    except (ImportError, RuntimeError):
        return None

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

    n_epoch = 10000
    n_iter = 5
    rng = np.random.default_rng(42)

    sat_batch = np.tile(sat_ecef, (n_epoch, 1, 1))
    pr_batch = np.tile(pseudoranges, (n_epoch, 1)) + rng.normal(0, 3.0, (n_epoch, len(pseudoranges)))
    w_batch = np.tile(weights, (n_epoch, 1))

    # Warm-up
    wls_batch(sat_batch[:10], pr_batch[:10], w_batch[:10])

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        wls_batch(sat_batch, pr_batch, w_batch)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_t = np.mean(times)
    throughput = n_epoch / mean_t
    return {"time_ms": mean_t * 1e3, "throughput": throughput, "label": "10K epochs"}


def _run_acquisition_summary():
    """Run acquisition benchmark at 32 PRNs, 1ms and return key metric."""
    try:
        from gnss_gpu import Acquisition
    except (ImportError, RuntimeError):
        return None

    sampling_freq = 4.092e6
    n_prn = 32
    n_iter = 5

    acq = Acquisition(sampling_freq=sampling_freq, doppler_range=5000, doppler_step=500)

    rng = np.random.default_rng(42)
    n_samples = int(sampling_freq * 1e-3)
    signal = rng.standard_normal(n_samples).astype(np.float32)
    for prn in [1, 5, 10]:
        try:
            s = Acquisition.generate_test_signal(
                prn=prn, code_phase=100, doppler=1500.0, snr_db=15.0,
                sampling_freq=sampling_freq, duration_s=1e-3)
            signal[:len(s)] += s[:len(signal)]
        except Exception:
            pass

    prn_list = list(range(1, n_prn + 1))

    # Warm-up
    try:
        acq.acquire(signal, prn_list=prn_list)
    except Exception:
        return None

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        acq.acquire(signal, prn_list=prn_list)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_t = np.mean(times)
    throughput = n_prn / mean_t
    return {"time_ms": mean_t * 1e3, "throughput": throughput, "label": "32 PRN, 1ms"}


def _run_skyplot_summary():
    """Run vulnerability map benchmark at 100x100 grid and return key metric."""
    try:
        from gnss_gpu import VulnerabilityMap
    except (ImportError, RuntimeError):
        return None

    origin_lla = (35.6812, 139.7671, 5.0)
    n_iter = 5

    # Generate satellite constellation
    sat_ecef = []
    orbit_radius = 26559.7e3
    for i in range(24):
        plane = i % 6
        slot = i // 6
        raan = np.radians(plane * 60)
        mean_anomaly = np.radians(slot * 90 + plane * 15)
        inclination = np.radians(55.0)

        x_orb = orbit_radius * np.cos(mean_anomaly)
        y_orb = orbit_radius * np.sin(mean_anomaly)

        x = x_orb * np.cos(raan) - y_orb * np.cos(inclination) * np.sin(raan)
        y = x_orb * np.sin(raan) + y_orb * np.cos(inclination) * np.cos(raan)
        z = y_orb * np.sin(inclination)
        sat_ecef.append([x, y, z])

    sat_ecef = np.array(sat_ecef, dtype=np.float64)

    # 100x100 grid
    vm = VulnerabilityMap(origin_lla, grid_size_m=495, resolution_m=5)
    n_points = vm.n_grid

    # Warm-up
    try:
        vm.evaluate(sat_ecef, elevation_mask_deg=10.0)
    except RuntimeError:
        return None

    times = []
    for _ in range(n_iter):
        vm2 = VulnerabilityMap(origin_lla, grid_size_m=495, resolution_m=5)
        t0 = time.perf_counter()
        vm2.evaluate(sat_ecef, elevation_mask_deg=10.0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_t = np.mean(times)
    throughput = n_points / mean_t
    return {"time_ms": mean_t * 1e3, "throughput": throughput, "label": "100x100 grid"}


def _run_pf_device_summary():
    """Run PF Device vs standard benchmark at 1M particles and return metrics."""
    try:
        from bench_pf_device import run_device_vs_standard_summary
    except ImportError:
        # Fallback: try adding benchmarks dir to sys.path
        import os
        bench_dir = os.path.dirname(os.path.abspath(__file__))
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        try:
            from bench_pf_device import run_device_vs_standard_summary
        except ImportError:
            return None

    try:
        data = run_device_vs_standard_summary(n_particles=1_000_000)
    except Exception:
        return None

    if not data:
        return None

    # Prefer device result; fall back to standard
    if "device" in data:
        entry = data["device"]
        if "standard" in data:
            speedup = data["standard"]["time_ms"] / entry["time_ms"]
            entry["label"] = f"1M parts (PFD {speedup:.1f}x)"
        return entry
    elif "standard" in data:
        return data["standard"]
    return None


def _run_raytrace_summary():
    """Run raytrace benchmark at 1000 triangles, 8 sats and return key metric."""
    try:
        from gnss_gpu import BuildingModel
    except (ImportError, RuntimeError):
        return None

    rng = np.random.default_rng(42)
    n_buildings = 84  # ~1008 triangles
    all_triangles = []
    for i in range(n_buildings):
        angle = 2 * np.pi * i / n_buildings
        dist = 50.0 + rng.uniform(0, 200)
        cx = dist * np.cos(angle)
        cy = dist * np.sin(angle)
        cz = rng.uniform(10, 50)
        hw = rng.uniform(5, 15)
        hd = rng.uniform(5, 15)
        hh = rng.uniform(10, 50)

        v = np.array([
            [cx - hw, cy - hd, cz - hh], [cx + hw, cy - hd, cz - hh],
            [cx + hw, cy + hd, cz - hh], [cx - hw, cy + hd, cz - hh],
            [cx - hw, cy - hd, cz + hh], [cx + hw, cy - hd, cz + hh],
            [cx + hw, cy + hd, cz + hh], [cx - hw, cy + hd, cz + hh],
        ], dtype=np.float64)
        faces = [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5], [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ]
        for f in faces:
            all_triangles.append([v[f[0]], v[f[1]], v[f[2]]])

    triangles = np.array(all_triangles, dtype=np.float64)
    n_tri = len(triangles)
    n_sat = 8
    n_iter = 10

    try:
        model = BuildingModel(triangles)
    except Exception:
        return None

    rx_ecef = np.array([0.0, 0.0, 1.5])
    rng2 = np.random.default_rng(123)
    sat_ecef = []
    for i in range(n_sat):
        el = rng2.uniform(10, 80)
        az = rng2.uniform(0, 360)
        dist = 20200e3
        x = dist * np.cos(np.radians(el)) * np.sin(np.radians(az))
        y = dist * np.cos(np.radians(el)) * np.cos(np.radians(az))
        z = dist * np.sin(np.radians(el))
        sat_ecef.append([x, y, z])
    sat_ecef = np.array(sat_ecef, dtype=np.float64)

    # Warm-up
    try:
        model.check_los(rx_ecef, sat_ecef)
    except Exception:
        return None

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model.check_los(rx_ecef, sat_ecef)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_t = np.mean(times)
    total_checks = n_sat * n_tri
    throughput = total_checks / mean_t
    return {"time_ms": mean_t * 1e3, "throughput": throughput,
            "label": f"{n_tri} tri, {n_sat} sats"}


def _fmt_throughput(value, unit):
    """Format throughput with SI prefix."""
    if value >= 1e9:
        return f"{value / 1e9:.2f} G{unit}"
    elif value >= 1e6:
        return f"{value / 1e6:.2f} M{unit}"
    elif value >= 1e3:
        return f"{value / 1e3:.2f} K{unit}"
    else:
        return f"{value:.1f} {unit}"


def main():
    """Run all benchmarks and print summary table."""
    print()
    _separator("=")
    print("gnss_gpu Performance Benchmark Summary")
    _separator("=")
    print()

    # Run detailed benchmarks if requested
    if "--detailed" in sys.argv:
        print("[Running detailed benchmarks...]")
        print()

        try:
            from bench_particle_filter import benchmark_pf, benchmark_convergence
            benchmark_pf()
            benchmark_convergence()
        except Exception as e:
            print(f"[Particle Filter detailed] SKIP: {e}")
        print()

        try:
            from bench_pf_device import (
                benchmark_device_vs_standard,
                benchmark_pf_device_detailed,
                benchmark_async_overlap,
            )
            benchmark_device_vs_standard()
            benchmark_pf_device_detailed()
            benchmark_async_overlap()
        except Exception as e:
            print(f"[PF Device detailed] SKIP: {e}")
        print()

        try:
            from bench_wls import benchmark_wls
            benchmark_wls()
        except Exception as e:
            print(f"[WLS detailed] SKIP: {e}")
        print()

        try:
            from bench_acquisition import benchmark_acquisition
            benchmark_acquisition()
        except Exception as e:
            print(f"[Acquisition detailed] SKIP: {e}")
        print()

        try:
            from bench_skyplot import benchmark_skyplot
            benchmark_skyplot()
        except Exception as e:
            print(f"[Skyplot detailed] SKIP: {e}")
        print()

        try:
            from bench_raytrace import benchmark_raytrace
            benchmark_raytrace()
        except Exception as e:
            print(f"[Raytrace detailed] SKIP: {e}")
        print()

        _separator("=")
        print("SUMMARY")
        _separator("=")
        print()

    # Collect summary results
    benchmarks = [
        ("WLS Batch", _run_wls_summary, "epoch/s"),
        ("Particle Filter", _run_particle_filter_summary, "part/s"),
        ("PF Device", _run_pf_device_summary, "part/s"),
        ("Signal Acquisition", _run_acquisition_summary, "PRN/s"),
        ("Vulnerability Map", _run_skyplot_summary, "pts/s"),
        ("Ray Tracing", _run_raytrace_summary, "checks/s"),
    ]

    col_module = 22
    col_input = 18
    col_time = 12
    col_tp = 20
    total_w = col_module + col_input + col_time + col_tp + 9

    print(f"{'Module':<{col_module}} | {'Input Size':<{col_input}} | "
          f"{'Time (ms)':>{col_time}} | {'Throughput':>{col_tp}}")
    print("-" * total_w)

    for name, runner, unit in benchmarks:
        try:
            result = runner()
        except Exception as e:
            result = None
            print(f"{name:<{col_module}} | {'[ERROR]':<{col_input}} | "
                  f"{'---':>{col_time}} | {str(e)[:col_tp]:>{col_tp}}")
            continue

        if result is None:
            print(f"{name:<{col_module}} | {'[SKIP: no CUDA]':<{col_input}} | "
                  f"{'---':>{col_time}} | {'---':>{col_tp}}")
        else:
            tp_str = _fmt_throughput(result["throughput"], unit)
            print(f"{name:<{col_module}} | {result['label']:<{col_input}} | "
                  f"{result['time_ms']:>{col_time}.2f} | {tp_str:>{col_tp}}")

    print("-" * total_w)
    print()


if __name__ == "__main__":
    main()
