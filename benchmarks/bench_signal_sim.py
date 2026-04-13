#!/usr/bin/env python3
"""Benchmark: GPU signal simulator vs gps-sdr-sim (CPU).

Compares IQ signal generation throughput for the same scenario:
  - 11 GPS satellites
  - 2.6 MHz sampling rate
  - Various durations (1ms to 10s)

gps-sdr-sim reference: ~1.8s for 10s signal on same hardware.
"""

import subprocess
import time

import numpy as np

from gnss_gpu.signal_sim import SignalSimulator


def benchmark_gpu():
    sim = SignalSimulator(sampling_freq=2.6e6, intermediate_freq=0, noise_floor_db=-20)

    channels = []
    np.random.seed(42)
    for prn in [5, 10, 12, 13, 14, 15, 18, 20, 23, 24, 28]:
        channels.append({
            "prn": prn, "code_phase": 0.0, "carrier_phase": 0.0,
            "doppler_hz": float(np.random.uniform(-3000, 3000)),
            "amplitude": 1.0, "nav_bit": 1,
        })

    # Warmup
    _ = sim.generate_epoch(channels, n_samples=2600)

    print("=== GPU Signal Simulator (gnss_gpu) ===")
    print(f"{'Duration':>10} {'Samples':>12} {'Time':>10} {'Throughput':>14} {'RT factor':>10}")
    print("-" * 62)

    results = {}
    for duration_s in [0.001, 0.01, 0.1, 1.0, 10.0]:
        n_samples = int(2.6e6 * duration_s)
        t0 = time.time()
        iq = sim.generate_epoch(channels, n_samples=n_samples)
        dt = time.time() - t0
        throughput = n_samples / dt / 1e6
        rt = duration_s / dt
        print(f"{duration_s:>10.3f}s {n_samples:>12,} {dt*1000:>9.1f}ms {throughput:>12.1f} MS/s {rt:>9.1f}x")
        results[duration_s] = {"time_ms": dt * 1000, "throughput_msps": throughput, "rt_factor": rt}

    return results


def benchmark_gpssdr():
    """Run gps-sdr-sim for reference timing."""
    print("\n=== gps-sdr-sim (CPU) ===")
    try:
        t0 = time.time()
        result = subprocess.run(
            [".venv/bin/gps-sdr-sim",
             "-e", "/tmp/gps-sdr-sim/brdc0010.22n",
             "-l", "35.681298,139.766247,10.0",
             "-d", "10", "-s", "2600000", "-b", "8",
             "-o", "/tmp/gpssim_bench.bin"],
            capture_output=True, text=True, timeout=60,
        )
        dt = time.time() - t0
        # Extract process time from output
        for line in result.stdout.split("\n"):
            if "Process time" in line:
                proc_time = float(line.split("=")[1].split("[")[0].strip())
                print(f"10s signal: {proc_time:.1f}s process, {dt:.1f}s wall")
                print(f"Throughput: {26e6 / proc_time / 1e6:.1f} MS/s")
                print(f"RT factor: {10.0 / proc_time:.1f}x")
                return proc_time
    except Exception as e:
        print(f"gps-sdr-sim not available: {e}")
        return None


def main():
    gpu_results = benchmark_gpu()
    cpu_time = benchmark_gpssdr()

    if cpu_time and 10.0 in gpu_results:
        gpu_time = gpu_results[10.0]["time_ms"] / 1000
        speedup = cpu_time / gpu_time
        print(f"\n=== Summary (10s signal, 11 satellites, 2.6 MHz) ===")
        print(f"gps-sdr-sim (CPU): {cpu_time:.2f}s")
        print(f"gnss_gpu (GPU):    {gpu_time:.3f}s")
        print(f"Speedup:           {speedup:.1f}x")


if __name__ == "__main__":
    main()
