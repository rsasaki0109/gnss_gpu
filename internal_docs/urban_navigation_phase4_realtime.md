# Phase 4 realtime runtime contract

The native `ParticleFilterDevice` already keeps particle state, temporary
resampling buffers, observation scratch, pinned host memory, and one CUDA
stream persistent. The Phase 4 work makes those capabilities enforceable at
the campaign level.

## Hard budgets

- normal/degraded epoch: at most 100 ms;
- heavy search epoch: at most 1,000 ms;
- peak GPU memory: at most 4,096 MB.

`RealtimeMonitor` measures complete epochs, including an explicit CUDA stream
synchronization at the deadline boundary. It records per-stage time,
particle count, memory, mode, and fallback state. Missing normal-runtime
evidence fails closed.

The common campaign promotion gate now requires `normal_latency_max_ms`,
`search_latency_max_ms`, and peak GPU memory in addition to P50/P95 latency.

## Adaptive particles and safe degradation

The default schedule uses 50k, 100k, and 500k levels.

- low ESS, multiple hypotheses, or outage moves toward search capacity;
- latency or memory pressure immediately moves down one level;
- sustained stable single-mode tracking moves down after hysteresis;
- pressure at minimum capacity enters explicit degraded fallback.

Fallback never changes the Evidence API or FIX gates.

`PersistentParticleRuntimePool` creates each capacity once, checks the planned
resident-memory budget, and migrates state by deterministic systematic
resampling. Revisited capacities reuse their existing CUDA allocation and
stream.

Run `python experiments/benchmark_phase4_realtime.py` for the synchronized
100k normal / 500k search benchmark. The output is a promotion-consumable JSON
record with device name, latency distribution, deadline misses, memory, and
M4 hashes.

Heavy search also runs persistent Numba-CUDA batches for a 16x32 arc screen,
256x64 candidate residual scores, and 256 affine refits with 16 sub-blocks.
Kernel compilation and device/pinned-host allocation occur before timing.

The locked GTX 1660 Ti result is
`internal_docs/phase4_realtime_benchmark_2026_07_29.json`: normal maximum
13.761 ms, search maximum 75.907 ms, zero deadline misses. Windows
`nvidia-smi` reported process memory as unavailable, so the recorded
152.588 MB is the conservative 320-byte-per-particle capacity estimate, not
an observed peak. The 4,096 MB gate therefore remains conservative but should
be supplemented by allocator telemetry on deployment hardware.
