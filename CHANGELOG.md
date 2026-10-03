# Changelog

## Unreleased

- `ParticleFilterDevice(resampling="megopolis_coalesced")`: opt-in Megopolis
  (Chesser et al. 2021) with ancestor-weight acceptance and block-shared,
  coalesced proposals. It converges to the weight distribution as iterations
  grow, unlike the default `"megopolis"`, and runs 3-13x faster (9 ms at
  B=15, 18 ms at B=240 for 1M particles). New `megopolis_iterations` argument
  (default 15). `ParticleFilterDevice` now rejects unknown `resampling` values.
- `ParticleFilterDevice` Megopolis resampling now mixes int32 ancestor indices
  and gathers the 16-double particle state once instead of copying it on
  every iteration. Bit-identical output; 1M-particle `predict → update` goes
  from 153 ms to 32 ms on a Turing GPU.
- Fixed a GPU memory leak in the host-buffered `ParticleFilter` Megopolis
  resampling: 4 of 9 device buffers were never freed (32 MB per call at 1M
  particles).

## 0.3.0 - 2026-07-29

- Added immutable evaluation contracts and mandatory negative holdouts.
- Added truth-free evidence detection, DDPR profiles, and weighted arc screens.
- Added IMU/map multi-hypothesis outage recovery.
- Added persistent CUDA batching, adaptive particle budgets, and runtime audits.
- Added cross-domain validation and fail-closed adaptation guidance.
- Added a ROS 2 lifecycle package with watchdog and deterministic replay.
- Added CUDA/ROS 2 containers, release automation, a public audit, and a
  deterministic reproducibility archive with benchmark, ablation, and failure
  gallery records.

See [the v0.3.0 release notes](RELEASE_NOTES_v0.3.0.md) for audited results and
known limits.
