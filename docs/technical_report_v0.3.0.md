# gnss_gpu v0.3.0 technical report

## Scope

v0.3 turns the urban-navigation research workspace into a guarded
GNSS/IMU/map/GPU platform with immutable negative controls, truth-free
acceptance evidence, multi-hypothesis outage recovery, enforced real-time
budgets, cross-domain validation, and a ROS 2 lifecycle safety boundary.

## Audited results

| Contract | Locked result |
|---|---|
| Historical negative controls | 4/4 rejected |
| DDPR structure | 2/3 WP163 ranks recovered; 0 WP164 false passes |
| Outage recovery | 3 evidence epochs; greedy 10.0 m vs retained 0.0 m |
| GTX 1660 Ti runtime | normal max 13.761 ms; search max 75.907 ms |
| Runtime capacity | 152.588 MiB conservative estimate; 0 deadline misses |
| Cross-domain positioning | epoch-weighted RMS 17.107 m to 16.916 m |
| Campaign coverage | 3 cities; 9 sites/routes; 5 dates; 3 receivers |
| ROS replay | 10 events; 1 restart; canonical event hash locked |
| ROS continuity | 2 simulated hours; 439,187 events; 1 watchdog trip; final normal |
| Production promotion | 10/11 gates pass; Tokyo accuracy gate remains closed |

The cross-domain result preserves Tokyo and improves the tracked Hong Kong
campaign. Every release claim is derived from locked, hash-checked evidence
included in the reproducibility archive.

## Safety and promotion

Promotion is fail closed. It requires truth-free production input, positive
full-denominator gain with zero loss, zero false FIX, all mandatory negative
holdouts rejected or explicitly abstained, exact M4 preservation, multi-city
non-degradation, reproducible input/config hashes, and the Phase 4
latency/memory limits.

The locked WP160 Tokyo audit proves truth-free production input, 55 gained
epochs, zero lost epochs, and zero false FIX over the full 11,924-epoch
denominator. It reaches 3,744 sub-50 cm epochs (31.3989%). Production promotion
therefore remains closed because the 45% target requires at least 5,366 epochs,
a gap of 1,622.

The ROS 2 lifecycle boundary rejects invalid parameters and timestamp
regressions, publishes diagnostics, enters safe fallback under watchdog
conditions, and produces a deterministic replay digest. A deterministic
two-hour, 360,000-tick continuity audit injects duplicates, timestamp faults, a
GNSS outage, and a restart; it records one watchdog trip, five recoveries, a
bounded core state, and a final normal navigation mode.

## Reproduction

Build and verify the release archive:

```bash
python tools/build_release_bundle.py \
  --output dist/reproducibility \
  --archive dist/gnss_gpu-v0.3.0-reproducibility.zip
python tools/build_release_bundle.py --verify dist/reproducibility
```

The archive contains the locked source evidence, a benchmark summary, ablation
records, a failure gallery, this report, and a SHA-256 manifest. ZIP entry order
and timestamps are deterministic.

## Limitations

- The final Tokyo sub-50 cm target of 45% is not met: the locked production
  candidate reaches 31.3989%, so the fail-closed promotion decision is false.
- Phase 3 city-scale accuracy is supported indirectly by later campaigns; its
  locked outage-controller audit is synthetic.
- Hong Kong is reproduced from a tracked result summary because raw data is not
  in this checkout.
- The lifecycle core and replay contract are host-tested; `rclpy` and colcon
  integration are validated by the ROS container build.
- The Windows Phase 4 memory value is a conservative capacity estimate because
  per-process `nvidia-smi` memory was unavailable.
