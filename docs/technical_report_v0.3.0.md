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
| WP172 candidate supply | 42.653 ms/epoch conservative sequential; 21.826 ms/epoch parallel |
| Runtime capacity | 152.588 MiB conservative estimate; 0 deadline misses |
| Cross-domain positioning | epoch-weighted RMS 17.107 m to 16.916 m |
| Campaign coverage | 3 cities; 9 sites/routes; 5 dates; 3 receivers |
| ROS replay | 10 events; 1 restart; canonical event hash locked |
| ROS continuity | 2 simulated hours; 439,187 events; 1 watchdog trip; final normal |
| Guarded MLAMBDA FIX | Tokyo 10.8688%; Nagoya 18.0667%; false-FIX 0 |
| Production promotion | 12/12 gates pass; Tokyo 46.5112% exceeds 45% |

The cross-domain result preserves Tokyo and improves the tracked Hong Kong
campaign. Every release claim is derived from locked, hash-checked evidence
included in the reproducibility archive.

## Safety and promotion

Promotion is fail closed. It requires truth-free production input, positive
full-denominator gain with zero loss, zero false FIX, all mandatory negative
holdouts rejected or explicitly abstained, exact M4 preservation, multi-city
non-degradation, reproducible input/config hashes, and the Phase 4
latency/memory limits.

WP172 seeds a demo5-continuous RTK candidate from the locked PF trajectory,
then accepts positions only when an independent RTK trajectory agrees within
1 m and the prefit residual RMS is at most 4 m. The gate was frozen on Nagoya
run1, where it gains 441 epochs with zero loss, before the Tokyo final audit.
Tokyo gains 1,802 epochs with zero loss and reaches 5,546/11,924 sub-50 cm
epochs (46.5112%). The selector is PF-only, uses no runtime FGO, and does not
load reference truth until the post-selection full-denominator audit.

WP173 promotes the existing libgnss++ MLAMBDA result to a declared FIX only
when the complete WP172 consensus gate passes, the LAMBDA ratio is at least
3.0, at least six satellites participate, and eligibility remains contiguous
for five epochs. These standard ratio/hold conditions were frozen on Nagoya:
1,370/7,583 epochs (18.0667%) are declared FIX with zero false FIX. Applied
unchanged to Tokyo, 1,296/11,924 epochs (10.8688%) are declared FIX with zero
false FIX. A failed ratio, satellite count, continuity check, or missing
candidate immediately falls back to FLOAT.

The two RTK candidate paths were replayed over all 11,924 Tokyo epochs on an
i7-9750H. Their conservative sequential sum is 42.653 ms/epoch, below the
100 ms production bound; concurrent wall time is 21.826 ms/epoch. The seeded
RTK output and final WP172 trajectory are byte-identical to the locks. The
independent RTK path itself is not byte-identical, but the guarded final
trajectory remains byte-identical, which is the product-level determinism
boundary.

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

- Tokyo clears the 45% positioning and 10% guarded FIX promotion floors, but
  most epochs intentionally remain FLOAT; the result does not claim broad RTK
  FIX availability.
- Tokyo run1 is not a virgin scientific holdout: earlier campaign diagnostics
  had already inspected it. WP172's numerical 1 m/4 m gates were inherited from
  physical safety limits, frozen and checked on Nagoya before the locked Tokyo
  rerun; the result is therefore an operational promotion audit, not an
  unbiased estimate from a never-observed benchmark.
- Phase 3 city-scale accuracy is supported indirectly by later campaigns; its
  locked outage-controller audit is synthetic.
- Hong Kong is reproduced from a tracked result summary because raw data is not
  in this checkout.
- The lifecycle core and replay contract are host-tested; `rclpy` and colcon
  integration are validated by the ROS container build.
- The Windows Phase 4 memory value is a conservative capacity estimate because
  per-process `nvidia-smi` memory was unavailable.
