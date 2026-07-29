# gnss_gpu v0.3.0

v0.3.0 packages the research workspace as an auditable GNSS/IMU/map/GPU
navigation platform. It adds immutable negative controls, truth-free evidence
gates, multi-hypothesis outage recovery, measured real-time contracts,
cross-domain validation, and a ROS 2 lifecycle safety boundary.

## Release assets

- CUDA wheel built for compute capabilities 7.5, 8.0, 8.6, and 8.9.
- Source distribution and deterministic reproducibility archive.
- `ghcr.io/rsasaki0109/gnss_gpu-cuda:v0.3.0`.
- `ghcr.io/rsasaki0109/gnss_gpu-ros2:v0.3.0`.
- Public release audit and tracked technical report.

Rebuild and verify the archive with:

```bash
python tools/build_release_bundle.py \
  --output dist/reproducibility \
  --archive dist/gnss_gpu-v0.3.0-reproducibility.zip
python tools/build_release_bundle.py --verify dist/reproducibility
```

## Audited results

- All 4 mandatory historical negative controls are rejected.
- DDPR recovers 2 of 3 WP163 reference ranks with no WP164 false pass.
- Retained hypotheses recover the locked outage in 3 evidence epochs; the
  greedy comparator ends 10 m away.
- GTX 1660 Ti maxima are 13.761 ms in normal mode and 75.907 ms in search mode,
  with no recorded deadline miss.
- Full WP172 candidate supply replays at a conservative 42.653 ms/epoch when
  its two RTK paths are summed sequentially (21.826 ms/epoch concurrently);
  the guarded final trajectory is byte-identical.
- The cross-domain campaign covers 3 cities, 9 sites/routes, 5 dates, and
  3 receivers. Epoch-weighted RMS changes from 17.107 m to 16.916 m, with
  Tokyo non-degradation and a Hong Kong gain.
- The deterministic ROS replay contains 10 events and one controlled restart.
- The PF-only WP173 production trajectory retains 5,546/11,924 Tokyo epochs
  below 50 cm (46.5112%) and declares 1,296 guarded MLAMBDA FIX epochs
  (10.8688%), with 1,802 gained, zero lost, and zero false FIX.
- Nagoya development declares 1,370/7,583 guarded FIX epochs (18.0667%) with
  zero false FIX.
- The fail-closed production audit passes all 12 gates.

## Important limits

The 45% Tokyo sub-50 cm promotion floor and 10% guarded FIX floor are met.
WP173 does not treat every internal LAMBDA solution as a declared FIX: it
requires the complete WP172 consensus plus ratio, satellite-count, and
five-epoch continuity gates. The locked Tokyo rerun is an operational
promotion audit rather than a virgin scientific holdout because earlier
campaign diagnostics had inspected Tokyo run1. The numerical gates were
frozen and checked on Nagoya. The locked Phase 3 outage audit is synthetic,
Hong Kong is reproduced from a tracked summary because raw data is absent,
and the Windows memory figure is a conservative capacity estimate. Promotion
remains fail closed for future changes.
