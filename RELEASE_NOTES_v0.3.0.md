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
- The cross-domain campaign covers 3 cities, 9 sites/routes, 5 dates, and
  3 receivers. Epoch-weighted RMS changes from 17.107 m to 16.916 m, with
  Tokyo non-degradation and a Hong Kong gain.
- The deterministic ROS replay contains 10 events and one controlled restart.

## Important limits

This release does not claim the program's 45% Tokyo sub-50 cm target. The
Phase 5 campaign validates RMS and non-degradation, not that target. The locked
Phase 3 outage audit is synthetic, Hong Kong is reproduced from a tracked
summary because raw data is absent, and the Windows memory figure is a
conservative capacity estimate. Promotion remains fail closed.
