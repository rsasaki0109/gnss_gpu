# Boreas reference and weather verification

Checked 2026-10-24 for `boreas-2020-11-26-13-58`.
Reference-file conventions are verified by documentation and internal numerical
checks. Weather classification and the GNSS antenna-to-pose axis mapping remain
unverified. No GNSS positioning-error or rain-effect result is claimed.

## Verified file conventions

The official [SDK reference](https://github.com/utiasASRL/pyboreas/blob/8e100ff82ecd4992511da61b016398dbd230f205/DATA_REFERENCE.md)
describes postprocessed GNSS/IMU/wheel-encoder poses, synchronized to UTC, in a
fixed ENU frame. Sensor poses are expressed at their respective sensor origins.
The frame origin is aligned with the first pose of this first sequence. This is
a GNSS/INS-derived reference sharing receiver measurements, not independent
truth for every GNSS error source.

The complete 77,647,402-byte `applanix/gps_post_process.csv` was downloaded
(215,097 rows). URL, length, ETag and SHA-256 are recorded in
`output/boreas-reference-audit/reference_download.json`.

| Field / file | Verified interpretation |
| --- | --- |
| `gps_post_process.csv: GPSTime` | Unix **seconds**, interpreted as UTC under the dataset convention; median interval 0.0050001 s |
| `radar_poses.csv: GPSTime` | Unix **microseconds**, UTC under the SDK convention |
| Postprocessed `latitude`, `longitude` | Radians; first values convert to approximately 43.78214756°, −79.46611149°, consistent with the Toronto route |
| `easting`, `northing`, `altitude` | Local ENU displacement; `altitude` is not absolute ellipsoid/MSL height |
| NMEA GGA/ZDA | UTC from checked ZDA dates; GGA MSL altitude plus geoid separation gives ellipsoid height when both exist |
| Raw Applanix Message 20 time-type byte | 1: time-1 configured as GPS time, time-2 as POS time under the reviewed schema; raw time-1 must not be silently treated as UTC |

The `GPSTime` column name alone does not justify adding an 18-second leap
offset. File-specific units and the documented synchronization convention are
required. The check below establishes consistency between reference products;
it does not independently measure hardware clock accuracy.

## Fixed extrinsic / timestamp consistency check

Downloaded official `T_applanix_lidar.txt` and `T_radar_lidar.txt` and formed:

```text
T_applanix_radar = T_applanix_lidar @ inverse(T_radar_lidar)
radar-origin translation in Applanix coordinates = [0, 0, 0.495] m
world_radar_origin = world_applanix_origin + R_world_applanix @ [0, 0, 0.495]
```

The rotation follows the SDK convention `Rx(-roll) Ry(-pitch) Rz(-yaw)`.
Interpolate translation and unwrapped Euler angles from the 200 Hz reference,
using only bracketed epochs with reference gaps <=0.02 s. Do not extrapolate.
There is no fitted spatial alignment, lever-arm adjustment or time-shift search.

| Fixed diagnostic time offset | Paired radar poses | RMS position discrepancy | Maximum discrepancy |
| --- | ---: | ---: | ---: |
| 0 s, documented convention | 4,142 | 0.00000743 m | 0.0000537 m |
| −18 s control | 4,142 | 148.81 m | 295.08 m |
| +18 s control | 4,142 | 148.84 m | 295.08 m |

All radar epochs are bracketed and none are excluded. The small zero-offset
discrepancy confirms compatible units, rotations, rigid translation and
reference interpolation. It is expected for two products derived from the same
postprocessing and **is not micrometre-level measured positioning accuracy**.
The ±18 s controls expose the consequence of an unjustified extra GPS/UTC
conversion; they are not candidate shifts tuned against native GNSS errors.

## GNSS antenna placement: configuration read, axes not established

The strict frame reader checks lengths, end markers and checksums before
reading Message 20. Its 84-byte layout matches the reviewed NOAA POS MV driver
schema. All 63 Message-20 records across the two acquired raw chunks agree:

* Reference-to-IMU vector: `[0, 0, 0]` m.
* Reference-to-primary-GPS vector: approximately `[0.38, 0.52, 0]` m.

These are vectors in the receiver's **configured reference frame**. Equivalence
of its axes to the SDK's postprocessed Applanix pose axes has not been
established, so the antenna vector has not been applied as a correction to
create GNSS error statistics. A known vector magnitude is not a verified
world-frame correction. The earlier radar-pose CSV also describes the radar
origin, which must not be treated as the GNSS antenna position.

## Weather: still unverified

Neither the acquired sequence's S3 root nor the pinned SDK supplies an official
per-sequence weather tag for this recording. The paper confirms that Boreas
contains rain/snow sequences, but that does not classify this particular run.
The official metadata website `https://boreas.utias.utoronto.ca/` currently
returns a proxy CONNECT 403 in this environment. That is separate from the
working S3 data distribution.

After publication, runtime revision **11** and the executor's policy snapshot
both include this website host. Nevertheless, HTTPS CONNECT still returns 403
and an HTTP request returns `403 Domain forbidden`. A simultaneous Boreas S3
listing succeeds with HTTP 200. Therefore, a missing saved host entry no longer
explains the failure, and no further environment publication is requested.
The exact proxy denial cause is not established: retained diagnostic records
are empty and the current debug bundle is unavailable. A local DNS lookup also
failed, which does not establish the destination's public DNS state because
this environment uses a proxy for HTTP egress.

Evidence is saved in `output/boreas-reference-audit/website_access_diagnosis.json`.
Public SDK repository/search pages were also checked, without finding a
verifiable per-sequence weather label. Until official metadata can be checked,
the recording remains **weather unknown**, and is not used as a rain-labelled
experiment. Oxford's separately confirmed provider withdrawal does not affect
Boreas S3 acquisition.

## Reproduction

With the downloaded reference, extrinsics and prior input files:

```bash
PYTHONPATH=python:. python experiments/audit_boreas_reference.py --raw \
  output/public-gnss-data/boreas-2020-11-26-13-58/default1126_185009.000 \
  output/public-gnss-data/boreas-2020-11-26-13-58/default1126_185009.001
pytest -q tests/test_import_boreas_gnss.py tests/test_audit_boreas_reference.py
```

Outputs: `output/boreas-reference-audit/reference_audit.json` and
`sensor_position_consistency.csv`, including input/code hashes, all radar
epochs, fixed clock controls and installation settings. **12 tests pass**,
including attitude-dependent lever arms, angle wrap, no extrapolation, gap
rejection, duplicate reference epochs and the original strict import checks.
