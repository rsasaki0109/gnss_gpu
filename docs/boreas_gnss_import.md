# Boreas native GNSS receiver records: acquired and imported

2026-10-24: after environment publication, runtime revision 9 and actual HTTP
requests confirm access to the RobotCar website and Boreas S3 distribution
(both return 200). The previous network block for these two hosts is resolved.

## Acquired data

From `boreas-2020-11-26-13-58`, downloaded:

* `applanix/radar_poses.csv`: 1,066,501 bytes, 4,142 postprocessed reference poses.
* `applanix/raw_logs/default1126_185009.000`: 13,094,600 bytes.
* `applanix/raw_logs/default1126_185009.001`: 13,093,844 bytes.

The full download URLs, lengths and SHA-256 digests are in
`output/public-gnss-data/boreas-2020-11-26-13-58/download_manifest.json`.
The official SDK's `DATA_LICENSE.md` states CC BY 4.0 for the dataset. No
third-party payload or decoder source is added to versioned project files.

## Import and checks

[The importer](../experiments/import_boreas_gnss.py) independently implements
the group-10001 layout documented by the NOAA HSTB POS MV driver:
`https://github.com/noaa-ocs-hydrography/drivers/blob/master/HSTB/drivers/pos_mv/PCSclassesV5R10.py`,
class `Primary_GPS_data_stream`. The fetched reference snapshot is saved at
`output/public-gnss-data/applanix_driver_reference.py.txt`.

It validates length-delimited `$GRP` and `$MSG` frames, their end markers and
16-bit checksums before concatenating the primary receiver's native stream.
It then validates NMEA XOR checksums and extracts GGA positions, native quality
codes, used-satellite counts and HDOP. UTC dates come from ZDA messages, not
the directory name. Date carry is limited to two seconds and handles midnight.
Missing fixes remain rows; HDOP and native quality codes are not true errors.
MSL altitude and geoid separation are retained separately; ellipsoid height is
their sum when both exist with metre units.

Observed import results:

| Check | Result |
| --- | --- |
| Complete checksum-verified Applanix frames | 331,572 |
| Extracted checksum-verified GGA epochs | 631 |
| UTC time span | 2020-11-26 18:49:50–19:00:20 |
| Native GGA quality-code counts | 1: 76; 2: 24; 4: 531 |
| Reference pose UTC extent | 18:58:17.528152–19:15:32.793469 |
| GGA timestamps inside reference extent | 123 |

All extracted epochs have a nonzero native quality code. This is a description
of the receiver reports in this snippet, not a measured success probability or
proof of accuracy. Quality code 4 is a receiver-reported solution category,
not an independent confirmation that its ambiguities are correct.

The reference CSV is postprocessed GNSS/INS in the dataset's ENU reference;
GGA is geodetic position from the primary receiver. UTC/GPS conventions,
reference origin and antenna/reference lever arms must be checked before
comparing position errors. No fitted alignment has been used to create an
apparent accuracy result. The acquired sequence's rain/wetness label is still
unverified, so no weather effect is inferred from status transitions.
The subsequent [reference audit](boreas_reference_audit.md) verifies file units,
reference sensor transforms and relative timestamp consistency, and reads the
antenna installation vector. Antenna axis mapping and weather labels remain open.

The native stream also contains Trimble RT27 messages. An exploratory RTKLIB
2.4.2 `convbin -r rt17` conversion produced navigation records but **no RINEX
observations**; that decoder documents no RT27 observation support. The native
stream is saved as `primary_gps.bin`, not mislabeled as decoded observations
or RF IQ. RF tracking and C/N0 validation therefore remain outstanding.

## Reproduction and artifacts

After acquiring the two source chunks, run in the configured CPU environment:

```bash
PYTHONPATH=python:. python experiments/import_boreas_gnss.py \
  output/public-gnss-data/boreas-2020-11-26-13-58/default1126_185009.000 \
  output/public-gnss-data/boreas-2020-11-26-13-58/default1126_185009.001
pytest -q tests/test_import_boreas_gnss.py
```

`output/boreas-gnss-import/` contains `receiver_gga.csv`, `receiver_status.png`,
the native `primary_gps.bin`, and provenance with source/input digests and
counts. `status_intervals.csv` now describes nine contiguous native-quality
intervals, splitting on quality changes or gaps above 1.5 s. Durations cover
first-to-last observed timestamps only; a single observation has zero span.
For example, quality code 2 is reported at 16 consecutive epochs from
18:59:18 through 18:59:33 UTC, with 12 used satellites and HDOP 0.9–1.0.
This is a receiver status transition, not proof of positioning error or rain.
Eight tests pass, covering corruption/truncation, embedded frame markers,
split native payloads, UTC midnight carry, altitude convention, no-fix
retention, duplicate epochs, stale/missing dates, NMEA checksum rejection,
gap splitting and observed-span duration semantics.

## Oxford download host discovered

The RobotCar rain-sequence page is now accessible. Its documented GPS archive
link for `2015-10-29-12-18-17` points to a **different host**:
`mrgdatashare.robots.ox.ac.uk`. The official documentation confirms `gps.csv`
contains GPS-only positions/uncertainty at 5 Hz and `ins.csv` fused GPS/INS at
50 Hz. The docs also warn that INS attitude headers say degrees but values
are radians. These payload fields have not yet been inspected locally.

After publication, runtime revision **10** includes the additional download
host in the active configuration. Actual HTTP and HTTPS requests still return
**503 upstream connection errors**. The official
[downloads page](https://robotcar-dataset.robots.ox.ac.uk/downloads/) was then
retrieved successfully and contains this notice:

> Notice 2026-10-22
> Please note that we are withdrawing the RobotCar Dataset until further notice.

The saved page and its checksum are recorded in the
[public-data source register](public_gnss_weather_data.json). The provider's
withdrawal is confirmed; the exact transport cause of each 503 is not known.
Further environment publication is not a remedy for provider availability.
Oxford is paused as a payload source. Boreas acquisition and the import above
already work; its weather labels and positioning reference still require
verification before weather/performance conclusions.
