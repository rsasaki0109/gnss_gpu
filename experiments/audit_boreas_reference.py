"""Verify Boreas reference units and rigid sensor-position consistency.

This compares two products of the same postprocessing pipeline, not truth
accuracy. Uses fixed official extrinsics, no fitted alignment or time offset.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from experiments.import_boreas_gnss import applanix_frames


def sensor_rotation(roll, pitch, yaw):
    """Boreas SDK convention: Rx(-roll) Ry(-pitch) Rz(-yaw)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1., 0., 0.], [0., cr, sr], [0., -sr, cr]])
    ry = np.array([[cp, 0., -sp], [0., 1., 0.], [sp, 0., cp]])
    rz = np.array([[cy, sy, 0.], [-sy, cy, 0.], [0., 0., 1.]])
    return rx@ry@rz


def predict_sensor_positions(reference, query_seconds, sensor_in_reference, max_gap_s=.02):
    """Interpolate only bracketed epochs, with a fixed gap limit; never extrapolate.

    Reference columns: UTC seconds, xyz, roll/pitch/yaw (radians). Translation
    is in the reference sensor frame and must be rotated at each query epoch.
    """
    reference, query = np.asarray(reference), np.asarray(query_seconds)
    times = reference[:, 0]
    if (not np.all(np.isfinite(reference)) or not np.all(np.isfinite(query))
            or np.any(np.diff(times) <= 0) or len(times) < 2):
        raise ValueError("finite chronological reference required")
    after = np.searchsorted(times, query, side="right")
    before = np.clip(after-1, 0, len(times)-2)
    after = before+1
    valid = (query >= times[0]) & (query <= times[-1]) & (times[after]-times[before] <= max_gap_s)
    state = reference[:, 1:7].copy()
    state[:, 3:6] = np.unwrap(state[:, 3:6], axis=0)
    interpolated = np.column_stack([np.interp(query[valid], times, state[:, i]) for i in range(6)])
    predicted = np.full((len(query), 3), np.nan)
    if np.any(valid):
        predicted[valid] = [row[:3]+sensor_rotation(*row[3:6])@sensor_in_reference for row in interpolated]
    return predicted, valid


def installation_settings(paths):
    """Read only the exact 84-byte Message 20 layout reviewed in NOAA's schema."""
    settings = Counter()
    for path in paths:
        for kind, identifier, packet in applanix_frames(Path(path).read_bytes()):
            if kind == b"$MSG" and identifier == 20:
                if len(packet) != 92:
                    raise ValueError("unsupported Message 20 layout")
                imu = struct.unpack_from("<3f", packet, 13)
                antenna = struct.unpack_from("<3f", packet, 25)
                settings[(imu, antenna, packet[10])] += 1
    return [dict(reference_to_imu_m=list(key[0]), reference_to_primary_gps_m=list(key[1]),
                 time_type_byte=key[2], occurrences=count) for key, count in settings.items()]


def run(directory, radar_path, raw_paths):
    directory, radar_path = Path(directory), Path(radar_path)
    reference_path = directory/"gps_post_process.csv"
    reference = np.loadtxt(reference_path, delimiter=",", skiprows=1,
                           usecols=(0, 1, 2, 3, 7, 8, 9, 16, 17))
    radar = np.loadtxt(radar_path, delimiter=",", skiprows=1)
    al_path, rl_path = directory/"T_applanix_lidar.txt", directory/"T_radar_lidar.txt"
    a_l, r_l = np.loadtxt(al_path), np.loadtxt(rl_path)
    for matrix in (a_l, r_l):
        if (matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.)):
            raise ValueError("proper homogeneous rigid extrinsics required")
    a_r = a_l@np.linalg.inv(r_l)
    comparisons = {}
    # Fixed controls test a mistaken GPS/UTC leap-second adjustment; no search
    # over shifts and no tuning to the native GGA errors is performed.
    for offset in (0., -18., 18.):
        predicted, valid = predict_sensor_positions(reference[:, :7], radar[:, 0]/1e6+offset, a_r[:3, 3])
        delta = np.linalg.norm(predicted[valid]-radar[valid, 1:4], axis=1)
        comparisons[str(offset)] = dict(pairs=int(sum(valid)), excluded=int(sum(~valid)),
                                        rms_m=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
                                        max_m=float(max(delta)) if len(delta) else None)
        if offset == 0.:
            rows = [dict(radar_time_utc_us=int(radar[i, 0]), reference_bracket_valid=bool(valid[i]),
                         sensor_position_consistency_m=float(np.linalg.norm(predicted[i]-radar[i, 1:4])) if valid[i] else None)
                    for i in range(len(radar))]
    with (directory/"sensor_position_consistency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paths = [reference_path, radar_path, al_path, rl_path, *map(Path, raw_paths)]
    report = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  reference_rows=len(reference), reference_median_interval_s=float(np.median(np.diff(reference[:, 0]))),
                  reference_first_latitude_longitude_deg=list(np.rad2deg(reference[0, -2:])),
                  reference_units=dict(gps_post_process_time="Unix seconds, treated as UTC per dataset convention",
                                       radar_pose_time="Unix microseconds, UTC per official documentation",
                                       latitude_longitude="radians; easting/northing/altitude are local ENU, not geodetic height"),
                  T_applanix_radar=a_r.tolist(), fixed_clock_offset_controls_s=comparisons,
                  installation_message_20=installation_settings(raw_paths),
                  limitations=["Internal consistency is not independent positioning accuracy.",
                               "GNSS/INS reference shares measurements with the native receiver.",
                               "Message 20 vectors are in the configured reference frame; mapping to the SDK pose frame remains unverified.",
                               "Weather label could not be verified from accessible sources; official metadata website returns proxy 403."])
    (directory/"reference_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("reference_rows", "reference_median_interval_s", "fixed_clock_offset_controls_s", "installation_message_20")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/boreas-reference-audit"))
    parser.add_argument("--radar-poses", type=Path, default=Path("output/public-gnss-data/boreas-2020-11-26-13-58/radar_poses.csv"))
    parser.add_argument("--raw", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.radar_poses, args.raw)
