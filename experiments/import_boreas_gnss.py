"""Extract native receiver GGA/ZDA from checksum-verified Applanix group 10001.

Group layout reference: NOAA HSTB/drivers/pos_mv/PCSclassesV5R10.py,
Primary_GPS_data_stream (group 10001). Independently implemented narrow reader;
does not decode RT27 observations or interpret receiver status as true error.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def applanix_frames(content):
    """Yield complete validated frames without resynchronizing past corruption."""
    offset = 0
    while offset < len(content):
        if len(content)-offset < 8 or content[offset:offset+4] not in (b"$GRP", b"$MSG"):
            raise ValueError(f"invalid/truncated Applanix header at {offset}")
        kind = content[offset:offset+4]
        identifier, size = struct.unpack_from("<HH", content, offset+4)
        end = offset+8+size
        packet = content[offset:end]
        if end > len(content) or size < 4 or len(packet) % 2 or packet[-2:] != b"$#":
            raise ValueError(f"invalid/truncated Applanix frame at {offset}")
        if sum(struct.unpack("<"+"H"*(len(packet)//2), packet)) & 0xffff:
            raise ValueError(f"Applanix checksum mismatch at {offset}")
        yield kind, identifier, packet
        offset = end


def primary_stream(content):
    payload, counts = bytearray(), Counter()
    for kind, identifier, packet in applanix_frames(content):
        counts[f"{kind.decode()}:{identifier}"] += 1
        if kind == b"$GRP" and identifier == 10001:
            if len(packet) < 46:
                raise ValueError("primary GPS frame too short")
            length = struct.unpack_from("<H", packet, 40)[0]
            if 42+length > len(packet)-4 or len(packet)-4-(42+length) > 3:
                raise ValueError("primary GPS payload length/padding mismatch")
            payload.extend(packet[42:42+length])
    return bytes(payload), counts


def nmea_fields(sentence):
    body, encoded = sentence[1:].rsplit(b"*", 1)
    checksum = 0
    for byte in body:
        checksum ^= byte
    if checksum != int(encoded, 16):
        raise ValueError("NMEA checksum mismatch")
    return body.decode("ascii").split(",")


def coordinate(value, hemisphere, latitude):
    if not value or not hemisphere:
        return None
    number = float(value)
    degrees = int(number//100)
    minutes = number-100*degrees
    allowed = ("N", "S") if latitude else ("E", "W")
    result = degrees+minutes/60
    if (hemisphere not in allowed or not math.isfinite(number) or number < 0
            or not 0 <= minutes < 60 or result > (90 if latitude else 180)):
        raise ValueError("invalid NMEA coordinate")
    return -result if hemisphere in ("S", "W") else result


def receiver_fixes(stream):
    messages = [nmea_fields(m.group()) for m in re.finditer(
        rb"\$[A-Z]{2}(?:GGA|ZDA),[^\r\n$]*?\*[0-9A-Fa-f]{2}", stream)]
    anchors = []
    for fields in messages:
        if fields[0].endswith("ZDA"):
            timestamp = datetime.strptime(
                f"{fields[4]}-{fields[3]}-{fields[2]} {fields[1]}", "%Y-%m-%d %H%M%S.%f").replace(tzinfo=timezone.utc)
            anchors.append(timestamp)
    if not anchors:
        raise ValueError("checksum-verified ZDA date required")
    anchor = anchors[0]
    rows, seen = [], set()
    for fields in messages:
        if fields[0].endswith("ZDA"):
            anchor = datetime.strptime(
                f"{fields[4]}-{fields[3]}-{fields[2]} {fields[1]}", "%Y-%m-%d %H%M%S.%f").replace(tzinfo=timezone.utc)
        if not fields[0].endswith("GGA"):
            continue
        stamp = datetime.strptime(anchor.strftime("%Y-%m-%d")+" "+fields[1], "%Y-%m-%d %H%M%S.%f").replace(tzinfo=timezone.utc)
        if stamp-anchor > timedelta(hours=12):
            stamp -= timedelta(days=1)
        elif anchor-stamp > timedelta(hours=12):
            stamp += timedelta(days=1)
        # A segment may end between GGA and ZDA. Carry the checked date for
        # adjacent seconds only, never guess a date from the file name.
        if abs(stamp-anchor) > timedelta(seconds=2):
            raise ValueError("GGA too far from verified ZDA date anchor")
        if stamp in seen:
            raise ValueError("duplicate GGA epoch")
        seen.add(stamp)
        lat, lon = coordinate(fields[2], fields[3], True), coordinate(fields[4], fields[5], False)
        quality = int(fields[6])
        msl = float(fields[9]) if fields[9] else None
        geoid = float(fields[11]) if fields[11] else None
        if (msl is not None and fields[10] != "M") or (geoid is not None and fields[12] != "M"):
            raise ValueError("NMEA altitude units must be metres")
        rows.append(dict(time_utc=stamp.isoformat(), latitude_deg=lat, longitude_deg=lon,
                         altitude_msl_m=msl, geoid_separation_m=geoid,
                         ellipsoid_height_m=msl+geoid if msl is not None and geoid is not None else None,
                         quality_code=quality, satellites_used=int(fields[7]) if fields[7] else None,
                         hdop=float(fields[8]) if fields[8] else None,
                         receiver_reports_fix=quality > 0 and lat is not None and lon is not None))
    if not rows or any(rows[i]["time_utc"] >= rows[i+1]["time_utc"] for i in range(len(rows)-1)):
        raise ValueError("nonempty chronological GGA recording required")
    return rows


def status_intervals(rows, max_gap_s=1.5):
    """Describe contiguous native quality reports; never bridge missing epochs.

    Durations cover first-to-last observed timestamps only. One observed fix
    has zero span, not an assumed one-second validity interval.
    """
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("positive finite max_gap_s required")
    groups = []
    for row in rows:
        if groups:
            previous = groups[-1][-1]
            gap = (datetime.fromisoformat(row["time_utc"])-datetime.fromisoformat(previous["time_utc"])).total_seconds()
            if gap <= 0:
                raise ValueError("strictly chronological status records required")
        if (not groups or row["quality_code"] != groups[-1][-1]["quality_code"]
                or gap > max_gap_s):
            groups.append([])
        groups[-1].append(row)
    output = []
    for group in groups:
        satellites = [r["satellites_used"] for r in group if r["satellites_used"] is not None]
        hdop = [r["hdop"] for r in group if r["hdop"] is not None]
        output.append(dict(start_utc=group[0]["time_utc"], end_utc=group[-1]["time_utc"],
                           quality_code=group[0]["quality_code"], observed_epochs=len(group),
                           observed_span_s=(datetime.fromisoformat(group[-1]["time_utc"])-datetime.fromisoformat(group[0]["time_utc"])).total_seconds(),
                           satellites_min=min(satellites, default=None), satellites_max=max(satellites, default=None),
                           hdop_min=min(hdop, default=None), hdop_max=max(hdop, default=None)))
    return output


def run(inputs, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    stream, counts, sources = bytearray(), Counter(), []
    for path in map(Path, inputs):
        content = path.read_bytes()
        chunk, frame_counts = primary_stream(content)
        stream.extend(chunk); counts.update(frame_counts)
        sources.append(dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    rows = receiver_fixes(bytes(stream))
    intervals = status_intervals(rows)
    with (output/"status_intervals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(intervals[0])); writer.writeheader(); writer.writerows(intervals)
    with (output/"receiver_gga.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"primary_gps.bin").write_bytes(stream)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    start = datetime.fromisoformat(rows[0]["time_utc"])
    minutes = [(datetime.fromisoformat(r["time_utc"])-start).total_seconds()/60 for r in rows]
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6), layout="constrained")
    for ax, key, label in zip(axes, ("quality_code", "satellites_used", "hdop"),
                              ("Native quality code", "Satellites used", "HDOP")):
        ax.step(minutes, [r[key] for r in rows], where="post")
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[-1].set_xlabel("Minutes since first extracted GGA")
    figure.suptitle("Boreas receiver-reported status; no weather attribution or true-error claim")
    figure.savefig(output/"receiver_status.png", dpi=160)
    plt.close(figure)
    summary = dict(inputs=sources, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   checksummed_applanix_frames=sum(counts.values()), frame_counts=dict(counts),
                   primary_stream_bytes=len(stream), primary_stream_sha256=hashlib.sha256(stream).hexdigest(),
                   gga_epochs=len(rows), start_utc=rows[0]["time_utc"], end_utc=rows[-1]["time_utc"],
                   quality_code_counts=dict(Counter(str(row["quality_code"]) for row in rows)),
                   receiver_reported_fix_epochs=sum(row["receiver_reports_fix"] for row in rows),
                   status_intervals=len(intervals), status_max_gap_s=1.5,
                   time_source="checksum-verified ZDA dates, GGA UTC times; <=2 s date carry with midnight rollover",
                   limitations=["Receiver-reported status/HDOP are not true error or certified integrity.",
                                "Rain/wetness label for this sequence has not been verified.",
                                "Primary stream includes RT27; RTKLIB 2.4.2 trial produced navigation only, no observations.",
                                "Postprocessed radar poses are GNSS/INS reference, not independent GNSS truth."])
    (output/"provenance.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("checksummed_applanix_frames", "gga_epochs", "start_utc", "end_utc", "quality_code_counts")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("output/boreas-gnss-import"))
    arguments = parser.parse_args()
    run(arguments.inputs, arguments.output)
