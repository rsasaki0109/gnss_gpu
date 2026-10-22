"""Export cached UrbanNav rover Sxx observations without resampling or rounding."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
SATELLITE = re.compile(r"^([A-Z])\s*(\d{1,2})$")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_strength(path, week, start, end, receiver):
    """Narrow RINEX 3.02/3.03 reader; rejects unsupported time/unit/scaling cases."""
    types, expected, signals, counts = {}, {}, {}, Counter()
    header = {"producer": "", "unitDeclaration": "", "clockOffsetsApplied": 0}
    current = None
    with Path(path).open(encoding="ascii") as stream:
        first = stream.readline()
        version = float(first[:9])
        if version not in (3.02, 3.03) or first[20] != "O":
            raise ValueError("Only RINEX 3.02/3.03 observation files are supported")
        for line in stream:
            label = line[60:80].strip()
            if label == "END OF HEADER":
                break
            if label == "PGM / RUN BY / DATE":
                header["producer"] = line[:20].strip()
            elif label == "SIGNAL STRENGTH UNIT":
                header["unitDeclaration"] = line[:20].strip()
            elif label == "TIME OF FIRST OBS":
                header["timeSystem"] = line[48:51].strip()
            elif label == "RCV CLOCK OFFS APPL":
                header["clockOffsetsApplied"] = int(line[:6])
            elif label == "SYS / SCALE FACTOR" and int(line[2:6]) != 1:
                raise ValueError("Scaled observations are not supported")
            elif label == "SYS / # / OBS TYPES":
                if line[0] != " ":
                    current = line[0]
                    expected[current] = int(line[3:6])
                    types[current] = []
                if current is None:
                    raise ValueError("Orphan observation header continuation")
                types[current].extend(line[7:60].split())
        else:
            raise ValueError("Missing END OF HEADER")
        if header.get("timeSystem") != "GPS":
            raise ValueError("Observation epochs must explicitly use GPS time")
        if any(len(types[system]) != count for system, count in expected.items()):
            raise ValueError("Incomplete observation-code header")
        if header["unitDeclaration"] == "DBHZ":
            unit_status = "declared"
        elif not header["unitDeclaration"] and header["producer"] == "RTKCONV 2.4.3 b31":
            unit_status = "inferred-rtkconv"
        else:
            raise ValueError("Unknown signal-strength units; no implicit conversion")
        codes = {"G": "S1C", "R": "S1C", "E": "S1X" if receiver == "trimble" else "S1C", "J": "S1C", "C": "S1I" if version == 3.02 else "S2I"}
        indices = {s: types[s].index(c) for s, c in codes.items() if c in types.get(s, [])}
        pending = None

        def read():
            nonlocal pending
            if pending is not None:
                value, pending = pending, None
                return value
            return stream.readline()

        previous, epoch_times = None, []
        while line := read():
            if not line.strip():
                continue
            if not line.startswith(">"):
                raise ValueError("Expected observation epoch")
            fields = line[1:].split()
            flag, number = int(fields[6]), int(fields[7])
            if flag in (2, 3, 5):
                for _ in range(number):
                    if not read():
                        raise ValueError("Truncated event record")
                counts["events"] += 1
                continue
            if flag == 4:
                raise ValueError("Mid-file header updates are not supported")
            if flag not in (0, 1, 6):
                raise ValueError("Unsupported epoch flag")
            seconds = float(fields[5])
            if not math.isfinite(seconds) or not 0 <= seconds < 60:
                raise ValueError("Unsupported epoch seconds")
            date = datetime(*map(int, fields[:5]), tzinfo=timezone.utc)
            whole = int((date - GPS_EPOCH).total_seconds())
            source_week, remainder = divmod(whole, 604800)
            time = (source_week - week) * 604800 + remainder + seconds
            if len(fields) > 8 and float(fields[8]) != 0 and not header["clockOffsetsApplied"]:
                raise ValueError("Unapplied receiver clock offsets are not supported")
            if flag != 6:
                if previous is not None and time <= previous:
                    raise ValueError("Observation epochs must strictly increase")
                previous = time
                if start <= time <= end:
                    epoch_times.append(time)
            seen = set()
            for _ in range(number):
                record = read().rstrip("\r\n")
                match = SATELLITE.fullmatch(record[:3])
                if not match:
                    raise ValueError("Invalid or truncated satellite observation")
                system, prn = match.groups()
                satellite = f"{system}{int(prn):02d}"
                if satellite in seen or system not in types:
                    raise ValueError("Duplicate satellite or missing observation header")
                seen.add(satellite)
                while len(record) < 3 + 16 * len(types[system]):
                    continuation = read()
                    if continuation.startswith("   "):
                        record += continuation[3:].rstrip("\r\n")
                    else:
                        pending = continuation or None
                        break
                if flag == 6 or system not in indices or not start <= time <= end:
                    continue
                offset = 3 + 16 * indices[system]
                value = record[offset:offset + 14].strip()
                if not value or float(value) == 0:
                    counts["blankOrZero"] += 1
                    continue
                strength = float(value)
                if not math.isfinite(strength):
                    raise ValueError("Non-finite recorded signal strength")
                signals.setdefault(satellite, {"code": codes[system], "samples": []})["samples"].append([time, strength])
                counts[system] += 1
    if len(epoch_times) < 2:
        raise ValueError("No overlapping observation interval")
    cadence = statistics.median(b - a for a, b in zip(epoch_times, epoch_times[1:]))
    return {"version": 1, "receiver": receiver, "week": week, "unit": "dB-Hz", "unitStatus": unit_status, "header": header, "codes": codes, "cadence": cadence, "counts": dict(counts), "signals": dict(sorted(signals.items())), "source": {"file": Path(path).name, "sha256": digest(path), "bytes": Path(path).stat().st_size}, "timeAlignment": "Original GPS epochs; no interpolation, receiver-clock repair or time quantization", "bandPolicy": "One explicit first-band Sxx per system/receiver; no fallback to another tracking code. BeiDou 3.02 S1I is B1I, equivalent to 3.03 S2I."}


def build(source, output, scene_path):
    scene_bytes = Path(scene_path).read_bytes()
    scene = json.loads(gzip.decompress(scene_bytes))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "sceneSha256": hashlib.sha256(scene_bytes).hexdigest(), "referenceSha256": scene["provenance"]["reference_sha256"], "sourceURL": "https://github.com/IPNL-POLYU/UrbanNavDataset", "license": "UrbanNav-LICENSE.txt", "receivers": {}}
    for receiver in ("trimble", "ublox"):
        data = read_strength(source / f"rover_{receiver}.obs", scene["week"], scene["track"][0][0] - .002, scene["track"][-1][0] + .002, receiver)
        content = bytearray(gzip.compress(json.dumps(data, separators=(",", ":"), allow_nan=False).encode(), compresslevel=9, mtime=0))
        content[9] = 255  # Stable gzip OS byte across Python 3.12 platforms.
        content = bytes(content)
        name = f"odaiba-strength-{receiver}.json.gz"
        (output / name).write_bytes(content)
        manifest["receivers"][receiver] = {"file": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "samples": sum(len(s["samples"]) for s in data["signals"].values()), "unitStatus": data["unitStatus"], "source": data["source"]}
    (output / "strength-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/urbannav/Tokyo/Odaiba"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/plateau"))
    parser.add_argument("--scene", type=Path, default=Path("docs/assets/plateau/odaiba-scene.json.gz"))
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.scene)))
