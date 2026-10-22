"""Export cached official CityGML, UrbanNav and GSIGEO data for the browser.

No mesh simplification or coordinate quantization is applied. Shapely >= 2.1
is needed only when rebuilding the static mesh, not by the browser or server.
The GSI grid is used offline for research and is not redistributed in the site.

Rebuild with Python >= 3.10 and shapely==2.1.2. Download the sources named in
docs/assets/plateau/ATTRIBUTION.txt, put the 28 Odaiba *.gml tiles in
data/plateau/odaiba/, reference.csv and base.nav in
data/urbannav/Tokyo/Odaiba/, and gsigeo2011_ver2_1.asc in data/geoid/.
Then run: python tools/build_plateau_browser_scene.py
All input/output paths can also be supplied through the CLI flags.
The browser uses the checked-in compressed outputs; CI tests do not download
the original multi-gigabyte city archive or rebuild the full city.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET

GML = "{http://www.opengis.net/gml}"
XLINK = "{http://www.w3.org/1999/xlink}href"
KINDS = {"Building", "BuildingPart", "Bridge", "BridgePart"}
SURFACES = {"WallSurface": 1, "RoofSurface": 2, "GroundSurface": 3}
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def allowed(path):
    path = Path(path)
    if ".claude" in (part.casefold() for part in path.parts):
        raise ValueError("Excluded source path")
    return path


def sha(path):
    with allowed(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ecef(lat, lon, height):
    lat, lon = math.radians(lat), math.radians(lon)
    e2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)
    n = 6378137 / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return (
        (n + height) * math.cos(lat) * math.cos(lon),
        (n + height) * math.cos(lat) * math.sin(lon),
        (n * (1 - e2) + height) * math.sin(lat),
    )


def frame(lat, lon):
    lat, lon = math.radians(lat), math.radians(lon)
    return [
        (-math.sin(lon), math.cos(lon), 0),
        (-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)),
        (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)),
    ]


class GeoidGrid:
    def __init__(self, path):
        with allowed(path).open(encoding="ascii") as stream:
            header = stream.readline().split()
            self.lat0, self.lon0, dy, self.dx = map(float, header[:4])
            self.ny, self.nx = map(int, header[4:6])
            # GSI's one arc-minute spacing is rounded in the ASCII header.
            self.dy = 1 / 60 if abs(dy - 1 / 60) < 1e-6 else dy
            self.values = array(
                "d", (float(v) for line in stream for v in line.split())
            )
        if len(self.values) != self.nx * self.ny:
            raise ValueError("Incomplete geoid grid")

    def __call__(self, lat, lon):
        y, x = (lat - self.lat0) / self.dy, (lon - self.lon0) / self.dx
        iy, ix = math.floor(y), math.floor(x)
        if not (0 <= ix < self.nx - 1 and 0 <= iy < self.ny - 1):
            raise ValueError("Point outside geoid grid")
        cells = [
            self.values[(iy + dy) * self.nx + ix + dx]
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
        ]
        if any(not math.isfinite(v) or abs(v) > 200 for v in cells):
            raise ValueError(f"Missing geoid cell at {lat}, {lon}")
        fy, fx = y - iy, x - ix
        return (1 - fy) * ((1 - fx) * cells[0] + fx * cells[1]) + fy * (
            (1 - fx) * cells[2] + fx * cells[3]
        )


def local(tag):
    return tag.rsplit("}", 1)[-1]


def ring_points(ring):
    pos = ring.find(f"{GML}posList")
    values = (
        [float(v) for v in pos.text.split()]
        if pos is not None
        else [float(v) for p in ring.findall(f"{GML}pos") for v in p.text.split()]
    )
    if len(values) % 3:
        raise ValueError("Non-3D polygon ring")
    points = [tuple(values[i : i + 3]) for i in range(0, len(values), 3)]
    if points and points[0] == points[-1]:
        points.pop()
    return [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]


def triangulate(rings, stats=None):
    """Constrained triangulation of a 3D face, preserving its holes/vertices."""
    from shapely import Polygon, constrained_delaunay_triangles, make_valid

    if stats is None:
        stats = Counter()

    outer = rings[0]
    if len(outer) < 3:
        return []
    n = [
        sum(
            (a[(j + 1) % 3] - b[(j + 1) % 3]) * (a[(j + 2) % 3] + b[(j + 2) % 3])
            for a, b in zip(outer, outer[1:] + outer[:1])
        )
        for j in range(3)
    ]
    axis = max(range(3), key=lambda i: abs(n[i]))
    if abs(n[axis]) < 1e-12:
        return []
    axes = [i for i in range(3) if i != axis]
    lookup = {}
    projected = []
    for ring in rings:
        flat = []
        for point in ring:
            key = tuple(point[i] for i in axes)
            if key in lookup and lookup[key] != point:
                raise ValueError("Ambiguous face projection")
            lookup[key] = point
            flat.append(key)
        projected.append(flat)
    polygon = Polygon(projected[0], projected[1:])
    if polygon.is_empty or polygon.area < 1e-12:
        return []
    if not polygon.is_valid:
        polygon = make_valid(polygon, method="linework")
        stats["topology_repaired_faces"] += 1

    def polygon_parts(geometry):
        if geometry.geom_type == "Polygon":
            yield geometry
        elif hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                yield from polygon_parts(part)

    def restore(point):
        key = tuple(point)
        if key in lookup:
            return lookup[key]
        candidates = []
        for ring, flat in zip(rings, projected):
            for i, a in enumerate(flat):
                b = flat[(i + 1) % len(flat)]
                delta = [b[j] - a[j] for j in range(2)]
                norm = sum(v * v for v in delta)
                if norm == 0:
                    continue
                t = sum((key[j] - a[j]) * delta[j] for j in range(2)) / norm
                if (
                    -1e-10 <= t <= 1 + 1e-10
                    and math.hypot(*(key[j] - a[j] - t * delta[j] for j in range(2)))
                    < 1e-6
                ):
                    candidates.append(
                        tuple(
                            ring[i][j] + t * (ring[(i + 1) % len(ring)][j] - ring[i][j])
                            for j in range(3)
                        )
                    )
        if not candidates or any(
            math.dist(candidates[0], other) > 0.001 for other in candidates[1:]
        ):
            raise ValueError("Ambiguous nonplanar source-boundary intersection")
        lookup[key] = candidates[0]
        stats["inserted_boundary_vertices"] += 1
        return candidates[0]

    triangles = []
    total = 0.0
    for part in polygon_parts(polygon):
        for tri in constrained_delaunay_triangles(part).geoms:
            points = [restore(p) for p in list(tri.exterior.coords)[:3]]
            triangles.append(points)
            total += tri.area
    if abs(total - polygon.area) > max(1e-7, polygon.area * 1e-10):
        raise ValueError("Triangulation changed face area")
    return triangles


def read_navigation(path, start, stop, week, systems="GECJR"):
    with allowed(path).open(encoding="ascii") as stream:
        lines = iter(stream)
        gpga = None
        leap_seconds = None
        for line in lines:
            if line.startswith("GPGA") and "TIME SYSTEM CORR" in line:
                gpga = [float(line[5:22].replace("D", "E")), float(line[22:38].replace("D", "E")), int(line[38:45]), int(line[45:50])]
            if "LEAP SECONDS" in line:
                leap_seconds = int(line[:6])
            if "END OF HEADER" in line:
                break
        records = []
        for line in lines:
            if not line.strip():
                continue
            system = line[0]
            block = [next(lines) for _ in range(3 if system in "RS" else 7)]
            if system not in systems:
                continue
            fields = [
                float(
                    row[4 + 19 * j : 4 + 19 * (j + 1)].replace("D", "E").strip() or "0"
                )
                for row in block
                for j in range(4)
            ]
            stamp = list(map(int, line[4:23].split()))
            toc_abs = (
                datetime(*stamp, tzinfo=timezone.utc) - GPS_EPOCH
            ).total_seconds()
            clocks = [float(line[23 + 19 * i:23 + 19 * (i + 1)].replace("D", "E")) for i in range(3)]
            if system == "R":
                # This archive has no LEAP SECONDS header. Its 2018 UTC epochs
                # have GPST-UTC=18 s; other dates require explicit file metadata.
                if leap_seconds is None and stamp[0] != 2018:
                    raise ValueError("GLONASS outside the 2018 fixture requires LEAP SECONDS")
                offset = 18 if leap_seconds is None else leap_seconds
                if not 0 <= offset <= 64 or (stamp[0] == 2018 and offset != 18):
                    raise ValueError("Invalid GLONASS UTC offset for the recorded epoch")
                transmission = math.floor(toc_abs / 604800) * 604800 + clocks[2]
                if transmission - toc_abs > 302400:
                    transmission -= 604800
                if transmission - toc_abs < -302400:
                    transmission += 604800
                if (not all(math.isfinite(v) for v in fields + clocks) or fields[3] != 0 or clocks[2] == 0.9999e9
                    or toc_abs + offset < week * 604800 + start - 900 or toc_abs + offset > week * 604800 + stop + 900):
                    continue
                records.append({"id": line[:3], "toc": toc_abs, "week": int(toc_abs // 604800), "utcOffset": offset,
                    "clock": clocks, "position": [fields[k] * 1000 for k in (0, 4, 8)],
                    "velocity": [fields[k] * 1000 for k in (1, 5, 9)], "acceleration": [fields[k] * 1000 for k in (2, 6, 10)],
                    "health": fields[3], "frequencyChannel": int(fields[7]), "transmission": transmission})
                continue
            toe_abs = (int(fields[18]) + (1356 if system == "C" else 0)) * 604800 + fields[8] + (14 if system == "C" else 0)
            healthy = (int(fields[17]) & 5 and int(fields[17]) & 512 and int(fields[21]) & 7 == 0 and fields[20] >= 0) if system == "E" else fields[21] == 0
            maximum_age = 14400 if system == "E" else 3600 if system == "C" else 7200
            if (
                not healthy
                or toe_abs < week * 604800 + start - maximum_age
                or toe_abs > week * 604800 + stop + maximum_age
            ):
                continue
            clocks = [
                float(line[23 + 19 * i : 23 + 19 * (i + 1)].replace("D", "E"))
                for i in range(3)
            ]
            record = {"id": line[:3], "toc": toc_abs, "clock": clocks, "parameters": fields}
            if system == "E":
                if gpga is None:
                    raise ValueError("Galileo requires GPGA time correction")
                record["gpga"] = gpga
            records.append(record)
    if not records:
        raise ValueError("No matching healthy GNSS broadcast records")
    return records


def compress(path, payload):
    data = bytearray(gzip.compress(payload, compresslevel=9, mtime=0))
    data[9] = 255  # No platform-specific gzip header.
    path.write_bytes(data)
    return {
        "file": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
    }


def build(gml_dir, reference, nav, geoid, output):
    output = allowed(output)
    output.mkdir(parents=True, exist_ok=True)
    with allowed(reference).open() as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    track = [
        [
            float(row[k])
            for k in ("GPS TOW (s)", "ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)")
        ]
        for row in rows
    ]
    weeks = {int(row["GPS Week"]) for row in rows}
    if len(weeks) != 1:
        raise ValueError("Track crosses GPS week boundary")
    week = weeks.pop()
    origin = track[0][1:]
    rotation = frame(
        float(rows[0]["Latitude (deg)"]), float(rows[0]["Longitude (deg)"])
    )
    grid = GeoidGrid(geoid)
    positions, indices, owners, types = array("d"), array("I"), array("I"), bytearray()
    vertices = {}
    objects, sources, stats = [], [], Counter()
    geoid_values = []
    converted = {}

    def convert(point):
        if point in converted:
            return converted[point]
        correction = grid(point[0], point[1])
        geoid_values.append(correction)
        xyz = ecef(point[0], point[1], point[2] + correction)
        delta = [xyz[i] - origin[i] for i in range(3)]
        converted[point] = tuple(
            sum(axis[i] * delta[i] for i in range(3)) for axis in rotation
        )
        return converted[point]

    files = sorted(allowed(gml_dir).glob("*.gml"))
    if not files:
        raise ValueError("No cached CityGML files")
    for path in files:
        allowed(path)
        root = ET.parse(path).getroot()
        envelope = root.find(f".//{GML}Envelope")
        if envelope is None or not envelope.get("srsName", "").endswith("/6697"):
            raise ValueError(f"Unsupported CityGML CRS in {path.name}")
        parent = {child: node for node in root.iter() for child in node}
        own = {}
        for polygon in root.iter(f"{GML}Polygon"):
            node, lod, category = polygon, 0, 0
            while node is not root and local(node.tag) not in KINDS:
                category = SURFACES.get(local(node.tag), category)
                match = re.match(r"lod([1-4])", local(node.tag))
                if match:
                    lod = max(lod, int(match[1]))
                node = parent[node]
            if local(node.tag) in KINDS and lod:
                own.setdefault(node, []).append((lod, category, polygon))
        file_faces = 0
        for feature, polygons in own.items():
            lod = max(item[0] for item in polygons)
            object_id = len(objects)
            is_bridge = local(feature.tag).startswith("Bridge")
            obj = {
                "id": feature.get(f"{GML}id"),
                "kind": "bridge" if is_bridge else "building",
                "lod": lod,
                "source": path.name,
                "triangles": 0,
            }
            for level, category, polygon in polygons:
                if level != lod:
                    continue
                rings = []
                for kind in ("exterior", "interior"):
                    for ring in polygon.findall(f"{GML}{kind}/{GML}LinearRing"):
                        rings.append([convert(point) for point in ring_points(ring)])
                if not rings:
                    raise ValueError("Polygon without materialized rings")
                try:
                    triangles = triangulate(rings, stats)
                except ValueError as error:
                    raise ValueError(
                        f"{path.name} polygon {polygon.get(f'{GML}id')}: {error}"
                    ) from error
                stats["polygons"] += 1
                stats["holes"] += len(rings) - 1
                stats["degenerate_faces"] += not triangles
                file_faces += 1
                for triangle in triangles:
                    for point in triangle:
                        if point not in vertices:
                            vertices[point] = len(positions) // 3
                            positions.extend(point)
                        indices.append(vertices[point])
                    owners.append(object_id)
                    types.append(4 if is_bridge else category)
                    obj["triangles"] += 1
            objects.append(obj)
        sources.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha(path),
                "faces": file_faces,
            }
        )
        print(path.name, file_faces, "faces", flush=True)
    bounds = [
        [min(positions[i::3]) for i in range(3)],
        [max(positions[i::3]) for i in range(3)],
    ]
    if sys.byteorder != "little":
        for values in (positions, indices, owners):
            values.byteswap()
    payload = (
        b"PLTGEO1\0"
        + struct.pack("<II", len(positions) // 3, len(owners))
        + positions.tobytes()
        + indices.tobytes()
        + owners.tobytes()
        + types
    )
    mesh = compress(output / "odaiba-mesh.bin.gz", payload)
    scene = {
        "version": 1,
        "name": "Tokyo · Odaiba",
        "origin": origin,
        "rotation": rotation,
        "week": week,
        "track": track,
        "navigation": read_navigation(nav, track[0][0], track[-1][0], week),
        "objects": objects,
        "bounds": bounds,
        "stats": dict(
            stats,
            vertices=len(vertices),
            triangles=len(owners),
            buildings=sum(o["kind"] == "building" for o in objects),
            bridges=sum(o["kind"] == "bridge" for o in objects),
            epochs=len(track),
        ),
        "provenance": {
            "city_model": "PLATEAU Tokyo 23 wards 2022, EPSG:6697, highest available LoD per component",
            "city_url": "https://www.geospatial.jp/ckan/dataset/plateau-tokyo23ku-2022",
            "city_license": "CC BY 4.0",
            "files": sources,
            "track_source": "UrbanNav-TK-20181219 / Odaiba, Applanix POS LV620 reference.csv (all epochs)",
            "navigation_source": "UrbanNav base.nav, DLR/TUM CONGO/MGEX merged RINEX 3.03 GPS/Galileo/BeiDou/QZSS/GLONASS broadcast ephemerides",
            "urbannav_url": "https://github.com/IPNL-POLYU/UrbanNavDataset",
            "reference_sha256": sha(reference),
            "navigation_sha256": sha(nav),
            "geoid": {
                "name": "GSIGEO2011 ver.2.1, bilinear, 1 arc-minute × 1.5 arc-minute",
                "source": "https://web1.gsi.go.jp/common/000275010.zip",
                "sha256": sha(geoid),
                "min_m": min(geoid_values),
                "max_m": max(geoid_values),
            },
            "geometry": "Float64 ENU; highest LoD per component; constrained triangulation preserving holes. Self-touching source rings use GEOS make_valid(linework); any new boundary intersections interpolate original 3D edges. Repair counts are reported. No decimation or quantization.",
            "epochs": "2018 navigation/trajectory and 2022 city model are a cross-year reconstruction; dynamic objects are not represented.",
        },
    }
    info = compress(
        output / "odaiba-scene.json.gz",
        json.dumps(
            scene, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode(),
    )
    manifest = {
        "version": 1,
        "mesh": mesh,
        "scene": info,
        "stats": scene["stats"],
        "provenance": scene["provenance"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest



def update_navigation(nav, output):
    """Refresh broadcast records without regenerating or rounding city geometry."""
    output = allowed(output)
    manifest_path = output / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    path = output / manifest["scene"]["file"]
    if sha(path) != manifest["scene"]["sha256"]:
        raise ValueError("Existing scene checksum differs")
    scene = json.loads(gzip.decompress(path.read_bytes()))
    scene["navigation"] = read_navigation(nav, scene["track"][0][0], scene["track"][-1][0], scene["week"])
    scene["provenance"]["navigation_source"] = "UrbanNav base.nav, DLR/TUM CONGO/MGEX merged RINEX 3.03 GPS/Galileo/BeiDou/QZSS/GLONASS broadcast ephemerides"
    scene["provenance"]["navigation_sha256"] = sha(nav)
    manifest["scene"] = compress(path, json.dumps(scene, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
    manifest["provenance"] = scene["provenance"]
    newline = "\r\n" if b"\r\n" in original_manifest else "\n"
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").replace("\n", newline).encode())
    print(dict(Counter(r["id"][0] for r in scene["navigation"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gml-dir", type=Path, default=Path("data/plateau/odaiba"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/urbannav/Tokyo/Odaiba/reference.csv"),
    )
    parser.add_argument(
        "--nav", type=Path, default=Path("data/urbannav/Tokyo/Odaiba/base.nav")
    )
    parser.add_argument(
        "--geoid", type=Path, default=Path("data/geoid/gsigeo2011_ver2_1.asc")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/assets/plateau"))
    parser.add_argument("--navigation-only", action="store_true", help="Refresh navigation in an existing verified scene; keep the mesh unchanged")
    args = parser.parse_args()
    if args.navigation_only:
        update_navigation(args.nav, args.output)
        return
    build(args.gml_dir, args.reference, args.nav, args.geoid, args.output)


if __name__ == "__main__":
    main()
