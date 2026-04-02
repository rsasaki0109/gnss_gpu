#!/usr/bin/env python3
"""Fetch a trajectory-aligned PLATEAU building subset from a remote ZIP.

This script avoids downloading a full multi-GB PLATEAU archive by opening the
remote ZIP over HTTP range requests, identifying building GML files whose mesh
codes overlap a PPC trajectory, and extracting only those files locally.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import requests


TOKYO23_CITYGML_URL = (
    "https://assets.cms.plateau.reearth.io/assets/74/"
    "b317b5-a7b0-426f-ba8b-af5f777c76c5/"
    "13100_tokyo23-ku_2022_citygml_1_2_op.zip"
)
NAGOYA_CITYGML_URL = (
    "https://assets.cms.plateau.reearth.io/assets/79/"
    "e43a02-06b6-40c2-ae97-51eba1b4297b/"
    "23100_nagoya-shi_city_2022_citygml_4_op.zip"
)

PRESET_URLS = {
    "tokyo23": TOKYO23_CITYGML_URL,
    "nagoya": NAGOYA_CITYGML_URL,
}


class HTTPRangeReader(io.RawIOBase):
    """Minimal seekable reader backed by HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.size = self._resolve_size()
        self.pos = 0

    def _resolve_size(self) -> int:
        response = requests.head(self.url, allow_redirects=True, timeout=30)
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length is not None:
            return int(content_length)

        probe = requests.get(
            self.url,
            headers={"Range": "bytes=0-0"},
            allow_redirects=True,
            timeout=30,
        )
        probe.raise_for_status()

        content_range = probe.headers.get("content-range", "")
        if "/" in content_range:
            _, _, total = content_range.partition("/")
            if total.isdigit():
                return int(total)

        content_length = probe.headers.get("content-length")
        if content_length is not None:
            return int(content_length)

        raise RuntimeError(f"could not determine remote ZIP size for {self.url}")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            end = self.size - 1
        else:
            end = min(self.size - 1, self.pos + n - 1)
            if end < self.pos:
                return b""

        response = requests.get(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}"},
            allow_redirects=True,
            timeout=120,
        )
        response.raise_for_status()
        data = response.content
        self.pos += len(data)
        return data


def mesh3_code(lat_deg: float, lon_deg: float) -> str:
    """Return the standard Japanese third-level mesh code for a lat/lon."""
    lat_minutes = lat_deg * 60.0
    p = int(lat_minutes / 40.0)
    a = lat_minutes - p * 40.0
    q = int(lon_deg) - 100
    lon_frac = lon_deg - int(lon_deg)
    r = int(a / 5.0)
    s = int(lon_frac * 60.0 / 7.5)
    c = a - r * 5.0
    d = lon_frac * 60.0 - s * 7.5
    t = int(c * 60.0 / 30.0)
    u = int(d * 60.0 / 45.0)
    return f"{p:02d}{q:02d}{r}{s}{t}{u}"


def mesh3_center(code: str) -> tuple[float, float]:
    """Return the center latitude/longitude of a third-level mesh code."""
    if len(code) != 8 or not code.isdigit():
        raise ValueError(f"invalid third-level mesh code: {code}")

    p = int(code[0:2])
    q = int(code[2:4])
    r = int(code[4])
    s = int(code[5])
    t = int(code[6])
    u = int(code[7])

    lat = p * (40.0 / 60.0) + r * (5.0 / 60.0) + t * (30.0 / 3600.0) + 15.0 / 3600.0
    lon = q + 100.0 + s * (7.5 / 60.0) + u * (45.0 / 3600.0) + 22.5 / 3600.0
    return lat, lon


def expand_meshes(meshes: list[str], radius: int) -> list[str]:
    """Expand third-level mesh codes by a square neighborhood radius."""
    if radius <= 0:
        return sorted(set(meshes))

    expanded: set[str] = set()
    lat_step = 30.0 / 3600.0
    lon_step = 45.0 / 3600.0

    for mesh in meshes:
        lat0, lon0 = mesh3_center(mesh)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                expanded.add(mesh3_code(lat0 + dy * lat_step, lon0 + dx * lon_step))
    return sorted(expanded)


def load_reference_meshes(
    reference_csv: Path,
    max_rows: int | None = None,
    start_row: int = 0,
    mesh_radius: int = 0,
) -> list[str]:
    with open(reference_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if start_row:
        rows = rows[start_row:]
    if max_rows is not None:
        rows = rows[:max_rows]
    base_meshes = sorted(
        {
            mesh3_code(float(row["Latitude (deg)"]), float(row["Longitude (deg)"]))
            for row in rows
        }
    )
    return expand_meshes(base_meshes, mesh_radius)


def select_bldg_entries(zf: zipfile.ZipFile, meshes: list[str]) -> list[str]:
    names = zf.namelist()
    selected = [
        name
        for name in names
        if ("udx/bldg/" in name or "/udx/bldg/" in name)
        and name.endswith(".gml")
        and any(f"/{mesh}_" in name for mesh in meshes)
    ]
    return sorted(selected)


def extract_entries(zf: zipfile.ZipFile, entries: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in entries:
        target = output_dir / Path(name).name
        if target.exists():
            continue
        with zf.open(name) as src, open(target, "wb") as dst:
            dst.write(src.read())
        print(f"  extracted {target.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a PLATEAU subset for a PPC trajectory")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="PPC run directory containing reference.csv",
    )
    parser.add_argument(
        "--zip-url",
        type=str,
        default="",
        help="Direct PLATEAU CityGML ZIP URL",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESET_URLS),
        default="",
        help="Use a built-in PLATEAU ZIP URL preset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for extracted building GML files",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only use the first N rows of reference.csv when computing meshes",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Skip this many initial rows of reference.csv before computing meshes",
    )
    parser.add_argument(
        "--mesh-radius",
        type=int,
        default=0,
        help="Expand each trajectory mesh by this many neighboring third-level tiles",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path",
    )
    args = parser.parse_args()

    reference_csv = args.run_dir / "reference.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"reference.csv not found: {reference_csv}")

    zip_url = args.zip_url or PRESET_URLS.get(args.preset, "")
    if not zip_url:
        raise ValueError("either --zip-url or --preset is required")

    meshes = load_reference_meshes(
        reference_csv,
        max_rows=args.max_rows,
        start_row=args.start_row,
        mesh_radius=args.mesh_radius,
    )
    print(f"trajectory meshes ({len(meshes)}): {', '.join(meshes)}")

    with zipfile.ZipFile(HTTPRangeReader(zip_url)) as zf:
        entries = select_bldg_entries(zf, meshes)
        if not entries:
            raise RuntimeError("no matching building GML files found in the remote ZIP")
        print(f"matched {len(entries)} building tiles")
        extract_entries(zf, entries, args.output_dir)

    manifest = {
        "run_dir": str(args.run_dir),
        "reference_csv": str(reference_csv),
        "start_row": args.start_row,
        "max_rows": args.max_rows,
        "mesh_radius": args.mesh_radius,
        "zip_url": zip_url,
        "meshes": meshes,
        "n_entries": len(entries),
        "entries": entries,
        "output_dir": str(args.output_dir),
    }

    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
