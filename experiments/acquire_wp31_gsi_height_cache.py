#!/usr/bin/env python3
"""Acquire and freeze GSI DEM/geoid height for a truth-free candidate center."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.nmea_writer import _ecef_to_lla_py  # noqa: E402


DEM_DOC = "https://maps.gsi.go.jp/development/elevation_s.html"
GEOID_DOC = "https://vldb.gsi.go.jp/sokuchi/surveycalc/api_help.html"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed official hosts
        return json.loads(response.read().decode("utf-8"))


def build_cache(
    candidate_source: dict[str, Any],
    calibration_cache: dict[str, Any],
    *,
    query_basis: str,
    fetch_json: Callable[[str], dict[str, Any]] = _fetch_json,
    acquired_utc: str | None = None,
) -> dict[str, Any]:
    candidates = list(candidate_source.get("candidates", []))
    if not candidates:
        raise ValueError("candidate source is empty")
    positions = np.asarray([row["position_ecef"] for row in candidates], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("candidate positions are invalid")
    center = np.median(positions, axis=0)
    latitude, longitude, _height = _ecef_to_lla_py(*center)
    dem_url = (
        "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/"
        f"getelevation.php?lon={longitude:.8f}&lat={latitude:.8f}&outtype=JSON"
    )
    geoid_url = (
        "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh2011/cgi/"
        f"geoidcalc.pl?outputType=json&latitude={latitude:.8f}&longitude={longitude:.8f}"
    )
    dem = fetch_json(dem_url)
    geoid = fetch_json(geoid_url)
    elevation = float(dem["elevation"])
    geoid_height = float(geoid["OutputData"]["geoidHeight"])
    if not np.isfinite([elevation, geoid_height]).all():
        raise ValueError("GSI returned nonfinite height")
    segment = [int(value) for value in candidate_source["segment"]]
    return {
        "schema": "wp31_gsi_height_cache_v1",
        "acquired_utc": acquired_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_network_required": False,
        "dem_api_documentation": DEM_DOC,
        "geoid_api_documentation": GEOID_DOC,
        "geoid_model": "GSIGEO2011_Ver2.2",
        "calibration_points": list(calibration_cache["calibration_points"]),
        "target_point": {
            "name": f"stop_{segment[0]}_candidate_median_center",
            "latitude_deg": float(latitude),
            "longitude_deg": float(longitude),
            "elevation_m": elevation,
            "dem_source": str(dem["hsrc"]),
            "geoid_height_m": geoid_height,
            "geoid_model": "GSIGEO2011_Ver2.2",
            "query_basis": query_basis,
            "dem_query_url": dem_url,
            "geoid_query_url": geoid_url,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--query-basis", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_cache.read_text(encoding="utf-8"))
    result = build_cache(
        candidates,
        calibration,
        query_basis=args.query_basis,
    )
    result["target_point"]["candidate_source"] = str(args.candidates_json).replace("\\", "/")
    result["target_point"]["candidate_source_sha256"] = _sha256(args.candidates_json)
    result["calibration_cache"] = str(args.calibration_cache).replace("\\", "/")
    result["calibration_cache_sha256"] = _sha256(args.calibration_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
