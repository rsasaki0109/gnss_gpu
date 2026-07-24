#!/usr/bin/env python3
"""Acquire a provenance-locked GSI height cache for accepted static anchors."""

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed official GSI hosts
        return json.loads(response.read().decode("utf-8"))


def load_candidate(path: Path, candidate_id: int) -> list[float]:
    source = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row for row in source.get("candidates", [])
        if int(row.get("candidate_id", -1)) == int(candidate_id)
    ]
    if len(matches) != 1 or not bool(matches[0].get("applied", False)):
        raise ValueError("calibration candidate is absent or invalid")
    position = np.asarray(matches[0]["position_ecef"], dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("calibration candidate position is invalid")
    return position.tolist()


def acquire_point(
    name: str,
    position_ecef: list[float],
    *,
    fetch_json: Callable[[str], dict[str, Any]] = _fetch_json,
) -> dict[str, Any]:
    latitude, longitude, _height = _ecef_to_lla_py(*position_ecef)
    dem_url = (
        "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/"
        f"getelevation.php?lon={longitude:.8f}&lat={latitude:.8f}&outtype=JSON"
    )
    geoid_url = (
        "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh2011/cgi/"
        f"geoidcalc.pl?outputType=json&latitude={latitude:.8f}&longitude={longitude:.8f}"
    )
    dem, geoid = fetch_json(dem_url), fetch_json(geoid_url)
    elevation = float(dem["elevation"])
    geoid_height = float(geoid["OutputData"]["geoidHeight"])
    if not np.isfinite([elevation, geoid_height]).all():
        raise ValueError("GSI returned nonfinite height")
    return {
        "name": name,
        "latitude_deg": float(latitude),
        "longitude_deg": float(longitude),
        "elevation_m": elevation,
        "dem_source": str(dem["hsrc"]),
        "geoid_height_m": geoid_height,
        "geoid_model": "GSIGEO2011_Ver2.2",
        "antenna_position_ecef": position_ecef,
        "dem_query_url": dem_url,
        "geoid_query_url": geoid_url,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-point", nargs=3, action="append", metavar=("NAME", "CANDIDATES_JSON", "ID"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.calibration_point) < 2:
        parser.error("at least two calibration points are required")
    points = []
    for name, path_text, candidate_text in args.calibration_point:
        path = Path(path_text)
        candidate_id = int(candidate_text)
        point = acquire_point(name, load_candidate(path, candidate_id))
        point["position_source"] = f"{str(path).replace(chr(92), '/')} candidate {candidate_id}"
        point["position_source_sha256"] = _sha256(path)
        points.append(point)
    result = {
        "schema": "wp31_gsi_height_calibration_cache_v1",
        "acquired_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_network_required": False,
        "dem_api_documentation": DEM_DOC,
        "geoid_api_documentation": GEOID_DOC,
        "geoid_model": "GSIGEO2011_Ver2.2",
        "calibration_points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
