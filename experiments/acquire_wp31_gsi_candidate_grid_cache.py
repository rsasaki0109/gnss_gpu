#!/usr/bin/env python3
"""Freeze GSI DEM/geoid responses at every candidate in a proposal grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.nmea_writer import _ecef_to_lla_py  # noqa: E402


def _fetch(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed official GSI hosts
        return json.loads(response.read().decode("utf-8"))


def acquire_candidate_points(
    candidates: list[dict[str, Any]],
    *,
    fetch_json: Callable[[str], dict[str, Any]] = _fetch,
    max_workers: int = 2,
) -> list[dict[str, Any]]:
    def valid_fetch(url: str, required_key: str) -> dict[str, Any]:
        last: dict[str, Any] = {}
        for attempt in range(5):
            last = fetch_json(url)
            if required_key in last:
                return last
            if attempt < 4:
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"GSI response missing {required_key}: {last}")

    def acquire(row: dict[str, Any]) -> dict[str, Any]:
        latitude, longitude, _height = _ecef_to_lla_py(*np.asarray(row["position_ecef"], dtype=np.float64))
        dem_url = "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/" + f"getelevation.php?lon={longitude:.8f}&lat={latitude:.8f}&outtype=JSON"
        geoid_url = "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh2011/cgi/" + f"geoidcalc.pl?outputType=json&latitude={latitude:.8f}&longitude={longitude:.8f}"
        dem = valid_fetch(dem_url, "elevation")
        geoid = valid_fetch(geoid_url, "OutputData")
        return {
            "candidate_id": int(row["candidate_id"]),
            "latitude_deg": float(latitude),
            "longitude_deg": float(longitude),
            "elevation_m": float(dem["elevation"]),
            "dem_source": str(dem["hsrc"]),
            "geoid_height_m": float(geoid["OutputData"]["geoidHeight"]),
            "geoid_model": "GSIGEO2011_Ver2.2",
            "dem_query_url": dem_url,
            "geoid_query_url": geoid_url,
        }

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        points = list(executor.map(acquire, candidates))
    points.sort(key=lambda row: int(row["candidate_id"]))
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes, calibration_bytes = args.candidates_json.read_bytes(), args.calibration_cache.read_bytes()
    source, calibration = json.loads(source_bytes), json.loads(calibration_bytes)
    points = acquire_candidate_points(list(source["candidates"]), max_workers=args.max_workers)
    result = {
        "schema": "wp31_gsi_candidate_grid_cache_v1",
        "acquired_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_network_required": False,
        "geoid_model": "GSIGEO2011_Ver2.2",
        "segment": [int(value) for value in source["segment"]],
        "candidate_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "calibration_cache_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "calibration_points": list(calibration["calibration_points"]),
        "candidate_points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**result, "calibration_points": f"{len(result['calibration_points'])} points", "candidate_points": f"{len(points)} points"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
