"""Record the repository's independent Python CPU GPS orbit results for web tests.

Run in a dedicated Python process. Namespace packages avoid importing unrelated
native/GPU entry points; the ephemeris implementation itself is unmodified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import types


def build(root: Path, navigation: Path, scene_path: Path, output: Path, systems="G"):
    for path in (root, navigation, scene_path, output):
        if ".claude" in (part.casefold() for part in path.parts):
            raise ValueError("Excluded source path")
    for name, path in (
        ("gnss_gpu", root / "python/gnss_gpu"),
        ("gnss_gpu.io", root / "python/gnss_gpu/io"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    from gnss_gpu.ephemeris import Ephemeris
    from gnss_gpu.io.nav_rinex import read_nav_rinex_multi

    scene = json.loads(gzip.decompress(scene_path.read_bytes()))
    navs = read_nav_rinex_multi(navigation, systems=tuple(systems))
    cases = []
    for satellite in sorted(navs):
        records = [r for r in scene["navigation"] if r["id"] == satellite]
        if not records:
            continue
        record = min(records, key=lambda r: abs(r["parameters"][8] - 273600))
        nav = next(
            n
            for n in navs[satellite]
            if n.toe == record["parameters"][8]
            and n.af0 == record["clock"][0]
            and n.week == record["parameters"][18]
        )
        for time in (273375.025, 273975.1, 274615.925):
            native_time = time
            if nav.system == "E":
                a0, a1, tow, reference_week = record["gpga"]
                native_time += (a0 + a1 * ((scene["week"] - reference_week) * 604800 + time - tow)) / (1 - a1)
            position, clock = Ephemeris._compute_single_cpu(nav, native_time, obs_code="C1C")
            if nav.system == "C" and nav.prn <= 5:
                # Independent frame-matrix composition around the native CPU
                # MEO solution; BDS ICD 5.2.4.1 GEO uses a tilted orbital frame.
                import numpy as np
                import math
                angle = 7.292115e-5 * (time - 14 - nav.toe)
                def rz(a):
                    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
                tilt = math.radians(5)
                rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
                position = rz(-angle) @ rx @ rz(angle) @ position
            cases.append(
                {
                    "record": record,
                    "week": scene["week"],
                    "time": time,
                    "position": position.tolist(),
                    "clock": float(clock),
                }
            )
    fixture = {
        "method": "gnss_gpu.ephemeris.Ephemeris._compute_single_cpu (L1/E1/B1; GPGA alignment; BDS GEO independent frame-matrix composition; no GPU)",
        "source_sha256": hashlib.sha256(
            (root / "python/gnss_gpu/ephemeris.py").read_bytes()
        ).hexdigest(),
        "navigation_sha256": hashlib.sha256(navigation.read_bytes()).hexdigest(),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Recorded {len(cases)} independent CPU orbit/clock cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--nav", type=Path, default=Path("data/urbannav/Tokyo/Odaiba/base.nav")
    )
    parser.add_argument(
        "--scene", type=Path, default=Path("docs/assets/plateau/odaiba-scene.json.gz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tests/site/fixtures/gps-orbits.json")
    )
    parser.add_argument("--systems", default="G", choices=("G", "GECJ"))
    args = parser.parse_args()
    build(args.root, args.nav, args.scene, args.output, args.systems)
