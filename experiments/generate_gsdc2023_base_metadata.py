#!/usr/bin/env python3
"""Write ``base_position.csv`` and ``base_offset.csv`` for ``correct_pseudorange.m``.

MATLAB (``ref/gsdc2023/functions/correct_pseudorange.m``) expects, next to ``dataset_2023``::

    ../base/base_position.csv   columns: Base, Year, X, Y, Z  (ECEF meters, WGS84)
    ../base/base_offset.csv     columns: Base, E, N, U       (antenna offset meters, ENU)

``Year`` is matched against the first token of ``Course`` (e.g. ``2020`` from
``2020-06-25-...``).

Coordinates below are **approximate monument positions** (deg height → ECEF via WGS84)
suitable for unlocking the audit / pipeline; replace with survey-grade values from
your RINEX headers or NGS if you need MATLAB parity at the centimeter level.

Default sites (GSDC2023 literature / CORS-style IDs used with ``settings`` Base1)::

  SLAC — San Francisco Bay / Peninsula drives
  VDCY — Los Angeles ``us-ca-lax`` drives

Example::

  PYTHONPATH=python python3 experiments/generate_gsdc2023_base_metadata.py \\
    --output-dir ../ref/gsdc2023/base
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F**2


def _lla_deg_to_ecef_m(lat_deg: float, lon_deg: float, h_m: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    x = (n + h_m) * cos_lat * math.cos(lon)
    y = (n + h_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + h_m) * sin_lat
    return (x, y, z)


# Approximate LLH (deg, deg, ellipsoidal height m) — replace from RINEX APPROX POSITION if needed.
_DEFAULT_SITES: dict[str, tuple[float, float, float]] = {
    # Menlo Park / SLAC CORS neighborhood (not a survey tie).
    "SLAC": (37.4189, -122.1734, 90.0),
    # Van Nuys area (Los Angeles basin).
    "VDCY": (34.2215, -118.4903, 95.0),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO.parent / "ref" / "gsdc2023" / "base",
        help="directory for base_position.csv and base_offset.csv",
    )
    p.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024],
        help="years duplicated per base (Course year token)",
    )
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    pos_lines = ["Base,Year,X,Y,Z"]
    for name, (la, lo, h) in _DEFAULT_SITES.items():
        x, y, z = _lla_deg_to_ecef_m(la, lo, h)
        for yr in args.years:
            pos_lines.append(f"{name},{yr},{x:.4f},{y:.4f},{z:.4f}")

    off_lines = ["Base,E,N,U"]
    for name in _DEFAULT_SITES:
        off_lines.append(f"{name},0,0,0")

    (out / "base_position.csv").write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    (out / "base_offset.csv").write_text("\n".join(off_lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'base_position.csv'}")
    print(f"wrote {out / 'base_offset.csv'}")


if __name__ == "__main__":
    main()
