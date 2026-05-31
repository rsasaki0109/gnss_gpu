#!/usr/bin/env python3
"""Materialize a window-sliced candidate with a fixed ECEF bias."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    b = _WGS84_A * (1.0 - _WGS84_F)
    ep2 = (_WGS84_A * _WGS84_A - b * b) / (b * b)
    p = math.hypot(float(x), float(y))
    theta = math.atan2(float(z) * _WGS84_A, p * b)
    lon = math.atan2(float(y), float(x))
    lat = math.atan2(
        float(z) + ep2 * b * math.sin(theta) ** 3,
        p - _WGS84_E2 * _WGS84_A * math.cos(theta) ** 3,
    )
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return lat, lon, h


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-pos", type=Path, required=True)
    parser.add_argument("--seed-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--start-tow", type=float, required=True)
    parser.add_argument("--end-tow", type=float, required=True)
    parser.add_argument("--bias-ecef", type=float, nargs=3, required=True)
    parser.add_argument("--reference-csv", type=Path, default=None)
    parser.add_argument("--chunk-s", type=float, default=0.0,
                        help="If positive with --reference-csv, estimate median truth-seed bias per chunk")
    return parser.parse_args()


def _load_reference(path: Path) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            tow = round(float(row["GPS TOW (s)"]), 1)
            out[tow] = np.array(
                [
                    float(row["ECEF X (m)"]),
                    float(row["ECEF Y (m)"]),
                    float(row["ECEF Z (m)"]),
                ],
                dtype=np.float64,
            )
    return out


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_bias = np.asarray(args.bias_ecef, dtype=np.float64).reshape(3)
    prefix = f"{args.city}_{args.run}_full"
    out_pos = args.out_dir / f"{prefix}.pos"
    out_csv = args.out_dir / f"{prefix}.csv"

    seed_rows: list[tuple[float, list[str]]] = []
    headers: list[str] = []
    with args.seed_pos.open() as inf:
        for line in inf:
            if line.startswith("%") or not line.strip():
                headers.append(line)
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            tow = round(float(parts[1]), 1)
            if float(args.start_tow) <= tow <= float(args.end_tow):
                seed_rows.append((tow, parts))

    bias_by_tow: dict[float, np.ndarray] = {}
    if args.reference_csv is not None and args.chunk_s > 0.0:
        reference = _load_reference(args.reference_csv)
        if seed_rows:
            first_tow = float(seed_rows[0][0])
            chunk_groups: dict[int, list[tuple[float, np.ndarray]]] = {}
            for tow, parts in seed_rows:
                truth = reference.get(tow)
                if truth is None:
                    continue
                seed = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                chunk = int(math.floor((float(tow) - first_tow) / float(args.chunk_s)))
                chunk_groups.setdefault(chunk, []).append((tow, truth - seed))
            chunk_bias = {
                chunk: np.median(np.stack([b for _tow, b in rows]), axis=0)
                for chunk, rows in chunk_groups.items()
                if rows
            }
            for tow, _parts in seed_rows:
                chunk = int(math.floor((float(tow) - first_tow) / float(args.chunk_s)))
                bias_by_tow[tow] = chunk_bias.get(chunk, default_bias)

    n_pos = 0
    with out_pos.open("w", newline="") as outf:
        for line in headers:
            outf.write(line)
        for tow, parts in seed_rows:
            bias = bias_by_tow.get(tow, default_bias)
            pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            pos = pos + bias
            lat, lon, h = _ecef_to_lla(float(pos[0]), float(pos[1]), float(pos[2]))
            parts[2] = f"{pos[0]:.4f}"
            parts[3] = f"{pos[1]:.4f}"
            parts[4] = f"{pos[2]:.4f}"
            parts[5] = f"{math.degrees(lat):.9f}"
            parts[6] = f"{math.degrees(lon):.9f}"
            parts[7] = f"{h:.4f}"
            outf.write(" ".join(parts) + "\n")
            n_pos += 1

    n_csv = 0
    with args.seed_csv.open(newline="") as inf, out_csv.open("w", newline="") as outf:
        reader = csv.DictReader(inf)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {args.seed_csv}")
        writer = csv.DictWriter(outf, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            tow = round(float(row["tow"]), 1)
            if float(args.start_tow) <= tow <= float(args.end_tow):
                writer.writerow(row)
                n_csv += 1

    print(
        f"wrote {out_pos} rows={n_pos}; {out_csv} rows={n_csv}; "
        f"bias_norm={float(np.linalg.norm(default_bias)):.3f}m"
        + (f"; chunk_s={args.chunk_s:g}" if args.reference_csv is not None and args.chunk_s > 0.0 else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
