#!/usr/bin/env python3
"""Materialize an OSM road-centerline-corrected PPC candidate .pos file."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_phase67_osm_road_centerline_feasibility import _ecef_to_llh  # noqa: E402
from experiments.evaluate import lla_to_ecef  # noqa: E402
from sim_phase68_osm_road_centerline_correction import (  # noqa: E402
    _base_label,
    _correct_xy,
    _float,
    _road_union_from_osm,
    _write_csv,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _xyz_from_internal_row(row: dict[str, str]) -> np.ndarray:
    return np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)


def _read_pos_lines(path: Path) -> tuple[list[str], list[list[str]]]:
    header: list[str] = []
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                header.append(line.rstrip("\n"))
            else:
                rows.append(stripped.split())
    return header, rows


def _write_pos_lines(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in header:
            fh.write(f"{line}\n")
        for parts in rows:
            fh.write(" ".join(parts) + "\n")
    print(f"saved: {path}")


def _pos_by_tow(rows: list[list[str]]) -> dict[float, list[str]]:
    out: dict[float, list[str]] = {}
    for parts in rows:
        if len(parts) < 8:
            continue
        out[round(float(parts[1]), 1)] = parts
    return out


def _eligible_indices(prepared: list[dict[str, Any]], road_dist_min_m: float, min_contiguous_epochs: int) -> set[int]:
    eligible: set[int] = set()
    current: list[int] = []
    for idx, rec in enumerate(prepared):
        ok = float(rec["road_dist_m"]) >= road_dist_min_m
        contiguous = bool(current) and int(rec["epoch"]) == int(prepared[current[-1]]["epoch"]) + 1
        if ok and (not current or contiguous):
            current.append(idx)
            continue
        if len(current) >= min_contiguous_epochs:
            eligible.update(current)
        current = [idx] if ok else []
    if len(current) >= min_contiguous_epochs:
        eligible.update(current)
    return eligible


def materialize(
    *,
    internal_rows: list[dict[str, str]],
    source_pos: Path,
    source_diag_csv: Path | None,
    out_dir: Path,
    city: str,
    run: str,
    selected_label: str,
    start_epoch: int,
    end_epoch: int,
    bbox_margin_deg: float,
    epsg: int,
    alpha: float,
    road_dist_min_m: float,
    min_contiguous_epochs: int,
    require_runtime_selected: bool,
    pass_m: float,
) -> list[dict[str, Any]]:
    header, pos_rows = _read_pos_lines(source_pos)
    pos_lookup = _pos_by_tow(pos_rows)
    span_rows = internal_rows[start_epoch : end_epoch + 1]
    truth_ll = [_ecef_to_llh(_xyz_from_internal_row(row)) for row in span_rows]
    north = max(lat for lat, _lon, _h in truth_ll) + bbox_margin_deg
    south = min(lat for lat, _lon, _h in truth_ll) - bbox_margin_deg
    east = max(lon for _lat, lon, _h in truth_ll) + bbox_margin_deg
    west = min(lon for _lat, lon, _h in truth_ll) - bbox_margin_deg
    road_union, fwd, inv, n_edges = _road_union_from_osm(north=north, south=south, east=east, west=west, epsg=epsg)

    prepared: list[dict[str, Any]] = []
    for row in span_rows:
        epoch = int(float(row["epoch"]))
        tow = round(float(row["tow"]), 1)
        runtime_selected = _base_label(row)
        if require_runtime_selected and runtime_selected != selected_label:
            continue
        parts = pos_lookup.get(tow)
        if parts is None or len(parts) < 8:
            continue
        lat = float(parts[5])
        lon = float(parts[6])
        height_m = float(parts[7])
        x, y = fwd.transform(lon, lat)
        road_dist = float(Point(x, y).distance(road_union))
        truth = _xyz_from_internal_row(row)
        source_xyz = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
        source_error = float(np.linalg.norm(source_xyz - truth))
        prepared.append(
            {
                "epoch": epoch,
                "tow": tow,
                "point": Point(x, y),
                "height_m": height_m,
                "road_dist_m": road_dist,
                "truth": truth,
                "source_error_m": source_error,
                "runtime_selected": runtime_selected,
            }
        )

    eligible = _eligible_indices(prepared, road_dist_min_m, min_contiguous_epochs)
    corrected_by_tow: dict[float, tuple[float, float, float, float, float, float]] = {}
    summary_rows: list[dict[str, Any]] = []
    good = 0
    bad = 0
    for idx, rec in enumerate(prepared):
        corrected = idx in eligible
        err = float(rec["source_error_m"])
        road_dist = float(rec["road_dist_m"])
        lat = lon = height_m = float("nan")
        xyz = None
        if corrected:
            cx, cy, road_dist = _correct_xy(rec["point"], road_union, "alpha", alpha, 0.0)
            lon, lat = inv.transform(cx, cy)
            height_m = float(rec["height_m"])
            xyz = np.asarray(lla_to_ecef(math.radians(lat), math.radians(lon), height_m), dtype=np.float64)
            err = float(np.linalg.norm(xyz - np.asarray(rec["truth"], dtype=np.float64)))
            corrected_by_tow[float(rec["tow"])] = (float(xyz[0]), float(xyz[1]), float(xyz[2]), float(lat), float(lon), height_m)
        source_pass = float(rec["source_error_m"]) <= pass_m
        corrected_pass = err <= pass_m
        if corrected_pass and not source_pass:
            good += 1
        elif source_pass and not corrected_pass:
            bad += 1
        summary_rows.append(
            {
                "epoch": rec["epoch"],
                "tow": rec["tow"],
                "runtime_selected": rec["runtime_selected"],
                "road_dist_m": rec["road_dist_m"],
                "triggered": corrected,
                "source_error_m": rec["source_error_m"],
                "corrected_error_m": err,
            }
        )

    for parts in pos_rows:
        if len(parts) < 8:
            continue
        tow = round(float(parts[1]), 1)
        corrected = corrected_by_tow.get(tow)
        if corrected is None:
            continue
        x, y, z, lat, lon, height_m = corrected
        parts[2] = f"{x:.4f}"
        parts[3] = f"{y:.4f}"
        parts[4] = f"{z:.4f}"
        parts[5] = f"{lat:.9f}"
        parts[6] = f"{lon:.9f}"
        parts[7] = f"{height_m:.4f}"

    out_pos = out_dir / f"{city}_{run}_full.pos"
    _write_pos_lines(out_pos, header, pos_rows)
    if source_diag_csv is not None and source_diag_csv.is_file():
        out_diag = out_dir / f"{city}_{run}_full.csv"
        shutil.copyfile(source_diag_csv, out_diag)
        print(f"saved: {out_diag}")

    rollup = {
        "city": city,
        "run": run,
        "selected_label": selected_label,
        "source_pos": str(source_pos),
        "source_diag_csv": str(source_diag_csv or ""),
        "out_dir": str(out_dir),
        "road_edge_geometries": n_edges,
        "prepared_epochs": len(prepared),
        "triggered_epochs": len(corrected_by_tow),
        "alpha": alpha,
        "road_dist_min_m": road_dist_min_m,
        "min_contiguous_epochs": min_contiguous_epochs,
        "good_epochs": good,
        "bad_epochs": bad,
    }
    _write_csv(out_dir / f"{city}_{run}_phase70_osm_road_materialize_summary.csv", [rollup])
    _write_csv(out_dir / f"{city}_{run}_phase70_osm_road_materialize_epochs.csv", summary_rows)
    return [rollup]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--source-pos", type=Path, required=True)
    parser.add_argument("--source-diag-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/phase70_osm_road_hs_alpha05_triggered"))
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--selected-label", default="xd_gici_hs")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=-1)
    parser.add_argument("--bbox-margin-deg", type=float, default=0.002)
    parser.add_argument("--epsg", type=int, default=32653)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--road-dist-min-m", type=float, default=2.5)
    parser.add_argument("--min-contiguous-epochs", type=int, default=40)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--include-nonselected", action="store_true")
    args = parser.parse_args()

    internal_rows = _read_csv(args.internal_epochs_csv)
    end_epoch = len(internal_rows) - 1 if int(args.end_epoch) < 0 else int(args.end_epoch)
    summary = materialize(
        internal_rows=internal_rows,
        source_pos=args.source_pos,
        source_diag_csv=args.source_diag_csv,
        out_dir=args.out_dir,
        city=str(args.city),
        run=str(args.run),
        selected_label=str(args.selected_label),
        start_epoch=int(args.start_epoch),
        end_epoch=end_epoch,
        bbox_margin_deg=float(args.bbox_margin_deg),
        epsg=int(args.epsg),
        alpha=float(args.alpha),
        road_dist_min_m=float(args.road_dist_min_m),
        min_contiguous_epochs=int(args.min_contiguous_epochs),
        require_runtime_selected=not bool(args.include_nonselected),
        pass_m=float(args.pass_m),
    )
    if summary:
        print("summary:", summary[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
