"""Build per-epoch PLATEAU LOS/NLOS mask CSVs for PPC runs.

The output keeps the minimal contract used by exp_ppc_ctrbpf_fgo.py:
``tow,epoch_idx,prn,is_los`` where ``is_los=0`` marks a satellite as NLOS.
Extra columns are diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD_PYTHON = _REPO_ROOT / "build" / "python"
if _BUILD_PYTHON.exists():
    sys.path.insert(0, str(_BUILD_PYTHON))
sys.path.append(str(_REPO_ROOT / "python"))

from gnss_gpu.bvh import BVHAccelerator  # noqa: E402
from gnss_gpu.io.plateau import load_plateau  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.raytrace import BuildingModel  # noqa: E402


DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


def _parse_csv_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _parse_geoid_correction(text: str) -> str | float | None:
    value = str(text).strip().lower()
    if value in {"", "none", "off", "false"}:
        return None
    if value == "egm96":
        return "egm96"
    return float(value)


def _default_plateau_zone(run_dir: Path) -> int:
    city = run_dir.parent.name.lower()
    if city == "nagoya":
        return 7
    return 9


def _elevation_deg(rx_ecef: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray:
    rx = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    diff = sat - rx[None, :]
    ranges = np.linalg.norm(diff, axis=1)
    rx_norm = float(np.linalg.norm(rx))
    out = np.full(sat.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges > 1e-3) & np.isfinite(rx_norm) & (rx_norm > 1e-3)
    if not np.any(valid):
        return out
    up = rx / rx_norm
    sin_el = diff[valid] @ up / ranges[valid]
    out[valid] = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
    return out


def _read_receiver_pos_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    positions: list[list[float]] = []
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].lstrip().startswith(("%", "#")):
                continue
            parts = [part for cell in row for part in cell.strip().split()]
            if len(parts) < 5:
                continue
            try:
                tow = float(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except ValueError:
                continue
            if np.all(np.isfinite([tow, x, y, z])):
                times.append(tow)
                positions.append([x, y, z])
    if not times:
        raise ValueError(f"receiver pos file contained no usable rows: {path}")
    order = np.argsort(np.asarray(times, dtype=np.float64))
    return np.asarray(times, dtype=np.float64)[order], np.asarray(positions, dtype=np.float64)[order]


def _nearest_position(
    times: np.ndarray,
    positions: np.ndarray,
    tow: float,
    tolerance_s: float,
) -> tuple[np.ndarray | None, float]:
    idx = int(np.searchsorted(times, tow))
    candidates = []
    if idx < len(times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None, float("nan")
    best = min(candidates, key=lambda i: abs(float(times[i]) - tow))
    delta = float(times[best] - tow)
    if abs(delta) > tolerance_s:
        return None, delta
    return positions[best].copy(), delta


def _receiver_for_epoch(
    data: dict,
    epoch_idx: int,
    *,
    pos_times: np.ndarray | None,
    pos_positions: np.ndarray | None,
    pos_tolerance_s: float,
    missing_pos_policy: str,
) -> tuple[np.ndarray | None, str, float]:
    tow = float(data["times"][epoch_idx])
    if pos_times is None or pos_positions is None:
        return np.asarray(data["ground_truth"][epoch_idx], dtype=np.float64), "reference", 0.0
    pos, delta = _nearest_position(pos_times, pos_positions, tow, pos_tolerance_s)
    if pos is not None:
        return pos, "pos", delta
    if missing_pos_policy == "reference":
        return np.asarray(data["ground_truth"][epoch_idx], dtype=np.float64), "reference_fallback", delta
    return None, "missing_pos", delta


def _write_batch_rows(
    writer: csv.writer,
    data: dict,
    start_idx: int,
    receiver_positions: np.ndarray,
    receiver_sources: list[str],
    receiver_deltas: list[float],
    los: np.ndarray,
) -> tuple[int, int]:
    n_rows = 0
    n_nlos = 0
    for local_idx, rx in enumerate(receiver_positions):
        epoch_idx = start_idx + local_idx
        tow = float(data["times"][epoch_idx])
        sat_ids = list(data["used_prns"][epoch_idx])
        sat_ecef = np.asarray(data["sat_ecef"][epoch_idx], dtype=np.float64)
        elev = _elevation_deg(rx, sat_ecef)
        for sat_idx, prn in enumerate(sat_ids):
            is_los = bool(los[local_idx, sat_idx])
            n_rows += 1
            n_nlos += int(not is_los)
            writer.writerow(
                [
                    f"{tow:.3f}",
                    epoch_idx,
                    prn,
                    int(is_los),
                    prn[0] if prn else "",
                    prn[1:] if len(prn) > 1 else "",
                    f"{float(elev[sat_idx]):.3f}",
                    receiver_sources[local_idx],
                    f"{float(receiver_deltas[local_idx]):.3f}",
                ]
            )
    return n_rows, n_nlos


def _check_los(bvh: BVHAccelerator, rx_rows: list[np.ndarray], sat_rows: list[np.ndarray]) -> list[np.ndarray]:
    max_sats = max(int(sat_ecef.shape[0]) for sat_ecef in sat_rows)
    sat_padded = np.full((len(rx_rows), max_sats, 3), np.nan, dtype=np.float64)
    for row_idx, sat_ecef in enumerate(sat_rows):
        sat_padded[row_idx, : sat_ecef.shape[0], :] = sat_ecef
    try:
        los = np.asarray(
            bvh.check_los_batch(np.asarray(rx_rows, dtype=np.float64), sat_padded),
            dtype=bool,
        )
        return [los[i, : sat_rows[i].shape[0]].copy() for i in range(len(sat_rows))]
    except (ImportError, AttributeError):
        return [
            np.asarray(bvh.check_los(rx_rows[i], sat_rows[i]), dtype=bool)
            for i in range(len(sat_rows))
        ]


def _load_or_build_plateau_model(args: argparse.Namespace, zone: int, kinds: tuple[str, ...]) -> BuildingModel:
    if args.triangle_cache_npz and args.triangle_cache_npz.exists():
        print(f"[nlos] loading triangle cache {args.triangle_cache_npz}", flush=True)
        with np.load(args.triangle_cache_npz) as data:
            triangles = np.asarray(data["triangles"], dtype=np.float64)
        return BuildingModel(triangles)

    geoid_correction = _parse_geoid_correction(args.geoid_correction)
    print(f"[nlos] loading PLATEAU {args.plateau_dir} zone={zone} kinds={kinds}", flush=True)
    model = load_plateau(
        args.plateau_dir,
        zone=zone,
        kinds=kinds,
        geoid_correction=geoid_correction,
    )
    if args.triangle_cache_npz:
        args.triangle_cache_npz.parent.mkdir(parents=True, exist_ok=True)
        print(f"[nlos] writing triangle cache {args.triangle_cache_npz}", flush=True)
        np.savez(args.triangle_cache_npz, triangles=np.asarray(model.triangles, dtype=np.float64))
    return model


def build_nlos_csv(args: argparse.Namespace) -> dict[str, float]:
    run_dir = args.data_dir or (args.data_root / args.run)
    systems = _parse_csv_list(args.systems)
    kinds = _parse_csv_list(args.kinds)
    zone = int(args.plateau_zone) if int(args.plateau_zone) > 0 else _default_plateau_zone(run_dir)

    print(f"[nlos] loading PPC {run_dir} systems={systems}", flush=True)
    loader = PPCDatasetLoader(run_dir)
    data = loader.load_experiment_data(
        max_epochs=args.max_epochs if args.max_epochs > 0 else None,
        start_epoch=max(0, int(args.start_epoch)),
        systems=systems,
    )

    model = _load_or_build_plateau_model(args, zone, kinds)
    bvh = BVHAccelerator.from_building_model(model)
    print(f"[nlos] BVH triangles={bvh.n_triangles} nodes={bvh.n_nodes}", flush=True)

    pos_times = pos_positions = None
    if args.receiver_pos_file:
        pos_times, pos_positions = _read_receiver_pos_file(args.receiver_pos_file)
        print(
            f"[nlos] receiver positions from {args.receiver_pos_file} rows={len(pos_times)}",
            flush=True,
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_nlos = 0
    n_skipped_epochs = 0
    n_epochs = int(data["n_epochs"])
    with args.out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tow",
                "epoch_idx",
                "prn",
                "is_los",
                "system",
                "svid",
                "elevation_deg",
                "receiver_source",
                "receiver_time_delta_s",
            ]
        )
        for start in range(0, n_epochs, int(args.batch_size)):
            end = min(start + int(args.batch_size), n_epochs)
            rx_rows: list[np.ndarray] = []
            source_rows: list[str] = []
            delta_rows: list[float] = []
            sat_rows: list[np.ndarray] = []
            real_epoch_indices: list[int] = []
            for epoch_idx in range(start, end):
                rx, source, delta = _receiver_for_epoch(
                    data,
                    epoch_idx,
                    pos_times=pos_times,
                    pos_positions=pos_positions,
                    pos_tolerance_s=float(args.pos_time_tolerance_s),
                    missing_pos_policy=str(args.missing_pos_policy),
                )
                if rx is None:
                    n_skipped_epochs += 1
                    continue
                sat_ecef = np.asarray(data["sat_ecef"][epoch_idx], dtype=np.float64)
                rx_rows.append(rx)
                source_rows.append(source)
                delta_rows.append(delta)
                sat_rows.append(sat_ecef)
                real_epoch_indices.append(epoch_idx)
            if not rx_rows:
                continue
            los_rows = _check_los(bvh, rx_rows, sat_rows)
            for row_idx, epoch_idx in enumerate(real_epoch_indices):
                rows, nlos = _write_batch_rows(
                    writer,
                    data,
                    epoch_idx,
                    np.asarray([rx_rows[row_idx]], dtype=np.float64),
                    [source_rows[row_idx]],
                    [delta_rows[row_idx]],
                    np.asarray([los_rows[row_idx]], dtype=bool),
                )
                n_rows += rows
                n_nlos += nlos
            if args.progress_every_batches > 0:
                batch_idx = start // int(args.batch_size) + 1
                if batch_idx % int(args.progress_every_batches) == 0 or end >= n_epochs:
                    frac_done = 100.0 * float(end) / float(max(n_epochs, 1))
                    print(
                        f"[nlos] progress epochs={end}/{n_epochs} "
                        f"({frac_done:.1f}%) rows={n_rows} nlos={n_nlos}",
                        flush=True,
                    )

    frac = float(n_nlos / n_rows) if n_rows else 0.0
    print(
        f"[nlos] wrote {args.out_csv} rows={n_rows} nlos={n_nlos} "
        f"nlos_frac={frac:.4f} skipped_epochs={n_skipped_epochs}",
        flush=True,
    )
    return {
        "epochs": float(n_epochs),
        "rows": float(n_rows),
        "nlos": float(n_nlos),
        "nlos_frac": frac,
        "skipped_epochs": float(n_skipped_epochs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run", type=Path, default=None, help="PPC run under data-root, e.g. tokyo/run1")
    parser.add_argument("--data-dir", type=Path, default=None, help="Direct PPC run directory")
    parser.add_argument("--plateau-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--systems", default="G,E,J,C,R")
    parser.add_argument("--kinds", default="bldg,brid")
    parser.add_argument("--plateau-zone", type=int, default=0, help="0=auto from city; Tokyo=9, Nagoya=7")
    parser.add_argument("--geoid-correction", default="egm96", help="egm96, none, or constant metres")
    parser.add_argument(
        "--triangle-cache-npz",
        type=Path,
        default=None,
        help="Optional .npz triangle cache. Existing cache skips CityGML parsing; missing cache is written.",
    )
    parser.add_argument("--receiver-pos-file", type=Path, default=None)
    parser.add_argument("--pos-time-tolerance-s", type=float, default=0.11)
    parser.add_argument(
        "--missing-pos-policy",
        choices=("skip", "reference"),
        default="skip",
        help="When --receiver-pos-file is missing an epoch: skip rows, or fall back to reference.",
    )
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=0, help="0=all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    args = parser.parse_args()
    if args.data_dir is None and args.run is None:
        parser.error("one of --run or --data-dir is required")
    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    build_nlos_csv(args)


if __name__ == "__main__":
    main()
