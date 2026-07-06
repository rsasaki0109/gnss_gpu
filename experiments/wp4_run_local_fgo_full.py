#!/usr/bin/env python3
"""WP4 driver: sweep the DD/LAMBDA local-FGO pipeline over a full PPC run.

This is a thin orchestration layer around
``experiments/solve_ppc_segment_multifamily_fgo.py`` (used as-is, via its CLI
``main()`` with a monkeypatched ``sys.argv`` — no changes to the solver or to
``python/gnss_gpu/local_fgo.py``). It does three things the single-window
solver script does not:

1. Builds a full-coverage seed-position ``.pos`` file for the whole run by
   gap-filling a partial WLS/native-FGO backbone trajectory (from
   ``results/wp3a`` or ``results/wp3b``) with linear ECEF interpolation, so
   every rover epoch has a seed position to anchor/prior the local FGO.
2. Partitions the full rover-epoch timeline into contiguous TOW windows and
   drives the solver script window-by-window (a fresh window keeps each
   local FGO solve to a bounded, tractable size).
3. Merges each window's ``--out-pos`` (float+fixed FGO positions) and
   ``--out-fixed-only-pos`` (used to recover the per-epoch LAMBDA fix mask by
   diffing against the known seed, since the solver script does not itself
   emit a fix mask) into one full-run trajectory CSV.

See ``results/wp4/WP4_REPORT.md`` for the full writeup.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate import ecef_to_lla  # noqa: E402
from ppc_window_geometry import _datetime_to_tow  # noqa: E402

import solve_ppc_segment_multifamily_fgo as segment_solver  # noqa: E402

DATA_ROOT = Path("E:/datasets/PPC-Dataset-data")


# --------------------------------------------------------------------------
# Pure helpers (unit-testable without the dataset).
# --------------------------------------------------------------------------


def parse_rover_tows_from_obs(rover_obs_path: Path) -> np.ndarray:
    """Return the sorted GPS TOW (rounded to 0.1 s) of every epoch header in a RINEX obs file.

    Only reads epoch header lines (``> YYYY MM DD hh mm ss.sss flag n_sat``);
    does not parse per-satellite observation records, so this is cheap even
    for a full run's rover.obs.
    """
    tows: list[float] = []
    with Path(rover_obs_path).open() as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            parts = line[2:].split()
            if len(parts) < 6:
                continue
            sec = float(parts[5])
            sec_int = int(sec)
            usec = int(round((sec - sec_int) * 1.0e6))
            epoch_time = datetime(
                int(parts[0]), int(parts[1]), int(parts[2]),
                int(parts[3]), int(parts[4]), sec_int, usec,
            )
            tows.append(round(_datetime_to_tow(epoch_time), 1))
    return np.asarray(sorted(tows), dtype=np.float64)


def load_seed_ecef_csv(csv_path: Path) -> dict[float, np.ndarray]:
    """Load a ``tow,...,ecef_x,ecef_y,ecef_z,...`` backbone CSV into a TOW->ECEF map."""
    out: dict[float, np.ndarray] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                tow = round(float(row["tow"]), 1)
                ecef = np.array(
                    [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                    dtype=np.float64,
                )
            except (KeyError, ValueError):
                continue
            out[tow] = ecef
    return out


def fill_seed_gaps(
    tows: np.ndarray,
    seed_by_tow: dict[float, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fill missing seed epochs with linear ECEF interpolation over time.

    Returns ``(positions, is_interpolated)`` where ``positions`` has shape
    ``(len(tows), 3)`` and ``is_interpolated[i]`` is True when ``tows[i]``
    was not present in ``seed_by_tow`` and had to be filled. Epochs outside
    the known-seed TOW span are constant-extrapolated from the nearest
    known endpoint.
    """
    tows = np.asarray(tows, dtype=np.float64).ravel()
    known_tows = np.asarray(sorted(seed_by_tow), dtype=np.float64)
    if known_tows.size == 0:
        raise ValueError("seed_by_tow must contain at least one entry")
    known_ecef = np.vstack([seed_by_tow[float(t)] for t in known_tows])

    positions = np.empty((tows.size, 3), dtype=np.float64)
    is_interpolated = np.zeros(tows.size, dtype=bool)
    for i, tow in enumerate(tows):
        hit = seed_by_tow.get(float(tow))
        if hit is not None:
            positions[i] = hit
            continue
        is_interpolated[i] = True
        idx = int(np.searchsorted(known_tows, tow))
        if idx <= 0:
            positions[i] = known_ecef[0]
        elif idx >= known_tows.size:
            positions[i] = known_ecef[-1]
        else:
            t0, t1 = known_tows[idx - 1], known_tows[idx]
            frac = 0.0 if t1 == t0 else float((tow - t0) / (t1 - t0))
            positions[i] = known_ecef[idx - 1] + frac * (known_ecef[idx] - known_ecef[idx - 1])
    return positions, is_interpolated


def write_pos_file(path: Path, tows: np.ndarray, positions: np.ndarray, status: int = 4) -> None:
    """Write an RTKLIB-column-order ``.pos`` file: ``week tow x y z lat lon height Q ...``.

    ``experiments/exp_ppc_ctrbpf_fgo.py:_write_pos_file`` (used internally by
    ``solve_ppc_segment_multifamily_fgo.py`` for its own ``--out-pos``/
    ``--out-fixed-only-pos`` outputs) puts its ``status`` argument in an extra
    13th whitespace token that does *not* line up with the column
    ``exp_ppc_ctrbpf_fgo.py:_load_hybrid_pos_file`` actually reads
    (``parts[8]``, the real RTKLIB ``Q`` column) — every row it writes is
    parsed back with ``status=0`` and silently dropped (verified: round-tripping
    ``_write_pos_file`` through ``_load_hybrid_pos_file`` returns an empty
    dict). That mismatch would make ``--seed-pos`` unusable for any file this
    driver writes, so our own seed file is written in the real RTKLIB column
    order instead (``Q`` at index 8), which ``_load_hybrid_pos_file`` parses
    correctly. This is an additive-only local helper (no edits to
    ``exp_ppc_ctrbpf_fgo.py``); see ``read_pos_ecef`` for how this driver
    reads the solver's own ``_write_pos_file``-formatted outputs back
    (position columns only, status-independent).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        fh.write("% WP4 full-coverage seed (gap-filled WP3b backbone)\n")
        fh.write(
            "%  GPST_week   tow(s)      x-ecef(m)        y-ecef(m)        z-ecef(m)"
            "   lat(deg)   lon(deg)  height(m)   Q  ns   sdx    sdy    sdz   age  ratio\n"
        )
        for tow, pos in zip(tows, positions, strict=True):
            fh.write(
                f"2324 {float(tow):14.4f} "
                f"{pos[0]:16.4f} {pos[1]:16.4f} {pos[2]:16.4f}  "
                f"0.0 0.0 0.0 {int(status)}   0  0.000  0.000  0.000  0.00  0.0\n"
            )


def read_pos_ecef(path: Path) -> dict[float, np.ndarray]:
    """Read TOW->ECEF from any ``.pos`` file with ``week tow x y z ...`` column order.

    Status-independent by design (see ``write_pos_file`` docstring for why):
    only reads ``parts[1:5]`` (tow, x, y, z), which both ``write_pos_file``
    (this module) and ``exp_ppc_ctrbpf_fgo.py:_write_pos_file`` (the solver
    script's own writer) agree on.
    """
    out: dict[float, np.ndarray] = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                tow = round(float(parts[1]), 1)
                ecef = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
            except ValueError:
                continue
            if not np.all(np.isfinite(ecef)):
                continue
            out[tow] = ecef
    return out


def make_windows(tows: np.ndarray, window_epochs: int) -> list[tuple[float, float]]:
    """Partition a sorted TOW array into contiguous ``(start_tow, end_tow)`` windows."""
    tows = np.asarray(tows, dtype=np.float64).ravel()
    window_epochs = max(1, int(window_epochs))
    windows: list[tuple[float, float]] = []
    for start in range(0, tows.size, window_epochs):
        end = min(start + window_epochs, tows.size) - 1
        windows.append((float(tows[start]), float(tows[end])))
    return windows


def recover_fix_mask(
    seed_ecef: np.ndarray,
    fixed_only_ecef: np.ndarray,
    atol: float = 1.0e-6,
) -> np.ndarray:
    """Recover the per-epoch LAMBDA-fix mask without modifying the solver script.

    ``solve_ppc_segment_multifamily_fgo.py`` writes ``--out-fixed-only-pos``
    as the seed trajectory with *only* the LAMBDA-fixed epochs overwritten by
    the FGO solution (see its ``main()``: ``fixed_only_positions =
    seed_positions.copy(); fixed_only_positions[idx] = fgo_positions[idx]``
    for ``idx`` in the fixed set). Comparing that file against the known
    seed recovers exactly which epochs were fixed, with no solver changes.
    """
    seed_ecef = np.asarray(seed_ecef, dtype=np.float64).reshape(-1, 3)
    fixed_only_ecef = np.asarray(fixed_only_ecef, dtype=np.float64).reshape(-1, 3)
    if seed_ecef.shape != fixed_only_ecef.shape:
        raise ValueError("seed_ecef and fixed_only_ecef must have the same shape")
    return ~np.all(np.isclose(seed_ecef, fixed_only_ecef, atol=atol, rtol=0.0), axis=1)


def ecef_rows_to_llh_rows(
    tows: np.ndarray,
    ecef: np.ndarray,
    fix_mask: np.ndarray,
) -> list[dict[str, object]]:
    """Build ``tow,lat_deg,lon_deg,height_m,ecef_x,ecef_y,ecef_z,fix`` rows."""
    rows: list[dict[str, object]] = []
    for tow, xyz, is_fix in zip(tows, ecef, fix_mask, strict=True):
        lat_rad, lon_rad, height_m = ecef_to_lla(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        rows.append(
            {
                "tow": float(tow),
                "lat_deg": float(np.degrees(lat_rad)),
                "lon_deg": float(np.degrees(lon_rad)),
                "height_m": float(height_m),
                "ecef_x": float(xyz[0]),
                "ecef_y": float(xyz[1]),
                "ecef_z": float(xyz[2]),
                "fix": int(bool(is_fix)),
            }
        )
    return rows


def write_trajectory_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tow", "lat_deg", "lon_deg", "height_m", "ecef_x", "ecef_y", "ecef_z", "fix"]
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Dataset-dependent orchestration.
# --------------------------------------------------------------------------


@dataclass
class WindowRunResult:
    start_tow: float
    end_tow: float
    ok: bool
    elapsed_s: float
    n_epochs: int = 0
    n_dd_epochs: int = 0
    n_dd_pairs: int = 0
    n_fixed: int = 0
    n_fixed_observations: int = 0
    n_fixed_epochs: int = 0
    best_ratio: float = 0.0
    error: str = ""
    tows: np.ndarray | None = None
    fgo_ecef: np.ndarray | None = None
    fix_mask: np.ndarray | None = None


def run_one_window(
    *,
    run: str,
    data_root: Path,
    seed_pos_path: Path,
    start_tow: float,
    end_tow: float,
    out_dir: Path,
    tag: str,
    systems: str = "G,E",
    dd_families: str = "L1_E1_B1,L5_E5A_B2A",
    extra_args: list[str] | None = None,
) -> WindowRunResult:
    """Invoke ``solve_ppc_segment_multifamily_fgo.main()`` in-process for one window."""
    out_pos = out_dir / f"{tag}.pos"
    out_fixed_pos = out_dir / f"{tag}_fixed_only.pos"
    out_summary = out_dir / f"{tag}_summary.csv"
    argv = [
        "solve_ppc_segment_multifamily_fgo.py",
        "--run", str(run),
        "--data-root", str(data_root),
        "--seed-pos", str(seed_pos_path),
        "--start-tow", f"{start_tow:.1f}",
        "--end-tow", f"{end_tow:.1f}",
        "--systems", str(systems),
        "--dd-families", str(dd_families),
        "--out-pos", str(out_pos),
        "--out-fixed-only-pos", str(out_fixed_pos),
        "--out-summary", str(out_summary),
    ]
    if extra_args:
        argv.extend(extra_args)

    old_argv = sys.argv
    t0 = time.time()
    try:
        sys.argv = argv
        segment_solver.main()
        elapsed = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        return WindowRunResult(start_tow=start_tow, end_tow=end_tow, ok=False, elapsed_s=elapsed, error=repr(exc))
    finally:
        sys.argv = old_argv

    fgo_by_tow = read_pos_ecef(out_pos)
    fixed_only_by_tow = read_pos_ecef(out_fixed_pos)
    seed_by_tow = read_pos_ecef(seed_pos_path)
    tows = np.asarray(sorted(fgo_by_tow), dtype=np.float64)
    fgo_ecef = np.vstack([fgo_by_tow[t] for t in tows])
    fixed_only_ecef = np.vstack([fixed_only_by_tow[t] for t in tows])
    seed_ecef = np.vstack([seed_by_tow[round(float(t), 1)] for t in tows])
    fix_mask = recover_fix_mask(seed_ecef, fixed_only_ecef)

    summary_row: dict[str, str] = {}
    if out_summary.exists():
        with out_summary.open(newline="") as fh:
            reader = csv.DictReader(fh)
            summary_row = next(reader, {})

    def _get_int(key: str) -> int:
        try:
            return int(float(summary_row.get(key, 0) or 0))
        except ValueError:
            return 0

    return WindowRunResult(
        start_tow=start_tow,
        end_tow=end_tow,
        ok=True,
        elapsed_s=elapsed,
        n_epochs=int(tows.size),
        n_dd_epochs=_get_int("dd_epochs"),
        n_dd_pairs=_get_int("dd_pairs_total"),
        n_fixed=_get_int("lambda_n_fixed"),
        n_fixed_observations=_get_int("lambda_n_fixed_observations"),
        n_fixed_epochs=int(np.count_nonzero(fix_mask)),
        tows=tows,
        fgo_ecef=fgo_ecef,
        fix_mask=fix_mask,
    )


def build_full_coverage_seed(
    *,
    rover_obs_path: Path,
    backbone_csv_path: Path,
    out_pos_path: Path,
) -> tuple[np.ndarray, int]:
    tows = parse_rover_tows_from_obs(rover_obs_path)
    backbone = load_seed_ecef_csv(backbone_csv_path)
    positions, is_interpolated = fill_seed_gaps(tows, backbone)
    write_pos_file(out_pos_path, tows, positions)
    return tows, int(np.count_nonzero(is_interpolated))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run1")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--backbone-csv",
        type=Path,
        default=_PROJECT_ROOT / "results" / "wp3b" / "tokyo_run1_fgo_imu_doppler_huber.csv",
        help="Full-run WLS/native-FGO backbone CSV used as the local-FGO seed source.",
    )
    parser.add_argument("--out-dir", type=Path, default=_PROJECT_ROOT / "results" / "wp4")
    parser.add_argument("--window-epochs", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=0, help="0 = all windows.")
    parser.add_argument(
        "--stride-windows",
        type=int,
        default=1,
        help="Process every Nth window (stratified subset); 1 = every window.",
    )
    parser.add_argument("--systems", default="G,E")
    parser.add_argument("--dd-families", default="L1_E1_B1,L5_E5A_B2A")
    parser.add_argument(
        "--dd-base-interp",
        action="store_true",
        help="Forward --dd-base-interp to the solver (base logs at 1 Hz vs 5 Hz rover; "
        "without this, DD carrier is only found on ~20%% of epochs).",
    )
    parser.add_argument("--rebuild-seed", action="store_true")
    parser.add_argument(
        "--traj-out",
        type=Path,
        default=_PROJECT_ROOT / "results" / "wp4" / "tokyo_run1_local_fgo_lambda.csv",
    )
    parser.add_argument(
        "--segment-stats-out",
        type=Path,
        default=_PROJECT_ROOT / "results" / "wp4" / "per_segment_stats.csv",
    )
    args = parser.parse_args()

    city, run_name = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run_name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    seed_pos_path = out_dir / f"{city}_{run_name}_seed_full_coverage.pos"
    if args.rebuild_seed or not seed_pos_path.exists():
        rover_tows, n_interp = build_full_coverage_seed(
            rover_obs_path=run_dir / "rover.obs",
            backbone_csv_path=args.backbone_csv,
            out_pos_path=seed_pos_path,
        )
        print(f"seed: {rover_tows.size} epochs, {n_interp} interpolated ({100.0 * n_interp / rover_tows.size:.1f}%)")
    else:
        rover_tows = parse_rover_tows_from_obs(run_dir / "rover.obs")

    windows = make_windows(rover_tows, int(args.window_epochs))
    windows = windows[:: max(1, int(args.stride_windows))]
    if int(args.max_windows) > 0:
        windows = windows[: int(args.max_windows)]

    extra_args = ["--dd-base-interp"] if bool(args.dd_base_interp) else []

    results: list[WindowRunResult] = []
    t_start = time.time()
    for i, (start_tow, end_tow) in enumerate(windows):
        tag = f"win_{i:04d}_{start_tow:.1f}_{end_tow:.1f}"
        result = run_one_window(
            run=str(args.run),
            data_root=args.data_root,
            seed_pos_path=seed_pos_path,
            start_tow=start_tow,
            end_tow=end_tow,
            out_dir=windows_dir,
            tag=tag,
            systems=str(args.systems),
            dd_families=str(args.dd_families),
            extra_args=extra_args,
        )
        results.append(result)
        status = "ok" if result.ok else f"FAILED ({result.error})"
        print(
            f"[{i + 1}/{len(windows)}] tow={start_tow:.1f}:{end_tow:.1f} "
            f"{status} epochs={result.n_epochs} fixed_epochs={result.n_fixed_epochs} "
            f"dt={result.elapsed_s:.2f}s"
        )

    total_elapsed = time.time() - t_start

    all_tows: list[float] = []
    all_ecef: list[np.ndarray] = []
    all_fix: list[bool] = []
    for result in results:
        if not result.ok or result.tows is None:
            continue
        all_tows.extend(result.tows.tolist())
        all_ecef.extend(result.fgo_ecef.tolist())
        all_fix.extend(result.fix_mask.tolist())

    if all_tows:
        order = np.argsort(np.asarray(all_tows, dtype=np.float64))
        tows_arr = np.asarray(all_tows, dtype=np.float64)[order]
        ecef_arr = np.asarray(all_ecef, dtype=np.float64)[order]
        fix_arr = np.asarray(all_fix, dtype=bool)[order]
        rows = ecef_rows_to_llh_rows(tows_arr, ecef_arr, fix_arr)
        write_trajectory_csv(args.traj_out, rows)
        print(f"wrote {args.traj_out} ({len(rows)} rows)")
    else:
        print("no windows produced a solution; trajectory CSV not written")

    with Path(args.segment_stats_out).open("w", newline="") as fh:
        fieldnames = [
            "start_tow", "end_tow", "ok", "elapsed_s", "n_epochs",
            "n_dd_epochs", "n_dd_pairs", "n_fixed", "n_fixed_observations",
            "n_fixed_epochs", "error",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "start_tow": result.start_tow,
                    "end_tow": result.end_tow,
                    "ok": int(result.ok),
                    "elapsed_s": result.elapsed_s,
                    "n_epochs": result.n_epochs,
                    "n_dd_epochs": result.n_dd_epochs,
                    "n_dd_pairs": result.n_dd_pairs,
                    "n_fixed": result.n_fixed,
                    "n_fixed_observations": result.n_fixed_observations,
                    "n_fixed_epochs": result.n_fixed_epochs,
                    "error": result.error,
                }
            )
    print(f"wrote {args.segment_stats_out}")
    n_ok = sum(1 for r in results if r.ok)
    print(
        f"windows: {n_ok}/{len(results)} ok, total wall time {total_elapsed:.1f}s "
        f"({total_elapsed / max(1, len(results)):.2f}s/window avg)"
    )


if __name__ == "__main__":
    main()
