#!/usr/bin/env python3
"""WP5 driver: anchor the DD/LAMBDA local-FGO pipeline on libgnss++ RTK fixes.

Extends ``experiments/wp4_run_local_fgo_full.py`` (imported, not copied) with
the three things WP4's negative result said were missing (see
``results/wp4/WP4_REPORT.md`` "Suggested next step" and
``internal_docs/inuex35_tc_fgo_benchmark.md``):

1. **Hybrid seed**: instead of WP4's pure WLS/native-FGO backbone seed
   (85-105 m RMS -- "two to three orders of magnitude coarser than the
   cm-level anchor this machinery expects"), the seed is RTK positions
   where the libgnss++ ``.pos`` artifact has *any* valid Status (FIX or
   FLOAT) and the WP3b backbone (gap-filled, as in WP4) everywhere else.
2. **Per-epoch anchor priors**: each window is solved with
   ``--anchor-source rtk`` (new in ``solve_ppc_segment_multifamily_fgo.py``,
   see its module docstring / ``_build_rtk_anchor_priors``) -- a tight prior
   (default sigma 0.07 m) at RTK FIX epochs, a looser one (default 2.0 m) at
   RTK FLOAT epochs, and no per-epoch prior elsewhere (endpoint-only, as in
   WP4). Anchor gating uses only the RTK Status column, never ground truth.
3. **External AR validation**: ``--dd-pr`` (DD-pseudorange factors already
   existed in the solver script) feeds ``LocalFgoProblem.dd_pseudorange``
   into the new ``local_fgo.py`` DDPR cross-check gate
   (``--lambda-ddpr-reject-threshold``), and ``--lambda-min-epochs`` is
   raised from WP4's 2 to 5 (WP5 work item 3's minimum-segment-length gate).

See ``results/wp5/WP5_REPORT.md`` for the full writeup.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from wp4_run_local_fgo_full import (  # noqa: E402
    ecef_rows_to_llh_rows,
    fill_seed_gaps,
    load_seed_ecef_csv,
    make_windows,
    parse_rover_tows_from_obs,
    read_pos_ecef,
    recover_fix_mask,
    write_pos_file,
    write_trajectory_csv,
)

import solve_ppc_segment_multifamily_fgo as segment_solver  # noqa: E402

DATA_ROOT = Path("E:/datasets/PPC-Dataset-data")
DEFAULT_RTK_POS = (
    _PROJECT_ROOT / "experiments" / "results" / "libgnss_rtk_pos_v5" / "tokyo_run1_full.pos"
)
DEFAULT_BACKBONE_CSV = _PROJECT_ROOT / "results" / "wp3b" / "tokyo_run1_fgo_imu_doppler_huber.csv"


# --------------------------------------------------------------------------
# Pure helpers (unit-testable without the dataset).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RtkPosRecord:
    """One libgnss++ ``.pos`` row with optional quality metadata (WP12e)."""

    ecef: np.ndarray
    status: int
    nsats: int = 0
    pdop: float = 0.0
    ratio: float = 0.0


def load_rtk_pos_extended(path: Path) -> dict[float, RtkPosRecord]:
    """Parse a libgnss++ ``.pos`` file into ``TOW -> RtkPosRecord``.

    Column layout: ``GPS_Week GPS_TOW X Y Z Lat Lon Height Status NumSat PDOP Ratio ...``
    """
    out: dict[float, RtkPosRecord] = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                tow = round(float(parts[1]), 1)
                ecef = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
                status = int(float(parts[8]))
                nsats = int(float(parts[9])) if len(parts) > 9 else 0
                pdop = float(parts[10]) if len(parts) > 10 else 0.0
                ratio = float(parts[11]) if len(parts) > 11 else 0.0
            except ValueError:
                continue
            if not np.all(np.isfinite(ecef)):
                continue
            out[tow] = RtkPosRecord(
                ecef=ecef,
                status=status,
                nsats=max(0, nsats),
                pdop=max(0.0, pdop),
                ratio=max(0.0, ratio),
            )
    return out


def load_rtk_pos_with_status(path: Path) -> dict[float, tuple[np.ndarray, int]]:
    """Parse a libgnss++ ``.pos`` file into ``TOW -> (ECEF, Status)``.

    Column layout: ``GPS_Week GPS_TOW X Y Z Lat Lon Height Status ...``
    (see ``experiments/results/libgnss_rtk_pos_v5/tokyo_run1_full.pos``
    header). Unlike ``exp_ppc_ctrbpf_fgo._load_hybrid_pos_file`` (which
    drops Status==0 rows), this keeps every parseable row so callers can
    decide their own fix/float/none classification.
    """
    return {tow: (rec.ecef, rec.status) for tow, rec in load_rtk_pos_extended(path).items()}


def anchor_sigma_m(
    record: RtkPosRecord,
    anchor_class: int,
    *,
    fix_sigma_m: float,
    float_sigma_m: float,
    sigma_scale: float = 1.0,
    quality_weight: bool = True,
) -> float:
    """Truth-honest anchor σ for FIX (class 2) or FLOAT (class 1) rows (WP12e)."""

    if int(anchor_class) == 2:
        base = float(fix_sigma_m)
    elif int(anchor_class) == 1:
        pdop_sigma = float(record.pdop) * 0.8 if record.pdop > 0.0 else 0.0
        base = float(max(float_sigma_m, pdop_sigma, 2.0))
        base = min(base, 5.0)
    else:
        return float("inf")
    sigma = base * max(float(sigma_scale), 1.0e-6)
    if quality_weight:
        ns = max(int(record.nsats), 4)
        sigma *= math.sqrt(8.0 / float(ns))
        if int(anchor_class) == 1 and record.ratio > 0.0 and record.ratio < 3.0:
            sigma *= 1.5
    return float(max(sigma, 0.05))


def build_hybrid_seed(
    tows: np.ndarray,
    rtk_by_tow: dict[float, tuple[np.ndarray, int]],
    backbone_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid seed: RTK ECEF where the RTK ``.pos`` has *any* valid Status, else backbone.

    Returns ``(positions, is_rtk)`` where ``is_rtk[i]`` marks epochs sourced
    from the RTK file (Status != 0, i.e. FIX or FLOAT -- WP5 work item 2's
    "RTK positions where available").
    """
    tows = np.asarray(tows, dtype=np.float64).ravel()
    positions = np.asarray(backbone_positions, dtype=np.float64).copy()
    if positions.shape != (tows.size, 3):
        raise ValueError("backbone_positions must have shape (len(tows), 3)")
    is_rtk = np.zeros(tows.size, dtype=bool)
    for i, tow in enumerate(tows):
        hit = rtk_by_tow.get(round(float(tow), 1))
        if hit is None:
            continue
        ecef, status = hit
        if int(status) == 0:
            continue
        positions[i] = ecef
        is_rtk[i] = True
    return positions, is_rtk


def classify_anchor_status(
    tows: np.ndarray,
    rtk_by_tow: dict[float, tuple[np.ndarray, int]],
    fix_statuses: tuple[int, ...] = (4,),
    float_statuses: tuple[int, ...] = (1, 3),
) -> np.ndarray:
    """Per-epoch anchor class: 2=FIX, 1=FLOAT, 0=none (matches anchor CLI gating)."""
    out = np.zeros(len(tows), dtype=np.int8)
    for i, tow in enumerate(np.asarray(tows, dtype=np.float64)):
        hit = rtk_by_tow.get(round(float(tow), 1))
        if hit is None:
            continue
        _ecef, status = hit
        if int(status) in fix_statuses:
            out[i] = 2
        elif int(status) in float_statuses:
            out[i] = 1
    return out


def nearest_fix_distance_epochs(anchor_class: np.ndarray) -> np.ndarray:
    """For every epoch, the epoch-count distance to the nearest FIX (class==2) epoch.

    ``np.inf`` when there is no FIX epoch at all. Used for the "fix-extension
    length" diagnostic (work item 4/6): how far decimeter accuracy
    propagates outward from an anchor.
    """
    return nearest_anchor_distance_epochs(anchor_class, include_fix=True, include_float=False)


def nearest_anchor_distance_epochs(
    anchor_class: np.ndarray,
    *,
    include_fix: bool = True,
    include_float: bool = False,
) -> np.ndarray:
    """Epoch-count distance to the nearest enabled anchor class (WP12e)."""

    anchor_class = np.asarray(anchor_class)
    n = anchor_class.size
    dist = np.full(n, np.inf, dtype=np.float64)
    anchor_mask = np.zeros(n, dtype=bool)
    if include_fix:
        anchor_mask |= anchor_class == 2
    if include_float:
        anchor_mask |= anchor_class == 1
    anchor_idx = np.flatnonzero(anchor_mask)
    if anchor_idx.size == 0:
        return dist
    last = -np.inf
    for i in range(n):
        if anchor_mask[i]:
            last = i
        if np.isfinite(last):
            dist[i] = min(dist[i], i - last)
    last = np.inf
    for i in range(n - 1, -1, -1):
        if anchor_mask[i]:
            last = i
        if np.isfinite(last):
            dist[i] = min(dist[i], last - i)
    return dist


def compute_extension_stats(
    tows: np.ndarray,
    ecef: np.ndarray,
    anchor_class: np.ndarray,
    reference: dict[float, np.ndarray],
    *,
    pass_threshold_m: float = 0.5,
) -> dict[str, object]:
    """Work item 4/6: fix-extension length distribution.

    For every non-FIX epoch (``anchor_class != 2``) that is within the
    reference grid and passes ``pass_threshold_m``, record its epoch-count
    distance to the nearest FIX anchor. The distribution of these distances
    quantifies how far decimeter accuracy propagates outward from RTK FIX
    anchors through the DD-carrier/LAMBDA extension mechanism.
    """
    dist = nearest_fix_distance_epochs(anchor_class)
    extension_distances: list[float] = []
    n_ref_covered = 0
    n_pass = 0
    n_fix = int(np.count_nonzero(anchor_class == 2))
    n_float = int(np.count_nonzero(anchor_class == 1))
    n_none = int(np.count_nonzero(anchor_class == 0))
    for i, tow in enumerate(np.asarray(tows, dtype=np.float64)):
        ref = reference.get(round(float(tow), 1))
        if ref is None:
            continue
        n_ref_covered += 1
        err = float(np.linalg.norm(np.asarray(ecef[i], dtype=np.float64) - ref))
        if err >= pass_threshold_m:
            continue
        n_pass += 1
        if anchor_class[i] != 2 and np.isfinite(dist[i]):
            extension_distances.append(float(dist[i]))

    out: dict[str, object] = {
        "n_fix_epochs": n_fix,
        "n_float_epochs": n_float,
        "n_none_epochs": n_none,
        "n_ref_covered": n_ref_covered,
        "n_pass_lt_50cm": n_pass,
        "n_extension_pass_lt_50cm": len(extension_distances),
    }
    if extension_distances:
        vals = np.asarray(extension_distances, dtype=np.float64)
        out["extension_epochs_median"] = float(np.median(vals))
        out["extension_epochs_p90"] = float(np.percentile(vals, 90.0))
        out["extension_epochs_max"] = float(np.max(vals))
    else:
        out["extension_epochs_median"] = 0.0
        out["extension_epochs_p90"] = 0.0
        out["extension_epochs_max"] = 0.0
    return out


# --------------------------------------------------------------------------
# Dataset-dependent orchestration.
# --------------------------------------------------------------------------


@dataclass
class AnchoredWindowRunResult:
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
    anchor_fix_count: int = 0
    anchor_float_count: int = 0
    lambda_n_segments_rejected_short: int = 0
    lambda_n_ddpr_rejected_iterations: int = 0
    lambda_n_ddpr_rejected_observations: int = 0
    error: str = ""
    tows: np.ndarray | None = None
    fgo_ecef: np.ndarray | None = None
    fix_mask: np.ndarray | None = None


def run_one_window_anchored(
    *,
    run: str,
    data_root: Path,
    seed_pos_path: Path,
    rtk_pos_path: Path,
    start_tow: float,
    end_tow: float,
    out_dir: Path,
    tag: str,
    systems: str,
    dd_families: str,
    dd_base_interp: bool,
    anchor_fix_sigma_m: float,
    anchor_float_sigma_m: float,
    anchor_fix_statuses: str,
    anchor_float_statuses: str,
    lambda_min_epochs: int,
    lambda_ddpr_reject_threshold: float,
    use_dd_pr: bool,
) -> AnchoredWindowRunResult:
    """Invoke ``solve_ppc_segment_multifamily_fgo.main()`` in-process, WP5-anchored."""
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
        "--anchor-source", "rtk",
        "--anchor-pos", str(rtk_pos_path),
        "--anchor-fix-sigma-m", f"{anchor_fix_sigma_m:.4f}",
        "--anchor-float-sigma-m", f"{anchor_float_sigma_m:.4f}",
        "--anchor-fix-statuses", str(anchor_fix_statuses),
        "--anchor-float-statuses", str(anchor_float_statuses),
        "--lambda-min-epochs", str(int(lambda_min_epochs)),
        "--lambda-ddpr-reject-threshold", f"{lambda_ddpr_reject_threshold:.6f}",
        "--out-pos", str(out_pos),
        "--out-fixed-only-pos", str(out_fixed_pos),
        "--out-summary", str(out_summary),
    ]
    if dd_base_interp:
        argv.append("--dd-base-interp")
    if use_dd_pr:
        argv.append("--dd-pr")

    old_argv = sys.argv
    t0 = time.time()
    try:
        sys.argv = argv
        segment_solver.main()
        elapsed = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        return AnchoredWindowRunResult(
            start_tow=start_tow, end_tow=end_tow, ok=False, elapsed_s=elapsed, error=repr(exc)
        )
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

    return AnchoredWindowRunResult(
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
        anchor_fix_count=_get_int("anchor_fix_count"),
        anchor_float_count=_get_int("anchor_float_count"),
        lambda_n_segments_rejected_short=_get_int("lambda_n_segments_rejected_short"),
        lambda_n_ddpr_rejected_iterations=_get_int("lambda_n_ddpr_rejected_iterations"),
        lambda_n_ddpr_rejected_observations=_get_int("lambda_n_ddpr_rejected_observations"),
        tows=tows,
        fgo_ecef=fgo_ecef,
        fix_mask=fix_mask,
    )


def build_hybrid_coverage_seed(
    *,
    rover_obs_path: Path,
    backbone_csv_path: Path,
    rtk_pos_path: Path,
    out_pos_path: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    tows = parse_rover_tows_from_obs(rover_obs_path)
    backbone = load_seed_ecef_csv(backbone_csv_path)
    backbone_positions, is_interp = fill_seed_gaps(tows, backbone)
    rtk_by_tow = load_rtk_pos_with_status(rtk_pos_path)
    positions, is_rtk = build_hybrid_seed(tows, rtk_by_tow, backbone_positions)
    write_pos_file(out_pos_path, tows, positions)
    stats = {
        "n_epochs": int(tows.size),
        "n_rtk": int(np.count_nonzero(is_rtk)),
        "n_backbone": int(np.count_nonzero(~is_rtk)),
        "n_backbone_interpolated": int(np.count_nonzero(is_interp & ~is_rtk)),
    }
    return tows, stats


def _load_reference_ecef(run_dir: Path) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    with (run_dir / "reference.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                tow = round(float(row["GPS TOW (s)"]), 1)
                xyz = np.array(
                    [float(row["ECEF X (m)"]), float(row["ECEF Y (m)"]), float(row["ECEF Z (m)"])],
                    dtype=np.float64,
                )
            except (KeyError, ValueError):
                continue
            out[tow] = xyz
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run1")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--rtk-pos", type=Path, default=DEFAULT_RTK_POS)
    parser.add_argument("--backbone-csv", type=Path, default=DEFAULT_BACKBONE_CSV)
    parser.add_argument("--out-dir", type=Path, default=_PROJECT_ROOT / "results" / "wp5")
    parser.add_argument("--window-epochs", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=0, help="0 = all windows.")
    parser.add_argument("--stride-windows", type=int, default=1)
    parser.add_argument("--systems", default="G,E")
    parser.add_argument("--dd-families", default="L1_E1_B1,L5_E5A_B2A")
    parser.add_argument("--dd-base-interp", action="store_true", default=True)
    parser.add_argument("--no-dd-base-interp", dest="dd_base_interp", action="store_false")
    parser.add_argument("--no-dd-pr", dest="use_dd_pr", action="store_false", default=True)
    parser.add_argument("--anchor-fix-sigma-m", type=float, default=0.07)
    parser.add_argument("--anchor-float-sigma-m", type=float, default=2.0)
    parser.add_argument("--anchor-fix-statuses", default="4")
    parser.add_argument("--anchor-float-statuses", default="1,3")
    parser.add_argument("--lambda-min-epochs", type=int, default=5)
    parser.add_argument("--lambda-ddpr-reject-threshold", type=float, default=0.2)
    parser.add_argument("--rebuild-seed", action="store_true")
    parser.add_argument(
        "--traj-out",
        type=Path,
        default=_PROJECT_ROOT / "results" / "wp5" / "tokyo_run1_anchored_fgo.csv",
    )
    parser.add_argument(
        "--segment-stats-out",
        type=Path,
        default=_PROJECT_ROOT / "results" / "wp5" / "per_segment_stats.csv",
    )
    parser.add_argument(
        "--extension-stats-out",
        type=Path,
        default=_PROJECT_ROOT / "results" / "wp5" / "extension_stats.json",
    )
    args = parser.parse_args()

    city, run_name = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run_name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    seed_pos_path = out_dir / f"{city}_{run_name}_hybrid_seed.pos"
    if args.rebuild_seed or not seed_pos_path.exists():
        rover_tows, seed_stats = build_hybrid_coverage_seed(
            rover_obs_path=run_dir / "rover.obs",
            backbone_csv_path=args.backbone_csv,
            rtk_pos_path=args.rtk_pos,
            out_pos_path=seed_pos_path,
        )
        print(
            f"hybrid seed: {seed_stats['n_epochs']} epochs, "
            f"{seed_stats['n_rtk']} RTK ({100.0 * seed_stats['n_rtk'] / seed_stats['n_epochs']:.1f}%), "
            f"{seed_stats['n_backbone']} backbone "
            f"({seed_stats['n_backbone_interpolated']} of those interpolated)"
        )
    else:
        rover_tows = parse_rover_tows_from_obs(run_dir / "rover.obs")

    windows = make_windows(rover_tows, int(args.window_epochs))
    windows = windows[:: max(1, int(args.stride_windows))]
    if int(args.max_windows) > 0:
        windows = windows[: int(args.max_windows)]

    results: list[AnchoredWindowRunResult] = []
    t_start = time.time()
    for i, (start_tow, end_tow) in enumerate(windows):
        tag = f"win_{i:04d}_{start_tow:.1f}_{end_tow:.1f}"
        result = run_one_window_anchored(
            run=str(args.run),
            data_root=args.data_root,
            seed_pos_path=seed_pos_path,
            rtk_pos_path=args.rtk_pos,
            start_tow=start_tow,
            end_tow=end_tow,
            out_dir=windows_dir,
            tag=tag,
            systems=str(args.systems),
            dd_families=str(args.dd_families),
            dd_base_interp=bool(args.dd_base_interp),
            anchor_fix_sigma_m=float(args.anchor_fix_sigma_m),
            anchor_float_sigma_m=float(args.anchor_float_sigma_m),
            anchor_fix_statuses=str(args.anchor_fix_statuses),
            anchor_float_statuses=str(args.anchor_float_statuses),
            lambda_min_epochs=int(args.lambda_min_epochs),
            lambda_ddpr_reject_threshold=float(args.lambda_ddpr_reject_threshold),
            use_dd_pr=bool(args.use_dd_pr),
        )
        results.append(result)
        status = "ok" if result.ok else f"FAILED ({result.error})"
        print(
            f"[{i + 1}/{len(windows)}] tow={start_tow:.1f}:{end_tow:.1f} "
            f"{status} epochs={result.n_epochs} anchors=fix:{result.anchor_fix_count}/"
            f"float:{result.anchor_float_count} fixed_epochs={result.n_fixed_epochs} "
            f"ddpr_rejected_iters={result.lambda_n_ddpr_rejected_iterations} "
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

        rtk_by_tow = load_rtk_pos_with_status(args.rtk_pos)
        anchor_class = classify_anchor_status(
            tows_arr,
            rtk_by_tow,
            fix_statuses=tuple(int(s) for s in str(args.anchor_fix_statuses).split(",") if s.strip()),
            float_statuses=tuple(int(s) for s in str(args.anchor_float_statuses).split(",") if s.strip()),
        )
        reference = _load_reference_ecef(run_dir)
        extension_stats = compute_extension_stats(tows_arr, ecef_arr, anchor_class, reference)
        args.extension_stats_out.parent.mkdir(parents=True, exist_ok=True)
        import json

        args.extension_stats_out.write_text(json.dumps(extension_stats, indent=2), encoding="utf-8")
        print(f"wrote {args.extension_stats_out}: {extension_stats}")
    else:
        print("no windows produced a solution; trajectory CSV not written")

    with Path(args.segment_stats_out).open("w", newline="") as fh:
        fieldnames = [
            "start_tow", "end_tow", "ok", "elapsed_s", "n_epochs",
            "n_dd_epochs", "n_dd_pairs", "n_fixed", "n_fixed_observations",
            "n_fixed_epochs", "anchor_fix_count", "anchor_float_count",
            "lambda_n_segments_rejected_short", "lambda_n_ddpr_rejected_iterations",
            "lambda_n_ddpr_rejected_observations", "error",
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
                    "anchor_fix_count": result.anchor_fix_count,
                    "anchor_float_count": result.anchor_float_count,
                    "lambda_n_segments_rejected_short": result.lambda_n_segments_rejected_short,
                    "lambda_n_ddpr_rejected_iterations": result.lambda_n_ddpr_rejected_iterations,
                    "lambda_n_ddpr_rejected_observations": result.lambda_n_ddpr_rejected_observations,
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
