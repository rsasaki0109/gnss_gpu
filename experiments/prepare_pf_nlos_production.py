#!/usr/bin/env python3
"""Prepare real BVH NLOS masks and production-style PPC smoke on mobile SSD.

Typical flow (tokyo/run1 first):

  PYTHONPATH=python python experiments/prepare_pf_nlos_production.py check
  PYTHONPATH=python python experiments/prepare_pf_nlos_production.py fetch --run tokyo/run1
  PYTHONPATH=python python experiments/prepare_pf_nlos_production.py mask --run tokyo/run1 --max-epochs 120
  PYTHONPATH=python python experiments/prepare_pf_nlos_production.py smoke --run tokyo/run1 --max-epochs 120

Artifacts default to the mobile SSD when ``E:`` is present:

- ``E:/datasets/PPC-Dataset-data`` — PPC GNSS/IMU (already installed)
- ``E:/datasets/plateau/{city}_{run}`` — fetched CityGML subset
- ``E:/datasets/plateau_cache/{city}_{run}_triangles.npz`` — triangle cache
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPC_ROOT = Path("E:/datasets/PPC-Dataset-data")
FALLBACK_PPC_ROOT = PROJECT_ROOT / "datasets" / "PPC-Dataset-data"
DEFAULT_PLATEAU_ROOT = Path("E:/datasets/plateau")
DEFAULT_CACHE_ROOT = Path("E:/datasets/plateau_cache")
MASK_DIR = PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_phase33"

PRESET_BY_CITY = {
    "tokyo": "tokyo23",
    "nagoya": "nagoya",
}
ZONE_BY_CITY = {
    "tokyo": 9,
    "nagoya": 7,
}


def _ppc_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if DEFAULT_PPC_ROOT.exists():
        return DEFAULT_PPC_ROOT
    return FALLBACK_PPC_ROOT.resolve()


def _run_dir(ppc_root: Path, run: str) -> Path:
    return ppc_root / Path(run)


def _city_run(run: str) -> tuple[str, str]:
    city, run_name = str(run).strip().strip("/").split("/", 1)
    return city, run_name


def _plateau_dir(root: Path, run: str) -> Path:
    city, run_name = _city_run(run)
    return root / f"{city}_{run_name}"


def _triangle_cache(root: Path, run: str) -> Path:
    city, run_name = _city_run(run)
    return root / f"{city}_{run_name}_triangles.npz"


def _mask_csv(run: str) -> Path:
    city, run_name = _city_run(run)
    return MASK_DIR / f"{city}_{run_name}_per_epoch_nlos.csv"


def _child_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    pythonpath = str(PROJECT_ROOT / "python")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _run(cmd: list[str]) -> None:
    print("[prep] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=_child_env())


def cmd_check(args: argparse.Namespace) -> int:
    ppc_root = _ppc_root(args.data_root)
    run_dir = _run_dir(ppc_root, args.run)
    required = ("rover.obs", "base.obs", "base.nav", "reference.csv", "imu.csv")
    missing = [name for name in required if not (run_dir / name).is_file()]
    report = {
        "ppc_root": str(ppc_root),
        "run": args.run,
        "run_dir_ok": not missing,
        "missing_run_files": missing,
        "plateau_dir": str(_plateau_dir(args.plateau_root, args.run)),
        "plateau_dir_exists": _plateau_dir(args.plateau_root, args.run).exists(),
        "triangle_cache": str(_triangle_cache(args.cache_root, args.run)),
        "triangle_cache_exists": _triangle_cache(args.cache_root, args.run).exists(),
        "mask_csv": str(_mask_csv(args.run)),
        "mask_csv_exists": _mask_csv(args.run).is_file(),
        "demo_mask_exists": _mask_csv(args.run).is_file(),
    }
    try:
        from gnss_gpu.bvh import BVHAccelerator  # noqa: F401
        report["bvh_import_ok"] = True
    except Exception as exc:  # pragma: no cover - environment specific
        report["bvh_import_ok"] = False
        report["bvh_import_error"] = str(exc)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["run_dir_ok"] and report["bvh_import_ok"] else 2


def cmd_fetch(args: argparse.Namespace) -> int:
    ppc_root = _ppc_root(args.data_root)
    run_dir = _run_dir(ppc_root, args.run)
    city, _ = _city_run(args.run)
    preset = PRESET_BY_CITY.get(city)
    if preset is None:
        raise SystemExit(f"unsupported city in run: {args.run}")
    out_dir = _plateau_dir(args.plateau_root, args.run)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "fetch_plateau_subset.py"),
        "--run-dir",
        str(run_dir),
        "--preset",
        preset,
        "--output-dir",
        str(out_dir),
        "--mesh-radius",
        str(int(args.mesh_radius)),
    ]
    if args.include_bridges:
        cmd.append("--include-bridges")
    if int(args.max_rows) > 0:
        cmd.extend(["--max-rows", str(int(args.max_rows))])
    _run(cmd)
    return 0


def cmd_mask(args: argparse.Namespace) -> int:
    ppc_root = _ppc_root(args.data_root)
    city, _ = _city_run(args.run)
    plateau_dir = _plateau_dir(args.plateau_root, args.run)
    if not plateau_dir.is_dir():
        raise SystemExit(f"plateau dir missing: {plateau_dir} (run: prepare fetch)")
    cache = _triangle_cache(args.cache_root, args.run)
    out_csv = _mask_csv(args.run)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "build_per_epoch_nlos_csv.py"),
        "--data-root",
        str(ppc_root),
        "--run",
        args.run,
        "--plateau-dir",
        str(plateau_dir),
        "--plateau-zone",
        str(ZONE_BY_CITY[city]),
        "--triangle-cache-npz",
        str(cache),
        "--out-csv",
        str(out_csv),
        "--batch-size",
        str(int(args.batch_size)),
    ]
    if int(args.max_epochs) > 0:
        cmd.extend(["--max-epochs", str(int(args.max_epochs))])
    if int(args.start_epoch) > 0:
        cmd.extend(["--start-epoch", str(int(args.start_epoch))])
    cmd.extend(["--geoid-correction", str(args.geoid_correction)])
    _run(cmd)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_pf_nlos_smoke.py"),
        "--run",
        args.run,
        "--data-root",
        str(_ppc_root(args.data_root)),
        "--max-epochs",
        str(int(args.max_epochs)),
        "--n-particles",
        str(int(args.n_particles)),
    ]
    mask_csv = _mask_csv(args.run)
    if mask_csv.is_file():
        cmd.extend(["--mask-csv", str(mask_csv)])
    if args.skip_ab:
        cmd.append("--skip-ab")
    _run(cmd)
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", default="tokyo/run1")
    common.add_argument("--data-root", type=Path, default=None)
    common.add_argument("--plateau-root", type=Path, default=DEFAULT_PLATEAU_ROOT)
    common.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", parents=[common], help="Verify PPC data, BVH, plateau, and mask paths")
    check.set_defaults(func=cmd_check)

    fetch = sub.add_parser("fetch", parents=[common], help="Download trajectory-aligned PLATEAU subset to SSD")
    fetch.add_argument("--mesh-radius", type=int, default=1)
    fetch.add_argument("--include-bridges", action="store_true", default=True)
    fetch.add_argument("--max-rows", type=int, default=0, help="0=all reference rows")
    fetch.set_defaults(func=cmd_fetch)

    mask = sub.add_parser("mask", parents=[common], help="Build per-epoch BVH NLOS CSV for one run")
    mask.add_argument("--max-epochs", type=int, default=0, help="0=full run")
    mask.add_argument("--start-epoch", type=int, default=0)
    mask.add_argument("--batch-size", type=int, default=256)
    mask.add_argument(
        "--geoid-correction",
        default="none",
        help="egm96 needs pyproj grid files; use none when grids are unavailable",
    )
    mask.set_defaults(func=cmd_mask)

    smoke = sub.add_parser("smoke", parents=[common], help="Run baseline vs soft-k3 PPC smoke")
    smoke.add_argument("--max-epochs", type=int, default=120)
    smoke.add_argument("--n-particles", type=int, default=2000)
    smoke.add_argument("--skip-ab", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
