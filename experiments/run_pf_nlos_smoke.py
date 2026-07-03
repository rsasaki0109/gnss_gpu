#!/usr/bin/env python3
"""Run PPC PF-domain NLOS smoke A/B once datasets/PPC-Dataset-data exists.

Baseline vs soft-k3 mask on a single run (default ``tokyo/run1``). Seeds the
demo mask when the plateau_nlos_phase33 CSV is missing.

Prerequisites:
  1. Install PPC data:
       PYTHONPATH=python python experiments/download_ppc_dataset.py --zip <zip>
  2. (Optional) Replace demo mask with a real BVH mask from build_per_epoch_nlos_csv.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "PPC-Dataset-data"
DEFAULT_MASK_DIR = PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_phase33"
REQUIRED_RUN_FILES = ("rover.obs", "base.obs", "base.nav", "reference.csv", "imu.csv")


def _run_dir_ok(data_root: Path, run: str) -> bool:
    run_dir = data_root / run
    return all((run_dir / name).is_file() for name in REQUIRED_RUN_FILES)


def _mask_path(city: str, run_name: str) -> Path:
    return DEFAULT_MASK_DIR / f"{city}_{run_name}_per_epoch_nlos.csv"


def _seed_demo_mask() -> Path:
    cmd = [sys.executable, str(PROJECT_ROOT / "experiments" / "seed_pf_nlos_smoke_mask.py")]
    print("[smoke] seeding demo mask", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=_child_env())
    return _mask_path("tokyo", "run1")


def _child_env() -> dict[str, str]:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    pythonpath = str(PROJECT_ROOT / "python")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{__import__('os').pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _run_ppc(
    *,
    run: str,
    data_root: Path,
    results_prefix: str,
    pf_nlos_preset: str | None,
    pf_nlos_mask_path: str | None,
    max_epochs: int,
    n_particles: int,
) -> Path:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "exp_ppc_ctrbpf_fgo.py"),
        "--data-root",
        str(data_root),
        "--runs",
        run,
        "--methods",
        "rbpf+dd",
        "--n-particles",
        str(int(n_particles)),
        "--max-epochs",
        str(int(max_epochs)),
        "--results-prefix",
        results_prefix,
    ]
    if pf_nlos_preset:
        cmd.extend(["--pf-nlos-preset", pf_nlos_preset])
    if pf_nlos_mask_path:
        cmd.extend(["--pf-nlos-mask-path", pf_nlos_mask_path])
    print("[smoke] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=_child_env())
    return PROJECT_ROOT / "experiments" / "results" / f"{results_prefix}_runs.csv"


def _read_official_pct(runs_csv: Path, method: str | None = "rbpf+dd") -> float | None:
    if not runs_csv.is_file():
        return None
    method_aliases = {
        "rbpf+dd": ("rbpf+dd", "RBPF-velKF+DD"),
    }
    accepted = set(method_aliases.get(method or "", ()))
    if method:
        accepted.add(method)
    with runs_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        label = str(row.get("method", "")).strip()
        if accepted and label not in accepted:
            continue
        value = row.get("honest_ppc_pct")
        if value is None or value == "":
            continue
        return float(value)
    if rows and rows[0].get("honest_ppc_pct") not in (None, ""):
        return float(rows[0]["honest_ppc_pct"])
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument(
        "--mask-csv",
        type=Path,
        default=None,
        help="Override mask CSV (default: plateau_nlos_phase33/{city}_{run}_per_epoch_nlos.csv)",
    )
    parser.add_argument(
        "--skip-ab",
        action="store_true",
        help="Only run the mask-soft variant (no baseline comparison)",
    )
    args = parser.parse_args(argv)

    run = str(args.run).strip().strip("/")
    city, run_name = run.split("/", 1)
    data_root = args.data_root.resolve()

    if not _run_dir_ok(data_root, run):
        print(f"[error] missing PPC run data under {data_root / run}", file=sys.stderr)
        print(
            "Install with:\n"
            "  PYTHONPATH=python python experiments/download_ppc_dataset.py\n"
            "then download the official zip in a browser and rerun with --zip.",
            file=sys.stderr,
        )
        return 2

    mask_csv = args.mask_csv or _mask_path(city, run_name)
    if not mask_csv.is_file():
        if run == "tokyo/run1":
            mask_csv = _seed_demo_mask()
        else:
            print(f"[error] missing mask CSV: {mask_csv}", file=sys.stderr)
            return 2

    summary: dict[str, object] = {
        "run": run,
        "data_root": str(data_root),
        "mask_csv": str(mask_csv),
        "max_epochs": int(args.max_epochs),
    }

    if not args.skip_ab:
        baseline_csv = _run_ppc(
            run=run,
            data_root=data_root,
            results_prefix=f"ppc_pf_nlos_smoke_{city}_{run_name}_baseline",
            pf_nlos_preset=None,
            pf_nlos_mask_path=None,
            max_epochs=int(args.max_epochs),
            n_particles=int(args.n_particles),
        )
        summary["baseline_pct"] = _read_official_pct(baseline_csv)
        summary["baseline_runs_csv"] = str(baseline_csv)

    mask_soft_runs_csv = _run_ppc(
        run=run,
        data_root=data_root,
        results_prefix=f"ppc_pf_nlos_smoke_{city}_{run_name}_masksoft",
        pf_nlos_preset="soft-k3",
        pf_nlos_mask_path=str(mask_csv),
        max_epochs=int(args.max_epochs),
        n_particles=int(args.n_particles),
    )
    summary["mask_soft_pct"] = _read_official_pct(mask_soft_runs_csv)
    summary["mask_soft_runs_csv"] = str(mask_soft_runs_csv)

    if summary.get("baseline_pct") is not None and summary.get("mask_soft_pct") is not None:
        summary["delta_pp"] = float(summary["mask_soft_pct"]) - float(summary["baseline_pct"])

    out_json = (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / f"ppc_pf_nlos_smoke_{city}_{run_name}_summary.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
