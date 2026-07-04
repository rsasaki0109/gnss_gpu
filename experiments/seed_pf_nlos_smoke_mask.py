#!/usr/bin/env python3
"""Seed a demo NLOS mask CSV in the plateau_nlos_phase33 layout for smoke tests.

This does not replace real BVH-generated masks for PPC runs. It copies the
deterministic PLATEAU demo mask into the path expected by ``--pf-nlos-preset
soft-k3`` so local smoke wiring can be checked once PPC data is available.

Run from repo root:

    PYTHONPATH=python:. python experiments/seed_pf_nlos_smoke_mask.py
    PYTHONPATH=python:. python experiments/seed_pf_nlos_smoke_mask.py --city tokyo --run run1
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

from gnss_gpu.nlos_presets import PPC_NLOS_MASK_PATH_TEMPLATE  # noqa: E402


def _load_exporter():
    path = PROJECT_ROOT / "experiments" / "export_plateau_nlos_demo_mask.py"
    spec = importlib.util.spec_from_file_location("export_plateau_nlos_demo_mask", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_smoke_mask(*, city: str, run: str, out_dir: Path | None = None) -> Path:
    exporter = _load_exporter()
    out_root = out_dir or (PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_phase33")
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / f"{city}_{run}_per_epoch_nlos.csv"
    tmp = out_root / f".{city}_{run}_per_epoch_nlos.demo.csv"
    exporter.export_mask_csv(tmp)
    shutil.copyfile(tmp, target)
    tmp.unlink(missing_ok=True)
    print(f"[seed] wrote {target}")
    print(f"[seed] preset path: {PPC_NLOS_MASK_PATH_TEMPLATE.format(city=city, run=run)}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="tokyo")
    parser.add_argument("--run", default="run1")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    seed_smoke_mask(city=str(args.city), run=str(args.run), out_dir=args.out_dir)


if __name__ == "__main__":
    main()
