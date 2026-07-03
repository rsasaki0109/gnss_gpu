#!/usr/bin/env python3
"""Demo chooser for gnss_gpu examples.

Lists runnable demos, explains GPU/data requirements, and launches a script with
the correct ``PYTHONPATH``. Run from the repository root:

    PYTHONPATH=python python3 examples/run_demo.py --list
    PYTHONPATH=python python3 examples/run_demo.py urban_canyon
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GpuNeed = Literal["none", "preferred", "required"]

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = EXAMPLES_DIR.parent
BUILD_HINT = (
    "Build the CUDA/C++ extensions first - see README.md#building-the-cudac-kernels "
    "and copy the generated .so/.pyd files into python/gnss_gpu/."
)
RUN_HINT = "Run from the repository root, e.g.\n  PYTHONPATH=python python3 examples/run_demo.py --list"


@dataclass(frozen=True)
class DemoSpec:
    name: str
    script: str
    summary: str
    gpu: GpuNeed = "none"
    data_note: str = ""


DEMO_CATALOG: tuple[DemoSpec, ...] = (
    DemoSpec(
        "urban_canyon",
        "demo_urban_canyon_sim.py",
        "Urban-canyon SPP: naive WLS vs robust Cauchy down-weighting (~1 s, CPU-only).",
    ),
    DemoSpec(
        "pf_localization",
        "demo_pf_localization_improvement.py",
        "Print checked-in PF vs RTKLIB and PLATEAU mask replay summaries (CPU-only).",
    ),
    DemoSpec(
        "nlos_simulation",
        "demo_nlos_simulation.py",
        "Synthetic street-canyon NLOS measurement model and SPP comparison (CPU-only).",
    ),
    DemoSpec(
        "plateau_nlos",
        "demo_plateau_nlos_simulation.py",
        "PLATEAU CityGML sample with LOS/NLOS ray mask and SPP replay (CPU BVH fallback).",
    ),
    DemoSpec(
        "plateau_viz",
        "demo_plateau_nlos_visualization.py",
        "Write a standalone PLATEAU LOS/NLOS HTML report (CPU-only).",
    ),
    DemoSpec(
        "nlos_validation",
        "demo_nlos_validation.py",
        "Validate NLOS residual statistics against synthetic expectations (CPU-only).",
    ),
    DemoSpec(
        "nlos_diffraction",
        "demo_nlos_diffraction.py",
        "UTD diffraction excess-path demo with matplotlib figures (CPU-only).",
    ),
    DemoSpec(
        "diffraction_benchmark",
        "demo_diffraction_benchmark.py",
        "Benchmark diffraction kernel vs reference paths (CPU-only).",
    ),
    DemoSpec(
        "visualization",
        "demo_visualization.py",
        "Generate sample sky plots, trajectories, and acquisition grids (CPU-only).",
    ),
    DemoSpec(
        "signal_sim",
        "demo_signal_sim.py",
        "GPU signal simulation plus acquisition round-trip.",
        gpu="required",
    ),
    DemoSpec(
        "acquisition",
        "demo_acquisition.py",
        "GPU GPS acquisition with a Python fallback when CUDA is unavailable.",
        gpu="preferred",
    ),
    DemoSpec(
        "interference",
        "demo_interference.py",
        "GNSS interference detection and excision (GPU preferred, Python fallback).",
        gpu="preferred",
    ),
    DemoSpec(
        "rinex",
        "demo_rinex.py",
        "Parse RINEX, run WLS, and exercise a small particle filter.",
        gpu="preferred",
    ),
    DemoSpec(
        "full_pipeline",
        "demo_full_pipeline.py",
        "End-to-end urban multipath scenario with PF updates.",
        gpu="preferred",
    ),
    DemoSpec(
        "plateau_urban",
        "demo_plateau_urban.py",
        "PLATEAU 3D city model with particle-filter urban positioning.",
        gpu="preferred",
    ),
    DemoSpec(
        "real_data",
        "demo_real_data.py",
        "Real-data positioning pipeline (GPU preferred; may need downloaded inputs).",
        gpu="preferred",
        data_note="Check the script docstring and internal_docs/plan.md for dataset paths.",
    ),
    DemoSpec(
        "urbannav_residuals",
        "demo_urbannav_residuals.py",
        "Summarise UrbanNav pseudorange residual distributions.",
        gpu="required",
        data_note="Requires experiments/data/urbannav/<Odaiba|Shinjuku> (see script docstring).",
    ),
    DemoSpec(
        "urbannav_calibration",
        "demo_urbannav_calibration.py",
        "Calibrate parametric NLOS model against UrbanNav residuals.",
        gpu="required",
        data_note="Requires experiments/data/urbannav/<Odaiba|Shinjuku> (see script docstring).",
    ),
)

DEMO_BY_NAME = {spec.name: spec for spec in DEMO_CATALOG}


class DemoChooserError(RuntimeError):
    """User-facing chooser error with remediation hints."""


def _require_repo_root(project_root: Path) -> None:
    if not (project_root / "python" / "gnss_gpu").is_dir():
        raise DemoChooserError(
            f"Expected a gnss_gpu repository at {project_root}\n{RUN_HINT}"
        )
    if not (project_root / "examples").is_dir():
        raise DemoChooserError(
            f"Missing examples/ under {project_root}\n{RUN_HINT}"
        )


def _gpu_extensions_available() -> bool:
    try:
        import gnss_gpu._gnss_gpu  # noqa: F401
    except ImportError:
        return False
    return True


def _format_gpu(gpu: GpuNeed) -> str:
    return {"none": "no", "preferred": "preferred", "required": "yes"}[gpu]


def _format_demo_row(spec: DemoSpec) -> str:
    data = spec.data_note or "-"
    return (
        f"{spec.name:<22} gpu={_format_gpu(spec.gpu):<10} "
        f"{spec.script:<34} {spec.summary}"
        + (f"\n{'':22} data: {data}" if spec.data_note else "")
    )


def list_demos() -> str:
    cpu = [spec for spec in DEMO_CATALOG if spec.gpu == "none"]
    gpu = [spec for spec in DEMO_CATALOG if spec.gpu != "none"]
    lines = [
        "Start here (CPU-only, no CUDA build):",
        *(_format_demo_row(spec) for spec in cpu),
        "",
        "Kernel-backed demos (build CUDA/C++ first - see README):",
        *(_format_demo_row(spec) for spec in gpu),
        "",
        "Run: PYTHONPATH=python python3 examples/run_demo.py <name>",
        "More detail: examples/README.md",
    ]
    return "\n".join(lines)


def resolve_demo(name: str) -> DemoSpec:
    key = name.strip().lower().replace("-", "_")
    if key not in DEMO_BY_NAME:
        known = ", ".join(sorted(DEMO_BY_NAME))
        raise DemoChooserError(
            f"Unknown demo {name!r}. Known demos: {known}\n"
            "Use --list to see summaries and requirements."
        )
    return DEMO_BY_NAME[key]


def preflight_demo(spec: DemoSpec, project_root: Path, passthrough: list[str]) -> None:
    script_path = EXAMPLES_DIR / spec.script
    if not script_path.is_file():
        raise DemoChooserError(
            f"Demo script missing: examples/{spec.script}\n"
            "Your checkout may be incomplete; try git pull."
        )

    if spec.gpu == "required" and not _gpu_extensions_available():
        raise DemoChooserError(
            f"Demo {spec.name!r} requires compiled CUDA extensions.\n{BUILD_HINT}"
        )

    if spec.gpu == "preferred" and not _gpu_extensions_available():
        print(
            f"Note: CUDA extensions not found; {spec.name} may use slower Python fallbacks.",
            file=sys.stderr,
        )

    if spec.data_note and spec.name.startswith("urbannav"):
        run_name = passthrough[0] if passthrough else "Odaiba"
        data_dir = project_root / "experiments" / "data" / "urbannav" / run_name
        if not data_dir.is_dir():
            raise DemoChooserError(
                f"Demo {spec.name!r} expects UrbanNav data at "
                f"{data_dir.relative_to(project_root).as_posix()}\n"
                f"{spec.data_note}"
            )


def run_demo(
    spec: DemoSpec,
    project_root: Path,
    passthrough: list[str],
) -> int:
    preflight_demo(spec, project_root, passthrough)
    env = os.environ.copy()
    env["PYTHONPATH"] = "python:."
    cmd = [sys.executable, str(EXAMPLES_DIR / spec.script), *passthrough]
    print(f"Running: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=project_root, env=env, check=False)
    if completed.returncode != 0:
        raise DemoChooserError(
            f"Demo {spec.name!r} exited with status {completed.returncode}.\n"
            "See the traceback above. For requirements, run:\n"
            f"  PYTHONPATH=python python3 examples/run_demo.py --describe {spec.name}"
        )
    return completed.returncode


def describe_demo(name: str) -> str:
    spec = resolve_demo(name)
    lines = [
        f"name   : {spec.name}",
        f"script : examples/{spec.script}",
        f"gpu    : {_format_gpu(spec.gpu)}",
        f"about  : {spec.summary}",
    ]
    if spec.data_note:
        lines.append(f"data   : {spec.data_note}")
    if spec.gpu != "none":
        lines.append(f"build  : {BUILD_HINT}")
    lines.append(f"run    : PYTHONPATH=python python3 examples/run_demo.py {spec.name}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List and run gnss_gpu example demos with friendly requirement checks.",
        epilog=(
            "Examples:\n"
            "  PYTHONPATH=python python3 examples/run_demo.py --list\n"
            "  PYTHONPATH=python python3 examples/run_demo.py urban_canyon\n"
            "  PYTHONPATH=python python3 examples/run_demo.py urbannav_residuals Odaiba"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "demo",
        nargs="?",
        help="Demo name to run (see --list). Remaining args are forwarded to the script.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all demos grouped by CPU vs GPU requirements.",
    )
    parser.add_argument(
        "--describe",
        metavar="NAME",
        help="Print detailed requirements for one demo.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root (defaults to the parent of examples/).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    project_root = args.project_root.resolve()

    try:
        _require_repo_root(project_root)
        if args.list:
            print(list_demos())
            return 0
        if args.describe:
            print(describe_demo(args.describe))
            return 0
        if args.demo is None:
            parser.print_help()
            print("\nTip: start with --list, then run `urban_canyon` (CPU-only).")
            return 2

        spec = resolve_demo(args.demo)
        return run_demo(spec, project_root, passthrough)
    except DemoChooserError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
