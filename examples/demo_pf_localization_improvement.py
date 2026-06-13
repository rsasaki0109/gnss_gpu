#!/usr/bin/env python3
"""Zero-data demo for particle-filter localization improvement.

The script reads checked-in result artifacts and prints a compact comparison of
the OpenStreetMap particle-filter showcase against RTKLIB demo5, plus the
PLATEAU LOS/NLOS mask replay result for the particle-filter consumer.

Run from the repo root:

    PYTHONPATH=python:. python3 examples/demo_pf_localization_improvement.py

No CUDA build, downloaded UrbanNav data, or Python package import is required.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]

ODAIBA_FREEZE_JSON = Path("docs/assets/data/odaiba_pf_smoother_freeze.json")
PLATEAU_SUITE_CSV = Path("docs/assets/data/plateau_nlos_demo_suite_summary.csv")

VISUAL_ARTIFACTS = {
    "OSM GIF": Path("docs/assets/media/particles/particle_viz_odaiba.gif"),
    "OSM MP4": Path("docs/assets/media/particles/particle_viz_odaiba.mp4"),
    "LOS/NLOS GIF": Path("docs/assets/media/los-nlos/los_nlos_deckgl.gif"),
}

REBUILD_HINT = (
    "Run: PYTHONPATH=python:. python3 experiments/build_githubio_summary.py"
)


class MissingArtifact(RuntimeError):
    """Raised when a checked-in demo artifact is missing."""


def _project_path(project_root: Path, rel_path: Path) -> Path:
    return project_root / rel_path


def _require_file(project_root: Path, rel_path: Path) -> Path:
    path = _project_path(project_root, rel_path)
    if not path.is_file():
        raise MissingArtifact(f"Missing artifact: {rel_path.as_posix()}\n{REBUILD_HINT}")
    return path


def _read_json(project_root: Path, rel_path: Path) -> dict[str, Any]:
    path = _require_file(project_root, rel_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(project_root: Path, rel_path: Path) -> list[dict[str, str]]:
    path = _require_file(project_root, rel_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float:
    return float(value)


def _fmt_m(value: Any) -> str:
    return f"{_float(value):.2f}"


def _fmt_pct(value: Any) -> str:
    pct = _float(value)
    if abs(pct - round(pct)) < 0.05:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def _improvement_pct(baseline: float, improved: float) -> float:
    if baseline == 0.0:
        return 0.0
    return 100.0 * (baseline - improved) / baseline


def _load_odaiba(project_root: Path) -> dict[str, Any]:
    data = _read_json(project_root, ODAIBA_FREEZE_JSON)
    return {
        "dataset": data["dataset"],
        "epochs": int(data["n_epochs"]),
        "source": ODAIBA_FREEZE_JSON.as_posix(),
        "baseline": {
            "method": data["baseline_method"],
            "p50_m": _float(data["baseline_p50_m"]),
            "rms_2d_m": _float(data["baseline_rms_2d_m"]),
        },
        "particle_filter": {
            "method": data["method"],
            "variant": data.get("variant", ""),
            "p50_m": _float(data["pf_p50_m"]),
            "rms_2d_m": _float(data["pf_rms_2d_m"]),
        },
        "improvement": {
            "p50_pct": _float(data["p50_improvement_pct"]),
            "rms_2d_pct": _float(data["rms_improvement_pct"]),
        },
        "notes": data.get("notes", ""),
    }


def _load_plateau_pf(project_root: Path) -> dict[str, Any]:
    rows = _read_csv_rows(project_root, PLATEAU_SUITE_CSV)
    pf_row = next((row for row in rows if row.get("estimator", "").upper() == "PF"), None)
    if pf_row is None:
        raise MissingArtifact(
            f"Missing PF row in artifact: {PLATEAU_SUITE_CSV.as_posix()}\n{REBUILD_HINT}"
        )

    baseline_p50 = _float(pf_row["baseline_p50_m"])
    mask_soft_p50 = _float(pf_row["mask_soft_p50_m"])
    return {
        "estimator": "PF",
        "source": PLATEAU_SUITE_CSV.as_posix(),
        "baseline": {
            "p50_m": baseline_p50,
            "rms_m": _float(pf_row["baseline_rms_m"]),
        },
        "mask_soft": {
            "p50_m": mask_soft_p50,
            "rms_m": _float(pf_row["mask_soft_rms_m"]),
        },
        "improvement": {
            "p50_pct": _improvement_pct(baseline_p50, mask_soft_p50),
            "rms_pct": _float(pf_row["rms_gain_pct"]),
        },
        "wins_fraction": pf_row["wins_fraction"],
    }


def _visual_artifacts(project_root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for label, rel_path in VISUAL_ARTIFACTS.items():
        artifacts[label] = {
            "path": rel_path.as_posix(),
            "exists": _project_path(project_root, rel_path).is_file(),
        }
    return artifacts


def build_summary(project_root: Path, include_plateau: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "odaiba": _load_odaiba(project_root),
        "visual_artifacts": _visual_artifacts(project_root),
    }
    if include_plateau:
        summary["plateau_los_nlos_mask"] = _load_plateau_pf(project_root)
    return summary


def print_report(summary: dict[str, Any]) -> None:
    print("Particle-filter localization improvement demo")
    print("=============================================")
    print()

    odaiba = summary["odaiba"]
    baseline = odaiba["baseline"]
    particle_filter = odaiba["particle_filter"]
    improvement = odaiba["improvement"]

    print("Real UrbanNav Odaiba freeze")
    print(f"  Dataset : {odaiba['dataset']}")
    print(f"  Epochs  : {odaiba['epochs']}")
    print()
    print("  Method                                      P50 [m]   RMS [m]")
    print("  -------------------------------------------------------------")
    print(
        f"  {baseline['method']:<42}"
        f"{_fmt_m(baseline['p50_m']):>7}   {_fmt_m(baseline['rms_2d_m']):>7}"
    )
    print(
        f"  {particle_filter['method']:<42}"
        f"{_fmt_m(particle_filter['p50_m']):>7}   "
        f"{_fmt_m(particle_filter['rms_2d_m']):>7}"
    )
    print()
    print(
        "  Improvement vs "
        f"{baseline['method']}: P50 {_fmt_pct(improvement['p50_pct'])}, "
        f"RMS {_fmt_pct(improvement['rms_2d_pct'])}."
    )
    print()

    plateau = summary.get("plateau_los_nlos_mask")
    if plateau is not None:
        print("PLATEAU LOS/NLOS mask replay")
        print(f"  Estimator : {plateau['estimator']}")
        print(
            "  Baseline  : "
            f"P50 {_fmt_m(plateau['baseline']['p50_m'])} m / "
            f"RMS {_fmt_m(plateau['baseline']['rms_m'])} m"
        )
        print(
            "  Mask-soft : "
            f"P50 {_fmt_m(plateau['mask_soft']['p50_m'])} m / "
            f"RMS {_fmt_m(plateau['mask_soft']['rms_m'])} m"
        )
        print(
            "  Improvement: "
            f"P50 {_fmt_pct(plateau['improvement']['p50_pct'])}, "
            f"RMS {_fmt_pct(plateau['improvement']['rms_pct'])}"
        )
        print(f"  Wins      : {plateau['wins_fraction']}")
        print()

    print("Visual artifacts")
    for label, artifact in summary["visual_artifacts"].items():
        suffix = "" if artifact["exists"] else " [missing]"
        print(f"  {label:<12}: {artifact['path']}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the checked-in particle-filter localization improvement demo."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument(
        "--no-plateau",
        action="store_true",
        help="Skip the PLATEAU LOS/NLOS mask replay section.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for a machine-readable copy of the summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    try:
        summary = build_summary(project_root, include_plateau=not args.no_plateau)
    except MissingArtifact as exc:
        raise SystemExit(str(exc)) from exc

    print_report(summary)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
