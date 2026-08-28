#!/usr/bin/env python3
"""Run the full PLATEAU NLOS mask replay suite.

This exports the PLATEAU ray mask, replays it through SPP/PF/FGO consumers, and
then writes a combined JSON/Markdown/CSV summary.

Run from the repo root:

    PYTHONPATH=python:. python3 experiments/run_plateau_nlos_demo_suite.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

DEFAULT_MASK_CSV = RESULTS_DIR / "plateau_nlos_demo_mask.csv"
DEFAULT_MASK_SUMMARY_JSON = RESULTS_DIR / "plateau_nlos_demo_mask_summary.json"
DEFAULT_SPP_SUMMARY_JSON = RESULTS_DIR / "plateau_nlos_demo_spp_replay_summary.json"
DEFAULT_PF_SUMMARY_JSON = RESULTS_DIR / "plateau_nlos_demo_pf_replay_summary.json"
DEFAULT_FGO_SUMMARY_JSON = RESULTS_DIR / "plateau_nlos_demo_fgo_replay_summary.json"
DEFAULT_SUITE_JSON = RESULTS_DIR / "plateau_nlos_demo_suite_summary.json"
DEFAULT_SUITE_MD = RESULTS_DIR / "plateau_nlos_demo_suite_summary.md"
DEFAULT_SUITE_CSV = RESULTS_DIR / "plateau_nlos_demo_suite_summary.csv"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_suite(
    *,
    mask_csv: Path = DEFAULT_MASK_CSV,
    mask_summary_json: Path = DEFAULT_MASK_SUMMARY_JSON,
    spp_summary_json: Path = DEFAULT_SPP_SUMMARY_JSON,
    pf_summary_json: Path = DEFAULT_PF_SUMMARY_JSON,
    fgo_summary_json: Path = DEFAULT_FGO_SUMMARY_JSON,
    suite_json: Path = DEFAULT_SUITE_JSON,
    suite_md: Path = DEFAULT_SUITE_MD,
    suite_csv: Path = DEFAULT_SUITE_CSV,
    pf_particles: int = 3000,
    gml_path: Path | None = None,
) -> dict[str, object]:
    exporter = _load_module(
        "export_plateau_nlos_demo_mask",
        PROJECT_ROOT / "experiments" / "export_plateau_nlos_demo_mask.py",
    )
    spp = _load_module(
        "replay_plateau_nlos_demo_spp",
        PROJECT_ROOT / "experiments" / "replay_plateau_nlos_demo_spp.py",
    )
    pf = _load_module(
        "replay_plateau_nlos_demo_pf",
        PROJECT_ROOT / "experiments" / "replay_plateau_nlos_demo_pf.py",
    )
    fgo = _load_module(
        "replay_plateau_nlos_demo_fgo",
        PROJECT_ROOT / "experiments" / "replay_plateau_nlos_demo_fgo.py",
    )
    summarizer = _load_module(
        "summarize_plateau_nlos_replays",
        PROJECT_ROOT / "experiments" / "summarize_plateau_nlos_replays.py",
    )

    mask_summary = exporter.export_mask_csv(
        mask_csv,
        summary_json=mask_summary_json,
        gml_path=gml_path,
    )
    spp_summary = spp.replay_spp(
        mask_csv,
        summary_json=spp_summary_json,
        gml_path=gml_path,
    )
    pf_summary = pf.replay_pf(
        mask_csv,
        summary_json=pf_summary_json,
        gml_path=gml_path,
        n_particles=pf_particles,
    )
    fgo_summary = fgo.replay_fgo(
        mask_csv,
        summary_json=fgo_summary_json,
        gml_path=gml_path,
    )
    suite_summary = summarizer.build_suite_summary(
        mask_summary_json=mask_summary_json,
        spp_summary_json=spp_summary_json,
        pf_summary_json=pf_summary_json,
        fgo_summary_json=fgo_summary_json,
    )
    outputs = summarizer.write_suite_summary(
        suite_summary,
        out_json=suite_json,
        out_md=suite_md,
        out_csv=suite_csv,
    )
    suite_summary["outputs"] = outputs
    return {
        "mask": mask_summary,
        "spp": spp_summary,
        "pf": pf_summary,
        "fgo": fgo_summary,
        "suite": suite_summary,
    }


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-csv", type=Path, default=DEFAULT_MASK_CSV)
    parser.add_argument("--mask-summary-json", type=Path, default=DEFAULT_MASK_SUMMARY_JSON)
    parser.add_argument("--spp-summary-json", type=Path, default=DEFAULT_SPP_SUMMARY_JSON)
    parser.add_argument("--pf-summary-json", type=Path, default=DEFAULT_PF_SUMMARY_JSON)
    parser.add_argument("--fgo-summary-json", type=Path, default=DEFAULT_FGO_SUMMARY_JSON)
    parser.add_argument("--suite-json", type=Path, default=DEFAULT_SUITE_JSON)
    parser.add_argument("--suite-md", type=Path, default=DEFAULT_SUITE_MD)
    parser.add_argument("--suite-csv", type=Path, default=DEFAULT_SUITE_CSV)
    parser.add_argument("--pf-particles", type=int, default=3000)
    parser.add_argument("--gml", type=Path, default=None)
    args = parser.parse_args()
    if args.pf_particles <= 0:
        parser.error("--pf-particles must be positive")

    result = run_suite(
        mask_csv=args.mask_csv,
        mask_summary_json=args.mask_summary_json,
        spp_summary_json=args.spp_summary_json,
        pf_summary_json=args.pf_summary_json,
        fgo_summary_json=args.fgo_summary_json,
        suite_json=args.suite_json,
        suite_md=args.suite_md,
        suite_csv=args.suite_csv,
        pf_particles=args.pf_particles,
        gml_path=args.gml,
    )
    suite = result["suite"]
    print("PLATEAU NLOS demo suite complete")
    print("=" * 70)
    for row in suite["rows"]:  # type: ignore[index]
        print(
            f"{row['estimator']:<4} raw RMS {row['baseline_rms_m']:>5.2f} m -> "
            f"mask-soft {row['mask_soft_rms_m']:>5.2f} m "
            f"({row['rms_gain_pct']:>4.1f}% gain, wins {row['wins_fraction']})"
        )
    outputs = suite["outputs"]  # type: ignore[index]
    print(f"summary: {outputs['summary_json']}")
    print(f"markdown: {outputs['summary_markdown']}")
    print(f"csv: {outputs['summary_csv']}")
    return result


if __name__ == "__main__":
    main()
