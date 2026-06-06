#!/usr/bin/env python3
"""Summarize PLATEAU NLOS mask replay outputs across SPP, PF, and FGO.

Run from the repo root after generating the individual replay summaries:

    PYTHONPATH=python:. python3 experiments/summarize_plateau_nlos_replays.py
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

DEFAULT_MASK_SUMMARY_JSON = RESULTS_DIR / "plateau_nlos_demo_mask_summary.json"
DEFAULT_SPP_SUMMARY_JSON = RESULTS_DIR / "plateau_nlos_demo_spp_replay_summary.json"
DEFAULT_PF_SUMMARY_JSON = RESULTS_DIR / "plateau_nlos_demo_pf_replay_summary.json"
DEFAULT_FGO_SUMMARY_JSON = RESULTS_DIR / "plateau_nlos_demo_fgo_replay_summary.json"
DEFAULT_OUT_JSON = RESULTS_DIR / "plateau_nlos_demo_suite_summary.json"
DEFAULT_OUT_MD = RESULTS_DIR / "plateau_nlos_demo_suite_summary.md"
DEFAULT_OUT_CSV = RESULTS_DIR / "plateau_nlos_demo_suite_summary.csv"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _metric(summary: dict[str, object], key: str) -> dict[str, float]:
    return dict(summary[key])  # type: ignore[arg-type]


def _row(
    estimator: str,
    summary: dict[str, object],
    *,
    baseline_key: str,
    mask_key: str,
    robust_key: str | None = None,
    wins_key: str = "mask_soft_wins",
) -> dict[str, object]:
    baseline = _metric(summary, baseline_key)
    mask = _metric(summary, mask_key)
    robust = _metric(summary, robust_key) if robust_key else None
    n_epochs = int(summary["n_solved_epochs"])
    baseline_rms = float(baseline["rms_m"])
    mask_rms = float(mask["rms_m"])
    gain_pct = float(summary.get("rms_gain_vs_naive_pct", 100.0 * (1.0 - mask_rms / baseline_rms)))
    return {
        "estimator": estimator,
        "baseline": baseline_key,
        "baseline_p50_m": _round(float(baseline["p50_m"])),
        "baseline_rms_m": _round(baseline_rms),
        "robust_rms_m": _round(float(robust["rms_m"])) if robust is not None else "",
        "mask_soft": mask_key,
        "mask_soft_p50_m": _round(float(mask["p50_m"])),
        "mask_soft_rms_m": _round(mask_rms),
        "rms_gain_pct": _round(gain_pct, 1),
        "mask_soft_wins": int(summary[wins_key]),
        "n_solved_epochs": n_epochs,
        "wins_fraction": f"{int(summary[wins_key])}/{n_epochs}",
    }


def build_suite_summary(
    *,
    mask_summary_json: Path = DEFAULT_MASK_SUMMARY_JSON,
    spp_summary_json: Path = DEFAULT_SPP_SUMMARY_JSON,
    pf_summary_json: Path = DEFAULT_PF_SUMMARY_JSON,
    fgo_summary_json: Path = DEFAULT_FGO_SUMMARY_JSON,
) -> dict[str, object]:
    mask_summary = _read_json(mask_summary_json)
    spp_summary = _read_json(spp_summary_json)
    pf_summary = _read_json(pf_summary_json)
    fgo_summary = _read_json(fgo_summary_json)

    rows = [
        _row(
            "SPP",
            spp_summary,
            baseline_key="naive",
            robust_key="robust",
            mask_key="mask_soft",
        ),
        _row(
            "PF",
            pf_summary,
            baseline_key="naive_pf",
            mask_key="mask_soft_pf",
        ),
        _row(
            "FGO",
            fgo_summary,
            baseline_key="naive_fgo",
            robust_key="robust_fgo",
            mask_key="mask_soft_fgo",
        ),
    ]
    best = min(rows, key=lambda row: float(row["mask_soft_rms_m"]))
    worst_gain = min(float(row["rms_gain_pct"]) for row in rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mask": {
            "rows": int(mask_summary["rows"]),
            "epochs": int(mask_summary["epochs"]),
            "satellites": int(mask_summary["satellites"]),
            "nlos": int(mask_summary["nlos"]),
            "nlos_frac": float(mask_summary["nlos_frac"]),
            "ray_source": str(mask_summary.get("ray_source", "")),
        },
        "rows": rows,
        "best_mask_soft_estimator": str(best["estimator"]),
        "best_mask_soft_rms_m": float(best["mask_soft_rms_m"]),
        "min_rms_gain_pct": float(worst_gain),
        "source_summaries": {
            "mask": _display_path(mask_summary_json),
            "spp": _display_path(spp_summary_json),
            "pf": _display_path(pf_summary_json),
            "fgo": _display_path(fgo_summary_json),
        },
    }
    return summary


def write_suite_summary(
    summary: dict[str, object],
    *,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_csv: Path = DEFAULT_OUT_CSV,
) -> dict[str, str]:
    out_json = Path(out_json)
    out_md = Path(out_md)
    out_csv = Path(out_csv)
    for path in (out_json, out_md, out_csv):
        path.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_markdown_summary(summary), encoding="utf-8")
    _write_csv(summary, out_csv)
    return {
        "summary_json": _display_path(out_json),
        "summary_markdown": _display_path(out_md),
        "summary_csv": _display_path(out_csv),
    }


def _write_csv(summary: dict[str, object], out_csv: Path) -> None:
    rows = list(summary["rows"])  # type: ignore[arg-type]
    columns = [
        "estimator",
        "baseline_rms_m",
        "mask_soft_rms_m",
        "rms_gain_pct",
        "baseline_p50_m",
        "mask_soft_p50_m",
        "wins_fraction",
        "robust_rms_m",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def _markdown_summary(summary: dict[str, object]) -> str:
    mask = summary["mask"]  # type: ignore[index]
    rows = list(summary["rows"])  # type: ignore[arg-type]
    lines = [
        "# PLATEAU NLOS Demo Suite",
        "",
        (
            f"Mask: {mask['epochs']} epochs x {mask['satellites']} satellites, "
            f"NLOS {mask['nlos']}/{mask['rows']} ({float(mask['nlos_frac']) * 100.0:.1f}%), "
            f"ray source: {mask['ray_source']}."
        ),
        "",
        "| Estimator | Baseline RMS (m) | Mask-soft RMS (m) | RMS gain | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {estimator} | {baseline_rms_m:.2f} | {mask_soft_rms_m:.2f} | "
            "{rms_gain_pct:.1f}% | {wins_fraction} |".format(**row)
        )
    lines.extend(
        [
            "",
            (
                f"Best mask-soft RMS: {summary['best_mask_soft_estimator']} "
                f"at {float(summary['best_mask_soft_rms_m']):.2f} m."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-summary-json", type=Path, default=DEFAULT_MASK_SUMMARY_JSON)
    parser.add_argument("--spp-summary-json", type=Path, default=DEFAULT_SPP_SUMMARY_JSON)
    parser.add_argument("--pf-summary-json", type=Path, default=DEFAULT_PF_SUMMARY_JSON)
    parser.add_argument("--fgo-summary-json", type=Path, default=DEFAULT_FGO_SUMMARY_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    summary = build_suite_summary(
        mask_summary_json=args.mask_summary_json,
        spp_summary_json=args.spp_summary_json,
        pf_summary_json=args.pf_summary_json,
        fgo_summary_json=args.fgo_summary_json,
    )
    outputs = write_suite_summary(
        summary,
        out_json=args.out_json,
        out_md=args.out_md,
        out_csv=args.out_csv,
    )
    summary["outputs"] = outputs
    print("PLATEAU NLOS replay suite")
    print("=" * 70)
    for row in summary["rows"]:  # type: ignore[index]
        print(
            f"{row['estimator']:<4} raw RMS {row['baseline_rms_m']:>5.2f} m -> "
            f"mask-soft {row['mask_soft_rms_m']:>5.2f} m "
            f"({row['rms_gain_pct']:>4.1f}% gain, wins {row['wins_fraction']})"
        )
    print(f"summary: {outputs['summary_json']}")
    print(f"markdown: {outputs['summary_markdown']}")
    print(f"csv: {outputs['summary_csv']}")
    return summary


if __name__ == "__main__":
    main()
