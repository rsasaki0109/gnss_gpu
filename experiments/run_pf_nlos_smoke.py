#!/usr/bin/env python3
"""Run PPC PF-domain NLOS smoke A/B once datasets/PPC-Dataset-data exists.

Baseline vs soft-k3 mask on a single run (default ``tokyo/run1``). Seeds the
demo mask when the plateau_nlos_phase33 CSV is missing.

Profiles:
  minimal — ``rbpf+dd`` only (wiring check)
  gate    — ``rbpf+dd+gate`` (production PF stack without hybrid/rtkdiag assets)
  signal  — ``rbpf+dd+gate+hybrid`` at ``--start-epoch 1000`` with reference-oracle
            hybrid positions (measures mask A/B when official libgnss pool is absent)
  full    — ``rbpf+dd+gate+hybrid+rtkdiag_pf`` (libgnss hybrid + diag CSV pool)

Prerequisites:
  1. Install PPC data:
       PYTHONPATH=python python experiments/download_ppc_dataset.py --zip <zip>
  2. (Optional) Replace demo mask with a real BVH mask from build_per_epoch_nlos_csv.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "PPC-Dataset-data"
DEFAULT_MASK_DIR = PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_phase33"
ORACLE_HYBRID_DIR = PROJECT_ROOT / "experiments" / "results" / "pf_nlos_oracle_hybrid"
LIBGNSS_HYBRID_DIR = PROJECT_ROOT / "experiments" / "results" / "libgnss_rtk_pos_v5"
LIBGNSS_DIAG_DIR = PROJECT_ROOT / "experiments" / "results" / "libgnss_rtk_pos_v5_diag"
SMOKE_POS_DIR = PROJECT_ROOT / "experiments" / "results" / "pf_nlos_smoke_pos"
REQUIRED_RUN_FILES = ("rover.obs", "base.obs", "base.nav", "reference.csv", "imu.csv")

PROFILE_METHODS = {
    "minimal": "rbpf+dd",
    "gate": "rbpf+dd+gate",
    "signal": "rbpf+dd+gate+hybrid",
    "full": "rbpf+dd+gate+hybrid+rtkdiag_pf",
}

PROFILE_DEFAULT_START_EPOCH = {
    "signal": 1000,
    "full": 1000,
}

METHOD_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "rbpf+dd": ("rbpf+dd", "RBPF-velKF+DD"),
    "rbpf+dd+gate": ("rbpf+dd+gate", "RBPF-velKF+DD+gate"),
    "rbpf+dd+gate+hybrid": ("rbpf+dd+gate+hybrid", "RBPF-velKF+DD+gate+hybrid"),
    "rbpf+dd+gate+hybrid+rtkdiag_pf": (
        "rbpf+dd+gate+hybrid+rtkdiag_pf",
        "RBPF-velKF+DD+gate+hybrid+rtkdiag_pf",
    ),
}

METRIC_COLUMNS = (
    "method",
    "honest_ppc_pct",
    "honest_pass_m",
    "honest_total_m",
    "segment_ppc_pct",
    "segment_pass_m",
    "segment_total_m",
    "segment_epoch_pass_pct",
    "coverage_pct",
    "rbpf_kf_gate_active",
    "rbpf_kf_applied",
    "dd_epochs_applied",
    "hybrid_applied",
    "rtkdiag_pf_pu_applied",
)


def _run_dir_ok(data_root: Path, run: str) -> bool:
    run_dir = data_root / run
    return all((run_dir / name).is_file() for name in REQUIRED_RUN_FILES)


def _mask_path(city: str, run_name: str) -> Path:
    return DEFAULT_MASK_DIR / f"{city}_{run_name}_per_epoch_nlos.csv"


def _ensure_oracle_hybrid(data_root: Path, run: str) -> Path:
    """Materialize ``{city}_{run}_full.pos`` from PPC reference.csv for smoke."""
    city, run_name = run.split("/", 1)
    out_dir = ORACLE_HYBRID_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city}_{run_name}_full.pos"
    ref_path = data_root / run / "reference.csv"
    if out_path.is_file() and out_path.stat().st_mtime >= ref_path.stat().st_mtime:
        return out_dir
    lines = ["% reference oracle hybrid for PF NLOS smoke\n"]
    with ref_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            week = int(float(str(row["GPS Week"]).strip()))
            tow = float(str(row["GPS TOW (s)"]).strip())
            x = float(str(row["ECEF X (m)"]).strip())
            y = float(str(row["ECEF Y (m)"]).strip())
            z = float(str(row["ECEF Z (m)"]).strip())
            lines.append(f"{week} {tow:.3f} {x:.6f} {y:.6f} {z:.6f} 0 0 0 4\n")
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[smoke] oracle hybrid: {out_path} ({len(lines) - 1} rows)", flush=True)
    return out_dir


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


def _method_labels(methods: str) -> set[str]:
    labels: set[str] = set()
    for token in methods.split(","):
        token = token.strip()
        if not token:
            continue
        labels.update(METHOD_LABEL_ALIASES.get(token, (token,)))
    return labels


def _read_run_row(runs_csv: Path, methods: str) -> dict[str, float | str | None]:
    if not runs_csv.is_file():
        return {}
    accepted = _method_labels(methods)
    with runs_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[0]
    for candidate in rows:
        label = str(candidate.get("method", "")).strip()
        if accepted and label in accepted:
            row = candidate
            break
    out: dict[str, float | str | None] = {}
    for key in METRIC_COLUMNS:
        raw = row.get(key)
        if raw is None or raw == "":
            out[key] = None
            continue
        if key == "method":
            out[key] = str(raw)
            continue
        try:
            out[key] = float(raw)
        except ValueError:
            out[key] = None
    return out


def _read_official_pct(runs_csv: Path, methods: str) -> float | None:
    row = _read_run_row(runs_csv, methods)
    value = row.get("honest_ppc_pct")
    return float(value) if value is not None else None


def _resolve_methods(args: argparse.Namespace) -> str:
    if args.methods:
        return str(args.methods).strip()
    return PROFILE_METHODS[str(args.profile)]


def _libgnss_hybrid_pos(city: str, run_name: str) -> Path:
    return LIBGNSS_HYBRID_DIR / f"{city}_{run_name}_full.pos"


def _libgnss_diag_csv(city: str, run_name: str) -> Path:
    return LIBGNSS_DIAG_DIR / f"{city}_{run_name}_full.csv"


def _resolve_hybrid_pos_dir(
    data_root: Path,
    run: str,
    explicit: Path | None,
    profile: str,
) -> Path | None:
    city, run_name = run.split("/", 1)
    if explicit is not None:
        return explicit.resolve()
    if _libgnss_hybrid_pos(city, run_name).is_file():
        return LIBGNSS_HYBRID_DIR.resolve()
    if profile == "signal":
        return _ensure_oracle_hybrid(data_root, run)
    return None


def _profile_ppc_extra_args(profile: str) -> list[str]:
    if profile != "full":
        return []
    return [
        "--rtkdiag-candidate-pos-dirs",
        str(LIBGNSS_HYBRID_DIR),
        "--rtkdiag-candidate-diag-dirs",
        str(LIBGNSS_DIAG_DIR),
        "--rtkdiag-candidate-labels",
        "libgnss_full",
        "--rtkdiag-candidate-select-mode",
        "residual",
        "--rtkdiag-candidate-emit-mode",
        "candidate",
        "--rtkdiag-candidate-fallback-mode",
        "hybrid",
        "--rtkdiag-candidate-residual-rms-max",
        "50.0",
        "--rtkdiag-candidate-ratio-min",
        "1.0",
        "--write-internal-diagnostics",
        "--pos-dir",
        str(SMOKE_POS_DIR),
    ]


def _ppc_extra_args(args: argparse.Namespace) -> list[str]:
    merged = _profile_ppc_extra_args(str(args.profile))
    if args.ppc_extra_args:
        merged.extend(shlex.split(str(args.ppc_extra_args), posix=False))
    return merged


def _validate_profile_args(
    args: argparse.Namespace,
    methods: str,
    run: str,
    hybrid_pos_dir: Path | None,
) -> None:
    city, run_name = run.split("/", 1)
    if "hybrid" in methods and hybrid_pos_dir is None:
        raise SystemExit(
            "--hybrid-pos-dir is required for hybrid/rtkdiag profiles "
            "(or ensure libgnss .pos exists under experiments/results/libgnss_rtk_pos_v5)"
        )
    if "rtkdiag_pf" in methods:
        diag_csv = _libgnss_diag_csv(city, run_name)
        if not diag_csv.is_file():
            raise SystemExit(
                f"missing RTK diagnostics CSV: {diag_csv}\n"
                "Generate with:\n"
                f"  python experiments/run_libgnss_rtk_wsl.py --run {run} "
                "--data-root <ppc-root> --with-diagnostics"
            )
        hybrid_pos = _libgnss_hybrid_pos(city, run_name)
        if hybrid_pos_dir is not None and not hybrid_pos.is_file():
            raise SystemExit(f"missing libgnss hybrid .pos: {hybrid_pos}")


def _run_ppc(
    *,
    run: str,
    data_root: Path,
    results_prefix: str,
    methods: str,
    pf_nlos_preset: str | None,
    pf_nlos_mask_path: str | None,
    pf_nlos_k_weak: float | None = None,
    pf_nlos_k_strong: float | None = None,
    max_epochs: int,
    n_particles: int,
    start_epoch: int,
    hybrid_pos_dir: Path | None,
    ppc_extra_args: list[str],
) -> Path:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "exp_ppc_ctrbpf_fgo.py"),
        "--data-root",
        str(data_root),
        "--runs",
        run,
        "--methods",
        methods,
        "--n-particles",
        str(int(n_particles)),
        "--max-epochs",
        str(int(max_epochs)),
        "--start-epoch",
        str(int(start_epoch)),
        "--results-prefix",
        results_prefix,
    ]
    if hybrid_pos_dir is not None:
        cmd.extend(["--hybrid-pos-dir", str(hybrid_pos_dir)])
    if pf_nlos_mask_path:
        cmd.extend(["--pf-nlos-mask-path", pf_nlos_mask_path])
        k_weak = pf_nlos_k_weak
        k_strong = pf_nlos_k_strong
        if k_weak is not None:
            cmd.extend(["--pf-nlos-k-weak", str(float(k_weak))])
        if k_strong is not None:
            cmd.extend(["--pf-nlos-k-strong", str(float(k_strong))])
    elif pf_nlos_preset:
        cmd.extend(["--pf-nlos-preset", pf_nlos_preset])
    if "hybrid" in methods:
        cmd.extend(["--hybrid-sigma-m", "1.0"])
    cmd.extend(ppc_extra_args)
    print("[smoke] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=_child_env())
    return PROJECT_ROOT / "experiments" / "results" / f"{results_prefix}_runs.csv"


def _smoke_prefix(city: str, run_name: str, profile: str, variant: str) -> str:
    return f"ppc_pf_nlos_smoke_{city}_{run_name}_{profile}_{variant}"


def _metric_delta(
    baseline: dict[str, float | str | None],
    variant: dict[str, float | str | None],
    key: str,
) -> float | None:
    b = baseline.get(key)
    v = variant.get(key)
    if isinstance(b, (int, float)) and isinstance(v, (int, float)):
        return float(v) - float(b)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_METHODS),
        default="signal",
        help="signal=oracle-hybrid mask A/B (default); gate=rbpf+dd+gate only",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Override --profile with a comma-separated exp_ppc_ctrbpf_fgo --methods string",
    )
    parser.add_argument(
        "--hybrid-pos-dir",
        type=Path,
        default=None,
        help="Required for --profile full (or any method containing +hybrid)",
    )
    parser.add_argument(
        "--ppc-extra-args",
        default="",
        help="Extra flags forwarded to exp_ppc_ctrbpf_fgo.py (quoted shell string)",
    )
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
    methods = _resolve_methods(args)
    ppc_extra_args = _ppc_extra_args(args)
    start_epoch = int(args.start_epoch)
    if start_epoch == 0 and args.profile in PROFILE_DEFAULT_START_EPOCH:
        start_epoch = int(PROFILE_DEFAULT_START_EPOCH[args.profile])
    hybrid_pos_dir = _resolve_hybrid_pos_dir(
        data_root,
        run,
        args.hybrid_pos_dir,
        str(args.profile),
    )
    _validate_profile_args(args, methods, run, hybrid_pos_dir)

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
        "start_epoch": start_epoch,
        "profile": str(args.profile),
        "methods": methods,
        "hybrid_pos_dir": str(hybrid_pos_dir) if hybrid_pos_dir is not None else None,
        "libgnss_diag_csv": str(_libgnss_diag_csv(city, run_name))
        if "rtkdiag_pf" in methods
        else None,
    }

    common_kw = dict(
        run=run,
        data_root=data_root,
        methods=methods,
        max_epochs=int(args.max_epochs),
        n_particles=int(args.n_particles),
        start_epoch=start_epoch,
        hybrid_pos_dir=hybrid_pos_dir,
        ppc_extra_args=ppc_extra_args,
    )

    profile = str(args.profile)

    if not args.skip_ab:
        baseline_csv = _run_ppc(
            results_prefix=_smoke_prefix(city, run_name, profile, "baseline"),
            pf_nlos_preset=None,
            pf_nlos_mask_path=None,
            **common_kw,
        )
        baseline_metrics = _read_run_row(baseline_csv, methods)
        summary["baseline_pct"] = baseline_metrics.get("honest_ppc_pct")
        summary["baseline_segment_pct"] = baseline_metrics.get("segment_ppc_pct")
        summary["baseline_metrics"] = baseline_metrics
        summary["baseline_runs_csv"] = str(baseline_csv)

    mask_soft_runs_csv = _run_ppc(
        results_prefix=_smoke_prefix(city, run_name, profile, "masksoft"),
        pf_nlos_preset=None,
        pf_nlos_mask_path=str(mask_csv),
        pf_nlos_k_weak=3.0,
        pf_nlos_k_strong=3.0,
        **common_kw,
    )
    mask_metrics = _read_run_row(mask_soft_runs_csv, methods)
    summary["mask_soft_pct"] = mask_metrics.get("honest_ppc_pct")
    summary["mask_soft_segment_pct"] = mask_metrics.get("segment_ppc_pct")
    summary["mask_soft_metrics"] = mask_metrics
    summary["mask_soft_runs_csv"] = str(mask_soft_runs_csv)

    baseline_metrics_obj = summary.get("baseline_metrics")
    if isinstance(baseline_metrics_obj, dict):
        summary["delta_pp"] = _metric_delta(baseline_metrics_obj, mask_metrics, "honest_ppc_pct")
        summary["delta_segment_pp"] = _metric_delta(
            baseline_metrics_obj, mask_metrics, "segment_ppc_pct"
        )
        summary["delta_segment_pass_m"] = _metric_delta(
            baseline_metrics_obj, mask_metrics, "segment_pass_m"
        )

    out_json = (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / f"ppc_pf_nlos_smoke_{city}_{run_name}_{profile}_summary.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if str(args.profile) == "full":
        analyze = PROJECT_ROOT / "experiments" / "analyze_pf_nlos_smoke_epochs.py"
        if analyze.is_file():
            baseline_prefix = _smoke_prefix(city, run_name, profile, "baseline")
            mask_prefix = _smoke_prefix(city, run_name, profile, "masksoft")
            analyze_cmd = [
                sys.executable,
                str(analyze),
                "--baseline-prefix",
                baseline_prefix,
                "--mask-prefix",
                mask_prefix,
            ]
            print("[smoke] " + " ".join(analyze_cmd), flush=True)
            subprocess.run(analyze_cmd, cwd=PROJECT_ROOT, check=True, env=_child_env())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
