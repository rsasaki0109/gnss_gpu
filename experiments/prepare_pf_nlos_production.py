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
HYBRID_POS_DIR = PROJECT_ROOT / "experiments" / "results" / "libgnss_rtk_pos_v5"
GNSS_SOLVE_WSL = (
    PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "apps" / "gnss_solve"
)

PRESET_BY_CITY = {
    "tokyo": "tokyo23",
    "nagoya": "nagoya",
}
ZONE_BY_CITY = {
    "tokyo": 9,
    "nagoya": 7,
}
# Constant orthometric→ellipsoid N (m) when EGM96 grids are unavailable.
GEOID_CONSTANT_BY_CITY = {
    "tokyo": "36.7",
    "nagoya": "43.0",
}

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
SELECTOR_V3_FEATURES = RESULTS_DIR / "selector_training_features_v3.csv"
SELECTOR_V5_FEATURES = RESULTS_DIR / "selector_training_features_v5_nlos.csv"
V5_RANKER_PREDICTIONS = RESULTS_DIR / "selector_ranker_predictions_v5_nlos.csv"
WAVE2_SELECTOR_V5_FEATURES = RESULTS_DIR / "selector_training_features_v5_nlos_w2pool.csv"
WAVE2_RANKER_PREDICTIONS = RESULTS_DIR / "selector_ranker_predictions_v5_nlos_w2pool.csv"
WAVE2_RANKER_MODEL = RESULTS_DIR / "selector_ranker_model_v5_nlos_w2pool.txt"
MANIFEST_DIR = PROJECT_ROOT / "experiments" / "results" / "rtkdiag_manifest"
WAVE2_ROOT = PROJECT_ROOT / "experiments" / "results" / "libgnss_rtk_wave2"
PHASE33_CANDIDATE_DIRS = (
    RESULTS_DIR / "libgnss_diag_phase10/fgo_v2_gap",
    RESULTS_DIR / "libgnss_diag_phase19/gici_tc_esdfix",
)

FULL_RUNS = (
    "tokyo/run1",
    "tokyo/run2",
    "tokyo/run3",
    "nagoya/run1",
    "nagoya/run2",
    "nagoya/run3",
)


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


def _hybrid_pos(run: str) -> Path:
    city, run_name = _city_run(run)
    return HYBRID_POS_DIR / f"{city}_{run_name}_full.pos"


def _parse_runs(text: str) -> tuple[str, ...]:
    if str(text).strip().lower() in {"", "all"}:
        return FULL_RUNS
    return tuple(r.strip().strip("/") for r in str(text).split(",") if r.strip())


def _run_status(args: argparse.Namespace, run: str) -> dict[str, bool]:
    ppc_root = _ppc_root(args.data_root)
    run_dir = _run_dir(ppc_root, run)
    required = ("rover.obs", "base.obs", "base.nav", "reference.csv", "imu.csv")
    return {
        "ppc_ok": all((run_dir / name).is_file() for name in required),
        "plateau_ok": _plateau_dir(args.plateau_root, run).is_dir(),
        "cache_ok": _triangle_cache(args.cache_root, run).is_file(),
        "mask_ok": _mask_csv(run).is_file(),
        "hybrid_ok": _hybrid_pos(run).is_file(),
    }


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
    geoid = args.geoid_correction
    if geoid is None:
        geoid = GEOID_CONSTANT_BY_CITY.get(city, "none")
    cmd.extend(["--geoid-correction", str(geoid)])
    _run(cmd)
    return 0


def cmd_hybrid(args: argparse.Namespace) -> int:
    """Generate libgnss++ RTK hybrid .pos for one PPC run (WSL build required)."""
    city, run_name = _city_run(args.run)
    ppc_root = _ppc_root(args.data_root)
    run_dir = _run_dir(ppc_root, args.run)
    out_dir = HYBRID_POS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pos = out_dir / f"{city}_{run_name}_full.pos"
    if out_pos.is_file() and not args.force:
        print(f"[hybrid] reuse existing {out_pos}", flush=True)
        return 0
    if not GNSS_SOLVE_WSL.is_file():
        raise SystemExit(
            f"missing gnss_solve binary: {GNSS_SOLVE_WSL}\n"
            "Build in WSL: third_party/gnssplusplus with g++-10, then rerun hybrid."
        )
    script = PROJECT_ROOT / "experiments" / "scripts" / f"run_libgnss_rtk_{city}_{run_name}.sh"
    if script.is_file():
        cmd = ["wsl", "bash", str(script).replace("\\", "/").replace("C:", "/mnt/c", 1)]
        print("[prep] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=_child_env())
        return 0
    wrapper = PROJECT_ROOT / "experiments" / "run_libgnss_rtk_wsl.py"
    if wrapper.is_file():
        cmd = [
            sys.executable,
            str(wrapper),
            "--run",
            args.run,
            "--data-root",
            str(ppc_root),
            "--out-dir",
            str(out_dir),
        ]
        if args.force:
            cmd.append("--force")
        if args.with_diagnostics:
            cmd.append("--with-diagnostics")
        _run(cmd)
        return 0
    gnss = str(GNSS_SOLVE_WSL).replace("\\", "/")
    if gnss.startswith("C:"):
        gnss = "/mnt/c" + gnss[2:]
    data = str(run_dir).replace("\\", "/")
    if data.startswith("C:"):
        data = "/mnt/c" + data[2:]
    elif data.startswith("E:"):
        data = "/mnt/e" + data[2:]
    out = str(out_pos).replace("\\", "/")
    if out.startswith("C:"):
        out = "/mnt/c" + out[2:]
    profile_args = (
        "--preset low-cost --arfilter --arfilter-margin 0.35 "
        "--min-hold-count 8 --hold-ratio-threshold 2.6"
        if city == "tokyo"
        else "--preset low-cost --min-hold-count 7 --hold-ratio-threshold 2.4"
    )
    cmd = (
        f'wsl -e bash -lc "set -euo pipefail; '
        f'echo [hybrid] {args.run}; '
        f'"{gnss}" --rover "{data}/rover.obs" --base "{data}/base.obs" '
        f'--nav "{data}/base.nav" --skip-epochs 0 --out "{out}" --no-kml {profile_args}"'
    )
    print("[prep] " + cmd, flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, shell=True, env=_child_env())
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
        "--start-epoch",
        str(int(args.start_epoch)),
        "--n-particles",
        str(int(args.n_particles)),
    ]
    mask_csv = _mask_csv(args.run)
    if mask_csv.is_file():
        cmd.extend(["--mask-csv", str(mask_csv)])
    cmd.extend(["--profile", str(args.profile)])
    hybrid_dir = args.hybrid_pos_dir
    if hybrid_dir is not None and (hybrid_dir / f"{_city_run(args.run)[0]}_{_city_run(args.run)[1]}_full.pos").is_file():
        cmd.extend(["--hybrid-pos-dir", str(hybrid_dir)])
    elif args.profile == "signal":
        pass  # run_pf_nlos_smoke falls back to reference oracle when hybrid is missing
    if args.skip_ab:
        cmd.append("--skip-ab")
    _run(cmd)
    return 0


def cmd_gaps(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "find_pf_nlos_hybrid_gaps.py"),
        "--run",
        args.run,
        "--window",
        str(int(args.window)),
        "--top",
        str(int(args.top)),
    ]
    if args.hybrid_pos_dir is not None:
        city, run_name = _city_run(args.run)
        pos = args.hybrid_pos_dir / f"{city}_{run_name}_full.pos"
        cmd.extend(["--pos-file", str(pos)])
    _run(cmd)
    return 0


def cmd_wave2_check(args: argparse.Namespace) -> int:
    """Report readiness for ranker/rtkdiag Wave 2 (v5_nlos + candidate pool)."""
    runs = _parse_runs(args.runs)
    masks = {run: _mask_csv(run).is_file() for run in runs}
    report: dict[str, object] = {
        "wave2_target": "ranker/rtkdiag with v5_nlos + 50+ candidate pool (see scripts_run_phase33_perrun_production.sh)",
        "masks_ready": masks,
        "all_masks_ready": all(masks.values()),
        "selector_v3_features": str(SELECTOR_V3_FEATURES),
        "selector_v3_exists": SELECTOR_V3_FEATURES.is_file(),
        "selector_v5_features": str(SELECTOR_V5_FEATURES),
        "selector_v5_exists": SELECTOR_V5_FEATURES.is_file(),
        "v5_ranker_predictions": str(V5_RANKER_PREDICTIONS),
        "v5_ranker_predictions_exists": V5_RANKER_PREDICTIONS.is_file(),
        "sample_candidate_dirs": {
            str(p): p.is_dir() for p in PHASE33_CANDIDATE_DIRS
        },
        "wave2_manifests": {
            run: (MANIFEST_DIR / f"{run.replace('/', '_')}.json").is_file()
            for run in runs
        },
        "wave2_manifests_ready": all(
            (MANIFEST_DIR / f"{run.replace('/', '_')}.json").is_file() for run in runs
        ),
        "candidate_pool_hint": (
            "Wave 2 bootstrap: experiments/bootstrap_rtkdiag_candidate_pool.py "
            "writes experiments/results/rtkdiag_manifest/{city}_{run}.json"
        ),
        "ready_for_ranker_features": SELECTOR_V3_FEATURES.is_file() and all(masks.values()),
        "ready_for_wave2_features": (
            all(masks.values())
            and all((MANIFEST_DIR / f"{run.replace('/', '_')}.json").is_file() for run in runs)
        ),
        "ready_for_phase33_smoke": (
            SELECTOR_V3_FEATURES.is_file()
            and V5_RANKER_PREDICTIONS.is_file()
            and any(p.is_dir() for p in PHASE33_CANDIDATE_DIRS)
        ),
        "next_commands": [
            "python experiments/prepare_pf_nlos_production.py wave2-bootstrap --runs all",
            "python experiments/prepare_pf_nlos_production.py wave2-features --runs all",
            "python experiments/prepare_pf_nlos_production.py wave2-train",
            "python experiments/prepare_pf_nlos_production.py wave2-smoke --run nagoya/run2",
        ],
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ready_for_ranker_features"] else 2


def cmd_ranker_features(args: argparse.Namespace) -> int:
    """Merge plateau_nlos_phase33 mask stats into selector v3 training features."""
    if not SELECTOR_V3_FEATURES.is_file():
        raise SystemExit(
            f"missing base features: {SELECTOR_V3_FEATURES}\n"
            "Generate selector_training_features_v3.csv from the Phase 11/29 pipeline first."
        )
    for run in _parse_runs(args.runs):
        if not _mask_csv(run).is_file():
            raise SystemExit(f"missing mask for {run}: {_mask_csv(run)}")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "augment_selector_training_features_with_nlos.py"),
        "--in-csv",
        str(SELECTOR_V3_FEATURES),
        "--mask-dir",
        str(MASK_DIR),
        "--out-csv",
        str(SELECTOR_V5_FEATURES),
    ]
    _run(cmd)
    return 0


def cmd_wave2_bootstrap(args: argparse.Namespace) -> int:
    ppc_root = _ppc_root(args.data_root)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "bootstrap_rtkdiag_candidate_pool.py"),
        "--runs",
        str(args.runs),
        "--data-root",
        str(ppc_root),
    ]
    if args.force:
        cmd.append("--force")
    _run(cmd)
    return 0


def cmd_wave2_features(args: argparse.Namespace) -> int:
    """Extract selector v3 from Wave 2 manifests, then merge PLATEAU NLOS columns."""
    runs = _parse_runs(args.runs)
    for run in runs:
        city, run_name = _city_run(run)
        manifest = MANIFEST_DIR / f"{city}_{run_name}.json"
        if not manifest.is_file():
            raise SystemExit(f"missing manifest: {manifest} (run wave2-bootstrap first)")
    _run([sys.executable, str(PROJECT_ROOT / "experiments" / "extract_selector_training_features_v3.py")])
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "augment_selector_training_features_with_nlos.py"),
        "--in-csv",
        str(SELECTOR_V3_FEATURES),
        "--mask-dir",
        str(MASK_DIR),
        "--out-csv",
        str(WAVE2_SELECTOR_V5_FEATURES),
    ]
    _run(cmd)
    return 0


def cmd_wave2_train(args: argparse.Namespace) -> int:
    if not WAVE2_SELECTOR_V5_FEATURES.is_file():
        raise SystemExit(f"missing {WAVE2_SELECTOR_V5_FEATURES} (run wave2-features first)")
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "train_selector_ranker_v5_nlos.py"),
            "--features-csv",
            str(WAVE2_SELECTOR_V5_FEATURES),
            "--predictions-csv",
            str(WAVE2_RANKER_PREDICTIONS),
            "--model-out",
            str(WAVE2_RANKER_MODEL),
        ]
    )
    return 0


def _load_manifest(run: str) -> tuple[str, str]:
    city, run_name = _city_run(run)
    manifest_path = MANIFEST_DIR / f"{city}_{run_name}.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dirs = ",".join(str(PROJECT_ROOT / d) for d in payload["dirs"])
    labels = ",".join(str(x) for x in payload["labels"])
    return dirs, labels


def cmd_wave2_smoke(args: argparse.Namespace) -> int:
    """Single-run rtkdiag_pf smoke with Wave 2 candidate pool + v5_nlos ranker."""
    if not WAVE2_RANKER_PREDICTIONS.is_file():
        raise SystemExit(f"missing ranker CSV: {WAVE2_RANKER_PREDICTIONS} (run wave2-train first)")
    run = str(args.run).strip().strip("/")
    city, run_name = _city_run(run)
    dirs, labels = _load_manifest(run)
    k = 99 if city == "nagoya" and run_name == "run2" else int(args.rms_prefilter_k)
    prefix = f"ppc_pf_nlos_wave2_{city}_{run_name}"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "exp_ppc_ctrbpf_fgo.py"),
        "--data-root",
        str(_ppc_root(args.data_root)),
        "--runs",
        run,
        "--methods",
        "rbpf+dd+gate+hybrid+rtkdiag_pf",
        "--n-particles",
        str(int(args.n_particles)),
        "--max-epochs",
        str(int(args.max_epochs)),
        "--start-epoch",
        str(int(args.start_epoch)),
        "--results-prefix",
        prefix,
        "--hybrid-pos-dir",
        str(HYBRID_POS_DIR),
        "--hybrid-sigma-m",
        "1.0",
        "--rtkdiag-candidate-pos-dirs",
        dirs,
        "--rtkdiag-candidate-diag-dirs",
        dirs,
        "--rtkdiag-candidate-labels",
        labels,
        "--rtkdiag-candidate-select-mode",
        "ranker",
        "--rtkdiag-candidate-ranker-score-path",
        str(WAVE2_RANKER_PREDICTIONS),
        "--rtkdiag-candidate-emit-mode",
        "candidate",
        "--rtkdiag-candidate-fallback-mode",
        "hybrid",
        "--rtkdiag-candidate-residual-rms-max",
        "50.0",
        "--rtkdiag-candidate-ratio-min",
        "1.0",
        "--rtkdiag-candidate-rms-prefilter-k",
        str(k),
        "--rtkdiag-candidate-recenter-max-shift-m",
        "10000.0",
        "--rtkdiag-candidate-emit-max-diff-m",
        "0.4",
        "--rtkdiag-candidate-max-to-hybrid-m",
        "0",
        "--rtkdiag-candidate-bridge-enable",
        "--rtkdiag-candidate-bridge-max-s",
        "6.0",
        "--rtkdiag-candidate-bridge-residual-rms-m",
        "0.2",
    ]
    _run(cmd)
    runs_csv = RESULTS_DIR / f"{prefix}_runs.csv"
    print(f"[wave2-smoke] wrote {runs_csv}", flush=True)
    return 0


def cmd_wave2_oracle(args: argparse.Namespace) -> int:
    """Oracle ceiling: best-of hybrid + Wave 2 candidates on one run window."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "analyze_wave2_oracle_ceiling.py"),
        "--run",
        str(args.run).strip().strip("/"),
        "--data-root",
        str(_ppc_root(args.data_root)),
        "--hybrid-pos-dir",
        str(HYBRID_POS_DIR),
        "--manifest-dir",
        str(MANIFEST_DIR),
        "--start-epoch",
        str(int(args.start_epoch)),
        "--max-epochs",
        str(int(args.max_epochs)),
    ]
    _run(cmd)
    return 0


def cmd_batch_prep(args: argparse.Namespace) -> int:
    """fetch + full mask + hybrid for each run missing SSD artifacts."""
    runs = _parse_runs(args.runs)
    report: list[dict[str, object]] = []
    for run in runs:
        status = _run_status(args, run)
        row: dict[str, object] = {"run": run, "before": status, "steps": []}
        if not status["ppc_ok"]:
            print(f"[batch] skip {run}: missing PPC data", flush=True)
            row["skipped"] = "missing_ppc"
            report.append(row)
            continue
        run_args = argparse.Namespace(
            **{
                **vars(args),
                "run": run,
                "max_epochs": int(getattr(args, "max_epochs", 0)),
                "start_epoch": int(getattr(args, "start_epoch", 0)),
            }
        )
        if not status["plateau_ok"] or args.force_fetch:
            cmd_fetch(run_args)
            row["steps"].append("fetch")
        elif status["plateau_ok"]:
            print(f"[batch] reuse plateau {run}", flush=True)
        status = _run_status(args, run)
        if not status["mask_ok"] or args.force_mask:
            cmd_mask(run_args)
            row["steps"].append("mask")
        elif status["mask_ok"]:
            print(f"[batch] reuse mask {run}", flush=True)
        if not status["hybrid_ok"] or args.force_hybrid:
            cmd_hybrid(run_args)
            row["steps"].append("hybrid")
        elif status["hybrid_ok"]:
            print(f"[batch] reuse hybrid {run}", flush=True)
        row["after"] = _run_status(args, run)
        report.append(row)
    out = PROJECT_ROOT / "experiments" / "results" / "pf_nlos_batch_prep_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    failed = [
        r
        for r in report
        if "skipped" not in r and not r.get("after", {}).get("mask_ok", False)
    ]
    return 0 if not failed else 2


def cmd_batch_smoke(args: argparse.Namespace) -> int:
    """Run signal-profile mask A/B smoke on all prepared runs; aggregate deltas."""
    runs = _parse_runs(args.runs)
    summaries: list[dict[str, object]] = []
    for run in runs:
        status = _run_status(args, run)
        if not status["ppc_ok"]:
            summaries.append({"run": run, "skipped": "missing_ppc"})
            continue
        if not status["mask_ok"]:
            summaries.append({"run": run, "skipped": "missing_mask"})
            continue
        run_args = argparse.Namespace(
            **{
                **vars(args),
                "run": run,
                "max_epochs": int(getattr(args, "max_epochs", 0)),
                "start_epoch": int(getattr(args, "start_epoch", 0)),
            }
        )
        cmd_smoke(run_args)
        city, run_name = _city_run(run)
        summary_path = (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / f"ppc_pf_nlos_smoke_{city}_{run_name}_{args.profile}_summary.json"
        )
        if summary_path.is_file():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        else:
            summaries.append({"run": run, "skipped": "missing_summary"})
    agg = {
        "profile": str(args.profile),
        "start_epoch": int(args.start_epoch),
        "max_epochs": int(args.max_epochs),
        "runs": summaries,
        "delta_pp_by_run": {
            str(s.get("run", "?")): s.get("delta_pp")
            for s in summaries
            if "delta_pp" in s
        },
    }
    out = PROJECT_ROOT / "experiments" / "results" / "ppc_pf_nlos_batch_smoke_summary.json"
    out.write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(agg, indent=2), flush=True)
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
        default=None,
        help="egm96, none, or constant metres; default uses city constant when EGM96 grids are missing",
    )
    mask.set_defaults(func=cmd_mask)

    hybrid = sub.add_parser("hybrid", parents=[common], help="Generate libgnss RTK .pos for one run via WSL")
    hybrid.add_argument("--force", action="store_true", help="Regenerate even if .pos exists")
    hybrid.add_argument(
        "--with-diagnostics",
        action="store_true",
        help="Also emit gnss_solve diagnostics CSV (required for rtkdiag_pf smoke)",
    )
    hybrid.set_defaults(func=cmd_hybrid)

    smoke = sub.add_parser("smoke", parents=[common], help="Run baseline vs soft-k3 PPC smoke")
    smoke.add_argument("--max-epochs", type=int, default=120)
    smoke.add_argument("--start-epoch", type=int, default=1000)
    smoke.add_argument("--n-particles", type=int, default=2000)
    smoke.add_argument(
        "--profile",
        choices=("minimal", "gate", "signal", "full"),
        default="signal",
        help="signal=rbpf+dd+gate+hybrid (default); pass --hybrid-pos-dir for libgnss",
    )
    smoke.add_argument(
        "--hybrid-pos-dir",
        type=Path,
        default=HYBRID_POS_DIR,
        help="libgnss RTK .pos directory (default: experiments/results/libgnss_rtk_pos_v5)",
    )
    smoke.add_argument("--skip-ab", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    gaps = sub.add_parser("gaps", parents=[common], help="Find hybrid-low PPC windows for mask A/B")
    gaps.add_argument("--window", type=int, default=1200)
    gaps.add_argument("--top", type=int, default=5)
    gaps.add_argument("--hybrid-pos-dir", type=Path, default=HYBRID_POS_DIR)
    gaps.set_defaults(func=cmd_gaps)

    wave2_common = argparse.ArgumentParser(add_help=False)
    wave2_common.add_argument("--runs", default="all")

    wave2_check = sub.add_parser("wave2-check", parents=[wave2_common], help="Check ranker/rtkdiag Wave 2 prerequisites")
    wave2_check.set_defaults(func=cmd_wave2_check)

    ranker_features = sub.add_parser(
        "ranker-features",
        parents=[wave2_common],
        help="Build selector_training_features_v5_nlos.csv from v3 + plateau_nlos_phase33 masks",
    )
    ranker_features.set_defaults(func=cmd_ranker_features)

    wave2_common = argparse.ArgumentParser(add_help=False)
    wave2_common.add_argument("--runs", default="all")
    wave2_common.add_argument("--data-root", type=Path, default=None)
    wave2_common.add_argument("--force", action="store_true")

    wave2_bootstrap = sub.add_parser(
        "wave2-bootstrap",
        parents=[wave2_common],
        help="Generate Wave 2 libgnss candidate pool + rtkdiag manifests",
    )
    wave2_bootstrap.set_defaults(func=cmd_wave2_bootstrap)

    wave2_features = sub.add_parser(
        "wave2-features",
        parents=[wave2_common],
        help="Extract selector v3 from Wave 2 manifests and build v5_nlos features",
    )
    wave2_features.set_defaults(func=cmd_wave2_features)

    wave2_train = sub.add_parser("wave2-train", help="Train v5_nlos ranker (lightgbm LORO)")
    wave2_train.set_defaults(func=cmd_wave2_train)

    wave2_smoke = sub.add_parser("wave2-smoke", parents=[common], help="Ranker rtkdiag smoke on one run")
    wave2_smoke.add_argument("--max-epochs", type=int, default=1200)
    wave2_smoke.add_argument("--start-epoch", type=int, default=1000)
    wave2_smoke.add_argument("--n-particles", type=int, default=2000)
    wave2_smoke.add_argument("--rms-prefilter-k", type=int, default=3)
    wave2_smoke.set_defaults(func=cmd_wave2_smoke)

    wave2_oracle = sub.add_parser("wave2-oracle", parents=[common], help="Oracle ceiling on Wave 2 pool")
    wave2_oracle.add_argument("--max-epochs", type=int, default=1200)
    wave2_oracle.add_argument("--start-epoch", type=int, default=1000)
    wave2_oracle.set_defaults(func=cmd_wave2_oracle)

    batch_common = argparse.ArgumentParser(add_help=False)
    batch_common.add_argument("--runs", default="all", help="all or comma-separated city/run list")
    batch_common.add_argument("--data-root", type=Path, default=None)
    batch_common.add_argument("--plateau-root", type=Path, default=DEFAULT_PLATEAU_ROOT)
    batch_common.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)

    batch_prep = sub.add_parser(
        "batch-prep",
        parents=[batch_common],
        help="fetch+mask+hybrid for all runs missing SSD artifacts",
    )
    batch_prep.add_argument("--max-epochs", type=int, default=0)
    batch_prep.add_argument("--start-epoch", type=int, default=0)
    batch_prep.add_argument("--mesh-radius", type=int, default=1)
    batch_prep.add_argument("--include-bridges", action="store_true", default=True)
    batch_prep.add_argument("--max-rows", type=int, default=0)
    batch_prep.add_argument("--batch-size", type=int, default=256)
    batch_prep.add_argument("--geoid-correction", default=None)
    batch_prep.add_argument("--force-fetch", action="store_true")
    batch_prep.add_argument("--force-mask", action="store_true")
    batch_prep.add_argument("--force-hybrid", action="store_true")
    batch_prep.add_argument("--force", action="store_true", help="Alias for --force-hybrid")
    batch_prep.add_argument("--with-diagnostics", action="store_true")
    batch_prep.set_defaults(func=cmd_batch_prep)

    batch_smoke = sub.add_parser(
        "batch-smoke",
        parents=[batch_common],
        help="signal smoke A/B on all prepared runs",
    )
    batch_smoke.add_argument("--max-epochs", type=int, default=1200)
    batch_smoke.add_argument("--start-epoch", type=int, default=1000)
    batch_smoke.add_argument("--n-particles", type=int, default=2000)
    batch_smoke.add_argument("--profile", choices=("minimal", "gate", "signal", "full"), default="signal")
    batch_smoke.add_argument("--hybrid-pos-dir", type=Path, default=HYBRID_POS_DIR)
    batch_smoke.add_argument("--skip-ab", action="store_true")
    batch_smoke.set_defaults(func=cmd_batch_smoke)

    args = parser.parse_args(argv)
    if args.command == "batch-prep" and args.force:
        args.force_hybrid = True
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
