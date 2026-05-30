#!/usr/bin/env python3
"""Replay the Phase79 top1 n/r2 overlay in the six-run production frame."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results"
PHASE10 = RESULTS / "libgnss_diag_phase10"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))

from sim_ppc_addcand_sweep import _VARIANTS_COMBO, _VARIANTS_INDIV  # noqa: E402
from sim_ppc_selector_sweep import _CANDIDATES_PHASE11V  # noqa: E402
from run_phase73_nr2_selector_variant import PHASE71_LABEL_DIRS  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    city: str
    run: str
    score_path: str
    select_mode: str
    rms_prefilter_k: int
    label_source_csv: str

    @property
    def key(self) -> str:
        return f"{self.city}_{self.run}"

    @property
    def run_arg(self) -> str:
        return f"{self.city}/{self.run}"


RUN_CONFIGS: dict[str, RunConfig] = {
    "tokyo_run1": RunConfig(
        "tokyo",
        "run1",
        "experiments/results/selector_ranker_predictions.csv",
        "ranker",
        3,
        "experiments/results/ppc_ctrbpf_fgo_phase43_prod_tokyo_run1_full_runs.csv",
    ),
    "tokyo_run2": RunConfig(
        "tokyo",
        "run2",
        "experiments/results/selector_ranker_predictions_v3.csv",
        "ranker",
        3,
        "experiments/results/ppc_phase71_osmroad_prod_tokyo_run2_full_runs.csv",
    ),
    "tokyo_run3": RunConfig(
        "tokyo",
        "run3",
        "experiments/results/selector_ranker_predictions.csv",
        "ranker",
        3,
        "experiments/results/ppc_phase71_osmroad_prod_tokyo_run3_full_runs.csv",
    ),
    "nagoya_run1": RunConfig(
        "nagoya",
        "run1",
        "experiments/results/selector_ranker_predictions_v3.csv",
        "ranker",
        99,
        "experiments/results/ppc_phase71_osmroad_prod_nagoya_run1_full_runs.csv",
    ),
    "nagoya_run2": RunConfig(
        "nagoya",
        "run2",
        "/tmp/selector_ranker_predictions_phase79_nr2_top1_label_pair_overlay.csv",
        "ranker_gici_cluster_override_phase79_top1",
        99,
        "experiments/results/ppc_phase71_osmroad_prod_nagoya_run2_full_runs.csv",
    ),
    "nagoya_run3": RunConfig(
        "nagoya",
        "run3",
        "experiments/results/selector_ranker_predictions.csv",
        "ranker",
        3,
        "experiments/results/ppc_phase71_osmroad_prod_nagoya_run3_full_runs.csv",
    ),
}


EXTRA_LABEL_DIRS = {
    "xcsig005_em10": PHASE10 / "full_ratio15_lock3_trustedseed_csig005_em10",
    "xd_r2_nohold": PHASE10 / "full_ratio2_lock3_trustedseed_nohold",
    "xd_r25_nohold": PHASE10 / "full_ratio25_lock3_trustedseed_nohold",
    "xd_ratio3_gate10_min6": PHASE10 / "full_ratio3_lock3_trustedseed_gate10_min6",
    "xd_ratio4": PHASE10 / "full_ratio4_lock3_trustedseed",
    "xr17_glonassar": PHASE10 / "full_ratio17_lock3_trustedseed_glonassar",
    "xr25_glonassar": PHASE10 / "full_ratio25_lock3_trustedseed_glonassar",
    "xpsig05": PHASE10 / "full_ratio15_lock3_trustedseed_psig05",
    "xd_elev10_outlier3_v2_6runs": PHASE10 / "elev10_outlier3_v2_6runs",
    "xd_strict_r5_outlier3_v1_6runs": PHASE10 / "strict_r5_outlier3_v1_6runs",
    "xd_survey_outlier3_v2_6runs": PHASE10 / "survey_outlier3_v2_6runs",
    "xd_n3_loose_hold4_ratio15_gate10_min6": PHASE10 / "n3_loose_hold4_ratio15_gate10_min6",
    "xd_t1_glo_autocal": PHASE10 / "t1_glo_autocal",
    "xd_dev_demo5_trusted_o3": PHASE10 / "dev_demo5_trusted_o3",
    "xd_demo5_continuous_nojump": PHASE10 / "demo5_continuous_nojump",
    "xd_gici_osmroad_hs": RESULTS / "phase70_osm_road_hs_alpha05_triggered",
}


def _comma(values: list[str]) -> str:
    return ",".join(values)


def _label_dir_map(osm_dir: Path | None = None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for label, subdir, _runs in _CANDIDATES_PHASE11V:
        out[label] = PHASE10 / subdir
    for _name, variants in list(_VARIANTS_INDIV) + list(_VARIANTS_COMBO):
        for label, subdir in variants:
            out[label] = PHASE10 / subdir
    for label, path in PHASE71_LABEL_DIRS:
        out[label] = Path(path)
    out.update(EXTRA_LABEL_DIRS)
    if osm_dir is not None:
        out["xd_gici_osmroad_hs"] = osm_dir
    return out


def _read_labels(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    labels = [label for label in row["rtkdiag_candidate_labels"].split(",") if label]
    if not labels:
        raise ValueError(f"no rtkdiag_candidate_labels in {path}")
    return labels


def _read_phase71_summary() -> dict[str, dict[str, str]]:
    path = RESULTS / "phase71_osmroad_production_summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["run"]: row for row in csv.DictReader(fh)}


def _normalize_run(value: str) -> str:
    key = value.replace("/", "_")
    if key not in RUN_CONFIGS:
        choices = ", ".join(RUN_CONFIGS)
        raise argparse.ArgumentTypeError(f"unknown run {value!r}; choices: {choices}")
    return key


def _result_row(prefix: str) -> dict[str, str]:
    path = RESULTS / f"{prefix}_runs.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.DictReader(fh))


def _ensure_phase79_overlay(score_path: Path) -> None:
    default_path = Path("/tmp/selector_ranker_predictions_phase79_nr2_top1_label_pair_overlay.csv")
    if score_path.exists() or score_path != default_path:
        return
    cmd = [sys.executable, str(SCRIPT_DIR / "build_phase79_nr2_top1_label_pair_overlay.py")]
    env = os.environ.copy()
    env["PYTHONPATH"] = "python"
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _run_one(
    cfg: RunConfig,
    *,
    score_path: str,
    osm_dir: Path | None,
    results_prefix: str,
    pos_dir: str,
    write_internal_diagnostics: bool,
    dry_run: bool,
) -> None:
    labels = _read_labels(REPO / cfg.label_source_csv)
    mapping = _label_dir_map(osm_dir)
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise ValueError(f"{cfg.key} has unmapped labels: {missing}")
    missing_dirs = [f"{label}={mapping[label]}" for label in labels if not mapping[label].exists()]
    if missing_dirs:
        raise FileNotFoundError(f"{cfg.key} has missing candidate dirs: {missing_dirs}")

    dirs = [str(mapping[label]) for label in labels]
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "exp_ppc_ctrbpf_fgo.py"),
        "--runs",
        cfg.run_arg,
        "--methods",
        "rbpf+dd+gate+hybrid+rtkdiag_pf",
        "--hybrid-pos-dir",
        "experiments/results/libgnss_rtk_pos_v5",
        "--hybrid-sigma-m",
        "1.0",
        "--rtkdiag-candidate-pos-dirs",
        _comma(dirs),
        "--rtkdiag-candidate-diag-dirs",
        _comma(dirs),
        "--rtkdiag-candidate-labels",
        _comma(labels),
        "--rtkdiag-candidate-block-labels-by-run",
        "",
        "--rtkdiag-candidate-run-index-policy",
        "phase11ep",
        "--rtkdiag-candidate-select-mode",
        cfg.select_mode,
        "--rtkdiag-candidate-ranker-score-path",
        score_path,
        "--rtkdiag-candidate-emit-mode",
        "candidate",
        "--rtkdiag-candidate-residual-rms-max",
        "50.0",
        "--rtkdiag-candidate-ratio-min",
        "1.0",
        "--rtkdiag-candidate-recenter-max-shift-m",
        "10000.0",
        "--rtkdiag-candidate-emit-max-diff-m",
        "0.4",
        "--rtkdiag-candidate-max-to-hybrid-m",
        "0",
        "--rtkdiag-candidate-fallback-mode",
        "hybrid",
        "--rtkdiag-candidate-bridge-enable",
        "--rtkdiag-candidate-bridge-max-s",
        "6.0",
        "--rtkdiag-candidate-bridge-residual-rms-m",
        "0.2",
        "--rtkdiag-candidate-rms-prefilter-k",
        str(cfg.rms_prefilter_k),
        "--n-particles",
        "2000",
        "--pos-dir",
        pos_dir,
        "--results-prefix",
        results_prefix,
    ]
    if write_internal_diagnostics:
        cmd.append("--write-internal-diagnostics")
    if dry_run:
        print(" ".join(cmd))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = "python"
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _write_summary(
    prefix_base: str,
    runs: list[str],
    *,
    summary_path: Path,
    score_paths: dict[str, str],
) -> Path:
    phase71 = _read_phase71_summary()
    rows: list[dict[str, object]] = []
    for key in runs:
        cfg = RUN_CONFIGS[key]
        prefix = f"{prefix_base}_{key}_full"
        row79 = _result_row(prefix)
        if key in phase71:
            base_ppc = float(phase71[key]["phase71_ppc_pct"])
            base_pass = float(phase71[key]["phase71_pass_m"])
        else:
            base = _result_row(Path(cfg.label_source_csv).name.removesuffix("_runs.csv"))
            base_ppc = float(base["honest_ppc_pct"])
            base_pass = float(base["honest_pass_m"])
        ppc79 = float(row79["honest_ppc_pct"])
        pass79 = float(row79["honest_pass_m"])
        rows.append(
            {
                "run": key,
                "phase71_ppc_pct": base_ppc,
                "phase79_ppc_pct": ppc79,
                "delta_pp": ppc79 - base_ppc,
                "phase71_pass_m": base_pass,
                "phase79_pass_m": pass79,
                "delta_pass_m": pass79 - base_pass,
                "select_mode": cfg.select_mode,
                "rms_prefilter_k": cfg.rms_prefilter_k,
                "ranker_score_path": score_paths.get(key, cfg.score_path),
                "phase79_result_csv": f"experiments/results/{prefix}_runs.csv",
            }
        )

    path = summary_path if summary_path.is_absolute() else REPO / summary_path
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    avg71 = sum(float(row["phase71_ppc_pct"]) for row in rows) / len(rows)
    avg79 = sum(float(row["phase79_ppc_pct"]) for row in rows) / len(rows)
    print(f"Saved: {path}")
    print(f"Phase71 official: {avg71:.6f}%")
    print(f"Phase79 official: {avg79:.6f}%")
    print(f"Delta: {avg79 - avg71:+.6f}pp")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=_normalize_run, default=list(RUN_CONFIGS))
    parser.add_argument("--results-prefix-base", default="ppc_phase79_top1_sixrun")
    parser.add_argument("--pos-dir-base", default="/tmp/ppc_phase79_top1_sixrun")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("experiments/results/phase79_top1_sixrun_neutral_summary.csv"),
    )
    parser.add_argument(
        "--osm-dir",
        type=Path,
        default=Path("experiments/results/phase70_osm_road_hs_alpha05_triggered"),
        help="Candidate directory for xd_gici_osmroad_hs.",
    )
    parser.add_argument(
        "--phase79-score-path",
        default="/tmp/selector_ranker_predictions_phase79_nr2_top1_label_pair_overlay.csv",
        help="Ranker score CSV used only for nagoya/run2.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-internal-diagnostics", action="store_true")
    args = parser.parse_args(argv)

    run_keys = list(dict.fromkeys(args.runs))
    phase79_score_path = Path(args.phase79_score_path)
    if "nagoya_run2" in run_keys:
        _ensure_phase79_overlay(phase79_score_path)

    score_paths: dict[str, str] = {}
    for key in run_keys:
        cfg = RUN_CONFIGS[key]
        score_path = args.phase79_score_path if key == "nagoya_run2" else cfg.score_path
        score_paths[key] = score_path
        if key == "nagoya_run2" and not Path(score_path).exists():
            raise FileNotFoundError(score_path)
        prefix = f"{args.results_prefix_base}_{key}_full"
        if args.skip_existing and (RESULTS / f"{prefix}_runs.csv").exists():
            print(f"SKIP {cfg.run_arg}: experiments/results/{prefix}_runs.csv")
            continue
        print(
            f"=== Phase79 top1 six-run {cfg.run_arg} "
            f"(mode={cfg.select_mode}, k={cfg.rms_prefilter_k}) ==="
        )
        _run_one(
            cfg,
            score_path=score_path,
            osm_dir=args.osm_dir,
            results_prefix=prefix,
            pos_dir=f"{args.pos_dir_base}_{key}",
            write_internal_diagnostics=args.write_internal_diagnostics,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            row = _result_row(prefix)
            print(f"FIN {cfg.run_arg}: {float(row['honest_ppc_pct']):.6f}%")

    if not args.dry_run:
        _write_summary(
            args.results_prefix_base,
            run_keys,
            summary_path=args.summary_path,
            score_paths=score_paths,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
