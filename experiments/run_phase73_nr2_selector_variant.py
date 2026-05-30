#!/usr/bin/env python3
"""Run nagoya/run2 Phase71 candidate set with a selectable ranker variant."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

PHASE71_LABEL_DIRS = [
    ("csig01hr", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_csig01_holdrlx"),
    ("csig05", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_csig05"),
    ("r15", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed"),
    ("r15g", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_gate30_min6"),
    ("r15nh", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_nohold"),
    ("r20g20", "experiments/results/libgnss_diag_phase10/full_ratio2_lock3_trustedseed_gate20_min6"),
    ("r25", "experiments/results/libgnss_diag_phase10/full_ratio25_lock3_trustedseed"),
    ("r25g", "experiments/results/libgnss_diag_phase10/full_ratio25_lock3_trustedseed_gate30_min6"),
    ("rtkout3", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_rtkout3"),
    ("mlc1oG", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_mlc1oG"),
    ("mlc1oGc005", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_mlc1oGc005"),
    ("rtkout5mlc1c005oG", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_rtkout5mlc1c005oG"),
    ("csig005", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_csig005"),
    ("csig01", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_csig01"),
    ("csig05_psig1", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_csig05_psig1"),
    ("mlc1oGc0001", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_mlc1oGc0001"),
    ("mlc1r10oG", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_mlc1r10oG"),
    ("n2loose3", "experiments/results/libgnss_diag_phase10/n2_loose_hold5_ratio20_gate8_min6"),
    ("oGc005", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_onlyG_csig005"),
    ("psig3", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_psig3"),
    ("rtkout1c005", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_rtkout1c005"),
    ("rtkout5c005em3", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_rtkout5c005em3"),
    ("rtkout5oG", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_rtkout5oG"),
    ("csig005_em10", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_csig005_em10"),
    ("onlyG_r05", "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_onlyG_r05"),
    ("xd_fgo_v2_gap", "experiments/results/libgnss_diag_phase10/fgo_v2_gap"),
    ("xd_fgo_v14_snr38", "experiments/results/libgnss_diag_phase10/fgo_v14_snr38"),
    ("xd_fgo_v17_el25", "experiments/results/libgnss_diag_phase10/fgo_v17_el25"),
    ("xd_gici_def", "experiments/results/libgnss_diag_phase19/gici_tc_esdfix"),
    ("xd_gici_z", "experiments/results/libgnss_diag_phase19/gici_full_zeroarm"),
    ("xd_gici_r", "experiments/results/libgnss_diag_phase19/gici_full_ratio25"),
    ("xd_gici_lp", "experiments/results/libgnss_diag_phase19/gici_full_loosepr"),
    ("xd_gici_lh", "experiments/results/libgnss_diag_phase19/gici_full_loosephase"),
    ("xd_gici_r4", "experiments/results/libgnss_diag_phase19/gici_full_ratio40"),
    ("xd_gici_combo", "experiments/results/libgnss_diag_phase19/gici_full_combo"),
    ("xd_gici_c4", "experiments/results/libgnss_diag_phase19/gici_full_combo4"),
    ("xd_gici_lprlph", "experiments/results/libgnss_diag_phase19/gici_full_lprlph"),
    ("xd_gici_zr", "experiments/results/libgnss_diag_phase19/gici_full_zr"),
    ("xd_gici_oa", "experiments/results/libgnss_diag_phase19/gici_full_onarm"),
    ("xd_gici_la", "experiments/results/libgnss_diag_phase19/gici_full_lowacc"),
    ("xd_gici_hs", "experiments/results/libgnss_diag_phase19/gici_full_hisnr"),
    ("xd_gici_hs45", "experiments/results/libgnss_diag_phase19/gici_full_hisnr45"),
    ("xd_gici_hs30", "experiments/results/libgnss_diag_phase19/gici_full_hisnr30"),
    ("xd_gici_he", "experiments/results/libgnss_diag_phase19/gici_full_hielev"),
    ("xd_gici_ir", "experiments/results/libgnss_diag_phase19/gici_full_imurot"),
    ("xd_gici_mb", "experiments/results/libgnss_diag_phase19/gici_full_himuba"),
    ("xd_gici_w5", "experiments/results/libgnss_diag_phase19/gici_full_window5"),
]


def _comma(values: list[str]) -> str:
    return ",".join(values)


def _parse_extra_label_dir(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("expected LABEL=DIR")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=DIR")
    return label, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker-score-path", required=True)
    parser.add_argument(
        "--select-mode",
        default="ranker_gici_cluster_override",
        choices=[
            "ranker",
            "ranker_gici_cluster_override",
            "ranker_gici_cluster_override_phase79_top1",
        ],
    )
    parser.add_argument("--results-prefix", required=True)
    parser.add_argument("--pos-dir", required=True)
    parser.add_argument("--osm-dir", default="experiments/results/phase70_osm_road_hs_alpha05_triggered")
    parser.add_argument(
        "--extra-label-dir",
        action="append",
        type=_parse_extra_label_dir,
        default=[],
        metavar="LABEL=DIR",
        help="Append an extra candidate label mapped to an existing candidate directory.",
    )
    parser.add_argument("--write-internal-diagnostics", action="store_true")
    args = parser.parse_args(argv)

    label_dirs = list(PHASE71_LABEL_DIRS) + [("xd_gici_osmroad_hs", args.osm_dir)] + list(args.extra_label_dir)
    labels = [label for label, _path in label_dirs]
    dirs = [path for _label, path in label_dirs]

    cmd = [
        sys.executable,
        str(REPO / "experiments/exp_ppc_ctrbpf_fgo.py"),
        "--runs",
        "nagoya/run2",
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
        args.select_mode,
        "--rtkdiag-candidate-ranker-score-path",
        args.ranker_score_path,
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
        "99",
        "--n-particles",
        "2000",
        "--pos-dir",
        args.pos_dir,
        "--results-prefix",
        args.results_prefix,
    ]
    if args.write_internal_diagnostics:
        cmd.append("--write-internal-diagnostics")

    env = os.environ.copy()
    env["PYTHONPATH"] = "python"
    return subprocess.run(cmd, cwd=REPO, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
