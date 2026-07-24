#!/usr/bin/env python3
"""Run the FGO-free WP23b partial-ambiguity basin RBPF on PPC."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
for _path in (_ROOT / "python", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _build_dd_measurements,
    _filter_data_by_systems,
    _load_full_reference,
    _reference_position_map,
)
from exp_ppc_tdcp_velocity import _epoch_measurements  # noqa: E402
from exp_urbannav_baseline import run_wls  # noqa: E402
from exp_wp23b_float_seed import _doppler_velocity  # noqa: E402
from gnss_gpu.ambiguity_basin_pf import (  # noqa: E402
    AmbiguityBasinParticleFilter,
    BasinKalmanState,
)
from gnss_gpu.ambiguity_respawn import (  # noqa: E402
    condition_respawn_position,
    ddpr_centered_ambiguity_seed,
)
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_float_kf import DDFloatKalmanFilter  # noqa: E402
from gnss_gpu.dd_integrity import (  # noqa: E402
    multipivot_ddpr_scores,
    satellite_pair_costs,
)
from gnss_gpu.gsdc_dgnss import DDWLSConfig, dd_pseudorange_position_update  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.dd_quality import _subset_dd_result, gate_dd_pseudorange  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.lambda_ambiguity import integer_search, integer_search_batch  # noqa: E402
from gnss_gpu.rtk_evidence import (  # noqa: E402
    EvidenceLedger,
    RTKEpochTrace,
    TrustedFixCommitPolicy,
    TrustedFixPolicyConfig,
    TrustedFixPolicyInput,
    ambiguity_assignment_id,
    ambiguity_assignment_json,
    replay_fix_decisions,
)
from gnss_gpu.recovery_proposals import (  # noqa: E402
    RecoveryAssignmentBank,
    RecoveryArcAssignmentBank,
    RecoveryPositionBank,
    SatelliteArcTracker,
    complete_versioned_assignment,
    covariance_axis_position_seeds,
)
from gnss_gpu.temporal_ambiguity import (  # noqa: E402
    TemporalAmbiguityCandidate,
    TemporalAmbiguityConfig,
    TemporalAmbiguityFilter,
)
from gnss_gpu.tdcp_velocity import estimate_displacement_from_tdcp  # noqa: E402
from gnss_gpu.widelane import (  # noqa: E402
    WidelaneDDPseudorangeComputer,
    WidelaneDDStats,
)

_BASIN_TRACE_FIELDS = (
    "epoch",
    "tow",
    "basin_id",
    "assignment_id",
    "assignment_json",
    "epoch_log_likelihood",
    "cumulative_log_marginal",
    "log_weight",
    "ecef_x",
    "ecef_y",
    "ecef_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "birth_epoch",
    "lineage",
    "proposal_sources",
)


class _StreamingCsvRows:
    """Bounded-memory CSV sink retaining the historical append call site."""

    def __init__(self, path: Path, fieldnames: tuple[str, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        self._writer.writeheader()
        self.count = 0

    def append(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)
        self.count += 1

    def close(self) -> None:
        self._fh.close()


def _write_trajectory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tow", "ecef_x", "ecef_y", "ecef_z", "fix"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def _load_diverse_position_seeds(
    path: Path,
    *,
    separation_m: float,
    max_positions: int,
) -> dict[int, tuple[np.ndarray, ...]]:
    """Load posterior-ranked, position-diverse seeds from a shadow PF trace."""

    rows_by_epoch: dict[int, list[tuple[float, np.ndarray]]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            position = np.asarray(
                [row["ecef_x"], row["ecef_y"], row["ecef_z"]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(position)):
                continue
            rows_by_epoch.setdefault(int(row["epoch"]), []).append(
                (float(row["log_weight"]), position)
            )

    result: dict[int, tuple[np.ndarray, ...]] = {}
    for epoch, ranked_rows in rows_by_epoch.items():
        selected: list[np.ndarray] = []
        for _log_weight, position in sorted(ranked_rows, key=lambda item: -item[0]):
            if all(
                np.linalg.norm(position - existing) > float(separation_m)
                for existing in selected
            ):
                selected.append(position)
            if len(selected) >= int(max_positions):
                break
        if selected:
            result[epoch] = tuple(selected)
    return result


def _append_distinct_position_seed(
    seeds: tuple[np.ndarray, ...],
    candidate: np.ndarray,
    *,
    separation_m: float = 1.0e-3,
) -> tuple[tuple[np.ndarray, ...], int]:
    """Append a finite seed once and return its stable source index."""

    existing = tuple(np.asarray(seed, dtype=np.float64).reshape(3) for seed in seeds)
    value = np.asarray(candidate, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError("position seed must be finite")
    for index, seed in enumerate(existing):
        if np.linalg.norm(value - seed) <= float(separation_m):
            return existing, int(index)
    return existing + (value.copy(),), len(existing)


def _widelane_integer_residual(
    assignment: tuple,
    fixed_dd_ambiguities: tuple[tuple[str, str, int], ...],
) -> tuple[int, float]:
    family_integers: dict[tuple[str, str, str], int] = {}
    for versioned_key, integer in assignment:
        ambiguity_key, _generation = versioned_key
        ref_sat, sat_id, _wavelength_nm = ambiguity_key
        if "@" not in ref_sat or "@" not in sat_id:
            continue
        ref_base, ref_family = ref_sat.split("@", 1)
        sat_base, sat_family = sat_id.split("@", 1)
        if ref_family != sat_family:
            continue
        family_integers[(ref_base, sat_base, ref_family)] = int(integer)

    residuals: list[int] = []
    for ref_sat, sat_id, wide_integer in fixed_dd_ambiguities:
        l1_key = (ref_sat, sat_id, "L1_E1_B1")
        l2_key = (ref_sat, sat_id, "L2_E5B_B2")
        if l1_key in family_integers and l2_key in family_integers:
            residuals.append(
                family_integers[l1_key]
                - family_integers[l2_key]
                - int(wide_integer)
            )
    if not residuals:
        return 0, float("nan")
    values = np.asarray(residuals, dtype=np.float64)
    return len(residuals), float(values @ values)


def _select_ambiguity_indices(
    keys: tuple,
    covariance: np.ndarray,
    available: np.ndarray,
    subset_size: int,
    *,
    prefer_multifrequency_pairs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    available_arr = np.asarray(available, dtype=np.int64)
    variances = np.diag(np.asarray(covariance, dtype=np.float64))[available_arr]
    variance_ranked = available_arr[np.argsort(variances)]
    if not prefer_multifrequency_pairs:
        selected = np.sort(variance_ranked[: int(subset_size)])
        return selected, variance_ranked

    available_set = set(int(index) for index in available_arr)
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for index, key in enumerate(keys):
        if index not in available_set:
            continue
        ref_sat, sat_id, _wavelength_nm = key
        if "@" not in ref_sat or "@" not in sat_id:
            continue
        ref_base, ref_family = ref_sat.split("@", 1)
        sat_base, sat_family = sat_id.split("@", 1)
        if ref_family != sat_family:
            continue
        grouped.setdefault((ref_base, sat_base), {})[ref_family] = index

    diagonal = np.diag(np.asarray(covariance, dtype=np.float64))
    pairs: list[tuple[float, int, int]] = []
    for families in grouped.values():
        if "L1_E1_B1" in families and "L2_E5B_B2" in families:
            l1_index = families["L1_E1_B1"]
            l2_index = families["L2_E5B_B2"]
            pairs.append(
                (
                    float(diagonal[l1_index] + diagonal[l2_index]),
                    l1_index,
                    l2_index,
                )
            )
    selected_order: list[int] = []
    for _pair_variance, l1_index, l2_index in sorted(pairs):
        if len(selected_order) + 2 > int(subset_size):
            break
        selected_order.extend((l1_index, l2_index))
    selected_set = set(selected_order)
    selected_order.extend(
        int(index)
        for index in variance_ranked
        if int(index) not in selected_set
    )
    ranked = np.asarray(selected_order, dtype=np.int64)
    return np.sort(ranked[: int(subset_size)]), ranked


def main(argv: list[str] | None = None) -> None:
    runtime_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run2")
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/PPC-Dataset-data"))
    parser.add_argument("--dd-systems", default="G,E,J,C")
    parser.add_argument(
        "--dd-carrier-families",
        default="",
        help=(
            "Comma-separated carrier families for multi-frequency DD carrier "
            "evidence; empty preserves the historical single-frequency path"
        ),
    )
    parser.add_argument("--pr-systems", default="G,E,J")
    parser.add_argument("--subset-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--lambda-engine",
        choices=("cpu", "gpu-batch"),
        default="cpu",
        help="ILS engine for genuinely batched multi-position respawn proposals",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("audit", "fast"),
        default="audit",
        help="fast omits large per-basin/integrity audit traces but preserves estimates",
    )
    parser.add_argument("--max-basins", type=int, default=128)
    parser.add_argument("--basin-diversity-reserve-fraction", type=float, default=0.0)
    parser.add_argument("--basin-diversity-radius-m", type=float, default=1.0)
    parser.add_argument("--basin-dedup-position-radius-m", type=float, default=float("inf"))
    parser.add_argument("--basin-source-reserve-fraction", type=float, default=0.0)
    parser.add_argument("--basin-protected-source-token", default="")
    parser.add_argument("--basin-protected-source-fraction", type=float, default=0.0)
    parser.add_argument("--birth-mass", type=float, default=0.01)
    parser.add_argument("--sigma-dd-pr-m", type=float, default=5.0)
    parser.add_argument(
        "--sigma-basin-dd-pr-m",
        type=float,
        default=0.0,
        help="Basin-only DDPR sigma; <=0 uses --sigma-dd-pr-m",
    )
    parser.add_argument("--sigma-float-cp-cycles", type=float, default=0.10)
    parser.add_argument("--float-slip-threshold-cycles", type=float, default=2.0)
    parser.add_argument("--sigma-fixed-cp-cycles", type=float, default=0.20)
    parser.add_argument("--enable-widelane-basin-evidence", action="store_true")
    parser.add_argument("--widelane-min-epochs", type=int, default=5)
    parser.add_argument("--widelane-max-std-cycles", type=float, default=0.75)
    parser.add_argument("--widelane-ratio-threshold", type=float, default=3.0)
    parser.add_argument("--widelane-min-fix-rate", type=float, default=0.3)
    parser.add_argument("--widelane-basin-sigma-m", type=float, default=1.0)
    parser.add_argument("--enable-widelane-integer-score", action="store_true")
    parser.add_argument("--widelane-integer-min-pairs", type=int, default=1)
    parser.add_argument("--widelane-integer-mismatch-penalty", type=float, default=10.0)
    parser.add_argument("--widelane-integer-missing-penalty", type=float, default=10.0)
    parser.add_argument("--prefer-paired-multifrequency-subset", action="store_true")
    parser.add_argument("--fix-gamma", type=float, default=0.99)
    parser.add_argument("--fix-streak", type=int, default=3)
    parser.add_argument(
        "--fix-consistency-m",
        type=float,
        default=0.5,
        help="Maximum MAP-basin versus independent float-KF position separation",
    )
    parser.add_argument(
        "--fix-ddpr-consistency-m",
        type=float,
        default=1.75,
        help="Maximum MAP-basin separation from the DDPR/Doppler-only guard KF",
    )
    parser.add_argument("--fix-min-dd-pairs", type=int, default=9)
    parser.add_argument("--fix-max-ddpr-age-epochs", type=int, default=4)
    parser.add_argument("--position-cluster-radius-m", type=float, default=0.5)
    parser.add_argument("--enable-temporal-lineage", action="store_true")
    parser.add_argument("--temporal-birth-mass", type=float, default=0.05)
    parser.add_argument("--temporal-change-cost", type=float, default=2.0)
    parser.add_argument("--temporal-incompatible-cost", type=float, default=12.0)
    parser.add_argument("--temporal-death-cost", type=float, default=6.0)
    parser.add_argument("--temporal-motion-sigma-m", type=float, default=3.0)
    parser.add_argument("--enable-integrity-lineage", action="store_true")
    parser.add_argument("--integrity-scale-m", type=float, default=3.0)
    parser.add_argument("--integrity-trim-pairs", type=int, default=0)
    parser.add_argument("--integrity-weight", type=float, default=5.0)
    parser.add_argument(
        "--integrity-exclude-max-cost-satellite",
        action="store_true",
        help="Exclude the largest guard-position incident pair-cost satellite",
    )
    parser.add_argument(
        "--integrity-satellite-cost-memory",
        type=float,
        default=0.0,
        help="EMA memory in [0,1) for causal per-satellite incident pair cost",
    )
    parser.add_argument("--integrity-tdcp-systems", default="G,E,J")
    parser.add_argument("--integrity-tdcp-min-sats", type=int, default=5)
    parser.add_argument("--integrity-tdcp-max-postfit-rms-m", type=float, default=0.5)
    parser.add_argument("--integrity-tdcp-slip-threshold-m", type=float, default=0.25)
    parser.add_argument("--enable-ddpr-respawn", action="store_true")
    parser.add_argument("--ddpr-respawn-trigger-m", type=float, default=1.75)
    parser.add_argument("--ddpr-respawn-mass", type=float, default=0.05)
    parser.add_argument("--ddpr-respawn-use-lambda-prior", action="store_true")
    parser.add_argument(
        "--ddpr-respawn-top-k",
        type=int,
        default=0,
        help="Respawn-only candidate count; <=0 uses --top-k",
    )
    parser.add_argument(
        "--ddpr-respawn-subset-size",
        type=int,
        default=0,
        help="Respawn-only ambiguity dimension; <=0 uses --subset-size",
    )
    parser.add_argument(
        "--ddpr-respawn-seed-radii-m",
        default="",
        help="Comma-separated covariance-axis position seed radii; empty uses center only",
    )
    parser.add_argument(
        "--ddpr-respawn-seed-directions",
        choices=("axes", "cube26"),
        default="axes",
    )
    parser.add_argument("--ddpr-respawn-shadow-seed-radii-m", default="")
    parser.add_argument("--ddpr-respawn-snapshot-seed-shadow-only", action="store_true")
    parser.add_argument("--ddpr-respawn-snapshot-seed-promote", action="store_true")
    parser.add_argument("--ddpr-respawn-snapshot-loo-shadow-only", action="store_true")
    parser.add_argument(
        "--ddpr-snapshot-pair-exclusion-position-shadow-only", action="store_true"
    )
    parser.add_argument("--ddpr-respawn-wls-seed-shadow-only", action="store_true")
    parser.add_argument(
        "--ddpr-respawn-trusted-fix-anchor-shadow-only", action="store_true"
    )
    parser.add_argument(
        "--trusted-fix-anchor-snapshot-reset-rms-m", type=float, default=0.0
    )
    parser.add_argument(
        "--trusted-fix-anchor-motion-fallback",
        choices=(
            "doppler",
            "doppler-calibrated",
            "imu-preint",
            "last-tdcp",
            "hold",
        ),
        default="doppler",
    )
    parser.add_argument(
        "--trusted-fix-anchor-doppler-bias-window", type=int, default=25
    )
    parser.add_argument("--trusted-fix-anchor-shadow-radii-m", default="")
    parser.add_argument("--trusted-fix-anchor-float-line-radii-m", default="")
    parser.add_argument("--trusted-fix-anchor-float-line-promote", action="store_true")
    parser.add_argument("--external-position-seeds-csv", type=Path)
    parser.add_argument("--external-position-seed-separation-m", type=float, default=0.5)
    parser.add_argument("--external-position-seed-max", type=int, default=64)
    parser.add_argument("--external-position-seed-top-k", type=int, default=2)
    parser.add_argument(
        "--external-position-seed-mode",
        choices=("lambda", "rounded-direct"),
        default="lambda",
    )
    parser.add_argument("--external-position-seeds-promote", action="store_true")
    parser.add_argument(
        "--trusted-fix-anchor-imu-velocity-blend-alpha", type=float, default=0.3
    )
    parser.add_argument("--trusted-anchor-refinement-seeds", type=int, default=0)
    parser.add_argument("--trusted-anchor-refinement-top-k", type=int, default=24)
    parser.add_argument("--trusted-anchor-shadow-subset-size", type=int, default=0)
    parser.add_argument("--ddpr-respawn-snapshot-shadow-radii-m", default="")
    parser.add_argument("--ddpr-respawn-snapshot-extrapolation-scales", default="")
    parser.add_argument("--ddpr-respawn-snapshot-shadow-top-k", type=int, default=0)
    parser.add_argument("--ddpr-snapshot-prior-sigma-m", type=float, default=0.0)
    parser.add_argument("--ddpr-snapshot-max-shift-m", type=float, default=200.0)
    parser.add_argument("--ddpr-snapshot-pair-residual-max-m", type=float, default=0.0)
    parser.add_argument(
        "--ddpr-respawn-exclude-max-cost-satellite", action="store_true"
    )
    parser.add_argument("--ddpr-respawn-shadow-one-swap-top-k", type=int, default=0)
    parser.add_argument("--ddpr-respawn-shadow-window-count", type=int, default=0)
    parser.add_argument("--ddpr-respawn-shadow-window-top-k", type=int, default=0)
    parser.add_argument("--ddpr-respawn-history-seeds", type=int, default=0)
    parser.add_argument("--ddpr-respawn-history-separation-m", type=float, default=1.0)
    parser.add_argument("--ddpr-respawn-history-max-age-epochs", type=int, default=25)
    parser.add_argument(
        "--ddpr-respawn-history-selection",
        choices=("weight", "farthest"),
        default="weight",
    )
    parser.add_argument(
        "--ddpr-respawn-history-max-guard-distance-m", type=float, default=float("inf")
    )
    parser.add_argument(
        "--ddpr-respawn-history-propagate-velocity", action="store_true"
    )
    parser.add_argument("--ddpr-respawn-history-propagate-tdcp", action="store_true")
    parser.add_argument("--ddpr-respawn-assignment-history", type=int, default=0)
    parser.add_argument(
        "--ddpr-respawn-assignment-history-max-age-epochs", type=int, default=50
    )
    parser.add_argument(
        "--ddpr-respawn-assignment-pivot-rebase", action="store_true"
    )
    parser.add_argument("--ddpr-respawn-assignment-arc-shadow", action="store_true")
    parser.add_argument("--ddpr-respawn-assignment-arc-promote", action="store_true")
    parser.add_argument(
        "--ddpr-respawn-assignment-arc-slip-threshold-cycles",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--ddpr-respawn-assignment-arc-max-gap-epochs", type=int, default=1
    )
    parser.add_argument(
        "--ddpr-respawn-assignment-arc-completion-top-k", type=int, default=0
    )
    parser.add_argument(
        "--ddpr-respawn-assignment-arc-completion-per-assignment",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ddpr-respawn-assignment-arc-shadow-max-candidates", type=int, default=0
    )
    parser.add_argument("--ddpr-respawn-assignment-completion-top-k", type=int, default=0)
    parser.add_argument("--ddpr-respawn-assignment-completion-min-stable", type=int, default=4)
    parser.add_argument(
        "--ddpr-respawn-assignment-completion-shadow-only", action="store_true"
    )
    parser.add_argument("--out-diagnostics", type=Path, default=Path("results/wp23b/csv/basin_run2_epochs.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/wp23b/csv/basin_run2_summary.json"))
    parser.add_argument("--out-trajectory", type=Path, default=Path("results/wp23b/pos/basin_run2.csv"))
    parser.add_argument(
        "--out-trace",
        type=Path,
        default=None,
        help="Optional truth-free epoch trace for deterministic FIX replay",
    )
    parser.add_argument(
        "--out-evidence",
        type=Path,
        default=None,
        help="Optional observation evidence-provenance CSV",
    )
    parser.add_argument(
        "--out-basin-trace",
        type=Path,
        default=None,
        help="Optional truth-free per-basin trace for temporal replay",
    )
    parser.add_argument(
        "--out-integrity-satellite-diagnostics",
        type=Path,
        default=None,
        help="Optional truth-joined leave-one-satellite-out DDPR diagnostics",
    )
    args = parser.parse_args(argv)
    if args.runtime_mode == "fast" and (
        args.out_basin_trace is not None
        or args.out_integrity_satellite_diagnostics is not None
    ):
        parser.error(
            "--runtime-mode fast cannot retain per-basin or integrity audit traces"
        )
    if args.lambda_engine == "gpu-batch":
        # Warm up CUDA outside the measured epoch loop and fail closed before
        # loading the full dataset if the locked engine is unavailable.
        integer_search_batch(
            [np.zeros(8, dtype=np.float64)],
            [np.eye(8, dtype=np.float64)],
            n_candidates=24,
            engine="gpu-batch",
        )
    if not 0.0 <= float(args.integrity_satellite_cost_memory) < 1.0:
        parser.error("--integrity-satellite-cost-memory must be in [0, 1)")
    respawn_seed_radii = tuple(
        float(value)
        for value in str(args.ddpr_respawn_seed_radii_m).split(",")
        if value.strip()
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in respawn_seed_radii):
        parser.error("--ddpr-respawn-seed-radii-m values must be positive")
    shadow_seed_radii = tuple(
        float(value)
        for value in str(args.ddpr_respawn_shadow_seed_radii_m).split(",")
        if value.strip()
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in shadow_seed_radii):
        parser.error("--ddpr-respawn-shadow-seed-radii-m values must be positive")
    snapshot_shadow_radii = tuple(
        float(value)
        for value in str(args.ddpr_respawn_snapshot_shadow_radii_m).split(",")
        if value.strip()
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in snapshot_shadow_radii):
        parser.error("--ddpr-respawn-snapshot-shadow-radii-m values must be positive")
    snapshot_extrapolation_scales = tuple(
        float(value)
        for value in str(args.ddpr_respawn_snapshot_extrapolation_scales).split(",")
        if value.strip()
    )
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in snapshot_extrapolation_scales
    ):
        parser.error(
            "--ddpr-respawn-snapshot-extrapolation-scales values must be positive"
        )
    trusted_anchor_shadow_radii = tuple(
        float(value)
        for value in str(args.trusted_fix_anchor_shadow_radii_m).split(",")
        if value.strip()
    )
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in trusted_anchor_shadow_radii
    ):
        parser.error("--trusted-fix-anchor-shadow-radii-m values must be positive")
    trusted_anchor_float_line_radii = tuple(
        float(value)
        for value in str(args.trusted_fix_anchor_float_line_radii_m).split(",")
        if value.strip()
    )
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in trusted_anchor_float_line_radii
    ):
        parser.error(
            "--trusted-fix-anchor-float-line-radii-m values must be positive"
        )
    if float(args.external_position_seed_separation_m) < 0.0:
        parser.error("--external-position-seed-separation-m must be non-negative")
    if int(args.external_position_seed_max) <= 0:
        parser.error("--external-position-seed-max must be positive")
    if int(args.external_position_seed_top_k) <= 0:
        parser.error("--external-position-seed-top-k must be positive")
    if float(args.widelane_basin_sigma_m) <= 0.0:
        parser.error("--widelane-basin-sigma-m must be positive")
    if int(args.widelane_integer_min_pairs) <= 0:
        parser.error("--widelane-integer-min-pairs must be positive")
    if float(args.widelane_integer_mismatch_penalty) < 0.0:
        parser.error("--widelane-integer-mismatch-penalty must be non-negative")
    if float(args.widelane_integer_missing_penalty) < 0.0:
        parser.error("--widelane-integer-missing-penalty must be non-negative")
    external_position_seeds = (
        _load_diverse_position_seeds(
            args.external_position_seeds_csv,
            separation_m=float(args.external_position_seed_separation_m),
            max_positions=int(args.external_position_seed_max),
        )
        if args.external_position_seeds_csv is not None
        else {}
    )

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    dd_systems = tuple(x.strip() for x in str(args.dd_systems).split(",") if x.strip())
    dd_carrier_families = tuple(
        x.strip()
        for x in str(args.dd_carrier_families).split(",")
        if x.strip()
    )
    pr_systems = tuple(x.strip() for x in str(args.pr_systems).split(",") if x.strip())
    ppc_loader = PPCDatasetLoader(run_dir)
    data = ppc_loader.load_experiment_data(
        max_epochs=int(args.max_epochs),
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    wls_positions, _ = run_wls(_filter_data_by_systems(data, pr_systems))
    truth = _reference_position_map(_load_full_reference(run_dir / "reference.csv"))
    observation_cache = RinexObservationCache()
    carrier = DDCarrierComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=dd_systems,
        observation_cache=observation_cache,
    )
    pseudorange = DDPseudorangeComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=dd_systems,
        observation_cache=observation_cache,
    )
    widelane = (
        WidelaneDDPseudorangeComputer(
            run_dir / "base.obs",
            run_dir / "rover.obs",
            base_position=np.asarray(data["base_ecef"], dtype=np.float64),
            allowed_systems=tuple(
                system for system in dd_systems if system in ("G", "J")
            ),
            min_epochs=int(args.widelane_min_epochs),
            max_std_cycles=float(args.widelane_max_std_cycles),
            ratio_threshold=float(args.widelane_ratio_threshold),
            min_fix_rate=float(args.widelane_min_fix_rate),
        )
        if (
            args.enable_widelane_basin_evidence
            or args.enable_widelane_integer_score
        )
        else None
    )
    float_kf = DDFloatKalmanFilter(
        np.asarray(wls_positions[0, :3], dtype=np.float64),
        position_sigma_m=50.0,
        velocity_sigma_mps=10.0,
        accel_process_sigma_mps2=3.0,
        ambiguity_init_sigma_cycles=40.0,
        max_track_age_epochs=10,
    )
    ddpr_guard = BasinKalmanState.from_position(
        np.asarray(wls_positions[0, :3], dtype=np.float64),
        np.eye(3, dtype=np.float64) * 50.0**2,
        velocity_sigma_mps=10.0,
        accel_process_sigma_mps2=3.0,
    )
    basin_pf = AmbiguityBasinParticleFilter(
        max_basins=int(args.max_basins),
        fix_gamma_threshold=float(args.fix_gamma),
        fix_min_streak=int(args.fix_streak),
        min_fixed_ambiguities=int(args.subset_size),
        diversity_reserve_fraction=float(args.basin_diversity_reserve_fraction),
        diversity_radius_m=float(args.basin_diversity_radius_m),
        dedup_position_radius_m=float(args.basin_dedup_position_radius_m),
        source_reserve_fraction=float(args.basin_source_reserve_fraction),
        protected_source_token=str(args.basin_protected_source_token),
        protected_source_fraction=float(args.basin_protected_source_fraction),
    )
    policy_config = TrustedFixPolicyConfig(
        gamma_threshold=float(args.fix_gamma),
        min_streak=int(args.fix_streak),
        min_ambiguities=int(args.subset_size),
        max_float_separation_m=float(args.fix_consistency_m),
        max_ddpr_separation_m=float(args.fix_ddpr_consistency_m),
        min_ddpr_pairs=int(args.fix_min_dd_pairs),
        max_ddpr_age_epochs=int(args.fix_max_ddpr_age_epochs),
    )
    commit_policy = TrustedFixCommitPolicy(policy_config)
    evidence_ledger = EvidenceLedger()
    traces: list[RTKEpochTrace] = []
    basin_trace_rows = (
        _StreamingCsvRows(args.out_basin_trace, _BASIN_TRACE_FIELDS)
        if args.out_basin_trace is not None
        else None
    )
    integrity_satellite_rows: list[dict[str, object]] = []
    temporal_filter = (
        TemporalAmbiguityFilter(
            TemporalAmbiguityConfig(
                birth_mass=float(args.temporal_birth_mass),
                assignment_change_cost=float(args.temporal_change_cost),
                incompatible_cost=float(args.temporal_incompatible_cost),
                death_cost=float(args.temporal_death_cost),
                motion_sigma_m=float(args.temporal_motion_sigma_m),
            )
        )
        if args.enable_temporal_lineage else None
    )
    integrity_filter = (
        TemporalAmbiguityFilter(
            TemporalAmbiguityConfig(
                birth_mass=float(args.temporal_birth_mass),
                assignment_change_cost=float(args.temporal_change_cost),
                incompatible_cost=float(args.temporal_incompatible_cost),
                death_cost=float(args.temporal_death_cost),
                motion_sigma_m=float(args.temporal_motion_sigma_m),
            )
        )
        if args.enable_integrity_lineage else None
    )
    system_id_map = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4}
    integrity_tdcp_system_ids = {
        system_id_map[value.strip()]
        for value in str(args.integrity_tdcp_systems).split(",")
        if value.strip() in system_id_map
    }
    previous_tdcp_measurements = None
    integrity_satellite_cost_state: dict[str, float] = {}
    recovery_position_bank = (
        RecoveryPositionBank(
            max_seeds=int(args.ddpr_respawn_history_seeds),
            separation_m=float(args.ddpr_respawn_history_separation_m),
            max_age_epochs=int(args.ddpr_respawn_history_max_age_epochs),
            selection_mode=str(args.ddpr_respawn_history_selection),
        )
        if int(args.ddpr_respawn_history_seeds) > 0
        else None
    )
    times = np.asarray(data["times"], dtype=np.float64)
    trusted_anchor_imu_guide = None
    trusted_anchor_imu_heading_filter = None
    trusted_anchor_imu_times = None
    trusted_anchor_imu_accel = None
    trusted_anchor_imu_gyro = None
    trusted_anchor_imu_dt = None
    trusted_anchor_ecef_to_lla_rad = None
    trusted_anchor_ecef_to_enu_rotation = None
    if args.trusted_fix_anchor_motion_fallback == "imu-preint":
        from gnss_gpu.imu import ComplementaryHeadingFilter
        from gnss_gpu.pf_imu_preint_adapter import (
            ImuPreintPfGuide,
            ecef_to_enu_rotation,
            ecef_to_lla_rad,
        )

        imu_data = ppc_loader.load_imu()
        trusted_anchor_imu_times = np.asarray(imu_data["time"], dtype=np.float64)
        trusted_anchor_imu_accel = np.column_stack(
            [imu_data["acc_x"], imu_data["acc_y"], imu_data["acc_z"]]
        ).astype(np.float64)
        trusted_anchor_imu_gyro = np.column_stack(
            [imu_data["gyro_x"], imu_data["gyro_y"], imu_data["gyro_z"]]
        ).astype(np.float64) * (math.pi / 180.0)
        imu_dt_forward = np.diff(trusted_anchor_imu_times)
        trusted_anchor_imu_dt = (
            np.concatenate([imu_dt_forward, imu_dt_forward[-1:]])
            if imu_dt_forward.size
            else np.zeros(0, dtype=np.float64)
        )
        trusted_anchor_imu_heading_filter = ComplementaryHeadingFilter(
            {
                "tow": trusted_anchor_imu_times,
                "gyro": trusted_anchor_imu_gyro,
                "wheel_vel": np.full(trusted_anchor_imu_times.size, np.nan),
            },
            alpha=0.05,
        )
        trusted_anchor_imu_guide = ImuPreintPfGuide(
            trusted_anchor_imu_heading_filter,
            velocity_blend_alpha=float(
                args.trusted_fix_anchor_imu_velocity_blend_alpha
            ),
            use_heading_uncertainty=True,
        )
        trusted_anchor_ecef_to_lla_rad = ecef_to_lla_rad
        trusted_anchor_ecef_to_enu_rotation = ecef_to_enu_rotation
    respawn_subset_size = (
        int(args.ddpr_respawn_subset_size)
        if int(args.ddpr_respawn_subset_size) > 0
        else int(args.subset_size)
    )
    respawn_top_k = (
        int(args.ddpr_respawn_top_k)
        if int(args.ddpr_respawn_top_k) > 0
        else int(args.top_k)
    )
    trusted_anchor_shadow_subset_size = (
        int(args.trusted_anchor_shadow_subset_size)
        if int(args.trusted_anchor_shadow_subset_size) > 0
        else respawn_subset_size
    )
    recovery_assignment_bank = (
        RecoveryAssignmentBank(
            max_assignments=int(args.ddpr_respawn_assignment_history),
            max_age_epochs=int(args.ddpr_respawn_assignment_history_max_age_epochs),
            min_assignment_size=int(respawn_subset_size),
        )
        if int(args.ddpr_respawn_assignment_history) > 0
        else None
    )
    satellite_arc_tracker = (
        SatelliteArcTracker(
            slip_threshold_cycles=float(
                args.ddpr_respawn_assignment_arc_slip_threshold_cycles
            ),
            max_gap_epochs=int(args.ddpr_respawn_assignment_arc_max_gap_epochs),
        )
        if (
            args.ddpr_respawn_assignment_arc_shadow
            or args.ddpr_respawn_assignment_arc_promote
        )
        else None
    )
    recovery_arc_assignment_bank = (
        RecoveryArcAssignmentBank(
            max_assignments=int(args.ddpr_respawn_assignment_history),
            max_age_epochs=int(args.ddpr_respawn_assignment_history_max_age_epochs),
            min_assignment_size=int(respawn_subset_size),
        )
        if (
            args.ddpr_respawn_assignment_arc_shadow
            or args.ddpr_respawn_assignment_arc_promote
        )
        and int(args.ddpr_respawn_assignment_history) > 0
        else None
    )
    basin_ddpr_sigma = (
        float(args.sigma_basin_dd_pr_m)
        if float(args.sigma_basin_dd_pr_m) > 0.0
        else float(args.sigma_dd_pr_m)
    )
    rows: list[dict[str, object]] = []
    n_birth_epochs = 0
    n_declared_fix = 0
    n_false_fix = 0
    n_correct_fix = 0
    n_gamma_fix = 0
    n_consistency_reject = 0
    n_respawn_epochs = 0
    n_completion_shadow_epochs = 0
    n_completion_shadow_correct = 0
    n_position_shadow_epochs = 0
    n_position_shadow_correct = 0
    n_snapshot_loo_shadow_epochs = 0
    n_snapshot_loo_shadow_correct = 0
    n_trusted_anchor_shadow_epochs = 0
    n_trusted_anchor_shadow_correct = 0
    n_external_position_shadow_epochs = 0
    n_external_position_shadow_correct = 0
    n_trusted_anchor_snapshot_resets = 0
    n_trusted_refinement_shadow_epochs = 0
    n_trusted_refinement_shadow_correct = 0
    n_subset_shadow_epochs = 0
    n_subset_shadow_correct = 0
    n_float_resets = 0
    n_assignment_history_clears = 0
    n_arc_slips = 0
    n_arc_shadow_epochs = 0
    n_arc_shadow_correct = 0
    total_arc_shadow_compute_seconds = 0.0
    max_arc_shadow_compute_seconds = 0.0
    n_stale_generation_holdover_basins = 0
    n_temporal_map_sub50 = 0
    n_temporal_map_disagreement = 0
    max_temporal_gamma = 0.0
    n_integrity_map_sub50 = 0
    n_integrity_map_disagreement = 0
    n_integrity_anchor_epochs = 0
    n_integrity_tdcp_intervals = 0
    n_widelane_evidence_epochs = 0
    n_widelane_integer_score_epochs = 0
    n_integrity_satellite_exclusions = 0
    n_basin_oracle_sub50 = 0
    n_integrity_ball_gamma99 = 0
    n_integrity_ball_gamma99_correct = 0
    n_integrity_guard_pass = 0
    n_integrity_guard_pass_correct = 0
    max_integrity_gamma = 0.0
    max_integrity_ball_gamma = 0.0
    max_gamma = 0.0
    last_ddpr_epoch = -1_000_000
    last_ddpr_pairs = 0
    last_ddpr_nis = float("nan")
    arc_reference_position = None
    trusted_fix_anchor_position = None
    trusted_fix_anchor_age_epochs = 0
    trusted_fix_anchor_last_tdcp_velocity = None
    trusted_fix_anchor_doppler_bias_samples: list[np.ndarray] = []
    previous_scoring_ref = None

    epoch_compute_seconds: list[float] = []
    lambda_batch_calls = 0
    lambda_batch_problems = 0
    lambda_batch_compute_seconds = 0.0
    rss_samples: list[int] = []
    try:
        import psutil

        runtime_process = psutil.Process()
    except ImportError:
        runtime_process = None

    epoch_loop_start = time.perf_counter()
    for i, tow in enumerate(times):
        epoch_compute_start = time.perf_counter()
        evidence_start = len(evidence_ledger)
        observation_id = f"tow={float(tow):.3f}"
        epoch_dt = 0.0
        integrity_tdcp = None
        trusted_anchor_imu_displacement = None
        current_tdcp_measurements = None
        if i > 0:
            epoch_dt = max(float(times[i] - times[i - 1]), 1e-3)
            float_kf.predict(epoch_dt)
            basin_pf.predict(epoch_dt)
            ddpr_guard.predict(epoch_dt)
        if (
            integrity_filter is not None
            or args.ddpr_respawn_history_propagate_tdcp
            or satellite_arc_tracker is not None
            or args.ddpr_respawn_trusted_fix_anchor_shadow_only
        ):
            current_tdcp_measurements = [
                measurement
                for measurement in _epoch_measurements(data, i)
                if int(measurement.system_id) in integrity_tdcp_system_ids
            ]
            if i > 0 and previous_tdcp_measurements is not None:
                integrity_tdcp = estimate_displacement_from_tdcp(
                    float_kf.position_ecef,
                    previous_tdcp_measurements,
                    current_tdcp_measurements,
                    epoch_dt,
                    min_sats=int(args.integrity_tdcp_min_sats),
                    max_postfit_rms_m=float(args.integrity_tdcp_max_postfit_rms_m),
                    slip_residual_threshold_m=float(
                        args.integrity_tdcp_slip_threshold_m
                    ),
                )
                n_integrity_tdcp_intervals += int(integrity_tdcp is not None)
            previous_tdcp_measurements = current_tdcp_measurements
        velocity, doppler_rms = _doppler_velocity(data, i, float_kf.position_ecef)
        if velocity is not None:
            velocity_sigma = max(0.5, min(float(doppler_rms), 5.0))
            float_kf.update_velocity(velocity, sigma_mps=velocity_sigma)
            basin_velocity_evidence = (
                basin_pf.update_velocity(velocity, sigma_mps=velocity_sigma)
                if basin_pf.basins else None
            )
            ddpr_guard.update_velocity(velocity, sigma_mps=velocity_sigma)
            for target in ("float_kf", "ddpr_guard"):
                evidence_ledger.record(
                    epoch=i,
                    target=target,
                    source="doppler_velocity",
                    observation_id=observation_id,
                    n_rows=3,
                )
            if basin_velocity_evidence is not None:
                evidence_ledger.record(
                    epoch=i,
                    target="basin_pf",
                    source="doppler_velocity",
                    observation_id=observation_id,
                    n_rows=3,
                    log_evidence=basin_velocity_evidence.log_marginal,
                )
        if trusted_anchor_imu_guide is not None and i > 0:
            trusted_anchor_imu_heading_filter.update_heading_gyro(
                float(times[i - 1]), float(tow)
            )
            imu_start = int(
                np.searchsorted(
                    trusted_anchor_imu_times, float(times[i - 1]), side="left"
                )
            )
            imu_stop = int(
                np.searchsorted(trusted_anchor_imu_times, float(tow), side="left")
            )
            for imu_index in range(imu_start, imu_stop):
                trusted_anchor_imu_guide.add_sample(
                    trusted_anchor_imu_accel[imu_index],
                    trusted_anchor_imu_gyro[imu_index],
                    float(trusted_anchor_imu_dt[imu_index]),
                )
            imu_heading = None
            if velocity is not None:
                imu_lat, imu_lon = trusted_anchor_ecef_to_lla_rad(
                    float_kf.position_ecef
                )
                imu_velocity_enu = trusted_anchor_ecef_to_enu_rotation(
                    imu_lat, imu_lon
                ) @ velocity
                if math.hypot(
                    float(imu_velocity_enu[0]), float(imu_velocity_enu[1])
                ) > 0.5:
                    imu_heading = math.atan2(
                        float(imu_velocity_enu[0]), float(imu_velocity_enu[1])
                    )
            imu_velocity_guide, _imu_sigma = trusted_anchor_imu_guide.close_segment(
                float_kf.position_ecef,
                epoch_dt,
                v_gnss_ecef=velocity,
                spp_heading_rad=imu_heading,
                spp_displacement_m=(
                    None
                    if velocity is None
                    else float(np.linalg.norm(velocity * epoch_dt))
                ),
            )
            trusted_anchor_imu_guide.reset_segment()
            if imu_velocity_guide is not None:
                trusted_anchor_imu_displacement = imu_velocity_guide * epoch_dt
        if integrity_tdcp is not None and epoch_dt > 0.0:
            trusted_fix_anchor_last_tdcp_velocity = (
                integrity_tdcp.displacement_ecef_m / epoch_dt
            )
            if velocity is not None:
                trusted_fix_anchor_doppler_bias_samples.append(
                    velocity - trusted_fix_anchor_last_tdcp_velocity
                )
                bias_window = max(
                    1, int(args.trusted_fix_anchor_doppler_bias_window)
                )
                trusted_fix_anchor_doppler_bias_samples = (
                    trusted_fix_anchor_doppler_bias_samples[-bias_window:]
                )
        if trusted_fix_anchor_position is not None and i > 0:
            if integrity_tdcp is not None:
                trusted_fix_anchor_position = (
                    trusted_fix_anchor_position
                    + integrity_tdcp.displacement_ecef_m
                )
            elif (
                args.trusted_fix_anchor_motion_fallback == "last-tdcp"
                and trusted_fix_anchor_last_tdcp_velocity is not None
            ):
                trusted_fix_anchor_position = (
                    trusted_fix_anchor_position
                    + trusted_fix_anchor_last_tdcp_velocity * epoch_dt
                )
            elif (
                args.trusted_fix_anchor_motion_fallback == "imu-preint"
                and trusted_anchor_imu_displacement is not None
            ):
                trusted_fix_anchor_position = (
                    trusted_fix_anchor_position
                    + trusted_anchor_imu_displacement
                )
            elif (
                args.trusted_fix_anchor_motion_fallback == "doppler-calibrated"
                and velocity is not None
                and trusted_fix_anchor_doppler_bias_samples
            ):
                doppler_bias = np.median(
                    np.asarray(trusted_fix_anchor_doppler_bias_samples), axis=0
                )
                trusted_fix_anchor_position = (
                    trusted_fix_anchor_position
                    + (velocity - doppler_bias) * epoch_dt
                )
            elif (
                args.trusted_fix_anchor_motion_fallback == "doppler"
                and velocity is not None
            ):
                trusted_fix_anchor_position = (
                    trusted_fix_anchor_position + velocity * epoch_dt
                )
            trusted_fix_anchor_age_epochs += 1

        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][i], dtype=np.float64),
            np.asarray(data["system_ids"][i], dtype=np.int32),
            list(data["used_prns"][i]),
            np.asarray(data["weights"][i], dtype=np.float64),
            float_kf.position_ecef,
            dd_systems,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        dd_pr = pseudorange.compute_dd(
            float(tow), measurements, rover_position_approx=float_kf.position_ecef, min_common_sats=4
        )
        if dd_carrier_families:
            dd_cp = carrier.compute_dd_families(
                float(tow),
                measurements,
                rover_position_approx=float_kf.position_ecef,
                min_common_sats=4,
                carrier_families=dd_carrier_families,
            )
        else:
            dd_cp = carrier.compute_dd(
                float(tow),
                measurements,
                rover_position_approx=float_kf.position_ecef,
                min_common_sats=4,
            )
        wl_dd_pr = None
        wl_stats = WidelaneDDStats(reason="disabled")
        if widelane is not None:
            wl_dd_pr, wl_stats = widelane.compute_dd(
                float(tow),
                measurements,
                rover_position_approx=float_kf.position_ecef,
                min_common_sats=4,
                rover_weights=np.asarray(data["weights"][i], dtype=np.float64),
            )
            n_widelane_evidence_epochs += int(wl_dd_pr is not None)
        pr_diag = None
        cp_diag = None
        ddpr_snapshot_position = None
        ddpr_snapshot_diagnostics = None
        ddpr_snapshot_loo_positions: list[np.ndarray] = []
        ddpr_snapshot_pair_exclusion_positions: list[np.ndarray] = []
        if dd_pr is not None and int(dd_pr.n_dd) >= 3:
            pr_diag = float_kf.update_pseudorange(
                dd_pr, sigma_pr_m=float(args.sigma_dd_pr_m)
            )
            ddpr_guard.update_pseudorange(dd_pr, sigma_pr_m=float(args.sigma_dd_pr_m))
            last_ddpr_epoch = i
            last_ddpr_pairs = int(dd_pr.n_dd)
            last_ddpr_nis = float(pr_diag.normalized_innovation_sq)
            for target in ("float_kf", "ddpr_guard"):
                evidence_ledger.record(
                    epoch=i,
                    target=target,
                    source="dd_pseudorange",
                    observation_id=observation_id,
                    n_rows=int(dd_pr.n_dd),
                )
            if (
                args.ddpr_respawn_snapshot_seed_shadow_only
                or args.ddpr_respawn_snapshot_seed_promote
            ):
                snapshot_dd_pr = dd_pr
                if float(args.ddpr_snapshot_pair_residual_max_m) > 0.0:
                    snapshot_dd_pr, _snapshot_gate = gate_dd_pseudorange(
                        dd_pr,
                        np.asarray(wls_positions[i, :3], dtype=np.float64),
                        pair_residual_max_m=float(
                            args.ddpr_snapshot_pair_residual_max_m
                        ),
                        min_pairs=3,
                    )
                if snapshot_dd_pr is not None:
                    ddpr_snapshot_position, ddpr_snapshot_diagnostics = (
                        dd_pseudorange_position_update(
                            np.asarray(wls_positions[i, :3], dtype=np.float64),
                            snapshot_dd_pr,
                            DDWLSConfig(
                                min_dd_pairs=3,
                                dd_sigma_m=float(args.sigma_dd_pr_m),
                                prior_sigma_m=float(args.ddpr_snapshot_prior_sigma_m),
                                max_shift_m=float(args.ddpr_snapshot_max_shift_m),
                                max_iter=8,
                            ),
                        )
                    )
                    if (
                        trusted_fix_anchor_position is not None
                        and float(args.trusted_fix_anchor_snapshot_reset_rms_m) > 0.0
                        and bool(ddpr_snapshot_diagnostics.get("accepted", False))
                        and float(ddpr_snapshot_diagnostics["final_rms_m"])
                        <= float(args.trusted_fix_anchor_snapshot_reset_rms_m)
                    ):
                        trusted_fix_anchor_position = ddpr_snapshot_position.copy()
                        trusted_fix_anchor_age_epochs = 0
                        n_trusted_anchor_snapshot_resets += 1
                if args.ddpr_respawn_snapshot_loo_shadow_only:
                    for excluded_satellite in sorted(set(dd_pr.sat_ids)):
                        loo_mask = np.asarray(
                            [
                                str(satellite) != excluded_satellite
                                for satellite in dd_pr.sat_ids
                            ],
                            dtype=bool,
                        )
                        if int(np.count_nonzero(loo_mask)) < 3:
                            continue
                        loo_dd_pr = _subset_dd_result(dd_pr, loo_mask)
                        loo_position, loo_diagnostics = dd_pseudorange_position_update(
                            np.asarray(wls_positions[i, :3], dtype=np.float64),
                            loo_dd_pr,
                            DDWLSConfig(
                                min_dd_pairs=3,
                                dd_sigma_m=float(args.sigma_dd_pr_m),
                                prior_sigma_m=float(args.ddpr_snapshot_prior_sigma_m),
                                max_shift_m=float(args.ddpr_snapshot_max_shift_m),
                                max_iter=8,
                            ),
                        )
                        if bool(loo_diagnostics.get("accepted", False)):
                            ddpr_snapshot_loo_positions.append(loo_position)
                if args.ddpr_snapshot_pair_exclusion_position_shadow_only:
                    nonpivot_satellites = sorted(set(dd_pr.sat_ids))
                    for excluded_pair in itertools.combinations(nonpivot_satellites, 2):
                        excluded = set(excluded_pair)
                        pair_mask = np.asarray(
                            [str(satellite) not in excluded for satellite in dd_pr.sat_ids],
                            dtype=bool,
                        )
                        if int(np.count_nonzero(pair_mask)) < 3:
                            continue
                        pair_dd_pr = _subset_dd_result(dd_pr, pair_mask)
                        pair_position, pair_diagnostics = dd_pseudorange_position_update(
                            np.asarray(wls_positions[i, :3], dtype=np.float64),
                            pair_dd_pr,
                            DDWLSConfig(
                                min_dd_pairs=3,
                                dd_sigma_m=float(args.sigma_dd_pr_m),
                                prior_sigma_m=float(args.ddpr_snapshot_prior_sigma_m),
                                max_shift_m=float(args.ddpr_snapshot_max_shift_m),
                                max_iter=8,
                            ),
                        )
                        if bool(pair_diagnostics.get("accepted", False)):
                            ddpr_snapshot_pair_exclusion_positions.append(pair_position)
        if dd_cp is not None and int(dd_cp.n_dd) >= 3:
            cp_diag = float_kf.update_carrier(
                dd_cp,
                dd_pseudorange_result=dd_pr,
                sigma_cp_cycles=float(args.sigma_float_cp_cycles),
                slip_threshold_cycles=float(args.float_slip_threshold_cycles),
            )
            n_float_resets += int(cp_diag.ambiguities_reset)
            evidence_ledger.record(
                epoch=i,
                target="float_kf",
                source="dd_carrier",
                observation_id=observation_id,
                n_rows=int(dd_cp.n_dd),
            )

        generations = float_kf.ambiguity_generations()
        epoch_arc_slips = 0
        epoch_arc_slip_ids = ""
        if satellite_arc_tracker is not None and dd_cp is not None:
            if arc_reference_position is None:
                arc_reference_position = float_kf.position_ecef.copy()
            elif integrity_tdcp is not None:
                arc_reference_position = (
                    arc_reference_position + integrity_tdcp.displacement_ecef_m
                )
            elif epoch_dt > 0.0:
                arc_reference_position = (
                    arc_reference_position + float_kf.velocity_ecef * epoch_dt
                )
            arc_seed = ddpr_centered_ambiguity_seed(
                dd_cp,
                arc_reference_position,
                float_kf.covariance[:3, :3],
                sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
            )
            arc_slips = satellite_arc_tracker.update(
                i, arc_seed.keys, arc_seed.ahat_cycles
            )
            epoch_arc_slips = len(arc_slips)
            epoch_arc_slip_ids = ",".join(
                f"{satellite}@{wavelength}" for satellite, wavelength in arc_slips
            )
            n_arc_slips += epoch_arc_slips
        assignment_history_cleared = False
        if (
            recovery_assignment_bank is not None
            and args.ddpr_respawn_assignment_pivot_rebase
            and cp_diag is not None
            and int(cp_diag.ambiguities_reset) > 0
        ):
            recovery_assignment_bank.clear()
            assignment_history_cleared = True
            n_assignment_history_clears += 1
        if recovery_position_bank is not None and basin_pf.basins:
            recovery_position_bank.update(
                i,
                np.asarray(
                    [basin.conditional.mean[:3] for basin in basin_pf.basins],
                    dtype=np.float64,
                ),
                np.asarray(
                    [basin.log_weight for basin in basin_pf.basins],
                    dtype=np.float64,
                ),
                velocities_ecef=(
                    np.asarray(
                        [basin.conditional.mean[3:6] for basin in basin_pf.basins],
                        dtype=np.float64,
                    )
                    if args.ddpr_respawn_history_propagate_velocity
                    else None
                ),
                dt_seconds=(
                    float(epoch_dt)
                    if args.ddpr_respawn_history_propagate_velocity
                    else 0.0
                ),
                displacement_ecef_m=(
                    integrity_tdcp.displacement_ecef_m
                    if args.ddpr_respawn_history_propagate_tdcp
                    and integrity_tdcp is not None
                    else None
                ),
                reference_position_ecef=ddpr_guard.mean[:3],
                max_reference_distance_m=float(
                    args.ddpr_respawn_history_max_guard_distance_m
                ),
            )
        active_versioned = {(key, generation) for key, generation in generations.items()}
        basin_pf.retain_compatible(active_versioned)
        stale_generation_holdover_basins = sum(
            any(key not in active_versioned for key, _value in basin.assignment)
            for basin in basin_pf.basins
        )
        n_stale_generation_holdover_basins += stale_generation_holdover_basins
        if stale_generation_holdover_basins:
            raise RuntimeError(
                "ambiguity basin survived an incompatible generation reset"
            )
        if dd_pr is not None and basin_pf.basins:
            basin_pr_evidence = basin_pf.update_pseudorange(
                dd_pr, sigma_pr_m=basin_ddpr_sigma
            )
            evidence_ledger.record(
                epoch=i,
                target="basin_pf",
                source="dd_pseudorange",
                observation_id=observation_id,
                n_rows=int(dd_pr.n_dd),
                log_evidence=basin_pr_evidence.log_marginal,
            )
        if wl_dd_pr is not None and basin_pf.basins:
            widelane_evidence = basin_pf.update_pseudorange(
                wl_dd_pr,
                sigma_pr_m=float(args.widelane_basin_sigma_m),
            )
            evidence_ledger.record(
                epoch=i,
                target="basin_pf",
                source="widelane_dd_pseudorange",
                observation_id=observation_id,
                n_rows=int(wl_dd_pr.n_dd),
                log_evidence=widelane_evidence.log_marginal,
            )
        if dd_cp is not None and basin_pf.basins:
            basin_cp_evidence = basin_pf.update_fixed_carrier(
                dd_cp,
                generations,
                sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
            )
            evidence_ledger.record(
                epoch=i,
                target="basin_pf",
                source="dd_carrier",
                observation_id=observation_id,
                n_rows=int(dd_cp.n_dd),
                log_evidence=basin_cp_evidence.log_marginal,
            )

        pre_birth_map = (
            max(basin_pf.basins, key=lambda basin: basin.log_weight)
            if basin_pf.basins else None
        )
        respawn_triggered = bool(
            args.enable_ddpr_respawn
            and dd_cp is not None
            and dd_pr is not None
            and int(dd_pr.n_dd) >= int(args.fix_min_dd_pairs)
            and (
                pre_birth_map is None
                or np.linalg.norm(
                    pre_birth_map.conditional.mean[:3] - ddpr_guard.mean[:3]
                ) > float(args.ddpr_respawn_trigger_m)
            )
        )

        n_candidates = 0
        n_respawn_candidates = 0
        n_respawn_position_seeds = 0
        n_respawn_history_seeds = 0
        n_respawn_assignment_candidates = 0
        respawn_oracle_min_error = float("nan")
        respawn_oracle_rank = -1
        completion_shadow_candidates = 0
        completion_shadow_oracle_min_error = float("nan")
        position_shadow_candidates = 0
        position_shadow_oracle_min_error = float("nan")
        snapshot_loo_shadow_candidates = 0
        snapshot_loo_shadow_oracle_min_error = float("nan")
        trusted_anchor_shadow_candidates = 0
        trusted_anchor_shadow_oracle_min_error = float("nan")
        trusted_anchor_shadow_states: list[BasinKalmanState] = []
        external_position_shadow_candidates = 0
        external_position_shadow_oracle_min_error = float("nan")
        trusted_refinement_shadow_candidates = 0
        trusted_refinement_shadow_oracle_min_error = float("nan")
        subset_shadow_candidates = 0
        subset_shadow_oracle_min_error = float("nan")
        arc_shadow_candidates = 0
        arc_shadow_oracle_min_error = float("nan")
        arc_shadow_oracle_rank = -1
        arc_shadow_compute_seconds = 0.0
        arc_completion_search_workspaces = 0
        respawn_excluded_satellite = ""
        if dd_cp is not None and int(dd_cp.n_dd) >= int(args.subset_size):
            current_pairs = {
                (str(ref), str(sat)) for ref, sat in zip(dd_cp.ref_sat_ids, dd_cp.sat_ids)
            }
            current_keys = tuple(
                key for key in float_kf.ambiguity_seed().keys if key[:2] in current_pairs
            )
            seed = float_kf.ambiguity_seed(current_keys)
            if len(seed.keys) >= int(args.subset_size):
                order, _ranked = _select_ambiguity_indices(
                    seed.keys,
                    seed.qahat_cycles2,
                    np.arange(len(seed.keys), dtype=np.int64),
                    int(args.subset_size),
                    prefer_multifrequency_pairs=bool(
                        args.prefer_paired_multifrequency_subset
                    ),
                )
                keys = tuple(seed.keys[j] for j in order)
                ahat = seed.ahat_cycles[order]
                qahat = seed.qahat_cycles2[np.ix_(order, order)]
                candidates, residuals = integer_search(
                    ahat, qahat, n_candidates=int(args.top_k)
                )
                assignments = []
                conditionals = []
                for candidate in candidates:
                    position, covariance, _distance = float_kf.condition_position_on_integers(
                        keys, candidate
                    )
                    assignment = {
                        (key, generations[key]): int(value)
                        for key, value in zip(keys, candidate)
                    }
                    assignments.append(assignment)
                    conditionals.append(
                        BasinKalmanState.from_position(
                            position,
                            covariance,
                            velocity_ecef=float_kf.velocity_ecef,
                            velocity_sigma_mps=1.0,
                            accel_process_sigma_mps2=3.0,
                        )
                    )
                if assignments:
                    basin_pf.spawn(
                        assignments,
                        conditionals,
                        prior_mass=(1.0 if not basin_pf.basins else float(args.birth_mass)),
                    )
                    n_candidates = len(assignments)
                    n_birth_epochs += 1

        if respawn_triggered and dd_cp is not None:
            if args.ddpr_respawn_exclude_max_cost_satellite and dd_pr is not None:
                respawn_satellite_cost = satellite_pair_costs(
                    dd_pr,
                    ddpr_guard.mean[:3],
                    scale_m=float(args.integrity_scale_m),
                )
                respawn_excluded_satellite = max(
                    zip(
                        respawn_satellite_cost.satellite_ids,
                        respawn_satellite_cost.mean_pair_costs,
                    ),
                    key=lambda item: item[1],
                )[0]
            respawn_positions = covariance_axis_position_seeds(
                ddpr_guard.mean[:3],
                ddpr_guard.covariance[:3, :3],
                respawn_seed_radii,
                direction_mode=str(args.ddpr_respawn_seed_directions),
            )
            snapshot_promoted_position_index: int | None = None
            if (
                args.ddpr_respawn_snapshot_seed_promote
                and ddpr_snapshot_position is not None
                and ddpr_snapshot_diagnostics is not None
                and bool(ddpr_snapshot_diagnostics.get("accepted", False))
            ):
                respawn_positions, snapshot_promoted_position_index = (
                    _append_distinct_position_seed(
                        respawn_positions, ddpr_snapshot_position
                    )
                )
            if recovery_position_bank is not None:
                respawn_position_list = list(respawn_positions)
                for history_position in recovery_position_bank.positions:
                    if all(
                        np.linalg.norm(history_position - existing) > 1.0e-3
                        for existing in respawn_position_list
                    ):
                        respawn_position_list.append(history_position)
                        n_respawn_history_seeds += 1
                respawn_positions = tuple(respawn_position_list)
            n_respawn_position_seeds = len(respawn_positions)
            assignments = []
            conditionals = []
            respawn_source_ids: list[str] = []
            all_respawn_residuals: list[float] = []
            completion_shadow_states: list[BasinKalmanState] = []
            position_shadow_states: list[BasinKalmanState] = []
            snapshot_loo_shadow_states: list[BasinKalmanState] = []
            subset_shadow_states: list[BasinKalmanState] = []
            arc_shadow_states: list[BasinKalmanState] = []
            arc_ranked_proposals: list[tuple[dict, float]] = []
            if recovery_assignment_bank is not None:
                assignment_seed = ddpr_centered_ambiguity_seed(
                    dd_cp,
                    ddpr_guard.mean[:3],
                    ddpr_guard.covariance[:3, :3],
                    sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
                )
                observed_assignment_keys = tuple(
                    key
                    for key in assignment_seed.keys
                    if respawn_excluded_satellite not in key[:2]
                )
                if (
                    recovery_arc_assignment_bank is not None
                    and satellite_arc_tracker is not None
                ):
                    arc_compute_start = time.perf_counter()
                    arc_completion_top_k = int(
                        args.ddpr_respawn_assignment_arc_completion_top_k
                    )
                    arc_replays = recovery_arc_assignment_bank.compatible_assignments(
                        active_versioned,
                        observed_assignment_keys,
                        satellite_arc_tracker.generations,
                        min_size=(
                            int(args.ddpr_respawn_assignment_completion_min_stable)
                            if arc_completion_top_k > 0
                            else None
                        ),
                    )
                    arc_candidates: dict[
                        tuple[tuple[tuple[tuple[str, str, int], int], int], ...],
                        tuple[dict, float],
                    ] = {}
                    arc_completion_search_cache = {}
                    for arc_assignment in arc_replays:
                        if (
                            arc_completion_top_k > 0
                            and len(arc_assignment) < respawn_subset_size
                        ):
                            completed = complete_versioned_assignment(
                                assignment_seed.keys,
                                generations,
                                assignment_seed.ahat_cycles,
                                assignment_seed.qahat_cycles2,
                                arc_assignment,
                                target_size=respawn_subset_size,
                                n_candidates=arc_completion_top_k,
                                search_cache=arc_completion_search_cache,
                            )
                            per_assignment_limit = int(
                                args.ddpr_respawn_assignment_arc_completion_per_assignment
                            )
                            if per_assignment_limit > 0:
                                completed = completed[:per_assignment_limit]
                            for assignment, distance in completed:
                                canonical = tuple(sorted(assignment.items()))
                                previous = arc_candidates.get(canonical)
                                if previous is None or float(distance) < previous[1]:
                                    arc_candidates[canonical] = (
                                        assignment,
                                        float(distance),
                                    )
                        else:
                            arc_keys = tuple(key[0] for key in arc_assignment)
                            arc_integers = np.asarray(
                                [arc_assignment[key] for key in arc_assignment],
                                dtype=np.float64,
                            )
                            _position, _covariance, distance = condition_respawn_position(
                                assignment_seed, arc_keys, arc_integers
                            )
                            arc_candidates.setdefault(
                                tuple(sorted(arc_assignment.items())),
                                (arc_assignment, float(distance)),
                            )
                    ranked_arc_candidates = sorted(
                        arc_candidates.values(), key=lambda item: item[1]
                    )
                    arc_shadow_limit = int(
                        args.ddpr_respawn_assignment_arc_shadow_max_candidates
                    )
                    if arc_shadow_limit > 0:
                        ranked_arc_candidates = ranked_arc_candidates[:arc_shadow_limit]
                    arc_ranked_proposals = list(ranked_arc_candidates)
                    for arc_assignment, _proposal_distance in ranked_arc_candidates:
                        arc_keys = tuple(key[0] for key in arc_assignment)
                        arc_integers = np.asarray(
                            [arc_assignment[key] for key in arc_assignment],
                            dtype=np.float64,
                        )
                        position, covariance, _distance = condition_respawn_position(
                            assignment_seed, arc_keys, arc_integers
                        )
                        arc_shadow_states.append(
                            BasinKalmanState.from_position(position, covariance)
                        )
                    arc_completion_search_workspaces = len(
                        arc_completion_search_cache
                    )
                    arc_shadow_compute_seconds = (
                        time.perf_counter() - arc_compute_start
                    )
                    total_arc_shadow_compute_seconds += arc_shadow_compute_seconds
                    max_arc_shadow_compute_seconds = max(
                        max_arc_shadow_compute_seconds,
                        arc_shadow_compute_seconds,
                    )

                def assignment_replays(
                    *, min_size: int | None = None
                ) -> tuple[dict, ...]:
                    if args.ddpr_respawn_assignment_pivot_rebase:
                        return recovery_assignment_bank.rebased_assignments(
                            active_versioned,
                            observed_assignment_keys,
                            min_size=min_size,
                        )
                    return recovery_assignment_bank.compatible_assignments(
                        active_versioned,
                        observed_assignment_keys,
                        min_size=min_size,
                    )

                completion_top_k = int(args.ddpr_respawn_assignment_completion_top_k)
                if completion_top_k > 0:
                    partial_assignments = assignment_replays(
                        min_size=int(args.ddpr_respawn_assignment_completion_min_stable),
                    )
                    completed: dict[
                        tuple[tuple[tuple[tuple[str, str, int], int], int], ...],
                        tuple[dict, float],
                    ] = {}
                    for partial_assignment in partial_assignments:
                        for completed_assignment, distance in complete_versioned_assignment(
                            assignment_seed.keys,
                            generations,
                            assignment_seed.ahat_cycles,
                            assignment_seed.qahat_cycles2,
                            partial_assignment,
                            target_size=respawn_subset_size,
                            n_candidates=completion_top_k,
                        ):
                            canonical = tuple(sorted(completed_assignment.items()))
                            previous = completed.get(canonical)
                            if previous is None or distance < previous[1]:
                                completed[canonical] = (completed_assignment, distance)
                    completion_proposals = sorted(
                        completed.values(), key=lambda item: item[1]
                    )
                    if args.ddpr_respawn_assignment_completion_shadow_only:
                        replay_proposals = [
                            (assignment, float("nan"))
                            for assignment in assignment_replays()
                        ]
                        for assignment, _distance in completion_proposals:
                            shadow_keys = tuple(key[0] for key in assignment)
                            shadow_integers = np.asarray(
                                [assignment[key] for key in assignment], dtype=np.float64
                            )
                            position, covariance, _ = condition_respawn_position(
                                assignment_seed, shadow_keys, shadow_integers
                            )
                            completion_shadow_states.append(
                                BasinKalmanState.from_position(position, covariance)
                            )
                    else:
                        replay_proposals = completion_proposals
                else:
                    if args.ddpr_respawn_assignment_arc_promote:
                        replay_proposals = arc_ranked_proposals
                    else:
                        replay_proposals = [
                            (assignment, float("nan"))
                            for assignment in assignment_replays()
                        ]
                for assignment_index, (assignment, completion_distance) in enumerate(
                    replay_proposals
                ):
                    replay_keys = tuple(key[0] for key in assignment)
                    replay_integers = np.asarray(
                        [assignment[key] for key in assignment], dtype=np.float64
                    )
                    position, covariance, distance = condition_respawn_position(
                        assignment_seed, replay_keys, replay_integers
                    )
                    if np.isfinite(completion_distance):
                        distance = completion_distance
                    assignments.append(assignment)
                    conditionals.append(
                        BasinKalmanState.from_position(
                            position,
                            covariance,
                            velocity_ecef=ddpr_guard.mean[3:6],
                            velocity_sigma_mps=1.0,
                            accel_process_sigma_mps2=3.0,
                        )
                    )
                    assignment_source = (
                        "arc_assignment"
                        if args.ddpr_respawn_assignment_arc_promote
                        else "assignment"
                    )
                    respawn_source_ids.append(
                        f"{i}:{assignment_source}:{assignment_index}"
                    )
                    all_respawn_residuals.append(float(distance))
                n_respawn_assignment_candidates = len(replay_proposals)
            n_replayed_assignments = len(assignments)
            prepared_respawns = []
            for position_index, respawn_position in enumerate(respawn_positions):
                respawn_seed = ddpr_centered_ambiguity_seed(
                    dd_cp,
                    respawn_position,
                    ddpr_guard.covariance[:3, :3],
                    sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
                )
                available = np.asarray(
                    [
                        j
                        for j, key in enumerate(respawn_seed.keys)
                        if key in generations
                        and respawn_excluded_satellite not in key[:2]
                    ],
                    dtype=np.int64,
                )
                if available.size < respawn_subset_size:
                    continue
                selected, ranked = _select_ambiguity_indices(
                    respawn_seed.keys,
                    respawn_seed.qahat_cycles2,
                    available,
                    respawn_subset_size,
                    prefer_multifrequency_pairs=bool(
                        args.prefer_paired_multifrequency_subset
                    ),
                )
                respawn_keys = tuple(respawn_seed.keys[j] for j in selected)
                prepared_respawns.append(
                    (
                        position_index,
                        respawn_seed,
                        selected,
                        ranked,
                        respawn_keys,
                        respawn_seed.ahat_cycles[selected],
                        respawn_seed.qahat_cycles2[np.ix_(selected, selected)],
                    )
                )
            batch_compute_start = time.perf_counter()
            batch_results = integer_search_batch(
                [spec[5] for spec in prepared_respawns],
                [spec[6] for spec in prepared_respawns],
                n_candidates=respawn_top_k,
                engine=str(args.lambda_engine),
            )
            if prepared_respawns:
                lambda_batch_calls += 1
                lambda_batch_problems += len(prepared_respawns)
                lambda_batch_compute_seconds += time.perf_counter() - batch_compute_start
            for spec, (candidates, seed_residuals) in zip(
                prepared_respawns, batch_results
            ):
                position_index, respawn_seed, selected, ranked, respawn_keys = spec[:5]
                for candidate in candidates:
                    position, covariance, _distance = condition_respawn_position(
                        respawn_seed, respawn_keys, candidate
                    )
                    assignments.append(
                        {
                            (key, generations[key]): int(value)
                            for key, value in zip(respawn_keys, candidate)
                        }
                    )
                    conditionals.append(
                        BasinKalmanState.from_position(
                            position,
                            covariance,
                            velocity_ecef=ddpr_guard.mean[3:6],
                            velocity_sigma_mps=1.0,
                            accel_process_sigma_mps2=3.0,
                        )
                    )
                    respawn_source_ids.append(
                        f"{i}:snapshot:{position_index}"
                        if position_index == snapshot_promoted_position_index
                        else f"{i}:{position_index}"
                    )
                all_respawn_residuals.extend(
                    float(value) for value in np.asarray(seed_residuals).reshape(-1)
                )
                shadow_subset_top_k = int(args.ddpr_respawn_shadow_one_swap_top_k)
                if shadow_subset_top_k > 0 and ranked.size > respawn_subset_size:
                    alternate = int(ranked[respawn_subset_size])
                    for dropped in selected:
                        shadow_selected = np.sort(
                            np.asarray(
                                [value for value in selected if value != dropped]
                                + [alternate],
                                dtype=np.int64,
                            )
                        )
                        shadow_keys = tuple(
                            respawn_seed.keys[j] for j in shadow_selected
                        )
                        shadow_candidates, _ = integer_search(
                            respawn_seed.ahat_cycles[shadow_selected],
                            respawn_seed.qahat_cycles2[
                                np.ix_(shadow_selected, shadow_selected)
                            ],
                            n_candidates=shadow_subset_top_k,
                        )
                        for candidate in shadow_candidates:
                            position, covariance, _ = condition_respawn_position(
                                respawn_seed, shadow_keys, candidate
                            )
                            subset_shadow_states.append(
                                BasinKalmanState.from_position(position, covariance)
                            )
                shadow_window_count = int(args.ddpr_respawn_shadow_window_count)
                shadow_window_top_k = int(args.ddpr_respawn_shadow_window_top_k)
                for offset in range(1, shadow_window_count + 1):
                    if (
                        shadow_window_top_k <= 0
                        or offset + respawn_subset_size > ranked.size
                    ):
                        break
                    shadow_selected = np.sort(
                        ranked[offset : offset + respawn_subset_size]
                    )
                    shadow_keys = tuple(respawn_seed.keys[j] for j in shadow_selected)
                    shadow_candidates, _ = integer_search(
                        respawn_seed.ahat_cycles[shadow_selected],
                        respawn_seed.qahat_cycles2[
                            np.ix_(shadow_selected, shadow_selected)
                        ],
                        n_candidates=shadow_window_top_k,
                    )
                    for candidate in shadow_candidates:
                        position, covariance, _ = condition_respawn_position(
                            respawn_seed, shadow_keys, candidate
                        )
                        subset_shadow_states.append(
                            BasinKalmanState.from_position(position, covariance)
                        )
            shadow_position_specs: list[tuple[np.ndarray, int, bool]] = []
            if shadow_seed_radii:
                shadow_position_specs.extend(
                    (position, respawn_top_k, False)
                    for position in covariance_axis_position_seeds(
                            ddpr_guard.mean[:3],
                            ddpr_guard.covariance[:3, :3],
                            shadow_seed_radii,
                            direction_mode=str(args.ddpr_respawn_seed_directions),
                        )[1:]
                )
            snapshot_shadow_top_k = (
                int(args.ddpr_respawn_snapshot_shadow_top_k)
                if int(args.ddpr_respawn_snapshot_shadow_top_k) > 0
                else respawn_top_k
            )
            if (
                ddpr_snapshot_position is not None
                and ddpr_snapshot_diagnostics is not None
                and bool(ddpr_snapshot_diagnostics.get("accepted", False))
            ):
                shadow_position_specs.append(
                    (ddpr_snapshot_position, snapshot_shadow_top_k, False)
                )
                if snapshot_shadow_radii:
                    shadow_position_specs.extend(
                        (position, snapshot_shadow_top_k, False)
                        for position in covariance_axis_position_seeds(
                                ddpr_snapshot_position,
                                ddpr_guard.covariance[:3, :3],
                                snapshot_shadow_radii,
                                direction_mode=str(args.ddpr_respawn_seed_directions),
                            )[1:]
                    )
                snapshot_step = ddpr_snapshot_position - np.asarray(
                    wls_positions[i, :3], dtype=np.float64
                )
                shadow_position_specs.extend(
                    (
                        ddpr_snapshot_position + scale * snapshot_step,
                        snapshot_shadow_top_k,
                        False,
                    )
                    for scale in snapshot_extrapolation_scales
                )
            shadow_position_specs.extend(
                (position, snapshot_shadow_top_k, True)
                for position in ddpr_snapshot_loo_positions
            )
            trusted_anchor_shadow_positions: list[np.ndarray] = []
            if (
                trusted_fix_anchor_position is not None
                and not args.trusted_fix_anchor_float_line_promote
            ):
                trusted_anchor_shadow_positions.extend(
                    covariance_axis_position_seeds(
                        trusted_fix_anchor_position,
                        ddpr_guard.covariance[:3, :3],
                        trusted_anchor_shadow_radii,
                        direction_mode=str(args.ddpr_respawn_seed_directions),
                    )
                )
                float_line = float_kf.position_ecef - trusted_fix_anchor_position
                float_line_norm = float(np.linalg.norm(float_line))
                if float_line_norm > 1.0e-6:
                    float_line_direction = float_line / float_line_norm
                    for radius in trusted_anchor_float_line_radii:
                        trusted_anchor_shadow_positions.extend(
                            [
                                trusted_fix_anchor_position
                                + radius * float_line_direction,
                                trusted_fix_anchor_position
                                - radius * float_line_direction,
                            ]
                        )
                shadow_position_specs.extend(
                    (position, snapshot_shadow_top_k, False)
                    for position in trusted_anchor_shadow_positions
                )
            if args.ddpr_respawn_wls_seed_shadow_only:
                shadow_position_specs.append(
                    (
                        np.asarray(wls_positions[i, :3], dtype=np.float64),
                        snapshot_shadow_top_k,
                        False,
                    )
                )
            if shadow_position_specs:
                for shadow_position, shadow_top_k, is_snapshot_loo in shadow_position_specs:
                    shadow_seed = ddpr_centered_ambiguity_seed(
                        dd_cp,
                        shadow_position,
                        ddpr_guard.covariance[:3, :3],
                        sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
                    )
                    available = np.asarray(
                        [
                            j
                            for j, key in enumerate(shadow_seed.keys)
                            if key in generations
                            and respawn_excluded_satellite not in key[:2]
                        ],
                        dtype=np.int64,
                    )
                    if available.size < respawn_subset_size:
                        continue
                    selected, _ranked = _select_ambiguity_indices(
                        shadow_seed.keys,
                        shadow_seed.qahat_cycles2,
                        available,
                        respawn_subset_size,
                        prefer_multifrequency_pairs=bool(
                            args.prefer_paired_multifrequency_subset
                        ),
                    )
                    shadow_keys = tuple(shadow_seed.keys[j] for j in selected)
                    shadow_candidates, _ = integer_search(
                        shadow_seed.ahat_cycles[selected],
                        shadow_seed.qahat_cycles2[np.ix_(selected, selected)],
                        n_candidates=shadow_top_k,
                    )
                    for candidate in shadow_candidates:
                        position, covariance, _ = condition_respawn_position(
                            shadow_seed, shadow_keys, candidate
                        )
                        shadow_state = BasinKalmanState.from_position(position, covariance)
                        position_shadow_states.append(shadow_state)
                        if is_snapshot_loo:
                            snapshot_loo_shadow_states.append(shadow_state)
                        if (
                            any(
                                shadow_position is trusted_position
                                for trusted_position in trusted_anchor_shadow_positions
                            )
                        ):
                            trusted_anchor_shadow_states.append(shadow_state)
            if recovery_assignment_bank is not None:
                fresh_assignments = assignments[n_replayed_assignments:]
                fresh_residuals = all_respawn_residuals[n_replayed_assignments:]
                recovery_assignment_bank.update(
                    i,
                    fresh_assignments,
                    (-0.5 * np.asarray(fresh_residuals, dtype=np.float64)).tolist(),
                )
                if (
                    recovery_arc_assignment_bank is not None
                    and satellite_arc_tracker is not None
                ):
                    recovery_arc_assignment_bank.update(
                        i,
                        fresh_assignments,
                        (-0.5 * np.asarray(fresh_residuals, dtype=np.float64)).tolist(),
                        satellite_arc_tracker.generations,
                    )
            if conditionals:
                # Diagnostic only: truth never changes candidates, weights,
                # output selection, or the FIX gate.
                epoch_ref = truth.get(round(float(tow), 1))
                if epoch_ref is not None and conditionals:
                    candidate_errors = np.asarray(
                        [
                            float(np.linalg.norm(state.mean[:3] - epoch_ref))
                            for state in conditionals
                        ],
                        dtype=np.float64,
                    )
                    respawn_oracle_rank = int(np.argmin(candidate_errors)) + 1
                    respawn_oracle_min_error = float(np.min(candidate_errors))
                if epoch_ref is not None and completion_shadow_states:
                    shadow_errors = np.asarray(
                        [
                            float(np.linalg.norm(state.mean[:3] - epoch_ref))
                            for state in completion_shadow_states
                        ],
                        dtype=np.float64,
                    )
                    completion_shadow_candidates = len(completion_shadow_states)
                    completion_shadow_oracle_min_error = float(np.min(shadow_errors))
                    n_completion_shadow_epochs += 1
                    n_completion_shadow_correct += int(
                        completion_shadow_oracle_min_error < 0.5
                    )
                if epoch_ref is not None and position_shadow_states:
                    position_shadow_candidates = len(position_shadow_states)
                    position_shadow_oracle_min_error = min(
                        float(np.linalg.norm(state.mean[:3] - epoch_ref))
                        for state in position_shadow_states
                    )
                    n_position_shadow_epochs += 1
                    n_position_shadow_correct += int(
                        position_shadow_oracle_min_error < 0.5
                    )
                if epoch_ref is not None and snapshot_loo_shadow_states:
                    snapshot_loo_shadow_candidates = len(snapshot_loo_shadow_states)
                    snapshot_loo_shadow_oracle_min_error = min(
                        float(np.linalg.norm(state.mean[:3] - epoch_ref))
                        for state in snapshot_loo_shadow_states
                    )
                    n_snapshot_loo_shadow_epochs += 1
                    n_snapshot_loo_shadow_correct += int(
                        snapshot_loo_shadow_oracle_min_error < 0.5
                    )
                if epoch_ref is not None and trusted_anchor_shadow_states:
                    trusted_anchor_shadow_candidates = len(trusted_anchor_shadow_states)
                    trusted_anchor_shadow_oracle_min_error = min(
                        float(np.linalg.norm(state.mean[:3] - epoch_ref))
                        for state in trusted_anchor_shadow_states
                    )
                    n_trusted_anchor_shadow_epochs += 1
                    n_trusted_anchor_shadow_correct += int(
                        trusted_anchor_shadow_oracle_min_error < 0.5
                    )
                if epoch_ref is not None and subset_shadow_states:
                    subset_shadow_candidates = len(subset_shadow_states)
                    subset_shadow_oracle_min_error = min(
                        float(np.linalg.norm(state.mean[:3] - epoch_ref))
                        for state in subset_shadow_states
                    )
                    n_subset_shadow_epochs += 1
                    n_subset_shadow_correct += int(
                        subset_shadow_oracle_min_error < 0.5
                    )
                if epoch_ref is not None and arc_shadow_states:
                    arc_shadow_candidates = len(arc_shadow_states)
                    arc_shadow_errors = np.asarray(
                        [
                            float(np.linalg.norm(state.mean[:3] - epoch_ref))
                            for state in arc_shadow_states
                        ],
                        dtype=np.float64,
                    )
                    arc_shadow_oracle_rank = int(np.argmin(arc_shadow_errors)) + 1
                    arc_shadow_oracle_min_error = float(np.min(arc_shadow_errors))
                    n_arc_shadow_epochs += 1
                    n_arc_shadow_correct += int(arc_shadow_oracle_min_error < 0.5)
                if assignments:
                    basin_pf.spawn(
                        assignments,
                        conditionals,
                        prior_mass=float(args.ddpr_respawn_mass),
                        candidate_log_weights=(
                            -0.5
                            * np.asarray(all_respawn_residuals, dtype=np.float64)
                            if args.ddpr_respawn_use_lambda_prior else None
                        ),
                        candidate_source_ids=respawn_source_ids,
                    )
                    n_respawn_candidates = len(assignments)
                    n_respawn_epochs += 1

        if (
            (not respawn_triggered or args.trusted_fix_anchor_float_line_promote)
            and args.ddpr_respawn_trusted_fix_anchor_shadow_only
            and trusted_fix_anchor_position is not None
            and dd_cp is not None
            and dd_pr is not None
            and int(dd_pr.n_dd) >= int(args.fix_min_dd_pairs)
        ):
            trusted_shadow_assignments: list[dict] = []
            trusted_shadow_residuals: list[float] = []
            trusted_shadow_source_ids: list[str] = []
            shadow_top_k = (
                int(args.ddpr_respawn_snapshot_shadow_top_k)
                if int(args.ddpr_respawn_snapshot_shadow_top_k) > 0
                else respawn_top_k
            )
            all_epoch_shadow_positions = list(
                covariance_axis_position_seeds(
                    trusted_fix_anchor_position,
                    ddpr_guard.covariance[:3, :3],
                    trusted_anchor_shadow_radii,
                    direction_mode=str(args.ddpr_respawn_seed_directions),
                )
            )
            float_line = float_kf.position_ecef - trusted_fix_anchor_position
            float_line_norm = float(np.linalg.norm(float_line))
            if float_line_norm > 1.0e-6:
                float_line_direction = float_line / float_line_norm
                for radius in trusted_anchor_float_line_radii:
                    all_epoch_shadow_positions.extend(
                        [
                            trusted_fix_anchor_position
                            + radius * float_line_direction,
                            trusted_fix_anchor_position
                            - radius * float_line_direction,
                        ]
                    )
            for shadow_position_index, shadow_position in enumerate(
                all_epoch_shadow_positions
            ):
                shadow_seed = ddpr_centered_ambiguity_seed(
                    dd_cp,
                    shadow_position,
                    ddpr_guard.covariance[:3, :3],
                    sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
                )
                available = np.asarray(
                    [
                        index
                        for index, key in enumerate(shadow_seed.keys)
                        if key in generations
                    ],
                    dtype=np.int64,
                )
                if available.size < respawn_subset_size:
                    continue
                selected, _ranked = _select_ambiguity_indices(
                    shadow_seed.keys,
                    shadow_seed.qahat_cycles2,
                    available,
                    respawn_subset_size,
                    prefer_multifrequency_pairs=bool(
                        args.prefer_paired_multifrequency_subset
                    ),
                )
                shadow_keys = tuple(shadow_seed.keys[index] for index in selected)
                shadow_candidates, shadow_residuals = integer_search(
                    shadow_seed.ahat_cycles[selected],
                    shadow_seed.qahat_cycles2[np.ix_(selected, selected)],
                    n_candidates=shadow_top_k,
                )
                for candidate_index, (candidate, residual) in enumerate(
                    zip(shadow_candidates, shadow_residuals)
                ):
                    position, covariance, _ = condition_respawn_position(
                        shadow_seed, shadow_keys, candidate
                    )
                    trusted_anchor_shadow_states.append(
                        BasinKalmanState.from_position(position, covariance)
                    )
                    trusted_shadow_assignments.append(
                        {
                            (key, generations[key]): int(value)
                            for key, value in zip(shadow_keys, candidate)
                        }
                    )
                    trusted_shadow_residuals.append(float(residual))
                    trusted_shadow_source_ids.append(
                        f"{i}:trusted_float_line:{shadow_position_index}:{candidate_index}"
                    )
            epoch_ref = truth.get(round(float(tow), 1))
            if epoch_ref is not None and trusted_anchor_shadow_states:
                trusted_anchor_shadow_candidates = len(trusted_anchor_shadow_states)
                trusted_anchor_shadow_oracle_min_error = min(
                    float(np.linalg.norm(state.mean[:3] - epoch_ref))
                    for state in trusted_anchor_shadow_states
                )
                n_trusted_anchor_shadow_epochs += 1
                n_trusted_anchor_shadow_correct += int(
                    trusted_anchor_shadow_oracle_min_error < 0.5
                )
            if (
                args.trusted_fix_anchor_float_line_promote
                and trusted_shadow_assignments
            ):
                basin_pf.spawn(
                    trusted_shadow_assignments,
                    trusted_anchor_shadow_states,
                    prior_mass=float(args.ddpr_respawn_mass),
                    candidate_log_weights=(
                        -0.5 * np.asarray(trusted_shadow_residuals, dtype=np.float64)
                        if args.ddpr_respawn_use_lambda_prior
                        else None
                    ),
                    candidate_source_ids=trusted_shadow_source_ids,
                )

        if (
            i in external_position_seeds
            and dd_cp is not None
            and dd_pr is not None
            and int(dd_pr.n_dd) >= int(args.fix_min_dd_pairs)
        ):
            external_assignments: list[dict] = []
            external_states: list[BasinKalmanState] = []
            external_residuals: list[float] = []
            external_source_ids: list[str] = []
            for position_index, seed_position in enumerate(
                external_position_seeds[i]
            ):
                external_seed = ddpr_centered_ambiguity_seed(
                    dd_cp,
                    seed_position,
                    ddpr_guard.covariance[:3, :3],
                    sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
                )
                available = np.asarray(
                    [
                        index
                        for index, key in enumerate(external_seed.keys)
                        if key in generations
                    ],
                    dtype=np.int64,
                )
                if available.size < respawn_subset_size:
                    continue
                selected, _ranked = _select_ambiguity_indices(
                    external_seed.keys,
                    external_seed.qahat_cycles2,
                    available,
                    respawn_subset_size,
                    prefer_multifrequency_pairs=bool(
                        args.prefer_paired_multifrequency_subset
                    ),
                )
                keys = tuple(external_seed.keys[index] for index in selected)
                if args.external_position_seed_mode == "rounded-direct":
                    rounded = np.rint(external_seed.ahat_cycles[selected])
                    selected_variances = np.diag(
                        external_seed.qahat_cycles2
                    )[selected]
                    normalized = (
                        rounded - external_seed.ahat_cycles[selected]
                    ) / np.sqrt(
                        np.maximum(selected_variances, 1.0e-12)
                    )
                    candidates = np.asarray([rounded], dtype=np.float64)
                    residuals = np.asarray(
                        [float(normalized @ normalized)], dtype=np.float64
                    )
                else:
                    candidates, residuals = integer_search(
                        external_seed.ahat_cycles[selected],
                        external_seed.qahat_cycles2[np.ix_(selected, selected)],
                        n_candidates=int(args.external_position_seed_top_k),
                    )
                for candidate_index, (candidate, residual) in enumerate(
                    zip(candidates, residuals)
                ):
                    if args.external_position_seed_mode == "rounded-direct":
                        position = np.asarray(seed_position, dtype=np.float64)
                        covariance = external_seed.position_covariance
                    else:
                        position, covariance, _distance = condition_respawn_position(
                            external_seed, keys, candidate
                        )
                    external_assignments.append(
                        {
                            (key, generations[key]): int(value)
                            for key, value in zip(keys, candidate)
                        }
                    )
                    external_states.append(
                        BasinKalmanState.from_position(
                            position,
                            covariance,
                            velocity_ecef=ddpr_guard.mean[3:6],
                            velocity_sigma_mps=1.0,
                            accel_process_sigma_mps2=3.0,
                        )
                    )
                    external_residuals.append(float(residual))
                    external_source_ids.append(
                        f"{i}:external_position:{position_index}:{candidate_index}"
                    )
            epoch_ref = truth.get(round(float(tow), 1))
            if epoch_ref is not None and external_states:
                external_position_shadow_candidates = len(external_states)
                external_position_shadow_oracle_min_error = min(
                    float(np.linalg.norm(state.mean[:3] - epoch_ref))
                    for state in external_states
                )
                n_external_position_shadow_epochs += 1
                n_external_position_shadow_correct += int(
                    external_position_shadow_oracle_min_error < 0.5
                )
            if args.external_position_seeds_promote and external_assignments:
                basin_pf.spawn(
                    external_assignments,
                    external_states,
                    prior_mass=float(args.ddpr_respawn_mass),
                    candidate_log_weights=(
                        -0.5 * np.asarray(external_residuals, dtype=np.float64)
                        if args.ddpr_respawn_use_lambda_prior
                        else None
                    ),
                    candidate_source_ids=external_source_ids,
                )

        if (
            int(args.trusted_anchor_refinement_seeds) > 0
            and args.ddpr_respawn_trusted_fix_anchor_shadow_only
            and trusted_fix_anchor_position is not None
            and dd_cp is not None
            and dd_pr is not None
            and int(dd_pr.n_dd) >= int(args.fix_min_dd_pairs)
        ):
            refinement_seed = ddpr_centered_ambiguity_seed(
                dd_cp,
                trusted_fix_anchor_position,
                ddpr_guard.covariance[:3, :3],
                sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
            )
            refinement_available = np.asarray(
                [
                    index
                    for index, key in enumerate(refinement_seed.keys)
                    if key in generations
                ],
                dtype=np.int64,
            )
            if refinement_available.size >= trusted_anchor_shadow_subset_size:
                refinement_variances = np.diag(
                    refinement_seed.qahat_cycles2
                )[refinement_available]
                refinement_selected = np.sort(
                    refinement_available[
                        np.argsort(refinement_variances)[
                            :trusted_anchor_shadow_subset_size
                        ]
                    ]
                )
                refinement_keys = tuple(
                    refinement_seed.keys[index] for index in refinement_selected
                )
                initial_candidates, _ = integer_search(
                    refinement_seed.ahat_cycles[refinement_selected],
                    refinement_seed.qahat_cycles2[
                        np.ix_(refinement_selected, refinement_selected)
                    ],
                    n_candidates=int(args.trusted_anchor_refinement_seeds),
                )
                refinement_states: list[BasinKalmanState] = []
                for initial_candidate in initial_candidates:
                    refined_seed_position, _covariance, _ = condition_respawn_position(
                        refinement_seed, refinement_keys, initial_candidate
                    )
                    refined_seed = ddpr_centered_ambiguity_seed(
                        dd_cp,
                        refined_seed_position,
                        ddpr_guard.covariance[:3, :3],
                        sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
                    )
                    refined_available = np.asarray(
                        [
                            index
                            for index, key in enumerate(refined_seed.keys)
                            if key in generations
                        ],
                        dtype=np.int64,
                    )
                    if refined_available.size < trusted_anchor_shadow_subset_size:
                        continue
                    refined_variances = np.diag(
                        refined_seed.qahat_cycles2
                    )[refined_available]
                    refined_selected = np.sort(
                        refined_available[
                            np.argsort(refined_variances)[
                                :trusted_anchor_shadow_subset_size
                            ]
                        ]
                    )
                    refined_keys = tuple(
                        refined_seed.keys[index] for index in refined_selected
                    )
                    refined_candidates, _ = integer_search(
                        refined_seed.ahat_cycles[refined_selected],
                        refined_seed.qahat_cycles2[
                            np.ix_(refined_selected, refined_selected)
                        ],
                        n_candidates=int(args.trusted_anchor_refinement_top_k),
                    )
                    for refined_candidate in refined_candidates:
                        position, covariance, _ = condition_respawn_position(
                            refined_seed, refined_keys, refined_candidate
                        )
                        refinement_states.append(
                            BasinKalmanState.from_position(position, covariance)
                        )
                epoch_ref = truth.get(round(float(tow), 1))
                if epoch_ref is not None and refinement_states:
                    trusted_refinement_shadow_candidates = len(refinement_states)
                    trusted_refinement_shadow_oracle_min_error = min(
                        float(np.linalg.norm(state.mean[:3] - epoch_ref))
                        for state in refinement_states
                    )
                    n_trusted_refinement_shadow_epochs += 1
                    n_trusted_refinement_shadow_correct += int(
                        trusted_refinement_shadow_oracle_min_error < 0.5
                    )

        if (
            args.enable_widelane_integer_score
            and wl_stats.fixed_dd_ambiguities
            and basin_pf.basins
        ):
            score_by_id: dict[str, float] = {}
            min_pairs = int(args.widelane_integer_min_pairs)
            for basin in basin_pf.basins:
                n_pairs, squared_residual = _widelane_integer_residual(
                    basin.assignment,
                    wl_stats.fixed_dd_ambiguities,
                )
                if n_pairs >= min_pairs:
                    score_by_id[basin.basin_id] = -float(
                        args.widelane_integer_mismatch_penalty
                    ) * squared_residual
                else:
                    score_by_id[basin.basin_id] = -float(
                        args.widelane_integer_missing_penalty
                    )
            basin_pf.update_log_likelihoods(score_by_id)
            n_widelane_integer_score_epochs += 1
            evidence_ledger.record(
                epoch=i,
                target="basin_pf",
                source="widelane_integer_consistency",
                observation_id=observation_id,
                n_rows=len(wl_stats.fixed_dd_ambiguities),
            )

        primary_source_token = (
            str(args.basin_protected_source_token)
            if float(args.basin_protected_source_fraction) > 0.0
            else ""
        )
        posterior = (
            basin_pf.posterior_excluding_source_only(primary_source_token)
            if primary_source_token
            else basin_pf.posterior()
        )
        position_cluster = basin_pf.position_cluster_posterior(
            radius_m=float(args.position_cluster_radius_m)
        )
        max_gamma = max(max_gamma, float(posterior.gamma))
        map_candidates = [
            basin
            for basin in basin_pf.basins
            if basin.assignment == posterior.map_assignment
            and not (
                primary_source_token
                and basin.proposal_sources
                and all(
                    primary_source_token in source
                    for source in basin.proposal_sources
                )
            )
        ]
        map_basin = max(map_candidates, key=lambda basin: basin.log_weight) if map_candidates else None
        if basin_trace_rows is not None:
            for basin in basin_pf.basins:
                basin_trace_rows.append(
                    {
                        "epoch": i,
                        "tow": float(tow),
                        "basin_id": basin.basin_id,
                        "assignment_id": ambiguity_assignment_id(basin.assignment),
                        "assignment_json": ambiguity_assignment_json(basin.assignment),
                        "epoch_log_likelihood": float(basin.epoch_log_marginal),
                        "cumulative_log_marginal": float(basin.cumulative_log_marginal),
                        "log_weight": float(basin.log_weight),
                        "ecef_x": float(basin.conditional.mean[0]),
                        "ecef_y": float(basin.conditional.mean[1]),
                        "ecef_z": float(basin.conditional.mean[2]),
                        "velocity_x": float(basin.conditional.mean[3]),
                        "velocity_y": float(basin.conditional.mean[4]),
                        "velocity_z": float(basin.conditional.mean[5]),
                        "birth_epoch": int(basin.birth_epoch),
                        "lineage": "|".join(basin.lineage),
                        "proposal_sources": "|".join(basin.proposal_sources),
                    }
                )
        temporal_posterior = None
        temporal_map_basin = None
        if temporal_filter is not None:
            temporal_candidates = [
                TemporalAmbiguityCandidate(
                    candidate_id=ambiguity_assignment_id(basin.assignment),
                    assignment=basin.assignment,
                    epoch_log_likelihood=float(basin.epoch_log_marginal),
                    position_ecef=basin.conditional.mean[:3],
                    velocity_ecef=basin.conditional.mean[3:6],
                )
                for basin in basin_pf.basins
            ]
            temporal_posterior = temporal_filter.step(
                i, epoch_dt, temporal_candidates
            )
            temporal_map_basin = next(
                (
                    basin for basin in basin_pf.basins
                    if ambiguity_assignment_id(basin.assignment)
                    == temporal_posterior.map_candidate_id
                ),
                None,
            )
            max_temporal_gamma = max(
                max_temporal_gamma, float(temporal_posterior.gamma)
            )
            n_temporal_map_disagreement += int(
                temporal_map_basin is not None
                and map_basin is not None
                and temporal_map_basin.basin_id != map_basin.basin_id
            )
        integrity_posterior = None
        integrity_map_basin = None
        integrity_result = None
        integrity_excluded_satellite = ""
        integrity_position_ball = None
        if integrity_filter is not None and basin_pf.basins:
            integrity_scores = np.zeros(len(basin_pf.basins), dtype=np.float64)
            if dd_pr is not None:
                excluded_satellites: tuple[str, ...] = ()
                if args.integrity_exclude_max_cost_satellite:
                    satellite_cost = satellite_pair_costs(
                        dd_pr,
                        ddpr_guard.mean[:3],
                        scale_m=float(args.integrity_scale_m),
                    )
                    cost_memory = float(args.integrity_satellite_cost_memory)
                    for satellite, current_cost in zip(
                        satellite_cost.satellite_ids,
                        satellite_cost.mean_pair_costs,
                    ):
                        previous_cost = integrity_satellite_cost_state.get(satellite)
                        integrity_satellite_cost_state[satellite] = (
                            float(current_cost)
                            if previous_cost is None
                            else cost_memory * previous_cost
                            + (1.0 - cost_memory) * float(current_cost)
                        )
                    integrity_excluded_satellite = max(
                        satellite_cost.satellite_ids,
                        key=integrity_satellite_cost_state.__getitem__,
                    )
                    excluded_satellites = (integrity_excluded_satellite,)
                    n_integrity_satellite_exclusions += 1
                integrity_result = multipivot_ddpr_scores(
                    dd_pr,
                    np.asarray(
                        [basin.conditional.mean[:3] for basin in basin_pf.basins],
                        dtype=np.float64,
                    ),
                    scale_m=float(args.integrity_scale_m),
                    trim_largest_pairs=int(args.integrity_trim_pairs),
                    excluded_satellites=excluded_satellites,
                )
                integrity_scores = (
                    float(args.integrity_weight) * integrity_result.scores
                )
                n_integrity_anchor_epochs += 1
                evidence_ledger.record(
                    epoch=i,
                    target="integrity_lineage",
                    source="multipivot_dd_pseudorange",
                    observation_id=observation_id,
                    n_rows=int(dd_pr.n_dd),
                    log_evidence=float(np.max(integrity_scores)),
                )
            integrity_candidates = [
                TemporalAmbiguityCandidate(
                    candidate_id=ambiguity_assignment_id(basin.assignment),
                    assignment=basin.assignment,
                    epoch_log_likelihood=float(integrity_scores[index]),
                    position_ecef=basin.conditional.mean[:3],
                    velocity_ecef=basin.conditional.mean[3:6],
                )
                for index, basin in enumerate(basin_pf.basins)
            ]
            integrity_motion: dict[str, object] = {"motion_mode": "none"}
            if integrity_tdcp is not None:
                integrity_motion = {
                    "motion_mode": "external",
                    "external_displacement_ecef_m": integrity_tdcp.displacement_ecef_m,
                    "external_covariance_m2": integrity_tdcp.covariance_m2,
                }
                evidence_ledger.record(
                    epoch=i,
                    target="integrity_lineage",
                    source="tdcp_displacement",
                    observation_id=(
                        f"tow={float(times[i - 1]):.3f}->{float(tow):.3f}"
                    ),
                    n_rows=int(integrity_tdcp.n_used),
                )
            integrity_posterior = integrity_filter.step(
                i,
                epoch_dt,
                integrity_candidates,
                **integrity_motion,
            )
            integrity_position_ball = integrity_filter.map_position_ball(
                float(args.position_cluster_radius_m)
            )
            integrity_map_basin = next(
                (
                    basin
                    for basin in basin_pf.basins
                    if ambiguity_assignment_id(basin.assignment)
                    == integrity_posterior.map_candidate_id
                ),
                None,
            )
            max_integrity_gamma = max(
                max_integrity_gamma, float(integrity_posterior.gamma)
            )
            n_integrity_map_disagreement += int(
                integrity_map_basin is not None
                and map_basin is not None
                and integrity_map_basin.basin_id != map_basin.basin_id
            )
        elif integrity_filter is not None:
            integrity_filter.step(i, epoch_dt, ())
        map_float_separation = (
            float(np.linalg.norm(map_basin.conditional.mean[:3] - float_kf.position_ecef))
            if map_basin is not None
            else float("nan")
        )
        map_ddpr_separation = (
            float(np.linalg.norm(map_basin.conditional.mean[:3] - ddpr_guard.mean[:3]))
            if map_basin is not None
            else float("nan")
        )
        ddpr_age_epochs = i - last_ddpr_epoch
        assignment_id = (
            ambiguity_assignment_id(posterior.map_assignment)
            if map_basin is not None else ""
        )
        policy_input = TrustedFixPolicyInput(
            epoch=i,
            assignment_id=assignment_id,
            gamma=float(posterior.gamma),
            n_ambiguities=len(posterior.map_assignment),
            map_float_separation_m=map_float_separation,
            map_ddpr_separation_m=map_ddpr_separation,
            last_ddpr_pairs=last_ddpr_pairs,
            ddpr_age_epochs=ddpr_age_epochs,
        )
        commit = commit_policy.evaluate(policy_input)
        gamma_fixed = bool(
            commit.gamma_eligible and commit.fix_streak >= int(args.fix_streak)
        )
        if gamma_fixed != bool(posterior.fixed and map_basin is not None):
            raise RuntimeError(
                f"legacy/replayable gamma FIX mismatch at epoch {i}: "
                f"{posterior.fixed} != {gamma_fixed}"
            )
        gate = commit.gate
        consistency_pass = gate.passed
        fixed = commit.fixed
        n_gamma_fix += int(gamma_fixed)
        n_consistency_reject += int(gamma_fixed and not consistency_pass)
        output_position = (
            map_basin.conditional.mean[:3].copy() if fixed else float_kf.position_ecef
        )
        ref = truth.get(round(float(tow), 1))
        basin_oracle_min_error = (
            min(
                float(np.linalg.norm(basin.conditional.mean[:3] - ref))
                for basin in basin_pf.basins
            )
            if basin_pf.basins and ref is not None
            else float("nan")
        )
        n_basin_oracle_sub50 += int(basin_oracle_min_error < 0.5)
        if (
            args.out_integrity_satellite_diagnostics is not None
            and integrity_result is not None
            and ref is not None
        ):
            integrity_positions = np.asarray(
                [basin.conditional.mean[:3] for basin in basin_pf.basins],
                dtype=np.float64,
            )
            integrity_errors = np.linalg.norm(
                integrity_positions - ref[None, :], axis=1
            )
            full_error = float(integrity_errors[integrity_result.best_index])
            guard_satellite_cost = satellite_pair_costs(
                dd_pr,
                ddpr_guard.mean[:3],
                scale_m=float(args.integrity_scale_m),
            )
            selected_satellite_cost = satellite_pair_costs(
                dd_pr,
                integrity_positions[integrity_result.best_index],
                scale_m=float(args.integrity_scale_m),
            )
            guard_cost_by_satellite = dict(
                zip(
                    guard_satellite_cost.satellite_ids,
                    guard_satellite_cost.mean_pair_costs,
                )
            )
            selected_cost_by_satellite = dict(
                zip(
                    selected_satellite_cost.satellite_ids,
                    selected_satellite_cost.mean_pair_costs,
                )
            )
            satellite_ids = sorted(
                set(str(value) for value in dd_pr.ref_sat_ids + dd_pr.sat_ids)
            )
            for excluded_satellite in satellite_ids:
                try:
                    excluded_result = multipivot_ddpr_scores(
                        dd_pr,
                        integrity_positions,
                        scale_m=float(args.integrity_scale_m),
                        trim_largest_pairs=int(args.integrity_trim_pairs),
                        excluded_satellites=(excluded_satellite,),
                    )
                except ValueError:
                    continue
                ordered_scores = np.sort(excluded_result.scores)
                score_margin = (
                    float(ordered_scores[-1] - ordered_scores[-2])
                    if len(ordered_scores) >= 2
                    else float("inf")
                )
                excluded_error = float(
                    integrity_errors[excluded_result.best_index]
                )
                integrity_satellite_rows.append(
                    {
                        "epoch": i,
                        "tow": float(tow),
                        "excluded_satellite": excluded_satellite,
                        "full_selected_error_m": full_error,
                        "excluded_selected_error_m": excluded_error,
                        "full_selected_sub50cm": int(full_error < 0.5),
                        "excluded_selected_sub50cm": int(excluded_error < 0.5),
                        "exclusion_recovers_sub50cm": int(
                            full_error >= 0.5 and excluded_error < 0.5
                        ),
                        "exclusion_breaks_sub50cm": int(
                            full_error < 0.5 and excluded_error >= 0.5
                        ),
                        "oracle_min_error_m": basin_oracle_min_error,
                        "excluded_best_probability": float(
                            excluded_result.probabilities[
                                excluded_result.best_index
                            ]
                        ),
                        "excluded_score_margin": score_margin,
                        "excluded_best_assignment_id": ambiguity_assignment_id(
                            basin_pf.basins[excluded_result.best_index].assignment
                        ),
                        "guard_mean_pair_cost": float(
                            guard_cost_by_satellite[excluded_satellite]
                        ),
                        "selected_mean_pair_cost": float(
                            selected_cost_by_satellite[excluded_satellite]
                        ),
                        "n_constellations": int(
                            excluded_result.n_constellations
                        ),
                        "n_satellites": int(excluded_result.n_satellites),
                    }
                )
        output_error = (
            float(np.linalg.norm(output_position - ref))
            if ref is not None and np.all(np.isfinite(ref))
            else float("nan")
        )
        ddpr_snapshot_error = (
            float(np.linalg.norm(ddpr_snapshot_position - ref))
            if ddpr_snapshot_position is not None and ref is not None
            else float("nan")
        )
        ddpr_snapshot_pair_exclusion_oracle_error = (
            min(
                float(np.linalg.norm(position - ref))
                for position in ddpr_snapshot_pair_exclusion_positions
            )
            if ddpr_snapshot_pair_exclusion_positions and ref is not None
            else float("nan")
        )
        trusted_fix_anchor_error = (
            float(np.linalg.norm(trusted_fix_anchor_position - ref))
            if trusted_fix_anchor_position is not None and ref is not None
            else float("nan")
        )
        truth_displacement = (
            ref - previous_scoring_ref
            if ref is not None and previous_scoring_ref is not None
            else None
        )
        tdcp_truth_displacement_error = (
            float(
                np.linalg.norm(
                    integrity_tdcp.displacement_ecef_m - truth_displacement
                )
            )
            if integrity_tdcp is not None and truth_displacement is not None
            else float("nan")
        )
        doppler_truth_displacement_error = (
            float(np.linalg.norm(velocity * epoch_dt - truth_displacement))
            if velocity is not None and truth_displacement is not None
            else float("nan")
        )
        imu_truth_displacement_error = (
            float(
                np.linalg.norm(
                    trusted_anchor_imu_displacement - truth_displacement
                )
            )
            if trusted_anchor_imu_displacement is not None
            and truth_displacement is not None
            else float("nan")
        )
        trusted_fix_anchor_age_for_row = (
            int(trusted_fix_anchor_age_epochs)
            if trusted_fix_anchor_position is not None
            else -1
        )
        wls_error = (
            float(np.linalg.norm(np.asarray(wls_positions[i, :3]) - ref))
            if ref is not None
            else float("nan")
        )
        map_error = (
            float(np.linalg.norm(map_basin.conditional.mean[:3] - ref))
            if map_basin is not None and ref is not None
            else float("nan")
        )
        temporal_map_error = (
            float(np.linalg.norm(temporal_map_basin.conditional.mean[:3] - ref))
            if temporal_map_basin is not None and ref is not None
            else float("nan")
        )
        n_temporal_map_sub50 += int(temporal_map_error < 0.5)
        integrity_map_error = (
            float(np.linalg.norm(integrity_map_basin.conditional.mean[:3] - ref))
            if integrity_map_basin is not None and ref is not None
            else float("nan")
        )
        n_integrity_map_sub50 += int(integrity_map_error < 0.5)
        integrity_ball_error = (
            float(np.linalg.norm(integrity_position_ball.mean_position_ecef - ref))
            if integrity_position_ball is not None and ref is not None
            else float("nan")
        )
        integrity_ball_gamma = (
            0.0
            if integrity_position_ball is None
            else float(integrity_position_ball.probability)
        )
        max_integrity_ball_gamma = max(
            max_integrity_ball_gamma, integrity_ball_gamma
        )
        n_integrity_ball_gamma99 += int(integrity_ball_gamma > 0.99)
        n_integrity_ball_gamma99_correct += int(
            integrity_ball_gamma > 0.99 and integrity_ball_error < 0.5
        )
        integrity_map_float_separation = (
            float(
                np.linalg.norm(
                    integrity_map_basin.conditional.mean[:3]
                    - float_kf.position_ecef
                )
            )
            if integrity_map_basin is not None
            else float("nan")
        )
        integrity_map_ddpr_separation = (
            float(
                np.linalg.norm(
                    integrity_map_basin.conditional.mean[:3]
                    - ddpr_guard.mean[:3]
                )
            )
            if integrity_map_basin is not None
            else float("nan")
        )
        integrity_guard_pass = bool(
            integrity_map_basin is not None
            and integrity_map_float_separation <= float(args.fix_consistency_m)
            and integrity_map_ddpr_separation <= float(args.fix_ddpr_consistency_m)
            and last_ddpr_pairs >= int(args.fix_min_dd_pairs)
            and ddpr_age_epochs <= int(args.fix_max_ddpr_age_epochs)
        )
        n_integrity_guard_pass += int(integrity_guard_pass)
        n_integrity_guard_pass_correct += int(
            integrity_guard_pass and integrity_map_error < 0.5
        )
        cluster_error = (
            float(np.linalg.norm(position_cluster.mean_position_ecef - ref))
            if ref is not None and np.all(np.isfinite(position_cluster.mean_position_ecef))
            else float("nan")
        )
        if fixed:
            n_declared_fix += 1
            if output_error < 0.5:
                n_correct_fix += 1
            else:
                n_false_fix += 1
            if args.ddpr_respawn_trusted_fix_anchor_shadow_only:
                trusted_fix_anchor_position = output_position.copy()
                trusted_fix_anchor_age_epochs = 0
        traces.append(
            RTKEpochTrace(
                epoch=i,
                tow=float(tow),
                assignment_id=assignment_id,
                gamma=float(posterior.gamma),
                n_ambiguities=len(posterior.map_assignment),
                map_float_separation_m=map_float_separation,
                map_ddpr_separation_m=map_ddpr_separation,
                last_ddpr_pairs=int(last_ddpr_pairs),
                ddpr_age_epochs=int(ddpr_age_epochs),
                ecef_x=float(output_position[0]),
                ecef_y=float(output_position[1]),
                ecef_z=float(output_position[2]),
                gamma_eligible=commit.gamma_eligible,
                fix_streak=commit.fix_streak,
                fixed=commit.fixed,
                evidence_records=len(evidence_ledger) - evidence_start,
            )
        )
        rows.append(
            {
                "epoch": i,
                "tow": float(tow),
                "ecef_x": float(output_position[0]),
                "ecef_y": float(output_position[1]),
                "ecef_z": float(output_position[2]),
                "fix": int(fixed),
                "gamma_fixed": int(gamma_fixed),
                "consistency_pass": int(consistency_pass),
                "float_consistency_pass": int(gate.float_consistent),
                "ddpr_consistency_pass": int(gate.ddpr_consistent),
                "ddpr_support_pass": int(gate.ddpr_supported),
                "ddpr_freshness_pass": int(gate.ddpr_fresh),
                "map_float_separation_m": map_float_separation,
                "map_ddpr_separation_m": map_ddpr_separation,
                "ddpr_guard_error_m": (
                    float(np.linalg.norm(ddpr_guard.mean[:3] - ref))
                    if ref is not None else float("nan")
                ),
                "last_ddpr_pairs": int(last_ddpr_pairs),
                "ddpr_age_epochs": int(ddpr_age_epochs),
                "last_ddpr_nis": last_ddpr_nis,
                "output_error_m": output_error,
                "map_error_m": map_error,
                "temporal_lineage_enabled": int(temporal_filter is not None),
                "temporal_map_assignment_id": (
                    "" if temporal_posterior is None
                    else temporal_posterior.map_candidate_id
                ),
                "temporal_gamma": (
                    0.0 if temporal_posterior is None
                    else float(temporal_posterior.gamma)
                ),
                "temporal_ess": (
                    0.0 if temporal_posterior is None
                    else float(temporal_posterior.ess)
                ),
                "temporal_dwell_epochs": (
                    0 if temporal_posterior is None
                    else int(temporal_posterior.dwell_epochs)
                ),
                "temporal_map_error_m": temporal_map_error,
                "integrity_lineage_enabled": int(integrity_filter is not None),
                "integrity_anchor_available": int(integrity_result is not None),
                "integrity_excluded_satellite": integrity_excluded_satellite,
                "integrity_tdcp_available": int(integrity_tdcp is not None),
                "integrity_tdcp_postfit_rms_m": (
                    float("nan")
                    if integrity_tdcp is None
                    else float(integrity_tdcp.postfit_rms_m)
                ),
                "integrity_map_assignment_id": (
                    "" if integrity_posterior is None
                    else integrity_posterior.map_candidate_id
                ),
                "integrity_gamma": (
                    0.0
                    if integrity_posterior is None
                    else float(integrity_posterior.gamma)
                ),
                "integrity_ess": (
                    0.0
                    if integrity_posterior is None
                    else float(integrity_posterior.ess)
                ),
                "integrity_dwell_epochs": (
                    0
                    if integrity_posterior is None
                    else int(integrity_posterior.dwell_epochs)
                ),
                "integrity_map_error_m": integrity_map_error,
                "integrity_map_ecef_x": (
                    float("nan")
                    if integrity_map_basin is None
                    else float(integrity_map_basin.conditional.mean[0])
                ),
                "integrity_map_ecef_y": (
                    float("nan")
                    if integrity_map_basin is None
                    else float(integrity_map_basin.conditional.mean[1])
                ),
                "integrity_map_ecef_z": (
                    float("nan")
                    if integrity_map_basin is None
                    else float(integrity_map_basin.conditional.mean[2])
                ),
                "integrity_map_float_separation_m": integrity_map_float_separation,
                "integrity_map_ddpr_separation_m": integrity_map_ddpr_separation,
                "integrity_guard_pass": int(integrity_guard_pass),
                "integrity_position_ball_gamma": integrity_ball_gamma,
                "integrity_position_ball_members": (
                    0
                    if integrity_position_ball is None
                    else int(integrity_position_ball.n_members)
                ),
                "integrity_position_ball_spread_m": (
                    float("nan")
                    if integrity_position_ball is None
                    else float(integrity_position_ball.rms_spread_m)
                ),
                "integrity_position_ball_error_m": integrity_ball_error,
                "basin_oracle_min_error_m": basin_oracle_min_error,
                "basin_oracle_sub50cm_available": int(
                    basin_oracle_min_error < 0.5
                ),
                "position_cluster_error_m": cluster_error,
                "position_cluster_gamma": float(position_cluster.gamma),
                "position_cluster_spread_m": float(position_cluster.rms_spread_m),
                "position_cluster_members": int(position_cluster.n_members),
                "position_cluster_float_separation_m": (
                    float(
                        np.linalg.norm(
                            position_cluster.mean_position_ecef - float_kf.position_ecef
                        )
                    )
                    if np.all(np.isfinite(position_cluster.mean_position_ecef))
                    else float("nan")
                ),
                "position_cluster_ddpr_separation_m": (
                    float(
                        np.linalg.norm(
                            position_cluster.mean_position_ecef - ddpr_guard.mean[:3]
                        )
                    )
                    if np.all(np.isfinite(position_cluster.mean_position_ecef))
                    else float("nan")
                ),
                "float_error_m": (
                    float(np.linalg.norm(float_kf.position_ecef - ref))
                    if ref is not None else float("nan")
                ),
                "float_position_sigma_m": float(
                    np.sqrt(np.trace(float_kf.covariance[:3, :3]))
                ),
                "dd_pr_nis": (
                    float("nan") if pr_diag is None else pr_diag.normalized_innovation_sq
                ),
                "dd_cp_nis": (
                    float("nan") if cp_diag is None else cp_diag.normalized_innovation_sq
                ),
                "ddpr_snapshot_error_m": ddpr_snapshot_error,
                "wls_error_m": wls_error,
                "ddpr_snapshot_accepted": int(
                    ddpr_snapshot_diagnostics is not None
                    and bool(ddpr_snapshot_diagnostics.get("accepted", False))
                ),
                "ddpr_snapshot_initial_rms_m": (
                    float("nan")
                    if ddpr_snapshot_diagnostics is None
                    else float(ddpr_snapshot_diagnostics["initial_rms_m"])
                ),
                "ddpr_snapshot_final_rms_m": (
                    float("nan")
                    if ddpr_snapshot_diagnostics is None
                    else float(ddpr_snapshot_diagnostics["final_rms_m"])
                ),
                "ddpr_snapshot_shift_m": (
                    float("nan")
                    if ddpr_snapshot_diagnostics is None
                    else float(ddpr_snapshot_diagnostics["shift_m"])
                ),
                "tdcp_truth_displacement_error_m": tdcp_truth_displacement_error,
                "doppler_truth_displacement_error_m": doppler_truth_displacement_error,
                "imu_truth_displacement_error_m": imu_truth_displacement_error,
                "ddpr_snapshot_pair_exclusion_positions": len(
                    ddpr_snapshot_pair_exclusion_positions
                ),
                "ddpr_snapshot_pair_exclusion_oracle_error_m": (
                    ddpr_snapshot_pair_exclusion_oracle_error
                ),
                "trusted_fix_anchor_error_m": trusted_fix_anchor_error,
                "trusted_fix_anchor_age_epochs": trusted_fix_anchor_age_for_row,
                "trusted_fix_anchor_ecef_x": (
                    float("nan")
                    if trusted_fix_anchor_position is None
                    else float(trusted_fix_anchor_position[0])
                ),
                "trusted_fix_anchor_ecef_y": (
                    float("nan")
                    if trusted_fix_anchor_position is None
                    else float(trusted_fix_anchor_position[1])
                ),
                "trusted_fix_anchor_ecef_z": (
                    float("nan")
                    if trusted_fix_anchor_position is None
                    else float(trusted_fix_anchor_position[2])
                ),
                "ddpr_snapshot_ecef_x": (
                    float("nan")
                    if ddpr_snapshot_position is None
                    else float(ddpr_snapshot_position[0])
                ),
                "ddpr_snapshot_ecef_y": (
                    float("nan")
                    if ddpr_snapshot_position is None
                    else float(ddpr_snapshot_position[1])
                ),
                "ddpr_snapshot_ecef_z": (
                    float("nan")
                    if ddpr_snapshot_position is None
                    else float(ddpr_snapshot_position[2])
                ),
                "ddpr_guard_ecef_x": float(ddpr_guard.mean[0]),
                "ddpr_guard_ecef_y": float(ddpr_guard.mean[1]),
                "ddpr_guard_ecef_z": float(ddpr_guard.mean[2]),
                "ambiguities_reset": (
                    0 if cp_diag is None else int(cp_diag.ambiguities_reset)
                ),
                "assignment_history_cleared": int(assignment_history_cleared),
                "assignment_arc_slips": int(epoch_arc_slips),
                "assignment_arc_slip_ids": epoch_arc_slip_ids,
                "stale_generation_holdover_basins": int(
                    stale_generation_holdover_basins
                ),
                "gamma": float(posterior.gamma),
                "fix_streak": int(commit.fix_streak),
                "map_assignment_id": assignment_id,
                "n_basins": int(posterior.n_basins),
                "basin_ess": float(posterior.ess),
                "map_n_ambiguities": len(posterior.map_assignment),
                "n_candidates_born": n_candidates,
                "respawn_triggered": int(respawn_triggered),
                "n_respawn_candidates_born": n_respawn_candidates,
                "n_respawn_position_seeds": n_respawn_position_seeds,
                "n_respawn_history_seeds": n_respawn_history_seeds,
                "n_respawn_assignment_candidates": n_respawn_assignment_candidates,
                "respawn_oracle_min_error_m": respawn_oracle_min_error,
                "respawn_oracle_rank": int(respawn_oracle_rank),
                "completion_shadow_candidates": completion_shadow_candidates,
                "completion_shadow_oracle_min_error_m": completion_shadow_oracle_min_error,
                "position_shadow_candidates": position_shadow_candidates,
                "position_shadow_oracle_min_error_m": position_shadow_oracle_min_error,
                "snapshot_loo_shadow_candidates": snapshot_loo_shadow_candidates,
                "snapshot_loo_shadow_oracle_min_error_m": snapshot_loo_shadow_oracle_min_error,
                "trusted_anchor_shadow_candidates": trusted_anchor_shadow_candidates,
                "trusted_anchor_shadow_oracle_min_error_m": trusted_anchor_shadow_oracle_min_error,
                "external_position_shadow_candidates": external_position_shadow_candidates,
                "external_position_shadow_oracle_min_error_m": external_position_shadow_oracle_min_error,
                "trusted_refinement_shadow_candidates": trusted_refinement_shadow_candidates,
                "trusted_refinement_shadow_oracle_min_error_m": trusted_refinement_shadow_oracle_min_error,
                "subset_shadow_candidates": subset_shadow_candidates,
                "subset_shadow_oracle_min_error_m": subset_shadow_oracle_min_error,
                "arc_shadow_candidates": arc_shadow_candidates,
                "arc_shadow_oracle_min_error_m": arc_shadow_oracle_min_error,
                "arc_shadow_oracle_rank": int(arc_shadow_oracle_rank),
                "arc_shadow_compute_seconds": arc_shadow_compute_seconds,
                "arc_completion_search_workspaces": arc_completion_search_workspaces,
                "respawn_excluded_satellite": respawn_excluded_satellite,
                "widelane_candidate_pairs": int(wl_stats.n_candidate_pairs),
                "widelane_fixed_pairs": int(wl_stats.n_fixed_pairs),
                "widelane_dd_pairs": int(wl_stats.n_dd),
                "widelane_fix_rate": float(wl_stats.fix_rate),
                "widelane_reason": str(wl_stats.reason),
                "n_dd_pr": 0 if dd_pr is None else int(dd_pr.n_dd),
                "n_dd_cp": 0 if dd_cp is None else int(dd_cp.n_dd),
            }
        )

        previous_scoring_ref = None if ref is None else ref.copy()
        epoch_compute_seconds.append(time.perf_counter() - epoch_compute_start)
        if runtime_process is not None:
            rss_samples.append(int(runtime_process.memory_info().rss))

    evidence_audit = evidence_ledger.audit()
    replayed = replay_fix_decisions(traces, policy_config)
    replay_mismatches = sum(
        decision.fixed != trace.fixed or decision.fix_streak != trace.fix_streak
        for decision, trace in zip(replayed, traces)
    )
    if replay_mismatches:
        raise RuntimeError(f"online/replay FIX mismatch count: {replay_mismatches}")
    false_rate = 100.0 * n_false_fix / n_declared_fix if n_declared_fix else 0.0
    runtime_epoch_loop_seconds = time.perf_counter() - epoch_loop_start
    runtime_total_seconds = time.perf_counter() - runtime_start
    runtime_epochs_per_second = (
        len(rows) / runtime_epoch_loop_seconds
        if runtime_epoch_loop_seconds > 0 else 0.0
    )
    summary = {
        "run": str(args.run),
        "n_epochs": len(rows),
        "subset_size": int(args.subset_size),
        "top_k": int(args.top_k),
        "lambda_engine": str(args.lambda_engine),
        "runtime_mode": str(args.runtime_mode),
        "lambda_batch_calls": int(lambda_batch_calls),
        "lambda_batch_problems": int(lambda_batch_problems),
        "lambda_batch_compute_seconds": float(lambda_batch_compute_seconds),
        "runtime_seconds": float(runtime_epoch_loop_seconds),
        "runtime_total_seconds": float(runtime_total_seconds),
        "runtime_epochs_per_second": float(runtime_epochs_per_second),
        "runtime_total_epochs_per_second": (
            float(len(rows) / runtime_total_seconds)
            if runtime_total_seconds > 0 else 0.0
        ),
        "epoch_compute_p99_seconds": (
            float(np.percentile(epoch_compute_seconds, 99.0))
            if epoch_compute_seconds else 0.0
        ),
        "rss_start_bytes": int(rss_samples[0]) if rss_samples else None,
        "rss_end_bytes": int(rss_samples[-1]) if rss_samples else None,
        "rss_peak_bytes": int(max(rss_samples)) if rss_samples else None,
        "rss_growth_bytes": (
            int(rss_samples[-1] - rss_samples[0]) if rss_samples else None
        ),
        "rss_last_quarter_growth_bytes": (
            int(rss_samples[-1] - rss_samples[(3 * len(rss_samples)) // 4])
            if rss_samples else None
        ),
        "rss_second_half_slope_bytes_per_epoch": (
            float(
                np.polyfit(
                    np.arange(len(rss_samples) // 2, len(rss_samples)),
                    np.asarray(rss_samples[len(rss_samples) // 2 :], dtype=np.float64),
                    1,
                )[0]
            )
            if len(rss_samples) >= 4 else None
        ),
        "dd_carrier_families": list(dd_carrier_families),
        "basin_diversity_reserve_fraction": float(
            args.basin_diversity_reserve_fraction
        ),
        "basin_diversity_radius_m": float(args.basin_diversity_radius_m),
        "basin_dedup_position_radius_m": (
            float(args.basin_dedup_position_radius_m)
            if np.isfinite(float(args.basin_dedup_position_radius_m))
            else None
        ),
        "basin_source_reserve_fraction": float(args.basin_source_reserve_fraction),
        "basin_protected_source_token": str(args.basin_protected_source_token),
        "basin_protected_source_fraction": float(
            args.basin_protected_source_fraction
        ),
        "fix_gamma_threshold": float(args.fix_gamma),
        "fix_min_streak": int(args.fix_streak),
        "fix_float_consistency_m": float(args.fix_consistency_m),
        "fix_ddpr_consistency_m": float(args.fix_ddpr_consistency_m),
        "fix_min_dd_pairs": int(args.fix_min_dd_pairs),
        "fix_max_ddpr_age_epochs": int(args.fix_max_ddpr_age_epochs),
        "sigma_basin_dd_pr_m": float(basin_ddpr_sigma),
        "widelane_basin_evidence_enabled": bool(
            args.enable_widelane_basin_evidence
        ),
        "widelane_min_epochs": int(args.widelane_min_epochs),
        "widelane_max_std_cycles": float(args.widelane_max_std_cycles),
        "widelane_ratio_threshold": float(args.widelane_ratio_threshold),
        "widelane_min_fix_rate": float(args.widelane_min_fix_rate),
        "widelane_basin_sigma_m": float(args.widelane_basin_sigma_m),
        "widelane_evidence_epochs": int(n_widelane_evidence_epochs),
        "widelane_integer_score_enabled": bool(
            args.enable_widelane_integer_score
        ),
        "widelane_integer_min_pairs": int(args.widelane_integer_min_pairs),
        "widelane_integer_mismatch_penalty": float(
            args.widelane_integer_mismatch_penalty
        ),
        "widelane_integer_missing_penalty": float(
            args.widelane_integer_missing_penalty
        ),
        "widelane_integer_score_epochs": int(n_widelane_integer_score_epochs),
        "prefer_paired_multifrequency_subset": bool(
            args.prefer_paired_multifrequency_subset
        ),
        "birth_epochs": int(n_birth_epochs),
        "ddpr_respawn_enabled": bool(args.enable_ddpr_respawn),
        "ddpr_respawn_subset_size": int(respawn_subset_size),
        "ddpr_respawn_top_k": int(respawn_top_k),
        "ddpr_respawn_lambda_prior": bool(args.ddpr_respawn_use_lambda_prior),
        "ddpr_respawn_seed_radii_m": list(respawn_seed_radii),
        "ddpr_respawn_seed_directions": str(args.ddpr_respawn_seed_directions),
        "ddpr_respawn_shadow_seed_radii_m": list(shadow_seed_radii),
        "ddpr_respawn_snapshot_seed_shadow_only": bool(
            args.ddpr_respawn_snapshot_seed_shadow_only
        ),
        "ddpr_respawn_snapshot_seed_promote": bool(
            args.ddpr_respawn_snapshot_seed_promote
        ),
        "ddpr_respawn_snapshot_loo_shadow_only": bool(
            args.ddpr_respawn_snapshot_loo_shadow_only
        ),
        "ddpr_snapshot_pair_exclusion_position_shadow_only": bool(
            args.ddpr_snapshot_pair_exclusion_position_shadow_only
        ),
        "ddpr_respawn_wls_seed_shadow_only": bool(
            args.ddpr_respawn_wls_seed_shadow_only
        ),
        "ddpr_respawn_trusted_fix_anchor_shadow_only": bool(
            args.ddpr_respawn_trusted_fix_anchor_shadow_only
        ),
        "trusted_fix_anchor_snapshot_reset_rms_m": float(
            args.trusted_fix_anchor_snapshot_reset_rms_m
        ),
        "trusted_fix_anchor_motion_fallback": str(
            args.trusted_fix_anchor_motion_fallback
        ),
        "trusted_fix_anchor_doppler_bias_window": int(
            args.trusted_fix_anchor_doppler_bias_window
        ),
        "trusted_fix_anchor_shadow_radii_m": list(trusted_anchor_shadow_radii),
        "trusted_fix_anchor_float_line_radii_m": list(
            trusted_anchor_float_line_radii
        ),
        "trusted_fix_anchor_float_line_promote": bool(
            args.trusted_fix_anchor_float_line_promote
        ),
        "external_position_seeds_csv": (
            None
            if args.external_position_seeds_csv is None
            else str(args.external_position_seeds_csv)
        ),
        "external_position_seed_separation_m": float(
            args.external_position_seed_separation_m
        ),
        "external_position_seed_max": int(args.external_position_seed_max),
        "external_position_seed_top_k": int(args.external_position_seed_top_k),
        "external_position_seed_mode": str(args.external_position_seed_mode),
        "external_position_seeds_promote": bool(
            args.external_position_seeds_promote
        ),
        "trusted_fix_anchor_imu_velocity_blend_alpha": float(
            args.trusted_fix_anchor_imu_velocity_blend_alpha
        ),
        "trusted_anchor_refinement_seeds": int(
            args.trusted_anchor_refinement_seeds
        ),
        "trusted_anchor_refinement_top_k": int(
            args.trusted_anchor_refinement_top_k
        ),
        "trusted_anchor_shadow_subset_size": int(
            trusted_anchor_shadow_subset_size
        ),
        "trusted_refinement_shadow_epochs": int(
            n_trusted_refinement_shadow_epochs
        ),
        "trusted_refinement_shadow_correct_epochs": int(
            n_trusted_refinement_shadow_correct
        ),
        "trusted_fix_anchor_snapshot_resets": int(
            n_trusted_anchor_snapshot_resets
        ),
        "ddpr_respawn_snapshot_shadow_radii_m": list(snapshot_shadow_radii),
        "ddpr_respawn_snapshot_extrapolation_scales": list(
            snapshot_extrapolation_scales
        ),
        "ddpr_respawn_snapshot_shadow_top_k": int(
            args.ddpr_respawn_snapshot_shadow_top_k
        ),
        "ddpr_snapshot_prior_sigma_m": float(args.ddpr_snapshot_prior_sigma_m),
        "ddpr_snapshot_max_shift_m": float(args.ddpr_snapshot_max_shift_m),
        "ddpr_snapshot_pair_residual_max_m": float(
            args.ddpr_snapshot_pair_residual_max_m
        ),
        "ddpr_respawn_exclude_max_cost_satellite": bool(
            args.ddpr_respawn_exclude_max_cost_satellite
        ),
        "ddpr_respawn_shadow_one_swap_top_k": int(
            args.ddpr_respawn_shadow_one_swap_top_k
        ),
        "ddpr_respawn_shadow_window_count": int(
            args.ddpr_respawn_shadow_window_count
        ),
        "ddpr_respawn_shadow_window_top_k": int(
            args.ddpr_respawn_shadow_window_top_k
        ),
        "ddpr_respawn_history_seeds": int(args.ddpr_respawn_history_seeds),
        "ddpr_respawn_history_separation_m": float(
            args.ddpr_respawn_history_separation_m
        ),
        "ddpr_respawn_history_max_age_epochs": int(
            args.ddpr_respawn_history_max_age_epochs
        ),
        "ddpr_respawn_history_selection": str(args.ddpr_respawn_history_selection),
        "ddpr_respawn_history_max_guard_distance_m": (
            float(args.ddpr_respawn_history_max_guard_distance_m)
            if np.isfinite(float(args.ddpr_respawn_history_max_guard_distance_m))
            else None
        ),
        "ddpr_respawn_history_propagate_velocity": bool(
            args.ddpr_respawn_history_propagate_velocity
        ),
        "ddpr_respawn_history_propagate_tdcp": bool(
            args.ddpr_respawn_history_propagate_tdcp
        ),
        "ddpr_respawn_assignment_history": int(
            args.ddpr_respawn_assignment_history
        ),
        "ddpr_respawn_assignment_history_max_age_epochs": int(
            args.ddpr_respawn_assignment_history_max_age_epochs
        ),
        "ddpr_respawn_assignment_pivot_rebase": bool(
            args.ddpr_respawn_assignment_pivot_rebase
        ),
        "ddpr_respawn_assignment_history_clears": int(
            n_assignment_history_clears
        ),
        "ddpr_respawn_assignment_arc_shadow": bool(
            args.ddpr_respawn_assignment_arc_shadow
        ),
        "ddpr_respawn_assignment_arc_promote": bool(
            args.ddpr_respawn_assignment_arc_promote
        ),
        "ddpr_respawn_assignment_arc_slip_threshold_cycles": float(
            args.ddpr_respawn_assignment_arc_slip_threshold_cycles
        ),
        "ddpr_respawn_assignment_arc_max_gap_epochs": int(
            args.ddpr_respawn_assignment_arc_max_gap_epochs
        ),
        "ddpr_respawn_assignment_arc_completion_top_k": int(
            args.ddpr_respawn_assignment_arc_completion_top_k
        ),
        "ddpr_respawn_assignment_arc_completion_per_assignment": int(
            args.ddpr_respawn_assignment_arc_completion_per_assignment
        ),
        "ddpr_respawn_assignment_arc_shadow_max_candidates": int(
            args.ddpr_respawn_assignment_arc_shadow_max_candidates
        ),
        "ddpr_respawn_assignment_arc_slips": int(n_arc_slips),
        "ddpr_respawn_assignment_completion_top_k": int(
            args.ddpr_respawn_assignment_completion_top_k
        ),
        "ddpr_respawn_assignment_completion_min_stable": int(
            args.ddpr_respawn_assignment_completion_min_stable
        ),
        "ddpr_respawn_assignment_completion_shadow_only": bool(
            args.ddpr_respawn_assignment_completion_shadow_only
        ),
        "completion_shadow_epochs": int(n_completion_shadow_epochs),
        "completion_shadow_correct_epochs": int(n_completion_shadow_correct),
        "position_shadow_epochs": int(n_position_shadow_epochs),
        "position_shadow_correct_epochs": int(n_position_shadow_correct),
        "snapshot_loo_shadow_epochs": int(n_snapshot_loo_shadow_epochs),
        "snapshot_loo_shadow_correct_epochs": int(n_snapshot_loo_shadow_correct),
        "trusted_anchor_shadow_epochs": int(n_trusted_anchor_shadow_epochs),
        "trusted_anchor_shadow_correct_epochs": int(
            n_trusted_anchor_shadow_correct
        ),
        "external_position_shadow_epochs": int(n_external_position_shadow_epochs),
        "external_position_shadow_correct_epochs": int(
            n_external_position_shadow_correct
        ),
        "subset_shadow_epochs": int(n_subset_shadow_epochs),
        "subset_shadow_correct_epochs": int(n_subset_shadow_correct),
        "arc_shadow_epochs": int(n_arc_shadow_epochs),
        "arc_shadow_correct_epochs": int(n_arc_shadow_correct),
        "arc_shadow_compute_seconds": float(total_arc_shadow_compute_seconds),
        "arc_shadow_max_epoch_compute_seconds": float(
            max_arc_shadow_compute_seconds
        ),
        "stale_generation_holdover_basins": int(
            n_stale_generation_holdover_basins
        ),
        "ddpr_respawn_epochs": int(n_respawn_epochs),
        "temporal_lineage_enabled": bool(args.enable_temporal_lineage),
        "temporal_map_disagreement_epochs": int(n_temporal_map_disagreement),
        "temporal_map_sub50cm_epochs": int(n_temporal_map_sub50),
        "max_temporal_gamma": float(max_temporal_gamma),
        "integrity_lineage_enabled": bool(args.enable_integrity_lineage),
        "integrity_scale_m": float(args.integrity_scale_m),
        "integrity_trim_pairs": int(args.integrity_trim_pairs),
        "integrity_weight": float(args.integrity_weight),
        "integrity_exclude_max_cost_satellite": bool(
            args.integrity_exclude_max_cost_satellite
        ),
        "integrity_satellite_cost_memory": float(
            args.integrity_satellite_cost_memory
        ),
        "integrity_satellite_exclusions": int(n_integrity_satellite_exclusions),
        "integrity_anchor_epochs": int(n_integrity_anchor_epochs),
        "integrity_tdcp_intervals": int(n_integrity_tdcp_intervals),
        "integrity_map_disagreement_epochs": int(n_integrity_map_disagreement),
        "integrity_map_sub50cm_epochs": int(n_integrity_map_sub50),
        "basin_oracle_sub50cm_epochs": int(n_basin_oracle_sub50),
        "integrity_selection_given_oracle_pct": float(
            100.0 * n_integrity_map_sub50 / max(n_basin_oracle_sub50, 1)
        ),
        "max_integrity_gamma": float(max_integrity_gamma),
        "max_integrity_position_ball_gamma": float(max_integrity_ball_gamma),
        "integrity_position_ball_gamma99_epochs": int(n_integrity_ball_gamma99),
        "integrity_position_ball_gamma99_correct_epochs": int(
            n_integrity_ball_gamma99_correct
        ),
        "integrity_guard_pass_epochs": int(n_integrity_guard_pass),
        "integrity_guard_pass_correct_epochs": int(n_integrity_guard_pass_correct),
        "integrity_guard_pass_false_epochs": int(
            n_integrity_guard_pass - n_integrity_guard_pass_correct
        ),
        "float_ambiguity_resets": int(n_float_resets),
        "declared_fix_epochs": int(n_declared_fix),
        "gamma_fix_epochs": int(n_gamma_fix),
        "consistency_reject_epochs": int(n_consistency_reject),
        "correct_fix_epochs": int(n_correct_fix),
        "false_fix_epochs": int(n_false_fix),
        "false_fix_pct": float(false_rate),
        "max_gamma": float(max_gamma),
        "sub50cm_all_epochs": int(sum(float(row["output_error_m"]) < 0.5 for row in rows)),
        "evidence_records": int(evidence_audit.n_records),
        "evidence_updates": int(evidence_audit.n_updates),
        "evidence_beta_errors": int(evidence_audit.beta_error_count),
        "commit_replay_mismatches": int(replay_mismatches),
    }
    args.out_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with args.out_diagnostics.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_trajectory(args.out_trajectory, rows)
    if args.out_trace is not None:
        args.out_trace.parent.mkdir(parents=True, exist_ok=True)
        with args.out_trace.open("w", newline="") as fh:
            trace_rows = [trace.row() for trace in traces]
            writer = csv.DictWriter(fh, fieldnames=list(trace_rows[0]))
            writer.writeheader()
            writer.writerows(trace_rows)
    if args.out_evidence is not None:
        args.out_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.out_evidence.open("w", newline="") as fh:
            evidence_rows = evidence_ledger.rows()
            writer = csv.DictWriter(fh, fieldnames=list(evidence_rows[0]))
            writer.writeheader()
            writer.writerows(evidence_rows)
    if basin_trace_rows is not None:
        basin_trace_rows.close()
    if (
        args.out_integrity_satellite_diagnostics is not None
        and integrity_satellite_rows
    ):
        args.out_integrity_satellite_diagnostics.parent.mkdir(
            parents=True, exist_ok=True
        )
        with args.out_integrity_satellite_diagnostics.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(integrity_satellite_rows[0])
            )
            writer.writeheader()
            writer.writerows(integrity_satellite_rows)
    # Refresh total wall time after all requested output serialization.  This
    # is the honest end-to-end throughput denominator; epoch-loop throughput
    # remains separately reported above.
    runtime_total_seconds = time.perf_counter() - runtime_start
    summary["runtime_total_seconds"] = float(runtime_total_seconds)
    summary["runtime_total_epochs_per_second"] = (
        float(len(rows) / runtime_total_seconds)
        if runtime_total_seconds > 0 else 0.0
    )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_diagnostics}")
    print(f"wrote {args.out_trajectory}")


if __name__ == "__main__":
    main()
