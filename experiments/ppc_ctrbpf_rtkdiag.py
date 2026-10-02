"""RTK-diagnostic candidate loading, gating, ranking and run-index policies.

Moved verbatim out of ``exp_ppc_ctrbpf_fgo`` (which re-exports every name);
these are the helpers most PPC audit/compose scripts import. No behaviour
change.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from ppc_ctrbpf_config import CTRBPFConfig


def _load_rtk_diag_file(path: Path) -> dict[float, dict[str, str]]:
    """Parse gnss_solve --diagnostics-csv output keyed by rounded TOW."""
    rows: dict[float, dict[str, str]] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                tow = round(float(row["tow"]), 1)
            except (KeyError, TypeError, ValueError):
                continue
            rows[tow] = row
    return rows


def _diag_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _diag_bool(row: dict[str, str], key: str) -> bool:
    value = row.get(key)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", "", "nan"}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return bool(text)


def _rtkdiag_candidate_diag_policy_gate(
    row: dict[str, str],
    *,
    require_any_fields: tuple[str, ...],
    require_all_fields: tuple[str, ...],
    min_fields: tuple[tuple[str, float], ...],
    max_fields: tuple[tuple[str, float], ...],
) -> bool:
    if require_any_fields and not any(_diag_bool(row, key) for key in require_any_fields):
        return False
    if require_all_fields and not all(_diag_bool(row, key) for key in require_all_fields):
        return False
    for key, threshold in min_fields:
        value = _diag_float(row, key)
        if not np.isfinite(value) or value < float(threshold):
            return False
    for key, threshold in max_fields:
        value = _diag_float(row, key)
        if not np.isfinite(value) or value > float(threshold):
            return False
    return True


def _rtkdiag_candidate_gate(
    row: dict[str, str] | None,
    *,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float = 0.3,
) -> bool:
    """Main candidate gate.

    - status=4 (Fix): ratio>=ratio_min AND residual_rms<=residual_rms_max
    - status=5 (Float): residual_rms<=status5_residual_rms_max (ratio irrelevant);
      set status5_residual_rms_max=0.0 to disable status=5 acceptance.
    """
    if row is None:
        return False
    try:
        output_added = int(row.get("output_added", "0")) == 1
        status = int(row.get("final_status", "0"))
    except ValueError:
        return False
    if not output_added:
        return False
    residual_rms = _diag_float(row, "final_residual_rms")
    if status == 4:
        return (
            _diag_float(row, "final_ratio") >= float(ratio_min)
            and residual_rms <= float(residual_rms_max)
        )
    if status == 5 and float(status5_residual_rms_max) > 0.0:
        return residual_rms <= float(status5_residual_rms_max)
    return False


def _rtkdiag_candidate_float_gate(
    row: dict[str, str] | None,
    *,
    label: str,
    allowed_labels: tuple[str, ...],
    residual_rms_max: float,
    residual_abs_max: float,
    min_sats: int,
) -> bool:
    if row is None or not allowed_labels or label not in set(allowed_labels):
        return False
    try:
        output_added = int(row.get("output_added", "0")) == 1
        final_status = int(row.get("final_status", "0")) == 3
    except ValueError:
        return False
    return (
        output_added
        and final_status
        and _diag_float(row, "final_residual_rms") <= float(residual_rms_max)
        and _diag_float(row, "final_residual_abs_max") <= float(residual_abs_max)
        and _diag_float(row, "final_sats") >= float(min_sats)
    )


def _rtkdiag_candidate_status5_gate(
    row: dict[str, str] | None,
    *,
    label: str,
    allowed_labels: tuple[str, ...],
    residual_rms_max: float,
    min_sats: int,
) -> bool:
    if row is None or not allowed_labels or label not in set(allowed_labels):
        return False
    try:
        final_status = int(row.get("final_status", "0")) == 5
    except ValueError:
        return False
    return (
        final_status
        and _diag_float(row, "final_residual_rms") <= float(residual_rms_max)
        and _diag_float(row, "final_sats") >= float(min_sats)
    )


def _rtkdiag_nearest_candidate_row(
    candidate_pos: dict[float, np.ndarray],
    candidate_diag: dict[float, dict[str, str]],
    t_key: float,
    *,
    max_dt_s: float,
) -> tuple[float, np.ndarray | None, dict[str, str] | None]:
    max_steps = max(0, int(round(float(max_dt_s) * 10.0)))
    for step in range(max_steps + 1):
        offsets = (0.0,) if step == 0 else (-0.1 * step, 0.1 * step)
        for offset in offsets:
            cand_t_key = round(float(t_key) + float(offset), 1)
            cand = candidate_pos.get(cand_t_key)
            diag_row = candidate_diag.get(cand_t_key)
            if cand is not None and diag_row is not None:
                return cand_t_key, cand, diag_row
    return float(t_key), None, None


def _rtkdiag_candidate_sort_key(
    row: dict[str, str],
    *,
    mode: str,
) -> tuple[float, float]:
    """Rank gated RTK diagnostic candidates; smaller tuple is better."""
    ratio = _diag_float(row, "final_ratio")
    residual = _diag_float(row, "final_residual_rms")
    update_rows = _diag_float(row, "final_update_rows")
    if mode == "ratio":
        return (-ratio, residual)
    if mode == "score":
        return (residual / max(ratio, 1.0e-6), residual)
    if mode == "maxabs":
        return (_diag_float(row, "final_residual_abs_max"), residual)
    if mode == "nrows":
        return (-update_rows, residual)
    if mode == "rms_per_row":
        return (residual / max(update_rows, 1.0), residual)
    if mode == "score_per_row":
        return ((residual / max(ratio, 1.0e-6)) / max(update_rows, 1.0), residual)
    if mode == "score_per_row2":
        return ((residual / max(ratio, 1.0e-6)) / max(update_rows, 1.0) ** 2, residual)
    if mode == "score_per_row3":
        return ((residual / max(ratio, 1.0e-6)) / max(update_rows, 1.0) ** 3, residual)
    if mode == "rms_minus_alpha_rows":
        return (residual - 0.1 * update_rows, residual)
    if mode == "log_combined":
        import math as _m
        return (_m.log(residual + 1.0e-3) - 0.5 * _m.log(max(update_rows, 1.0)), residual)
    if mode == "composite_3axis_n2":
        # 3-axis sim BEST for n/r2: residual / (ratio^0.5 * rows^1.5 * abs_max^0.5)
        # sim 39.92% vs score 39.12 (+0.74pp); also n/r3 sim 59.04% vs 58.85 (+0.19pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.5
                       * max(update_rows, 1.0) ** 1.5
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_3axis_t2":
        # 3-axis sim BEST for t/r2: residual / (ratio^0.5 * rows^2.0)
        # sim 84.78% vs score_per_row 84.53 (+0.25pp).
        return (
            residual / (max(ratio, 1.0e-6) ** 0.5
                       * max(update_rows, 1.0) ** 2.0),
            residual,
        )
    if mode == "composite_3axis_n1":
        # 3-axis sim BEST for n/r1: residual / (rows^0.5 * abs_max^0.5)
        # sim 64.25% vs rms_per_row 63.89 (+0.37pp). Note: a=0 (no ratio dependence).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(update_rows, 1.0) ** 0.5
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_n2_v2":
        # Fine sim BEST for n/r2: residual / (ratio^0.4 * rows^1.0 * abs_max^0.7)
        # sim 40.14% vs composite_3axis_n2 39.92 (+0.22pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.4
                       * max(update_rows, 1.0) ** 1.0
                       * max(abs_max, 1.0e-3) ** 0.7),
            residual,
        )
    if mode == "composite_n3_v2":
        # Fine sim BEST for n/r3: residual / (ratio^0.2 * rows^0.5 * abs_max^0.5)
        # sim 59.10% vs composite_3axis_n2 59.04 (+0.05pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 0.5
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_n1_v2":
        # Fine sim BEST for n/r1: residual / (rows^0.5 * abs_max^0.3)
        # sim 64.31% vs composite_3axis_n1 64.25 (+0.06pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(update_rows, 1.0) ** 0.5
                       * max(abs_max, 1.0e-3) ** 0.3),
            residual,
        )
    if mode == "composite_t2_v2":
        # Fine sim BEST for t/r2: residual / (ratio^0.2 * rows^2.0 * abs_max^0.5)
        # sim 84.86% vs composite_3axis_t2 84.78 (+0.08pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 2.0
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_t3_v2":
        # Phase 11dm t/r3 re-sweep after 11dl pool:
        # residual / (ratio^1.3 * rows^1.5 * abs_max^-0.5).
        # Negative abs exponent intentionally penalizes tiny abs_max candidates
        # that became traps in the expanded 11dl pool.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 1.3
                       * max(update_rows, 1.0) ** 1.5
                       * max(abs_max, 1.0e-3) ** -0.5),
            residual,
        )
    if mode == "composite_t3_v4":
        # Phase 11eb t/r3 re-sweep after 11ea blocks:
        # residual / (ratio^1.5 * rows^1.5 * abs_max^-0.7).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 1.5
                       * max(update_rows, 1.0) ** 1.5
                       * max(abs_max, 1.0e-3) ** -0.7),
            residual,
        )
    if mode == "composite_t2_v3":
        # Phase 11ed t/r2 re-sweep after 11ec pool:
        # residual / (ratio^0.1 * rows^1.0 * abs_max^0.5), sim 85.099%.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.1
                       * max(update_rows, 1.0) ** 1.0
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_n1_v3":
        # Ultra-fine sim BEST for n/r1: residual / (rows^0.7 * abs_max^0.3)
        # sim 64.33% vs composite_n1_v2 64.31 (+0.02pp). a=0 (no ratio).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(update_rows, 1.0) ** 0.7
                       * max(abs_max, 1.0e-3) ** 0.3),
            residual,
        )
    if mode == "composite_n2_v3":
        # Ultra-fine sim BEST for n/r2: residual / (ratio^0.3 * rows^0.7 * abs_max^0.8)
        # sim 40.21% vs composite_n2_v2 40.14 (+0.07pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.3
                       * max(update_rows, 1.0) ** 0.7
                       * max(abs_max, 1.0e-3) ** 0.8),
            residual,
        )
    if mode == "composite_n2_v4":
        # Phase 11dm n/r2 re-sweep after 11dl pool:
        # residual / (ratio^0.2 * rows^0.3 * abs_max^0.8), sim 40.73%.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 0.3
                       * max(abs_max, 1.0e-3) ** 0.8),
            residual,
        )
    if mode in {"temporal_n2_v1", "temporal_n2_v2", "temporal_n2_v3"}:
        # Stateless fallback for diagnostics; the PF loop adds the temporal
        # previous-position penalty on top of this same base key.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 0.3
                       * max(abs_max, 1.0e-3) ** 0.8),
            residual,
        )
    if mode in {"temporal_hybdelta_t3_v1", "temporal_hybdelta_t3_v2", "temporal_hybdelta_t3_v3"}:
        return _rtkdiag_candidate_sort_key(row, mode="composite_t3_v2")
    if mode == "temporal_hybdelta_t3_v4":
        return _rtkdiag_candidate_sort_key(row, mode="composite_t3_v4")
    if mode == "temporal_hybdelta_n2_v1":
        return _rtkdiag_candidate_sort_key(row, mode="composite_n2_v4")
    if mode == "temporal_hybdelta_n3_v1":
        return _rtkdiag_candidate_sort_key(row, mode="composite_n3_v3")
    if mode == "temporal_hybdelta_n3_v2":
        return _rtkdiag_candidate_sort_key(row, mode="composite_n3_v4")
    if mode == "composite_n3_v3":
        # Ultra-fine sim BEST for n/r3: residual / (ratio^0.2 * rows^0.7 * abs_max^0.5)
        # sim 59.15% vs composite_n3_v2 59.10 (+0.05pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 0.7
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_n3_v4":
        # Phase 11ec n/r3 re-sweep after 11eb pool:
        # residual / (ratio^0.2 * rows^1.0 * abs_max^0.7), sim 62.02%.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 1.0
                       * max(abs_max, 1.0e-3) ** 0.7),
            residual,
        )
    # Conservative default: prefer the tightest measurement update residual.
    return (residual, -ratio)


def _rtkdiag_local_ungate_labels(
    windows: tuple[tuple[int, int, tuple[str, ...]], ...],
    epoch_idx: int,
) -> tuple[str, ...] | None:
    for start_idx, end_idx, labels in windows:
        if int(start_idx) <= int(epoch_idx) <= int(end_idx):
            return tuple(labels)
    return None


def _rtkdiag_local_ungate_labels_for_tow(
    windows: tuple[tuple[float, float, tuple[str, ...]], ...],
    tow: float,
) -> tuple[str, ...] | None:
    for start_tow, end_tow, labels in windows:
        if float(start_tow) <= float(tow) <= float(end_tow):
            return tuple(labels)
    return None


def _rtkdiag_fixed_output_ok(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    try:
        return int(row.get("output_added", "0")) == 1 and int(row.get("final_status", "0")) == 4
    except ValueError:
        return False


_RTKDIAG_POLICIES = {
    "phase10o", "phase10p", "phase10r",
    "phase11h", "phase11i", "phase11l", "phase11n",
    "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di", "phase11dk", "phase11dl", "phase11dm", "phase11dn", "phase11do", "phase11dp", "phase11dq", "phase11dr", "phase11ds", "phase11dt", "phase11du", "phase11dv", "phase11dw", "phase11dx", "phase11dy", "phase11dz", "phase11ea", "phase11eb", "phase11ec", "phase11ed", "phase11ee", "phase11ef", "phase11eg", "phase11eh", "phase11ei", "phase11ej", "phase11ek", "phase11el", "phase11em", "phase11en", "phase11eo", "phase11ep", "phase11eq", "phase11er", "phase11er_mrescue", "phase11er_ext", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue",
}


_NAGOYA_RUN2_PHASE11EQ_LABELS = {
    "fgo_v14_snr38",
    "full_ratio15_lock3_trustedseed_rtkout3oGem3",
    "dev_demo5_trusted_o3",
    "n2_nobds",
    "fgo_v1",
    "full_ratio15_lock3_trustedseed_rtkout3mlc1",
    "full_ratio15_lock3_trustedseed_rtkout5",
}


_NAGOYA_RUN2_PHASE11ER_EXT_LABELS = {
    *_NAGOYA_RUN2_PHASE11EQ_LABELS,
    "libgnss_ext_subset",
}


def _apply_rtkdiag_run_index_policy(
    variant: CTRBPFConfig,
    *,
    run: str,
    policy: str,
    city: str | None = None,
) -> CTRBPFConfig:
    if (
        policy not in _RTKDIAG_POLICIES
        or not bool(variant.enable_rtkdiag_pf_rescue)
    ):
        return variant
    if policy in {"phase11eq", "phase11er", "phase11er_mrescue", "phase11er_ext", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"}:
        if city == "nagoya" and run == "run2":
            max_to_hybrid_m = 60.0 if policy in {"phase11er_mrescue", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"} else 10.0
            float_labels = ("libgnss_ext_subset",) if policy in {"phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"} else ()
            status5_labels = ("fgo_v14_snr38",) if policy == "phase11er_ext_s5_mrescue" else ()
            return replace(
                variant,
                rtkdiag_candidate_select_mode="residual",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_max_to_hybrid_m=max_to_hybrid_m,
                rtkdiag_candidate_emit_mode="candidate",
                rtkdiag_candidate_fallback_mode="hybrid",
                rtkdiag_candidate_float_labels=float_labels,
                rtkdiag_candidate_float_residual_rms_max=1.0 if float_labels else 0.0,
                rtkdiag_candidate_float_abs_max=3.0 if float_labels else 0.0,
                rtkdiag_candidate_float_min_sats=8 if float_labels else 0,
                rtkdiag_candidate_status5_labels=status5_labels,
                rtkdiag_candidate_status5_tow_windows=(
                    (556116.0, 556122.4, status5_labels),
                ) if status5_labels else (),
                rtkdiag_candidate_status5_max_dt_s=0.2 if status5_labels else 0.0,
                rtkdiag_candidate_status5_residual_rms_max=1.0 if status5_labels else 0.0,
                rtkdiag_candidate_status5_min_sats=3 if status5_labels else 0,
            )
        return replace(
            variant,
            rtkdiag_candidate_emit_mode="candidate",
            rtkdiag_candidate_fallback_mode="hybrid",
        )
    if policy == "phase11ep":
        if city == "tokyo" and run == "run1":
            variant = replace(
                variant,
                rtkdiag_candidate_local_ungate_tow_windows=(
                    (188028.6, 188054.0, ()),
                    (188131.8, 188137.4, ()),
                    (188151.6, 188164.4, ()),
                    (188259.0, 188268.2, ()),
                    (188403.0, 188417.4, ()),
                    (188432.2, 188443.2, ()),
                    (188554.6, 188560.6, ()),
                    (189207.2, 189216.6, ()),
                ),
            )
        elif city == "tokyo" and run == "run2":
            variant = replace(
                variant,
                rtkdiag_candidate_local_ungate_tow_windows=((178248.4, 178255.4, ()),),
            )
        elif city == "nagoya" and run == "run1":
            variant = replace(
                variant,
                rtkdiag_candidate_local_ungate_tow_windows=(
                    (551063.2, 551076.2, ()),
                    (551106.6, 551113.8, ()),
                    (551315.0, 551349.8, ()),
                ),
            )
        elif city == "nagoya" and run == "run2":
            variant = replace(
                variant,
                rtkdiag_candidate_label_factors=(
                    (
                        "xd_fixedicb_raw_n2_icbsweep_5734_5773_l1p8_l2p0",
                        0.8,
                    ),
                ),
            )
        policy = "phase11eo"
    if policy == "phase11eo":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v8",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v10",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11en"
    if policy == "phase11en":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v7",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v9",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v6",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11em"
    if policy == "phase11em":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v8",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11el"
    if policy == "phase11el":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v7",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ek"
    if policy == "phase11ek":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v6",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v5",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ej"
    if policy == "phase11ej":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v6",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v5",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v4",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ei"
    if policy == "phase11ei":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v5",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11eh"
    if policy == "phase11eh":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v4",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v3",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11eg"
    if policy == "phase11eg":
        policy = "phase11ef"
    if policy == "phase11ef":
        policy = "phase11ee"
    if policy == "phase11ee":
        policy = "phase11ed"
    if policy == "phase11ed":
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="composite_t2_v3",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ec"
    if policy == "phase11ec":
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v2",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11eb"
    if policy == "phase11eb":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v4",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ea"
    if policy == "phase11ea":
        policy = "phase11dz"
    if policy == "phase11dz":
        policy = "phase11dy"
    if policy == "phase11dy":
        policy = "phase11dx"
    if policy == "phase11dx":
        policy = "phase11dw"
    if policy == "phase11dw":
        policy = "phase11dv"
    if policy == "phase11dv":
        policy = "phase11du"
    if policy == "phase11du":
        policy = "phase11dt"
    if policy == "phase11dt":
        policy = "phase11ds"
    if policy == "phase11ds":
        policy = "phase11dr"
    if policy == "phase11dr":
        policy = "phase11dq"
    if policy == "phase11dq":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v3",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v3",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11dp"
    if policy == "phase11dp":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v2",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v2",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11do"
    if policy == "phase11do":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v1",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n2_v1",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v1",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11dn"
    if policy == "phase11dn":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v1",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11dm"
    if policy == "phase11dm":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="composite_t3_v2",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="composite_n2_v4",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11di"
    if policy in {"phase11dk", "phase11dl"}:
        # Phase 11dk/dl keep Phase 11di selector/gate settings and only change
        # the per-run candidate allow-list in _filter_rtkdiag_candidates_by_policy.
        policy = "phase11di"
    if policy in {"phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        # Phase 11aa: extends Phase 11z by switching tokyo/run1 selector
        # to hybrid_anchor (consensus selector showed +0.20pp on this run /
        # +0.05pp aggregate). Other runs unchanged from Phase 11z.
        # Phase 11ab: same gates as 11aa, but r15ga (--glonass-ar autocal,
        # ratio 1.5) is added to the candidate pool — restricted via
        # blocked_labels to {(tokyo, run3), (nagoya, run1)} where offline
        # simulation showed positive aggregate (+88m / +0.19pp).
        # Phase 11ac: extends 11ab with two more selectively-eligible
        # candidates: r20ga (ratio2.0 + glonass-ar) for {(tokyo, run3),
        # (nagoya, run2)} and em10 (--elevation-mask-deg 10) for {(tokyo,
        # run3)}. Combined offline gain over 11ab is +81m / +0.18pp.
        # Phase 11ad: extends 11ac with two more candidates restricted to
        # {(tokyo, run3)}: psig1 (--pseudorange-sigma 1.0) and holdrlx
        # (--min-hold-count 3 --hold-ratio-threshold 1.5). Offline shows
        # +146m on tokyo/run3 alone (additive).
        if city == "tokyo" and run == "run1":
            # Phase 11bw: switch t/r1 to residual mode (selector sweep predicted +164m
            # vs score in small candidate pool; check if PF realises the gain).
            if policy in {"phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=2.5,
                    rtkdiag_candidate_residual_rms_max=1.4,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11bu: switch t/r1 to score mode (confirmed +3.34pp / +344m on t/r1 alone)
            # — hybrid_anchor was underusing high-precision combos.
            if policy in {"phase11bu", "phase11bv"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=2.5,
                    rtkdiag_candidate_residual_rms_max=1.4,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="hybrid_anchor",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            # Phase 11dc/dd: composite_t2_v2 = residual / (ratio^0.2 * rows^2.0 * abs_max^0.5).
            # Fine sim BEST = 84.86% vs db 84.78 (+0.08pp). dd: ultra-fine confirmed same optimum.
            if policy in {"phase11dc", "phase11dd", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_t2_v2",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=10.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11db: composite_3axis_t2 = residual / (ratio^0.5 * rows^2.0).
            # 3-axis sim BEST = 84.78% vs score_per_row 84.53 (+0.25pp).
            if policy == "phase11db":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_3axis_t2",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=10.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cu/cv: switch t/r2 score → score_per_row.
            if policy in {"phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=10.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cc: try t/r2 score → residual (mirror t/r1 pattern).
            if policy == "phase11cc":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=10.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            # Phase 11cu/cv: switch t/r3 score → score_per_row (sweep +0.12pp / +19.6m).
            if policy in {"phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cb: tighten ratio_min 1.0 → 1.7 for t/r3 score (rms_max kept 50).
            if policy == "phase11cb":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11ca: tighten rms_max 50 → 5 for t/r3 score (regressed -1.40pp).
            if policy == "phase11ca":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=5.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            # Phase 11dd: composite_n1_v3 = residual / (rows^0.7 * abs_max^0.3).
            # Ultra-fine sim BEST = 64.33% vs dc 64.31 (+0.02pp).
            if policy in {"phase11dd", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n1_v3",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11dc: composite_n1_v2 = residual / (rows^0.5 * abs_max^0.3).
            # Fine sim BEST = 64.31% vs db 64.25 (+0.06pp).
            if policy == "phase11dc":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n1_v2",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11db: composite_3axis_n1 = residual / (rows^0.5 * abs_max^0.5) (no ratio).
            # 3-axis sim BEST = 64.25% vs rms_per_row 63.89 (+0.37pp).
            if policy == "phase11db":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_3axis_n1",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cu/cv: switch n/r1 nrows → rms_per_row (sweep +0.14pp / +6.2m).
            if policy in {"phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="rms_per_row",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11bz: try n/r1 nrows → ratio (max ratio = highest ambiguity confidence).
            if policy == "phase11bz":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="ratio",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11by: try n/r1 nrows → residual (mirror t/r1 success).
            # Tight rms_max=1.0 → residual mode should pick the truly-best rms candidate.
            if policy == "phase11by":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11bv: try switching n/r1 from nrows to score (mirror t/r1 success).
            if policy == "phase11bv":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            # Phase 11dh: dd composite_n2_v3 + emit_mode="pf" + recenter=2.0m.
            # Test: if candidate > 2m from PF, skip recenter, drift_skip → emit hybrid.
            # Goal: filter outlier candidates by hybrid agreement.
            if policy == "phase11dh":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n2_v3",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="pf",
                    rtkdiag_candidate_recenter_max_shift_m=2.0,
                )
            # Phase 11dd: composite_n2_v3 = residual / (ratio^0.3 * rows^0.7 * abs_max^0.8).
            # Ultra-fine sim BEST = 40.21% vs dc 40.14 (+0.07pp).
            if policy in {"phase11dd", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n2_v3",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11dc: composite_n2_v2 = residual / (ratio^0.4 * rows^1.0 * abs_max^0.7).
            # Fine sim BEST = 40.14% vs db 39.92 (+0.22pp).
            if policy == "phase11dc":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n2_v2",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11da/db: composite_3axis_n2 = residual / (ratio^0.5 * rows^1.5 * abs_max^0.5).
            # 3-axis sim BEST = 39.92% vs cy 39.18% (+0.74pp).
            if policy in {"phase11da", "phase11db"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_3axis_n2",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cv/cz: switch n/r2 score → score_per_row2 (alpha sweep b=2、sim 39.43% vs 39.18%、+0.25pp).
            if policy in {"phase11cv", "phase11cz"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row2",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cu/cy: switch n/r2 score → score_per_row (filter fix 後 PF +0.06pp 確認).
            if policy in {"phase11cu", "phase11cy"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cp: try ratio mode on n/r2 (untested).
            if policy == "phase11cp":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="ratio",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11co: try consensus5 (median-anchored) on n/r2.
            if policy == "phase11co":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="consensus5",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cn: try wavg3 fusion mode on n/r2 (top 3 candidates weighted avg).
            if policy == "phase11cn":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="wavg3",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cj: relax sigma_m 0.02 → 1.0 on n/r2 (PF takes candidate weakly).
            if policy == "phase11cj":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                    rtkdiag_candidate_sigma_m=1.0,
                )
            # Phase 11ci: relax emit_max_diff_m 0.4 → 2.0 on n/r2 (no effect).
            if policy == "phase11ci":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                    rtkdiag_candidate_emit_max_diff_m=2.0,
                )
            # Phase 11bx: try n/r2 score → residual (regressed -4.49pp).
            if policy == "phase11bx":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            # Phase 11cv: switch n/r3 score → score_per_row3 (alpha sweep +1.06pp at b=3).
            if policy == "phase11cv":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row3",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11dd: n/r3 composite_n3_v3 = residual / (ratio^0.2 * rows^0.7 * abs_max^0.5).
            # Ultra-fine sim BEST = 59.15% vs dc 59.10 (+0.05pp).
            if policy in {"phase11dd", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n3_v3",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11dc: n/r3 composite_n3_v2 = residual / (ratio^0.2 * rows^0.5 * abs_max^0.5).
            # Fine sim BEST = 59.10% vs db 59.04 (+0.05pp).
            if policy == "phase11dc":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n3_v2",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11da/db: n/r3 composite_3axis_n2 (3-axis sim 59.04% vs cy 58.85%、+0.19pp).
            if policy in {"phase11da", "phase11db"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_3axis_n2",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cx/cy: n/r3 score → score_per_row3 (alpha sweep +1.06pp at b=3、PF +0.25pp 確認).
            if policy in {"phase11cx", "phase11cy", "phase11cz"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row3",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cu/cw: switch n/r3 score → score_per_row (sweep +0.81pp / +27m).
            if policy in {"phase11cu", "phase11cw"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cd: try n/r3 score → residual (last selector permutation).
            if policy == "phase11cd":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11z logic if city unknown.
    if policy == "phase11z":
        # Phase 11z: extends Phase 11y by pushing rms_max to 50 on
        # tokyo/run3 + nagoya/run2 and rms_max=30 on nagoya/run3 (extended2
        # gate sweep showed plateau between rms=50 and rms=100, with most
        # gain on nagoya/run2 +0.79pp).
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11y/phase11x logic if city unknown.
    if policy == "phase11y":
        # Phase 11y: extends Phase 11x by widening rms_max to 20 on the
        # heavy-NLOS runs (tokyo/run3, nagoya/run2, nagoya/run3) and
        # dropping ratio_min to 1.0 where the extended gate sweep showed
        # selector improvement. Other runs unchanged from Phase 11x.
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11x logic if city unknown.
    if policy == "phase11x":
        # Phase 11x: city x run gate from offline gate sweep on Phase 11v
        # candidate pool. Selector mode is unchanged from phase11n.
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.3,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.5,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.5,
                rtkdiag_candidate_residual_rms_max=7.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through: city unknown, behave like phase11n.
    if run == "run1":
        return replace(
            variant,
            rtkdiag_candidate_select_mode="nrows",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=1.4,
            rtkdiag_candidate_emit_mode="candidate",
        )
    if run == "run2":
        if policy in {"phase10r", "phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa"}:
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=6.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        return replace(
            variant,
            rtkdiag_candidate_select_mode="maxabs",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=6.0 if policy == "phase10p" else 5.0,
            rtkdiag_candidate_emit_mode="candidate",
        )
    if run == "run3":
        return replace(
            variant,
            rtkdiag_candidate_select_mode="score",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=7.0 if policy in {"phase10r", "phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa"} else (
                6.0 if policy == "phase10p" else 5.0
            ),
            rtkdiag_candidate_emit_mode="candidate",
        )
    return variant


def _filter_rtkdiag_candidates_by_policy(
    candidates: list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]],
    *,
    city: str,
    run: str,
    policy: str,
    blocked_labels: set[str] | None = None,
) -> list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]]:
    if policy in {"phase11eq", "phase11er", "phase11er_mrescue", "phase11er_ext", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"}:
        # Phase 11er is the conservative n/r2 rescue found from the
        # lowcase audit: residual-select across a small candidate stack,
        # candidate emit, hybrid fallback, and a 10 m hybrid-distance gate.
        # phase11er_mrescue keeps the same whitelist but widens the gate to
        # 60 m in _apply_rtkdiag_run_index_policy for 3 m outlier reduction.
        # phase11er_ext variants add one libgnss++ extended/subset AR candidate.
        # phase11er_ext_float_mrescue also allows tightly gated FLOAT output
        # from that extended candidate only. phase11er_ext_s5_mrescue adds a
        # narrow, audited fgo_v14 status-5 bridge over the n/r2 tail gap.
        # On every other run, block all candidates so the rtkdiag variant is a
        # hybrid passthrough and cannot regress unrelated runs.
        if city == "nagoya" and run == "run2":
            allowed = (
                _NAGOYA_RUN2_PHASE11ER_EXT_LABELS
                if policy in {"phase11er_ext", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"}
                else _NAGOYA_RUN2_PHASE11EQ_LABELS
            )
            extra_blocked = set(blocked_labels or set())
            extra_blocked.update(
                label
                for label, _, _ in candidates
                if label not in allowed
            )
            return _filter_rtkdiag_candidates_by_policy(
                candidates,
                city=city,
                run=run,
                policy="phase11ep",
                blocked_labels=extra_blocked,
            )
        return []
    if policy == "phase11eo":
        # Phase 11eo changes only t/r3 and n/r2 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11en.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11en",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11en":
        # Phase 11en changes only t/r3, n/r2, and n/r3 selector label
        # penalties. Candidate pool and block rules are exactly Phase 11em.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11em",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11em":
        # Phase 11em changes only n/r2 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11el.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11el",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11el":
        # Phase 11el changes only n/r2 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11ek.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ek",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ek":
        # Phase 11ek changes only n/r2 and n/r3 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11ej.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ej",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ej":
        # Phase 11ej changes only t/r3, n/r2, and n/r3 selector label
        # penalties. Candidate pool and block rules are exactly Phase 11ei.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ei",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ei":
        # Phase 11ei changes only t/r3 selector label penalties. Candidate
        # pool and block rules are exactly Phase 11eh.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11eh",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11eh":
        # Phase 11eh changes only n/r2 and n/r3 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11eg.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11eg",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11eg":
        # Phase 11eg: Phase 11ef plus n/r3-only micro-add of
        # csig01_psig1, em5oG, and mlc2nobds. Combo replay on nagoya/run3
        # gave +2.494m.
        extra_blocked = set(blocked_labels or set())
        if (city, run) != ("nagoya", "run3"):
            extra_blocked.update({"csig01_psig1", "em5oG", "mlc2nobds"})
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ef",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11ef":
        # Phase 11ef: Phase 11ee plus t/r3-only micro-add of
        # rtkout5minobs3. Single-add replay on tokyo/run3 gave +1.817m.
        extra_blocked = set(blocked_labels or set())
        if (city, run) != ("tokyo", "run3"):
            extra_blocked.add("rtkout5minobs3")
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ee",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11ee":
        # Phase 11ee: Phase 11ed base plus n/r2-only micro-add of
        # csig005_em10 and onlyG_r05. Both labels are blocked elsewhere to
        # avoid widening the pool on runs that were not replay-positive.
        extra_blocked = set(blocked_labels or set())
        if (city, run) != ("nagoya", "run2"):
            extra_blocked.update({"csig005_em10", "onlyG_r05"})
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ed",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11ed":
        # Phase 11ed changes only tokyo/run2 selector parameters; candidate
        # blocks are exactly the Phase 11ec pool.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ec",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ec":
        # Phase 11ec changes only nagoya/run3 selector parameters; candidate
        # blocks are exactly the Phase 11eb pool.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11eb",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11eb":
        # Phase 11eb changes only tokyo/run3 selector parameters; candidate
        # blocks are exactly the Phase 11ea pool.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ea",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ea":
        # Phase 11ea: tenth selected-loss block pass on top of 11dz.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"r25g20"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dz",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dz":
        # Phase 11dz: ninth selected-loss block pass on top of 11dy.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"csig005_holdvrlx", "r20"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dy",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dy":
        # Phase 11dy: eighth selected-loss block pass on top of 11dx.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"r20ga"},
            ("nagoya", "run3"): {"r30"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dx",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dx":
        # Phase 11dx: seventh selected-loss block pass on top of 11dw.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run1"): {"c005p1"},
            ("nagoya", "run2"): {"em5mlc2oG"},
            ("nagoya", "run3"): {"mlc1c005r10", "r20", "r20g40", "r30g"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dw",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dw":
        # Phase 11dw: sixth selected-loss block pass on top of 11dv.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("tokyo", "run1"): {"r20g10", "r20g15"},
            ("nagoya", "run2"): {"onlyG"},
            ("nagoya", "run3"): {"r15g", "r15g20", "r25g20"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dv",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dv":
        # Phase 11dv: fifth selected-loss block pass on top of 11du.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"r20g40", "ratio12oG", "nobds"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11du",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11du":
        # Phase 11du: fourth selected-loss block pass on top of 11dt.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"r15g20", "r20g", "csig1"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dt",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dt":
        # Phase 11dt: third selected-loss block pass on top of 11ds.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("tokyo", "run3"): {"oGr05", "psig2"},
            ("nagoya", "run2"): {"mlc1c005"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ds",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11ds":
        # Phase 11ds: second selected-loss block pass on top of 11dr.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("tokyo", "run1"): {"csig05hvr", "r25g15"},
            ("tokyo", "run2"): {"r15nh"},
            ("tokyo", "run3"): {"mlc1oGc005p1", "c005ga", "r05", "oGc01p1"},
            ("nagoya", "run1"): {"rtkout10", "csig05_em10"},
            ("nagoya", "run2"): {"n2loose"},
            ("nagoya", "run3"): {"r15nh", "r20g", "csig05", "mlc1"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dr",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dr":
        # Phase 11dr: keep Phase 11dq temporal selectors, but block labels
        # that single-run replay showed were locally replaceable by better
        # candidates.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("tokyo", "run1"): {"oGp1hr", "csig05psh"},
            ("nagoya", "run1"): {"c005hr", "mlc1c005p1", "oGc01"},
            ("nagoya", "run2"): {"rtkout5", "rtkout5c005", "oGr05", "n2loose2"},
            ("tokyo", "run3"): {"csig01", "mlc1oG", "csig05ps"},
            ("nagoya", "run3"): {"r15g15", "em3mlc1oG", "psig1", "csig05hr", "oGp1"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dq",
            blocked_labels=extra_blocked,
        )
    if policy in {"phase11do", "phase11dp", "phase11dq"}:
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dn",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11dn":
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dm",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11dm":
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dl",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11dl":
        # 11dk + final run-local positives from discovered diag dirs and
        # post-11dk all-known replay.  Offline combo: +10.410m / +0.022471pp.
        allowed: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run1"): {"xd_r25_nohold", "csig05_em10"},
            ("nagoya", "run2"): {"csig05_psig1", "em5mlc2oG"},
            ("nagoya", "run3"): {"xd_n3_loose_hold4_ratio15_gate10_min6"},
            ("tokyo", "run1"): {"xd_ratio4", "xd_r2_nohold", "xd_r25_nohold", "r10c005p1"},
            ("tokyo", "run2"): {"xd_ratio3_gate10_min6", "em5c005p1"},
            ("tokyo", "run3"): {"csig01_holdvrlx"},
        }
        surgical_labels = set().union(*allowed.values())
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(surgical_labels - allowed.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dk",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dk":
        # 11di true-base + run-local positive extras discovered by
        # sim_ppc_phase_csv_addcand.py --allowed-pairs (delta_pass_m > 1m).
        allowed: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run1"): {"ratio12", "csig05_psig1_holdvrlx"},
            ("nagoya", "run2"): {"mlc1oGc0001", "mlc1r10oG", "rtkout5oG", "psig3", "csig005_holdvrlx", "ratio12oG"},
            ("nagoya", "run3"): {"mlc1c005r10em3", "mlc1", "csig05_holdrlx_em10", "r10", "r08"},
            ("tokyo", "run1"): {"oGc005p1hr", "c005p1hr", "oGc005p2", "mlc1r10c005p1", "em3oG", "oGc005hr", "csig005_holdvrlx"},
            ("tokyo", "run2"): {"oGc005p05", "mlc1oGp1"},
            ("tokyo", "run3"): {"csig05_r10", "csig01_holdrlx"},
        }
        surgical_labels = set().union(*allowed.values())
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(surgical_labels - allowed.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11di",
            blocked_labels=extra_blocked,
        )
    effective_blocked_labels = set(blocked_labels or set())
    # Phase 11ce: experimentally block mlc1oG on n/r2 (currently dominant
    # selection ~1729 epochs, ~42% of selected). Test if this drives selection
    # toward better candidates.
    if policy == "phase11ce" and city == "nagoya" and run == "run2":
        effective_blocked_labels.add("mlc1oG")
    # Phase 11cf: 5 new candidates restricted to n/r2 only (regressed -0.86pp on n/r2).
    if policy == "phase11cf" and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.update({"c005em10", "mlc1c005r10oG", "GE", "psig01", "nopostflt"})
    # Phase 11cg: 5 different candidates targeting n/r2 (regressed -0.30pp).
    if policy == "phase11cg" and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.update({"c005hvrlx", "mlc1oGc0001", "GEonly", "noarfilt", "c005minar3"})
    # Phase 11ch: open phase11cf candidates GLOBALLY (block only on n/r2 where they regressed).
    if policy == "phase11ch" and (city, run) == ("nagoya", "run2"):
        effective_blocked_labels.update({"c005em10", "mlc1c005r10oG", "GE", "psig01", "nopostflt"})
    # Phase 11ck: modeauto + modestatic (regressed -7.70pp on n/r2).
    if policy == "phase11ck" and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.update({"modeauto", "modestatic"})
    # Phase 11cl: psig005 + oGc005hr + r12oG candidates restricted to n/r2.
    if policy == "phase11cl" and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.update({"psig005", "oGc005hr", "r12oG"})
    # Phase 11dg: surgical addition of 3 NEW candidates with strict per-run allowance
    # based on per-epoch oracle pick frequency (sim_ppc_oracle_label_freq.py):
    # - xr25_glonassar: allowed only on tokyo/run1 (8.6%) and tokyo/run2 (10.1%)
    # - xmlc1psig005: allowed only on tokyo/run2 (12.2%)
    # - xcsig005_em10: allowed only on nagoya/run2 (9.0%)
    if policy == "phase11dg":
        if (city, run) not in {("tokyo", "run1"), ("tokyo", "run2")}:
            effective_blocked_labels.add("xr25_glonassar")
        if (city, run) != ("tokyo", "run2"):
            effective_blocked_labels.add("xmlc1psig005")
        if (city, run) != ("nagoya", "run2"):
            effective_blocked_labels.add("xcsig005_em10")
    # Phase 11di: narrower surgical version after replaying additions against
    # the real Phase 11dd per-run pools. 11dg let several oracle-frequent
    # labels into runs where the 11dd composite selector actually regressed.
    if policy == "phase11di":
        if (city, run) != ("tokyo", "run1"):
            effective_blocked_labels.update({"xr25_glonassar", "xcsig005_em10"})
        if (city, run) != ("nagoya", "run1"):
            effective_blocked_labels.add("xr17_glonassar")
        if (city, run) != ("tokyo", "run3"):
            effective_blocked_labels.add("xpsig05")
        effective_blocked_labels.update({"xmlc1psig005", "xnobds_holdrlx"})
    if policy in {"phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and city == "nagoya" and run == "run2":
        # Phase 11cm: skip this 11h-era block on n/r2 to test if old reasoning still holds.
        # Phase 11cu/cv/cw: re-apply block (the cm-cp experiment showed regression).
        effective_blocked_labels.update({"r15g15", "r20g15", "r25g15", "r30g15"})
    if policy in {"phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (
        (city, run) in {("tokyo", "run2"), ("nagoya", "run1"), ("nagoya", "run2")}
    ):
        effective_blocked_labels.update({"r30", "r30g"})
    if policy in {"phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (
        (city, run) in {("nagoya", "run1"), ("nagoya", "run2")}
    ):
        effective_blocked_labels.add("r20g10")
    if policy in {"phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and city == "nagoya":
        effective_blocked_labels.update({"r15g10", "r25g10"})
    # Phase 11ab: r15ga (--glonass-ar autocal ratio 1.5) only helps tokyo/run3
    # and nagoya/run1 in offline simulation; block on the other 4 runs to
    # avoid the -30/-19/-5/-2 m/run regressions seen in the sim.
    if policy in {"phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("r15ga")
    # Phase 11ac: r20ga only helps tokyo/run3 (+12.6m) and nagoya/run2
    # (+47.7m) on top of 11ab base. Block elsewhere.
    if policy in {"phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("r20ga")
    # Phase 11ac: em10 (elev mask 10°) only helps tokyo/run3 (+21m). Block
    # elsewhere. Phase 11af also allows em10 on nagoya/run1 (+2.0m offline).
    if policy in {"phase11ac", "phase11ad", "phase11ae"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("em10")
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("em10")
    # Phase 11ad: psig1 (--pseudorange-sigma 1.0) gives +125m only on
    # tokyo/run3. Phase 11af also allows psig1 on nagoya/run3 (+4.1m offline).
    if policy in {"phase11ad", "phase11ae"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig1")
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("psig1")
    # Phase 11ad: holdrlx (relaxed hold ambiguity) gives +72m only on
    # tokyo/run3.
    if policy in {"phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("holdrlx")
    # Phase 11ae: r12ga (--ratio 1.2 + --glonass-ar autocal) gives +103m
    # additive on tokyo/run3 only. Phase 11af keeps it (no harm).
    if policy in {"phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("r12ga")
    # Phase 11ae: psig2 (--pseudorange-sigma 2.0) gives +44.5m additive on
    # tokyo/run3 only. Phase 11af/ag keeps it (no harm).
    if policy in {"phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig2")
    # Phase 11af: psig1hr (psig1 + holdrlx combined config) gives +90.8m
    # additive on tokyo/run3 only. Negative on nagoya runs.
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig1hr")
    # Phase 11af: nobds (--no-beidou) gives +29.7m on nagoya/run2 (largest),
    # +9.8m on tokyo/run3, +4.2m on tokyo/run1. Negative on nagoya/run1, run3.
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run2"), ("tokyo", "run3"), ("tokyo", "run1")}:
        effective_blocked_labels.add("nobds")
    # Phase 11ag: csig05 (--carrier-phase-sigma 0.0005) gives massive gains:
    # tokyo/run3 +95.8m, nagoya/run2 +95.8m, tokyo/run1 +9.8m. Negative on
    # nagoya/run1/run3 originally; Phase 11am also enables nagoya/run3 (+2.6m
    # additive after the other csig05* variants).
    if policy in {"phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("tokyo", "run1")}:
        effective_blocked_labels.add("csig05")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("tokyo", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05")
    # Phase 11ag: noglo (--no-glonass) gives +40.5m on tokyo/run1 only.
    if policy in {"phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("noglo")
    # Phase 11ag: csig1 (--carrier-phase-sigma 0.001) gives +0.6m on
    # nagoya/run1. Phase 11ah extends to {nagoya/run1, nagoya/run2, tokyo/run3}
    # (additive after csig05).
    if policy == "phase11ag" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("csig1")
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run1"), ("nagoya", "run2"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig1")
    # Phase 11ah: rout30 (--rtk-update-outlier-threshold 30) gives +3.6m
    # on nagoya/run3 only.
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run3"):
        effective_blocked_labels.add("rout30")
    # Phase 11ah: rout20 (--rtk-update-outlier-threshold 20) gives +2.2m
    # on nagoya/run1 only.
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rout20")
    # Phase 11ai: csig05hr (csig05 + holdrlx combined) gives massive gains:
    # tokyo/run1 +509.8m, tokyo/run3 +147.1m, nagoya/run3 +128.3m. Block on
    # nagoya/run2 (-7.7m) and nagoya/run1 (+0.4m, near-zero).
    if policy in {"phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05hr")
    # Phase 11ai: csig05ps (csig05 + psig1 combined) gives +22.5m on
    # tokyo/run3 and +3.4m on nagoya/run1. Phase 11aj also enables on
    # nagoya/run3 (+11.7m additive).
    # Phase 11ai/ak: csig05ps (csig05 + psig1) for {tokyo/run3, nagoya/run1}.
    # Phase 11aj added nagoya/run3 (-0.12pp PF on that run, so removed in 11ak).
    if policy in {"phase11ai", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("csig05ps")
    if policy == "phase11aj" and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05ps")
    # Phase 11aj/ak: csig01 (--carrier-phase-sigma 0.0001). Phase 11ak narrows
    # to {tokyo/run3, nagoya/run2} only — nagoya/run3 was -0.12pp in PF.
    if policy == "phase11aj" and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig01")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("csig01")
    # Phase 11aj: rout100 hurt nagoya/run1 by -1.49pp in PF — block in phase11ak.
    if policy == "phase11aj" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rout100")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rout100")
    # Phase 11aj: csig05nb (csig05 + nobds) +0.8m offline but -0.02pp PF.
    # Block entirely in phase11ak.
    if policy == "phase11aj" and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("csig05nb")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("csig05nb")
    # Phase 11ak: csig05hvr (csig05 + holdvrlx very loose hold) gives +53.2m
    # on tokyo/run1 only. Phase 11am also enables tokyo/run3 (+29.4m offline).
    if policy in {"phase11ak", "phase11al"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("csig05hvr")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig05hvr")
    # Phase 11ak: csig05psh (csig05 + psig1 + holdrlx triple) gives +44m on
    # tokyo/run3 and +21.5m on nagoya/run3. Phase 11am also enables tokyo/run1 (+4.1m).
    if policy in {"phase11ak", "phase11al"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05psh")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3"), ("tokyo", "run1")}:
        effective_blocked_labels.add("csig05psh")
    # Phase 11ak: csig05em (csig05 + em10) gives +2.5m on nagoya/run1.
    if policy == "phase11ak" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("csig05em")
    if policy == "phase11al":
        effective_blocked_labels.add("csig05em")
    # Phase 11am: csig05em allowed only on tokyo/run3 (+20.5m offline). Block elsewhere.
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("csig05em")
    # Phase 11ak: csig01hr (csig01 + holdrlx) gives +42.8m on nagoya/run2.
    # Phase 11am widens to {tokyo/run1 +7.2m, tokyo/run3 +23.1m, nagoya/run2 +42.8m}.
    if policy in {"phase11ak", "phase11al"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("csig01hr")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run2"), ("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig01hr")
    # Phase 11an: c5p1hvr (csig05+psig1+holdvrlx 4-knob) gives +104.6m on
    # tokyo/run1 and +28.4m on nagoya/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("c5p1hvr")
    # Phase 11an: c1p1hr (csig01+psig1+holdrlx triple) gives +7.3m on tokyo/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("c1p1hr")
    # Phase 11an: c5hrem (csig05+holdrlx+em10 triple) gives +18.2m on tokyo/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("c5hrem")
    # Phase 11ao: csig005 (--carrier-phase-sigma 0.00005, super-tight) gives
    # +36.1m on nagoya/run2 (the lowest run). Marginal elsewhere.
    if policy in {"phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("csig005")
    # Phase 11ao: c5nbhr (csig05+nobds+holdrlx triple). Marginal +9.8m on
    # tokyo/run1 and +6.8m on nagoya/run1. Negative on others.
    if policy in {"phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c5nbhr")
    # Phase 11ap: c005hr (csig005+holdrlx) marginal +2.6m on tokyo/run1, +9.5m on nagoya/run1.
    if policy in {"phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c005hr")
    # Phase 11aq: r05 (--ratio 0.5) marginal +5.0/+3.9/+2.3m on tokyo/run1, tokyo/run3, nagoya/run1.
    if policy in {"phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("r05")
    # Phase 11aq: c005ga (csig005+glonassar) marginal +1.8/+3.3m on tokyo/run1, tokyo/run3.
    if policy in {"phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("c005ga")
    # Phase 11ar: onlyG (--no-glonass --no-beidou, only G+E+J). +9.4m tokyo/run1
    # and **+20.2m on nagoya/run2** (the lowest-scoring run). Negative on nagoya/run3.
    if policy in {"phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run2")}:
        effective_blocked_labels.add("onlyG")
    # Phase 11as: oGc05 (onlyG + csig05) +14.3m on tokyo/run1.
    if policy in {"phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("oGc05")
    # Phase 11as: oGc005 (onlyG + csig005) +10.4m on nagoya/run2.
    if policy in {"phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("oGc005")
    # Phase 11as: oGp1 (onlyG + psig1) +2.1/+3.0m on nagoya/run1, nagoya/run3.
    if policy in {"phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("oGp1")
    # Phase 11at: oGp1hr (onlyG + psig1 + holdrlx) +15.4m on tokyo/run1.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("oGp1hr")
    # Phase 11at: oGp1c05 (onlyG + psig1 + csig05) +9.2m nagoya/run3, +2.0m nagoya/run1.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("oGp1c05")
    # Phase 11at: oGr05 (onlyG + ratio 0.5) marginal +2.2/+3.7m on tokyo/run3, nagoya/run2.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("oGr05")
    # Phase 11at: oGc01 (onlyG + csig01) marginal +3.1m on nagoya/run1.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("oGc01")
    # Phase 11at: oGem10 (onlyG + em10) marginal +1.5m/+1.1m on tokyo/run3, nagoya/run3.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("oGem10")
    # Phase 11au: oGc005p1 (onlyG + csig005 + psig1) +37m tokyo/run2, +6.6m tokyo/run1, +1.2m nagoya/run1.
    if policy in {"phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run2"), ("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("oGc005p1")
    # Phase 11au: c005p1 (csig005 + psig1, no onlyG) +13.9m tokyo/run1, +15.1m tokyo/run2, +1.5m nagoya/run1.
    if policy in {"phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c005p1")
    # Phase 11au: oGc01p1 (onlyG + csig01 + psig1) +21.4m tokyo/run2, +3.4m tokyo/run3.
    if policy in {"phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run2"), ("tokyo", "run3")}:
        effective_blocked_labels.add("oGc01p1")
    # Phase 11aw: oGc00005p1 (onlyG + csig 0.00005 + psig 1) +16.7m tokyo/run1.
    if policy in {"phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("oGc00005p1")
    # Phase 11aw: oGc0001p1 (onlyG + csig 0.0001 + psig 1) +16.3m tokyo/run2, +5.4m tokyo/run1.
    if policy in {"phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run2"), ("tokyo", "run1")}:
        effective_blocked_labels.add("oGc0001p1")
    # Phase 11aw: nobdsc005p1 (no-beidou + csig005 + psig1) +2.7m nagoya/run3.
    if policy in {"phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run3"):
        effective_blocked_labels.add("nobdsc005p1")
    # Phase 11ay: em5 (--elevation-mask-deg 5) +13.8m nagoya/run1, +8.0m tokyo/run2.
    if policy in {"phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run1"), ("tokyo", "run2")}:
        effective_blocked_labels.add("em5")
    # Phase 11ay: mlc2oG (--min-lock-count 2 + onlyG) +7.5m tokyo/run1, +1.3m tokyo/run3.
    if policy in {"phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("mlc2oG")
    # Phase 11ba: mlc1oG (--min-lock-count 1 + onlyG) +11.3m t/r2, +9.3m t/r3, +3.6m n/r2.
    if policy in {"phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run2"), ("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("mlc1oG")
    # Phase 11ba: em3 (--elev-mask-deg 3) +5.4m nagoya/run1.
    if policy in {"phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("em3")
    # Phase 11bc: mlc1oGc005p1 (mlc1+onlyG+csig005+psig1) positive 5 runs, n/r2 only loser.
    if policy in {"phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run1"), ("nagoya", "run3"), ("tokyo", "run3"),
    }:
        effective_blocked_labels.add("mlc1oGc005p1")
    # Phase 11bc: em3mlc1oG (em3+mlc1+onlyG) n/r3 +12m, t/r3 +6m, n/r1 +1m.
    if policy in {"phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run3"), ("nagoya", "run3"), ("nagoya", "run1"),
    }:
        effective_blocked_labels.add("em3mlc1oG")
    # Phase 11bc: mlc1oGc005 (mlc1+onlyG+csig005) n/r2 +7m, t/r2 +3m, t/r3 +5m.
    if policy in {"phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("nagoya", "run2"), ("tokyo", "run2"), ("tokyo", "run3"),
    }:
        effective_blocked_labels.add("mlc1oGc005")
    # Phase 11be: mlc1c005p1 (mlc1+csig005+psig1, no onlyG) +5m n/r1, +4m n/r3, +2m n/r2.
    # Phase 11bf: drop n/r2 (lost -3.36m PF, displaced existing winners).
    _mlc1c005p1_runs = {("nagoya", "run1"), ("nagoya", "run3"), ("nagoya", "run2")}
    if policy in {"phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        _mlc1c005p1_runs = {("nagoya", "run1"), ("nagoya", "run3")}
    if policy in {"phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in _mlc1c005p1_runs:
        effective_blocked_labels.add("mlc1c005p1")
    # Phase 11be: mlc1oGc005em3 (mlc1+onlyG+csig005+em3) n/r2 +5m, t/r3 +4m, t/r1 +1m.
    # Phase 11bf: drop n/r2 (suspected over-trust contributor to -3.36m loss).
    _mlc1oGc005em3_runs = {("nagoya", "run2"), ("tokyo", "run3"), ("tokyo", "run1")}
    if policy in {"phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        _mlc1oGc005em3_runs = {("tokyo", "run3"), ("tokyo", "run1")}
    if policy in {"phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in _mlc1oGc005em3_runs:
        effective_blocked_labels.add("mlc1oGc005em3")
    # Phase 11be: mlc1oGc005r12 (mlc1+onlyG+csig005+ratio 1.2) t/r3 +5m, t/r1 +1m.
    if policy in {"phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run3"), ("tokyo", "run1"),
    }:
        effective_blocked_labels.add("mlc1oGc005r12")
    # Phase 11be: mlc1nobds (mlc1+no-beidou) t/r1 +5m.
    if policy in {"phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("mlc1nobds")
    # Phase 11bh: mlc1c005r10 (mlc1+csig005+ratio 1.0) +14m n/r3, +3m t/r1, +1.5m t/r2.
    if policy in {"phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("nagoya", "run3"), ("tokyo", "run1"), ("tokyo", "run2"),
    }:
        effective_blocked_labels.add("mlc1c005r10")
    # Phase 11bh: mlc1r10 (mlc1+ratio 1.0) +8m n/r1, +6.5m t/r3.
    if policy in {"phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("nagoya", "run1"), ("tokyo", "run3"),
    }:
        effective_blocked_labels.add("mlc1r10")
    # Phase 11bh: mlc1c005 (mlc1+csig005) +4.2m n/r2 (rare positive).
    if policy in {"phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("mlc1c005")
    # Phase 11bk: rtkout5 (--rtk-update-outlier-threshold 5) HUGE on tokyo/run1 (+74m, +0.72pp).
    # Also tokyo/run2 +6.94m, nagoya/run2 +2.91m. Other runs negative.
    if policy in {"phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run2"),
    }:
        effective_blocked_labels.add("rtkout5")
    # Phase 11bk: rtkout10 (--rtk-update-outlier-threshold 10) +3.95m nagoya/run1.
    if policy in {"phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rtkout10")
    # Phase 11bl: rtkout3 (--rtk-update-outlier-threshold 3) HUGE on tokyo/run1 (+167m, +1.62pp).
    # Marginal positive on nagoya/run2 (+1.5m). Other runs negative.
    if policy in {"phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run1"), ("nagoya", "run2"),
    }:
        effective_blocked_labels.add("rtkout3")
    # Phase 11bl: rtkout7 marginal positive on nagoya/run1 (+4.2m).
    if policy in {"phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rtkout7")
    # Phase 11bm: rtkout1 (--rtk-update-outlier-threshold 1) HUGE on tokyo/run1 (+175.5m, +1.70pp)
    # and tokyo/run3 (+14.4m). Other runs negative.
    if policy in {"phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run1"), ("tokyo", "run3"),
    }:
        effective_blocked_labels.add("rtkout1")
    # Phase 11bm: rtkout5c005 (rtkout5 + carrier-phase-sigma 0.0005) +15.0m nagoya/run2.
    if policy in {"phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("rtkout5c005")
    # Phase 11bm: rtkout5em3 (rtkout5 + elevation-mask-deg 3) +13.1m t/r2, +4.2m n/r1.
    if policy in {"phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run2"), ("nagoya", "run1"),
    }:
        effective_blocked_labels.add("rtkout5em3")
    # Phase 11bn: rtkout2/rtkout4/rtkout3c005 are not winners — block from all runs.
    # Were leaked into pool by phase 11bm dirs/labels but no per-run target; caused
    # -113m on t/r3 and -67m on n/r2 displacement in 11bm.
    if policy in {"phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout2")
        effective_blocked_labels.add("rtkout4")
        effective_blocked_labels.add("rtkout3c005")
    # Phase 11bo: rtkout3oG (rtkout3 + no-glonass + no-beidou) +61.5m on tokyo/run1.
    # Other runs negative.
    if policy in {"phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout3oG")
    # Phase 11bo: rtkout3em3 / rtkout3minobs3 are non-winners on every run; block globally.
    if policy in {"phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout3em3")
        effective_blocked_labels.add("rtkout3minobs3")
    # Phase 11bp: rtkout1c005oG (rtkout1 + carrier-phase-sigma 0.0005 + no-glonass/beidou)
    # +125.1m on tokyo/run1 (best variant in phase 11bp sweep). Other runs negative.
    if policy in {"phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout1c005oG")
    # Phase 11bp: rtkout5oGc005 (rtkout5 + no-glonass/beidou + csig005) +35.3m on tokyo/run3
    # (rare positive on this run). Other runs negative.
    if policy in {"phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("rtkout5oGc005")
    # Phase 11bp: rtkout1c005 +2.8m on n/r2 (marginal); also strong on t/r1 (+111m) but rtkout1c005oG
    # is even better there, so target n/r2 only.
    if policy in {"phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("rtkout1c005")
    # Phase 11bp: non-winner combos (rtkout1oG, rtkout1em3, rtkout1minobs3, rtkout10minobs3) — block globally.
    if policy in {"phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout1oG")
        effective_blocked_labels.add("rtkout1em3")
        effective_blocked_labels.add("rtkout1minobs3")
        effective_blocked_labels.add("rtkout10minobs3")
    # Phase 11bq: rtkout3c005oG (rtkout3 + csig005 + no-glonass/beidou) +404m on tokyo/run1 (HUGE).
    # Other runs marginally negative. Best single variant of session.
    if policy in {"phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout3c005oG")
    # Phase 11bq: rtkout5c005em3 (rtkout5 + csig005 + em3) +18m on n/r3 (first n/r3 winner) and
    # +4.5m on n/r2.
    if policy in {"phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("rtkout5c005em3")
    # Phase 11bq: non-winner combos — block globally.
    if policy in {"phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout1c005em3")
        effective_blocked_labels.add("rtkout1oGc005em3")
        effective_blocked_labels.add("rtkout3c005em3")
        effective_blocked_labels.add("rtkout1minobs4")
        effective_blocked_labels.add("rtkout1minobs6")
    # Phase 11br: rtkout3oGem3 (rtkout3 + no-glonass/beidou + em3) +73.5m on tokyo/run1 (offline,
    # vs phase11bp base — additive on top of phase11bq's rtkout3c005oG depends on epoch overlap).
    if policy in {"phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout3oGem3")
    # Block rtkout3oGem3 globally for phase11bq (kept as phase11br-only target).
    if policy == "phase11bq":
        effective_blocked_labels.add("rtkout3oGem3")
    # Phase 11bt: rtkout3mlc1c005 (rtkout3 + mlc1 + csig005) +278.1m on tokyo/run1 (offline winner).
    if policy in {"phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout3mlc1c005")
    # Phase 11bt: rtkout5mlc1c005oG (rtkout5 + mlc1 + csig005 + no-glonass/beidou)
    # +26.6m on t/r3 and +11.5m on n/r2 — winner on TWO runs.
    if policy in {"phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("rtkout5mlc1c005oG")
    # Phase 11bt: rtkout3mlc1 +17.1m on n/r3 (third n/r3 winner of session).
    if policy in {"phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run3"):
        effective_blocked_labels.add("rtkout3mlc1")
    # Phase 11bt: non-winner combos — block globally.
    if policy in {"phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout1mlc1")
        effective_blocked_labels.add("rtkout5mlc1")
        effective_blocked_labels.add("rtkout3mlc1oG")
        effective_blocked_labels.add("rtkout1mlc1c005")
    if not effective_blocked_labels:
        return candidates
    return [
        candidate for candidate in candidates
        if candidate[0] not in effective_blocked_labels
    ]
