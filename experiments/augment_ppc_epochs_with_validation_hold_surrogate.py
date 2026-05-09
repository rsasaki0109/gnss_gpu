#!/usr/bin/env python3
"""Add deployable epoch-level validation/hold surrogate features.

This is a predictor-side RTK-like state model.  It deliberately avoids demo5
labels and RTKLIB solver internals.  Unlike the earlier pseudo-solver state,
continuity alone is not allowed to create carry readiness: a validation-quality
condition must pass before hold state can accumulate.

The hold cooldown is intentionally short.  Longer bad-observation memory is
emitted as separate reject features so window-level models can learn whether a
single spike should suppress an otherwise good hold state.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_EPOCHS_CSV = (
    RESULTS_DIR
    / "ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_epochs.csv"
)
DEFAULT_PREFIX = "ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_validationhold"


def _float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_finite(*values: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else 0.0


def _min_finite(*values: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else float("nan")


def _pos(value: float) -> float:
    return max(0.0, value) if math.isfinite(value) else 0.0


def _ewma(previous: float, value: float, dt: float, tau_s: float) -> float:
    if not math.isfinite(previous):
        return value
    if dt <= 0.0:
        return previous
    alpha = 1.0 - math.exp(-dt / max(tau_s, 1.0))
    return (1.0 - alpha) * previous + alpha * value


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        print(f"saved empty: {path}")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def _inputs(row: dict[str, str]) -> dict[str, float]:
    phase_present = _float(row.get("rinex_phase_present_count"), 0.0)
    phase30 = _float(row.get("rinex_phase_streak_ge30p0s_count"), 0.0)
    phase60 = _float(row.get("rinex_phase_streak_ge60p0s_count"), 0.0)
    gf30 = _float(row.get("rinex_gf_streak_ge30p0s_count"), 0.0)
    gf60 = _float(row.get("rinex_gf_streak_ge60p0s_count"), 0.0)

    sim_los30 = _max_finite(
        _float(row.get("sim_los_cont_ge30p0s_count"), 0.0),
        _float(row.get("sim_los_system_g_cont_ge30p0s_count"), 0.0),
    )
    sim_los60 = _max_finite(
        _float(row.get("sim_los_cont_ge60p0s_count"), 0.0),
        _float(row.get("sim_los_system_g_cont_ge60p0s_count"), 0.0),
    )
    sim_adop30 = _float(row.get("sim_adop_cont_ge30p0s_count"), 0.0)
    sim_adop60 = _float(row.get("sim_adop_cont_ge60p0s_count"), 0.0)
    sim_adop_los = _float(row.get("sim_adop_los_count"), 0.0)
    sim_adop_all = _float(row.get("sim_adop_all_count"), 0.0)
    sim_adop_mean_el = _max_finite(
        _float(row.get("sim_adop_cont_ge30p0s_mean_el_deg"), 0.0),
        _float(row.get("sim_adop_los_mean_el_deg"), 0.0),
    )
    log_adop = _min_finite(
        _float(row.get("sim_adop_cont_ge30p0s_log10_adop"), 3.0),
        _float(row.get("sim_adop_cont_ge60p0s_log10_adop"), 3.0),
        _float(row.get("sim_adop_los_log10_adop"), 3.0),
        _float(row.get("sim_adop_all_log10_adop"), 3.0),
    )

    jump025 = _float(row.get("rinex_phase_jump_ge0p25cy_count"), 0.0)
    jump05 = _float(row.get("rinex_phase_jump_ge0p5cy_count"), 0.0)
    jump10 = _float(row.get("rinex_phase_jump_ge1p0cy_count"), 0.0)
    gf_slip02 = _float(row.get("rinex_gf_slip_ge0p2m_count"), 0.0)
    gf_slip05 = _float(row.get("rinex_gf_slip_ge0p5m_count"), 0.0)
    break_count = _float(row.get("rinex_phase_break_count"), 0.0)
    lost_count = _float(row.get("rinex_phase_lost_count"), 0.0)
    lli_count = _float(row.get("rinex_phase_lli_count"), 0.0)
    raw_p50 = _float(row.get("rinex_phase_raw_delta_cycles_p50"), 0.0)
    raw_p90 = _float(row.get("rinex_phase_raw_delta_cycles_p90"), 0.0)
    raw_max = _float(row.get("rinex_phase_raw_delta_cycles_max"), 0.0)
    dop_min_p50 = _float(row.get("rinex_phase_doppler_min_residual_cycles_p50"), 0.0)
    dop_min_p90 = _float(row.get("rinex_phase_doppler_min_residual_cycles_p90"), 0.0)
    dop_min_max = _float(row.get("rinex_phase_doppler_min_residual_cycles_max"), 0.0)

    continuity = _max_finite(phase30, gf30, sim_adop30, sim_adop_all)
    long_continuity = _max_finite(phase60, gf60, sim_adop60)
    los_support = _max_finite(sim_los30, sim_los60, sim_adop_los)
    clean_phase = max(0.0, phase_present - 1.5 * jump05 - gf_slip02 - 0.5 * lli_count)
    geometry_bonus = 0.02 * sim_adop_mean_el + _pos(-log_adop)
    quality = continuity + 0.45 * long_continuity + 0.30 * los_support + 0.12 * clean_phase + geometry_bonus

    count_block = (
        0.6 * jump025
        + 1.5 * jump05
        + 2.0 * jump10
        + 1.2 * gf_slip02
        + 1.8 * gf_slip05
        + 0.5 * break_count
        + 0.5 * lost_count
        + 0.35 * lli_count
    )
    raw_block = _pos((raw_p50 - 250.0) / 250.0) + _pos((raw_p90 - 500.0) / 500.0) + _pos((raw_max - 1000.0) / 1000.0)
    doppler_block = _pos((dop_min_p50 - 0.15) / 0.15) + _pos((dop_min_p90 - 0.50) / 0.50) + _pos((dop_min_max - 1.00) / 1.00)
    block = count_block + raw_block + doppler_block

    severe_block = (
        jump05 >= 3.0
        or jump10 >= 1.0
        or raw_p50 >= 600.0
        or raw_p90 >= 1000.0
        or raw_max >= 5000.0
        or dop_min_p90 >= 0.70
        or dop_min_max >= 1.50
    )
    hard_block = (
        jump05 >= 1.0
        or jump10 >= 1.0
        or gf_slip05 >= 1.0
        or raw_p50 >= 600.0
        or raw_p90 >= 1000.0
        or raw_max >= 5000.0
        or dop_min_p90 >= 0.70
        or dop_min_max >= 1.50
    )

    return {
        "continuity": continuity,
        "long_continuity": long_continuity,
        "los_support": los_support,
        "clean_phase": clean_phase,
        "geometry_bonus": geometry_bonus,
        "quality": quality,
        "count_block": count_block,
        "raw_block": raw_block,
        "doppler_block": doppler_block,
        "block": block,
        "hard_block": 1.0 if hard_block else 0.0,
        "severe_block": 1.0 if severe_block else 0.0,
        "jump05": jump05,
        "gf_slip02": gf_slip02,
        "raw_p50": raw_p50,
        "raw_p90": raw_p90,
        "dop_min_p90": dop_min_p90,
    }


def _augment_group(
    rows: list[dict[str, str]],
    *,
    max_gap_s: float,
    hard_cooldown_s: float,
    severe_cooldown_s: float,
    severe_memory_s: float,
) -> list[dict[str, object]]:
    rows_sorted = sorted(rows, key=lambda row: _float(row.get("gps_tow"), 0.0))
    prev_tow: float | None = None
    quality_ewma_30 = float("nan")
    quality_ewma_120 = float("nan")
    block_ewma_30 = float("nan")
    block_ewma_120 = float("nan")
    pass_ewma_30 = float("nan")
    recent_pass_s = 0.0
    recent_block_s = 0.0
    severe_recent_s = 0.0
    reject_recent_s = 0.0
    block_cooldown_s = 0.0
    hold_state = 0.0
    hold_age_s = 0.0
    hold_since_reset_s = 9999.0
    hold_reset_count = 0.0
    validation_pass_count = 0.0
    validation_block_count = 0.0
    clean_streak_s = 0.0
    strict_clean_streak_s = 0.0
    out_rows: list[dict[str, object]] = []

    for row in rows_sorted:
        out: dict[str, object] = dict(row)
        tow = _float(row.get("gps_tow"))
        if prev_tow is None or not math.isfinite(tow):
            dt = 0.0
            gap_reset = False
        else:
            dt = max(0.0, tow - prev_tow)
            gap_reset = dt > max_gap_s

        values = _inputs(row)
        quality_ewma_30 = _ewma(quality_ewma_30, values["quality"], dt, 30.0)
        quality_ewma_120 = _ewma(quality_ewma_120, values["quality"], dt, 120.0)
        block_ewma_30 = _ewma(block_ewma_30, values["block"], dt, 30.0)
        block_ewma_120 = _ewma(block_ewma_120, values["block"], dt, 120.0)

        severe_block = bool(values["severe_block"]) or gap_reset
        hard_block = bool(values["hard_block"]) or gap_reset
        block_spike = values["block"] >= 10.0
        reject_block = severe_block or block_spike
        if severe_block or values["block"] >= 10.0:
            block_cooldown_s = max(block_cooldown_s, severe_cooldown_s)
            severe_recent_s = max(severe_recent_s, severe_memory_s)
            reject_recent_s = max(reject_recent_s, severe_memory_s)
        elif hard_block or values["block"] >= 2.0:
            block_cooldown_s = max(block_cooldown_s, hard_cooldown_s)
            reject_recent_s = max(reject_recent_s, hard_cooldown_s)
        else:
            block_cooldown_s = max(0.0, block_cooldown_s - dt)
            severe_recent_s = max(0.0, severe_recent_s - dt)
            reject_recent_s = max(0.0, reject_recent_s - dt)
        validation_pass = (
            not hard_block
            and block_cooldown_s <= 0.0
            and values["quality"] >= 5.0
            and quality_ewma_30 >= 4.5
            and values["continuity"] >= 3.0
            and values["block"] <= 1.25
            and block_ewma_30 <= 0.90
        )
        validation_soft_pass = (
            not hard_block
            and block_cooldown_s <= 2.0
            and values["quality"] >= 4.0
            and values["continuity"] >= 2.0
            and values["block"] <= 1.75
            and block_ewma_30 <= 1.25
        )
        pass_ewma_30 = _ewma(pass_ewma_30, 1.0 if validation_pass else 0.0, dt, 30.0)

        if validation_pass:
            validation_pass_count += 1.0
            recent_pass_s += dt
        elif validation_soft_pass:
            recent_pass_s = max(0.0, recent_pass_s + 0.25 * dt)
        else:
            recent_pass_s = max(0.0, recent_pass_s - 1.5 * dt)

        if hard_block or values["block"] >= 2.0 or block_cooldown_s > 0.0:
            validation_block_count += 1.0
            recent_block_s += dt
        else:
            recent_block_s = max(0.0, recent_block_s - dt)

        if hard_block or block_cooldown_s > 0.0:
            hold_state = 0.0
            hold_age_s = 0.0
            hold_since_reset_s = 0.0
            if hard_block:
                hold_reset_count += 1.0
        elif validation_pass:
            hold_state = min(1.0, hold_state + dt / 20.0)
            hold_age_s += dt
            hold_since_reset_s = min(9999.0, hold_since_reset_s + dt)
        elif validation_soft_pass and hold_state > 0.0:
            hold_state = min(1.0, hold_state + dt / 80.0)
            hold_age_s += 0.5 * dt
            hold_since_reset_s = min(9999.0, hold_since_reset_s + dt)
        else:
            decay = dt / 60.0 if values["block"] <= 1.5 else dt / 25.0
            hold_state = max(0.0, hold_state - decay)
            hold_age_s = max(0.0, hold_age_s - 0.5 * dt)
            hold_since_reset_s = min(9999.0, hold_since_reset_s + dt)

        if hard_block or block_spike or gap_reset or reject_block:
            clean_streak_s = 0.0
        else:
            clean_streak_s += dt
        if validation_pass:
            strict_clean_streak_s += dt
        elif hard_block or block_spike or gap_reset or reject_block or not validation_soft_pass:
            strict_clean_streak_s = 0.0
        # soft_pass alone neither grows nor resets strict streak

        low_block_factor = math.exp(-min(block_ewma_120 if math.isfinite(block_ewma_120) else 0.0, 8.0) / 2.5)
        validation_confidence = max(0.0, min(1.0, pass_ewma_30 if math.isfinite(pass_ewma_30) else 0.0))
        hold_carry_score = hold_state * max(0.0, quality_ewma_120 if math.isfinite(quality_ewma_120) else 0.0) * low_block_factor
        hold_ready = (
            hold_state >= 0.35
            and recent_pass_s >= 5.0
            and block_cooldown_s <= 0.0
            and block_ewma_30 <= 1.25
        )
        strict_hold_ready = (
            hold_state >= 0.65
            and recent_pass_s >= 12.0
            and block_cooldown_s <= 0.0
            and block_ewma_30 <= 0.75
        )

        out.update(
            {
                "validation_quality_score": values["quality"],
                "validation_quality_ewma_30s": quality_ewma_30 if math.isfinite(quality_ewma_30) else 0.0,
                "validation_quality_ewma_120s": quality_ewma_120 if math.isfinite(quality_ewma_120) else 0.0,
                "validation_block_score": values["block"],
                "validation_block_count_score": values["count_block"],
                "validation_raw_block_score": values["raw_block"],
                "validation_doppler_block_score": values["doppler_block"],
                "validation_block_ewma_30s": block_ewma_30 if math.isfinite(block_ewma_30) else 0.0,
                "validation_block_ewma_120s": block_ewma_120 if math.isfinite(block_ewma_120) else 0.0,
                "validation_hard_block": 1.0 if hard_block else 0.0,
                "validation_severe_block": 1.0 if severe_block else 0.0,
                "validation_block_spike": 1.0 if block_spike else 0.0,
                "validation_reject_block": 1.0 if reject_block else 0.0,
                "validation_gap_reset": 1.0 if gap_reset else 0.0,
                "validation_pass": 1.0 if validation_pass else 0.0,
                "validation_soft_pass": 1.0 if validation_soft_pass else 0.0,
                "validation_pass_ewma_30s": pass_ewma_30 if math.isfinite(pass_ewma_30) else 0.0,
                "validation_recent_pass_s": recent_pass_s,
                "validation_recent_block_s": recent_block_s,
                "validation_severe_recent_s": severe_recent_s,
                "validation_reject_recent_s": reject_recent_s,
                "validation_block_cooldown_s": block_cooldown_s,
                "validation_pass_count": validation_pass_count,
                "validation_block_count": validation_block_count,
                "validation_continuity_support": values["continuity"],
                "validation_long_continuity_support": values["long_continuity"],
                "validation_los_support": values["los_support"],
                "validation_clean_phase_support": values["clean_phase"],
                "hold_state": hold_state,
                "hold_age_s": hold_age_s,
                "hold_since_reset_s": hold_since_reset_s,
                "hold_reset_count": hold_reset_count,
                "hold_ready": 1.0 if hold_ready else 0.0,
                "hold_strict_ready": 1.0 if strict_hold_ready else 0.0,
                "hold_carry_score": hold_carry_score,
                "hold_validation_confidence": validation_confidence,
                "hold_state_x_quality_ewma_120s": hold_state * (quality_ewma_120 if math.isfinite(quality_ewma_120) else 0.0),
                "hold_state_x_low_block_factor": hold_state * low_block_factor,
                "clean_streak_s": clean_streak_s,
                "strict_clean_streak_s": strict_clean_streak_s,
            }
        )
        out_rows.append(out)
        if math.isfinite(tow):
            prev_tow = tow
    return out_rows


def augment_rows(
    rows: list[dict[str, str]],
    *,
    max_gap_s: float,
    hard_cooldown_s: float,
    severe_cooldown_s: float,
    severe_memory_s: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["city"], row["run"])].append(row)

    out_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (city, run), group_rows in sorted(grouped.items()):
        augmented = _augment_group(
            group_rows,
            max_gap_s=max_gap_s,
            hard_cooldown_s=hard_cooldown_s,
            severe_cooldown_s=severe_cooldown_s,
            severe_memory_s=severe_memory_s,
        )
        out_rows.extend(augmented)
        summary: dict[str, object] = {"city": city, "run": run, "epochs": len(augmented)}
        for name in (
            "validation_quality_score",
            "validation_block_score",
            "validation_pass",
            "validation_hard_block",
            "validation_severe_block",
            "validation_reject_block",
            "validation_recent_pass_s",
            "validation_recent_block_s",
            "validation_reject_recent_s",
            "hold_state",
            "hold_ready",
            "hold_strict_ready",
            "hold_carry_score",
        ):
            values = [_float(row.get(name), 0.0) for row in augmented]
            summary[f"{name}_mean"] = sum(values) / len(values) if values else 0.0
            summary[f"{name}_max"] = max(values) if values else 0.0
        summary_rows.append(summary)
    return out_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add validation/hold surrogate features to PPC epoch rows")
    parser.add_argument("--epochs-csv", type=Path, default=DEFAULT_EPOCHS_CSV)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--hard-cooldown-s", type=float, default=10.0)
    parser.add_argument("--severe-cooldown-s", type=float, default=60.0)
    parser.add_argument("--severe-memory-s", type=float, default=120.0)
    parser.add_argument("--results-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.epochs_csv.open(newline="", encoding="utf-8") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    out_rows, summary_rows = augment_rows(
        rows,
        max_gap_s=args.max_gap_s,
        hard_cooldown_s=args.hard_cooldown_s,
        severe_cooldown_s=args.severe_cooldown_s,
        severe_memory_s=args.severe_memory_s,
    )
    prefix = RESULTS_DIR / args.results_prefix
    _write_rows(prefix.with_name(prefix.name + "_epochs.csv"), out_rows)
    _write_rows(prefix.with_name(prefix.name + "_summary.csv"), summary_rows)


if __name__ == "__main__":
    main()
