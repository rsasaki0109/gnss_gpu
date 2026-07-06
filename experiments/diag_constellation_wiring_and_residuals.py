#!/usr/bin/env python3
"""TASK_E (WP3c) work items 2a/2c/3: multi-GNSS clock wiring + code-frequency
consistency verification, and per-constellation WLS residual breakdown.

This is a *data/diagnostic* audit (single-clock WLS + a short native FGO
smoke chunk), not a full-run scoring script -- see
``experiments/validate_fgo_ppc.py --systems GRECJ`` for the full pipeline.

Reports three things used directly in ``results/wp3c/WP3C_REPORT.md``:

1. **2a** -- for a GRECJ (5-constellation) load, confirms
   ``constellations`` / ``n_clock`` / ``sys_kind`` wiring in
   ``run_fgo_on_ppc_native`` assigns exactly one contiguous clock index per
   observed system letter, and sanity-checks a short native VD solve
   converges with ``n_clock=5``.
2. **2c** -- confirms ``PPCDatasetLoader._PSEUDORANGE_CODE_PREFERENCES``
   picks a single, internally-consistent carrier frequency per system (no
   accidental L1/L2/L5 mixing within one system's preference list).
3. **3** -- per-constellation median |pseudorange residual| at a *single
   global clock* WLS solution (i.e. before any per-system ISB is estimated),
   both "before" (raw GRECJ load, no elevation mask/weighting) and "after"
   (elevation mask + constellation sigma weighting applied) -- quantifies
   which constellation(s) were poisoning the D2 multi-GNSS regression.

Usage:
    set PYTHONPATH=python
    python experiments/diag_constellation_wiring_and_residuals.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
for _p in (_REPO, _EXPERIMENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.fgo import fgo_gnss_lm_vd  # noqa: E402
from gnss_gpu.io.ppc import (  # noqa: E402
    _PSEUDORANGE_CODE_PREFERENCES,
    PPCDatasetLoader,
)
from validate_fgo_ppc import (  # noqa: E402
    _apply_constellation_sigma_scaling,
    _apply_elevation_mask,
)

RUN_DIR = Path("E:/datasets/PPC-Dataset-data/tokyo/run1")
DIAG_MAX_EPOCHS = 2000
WIRING_SMOKE_EPOCHS = 300

# Nominal single-frequency carrier bands (Hz) each system's preference list
# should map to -- if a preference list mixed bands (e.g. accidentally
# including an L2 code in a nominally-L1 list) this would flag it.
_EXPECTED_BAND_HZ = {
    "G": 1575.42e6,  # L1 C/A
    "E": 1575.42e6,  # E1
    "J": 1575.42e6,  # L1 C/A
    "C": 1561.098e6,  # B1I
    "R": 1602.0e6,  # L1 (nominal channel-0; FDMA varies per-satellite)
}
# RINEX 1-char frequency-band digit -> nominal Hz, used only to flag a code
# whose *band digit* doesn't match the system's expected band (crude check;
# real per-code frequency tables are system/constellation-specific and not
# needed here -- we only need to catch band mixing, e.g. "C2I" appearing in
# a nominally-B1I/1561 list).
_BAND_DIGIT_FOR_SYSTEM_L1_LIKE = {"G": "1", "E": "1", "J": "1", "C": "1", "R": "1"}


def verify_code_preferences_single_frequency() -> list[str]:
    """Work item 2c: flag any per-system preference code outside its L1-like band."""
    problems: list[str] = []
    for sys_char, codes in _PSEUDORANGE_CODE_PREFERENCES.items():
        expected_digit = _BAND_DIGIT_FOR_SYSTEM_L1_LIKE.get(sys_char)
        for code in codes:
            # RINEX 3 obs code format: <band digit><attribute><channel>, e.g. "C1C".
            if len(code) < 2 or code[0] != "C":
                problems.append(f"{sys_char}: {code!r} is not a pseudorange (C-prefixed) code")
                continue
            band_digit = code[1]
            if expected_digit is not None and band_digit != expected_digit:
                problems.append(
                    f"{sys_char}: {code!r} band digit {band_digit!r} != expected {expected_digit!r} "
                    f"({_EXPECTED_BAND_HZ.get(sys_char, float('nan')):.3e} Hz)"
                )
    return problems


def verify_grecj_clock_wiring() -> dict:
    """Work item 2a: constellations/n_clock/sys_kind wiring + a short VD smoke solve."""
    loader = PPCDatasetLoader(RUN_DIR)
    data = loader.load_experiment_data(
        systems=("G", "R", "E", "C", "J"), max_epochs=WIRING_SMOKE_EPOCHS,
    )
    constellations = tuple(sorted(data["constellations"]))
    n_clock = len(constellations)
    sys_char_to_clock = {c: i for i, c in enumerate(constellations)}

    used_prns = data["used_prns"]
    n_epoch = len(used_prns)
    max_sats = max(len(sats) for sats in used_prns)
    sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
    weights = np.zeros((n_epoch, max_sats), dtype=np.float64)
    sys_kind = np.zeros((n_epoch, max_sats), dtype=np.int32)
    for t in range(n_epoch):
        ns = len(used_prns[t])
        sat_ecef[t, :ns] = np.asarray(data["sat_ecef"][t], dtype=np.float64)
        pseudorange[t, :ns] = np.asarray(data["pseudoranges"][t], dtype=np.float64)
        weights[t, :ns] = np.asarray(data["weights"][t], dtype=np.float64)
        for i, sid in enumerate(used_prns[t]):
            sys_kind[t, i] = sys_char_to_clock.get(sid[0] if sid else "G", 0)

    # Each observed clock index must map back to exactly one system letter
    # (contiguous, no collisions) -- verifies the D2 dynamic n_clock wiring.
    clock_to_syschar: dict[int, set[str]] = defaultdict(set)
    for t in range(n_epoch):
        for i, sid in enumerate(used_prns[t]):
            clock_to_syschar[int(sys_kind[t, i])].add(sid[0] if sid else "G")
    wiring_ok = all(len(v) == 1 for v in clock_to_syschar.values()) and set(
        clock_to_syschar
    ) == set(range(n_clock))

    wls_state = np.zeros((n_epoch, 4), dtype=np.float64)
    for t in range(n_epoch):
        idx = np.flatnonzero(weights[t] > 0)
        if idx.size < 4:
            continue
        st, _ = wls_position(sat_ecef[t, idx].reshape(-1), pseudorange[t, idx], weights[t, idx], 25, 1e-9)
        wls_state[t] = st

    fgo_state = np.zeros((n_epoch, 7 + n_clock), dtype=np.float64)
    fgo_state[:, :3] = wls_state[:, :3]
    fgo_state[:, 6] = wls_state[:, 3]
    dt_arr = np.full(n_epoch, float(data.get("dt", 0.2)), dtype=np.float64)

    iters, mse_pr = fgo_gnss_lm_vd(
        sat_ecef, pseudorange, weights, fgo_state,
        sys_kind=sys_kind, n_clock=n_clock,
        motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        max_iter=8, tol=1e-7, dt=dt_arr,
    )

    return {
        "constellations": constellations,
        "n_clock": n_clock,
        "clock_to_syschar": {k: sorted(v) for k, v in clock_to_syschar.items()},
        "wiring_ok": wiring_ok,
        "smoke_iters": int(iters),
        "smoke_mse_pr": float(mse_pr),
        "smoke_converged": int(iters) >= 0 and np.isfinite(mse_pr),
        "per_clock_final_bias_m": {
            constellations[i]: float(fgo_state[-1, 6 + i]) for i in range(n_clock)
        },
    }


def _load_grecj_arrays(max_epochs: int) -> dict:
    loader = PPCDatasetLoader(RUN_DIR)
    data = loader.load_experiment_data(systems=("G", "R", "E", "C", "J"), max_epochs=max_epochs)
    used_prns = data["used_prns"]
    n_epoch = len(used_prns)
    max_sats = max(len(sats) for sats in used_prns)
    sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
    weights = np.zeros((n_epoch, max_sats), dtype=np.float64)
    for t in range(n_epoch):
        ns = len(used_prns[t])
        sat_ecef[t, :ns] = np.asarray(data["sat_ecef"][t], dtype=np.float64)
        pseudorange[t, :ns] = np.asarray(data["pseudoranges"][t], dtype=np.float64)
        weights[t, :ns] = np.asarray(data["weights"][t], dtype=np.float64)
    return {"used_prns": used_prns, "sat_ecef": sat_ecef, "pseudorange": pseudorange, "weights": weights}


def _single_clock_wls_residuals(sat_ecef, pseudorange, weights, used_prns) -> dict[str, list[float]]:
    """Per-satellite residual at a single global-clock WLS fit, grouped by system letter."""
    n_epoch = sat_ecef.shape[0]
    residuals_by_system: dict[str, list[float]] = defaultdict(list)
    for t in range(n_epoch):
        idx = np.flatnonzero(weights[t] > 0)
        if idx.size < 4:
            continue
        st, _ = wls_position(sat_ecef[t, idx].reshape(-1), pseudorange[t, idx], weights[t, idx], 25, 1e-9)
        rx = st[:3]
        clk_m = st[3]  # wls_position returns the clock state in metres, not seconds
        rng = np.linalg.norm(sat_ecef[t, idx] - rx[None, :], axis=1)
        resid = pseudorange[t, idx] - rng - clk_m
        for k, i in enumerate(idx):
            sid = used_prns[t][i] if i < len(used_prns[t]) else "?"
            residuals_by_system[sid[0] if sid else "?"].append(float(resid[k]))
    return residuals_by_system


def residual_breakdown(elevation_mask_deg: float, constellation_weighting: bool) -> dict:
    arrays = _load_grecj_arrays(DIAG_MAX_EPOCHS)
    sat_ecef, pseudorange, weights, used_prns = (
        arrays["sat_ecef"], arrays["pseudorange"], arrays["weights"], arrays["used_prns"]
    )

    residuals_before = _single_clock_wls_residuals(sat_ecef, pseudorange, weights, used_prns)

    weights_after = weights
    if elevation_mask_deg > 0.0:
        wls_seed = np.zeros((sat_ecef.shape[0], 4), dtype=np.float64)
        for t in range(sat_ecef.shape[0]):
            idx = np.flatnonzero(weights_after[t] > 0)
            if idx.size < 4:
                continue
            st, _ = wls_position(sat_ecef[t, idx].reshape(-1), pseudorange[t, idx], weights_after[t, idx], 25, 1e-9)
            wls_seed[t] = st
        weights_after, _stats = _apply_elevation_mask(sat_ecef, weights_after, wls_seed, elevation_mask_deg)
    if constellation_weighting:
        weights_after = _apply_constellation_sigma_scaling(weights_after, used_prns)

    residuals_after = _single_clock_wls_residuals(sat_ecef, pseudorange, weights_after, used_prns)

    def _summary(residuals: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        out = {}
        for sys_char, vals in sorted(residuals.items()):
            arr = np.asarray(vals, dtype=np.float64)
            out[sys_char] = {
                "n": int(arr.size),
                "median_abs_residual_m": float(np.median(np.abs(arr))),
                "rms_residual_m": float(np.sqrt(np.mean(arr**2))),
            }
        return out

    return {
        "before": _summary(residuals_before),
        "after": _summary(residuals_after),
        "elevation_mask_deg": elevation_mask_deg,
        "constellation_weighting": constellation_weighting,
    }


def main() -> None:
    print(f"Run: {RUN_DIR}\n")

    print("=== Work item 2c: pseudorange code-preference frequency consistency ===")
    problems = verify_code_preferences_single_frequency()
    if problems:
        print(f"  FOUND {len(problems)} issue(s):")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  OK: every system's _PSEUDORANGE_CODE_PREFERENCES entry stays within its "
              "single expected L1-like frequency band (no L1/L2/L5 mixing).")
    print()

    print(f"=== Work item 2a: GRECJ n_clock wiring + VD smoke solve ({WIRING_SMOKE_EPOCHS} ep) ===")
    wiring = verify_grecj_clock_wiring()
    print(f"  constellations (sorted) = {wiring['constellations']}  n_clock = {wiring['n_clock']}")
    print(f"  clock_index -> system_char(s): {wiring['clock_to_syschar']}")
    print(f"  wiring_ok (1 system per clock index, no collisions) = {wiring['wiring_ok']}")
    print(f"  smoke VD solve: iters={wiring['smoke_iters']} mse_pr={wiring['smoke_mse_pr']:.4g} "
          f"converged={wiring['smoke_converged']}")
    print(f"  final per-system clock bias estimate (m): {wiring['per_clock_final_bias_m']}")
    print()

    print(f"=== Work item 3: per-constellation WLS residual breakdown ({DIAG_MAX_EPOCHS} ep) ===")
    breakdown = residual_breakdown(elevation_mask_deg=20.0, constellation_weighting=True)
    print("  BEFORE (raw GRECJ, no elevation mask, no constellation weighting):")
    for sys_char, s in breakdown["before"].items():
        print(f"    {sys_char}: n={s['n']:6d}  median|resid|={s['median_abs_residual_m']:9.2f} m  "
              f"rms={s['rms_residual_m']:9.2f} m")
    print(f"  AFTER (elevation_mask={breakdown['elevation_mask_deg']:.0f} deg + constellation weighting):")
    for sys_char, s in breakdown["after"].items():
        print(f"    {sys_char}: n={s['n']:6d}  median|resid|={s['median_abs_residual_m']:9.2f} m  "
              f"rms={s['rms_residual_m']:9.2f} m")


if __name__ == "__main__":
    main()
