"""Extract 5 NLOS-classifier features per (epoch, sat) for PPC runs.

Features (per Mou et al. 2024 stacking ensemble):
    elev_deg     - satellite elevation in degrees from truth ECEF
    snr_dbhz     - C/N0 from rover.obs (S1C etc.)
    d_snr_dbhz   - SNR(t) - SNR(t-1) for same sat
    pr_res_m     - pseudorange residual after geometric + sat clock + rx clock removal
    pr_res_std_m - rolling 5-epoch std-dev of pr_res_m for same sat

Pseudo-label (for supervised training):
    is_nlos = 1 if |pr_res_m| > 8m
              0 if |pr_res_m| < 3m
              -1 (drop) otherwise

Usage:
    python experiments/extract_nlos_features.py --run tokyo/run1 --out experiments/results/nlos_features/tokyo_run1.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader


C_LIGHT = 299792458.0


def _elevation_rad(rx_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    dx = sat_ecef - rx_ecef
    range_m = np.linalg.norm(dx)
    if range_m < 1e-3:
        return 0.0
    rx_norm = np.linalg.norm(rx_ecef)
    if rx_norm < 1e-3:
        return 0.0
    up = rx_ecef / rx_norm
    sin_el = float(np.dot(dx, up) / range_m)
    sin_el = max(-1.0, min(1.0, sin_el))
    return float(np.arcsin(sin_el))


def _fit_clock_atm(pr_corr: np.ndarray, geom_range: np.ndarray, elev_deg: np.ndarray,
                    snr_dbhz: np.ndarray, sys_chars: list[str]) -> tuple[np.ndarray, dict[str, float], float]:
    """Per-epoch LS fit: estimate per-system clock + 1 zenith delay.

    Model: pr_corr_i = geom_i + clk_sys(i) + zd / sin(el_i) + multipath/NLOS_i + noise

    Iterates with outlier rejection (residuals > 5*MAD discarded).
    Returns (residuals_m_for_each_sat, dict_of_system_clocks, zenith_delay_m).
    """
    n = pr_corr.shape[0]
    if n < 4:
        return np.zeros(n), {}, 0.0

    sin_el = np.sin(np.radians(np.maximum(elev_deg, 5.0)))
    csv_unique_sys = sorted(set(sys_chars))
    n_sys = len(csv_unique_sys)
    sys_idx = {s: i for i, s in enumerate(csv_unique_sys)}
    # Design: [clk_G, clk_E, clk_R, clk_C, clk_J, ..., zd]
    A_full = np.zeros((n, n_sys + 1))
    for i, sc in enumerate(sys_chars):
        A_full[i, sys_idx[sc]] = 1.0
    A_full[:, -1] = 1.0 / sin_el
    y_full = pr_corr - geom_range
    # SNR-based weighting
    w = np.clip(snr_dbhz, 20.0, 50.0) / 50.0  # 0.4..1.0
    sqrtw = np.sqrt(w)

    keep = np.ones(n, dtype=bool)
    for _ in range(3):
        A = A_full[keep]
        y = y_full[keep]
        sw = sqrtw[keep]
        if A.shape[0] < n_sys + 2:
            break
        Aw = A * sw[:, None]
        yw = y * sw
        try:
            x, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        except np.linalg.LinAlgError:
            break
        res_all = y_full - A_full @ x
        mad = np.median(np.abs(res_all[keep] - np.median(res_all[keep])))
        if mad < 0.5:
            mad = 0.5
        new_keep = np.abs(res_all - np.median(res_all[keep])) < 5.0 * 1.4826 * mad
        if np.array_equal(new_keep, keep) or new_keep.sum() < n_sys + 2:
            break
        keep = new_keep
    sys_clk = {csv_unique_sys[i]: float(x[i]) for i in range(n_sys)}
    zd = float(x[-1])
    return res_all, sys_clk, zd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data"))
    parser.add_argument("--run", required=True, help="e.g. tokyo/run1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=0, help="0 = all epochs")
    parser.add_argument("--rolling-window", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.data_root / args.run
    loader = PPCDatasetLoader(run_dir)
    data = loader.load_experiment_data(systems=("G", "E", "R", "J"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_epochs = len(data["times"])
    if args.max_epochs > 0:
        n_epochs = min(n_epochs, args.max_epochs)
    print(f"[extract] {args.run}: {n_epochs} epochs, writing {args.out}")

    last_snr: dict[str, float] = {}  # sat_id -> previous SNR
    pr_res_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=args.rolling_window))

    n_rows = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch_idx", "tow", "sat_id", "elev_deg", "snr_dbhz",
            "d_snr_dbhz", "pr_res_m", "pr_res_std_m", "is_nlos"
        ])
        for ei in range(n_epochs):
            tow = float(data["times"][ei])
            sat_ids = data["used_prns"][ei]
            sat_ecef = np.asarray(data["sat_ecef"][ei], dtype=np.float64)
            pr_corr = np.asarray(data["pseudoranges"][ei], dtype=np.float64)
            snr = np.asarray(data["weights"][ei], dtype=np.float64)
            truth = np.asarray(data["ground_truth"][ei], dtype=np.float64)
            n = len(sat_ids)
            if n < 4 or truth is None or not np.all(np.isfinite(truth)):
                # advance SNR history for dropped epoch — keep simple
                continue
            # Geometric range from truth to each sat
            geom = np.linalg.norm(sat_ecef - truth[None, :], axis=1)
            elev_deg = np.degrees([_elevation_rad(truth, sat_ecef[i]) for i in range(n)])
            sys_chars = [s[0] for s in sat_ids]
            pr_res, _sys_clk, _zd = _fit_clock_atm(pr_corr, geom, elev_deg, snr, sys_chars)
            for i, sat in enumerate(sat_ids):
                el = float(elev_deg[i])
                s = float(snr[i])
                ds = s - last_snr[sat] if sat in last_snr else 0.0
                last_snr[sat] = s
                r = float(pr_res[i])
                pr_res_hist[sat].append(r)
                if len(pr_res_hist[sat]) >= 2:
                    rs = float(np.std(pr_res_hist[sat]))
                else:
                    rs = 0.0
                abs_r = abs(r)
                if abs_r < 3.0:
                    label = 0
                elif abs_r > 8.0:
                    label = 1
                else:
                    label = -1
                w.writerow([ei, f"{tow:.3f}", sat, f"{el:.3f}", f"{s:.2f}",
                            f"{ds:.2f}", f"{r:.3f}", f"{rs:.3f}", label])
                n_rows += 1
    print(f"[extract] wrote {n_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
