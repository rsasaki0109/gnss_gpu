"""Shared loaders for the public ``gtsam_gnss/examples/data`` RINEX clip (GPS C1C)."""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Parent of ``experiments/`` is the gnss_gpu Python tree root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.ephemeris import Ephemeris  # noqa: E402
from gnss_gpu.io.nav_rinex import (  # noqa: E402
    _datetime_to_gps_seconds_of_week,
    _datetime_to_gps_week,
    read_gps_klobuchar_from_nav_header,
    read_nav_rinex,
)
from gnss_gpu.io.rinex import read_rinex_obs  # noqa: E402
from gnss_gpu.spp import correct_pseudoranges  # noqa: E402

C_LIGHT = 299792458.0


def load_reference_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (tow_s, ecef Nxn) from reference.csv."""
    tow_list: list[float] = []
    ecef: list[list[float]] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tow_list.append(float(row["GPS TOW (s)"]))
            ecef.append(
                [
                    float(row["ECEF X (m)"]),
                    float(row["ECEF Y (m)"]),
                    float(row["ECEF Z (m)"]),
                ]
            )
    return np.asarray(tow_list, dtype=np.float64), np.asarray(ecef, dtype=np.float64)


def nearest_ref_error(
    tow: float, ref_tow: np.ndarray, ref_ecef: np.ndarray, est: np.ndarray
) -> float:
    i = int(np.argmin(np.abs(ref_tow - tow)))
    d = est[:3] - ref_ecef[i]
    return float(np.linalg.norm(d[:2]))


def _try_pybind_export(
    obs_p: Path, nav_p: Path, el_mask_deg: float
) -> dict[tuple[int, float, str], dict[str, float]] | None:
    """Try to use pybind11 RTKLIB SPP export.  Returns None if not available."""
    try:
        from gnss_gpu._gnss_gpu_rtklib_spp import export_spp_meas
    except ImportError:
        return None
    raw = export_spp_meas(str(obs_p), str(nav_p), el_mask_deg)
    result: dict[tuple[int, float, str], dict[str, float]] = {}
    n = len(raw["gps_tow"])
    for i in range(n):
        wk = int(raw["gps_week"][i])
        tow = round(float(raw["gps_tow"][i]), 4)
        sid = raw["sat_id"][i].strip()
        result[(wk, tow, sid)] = {
            "prange_m": float(raw["prange_m"][i]),
            "iono_m": float(raw["iono_m"][i]),
            "trop_m": float(raw["trop_m"][i]),
            "sat_clk_m": float(raw["sat_clk_m"][i]),
            "satx": float(raw["satx"][i]),
            "saty": float(raw["saty"][i]),
            "satz": float(raw["satz"][i]),
            "el_rad": float(raw["el_rad"][i]),
            "var_total": float(raw["var_total"][i]),
        }
    return result


def _run_rtklib_export_spp_meas(
    exe: Path, obs_p: Path, nav_p: Path, *, el_mask_deg: float
) -> Path:
    fd, out_path = tempfile.mkstemp(suffix="_rtklib_spp.csv", text=True)
    os.close(fd)
    out_p = Path(out_path)
    cmd = [str(exe), str(obs_p), str(nav_p), "-m", str(el_mask_deg)]
    with open(out_p, "w", encoding="utf-8") as fp:
        subprocess.run(cmd, check=True, stdout=fp, stderr=subprocess.PIPE, text=True)
    return out_p


def _load_rtklib_spp_export(path: Path) -> dict[tuple[int, float, str], dict[str, float]]:
    """Parse ``export_spp_meas`` CSV: key (gps_week, tow rounded, sat_id)."""
    out: dict[tuple[int, float, str], dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            wk = int(row["gps_week"])
            tow = round(float(row["gps_tow"]), 4)
            sid = row["sat_id"].strip()
            d: dict[str, float] = {
                "prange_m": float(row["prange_m"]),
                "iono_m": float(row["iono_m"]),
                "trop_m": float(row["trop_m"]),
                "sat_clk_m": float(row["sat_clk_m"]),
                "satx": float(row["satx"]),
                "saty": float(row["saty"]),
                "satz": float(row["satz"]),
            }
            if "el_rad" in row and "var_total" in row:
                d["el_rad"] = float(row["el_rad"])
                d["var_total"] = float(row["var_total"])
            out[(wk, tow, sid)] = d
    return out


@dataclass
class PublicGtsamBatch:
    data_dir: Path
    epochs_data: list[tuple[float, list[str], np.ndarray, int]]
    n_epoch: int
    max_sats: int
    sat_ecef: np.ndarray
    pseudorange: np.ndarray
    weights: np.ndarray
    ref_tow: np.ndarray
    ref_ecef: np.ndarray


def build_public_gtsam_arrays(
    data_dir: Path,
    max_epochs: int,
    *,
    rtklib_export_spp_exe: Path | None = None,
    el_mask_deg: float = 15.0,
    use_pybind: bool = True,
    weight_mode: str = "sin2el",
) -> PublicGtsamBatch:
    """Build preprocessed PR / sat ECEF / weights aligned with ``validate_fgo_gtsam_public_dataset``.

    If ``use_pybind`` is True (default), tries the pybind11 RTKLIB SPP binding
    first (``gnss_gpu._gnss_gpu_rtklib_spp``).  If unavailable or disabled,
    falls back to subprocess + CSV via ``rtklib_export_spp_exe``.

    If ``rtklib_export_spp_exe`` points to RTKLIB ``export_spp_meas`` (demo5), uses
    RTKLIB ``pntpos`` prange + iono + trop + broadcast clock + ``geodist`` sat ECEF
    to match demo5's SPP observation model. Otherwise uses ``correct_pseudoranges`` +
    ``Ephemeris`` (legacy gnss_gpu path).

    ``weight_mode`` controls the per-satellite weight strategy when RTKLIB
    observations are used:

    - ``"sin2el"`` (default): sin²(elevation) from RTKLIB ``el_rad``.  Gives the
      best FGO accuracy on the public clip (~0.96 m vs reference, beating RTKLIB
      SPP ~1.67 m).
    - ``"rtklib"``  : inverse of RTKLIB ``var_total`` (matches RTKLIB pntpos
      exactly but removes the FGO accuracy advantage).
    """
    obs_p = data_dir / "rover_1Hz.obs"
    nav_p = data_dir / "base.nav"
    ref_p = data_dir / "reference.csv"
    for need in (obs_p, nav_p, ref_p):
        if not need.is_file():
            raise FileNotFoundError(f"Missing dataset file: {need}")

    rinex = read_rinex_obs(obs_p)
    nav = read_nav_rinex(nav_p, systems=("G",))
    eph = Ephemeris(nav)
    ion_a, ion_b = read_gps_klobuchar_from_nav_header(nav_p)

    rtk_csv_path: Path | None = None
    rtk_meas: dict[tuple[int, float, str], dict[str, float]] | None = None

    # Try pybind11 path first (faster, no subprocess / temp-file overhead)
    if use_pybind and rtklib_export_spp_exe is not None:
        rtk_meas = _try_pybind_export(obs_p, nav_p, el_mask_deg)

    # Fall back to subprocess + CSV
    if rtk_meas is None and rtklib_export_spp_exe is not None:
        if not rtklib_export_spp_exe.is_file():
            raise FileNotFoundError(f"RTKLIB export tool not found: {rtklib_export_spp_exe}")
        rtk_csv_path = _run_rtklib_export_spp_meas(
            rtklib_export_spp_exe, obs_p, nav_p, el_mask_deg=el_mask_deg
        )
        try:
            rtk_meas = _load_rtklib_spp_export(rtk_csv_path)
        finally:
            rtk_csv_path.unlink(missing_ok=True)

    epochs_data: list[tuple[float, list[str], np.ndarray, int]] = []
    max_sats = 0
    for ep in rinex.epochs:
        pr_map: dict[str, float] = {}
        for sat, obs in ep.observations.items():
            if not sat.startswith("G"):
                continue
            if "C1C" in obs and obs["C1C"] and obs["C1C"] != 0.0:
                pr_map[sat] = obs["C1C"]
        if len(pr_map) < 4:
            continue
        tow = _datetime_to_gps_seconds_of_week(ep.time)
        wk = _datetime_to_gps_week(ep.time)
        sats = sorted(pr_map.keys())
        pr = np.array([pr_map[s] for s in sats], dtype=np.float64)
        epochs_data.append((tow, sats, pr, wk))
        max_sats = max(max_sats, len(sats))
        if len(epochs_data) >= max_epochs:
            break

    n_epoch = len(epochs_data)
    if n_epoch < 5:
        raise RuntimeError("Not enough valid GPS epochs (need >= 5).")

    sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
    raw_weights = np.zeros((n_epoch, max_sats), dtype=np.float64)

    approx0 = rinex.header.approx_position.copy()
    if np.linalg.norm(approx0) < 1e3:
        approx0 = np.array([-3810234.0, 3567867.0, 3652898.0], dtype=np.float64)

    corr_kw: dict = {"el_mask_rad": math.radians(el_mask_deg)}
    if ion_a is not None and ion_b is not None:
        corr_kw["iono_alpha"] = ion_a
        corr_kw["iono_beta"] = ion_b

    for t, (tow, sats, pr_raw, wk) in enumerate(epochs_data):
        ns = len(sats)
        rx_est = approx0.astype(np.float64, copy=True)
        sat_buf = np.zeros((ns, 3), dtype=np.float64)
        clk_buf = np.zeros(ns, dtype=np.float64)
        ok = np.zeros(ns, dtype=bool)
        pr_row = np.zeros(max_sats, dtype=np.float64)
        w_row = np.zeros(max_sats, dtype=np.float64)
        pr_tmp = np.zeros(ns, dtype=np.float64)
        w_tmp = np.zeros(ns, dtype=np.float64)

        if rtk_meas is not None:
            # RTKLIB pntpos model: prange = geom + rcv_clk + sat_clk_m + iono + trop
            # (sat_clk_m = -CLIGHT*dts as emitted by export_spp_meas).
            for _pass in range(2):
                pr_tmp[:] = 0.0
                w_tmp[:] = 0.0
                for si, sid in enumerate(sats):
                    row = rtk_meas.get((wk, round(float(tow), 4), sid))
                    if row is None:
                        continue
                    sat_buf[si, 0] = row["satx"]
                    sat_buf[si, 1] = row["saty"]
                    sat_buf[si, 2] = row["satz"]
                    pr_clean = (
                        row["prange_m"]
                        - row["iono_m"]
                        - row["trop_m"]
                        - row["sat_clk_m"]
                    )
                    pr_tmp[si] = pr_clean
                    if weight_mode == "rtklib" and "var_total" in row and row["var_total"] > 0.0:
                        w_tmp[si] = 1.0 / row["var_total"]
                    elif "el_rad" in row:
                        sin_el = max(math.sin(row["el_rad"]), 0.1)
                        w_tmp[si] = sin_el * sin_el
                    else:
                        # Fallback: use gnss_gpu elevation-based weight
                        _pr_ign, wsp = correct_pseudoranges(
                            sat_buf[si].reshape(1, 3),
                            np.array([pr_raw[si]], dtype=np.float64),
                            rx_est,
                            tow,
                            **corr_kw,
                        )
                        w_tmp[si] = float(wsp[0])
                idx = np.flatnonzero(w_tmp > 0)
                if idx.size >= 4:
                    st, _ = wls_position(
                        sat_buf[idx, :].reshape(-1),
                        pr_tmp[idx],
                        w_tmp[idx],
                        25,
                        1e-9,
                    )
                    rx_est = np.asarray(st[:3], dtype=np.float64).copy()
        else:
            # Two-pass SPP-style prep: iono/tropo mapping depends on receiver LLH; refines
            # toward RTKLIB-style broadcast corrections (demo5 uses ~similar iteration).
            for _pass in range(2):
                for si, sid in enumerate(sats):
                    prn_int = int(sid[1:])
                    tau = 0.0
                    sat_c = None
                    clk_s = 0.0
                    for _lt in range(4):
                        t_tx = (tow - tau / C_LIGHT) % 604800.0
                        sat_pos, sat_clk_arr, _ = eph.compute(t_tx, [prn_int], ["C1C"])
                        if len(sat_pos) == 0:
                            break
                        sat_c = sat_pos[0]
                        clk_s = float(sat_clk_arr[0])
                        rho = float(np.linalg.norm(rx_est - sat_c))
                        tau = rho / C_LIGHT
                    if sat_c is None:
                        ok[si] = False
                        continue
                    sat_buf[si] = sat_c
                    clk_buf[si] = clk_s
                    ok[si] = True

                pr_tmp[:] = 0.0
                w_tmp[:] = 0.0
                for si, sid in enumerate(sats):
                    if not ok[si]:
                        continue
                    pr_cor, wsp = correct_pseudoranges(
                        sat_buf[si].reshape(1, 3),
                        np.array([pr_raw[si]], dtype=np.float64),
                        rx_est,
                        tow,
                        **corr_kw,
                    )
                    pr_tmp[si] = pr_cor[0] + C_LIGHT * clk_buf[si]
                    w_tmp[si] = float(wsp[0])

                idx = np.flatnonzero(w_tmp > 0)
                if idx.size >= 4:
                    st, _ = wls_position(
                        sat_buf[idx, :].reshape(-1),
                        pr_tmp[idx],
                        w_tmp[idx],
                        25,
                        1e-9,
                    )
                    rx_est = np.asarray(st[:3], dtype=np.float64).copy()

        sat_ecef[t, :ns] = sat_buf
        pr_row[:ns] = pr_tmp
        w_row[:ns] = w_tmp
        pseudorange[t, :ns] = pr_row[:ns]
        raw_weights[t, :ns] = w_row[:ns]

    ref_tow, ref_ecef = load_reference_csv(ref_p)
    return PublicGtsamBatch(
        data_dir=data_dir,
        epochs_data=epochs_data,
        n_epoch=n_epoch,
        max_sats=max_sats,
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        weights=raw_weights.copy(),
        ref_tow=ref_tow,
        ref_ecef=ref_ecef,
    )
