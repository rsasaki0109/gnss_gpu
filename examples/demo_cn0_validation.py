"""Validate ray-traced LOS/NLOS + diffraction physics against measured C/N0.

Pseudorange residuals (see ``demo_diffraction_benchmark.py``) are one way to
check the ray-traced geometry against real UrbanNav data, but they mix in
clock, atmosphere and ephemeris error. The RINEX rover files also carry
``S1C`` -- the receiver-measured carrier-to-noise density (C/N0, dB-Hz) -- an
independent channel that only depends on received signal power. This demo
checks whether the ray tracer's predicted LOS/NLOS state and diffraction
attenuation explain what the receiver actually measured:

  1. Do predicted-NLOS satellites read systematically weaker C/N0 than
     predicted-LOS ones (mean/median gap, AUC of C/N0 as an LOS classifier)?
  2. Among NLOS satellites, does the ray-traced diffraction attenuation
     correlate with the measured C/N0 deficit relative to an elevation-matched
     LOS baseline (Pearson / Spearman)?

Reuses the same site wiring as ``demo_diffraction_benchmark.py``: UrbanNav
rover observations + PLATEAU building mesh, satellites placed at signal
transmission time, ``BuildingModel.check_los`` for the LOS/NLOS label, and
``compute_utd_diffraction_paths`` (Kouyoumjian-Pathak UTD) over extracted
wedge edges for the NLOS diffraction amplitude.

Requires local UrbanNav run directories and matching PLATEAU meshes
(defaults: experiments/data/urbannav/{Odaiba,Shinjuku} +
experiments/data/plateau_{odaiba,shinjuku}).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Use the installed gnss_gpu (carries the compiled ray-tracing extensions);
# only add experiments/ for the edge extractor, same as demo_diffraction_benchmark.
for _p in (str(_REPO_ROOT / "experiments"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.io.plateau import load_plateau
from gnss_gpu.io.nav_rinex import _datetime_to_gps_seconds_of_week
from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths
from gnss_gpu.validation.real_residuals import elevation_azimuth
from gnss_gpu.validation.cn0_validation import (
    attenuation_deficit_correlation,
    cn0_deficit,
    cn0_los_nlos_separation,
    elevation_binned_los_baseline,
)
from utd_edge_features import extract_diffraction_edges


def _build_measured_cn0_lookup(rover_obs, snr_code: str = "S1C") -> dict[float, dict[str, float]]:
    """Map rounded GPS-time-of-week -> {sat_id: measured C/N0 [dB-Hz]}.

    Rounded to milliseconds so it can be matched against
    ``UrbanNavLoader.load_experiment_data``'s per-epoch ``times`` (computed
    with the same ``_datetime_to_gps_seconds_of_week`` from the same file).
    """
    lut: dict[float, dict[str, float]] = {}
    for ep in rover_obs.epochs:
        tow = round(_datetime_to_gps_seconds_of_week(ep.time), 3)
        sat_map: dict[str, float] = {}
        for sat_id, obs in ep.observations.items():
            val = float(obs.get(snr_code, 0.0))
            if val > 0.0:
                sat_map[sat_id] = val
        lut[tow] = sat_map
    return lut


def _load_plateau_model(plateau_dir: Path, geoid: str = "egm96"):
    try:
        return load_plateau(plateau_dir, zone=9, kinds=("bldg",), geoid_correction=geoid)
    except Exception as exc:  # egm96 needs pyproj; fall back to a constant offset.
        print(f"  egm96 correction failed ({exc}); using constant Tokyo geoid offset +36.7 m",
              flush=True)
        return load_plateau(plateau_dir, zone=9, kinds=("bldg",), geoid_correction=36.7)


def run_site(
    site: str,
    data_root: Path,
    plateau_root: Path,
    *,
    max_epochs: int = 60,
    elevation_mask_deg: float = 10.0,
    max_paths: int = 2,
    out_dir: Path | None = None,
) -> dict:
    run_dir = data_root / site
    plateau_dir = plateau_root / f"plateau_{site.lower()}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"UrbanNav run directory not found: {run_dir}")
    if not plateau_dir.is_dir():
        raise FileNotFoundError(f"PLATEAU mesh directory not found: {plateau_dir}")

    print(f"\n=== {site} ===", flush=True)
    print("loading UrbanNav rover observations ...", flush=True)
    loader = UrbanNavLoader(run_dir)
    data = loader.load_experiment_data(max_epochs=max_epochs, correct_transmission_time=True)
    ground_truth = np.asarray(data["ground_truth"], dtype=float)
    print(f"  epochs={len(data['times'])} (transmission-time corrected)", flush=True)

    rover_obs = loader.load_rover_obs()
    cn0_lookup = _build_measured_cn0_lookup(rover_obs)

    print("loading PLATEAU mesh ...", flush=True)
    model = _load_plateau_model(plateau_dir)
    print(f"  triangles={len(model.triangles)}", flush=True)

    print("extracting diffraction edges (UTD wedge geometry) ...", flush=True)
    edges = extract_diffraction_edges(
        model.triangles, route_ecef=ground_truth, route_margin_m=250.0,
        min_edge_length_m=3.0, min_dihedral_deg=20.0,
        include_boundary_edges=True, voxel_size_m=2.0, max_edges=4000)
    print(f"  edges={edges.size}", flush=True)

    mask = np.radians(elevation_mask_deg)
    kw = dict(max_paths=max_paths, max_edge_range_m=250.0,
              max_ray_edge_distance_m=20.0, max_excess_path_m=120.0)

    elevation_deg: list[float] = []
    measured_cn0: list[float] = []
    is_los_flags: list[bool] = []
    # For NLOS satellite-epochs with a predicted UTD diffraction path: index
    # into the arrays above + the strongest (min-attenuation) path's loss.
    nlos_index: list[int] = []
    nlos_attenuation_db: list[float] = []

    n_epochs_used = 0
    for i in range(len(data["times"])):
        sat = np.asarray(data["sat_ecef"][i], dtype=float)
        if sat.shape[0] == 0:
            continue
        rx = ground_truth[i]
        el, _ = elevation_azimuth(rx, sat)

        try:
            is_los = np.asarray(model.check_los(rx, sat), dtype=bool)
        except Exception:
            is_los = np.ones(sat.shape[0], dtype=bool)

        ut = compute_utd_diffraction_paths(rx, sat, edges, mode="absorbing", **kw)

        tow_key = round(float(data["times"][i]), 3)
        obs_map = cn0_lookup.get(tow_key, {})
        sat_ids = data["used_prns"][i]
        n_epochs_used += 1

        for s in range(sat.shape[0]):
            if el[s] < mask:
                continue
            cn0 = obs_map.get(sat_ids[s], np.nan)
            if not np.isfinite(cn0) or cn0 <= 0.0:
                continue

            elevation_deg.append(float(np.degrees(el[s])))
            measured_cn0.append(float(cn0))
            is_los_flags.append(bool(is_los[s]))
            idx = len(elevation_deg) - 1

            if not is_los[s] and ut[s]:
                best = min(ut[s], key=lambda p: p.attenuation_db)
                nlos_index.append(idx)
                nlos_attenuation_db.append(float(best.attenuation_db))

    elevation_deg = np.asarray(elevation_deg, dtype=float)
    measured_cn0 = np.asarray(measured_cn0, dtype=float)
    is_los_flags = np.asarray(is_los_flags, dtype=bool)
    nlos_index = np.asarray(nlos_index, dtype=np.int64)
    nlos_attenuation_db = np.asarray(nlos_attenuation_db, dtype=float)

    print(f"\nsatellite-epochs with measured C/N0: {measured_cn0.size} "
          f"(LOS={int(is_los_flags.sum())}, NLOS={int((~is_los_flags).sum())})", flush=True)

    sep = cn0_los_nlos_separation(measured_cn0, is_los_flags)
    baseline = elevation_binned_los_baseline(elevation_deg, measured_cn0, is_los_flags)

    if nlos_index.size:
        elev_nlos = elevation_deg[nlos_index]
        cn0_nlos = measured_cn0[nlos_index]
        deficit = cn0_deficit(elev_nlos, cn0_nlos, baseline)
    else:
        deficit = np.array([])
    corr = attenuation_deficit_correlation(nlos_attenuation_db, deficit)

    print(f"\n{site} C/N0 LOS-vs-NLOS separation:")
    print(f"  n_los={sep['n_los']}  n_nlos={sep['n_nlos']}")
    print(f"  mean C/N0   LOS={sep['mean_los_dbhz']:.2f} dB-Hz  "
          f"NLOS={sep['mean_nlos_dbhz']:.2f} dB-Hz  gap={sep['mean_gap_dbhz']:+.2f} dB")
    print(f"  median C/N0 LOS={sep['median_los_dbhz']:.2f} dB-Hz  "
          f"NLOS={sep['median_nlos_dbhz']:.2f} dB-Hz  gap={sep['median_gap_dbhz']:+.2f} dB")
    print(f"  AUC(C/N0 -> LOS)={sep['auc']:.3f}  (0.5=useless, 1=perfect)")

    print(f"\n{site} diffraction attenuation vs measured C/N0 deficit "
          f"(NLOS sats with a predicted UTD path, n={corr['n']}):")
    print(f"  Pearson r ={corr['pearson_r']:+.3f}")
    print(f"  Spearman r={corr['spearman_r']:+.3f}")

    plot_path = _make_plot(site, measured_cn0, is_los_flags, sep,
                            nlos_attenuation_db, deficit, corr, out_dir)
    if plot_path is not None:
        print(f"\nplot: {plot_path}")

    return {
        "site": site,
        "n_epochs": n_epochs_used,
        "n_sat_epochs": int(measured_cn0.size),
        "separation": sep,
        "baseline": baseline,
        "correlation": corr,
        "plot_path": str(plot_path) if plot_path is not None else None,
        "arrays": {
            "elevation_deg": elevation_deg,
            "measured_cn0_dbhz": measured_cn0,
            "is_los": is_los_flags,
            "nlos_attenuation_db": nlos_attenuation_db,
            "nlos_cn0_deficit_dbhz": deficit,
        },
    }


def _make_plot(site, cn0, is_los, sep, atten_db, deficit, corr, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(plot skipped: {exc})")
        return None

    out_dir = Path(out_dir) if out_dir else _REPO_ROOT / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cn0_validation_{site}.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    los_vals = cn0[is_los]
    nlos_vals = cn0[~is_los]
    bins = np.linspace(
        float(np.nanmin(cn0)) if cn0.size else 0.0,
        float(np.nanmax(cn0)) if cn0.size else 1.0,
        30,
    )
    if los_vals.size:
        ax1.hist(los_vals, bins=bins, alpha=0.6, label=f"predicted LOS (n={los_vals.size})",
                  color="#1f77b4")
    if nlos_vals.size:
        ax1.hist(nlos_vals, bins=bins, alpha=0.6, label=f"predicted NLOS (n={nlos_vals.size})",
                  color="#d62728")
    ax1.set_xlabel("measured C/N0 [dB-Hz]")
    ax1.set_ylabel("count")
    ax1.set_title(f"{site}: measured C/N0 by predicted LOS/NLOS")
    auc = sep["auc"]
    gap = sep["mean_gap_dbhz"]
    auc_txt = f"{auc:.3f}" if np.isfinite(auc) else "n/a"
    gap_txt = f"{gap:+.2f} dB" if np.isfinite(gap) else "n/a"
    ax1.text(0.02, 0.98, f"AUC={auc_txt}\nmean gap={gap_txt}",
              transform=ax1.transAxes, va="top", ha="left",
              bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    if atten_db.size and deficit.size:
        finite = np.isfinite(atten_db) & np.isfinite(deficit)
        ax2.scatter(atten_db[finite], deficit[finite], s=14, alpha=0.6, color="#2ca02c")
    ax2.set_xlabel("predicted UTD diffraction attenuation [dB]")
    ax2.set_ylabel("measured C/N0 deficit vs elevation-matched LOS baseline [dB]")
    ax2.set_title(f"{site}: predicted attenuation vs measured C/N0 deficit")
    pr = corr["pearson_r"]
    sr = corr["spearman_r"]
    pr_txt = f"{pr:+.3f}" if np.isfinite(pr) else "n/a"
    sr_txt = f"{sr:+.3f}" if np.isfinite(sr) else "n/a"
    ax2.text(0.02, 0.98, f"n={corr['n']}\nPearson r={pr_txt}\nSpearman r={sr_txt}",
              transform=ax2.transAxes, va="top", ha="left",
              bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main(
    sites=("Odaiba", "Shinjuku"),
    data_root: str | Path | None = None,
    plateau_root: str | Path | None = None,
    max_epochs: int = 60,
    elevation_mask_deg: float = 10.0,
    out_dir: str | Path | None = None,
) -> dict:
    data_root = Path(data_root) if data_root else _REPO_ROOT / "experiments" / "data" / "urbannav"
    plateau_root = Path(plateau_root) if plateau_root else _REPO_ROOT / "experiments" / "data"
    out_dir = Path(out_dir) if out_dir else _REPO_ROOT / "results" / "validation"

    results = {}
    for site in sites:
        results[site] = run_site(
            site, data_root, plateau_root,
            max_epochs=max_epochs, elevation_mask_deg=elevation_mask_deg,
            out_dir=out_dir,
        )

    print("\n" + "=" * 72)
    print(f"{'site':<10}{'n_los':>8}{'n_nlos':>8}{'gap_dB':>10}{'AUC':>8}"
          f"{'pearson':>10}{'spearman':>10}{'n_pairs':>10}")
    for site, res in results.items():
        sep = res["separation"]
        corr = res["correlation"]
        print(f"{site:<10}{sep['n_los']:>8}{sep['n_nlos']:>8}"
              f"{sep['mean_gap_dbhz']:>10.2f}{sep['auc']:>8.3f}"
              f"{corr['pearson_r']:>10.3f}{corr['spearman_r']:>10.3f}{corr['n']:>10}")
    print("=" * 72)
    return results


if __name__ == "__main__":
    site_arg = sys.argv[1] if len(sys.argv) > 1 else None
    max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    sites_arg = (site_arg,) if site_arg else ("Odaiba", "Shinjuku")
    main(sites=sites_arg, max_epochs=max_ep)
