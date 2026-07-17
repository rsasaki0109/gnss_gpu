"""Per-satellite, per-epoch validation: does the physics predict WHICH satellite
is biased, by HOW MUCH, at EACH epoch?

``examples/demo_diffraction_benchmark.py`` answers a distribution-level
question: pool every candidate satellite-epoch's simulated bias into one
distribution and compare it (Wasserstein-1 / KS) against the pooled real
residual distribution. That is necessary but not sufficient for the claim
this project actually wants to make: that the ray-traced diffraction bias for
satellite S at epoch T can be subtracted from satellite S's own measurement
at epoch T as a *correction* -- which requires the model to track individual
satellite-epochs, not just get the overall histogram right.

This demo reuses the exact same building blocks as
``demo_diffraction_benchmark.main`` (UrbanNav + PLATEAU loading, edge
extraction, ``compute_diffraction_paths`` / ``compute_utd_diffraction_paths``,
reflection pooling, the clock/tropo-purified residual) but does not edit that
file. Its result dict only exposes pooled arrays (many phase-swept bias
*samples* per candidate, not one scalar prediction per satellite-epoch), which
cannot be paired 1:1 with the measured residual. So the per-(satellite,
epoch) candidate loop is reimplemented here, collapsing each candidate's
phase-swept bias samples to a single scalar prediction (their mean, i.e. the
model's expected bias for that satellite at that epoch given an unknown
carrier phase) so it can be matched exactly against that satellite's real
residual at that epoch.

Run for both UrbanNav sites (Odaiba, Shinjuku) x both diffraction amplitude
models (knife_edge, utd); prints a site x model correlation / sign-agreement
/ correction-gain summary table and a per-satellite breakdown, and writes one
scatter figure per site to ``results/validation/per_satellite_<site>.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Same rationale as demo_diffraction_benchmark.py: use the repo's own
# gnss_gpu (compiled extensions such as _raytrace live under python/), only
# add experiments/ so the UTD edge extractor helper can be imported.
for _p in (str(_REPO_ROOT / "experiments"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.io.plateau import load_plateau
from gnss_gpu.diffraction import compute_diffraction_paths
from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.validation.real_residuals import elevation_azimuth, tropo_delays
from gnss_gpu.validation.diffraction_benchmark import (
    candidates_from_paths,
    predict_bias_samples_m,
)
from gnss_gpu.validation.nlos_bias import pooled_nlos_bias_m, reflection_replicas
from gnss_gpu.validation.per_satellite import evaluate_predictions
from utd_edge_features import extract_diffraction_edges

MODELS = ("knife_edge", "utd")
SITES = ("Odaiba", "Shinjuku")
NLOS_THRESHOLD_M = 1.0


def _los_predicted_bias_m(paths, n_phase: int) -> float:
    """Scalar LOS multipath prediction: mean bias over an unknown carrier phase."""
    if not paths:
        return float("nan")
    samples = predict_bias_samples_m(candidates_from_paths(paths), n_phase=n_phase)
    return float(np.mean(samples)) if samples.size else float("nan")


def collect_aligned_arrays(
    site,
    data_root=None,
    plateau_dir=None,
    max_epochs=60,
    epoch_stride=1,
    elevation_mask_deg=10.0,
    n_phase=16,
    max_paths=2,
    geoid="egm96",
    pool_reflections=True,
    refl_cull_radius_m=150.0,
    refl_material="concrete",
    max_refl_paths=3,
    refl_point_tol_m=3.0,
    correct_transmission_time=True,
):
    """Build aligned per-(satellite, epoch) predicted-vs-measured arrays.

    Mirrors ``demo_diffraction_benchmark.main``'s candidate-satellite loop
    (same geometry, same defaults, same reflection-pooling logic) but instead
    of pooling phase-swept bias *samples* across all candidates into one
    distribution, it records a single scalar prediction per model per
    satellite-epoch next to that satellite-epoch's own measured residual.

    Returns a dict with parallel 1-D arrays ``sat_id``, ``epoch_idx``,
    ``is_los``, ``is_nlos``, ``measured`` (signed, tropo-purified,
    clock-removed pseudorange residual, metres) and ``predicted`` (a dict
    mapping model name -> array of the same length; NaN where that model had
    no trackable candidate for that satellite-epoch).
    """
    data_root = Path(data_root) if data_root else _REPO_ROOT / "experiments" / "data" / "urbannav"
    run_dir = data_root / site
    plateau_dir = Path(plateau_dir) if plateau_dir else (
        _REPO_ROOT / "experiments" / "data" / f"plateau_{site.lower()}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"UrbanNav run directory not found: {run_dir}")
    if not plateau_dir.is_dir():
        raise FileNotFoundError(f"PLATEAU mesh directory not found: {plateau_dir}")

    print(f"[{site}] loading UrbanNav ...", flush=True)
    loader = UrbanNavLoader(run_dir)
    data = loader.load_experiment_data(
        max_epochs=max_epochs, correct_transmission_time=correct_transmission_time)
    ground_truth = np.asarray(data["ground_truth"], dtype=float)
    print(f"[{site}]   epochs={len(data['times'])}", flush=True)

    print(f"[{site}] loading PLATEAU mesh ...", flush=True)
    try:
        model = load_plateau(plateau_dir, zone=9, kinds=("bldg",), geoid_correction=geoid)
    except Exception as exc:  # egm96 needs pyproj; fall back to a constant offset.
        print(f"[{site}]   egm96 correction failed ({exc}); "
              f"using constant Tokyo geoid offset +36.7 m", flush=True)
        model = load_plateau(plateau_dir, zone=9, kinds=("bldg",), geoid_correction=36.7)
    print(f"[{site}]   triangles={len(model.triangles)}", flush=True)

    print(f"[{site}] extracting diffraction edges (UTD wedge geometry) ...", flush=True)
    edges = extract_diffraction_edges(
        model.triangles, route_ecef=ground_truth, route_margin_m=250.0,
        min_edge_length_m=3.0, min_dihedral_deg=20.0,
        include_boundary_edges=True, voxel_size_m=2.0, max_edges=4000)
    print(f"[{site}]   edges={edges.size}", flush=True)

    mask = np.radians(elevation_mask_deg)
    kw = dict(max_paths=max_paths, max_edge_range_m=250.0,
              max_ray_edge_distance_m=20.0, max_excess_path_m=120.0)

    tri_all = np.asarray(model.triangles, dtype=float)
    tri_centroid = tri_all.mean(axis=1) if pool_reflections else None

    sat_ids, epoch_idx, is_los_flags, measured = [], [], [], []
    predicted = {name: [] for name in MODELS}

    n_epochs = len(data["times"])
    for i in range(0, n_epochs, epoch_stride):
        sat = np.asarray(data["sat_ecef"][i], dtype=float)
        pr = np.asarray(data["pseudoranges"][i], dtype=float)
        if sat.shape[0] == 0:
            continue
        prns = data["used_prns"][i]
        rx = ground_truth[i]
        geom = np.linalg.norm(sat - rx, axis=1)
        el, _ = elevation_azimuth(rx, sat)
        atmo = tropo_delays(rx, sat)
        pre = pr - geom - atmo

        try:
            is_los = np.asarray(model.check_los(rx, sat), dtype=bool)
        except Exception:
            is_los = np.ones(sat.shape[0], dtype=bool)

        ref = is_los & (el >= mask)
        clock = float(np.median(pre[ref])) if int(ref.sum()) >= 3 else float(np.median(pre))
        resid = pre - clock  # tropo-purified, receiver-clock-removed residual

        ke = compute_diffraction_paths(rx, sat, edges, **kw)
        ut = compute_utd_diffraction_paths(rx, sat, edges, mode="absorbing", **kw)

        refl_by_sat = {}
        if pool_reflections:
            nlos_idx = np.where(~is_los & (el >= mask))[0]
            if nlos_idx.size:
                near = tri_all[np.linalg.norm(tri_centroid - rx, axis=1)
                               < refl_cull_radius_m]
                bm = BuildingModel(near if near.shape[0] else tri_all[:1])
                rpaths = bm.compute_reflection_paths(
                    rx, sat[nlos_idx], max_paths=max_refl_paths,
                    reflection_point_tol_m=refl_point_tol_m)
                for k, s in enumerate(nlos_idx):
                    reps = reflection_replicas(rpaths[k], material=refl_material)
                    if reps:
                        refl_by_sat[int(s)] = reps

        for s in range(sat.shape[0]):
            if el[s] < mask:
                continue
            has_ke, has_ut = bool(ke[s]), bool(ut[s])
            refl = refl_by_sat.get(s, [])
            if not (has_ke or has_ut or refl):
                continue

            if is_los[s]:
                # LOS: bounded multipath bias, scalar = phase-averaged expectation.
                pk = _los_predicted_bias_m(ke[s], n_phase)
                pu = _los_predicted_bias_m(ut[s], n_phase)
                if np.isnan(pk) and np.isnan(pu):
                    continue
            else:
                # NLOS: full excess delay of the strongest trackable pooled replica.
                pk_val = pooled_nlos_bias_m(ke[s], refl) if (has_ke or refl) else None
                pu_val = pooled_nlos_bias_m(ut[s], refl) if (has_ut or refl) else None
                if pk_val is None and pu_val is None:
                    continue
                pk = float(pk_val) if pk_val is not None else float("nan")
                pu = float(pu_val) if pu_val is not None else float("nan")

            sid = prns[s] if s < len(prns) else f"sat{s}"
            sat_ids.append(sid)
            epoch_idx.append(i)
            is_los_flags.append(bool(is_los[s]))
            measured.append(float(resid[s]))
            predicted["knife_edge"].append(pk)
            predicted["utd"].append(pu)

    is_los_arr = np.asarray(is_los_flags, dtype=bool)
    return {
        "site": site,
        "sat_id": np.asarray(sat_ids, dtype=object),
        "epoch_idx": np.asarray(epoch_idx, dtype=int),
        "is_los": is_los_arr,
        "is_nlos": ~is_los_arr,
        "measured": np.asarray(measured, dtype=float),
        "predicted": {name: np.asarray(vals, dtype=float) for name, vals in predicted.items()},
    }


def _print_summary_row(site, model, res):
    print(f"{site:<10}{model:<12}{res['n']:>6}{res['n_nlos']:>7}"
          f"{res['pearson_all']:>9.3f}{res['pearson_nlos']:>9.3f}"
          f"{res['spearman_all']:>9.3f}"
          f"{res['sign_agreement_nlos']:>9.3f}({res['sign_agreement_nlos_n']:>3})"
          f"{res['rms_raw_m']:>9.2f}{res['rms_corrected_m']:>9.2f}"
          f"{res['correction_gain_pct']:>8.1f}%")


def _print_per_satellite(site, model, rows, top_n=8):
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: -r["n"])[:top_n]
    print(f"\n  {site} / {model} per-satellite (top {len(rows_sorted)} by sample count):")
    print(f"    {'sat':<6}{'n':>5}{'n_nlos':>7}{'pearson':>9}{'sign_agr':>10}{'gain_%':>9}")
    for r in rows_sorted:
        sign_str = "n/a" if np.isnan(r["sign_agreement"]) else f"{r['sign_agreement']:.2f}"
        print(f"    {str(r['sat_id']):<6}{r['n']:>5}{r['n_nlos']:>7}"
              f"{r['pearson']:>9.3f}{sign_str:>10}{r['gain_pct']:>8.1f}%")


def _plot_site(site, arrays, results_by_model, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 5.5), sharex=True, sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    is_nlos = arrays["is_nlos"]
    measured = arrays["measured"]

    for ax, model in zip(axes, MODELS):
        pred = arrays["predicted"][model]
        mask = np.isfinite(pred) & np.isfinite(measured)
        los_mask = mask & ~is_nlos
        nlos_mask = mask & is_nlos
        ax.scatter(pred[los_mask], measured[los_mask], s=14, alpha=0.5,
                   color="steelblue", label="LOS")
        ax.scatter(pred[nlos_mask], measured[nlos_mask], s=22, alpha=0.7,
                   color="crimson", marker="^", label="NLOS")
        if mask.any():
            lo = float(min(pred[mask].min(), measured[mask].min()))
            hi = float(max(pred[mask].max(), measured[mask].max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x")
        res = results_by_model[model]
        ax.set_title(
            f"{model}\n"
            f"r(all)={res['pearson_all']:.2f}  r(NLOS)={res['pearson_nlos']:.2f}\n"
            f"sign-agr(NLOS)={res['sign_agreement_nlos']:.2f}  "
            f"gain={res['correction_gain_pct']:.1f}%")
        ax.set_xlabel("predicted bias [m]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("measured residual [m]")
    fig.suptitle(f"Per-satellite, per-epoch: predicted vs measured bias ({site})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_satellite_{site}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main(sites=SITES, max_epochs=60, epoch_stride=1, nlos_threshold_m=NLOS_THRESHOLD_M,
         out_dir=None):
    out_dir = Path(out_dir) if out_dir else _REPO_ROOT / "results" / "validation"
    all_results = {}

    print(f"{'site':<10}{'model':<12}{'n':>6}{'n_nlos':>7}{'r_all':>9}{'r_nlos':>9}"
          f"{'rho_all':>9}{'sign_agr(n)':>13}{'rms0':>9}{'rmsC':>9}{'gain':>9}")
    print("-" * 106)

    for site in sites:
        try:
            arrays = collect_aligned_arrays(
                site, max_epochs=max_epochs, epoch_stride=epoch_stride)
        except FileNotFoundError as exc:
            print(f"[{site}] SKIPPED: {exc}")
            continue

        n_total = arrays["measured"].size
        n_nlos = int(arrays["is_nlos"].sum())
        print(f"\n[{site}] candidate satellite-epochs: {n_total} "
              f"(NLOS={n_nlos}, LOS={n_total - n_nlos})", flush=True)
        if n_total == 0:
            print(f"[{site}] no candidate satellite-epochs; skipping.")
            continue

        results_by_model = {}
        for model in MODELS:
            res = evaluate_predictions(
                arrays["predicted"][model], arrays["measured"],
                sat_ids=arrays["sat_id"], is_nlos=arrays["is_nlos"],
                nlos_threshold_m=nlos_threshold_m)
            results_by_model[model] = res
            _print_summary_row(site, model, res)

        for model in MODELS:
            _print_per_satellite(site, model, results_by_model[model].get("per_satellite", []))

        try:
            out_path = _plot_site(site, arrays, results_by_model, out_dir)
            print(f"\n[{site}] figure: {out_path}")
        except Exception as exc:
            print(f"[{site}] (plot skipped: {exc})")

        all_results[site] = {"arrays": arrays, "models": results_by_model}

    return all_results


if __name__ == "__main__":
    site_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sites_arg = (site_arg,) if site_arg else SITES
    max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    main(sites=sites_arg, max_epochs=max_ep)
