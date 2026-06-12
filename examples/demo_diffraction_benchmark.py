"""Quantitative benchmark: knife-edge vs UTD diffraction on real UrbanNav data.

For each epoch we find the satellites whose line of sight grazes a PLATEAU
building edge (diffraction candidates). The geometry (which edge, excess delay)
is identical for both diffraction models -- only the diffracted-replica
*amplitude* differs (ITU-R P.526 knife edge vs Kouyoumjian-Pathak UTD). Each
model therefore predicts a different multipath code-bias distribution for the
same candidate set; we compare both against the real tropo-purified pseudorange
residuals of those candidate satellites using Wasserstein-1 and KS distances.

The model whose predicted distribution is closer to the measured one is the
better diffraction model for this data -- the quantitative claim behind
"UTD > knife edge" (Zhang & Hsu, NAVIGATION 2021).

Requires a local UrbanNav run directory and the matching PLATEAU mesh
(defaults: experiments/data/urbannav/Odaiba + experiments/data/plateau_odaiba).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Use the installed gnss_gpu (it carries the compiled .so extensions such as
# _raytrace for check_los); only add experiments/ for the edge extractor. The
# pure-Python modules added in this work are kept in sync in site-packages.
for _p in (str(_REPO_ROOT / "experiments"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.io.plateau import load_plateau
from gnss_gpu.diffraction import compute_diffraction_paths
from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.validation.real_residuals import (
    elevation_azimuth,
    tropo_delays,
)
from gnss_gpu.validation.diffraction_benchmark import (
    benchmark_models,
    candidates_from_paths,
    predict_bias_samples_m,
)
from gnss_gpu.validation.nlos_bias import (
    nlos_bias_m,
    pooled_nlos_bias_m,
    reflection_replicas,
)
from gnss_gpu.validation.reference_quality import (
    format_reference_quality,
    residual_reference_quality,
)
from utd_edge_features import extract_diffraction_edges


def main(site="Odaiba", data_root=None, plateau_dir=None, max_epochs=120,
         epoch_stride=2, elevation_mask_deg=10.0, n_phase=16,
         max_paths=2, geoid="egm96", pool_reflections=True,
         refl_cull_radius_m=150.0, refl_material="concrete",
         max_refl_paths=3, refl_point_tol_m=3.0,
         correct_transmission_time=True):
    data_root = Path(data_root) if data_root else _REPO_ROOT / "experiments" / "data" / "urbannav"
    run_dir = data_root / site
    plateau_dir = Path(plateau_dir) if plateau_dir else (
        _REPO_ROOT / "experiments" / "data" / f"plateau_{site.lower()}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"UrbanNav run directory not found: {run_dir}")
    if not plateau_dir.is_dir():
        raise FileNotFoundError(f"PLATEAU mesh directory not found: {plateau_dir}")

    print(f"loading UrbanNav {site} ...", flush=True)
    loader = UrbanNavLoader(run_dir)
    # The loader corrects satellite positions to signal-transmission time
    # (Sagnac-rotated) by default; this removes a per-satellite range error of
    # tens of metres that otherwise dominates the residual and makes it useless
    # as an NLOS reference. Without it the benchmark would compare the model
    # against contamination, not NLOS.
    data = loader.load_experiment_data(
        max_epochs=max_epochs,
        correct_transmission_time=correct_transmission_time,
    )
    ground_truth = np.asarray(data["ground_truth"], dtype=float)
    print(f"  epochs={len(data['times'])}"
          f"{'  (transmission-time corrected)' if correct_transmission_time else ''}",
          flush=True)

    print("loading PLATEAU mesh ...", flush=True)
    try:
        model = load_plateau(plateau_dir, zone=9, kinds=("bldg",),
                             geoid_correction=geoid)
    except Exception as exc:  # egm96 needs pyproj; fall back to a constant offset.
        print(f"  egm96 correction failed ({exc}); "
              f"using constant Tokyo geoid offset +36.7 m", flush=True)
        model = load_plateau(plateau_dir, zone=9, kinds=("bldg",),
                             geoid_correction=36.7)
    print(f"  triangles={len(model.triangles)}", flush=True)

    print("extracting diffraction edges (UTD wedge geometry) ...", flush=True)
    edges = extract_diffraction_edges(
        model.triangles, route_ecef=ground_truth, route_margin_m=250.0,
        min_edge_length_m=3.0, min_dihedral_deg=20.0,
        include_boundary_edges=True, voxel_size_m=2.0, max_edges=4000)
    welded = int(edges.size - np.count_nonzero(edges.is_boundary))
    print(f"  edges={edges.size} (welded={welded}, boundary={edges.size - welded})",
          flush=True)

    mask = np.radians(elevation_mask_deg)
    kw = dict(max_paths=max_paths, max_edge_range_m=250.0,
              max_ray_edge_distance_m=20.0, max_excess_path_m=120.0)

    # Reflection pooling: for NLOS satellites, a specular building reflection (if
    # one reaches the receiver) is usually a stronger replica with a larger excess
    # delay than grazing diffraction, so it should dominate the tracked NLOS bias
    # and push the predicted magnitude toward the real urban scale. We test that
    # empirically here -- per epoch the image method is run on an rx-local mesh
    # cull and the resulting reflections (amplitude = Fresnel |Gamma|) are pooled
    # with the diffracted replicas before picking the tracked path.
    tri_all = np.asarray(model.triangles, dtype=float)
    tri_centroid = tri_all.mean(axis=1) if pool_reflections else None
    n_nlos_with_refl = 0  # NLOS sat-epochs that gained a reflection candidate

    real_cand, sim_knife, sim_utd = [], [], []
    ke_att_db, ut_att_db = [], []  # diffraction attenuation per path (amplitude level)
    # True-NLOS section: |residual| of NLOS candidate sats and the tracked-replica
    # bias predicted by each diffraction amplitude model.
    nlos_real, nlos_knife, nlos_utd = [], [], []
    # Reflection-only NLOS bias, paired with the real |residual| of the same sats,
    # to isolate what specular reflections alone contribute to the magnitude.
    nlos_real_r, nlos_refl = [], []
    # Every visible satellite's |residual| + LOS flag, for the reference-quality
    # guard (is the residual a usable NLOS ground truth in the first place?).
    all_abs_resid, all_is_los = [], []
    n_cand_sat = 0
    for i in range(0, len(data["times"]), epoch_stride):
        sat = np.asarray(data["sat_ecef"][i], dtype=float)
        pr = np.asarray(data["pseudoranges"][i], dtype=float)
        if sat.shape[0] == 0:
            continue
        rx = ground_truth[i]
        # data["sat_ecef"] is already transmission-time / Sagnac corrected by the
        # loader (correct_transmission_time). The tens-of-metres correction is
        # negligible for the angular geometry below (elevation / LOS / diffraction)
        # at 2e7 m range; it only matters here, in the geometric range.
        geom = np.linalg.norm(sat - rx, axis=1)
        el, _ = elevation_azimuth(rx, sat)
        atmo = tropo_delays(rx, sat)
        pre = pr - geom - atmo

        try:
            is_los = np.asarray(model.check_los(rx, sat), dtype=bool)
        except Exception:
            is_los = np.ones(sat.shape[0], dtype=bool)

        # Receiver clock from LOS satellites (robust to NLOS long-bias pull).
        ref = is_los & (el >= mask)
        clock = float(np.median(pre[ref])) if int(ref.sum()) >= 3 else float(np.median(pre))
        resid = pre - clock  # tropo-purified, receiver-clock-removed residual

        ke = compute_diffraction_paths(rx, sat, edges, **kw)
        ut = compute_utd_diffraction_paths(rx, sat, edges, mode="absorbing", **kw)

        # Per-epoch specular reflections for the NLOS satellites only (image
        # method on the rx-local mesh cull). refl_by_sat[s] is the list of
        # Fresnel-weighted reflection replicas for satellite s (empty otherwise).
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
            # Reference-quality bookkeeping over every visible satellite (not just
            # diffraction candidates): can the residual tell LOS from NLOS at all?
            all_abs_resid.append(abs(float(resid[s])))
            all_is_los.append(bool(is_los[s]))
            has_ke, has_ut = bool(ke[s]), bool(ut[s])
            refl = refl_by_sat.get(s, [])
            if not (has_ke or has_ut or refl):
                continue
            if is_los[s]:
                # LOS: bounded multipath bias from the diffracted replica.
                n_cand_sat += 1
                real_cand.append(float(resid[s]))
                if has_ke:
                    sim_knife.extend(predict_bias_samples_m(
                        candidates_from_paths(ke[s]), n_phase=n_phase).tolist())
                    ke_att_db.extend(p.attenuation_db for p in ke[s])
                if has_ut:
                    sim_utd.extend(predict_bias_samples_m(
                        candidates_from_paths(ut[s]), n_phase=n_phase).tolist())
                    ut_att_db.extend(p.attenuation_db for p in ut[s])
            else:
                # NLOS: receiver tracks the strongest trackable replica across
                # the pooled reflection + diffraction candidates; its full excess
                # delay is the pseudorange bias.
                if refl:
                    n_nlos_with_refl += 1
                    # Reflection-only prediction, paired with this sat's residual,
                    # isolates the specular contribution to the bias magnitude.
                    br = nlos_bias_m(refl)
                    if br is not None:
                        nlos_real_r.append(abs(float(resid[s])))
                        nlos_refl.append(br)
                bk = pooled_nlos_bias_m(ke[s], refl) if (has_ke or refl) else None
                bu = pooled_nlos_bias_m(ut[s], refl) if (has_ut or refl) else None
                if bk is None and bu is None:
                    continue
                nlos_real.append(abs(float(resid[s])))
                if bk is not None:
                    nlos_knife.append(bk)
                if bu is not None:
                    nlos_utd.append(bu)

    # Guard first: is the residual even a usable NLOS ground truth? If LOS
    # satellites already carry tens of metres of residual and |residual| cannot
    # rank NLOS above LOS (AUC ~ 0.5), the diffraction/NLOS comparisons below are
    # measuring contamination, not the model.
    if all_abs_resid:
        q = residual_reference_quality(
            np.asarray(all_abs_resid), np.asarray(all_is_los, dtype=bool))
        print("\n" + format_reference_quality(q), flush=True)

    real_cand = np.asarray(real_cand)
    print(f"\ncandidate satellite-epochs: {n_cand_sat}", flush=True)
    print(f"real residual samples: {real_cand.size}, "
          f"sim(knife)={len(sim_knife)}, sim(utd)={len(sim_utd)}", flush=True)
    if real_cand.size == 0 or not sim_knife or not sim_utd:
        print("insufficient candidates for a benchmark (try more epochs).")
        return None

    res = benchmark_models(real_cand, {"knife_edge": sim_knife, "utd": sim_utd})
    print(f"\nreal candidate residuals: n={res['n_real']}, "
          f"mean={res['real_mean_bias_m']:+.2f} m")
    print(f"{'model':<12}{'n_sim':>8}{'mean_m':>10}{'W1':>10}{'KS':>10}")
    for name, m in res["models"].items():
        print(f"{name:<12}{m['n_sim']:>8}{m['mean_bias_m']:>10.2f}"
              f"{m['wasserstein']:>10.3f}{m['ks']:>10.3f}")
    print(f"\nbest by Wasserstein-1: {res['best_wasserstein']}")
    print(f"best by KS           : {res['best_ks']}")

    # Amplitude-level difference (where the two models diverge most, and what
    # Zhang & Hsu compared via C/N0): diffraction attenuation per path.
    ke_db = np.asarray(ke_att_db); ut_db = np.asarray(ut_att_db)
    if ke_db.size and ut_db.size:
        print("\ndiffraction attenuation [dB] per path (amplitude-level model gap):")
        print(f"  knife_edge: median={np.median(ke_db):6.2f}  "
              f"p90={np.percentile(ke_db, 90):6.2f}  n={ke_db.size}")
        print(f"  utd       : median={np.median(ut_db):6.2f}  "
              f"p90={np.percentile(ut_db, 90):6.2f}  n={ut_db.size}")
        print(f"  median gap (UTD - knife): {np.median(ut_db) - np.median(ke_db):+.2f} dB")

    # True-NLOS bias level: which model's tracked-replica bias distribution is
    # closer to the real |residual| of NLOS satellites.
    if nlos_real and nlos_knife and nlos_utd:
        nlos_res = benchmark_models(
            np.abs(nlos_real), {"knife_edge": nlos_knife, "utd": nlos_utd})
        print(f"\nTRUE-NLOS bias level (|residual| of NLOS candidate sats, "
              f"n_real={nlos_res['n_real']}, real_mean={nlos_res['real_mean_bias_m']:.2f} m):")
        print(f"{'model':<12}{'n_meas':>8}{'mean_m':>10}{'W1':>10}{'KS':>10}")
        for name, m in nlos_res["models"].items():
            print(f"{name:<12}{m['n_sim']:>8}{m['mean_bias_m']:>10.2f}"
                  f"{m['wasserstein']:>10.3f}{m['ks']:>10.3f}")
        print(f"  best by W1: {nlos_res['best_wasserstein']}, "
              f"best by KS: {nlos_res['best_ks']}")
        print(f"  measured NLOS sats: knife={len(nlos_knife)}, utd={len(nlos_utd)} "
              f"(UTD keeps more diffraction paths trackable)")
        if pool_reflections:
            print(f"  NLOS sats with a pooled specular reflection candidate: "
                  f"{n_nlos_with_refl} "
                  f"(R<{refl_cull_radius_m:.0f} m image method, {refl_material}, "
                  f"point_tol={refl_point_tol_m:.1f} m)")

    # Reflection-only NLOS bias: what specular reflections alone predict for the
    # sats they touch, vs those sats' real residuals. This is a control: if the
    # reflections were the tracked NLOS replica, predicted ~ real. In practice it
    # exposes whether the recovered reflections actually match reality.
    if nlos_refl:
        nlos_refl = np.asarray(nlos_refl); nlos_real_r = np.asarray(nlos_real_r)
        rr = benchmark_models(nlos_real_r, {"reflection_only": nlos_refl})
        m = rr["models"]["reflection_only"]
        print(f"\nREFLECTION-ONLY NLOS bias (sats with a specular reflection, "
              f"n={rr['n_real']}):")
        print(f"  real |residual|  mean={rr['real_mean_bias_m']:.2f} m  "
              f"median={np.median(nlos_real_r):.2f} m")
        print(f"  predicted (refl) mean={m['mean_bias_m']:.2f} m  "
              f"median={np.median(nlos_refl):.2f} m  "
              f"W1={m['wasserstein']:.3f}  KS={m['ks']:.3f}")
        verdict = ("match -> reflections explain these sats"
                   if m['wasserstein'] < 10.0 else
                   "MISMATCH -> recovered reflections do NOT match the measured "
                   "bias (these sats' large/true-NLOS error comes from geometry "
                   "this first-order model does not capture)")
        print(f"  {verdict}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(tempfile.mkdtemp(prefix="diffraction_benchmark_")) / f"{site}_cdf.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, vals in (("real", real_cand), ("knife_edge", np.asarray(sim_knife)),
                            ("utd", np.asarray(sim_utd))):
            v = np.sort(vals[np.isfinite(vals)])
            ax.plot(v, np.linspace(0, 1, v.size), label=label)
        ax.set_xlabel("pseudorange bias [m]"); ax.set_ylabel("CDF")
        ax.set_title(f"Diffraction model vs real residuals ({site})")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(out, dpi=110); print(f"\nCDF plot: {out}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")
    return res


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Odaiba"
    max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    main(site=site, max_epochs=max_ep)
