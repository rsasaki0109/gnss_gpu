"""Render the README NLOS-diffraction benchmark figure.

Runs the diffraction benchmark on real UrbanNav data over a PLATEAU city mesh
(via :func:`demo_diffraction_benchmark.main`) and renders a two-panel summary:

  (left)  After the transmission-time / Sagnac correction the pseudorange
          residual is a CLEAN NLOS ground truth -- LOS satellites sit near 0 m,
          NLOS satellites are clearly separated (high AUC).
  (right) On that clean reference the UTD diffraction model reproduces the
          measured multipath-bias distribution better than knife-edge
          (smaller Wasserstein-1).

Usage:
    # installed package (has the _raytrace .so used by check_los)
    python examples/plot_nlos_diffraction_figure.py [site] [max_epochs]

Writes docs/assets/figures/nlos_diffraction_benchmark.png.
"""

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo_diffraction_benchmark import main as run_benchmark

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT = _REPO_ROOT / "docs" / "assets" / "figures" / "nlos_diffraction_benchmark.png"

# Colour-blind-friendly palette.
C_REAL, C_KNIFE, C_UTD = "#222222", "#d55e00", "#0072b2"
C_LOS, C_NLOS = "#0072b2", "#d55e00"


def _cdf(ax, vals, **kw):
    v = np.sort(np.asarray(vals, float))
    v = v[np.isfinite(v)]
    ax.plot(v, np.linspace(0.0, 1.0, v.size), **kw)


def main(site: str = "Odaiba", max_epochs: int = 60) -> Path:
    res = run_benchmark(site=site, max_epochs=max_epochs)
    if res is None or "arrays" not in res:
        raise SystemExit("benchmark produced no data (need the installed package "
                         "with the _raytrace .so, and the UrbanNav/PLATEAU data)")

    a = res["arrays"]
    q = res.get("reference_quality", {})
    models = res["models"]
    w1_knife = models["knife_edge"]["wasserstein"]
    w1_utd = models["utd"]["wasserstein"]

    abs_resid = a["all_abs_resid"]
    is_los = a["all_is_los"]
    los = abs_resid[is_los]
    nlos = abs_resid[~is_los]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # --- Left: residual is a clean NLOS ground truth ---
    _cdf(axL, los, color=C_LOS, lw=2.2,
         label=f"LOS sats  (median {np.median(los):.1f} m)")
    if nlos.size:
        _cdf(axL, nlos, color=C_NLOS, lw=2.2,
             label=f"NLOS sats (median {np.median(nlos):.1f} m)")
    axL.set_xlim(0, 30)
    axL.set_xlabel("|pseudorange residual| [m]")
    axL.set_ylabel("CDF")
    axL.set_title("Corrected residual separates LOS from NLOS")
    auc = q.get("auc")
    verdict = "CLEAN" if q.get("is_clean_reference") else ""
    tag = f"AUC = {auc:.2f}   {verdict}".strip() if auc is not None else ""
    if tag:
        axL.text(0.97, 0.07, tag, transform=axL.transAxes, ha="right",
                 fontsize=11, fontweight="bold", color="#1a7a1a",
                 bbox=dict(boxstyle="round", fc="white", ec="#1a7a1a", alpha=0.9))
    axL.legend(loc="lower right", bbox_to_anchor=(1.0, 0.18), fontsize=9)
    axL.grid(True, alpha=0.3)

    # --- Right: UTD reproduces the measured bias better than knife-edge ---
    _cdf(axR, np.abs(a["real_cand"]), color=C_REAL, lw=2.4, label="real residual")
    _cdf(axR, np.abs(a["sim_knife"]), color=C_KNIFE, lw=2.0, ls="--",
         label=f"knife-edge  (W1 {w1_knife:.2f})")
    _cdf(axR, np.abs(a["sim_utd"]), color=C_UTD, lw=2.0,
         label=f"UTD  (W1 {w1_utd:.2f})")
    axR.set_xlim(0, 12)
    axR.set_xlabel("|multipath bias| [m]")
    axR.set_ylabel("CDF")
    axR.set_title("UTD matches real multipath better than knife-edge")
    axR.legend(loc="lower right", fontsize=9)
    axR.grid(True, alpha=0.3)

    fig.suptitle(
        f"Ray-traced NLOS diffraction vs real UrbanNav {site} residuals "
        f"(PLATEAU mesh, transmission-time corrected)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=130)
    print(f"wrote {_OUT}")
    return _OUT


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Odaiba"
    max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    main(site=site, max_epochs=max_ep)
