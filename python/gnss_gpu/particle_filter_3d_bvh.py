"""GPU-accelerated BVH-accelerated 3D-aware Particle Filter for GNSS positioning.

Extends ParticleFilter3D to use an O(log n) BVH ray traversal instead of the
O(n) linear triangle scan.  This makes the weight update step practical for
large urban meshes (10K+ triangles) without sacrificing accuracy.

Usage
-----
    from gnss_gpu.particle_filter_3d_bvh import ParticleFilter3DBVH
    from gnss_gpu.bvh import BVHAccelerator

    bvh = BVHAccelerator.from_building_model(building)
    pf = ParticleFilter3DBVH(bvh=bvh, n_particles=1_000_000, ...)
    pf.initialize(position_ecef=pos0, clock_bias=cb0)
    for obs in observations:
        pf.predict(dt=1.0)
        pf.update(obs.sat_ecef, obs.pseudoranges)
    pos_est = pf.estimate()
"""

import numpy as np

from gnss_gpu.input_validation import (
    finite_float,
    positive_float,
    validate_gnss_observation_epoch,
)
from gnss_gpu.particle_filter_3d import ParticleFilter3D
from gnss_gpu.bvh import BVHAccelerator


class ParticleFilter3DBVH(ParticleFilter3D):
    """Particle filter with BVH-accelerated 3D building-aware NLOS handling.

    This class has the same interface as :class:`ParticleFilter3D` but
    replaces the per-particle O(n_triangles) linear scan with an
    O(log n_triangles) BVH traversal.  It is drop-in compatible: swap
    ``building_model`` (a ``BuildingModel``) for ``bvh`` (a
    ``BVHAccelerator``) and everything else stays the same.

    Parameters
    ----------
    bvh : BVHAccelerator
        Pre-built BVH acceleration structure containing the building mesh.
    sigma_los : float
        Observation sigma for LOS satellites [m] (tight, e.g., 3.0).
    sigma_nlos : float
        Observation sigma for NLOS satellites [m] (loose, e.g., 30.0).
    nlos_bias : float
        Expected positive pseudorange bias for NLOS satellites [m]. Applied
        only when the residual itself is positive.
    blocked_nlos_prob : float
        Prior probability of NLOS when the ray tracer says blocked.
    clear_nlos_prob : float
        Prior probability of NLOS when the ray tracer says clear.
    nlos_bias_slope : float
        Optional elevation slope on the NLOS bias [m/deg]: the bias grows by
        this amount for each degree of elevation below
        ``nlos_bias_elev_ref_deg``. Default 0 disables the term.
    nlos_bias_elev_ref_deg : float
        Elevation [deg] anchoring the bias ramp and the blocking prior.
    nlos_prob_high_elev : float
        Optional elevation-conditioned blocking prior: P(NLOS | blocked) at
        high elevation. A negative value (default) disables the modulation.
    nlos_beta : float
        Generalized-Gaussian shape exponent for the NLOS likelihood:
        2 = Gaussian (default), 1 = Laplace (heavy-tailed).

    Notes
    -----
    The elevation slope, elevation-conditioned prior, and Laplace shape are
    *opt-in* capabilities. Real-data validation (UrbanNav Odaiba + PLATEAU,
    300 epochs) found that enabling the per-particle ray-conditioned NLOS
    likelihood degrades the filter regardless of these knobs: particles can
    "explain away" a measurement by drifting into NLOS geometry, so the cloud
    is pulled into building shadows. The defaults therefore preserve the
    conservative behaviour and the model is not promoted to a mainline lever.
    See ``internal_docs/decisions.md`` D-037.

    **kwargs
        Additional keyword arguments forwarded to ``ParticleFilter.__init__``.
    """

    def __init__(self, bvh, sigma_los=3.0, sigma_nlos=30.0,
                 nlos_bias=20.0, blocked_nlos_prob=1.0,
                 clear_nlos_prob=0.0, nlos_bias_slope=0.0,
                 nlos_bias_elev_ref_deg=35.0, nlos_prob_high_elev=-1.0,
                 nlos_beta=2.0, **kwargs):
        # Bypass ParticleFilter3D.__init__ which requires a BuildingModel.
        # Instead call the grandparent (ParticleFilter) directly, then
        # set up the BVH-specific attributes.
        from gnss_gpu.particle_filter import ParticleFilter
        ParticleFilter.__init__(self, **kwargs)

        if not isinstance(bvh, BVHAccelerator):
            raise TypeError("bvh must be a BVHAccelerator instance")

        self.bvh = bvh
        self.sigma_los = positive_float("sigma_los", sigma_los)
        self.sigma_nlos = positive_float("sigma_nlos", sigma_nlos)
        self.nlos_bias = finite_float("nlos_bias", nlos_bias)
        self.blocked_nlos_prob = finite_float("blocked_nlos_prob", blocked_nlos_prob)
        self.clear_nlos_prob = finite_float("clear_nlos_prob", clear_nlos_prob)
        self.nlos_bias_slope = finite_float("nlos_bias_slope", nlos_bias_slope)
        self.nlos_bias_elev_ref_deg = finite_float(
            "nlos_bias_elev_ref_deg", nlos_bias_elev_ref_deg)
        self.nlos_prob_high_elev = finite_float(
            "nlos_prob_high_elev", nlos_prob_high_elev)
        self.nlos_beta = positive_float("nlos_beta", nlos_beta)

        from gnss_gpu._gnss_gpu_pf3d_bvh import pf_weight_3d_bvh as _pf_weight_3d_bvh
        self._pf_weight_3d_bvh = _pf_weight_3d_bvh

    # ------------------------------------------------------------------
    # Override ParticleFilter3D.update() to use the BVH kernel
    # ------------------------------------------------------------------

    def update(self, sat_ecef, pseudoranges, weights=None):
        """Weight update using BVH-accelerated 3D ray tracing.

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].
        pseudoranges : array_like, shape (n_sat,)
            Observed pseudoranges [m].
        weights : array_like, shape (n_sat,), optional
            Per-satellite weights. Defaults to ones.
        """
        if not self._initialized:
            raise RuntimeError(
                "ParticleFilter3DBVH not initialized. Call initialize() first.")

        sat, pr, weights, n_sat = validate_gnss_observation_epoch(
            sat_ecef, pseudoranges, weights)

        nodes_flat = self.bvh._nodes_flat
        sorted_tris = self.bvh._sorted_tris
        if sorted_tris.ndim != 3 or sorted_tris.shape[1:] != (3, 3):
            raise ValueError("bvh sorted triangles must have shape (n_tri, 3, 3)")
        n_tri = sorted_tris.shape[0]
        if n_tri > 0:
            if nodes_flat.ndim != 2 or nodes_flat.shape[1] != 10 or nodes_flat.shape[0] < 1:
                raise ValueError("bvh nodes_flat must have shape (n_nodes, 10) with n_nodes >= 1")
            if not np.all(np.isfinite(nodes_flat)):
                raise ValueError("bvh nodes_flat must be finite")
            if not np.all(np.isfinite(sorted_tris)):
                raise ValueError("bvh sorted triangles must be finite")

        self._pf_weight_3d_bvh(
            self._px, self._py, self._pz, self._pcb,
            sat.ravel(), pr, weights,
            nodes_flat,
            sorted_tris,
            self._log_weights,
            self.n_particles, n_sat,
            float(self.sigma_los), float(self.sigma_nlos),
            float(self.nlos_bias),
            float(self.blocked_nlos_prob),
            float(self.clear_nlos_prob),
            float(self.nlos_bias_slope),
            float(self.nlos_bias_elev_ref_deg),
            float(self.nlos_prob_high_elev),
            float(self.nlos_beta))

        # Adaptive resampling based on ESS
        ess = self.get_ess()
        if ess < self.ess_threshold * self.n_particles:
            self._resample()
