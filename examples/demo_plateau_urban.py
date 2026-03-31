#!/usr/bin/env python3
"""Demo: PLATEAU 3D City Model + Particle Filter Urban Positioning Experiment.

Loads PLATEAU CityGML building data (Tokyo Station area), simulates a
pedestrian walking through the urban canyon, and compares three positioning
methods under multipath-contaminated observations:

  1. WLS  -- standard weighted least squares (no building model)
  2. PF   -- standard particle filter (no building model)
  3. PF3D -- 3D-aware particle filter (uses building model for NLOS-aware
             likelihood weighting)

Usage:
    PYTHONPATH=python python3 examples/demo_plateau_urban.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# WGS-84 helpers (pure-Python, no CUDA dependency)
# ---------------------------------------------------------------------------

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2


def _lla_to_ecef(lat_rad, lon_rad, alt):
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.array([x, y, z])


def _ecef_to_lla(x, y, z):
    b = WGS84_A * (1.0 - WGS84_F)
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * WGS84_A, p * b)
    lat = np.arctan2(
        z + WGS84_E2 / (1.0 - WGS84_E2) * b * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    lon = np.arctan2(y, x)
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt


def _enu_to_ecef_rotation(lat_rad, lon_rad):
    """ENU-to-ECEF rotation matrix."""
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    return np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [ 0.0,      cos_lat,           sin_lat],
    ])


# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------

# Tokyo Station area -- matches sample_plateau.gml zone 9 data
ORIGIN_LLA_DEG = (35.6812, 139.7671, 5.0)
ORIGIN_LLA_RAD = (np.radians(ORIGIN_LLA_DEG[0]),
                  np.radians(ORIGIN_LLA_DEG[1]),
                  ORIGIN_LLA_DEG[2])
ORIGIN_ECEF = _lla_to_ecef(*ORIGIN_LLA_RAD)
R_ENU2ECEF = _enu_to_ecef_rotation(ORIGIN_LLA_RAD[0], ORIGIN_LLA_RAD[1])

# 8 hardcoded GPS satellites in ECEF [m]
SAT_ECEF = np.array([
    [-14985000.0,  -3988000.0, 21474000.0],   # PRN 01 - high elevation
    [ -9575000.0,  15498000.0, 19457000.0],   # PRN 03 - medium elevation
    [  7624000.0, -16218000.0, 19843000.0],   # PRN 06 - medium elevation
    [ 16305000.0,  12037000.0, 17183000.0],   # PRN 09 - medium-low elevation
    [-20889000.0,  13759000.0,  8291000.0],   # PRN 11 - low elevation
    [  5463000.0,  24413000.0,  8934000.0],   # PRN 14 - low elevation
    [ 22169000.0,   3975000.0, 13781000.0],   # PRN 17 - medium elevation
    [-11527000.0, -19421000.0, 13682000.0],   # PRN 22 - medium-low elevation
], dtype=np.float64)

N_SAT = len(SAT_ECEF)
N_EPOCHS = 60
DT = 1.0
TRUE_CB_M = 3000.0  # receiver clock bias in metres


# ---------------------------------------------------------------------------
# CPU fallback: LOS check using Moeller-Trumbore triangle intersection
# ---------------------------------------------------------------------------

def _ray_triangle_intersect(origin, direction, v0, v1, v2):
    """Moeller-Trumbore ray-triangle intersection. Returns True if hit."""
    EPS = 1e-9
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(direction, e2)
    a = np.dot(e1, h)
    if -EPS < a < EPS:
        return False
    f = 1.0 / a
    s = origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, e1)
    v = f * np.dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(e2, q)
    return t > EPS and t < 1.0  # hit must be between origin and satellite


def _check_los_cpu(rx_ecef, sat_ecef, triangles):
    """Pure-Python LOS check against a triangle mesh.

    Args:
        rx_ecef: (3,) receiver position.
        sat_ecef: (n_sat, 3) satellite positions.
        triangles: (N, 3, 3) triangle mesh.

    Returns:
        is_los: (n_sat,) boolean array.
    """
    n_sat = sat_ecef.shape[0]
    n_tri = triangles.shape[0]
    is_los = np.ones(n_sat, dtype=bool)
    for j in range(n_sat):
        direction = sat_ecef[j] - rx_ecef
        for k in range(n_tri):
            if _ray_triangle_intersect(rx_ecef, direction,
                                       triangles[k, 0],
                                       triangles[k, 1],
                                       triangles[k, 2]):
                is_los[j] = False
                break
    return is_los


# ---------------------------------------------------------------------------
# CPU fallback: WLS solver
# ---------------------------------------------------------------------------

def _wls_solve_py(sat_ecef, pseudoranges, weights, max_iter=10, tol=1e-4):
    n_sat = len(pseudoranges)
    state = np.zeros(4)
    for _iteration in range(max_iter):
        H = np.zeros((n_sat, 4))
        pred_pr = np.zeros(n_sat)
        for j in range(n_sat):
            dx = sat_ecef[j] - state[:3]
            r = np.linalg.norm(dx)
            if r < 1.0:
                r = 1.0
            pred_pr[j] = r + state[3]
            H[j, :3] = -dx / r
            H[j, 3] = 1.0
        residual = pseudoranges - pred_pr
        W = np.diag(weights)
        try:
            delta = np.linalg.solve(H.T @ W @ H, H.T @ W @ residual)
        except np.linalg.LinAlgError:
            break
        state += delta
        if np.linalg.norm(delta[:3]) < tol:
            return state, _iteration + 1
    return state, max_iter


# ---------------------------------------------------------------------------
# CPU fallback: Simple Particle Filter
# ---------------------------------------------------------------------------

class _SimpleParticleFilter:
    """Pure-Python particle filter for GNSS positioning."""

    def __init__(self, n_particles=10000, sigma_pos=1.0, sigma_cb=300.0,
                 sigma_pr=5.0, seed=42):
        self.n = n_particles
        self.sigma_pos = sigma_pos
        self.sigma_cb = sigma_cb
        self.sigma_pr = sigma_pr
        self.rng = np.random.default_rng(seed)
        self.particles = None
        self.log_w = None

    def initialize(self, pos_ecef, cb=0.0, spread_pos=100.0, spread_cb=1000.0):
        self.particles = np.column_stack([
            self.rng.normal(pos_ecef[0], spread_pos, self.n),
            self.rng.normal(pos_ecef[1], spread_pos, self.n),
            self.rng.normal(pos_ecef[2], spread_pos, self.n),
            self.rng.normal(cb, spread_cb, self.n),
        ])
        self.log_w = np.zeros(self.n)

    def predict(self, velocity_ecef=None, dt=1.0):
        if velocity_ecef is not None:
            self.particles[:, :3] += velocity_ecef * dt
        self.particles[:, :3] += self.rng.normal(0, self.sigma_pos * dt,
                                                  (self.n, 3))
        self.particles[:, 3] += self.rng.normal(0, self.sigma_cb * dt, self.n)

    def update(self, sat_ecef, pseudoranges, sat_weights=None):
        n_sat = len(pseudoranges)
        log_lik = np.zeros(self.n)
        for j in range(n_sat):
            dx = sat_ecef[j] - self.particles[:, :3]
            r = np.sqrt(np.sum(dx ** 2, axis=1))
            pred_pr = r + self.particles[:, 3]
            diff = pseudoranges[j] - pred_pr
            w = 1.0 if sat_weights is None else sat_weights[j]
            log_lik += -0.5 * w * (diff / self.sigma_pr) ** 2
        self.log_w += log_lik
        max_lw = np.max(self.log_w)
        self.log_w -= max_lw
        w = np.exp(self.log_w)
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        if ess < 0.5 * self.n:
            indices = self.rng.choice(self.n, size=self.n, p=w)
            self.particles = self.particles[indices]
            self.log_w[:] = 0.0

    def estimate(self):
        w = np.exp(self.log_w)
        w /= w.sum()
        return np.average(self.particles, axis=0, weights=w)


# ---------------------------------------------------------------------------
# CPU fallback: 3D-aware Particle Filter
# ---------------------------------------------------------------------------

class _SimpleParticleFilter3D(_SimpleParticleFilter):
    """Particle filter with building-aware NLOS handling (CPU fallback)."""

    def __init__(self, triangles, sigma_los=3.0, sigma_nlos=30.0,
                 nlos_bias=20.0, **kwargs):
        super().__init__(**kwargs)
        self.triangles = triangles
        self.sigma_los = sigma_los
        self.sigma_nlos = sigma_nlos
        self.nlos_bias = nlos_bias

    def update(self, sat_ecef, pseudoranges, sat_weights=None):
        """NLOS-aware weight update.

        For a subset of representative particles, check LOS/NLOS and adjust
        the likelihood accordingly. NLOS satellites get wider sigma and a bias
        correction.
        """
        n_sat = len(pseudoranges)

        # Use the current weighted mean as the representative position for
        # the LOS check (checking every particle is too expensive on CPU).
        est = self.estimate()
        is_los = _check_los_cpu(est[:3], sat_ecef, self.triangles)

        log_lik = np.zeros(self.n)
        for j in range(n_sat):
            dx = sat_ecef[j] - self.particles[:, :3]
            r = np.sqrt(np.sum(dx ** 2, axis=1))
            pred_pr = r + self.particles[:, 3]
            diff = pseudoranges[j] - pred_pr

            if is_los[j]:
                sigma = self.sigma_los
            else:
                # NLOS: correct for expected positive bias, use wider sigma
                diff = diff + self.nlos_bias
                sigma = self.sigma_nlos

            w = 1.0 if sat_weights is None else sat_weights[j]
            log_lik += -0.5 * w * (diff / sigma) ** 2

        self.log_w += log_lik
        max_lw = np.max(self.log_w)
        self.log_w -= max_lw
        w = np.exp(self.log_w)
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        if ess < 0.5 * self.n:
            indices = self.rng.choice(self.n, size=self.n, p=w)
            self.particles = self.particles[indices]
            self.log_w[:] = 0.0


# ---------------------------------------------------------------------------
# VulnerabilityMap CPU fallback
# ---------------------------------------------------------------------------

def _compute_dop_cpu(rx_ecef, sat_ecef, el_mask_rad=np.radians(10.0)):
    """Compute HDOP at a single grid point (pure Python)."""
    up = rx_ecef / np.linalg.norm(rx_ecef)
    visible = []
    for j in range(sat_ecef.shape[0]):
        dx = sat_ecef[j] - rx_ecef
        r = np.linalg.norm(dx)
        sin_el = np.dot(dx / r, up)
        if sin_el > np.sin(el_mask_rad):
            visible.append(j)
    if len(visible) < 4:
        return 99.9, len(visible)
    H = np.zeros((len(visible), 4))
    for idx, j in enumerate(visible):
        dx = sat_ecef[j] - rx_ecef
        r = np.linalg.norm(dx)
        H[idx, :3] = -dx / r
        H[idx, 3] = 1.0
    try:
        Q = np.linalg.inv(H.T @ H)
        hdop = np.sqrt(Q[0, 0] + Q[1, 1])
    except np.linalg.LinAlgError:
        hdop = 99.9
    return hdop, len(visible)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  PLATEAU 3D City Model + Particle Filter Urban Positioning")
    print("=" * 72)

    rng = np.random.default_rng(42)
    base_dir = Path(__file__).resolve().parent.parent
    gml_path = base_dir / "data" / "sample_plateau.gml"

    # ------------------------------------------------------------------
    # Step 1: Load PLATEAU CityGML data
    # ------------------------------------------------------------------
    print(f"\n[1] Loading PLATEAU CityGML data")
    print(f"    File: {gml_path}")

    try:
        from gnss_gpu.io.plateau import PlateauLoader
        loader = PlateauLoader(zone=9)
        building_model = loader.load_citygml(gml_path)
        triangles = building_model.triangles
        bm_source = "gnss_gpu (PlateauLoader)"
    except Exception as e:
        print(f"    PlateauLoader failed ({e}), building model from CityGML parser")
        # Fallback: parse CityGML directly and build triangles with a
        # simplified coordinate conversion
        from gnss_gpu.io.citygml import parse_citygml
        buildings = parse_citygml(gml_path)
        loader = PlateauLoader(zone=9)
        all_tris = []
        for bldg in buildings:
            for polygon in bldg.polygons:
                ecef_coords = loader._polygon_to_ecef(polygon)
                tris = loader._polygon_to_triangles(ecef_coords)
                if tris is not None:
                    all_tris.append(tris)
        if all_tris:
            triangles = np.concatenate(all_tris, axis=0)
        else:
            triangles = np.empty((0, 3, 3), dtype=np.float64)

        try:
            from gnss_gpu.raytrace import BuildingModel
            building_model = BuildingModel(triangles)
        except Exception:
            building_model = None
        bm_source = "CityGML parser (fallback)"

    print(f"    Source    : {bm_source}")
    print(f"    Triangles: {triangles.shape[0]}")
    print(f"    Buildings: 3 (bldg_001: 20m, bldg_002: 45m, bldg_003: 12m)")

    # ------------------------------------------------------------------
    # Step 2: Define pedestrian trajectory (~100m walk, 60 epochs)
    # ------------------------------------------------------------------
    print(f"\n[2] Defining pedestrian trajectory")
    print(f"    Epochs: {N_EPOCHS} at {DT:.0f}s intervals")

    # Walk: start at (-10, -20) ENU, walk east ~60m then turn north ~40m
    # This path goes between buildings, creating realistic NLOS conditions
    traj_enu = np.zeros((N_EPOCHS, 3))
    speed = 1.2  # m/s (pedestrian walking speed)
    turn_epoch = 35  # epoch where we turn north

    for i in range(N_EPOCHS):
        if i < turn_epoch:
            # Walk east
            traj_enu[i, 0] = -10.0 + speed * i * DT
            traj_enu[i, 1] = -20.0
        else:
            # Turn north
            traj_enu[i, 0] = -10.0 + speed * turn_epoch * DT
            traj_enu[i, 1] = -20.0 + speed * (i - turn_epoch) * DT
        traj_enu[i, 2] = 1.5  # receiver height above ground

    # Convert ENU to ECEF
    traj_ecef = np.zeros((N_EPOCHS, 3))
    for i in range(N_EPOCHS):
        traj_ecef[i] = ORIGIN_ECEF + R_ENU2ECEF @ traj_enu[i]

    total_dist = np.sum(np.sqrt(np.sum(np.diff(traj_enu, axis=0) ** 2, axis=1)))
    print(f"    Total distance : {total_dist:.1f} m")
    print(f"    Start (ENU)    : ({traj_enu[0, 0]:.1f}, {traj_enu[0, 1]:.1f}) m")
    print(f"    Turn  (ENU)    : ({traj_enu[turn_epoch, 0]:.1f}, "
          f"{traj_enu[turn_epoch, 1]:.1f}) m  [epoch {turn_epoch}]")
    print(f"    End   (ENU)    : ({traj_enu[-1, 0]:.1f}, {traj_enu[-1, 1]:.1f}) m")

    # ------------------------------------------------------------------
    # Step 3: For each epoch, compute satellite geometry and pseudoranges
    # ------------------------------------------------------------------
    print(f"\n[3] Generating observations ({N_SAT} satellites x {N_EPOCHS} epochs)")

    # Check LOS/NLOS using the building model
    los_mask = np.ones((N_EPOCHS, N_SAT), dtype=bool)

    try:
        # Try GPU ray tracing
        for i in range(N_EPOCHS):
            los_mask[i] = building_model.check_los(traj_ecef[i], SAT_ECEF)
        los_source = "GPU ray tracing"
    except Exception:
        # CPU fallback
        for i in range(N_EPOCHS):
            los_mask[i] = _check_los_cpu(traj_ecef[i], SAT_ECEF, triangles)
        los_source = "CPU ray tracing"

    n_nlos_total = np.sum(~los_mask)
    n_los_total = np.sum(los_mask)
    print(f"    LOS check  : {los_source}")
    print(f"    LOS signals: {n_los_total} / {N_EPOCHS * N_SAT} "
          f"({100 * n_los_total / (N_EPOCHS * N_SAT):.1f}%)")
    print(f"    NLOS signals: {n_nlos_total} / {N_EPOCHS * N_SAT} "
          f"({100 * n_nlos_total / (N_EPOCHS * N_SAT):.1f}%)")

    # Generate clean pseudoranges
    clean_pr = np.zeros((N_EPOCHS, N_SAT))
    for i in range(N_EPOCHS):
        ranges = np.sqrt(np.sum((SAT_ECEF - traj_ecef[i]) ** 2, axis=1))
        clean_pr[i] = ranges + TRUE_CB_M

    # Add multipath bias for NLOS satellites (20-50m positive bias)
    multipath_bias = np.zeros((N_EPOCHS, N_SAT))
    for i in range(N_EPOCHS):
        for j in range(N_SAT):
            if not los_mask[i, j]:
                multipath_bias[i, j] = rng.uniform(20.0, 50.0)

    # Add noise: 3m for LOS, 15m for NLOS
    noise = np.zeros((N_EPOCHS, N_SAT))
    for i in range(N_EPOCHS):
        for j in range(N_SAT):
            if los_mask[i, j]:
                noise[i, j] = rng.normal(0, 3.0)
            else:
                noise[i, j] = rng.normal(0, 15.0)

    observed_pr = clean_pr + multipath_bias + noise

    print(f"    Multipath bias : mean={np.mean(multipath_bias[~los_mask]):.1f} m "
          f"(NLOS only)" if n_nlos_total > 0 else "    No NLOS signals")
    print(f"    Noise (LOS)    : sigma=3.0 m")
    print(f"    Noise (NLOS)   : sigma=15.0 m")

    # ------------------------------------------------------------------
    # Step 4: WLS positioning (standard, no building model)
    # ------------------------------------------------------------------
    print(f"\n[4] WLS positioning (no building model)")

    weights_uniform = np.ones(N_SAT)
    wls_results = np.zeros((N_EPOCHS, 4))

    try:
        from gnss_gpu import wls_position
        for i in range(N_EPOCHS):
            result, _iters = wls_position(SAT_ECEF, observed_pr[i],
                                          weights_uniform)
            wls_results[i] = result
        wls_source = "GPU"
    except (ImportError, Exception):
        for i in range(N_EPOCHS):
            wls_results[i], _ = _wls_solve_py(SAT_ECEF, observed_pr[i],
                                               weights_uniform)
        wls_source = "CPU"

    wls_errors = np.array([np.linalg.norm(wls_results[i, :3] - traj_ecef[i])
                           for i in range(N_EPOCHS)])
    print(f"    Source: {wls_source}")
    print(f"    Mean error: {np.mean(wls_errors):.2f} m")

    # ------------------------------------------------------------------
    # Step 5: ParticleFilter (standard, no building model)
    # ------------------------------------------------------------------
    print(f"\n[5] ParticleFilter (standard, no building model)")
    n_particles = 20000

    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(n_particles=n_particles, sigma_pos=2.0,
                            sigma_cb=300.0, sigma_pr=10.0, seed=42)
        pf.initialize(wls_results[0, :3], clock_bias=wls_results[0, 3],
                       spread_pos=50.0, spread_cb=500.0)
        pf_results = np.zeros((N_EPOCHS, 4))
        for i in range(N_EPOCHS):
            vel = None
            if i > 0:
                vel = (traj_ecef[i] - traj_ecef[i - 1]) / DT
            pf.predict(velocity=vel, dt=DT)
            pf.update(SAT_ECEF, observed_pr[i])
            pf_results[i] = pf.estimate()
        pf_source = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    GPU unavailable ({e}), using CPU fallback")
        pf = _SimpleParticleFilter(n_particles=n_particles, sigma_pos=2.0,
                                    sigma_cb=300.0, sigma_pr=10.0, seed=42)
        pf.initialize(wls_results[0, :3], cb=wls_results[0, 3],
                       spread_pos=50.0, spread_cb=500.0)
        pf_results = np.zeros((N_EPOCHS, 4))
        for i in range(N_EPOCHS):
            vel = None
            if i > 0:
                vel = (traj_ecef[i] - traj_ecef[i - 1]) / DT
            pf.predict(velocity_ecef=vel, dt=DT)
            pf.update(SAT_ECEF, observed_pr[i])
            pf_results[i] = pf.estimate()
        pf_source = "CPU"

    pf_errors = np.array([np.linalg.norm(pf_results[i, :3] - traj_ecef[i])
                          for i in range(N_EPOCHS)])
    print(f"    Source    : {pf_source}")
    print(f"    Particles: {n_particles}")
    print(f"    Mean error: {np.mean(pf_errors):.2f} m")

    # ------------------------------------------------------------------
    # Step 6: ParticleFilter3D (with building model for NLOS-aware likelihood)
    # ------------------------------------------------------------------
    print(f"\n[6] ParticleFilter3D (NLOS-aware with building model)")

    try:
        from gnss_gpu import ParticleFilter3D
        pf3d = ParticleFilter3D(
            building_model=building_model,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            n_particles=n_particles, sigma_pos=2.0, sigma_cb=300.0,
            sigma_pr=10.0, seed=42,
        )
        pf3d.initialize(wls_results[0, :3], clock_bias=wls_results[0, 3],
                         spread_pos=50.0, spread_cb=500.0)
        pf3d_results = np.zeros((N_EPOCHS, 4))
        for i in range(N_EPOCHS):
            vel = None
            if i > 0:
                vel = (traj_ecef[i] - traj_ecef[i - 1]) / DT
            pf3d.predict(velocity=vel, dt=DT)
            pf3d.update(SAT_ECEF, observed_pr[i])
            pf3d_results[i] = pf3d.estimate()
        pf3d_source = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    GPU unavailable ({e}), using CPU fallback")
        pf3d = _SimpleParticleFilter3D(
            triangles=triangles,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            n_particles=n_particles, sigma_pos=2.0, sigma_cb=300.0,
            sigma_pr=10.0, seed=42,
        )
        pf3d.initialize(wls_results[0, :3], cb=wls_results[0, 3],
                         spread_pos=50.0, spread_cb=500.0)
        pf3d_results = np.zeros((N_EPOCHS, 4))
        for i in range(N_EPOCHS):
            vel = None
            if i > 0:
                vel = (traj_ecef[i] - traj_ecef[i - 1]) / DT
            pf3d.predict(velocity_ecef=vel, dt=DT)
            pf3d.update(SAT_ECEF, observed_pr[i])
            pf3d_results[i] = pf3d.estimate()
        pf3d_source = "CPU"

    pf3d_errors = np.array([np.linalg.norm(pf3d_results[i, :3] - traj_ecef[i])
                            for i in range(N_EPOCHS)])
    print(f"    Source    : {pf3d_source}")
    print(f"    Particles: {n_particles}")
    print(f"    Mean error: {np.mean(pf3d_errors):.2f} m")

    # ------------------------------------------------------------------
    # Step 7: Epoch-by-epoch comparison
    # ------------------------------------------------------------------
    print(f"\n[7] Epoch-by-epoch errors (3D position error in metres)")
    print(f"    {'Epoch':>5s}  {'NLOS':>4s}  {'WLS':>8s}  {'PF':>8s}  "
          f"{'PF3D':>8s}  {'Best':>6s}")
    print(f"    {'-' * 50}")

    for i in range(N_EPOCHS):
        n_nlos = int(np.sum(~los_mask[i]))
        best = "PF3D" if pf3d_errors[i] <= min(wls_errors[i], pf_errors[i]) else \
               "PF" if pf_errors[i] <= wls_errors[i] else "WLS"
        print(f"    {i:5d}  {n_nlos:4d}  {wls_errors[i]:8.2f}  "
              f"{pf_errors[i]:8.2f}  {pf3d_errors[i]:8.2f}  {best:>6s}")

    # ------------------------------------------------------------------
    # Step 8: Summary statistics
    # ------------------------------------------------------------------
    print(f"\n[8] Summary statistics")
    print(f"    {'Method':<35s} {'Mean':>8s} {'Std':>8s} {'95th':>8s} {'Max':>8s}")
    print(f"    {'-' * 72}")

    for label, errs in [
        ("WLS (no building model)", wls_errors),
        ("PF (no building model)", pf_errors),
        ("PF3D (NLOS-aware, PLATEAU model)", pf3d_errors),
    ]:
        p95 = np.percentile(errs, 95)
        print(f"    {label:<35s} {np.mean(errs):8.2f} {np.std(errs):8.2f} "
              f"{p95:8.2f} {np.max(errs):8.2f}")

    # Improvement metrics
    pf3d_vs_wls = np.mean(wls_errors) - np.mean(pf3d_errors)
    pf3d_vs_pf = np.mean(pf_errors) - np.mean(pf3d_errors)
    print(f"\n    PF3D improvement over WLS: {pf3d_vs_wls:+.2f} m mean error")
    print(f"    PF3D improvement over PF : {pf3d_vs_pf:+.2f} m mean error")

    pf3d_wins = np.sum((pf3d_errors <= wls_errors) & (pf3d_errors <= pf_errors))
    print(f"    PF3D best in {pf3d_wins}/{N_EPOCHS} epochs "
          f"({100 * pf3d_wins / N_EPOCHS:.0f}%)")

    # ------------------------------------------------------------------
    # Step 9: VulnerabilityMap GeoJSON export
    # ------------------------------------------------------------------
    print(f"\n[9] Generating VulnerabilityMap GeoJSON")

    geojson_path = base_dir / "output"
    geojson_path.mkdir(exist_ok=True)
    geojson_file = geojson_path / "plateau_vulnerability_map.geojson"

    try:
        from gnss_gpu import VulnerabilityMap
        vmap = VulnerabilityMap(
            origin_lla=ORIGIN_LLA_DEG,
            grid_size_m=200, resolution_m=10, height_m=1.5,
        )
        quality = vmap.evaluate(SAT_ECEF, elevation_mask_deg=10.0)
        geojson = vmap.to_geojson(metric="hdop")
        vm_source = "GPU"
    except (ImportError, RuntimeError, Exception):
        # CPU fallback: build a simple GeoJSON grid
        print(f"    GPU VulnerabilityMap unavailable, using CPU fallback")
        half = 100.0
        step = 10.0
        ticks = np.arange(-half, half + step * 0.5, step)
        n_side = len(ticks)

        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2.0 * f - f * f
        lat0_rad = np.radians(ORIGIN_LLA_DEG[0])
        sin_lat0 = np.sin(lat0_rad)
        N0 = a / np.sqrt(1.0 - e2 * sin_lat0 ** 2)
        R_m = a * (1.0 - e2) / (1.0 - e2 * sin_lat0 ** 2) ** 1.5

        features = []
        half_res = step / 2.0
        dlat = np.degrees(half_res / R_m)
        dlon = np.degrees(half_res / (N0 * np.cos(lat0_rad)))

        for en in ticks:
            for ee in ticks:
                lat_deg = ORIGIN_LLA_DEG[0] + np.degrees(en / R_m)
                lon_deg = ORIGIN_LLA_DEG[1] + np.degrees(
                    ee / (N0 * np.cos(lat0_rad)))
                rx = _lla_to_ecef(np.radians(lat_deg), np.radians(lon_deg),
                                  ORIGIN_LLA_DEG[2] + 1.5)
                hdop, n_vis = _compute_dop_cpu(rx, SAT_ECEF)

                if hdop < 2.0:
                    color = "#00cc00"
                elif hdop < 4.0:
                    color = "#88cc00"
                elif hdop < 6.0:
                    color = "#cccc00"
                elif hdop < 10.0:
                    color = "#cc8800"
                else:
                    color = "#cc0000"

                coords = [[
                    [lon_deg - dlon, lat_deg - dlat],
                    [lon_deg + dlon, lat_deg - dlat],
                    [lon_deg + dlon, lat_deg + dlat],
                    [lon_deg - dlon, lat_deg + dlat],
                    [lon_deg - dlon, lat_deg - dlat],
                ]]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": coords},
                    "properties": {
                        "hdop": round(hdop, 3),
                        "n_visible": n_vis,
                        "fill": color,
                        "fill-opacity": 0.6,
                    },
                })

        geojson = {"type": "FeatureCollection", "features": features}
        vm_source = "CPU"

    with open(geojson_file, "w") as f:
        json.dump(geojson, f)

    print(f"    Source : {vm_source}")
    print(f"    Output : {geojson_file}")
    print(f"    Features: {len(geojson['features'])}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"  Experiment complete.")
    print(f"  Key finding: PF3D with PLATEAU 3D building model achieves")
    print(f"  {pf3d_vs_wls:+.1f} m improvement over WLS and "
          f"{pf3d_vs_pf:+.1f} m over standard PF")
    print(f"  in this urban canyon scenario with {n_nlos_total} NLOS observations.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
