"""Geometry-aware urban GNSS IQ signal simulator.

Chains 3D city models, GPU ray-tracing, ephemeris, atmosphere, and
CUDA signal generation into a single pipeline:

    PLATEAU CityGML + User Trajectory + Broadcast Ephemeris
        -> GPU LOS/NLOS classification (BVH ray-tracing)
        -> GPU multipath excess delay computation
        -> Atmospheric delay (Saastamoinen + Klobuchar)
        -> Per-satellite signal parameters (code phase, Doppler, amplitude)
        -> CUDA IQ signal generation with multipath replica injection
        -> Output: urban GNSS IF samples with scene-driven LOS/NLOS effects
"""

import math

import numpy as np

from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.fresnel import reflection_coefficient
from gnss_gpu.diffraction import (
    compute_diffraction_paths,
    compute_diffraction_paths_gpu,
)
from gnss_gpu.double_reflection import compute_double_reflection_paths
from gnss_gpu.reflection_diffraction import compute_reflection_diffraction_paths
from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths

C_LIGHT = 299792458.0
GPS_L1_FREQ = 1575.42e6
GPS_L1_WAVELENGTH = C_LIGHT / GPS_L1_FREQ
CA_CHIP_RATE = 1.023e6


def ecef_to_lla(x, y, z):
    """Convert ECEF to geodetic (lat, lon, alt) in radians and meters."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * N * sin_lat, p)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else abs(z) - N * (1 - e2)
    return lat, lon, alt


def _sat_elevation_azimuth(rx_ecef, sat_ecef):
    """Compute elevation and azimuth from receiver to satellite."""
    rx = np.asarray(rx_ecef, dtype=np.float64)
    lat, lon, _ = ecef_to_lla(rx[0], rx[1], rx[2])
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    # ENU rotation matrix
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])

    diff = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3) - rx
    enu = (R @ diff.T).T  # [n_sat, 3]
    e, n, u = enu[:, 0], enu[:, 1], enu[:, 2]

    horiz = np.sqrt(e * e + n * n)
    el = np.arctan2(u, horiz)
    az = np.arctan2(e, n)
    return el, az


class UrbanSignalSimulator:
    """Urban GNSS IQ signal simulator with a 3D scene model."""

    def __init__(self, building_model=None, sampling_freq=2.6e6,
                 intermediate_freq=0.0, noise_floor_db=-20.0,
                 elevation_mask_deg=10.0,
                 nlos_attenuation_db=6.0, fresnel_coeff=0.5,
                 max_reflection_paths=0, reflector_material=None,
                 reflection_polarization="rhcp",
                 carrier_freq_hz=GPS_L1_FREQ,
                 ground_reflection=False, ground_height_m=0.0,
                 ground_material="dry_ground",
                 max_diffraction_paths=0, diffraction_edges=None,
                 diffraction_edge_kwargs=None, diffraction_path_kwargs=None,
                 diffraction_use_gpu=False,
                 diffraction_model="knife_edge", utd_mode="absorbing",
                 max_double_reflection_paths=0,
                 max_reflection_diffraction_paths=0,
                 reflection_diffraction_path_kwargs=None,
                 reflection_diffraction_orders=("RD", "DR")):
        """
        Args:
            building_model: BuildingModel or BVHAccelerator instance.
            sampling_freq: IQ sampling frequency [Hz].
            intermediate_freq: IF frequency [Hz].
            noise_floor_db: Noise floor [dB].
            elevation_mask_deg: Minimum satellite elevation [deg].
            nlos_attenuation_db: Signal attenuation for NLOS satellites [dB].
            fresnel_coeff: Reflection coefficient for multipath [0-1].
            max_reflection_paths: Maximum first-order reflection paths to add
                per satellite. 0 disables physical reflection paths and keeps
                legacy single multipath behavior.
            reflector_material: Material name, (eps_r, sigma) tuple, or complex
                permittivity for angle-dependent Fresnel reflection. None keeps
                legacy fixed fresnel_coeff behavior.
            reflection_polarization: Polarization mode for Fresnel reflection
                ("rhcp", "rhcp_cross", "parallel", "perpendicular", "average").
            carrier_freq_hz: Carrier frequency [Hz] for complex permittivity.
            max_diffraction_paths: Maximum knife-edge diffraction paths to add
                per satellite. 0 disables diffraction.
            diffraction_edges: Precomputed DiffractionEdgeSet (start/end/midpoint
                /size). If None and max_diffraction_paths>0, edges are lazily
                extracted from building_model.triangles via the experiments
                helper (best effort; degrades to no diffraction if unavailable).
            diffraction_edge_kwargs: kwargs for extract_diffraction_edges.
            diffraction_path_kwargs: kwargs for compute_diffraction_paths
                (e.g. max_ray_edge_distance_m, max_excess_path_m).
        """
        self.building_model = building_model
        self.sim = SignalSimulator(sampling_freq, intermediate_freq, noise_floor_db)
        self.elevation_mask_rad = math.radians(elevation_mask_deg)
        self.nlos_attenuation_db = nlos_attenuation_db
        self.fresnel_coeff = fresnel_coeff
        self.max_reflection_paths = int(max_reflection_paths)
        self.reflector_material = reflector_material
        self.reflection_polarization = reflection_polarization
        self.carrier_freq_hz = float(carrier_freq_hz)
        self.ground_reflection = bool(ground_reflection)
        self.ground_height_m = float(ground_height_m)
        self.ground_material = ground_material
        self.max_diffraction_paths = int(max_diffraction_paths)
        self.diffraction_use_gpu = bool(diffraction_use_gpu)
        self.diffraction_model = str(diffraction_model)
        self.utd_mode = str(utd_mode)
        self.diffraction_edge_kwargs = dict(diffraction_edge_kwargs or {})
        self.diffraction_path_kwargs = dict(diffraction_path_kwargs or {})
        self._diffraction_edges_cache = diffraction_edges
        self._diffraction_edges_resolved = diffraction_edges is not None
        self.max_double_reflection_paths = int(max_double_reflection_paths)
        self.max_reflection_diffraction_paths = int(max_reflection_diffraction_paths)
        self.reflection_diffraction_path_kwargs = dict(
            reflection_diffraction_path_kwargs or {})
        self.reflection_diffraction_orders = tuple(reflection_diffraction_orders)

    def _get_diffraction_edges(self):
        """Return DiffractionEdgeSet (precomputed or lazily extracted), or None."""
        if self._diffraction_edges_resolved:
            return self._diffraction_edges_cache
        self._diffraction_edges_resolved = True
        needs_edges = (
            self.max_diffraction_paths > 0
            or self.max_reflection_diffraction_paths > 0)
        if not needs_edges or self.building_model is None:
            self._diffraction_edges_cache = None
            return None
        tris = getattr(self.building_model, "triangles", None)
        if tris is None:
            self._diffraction_edges_cache = None
            return None
        try:
            import os
            import sys
            exp_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                "experiments",
            )
            if exp_dir not in sys.path:
                sys.path.insert(0, exp_dir)
            from utd_edge_features import extract_diffraction_edges
            self._diffraction_edges_cache = extract_diffraction_edges(
                tris, **self.diffraction_edge_kwargs)
        except Exception:
            self._diffraction_edges_cache = None
        return self._diffraction_edges_cache

    def compute_epoch(self, rx_ecef, sat_ecef, sat_clk=None, sat_vel=None,
                      rx_vel=None, rx_clock_bias=0.0,
                      prn_list=None, gps_time=0.0,
                      atmo_correction=None, iono_params=None,
                      n_samples=None):
        """Generate IQ signal for one epoch with urban environment effects.

        Args:
            rx_ecef: [3] receiver ECEF position [m].
            sat_ecef: [n_sat, 3] satellite ECEF positions [m].
            sat_clk: [n_sat] satellite clock corrections [s].
            sat_vel: [n_sat, 3] satellite ECEF velocities [m/s] (for Doppler).
            rx_vel: [3] receiver velocity [m/s] (for Doppler).
            rx_clock_bias: Receiver clock bias [m].
            prn_list: List of PRN numbers (1-32). Defaults to range(1, n_sat+1).
            gps_time: GPS time of week [s] (for ionosphere model).
            atmo_correction: AtmosphereCorrection instance (optional).
            iono_params: dict with 'alpha' and 'beta' arrays (Klobuchar).
            n_samples: Number of IQ samples (default: 1ms).

        Returns:
            dict with:
                'iq': float32 array [2*n_samples] interleaved I/Q
                'channels': list of per-satellite parameter dicts
                'is_los': boolean array
                'excess_delays': float array [m]
                'elevations': float array [rad]
        """
        rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
        sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        n_sat = sats.shape[0]
        sat_clock_m = np.zeros(n_sat, dtype=np.float64)

        if sat_clk is not None:
            sat_clk_arr = np.asarray(sat_clk, dtype=np.float64).reshape(-1)
            if sat_clk_arr.size != n_sat:
                raise ValueError("sat_clk must have one entry per satellite")
            sat_clock_m = sat_clk_arr * C_LIGHT

        if prn_list is None:
            prn_list = list(range(1, n_sat + 1))
        if n_samples is None:
            n_samples = int(self.sim.sampling_freq * 1e-3)

        # --- Elevation / azimuth ---
        el, az = _sat_elevation_azimuth(rx, sats)

        # --- Elevation mask ---
        visible = el >= self.elevation_mask_rad

        # --- LOS / NLOS classification ---
        is_los = np.ones(n_sat, dtype=bool)
        excess_delays = np.zeros(n_sat, dtype=np.float64)
        reflection_paths = [[] for _ in range(n_sat)]
        diffraction_paths = [[] for _ in range(n_sat)]
        double_reflection_paths = [[] for _ in range(n_sat)]
        reflection_diffraction_paths = [[] for _ in range(n_sat)]
        use_reflection_paths = (
            self.max_reflection_paths > 0
            and self.building_model is not None
            and hasattr(self.building_model, 'compute_reflection_paths')
        )
        use_diffraction_paths = self.max_diffraction_paths > 0
        use_double_reflection_paths = (
            self.max_double_reflection_paths > 0
            and self.building_model is not None
            and getattr(self.building_model, "triangles", None) is not None
        )
        use_reflection_diffraction_paths = (
            self.max_reflection_diffraction_paths > 0
            and self.building_model is not None
            and getattr(self.building_model, "triangles", None) is not None
        )
        vis_idx = np.where(visible)[0]

        if self.building_model is not None:
            if len(vis_idx) > 0:
                los_result = self.building_model.check_los(rx, sats[vis_idx])
                is_los_vis = np.asarray(los_result, dtype=bool)
                is_los[vis_idx] = is_los_vis

                if use_reflection_paths:
                    ground_plane = None
                    if self.ground_reflection:
                        lat, lon, _ = ecef_to_lla(rx[0], rx[1], rx[2])
                        up = np.array([
                            math.cos(lat) * math.cos(lon),
                            math.cos(lat) * math.sin(lon),
                            math.sin(lat),
                        ], dtype=np.float64)
                        up_len = float(np.linalg.norm(up))
                        if up_len > 0.0:
                            up = up / up_len
                        ground_point = np.asarray(rx, dtype=np.float64) - up * self.ground_height_m
                        ground_plane = (ground_point, up)

                    if ground_plane is not None:
                        try:
                            paths_per_vis = self.building_model.compute_reflection_paths(
                                rx, sats[vis_idx], max_paths=self.max_reflection_paths,
                                ground_plane=ground_plane)
                        except TypeError:
                            paths_per_vis = self.building_model.compute_reflection_paths(
                                rx, sats[vis_idx], max_paths=self.max_reflection_paths)
                    else:
                        paths_per_vis = self.building_model.compute_reflection_paths(
                            rx, sats[vis_idx], max_paths=self.max_reflection_paths)
                    for sat_idx, paths in zip(vis_idx, paths_per_vis):
                        sat_paths = [] if paths is None else list(paths)
                        reflection_paths[sat_idx] = sat_paths
                        if sat_paths:
                            excess_delays[sat_idx] = min(
                                float(path.excess_delay) for path in sat_paths)
                # Multipath excess delay (if supported by the model)
                elif hasattr(self.building_model, 'compute_multipath'):
                    delays, _ = self.building_model.compute_multipath(rx, sats[vis_idx])
                    excess_delays[vis_idx] = np.asarray(delays, dtype=np.float64)

        # --- Knife-edge diffraction paths ---
        if use_diffraction_paths and len(vis_idx) > 0:
            edges = self._get_diffraction_edges()
            if edges is not None and int(getattr(edges, "size", 0) or 0) > 0:
                if self.diffraction_model == "utd":
                    dpaths_per_vis = compute_utd_diffraction_paths(
                        rx, sats[vis_idx], edges,
                        max_paths=self.max_diffraction_paths,
                        mode=self.utd_mode,
                        **self.diffraction_path_kwargs)
                else:
                    diffraction_fn = (
                        compute_diffraction_paths_gpu
                        if self.diffraction_use_gpu
                        else compute_diffraction_paths
                    )
                    dpaths_per_vis = diffraction_fn(
                        rx, sats[vis_idx], edges,
                        max_paths=self.max_diffraction_paths,
                        **self.diffraction_path_kwargs)
                for sat_idx, dpaths in zip(vis_idx, dpaths_per_vis):
                    diffraction_paths[sat_idx] = [] if dpaths is None else list(dpaths)

        # --- Second-order (double-bounce) reflection paths ---
        if use_double_reflection_paths and len(vis_idx) > 0:
            tris = np.asarray(self.building_model.triangles, dtype=np.float64)
            if tris.size > 0:
                dbl_per_vis = compute_double_reflection_paths(
                    tris, rx, sats[vis_idx],
                    max_paths=self.max_double_reflection_paths)
                for sat_idx, dpaths in zip(vis_idx, dbl_per_vis):
                    double_reflection_paths[sat_idx] = [] if dpaths is None else list(dpaths)

        # --- Reflection+diffraction composite paths (rx->reflect->diffract->sat
        #     and rx->diffract->reflect->sat) ---
        if use_reflection_diffraction_paths and len(vis_idx) > 0:
            tris = np.asarray(self.building_model.triangles, dtype=np.float64)
            edges = self._get_diffraction_edges()
            if tris.size > 0 and edges is not None and int(getattr(edges, "size", 0) or 0) > 0:
                rd_per_vis = compute_reflection_diffraction_paths(
                    tris, edges, rx, sats[vis_idx],
                    max_paths=self.max_reflection_diffraction_paths,
                    orders=self.reflection_diffraction_orders,
                    **self.reflection_diffraction_path_kwargs)
                for sat_idx, dpaths in zip(vis_idx, rd_per_vis):
                    reflection_diffraction_paths[sat_idx] = (
                        [] if dpaths is None else list(dpaths))

        # --- Geometric range + atmospheric delays ---
        ranges = np.linalg.norm(sats - rx, axis=1)
        atmo_delay = np.zeros(n_sat, dtype=np.float64)

        if atmo_correction is not None:
            lat, lon, alt = ecef_to_lla(rx[0], rx[1], rx[2])
            rx_lla = np.array([lat, lon, alt])
            for i in range(n_sat):
                if not visible[i]:
                    continue
                tropo = atmo_correction.tropo(rx_lla, el[i])
                atmo_delay[i] = float(tropo)
                if iono_params is not None:
                    iono = atmo_correction.iono(
                        rx_lla, az[i], el[i], gps_time,
                        alpha=iono_params.get('alpha'),
                        beta=iono_params.get('beta'))
                    atmo_delay[i] += float(iono)

        # --- Doppler ---
        doppler = np.zeros(n_sat, dtype=np.float64)
        if sat_vel is not None:
            sv = np.asarray(sat_vel, dtype=np.float64).reshape(-1, 3)
            rv = np.zeros(3) if rx_vel is None else np.asarray(rx_vel, dtype=np.float64)
            for i in range(n_sat):
                if not visible[i]:
                    continue
                los_vec = (sats[i] - rx) / ranges[i]
                rel_vel = np.dot(sv[i] - rv, los_vec)
                doppler[i] = -rel_vel / GPS_L1_WAVELENGTH

        # --- Build per-satellite channel parameters ---
        channels = []
        n_reflection_paths = 0
        n_diffraction_paths = 0
        n_double_reflection_paths = 0
        n_reflection_diffraction_paths = 0
        for i in range(n_sat):
            if not visible[i]:
                continue

            # Pseudorange
            pr = ranges[i] + rx_clock_bias - sat_clock_m[i] + atmo_delay[i]

            # Code phase (chips into the current C/A code period)
            code_chips = (pr / C_LIGHT) * CA_CHIP_RATE
            code_phase = code_chips % 1023.0

            # Carrier phase
            carrier_phase = (pr / GPS_L1_WAVELENGTH) * 2.0 * math.pi
            carrier_phase = carrier_phase % (2.0 * math.pi)

            # Amplitude: LOS=1.0, NLOS=attenuated
            amplitude = 1.0
            if not is_los[i]:
                amplitude = 10.0 ** (-self.nlos_attenuation_db / 20.0)

            ch = {
                "prn": int(prn_list[i]),
                "code_phase": float(code_phase),
                "carrier_phase": float(carrier_phase),
                "doppler_hz": float(doppler[i]),
                "amplitude": float(amplitude),
                "nav_bit": 1,
            }
            channels.append(ch)

            if use_reflection_paths:
                # One replica per physical first-order reflection path.
                for path in reflection_paths[i]:
                    mp_pr = pr + float(path.excess_delay)
                    mp_code_phase = ((mp_pr / C_LIGHT) * CA_CHIP_RATE) % 1023.0
                    mp_carrier_phase = (mp_pr / GPS_L1_WAVELENGTH) * 2.0 * math.pi
                    mp_carrier_phase = mp_carrier_phase % (2.0 * math.pi)
                    if self.reflector_material is not None:
                        mat = (
                            self.ground_material
                            if getattr(path, "triangle_id", 0) == -1
                            else self.reflector_material
                        )
                        coeff = reflection_coefficient(
                            path.incidence_angle,
                            mat,
                            freq_hz=self.carrier_freq_hz,
                            polarization=self.reflection_polarization,
                        )
                        mp_amplitude = amplitude * float(coeff)
                    else:
                        mp_amplitude = amplitude * self.fresnel_coeff

                    mp_ch = {
                        "prn": int(prn_list[i]),
                        "code_phase": float(mp_code_phase),
                        "carrier_phase": float(mp_carrier_phase),
                        "doppler_hz": float(doppler[i]),
                        "amplitude": float(mp_amplitude),
                        "nav_bit": 1,
                    }
                    channels.append(mp_ch)
                    n_reflection_paths += 1
            else:
                # Add multipath replica (delayed + attenuated copy)
                if excess_delays[i] > 0.1:  # >0.1m excess delay
                    mp_pr = pr + excess_delays[i]
                    mp_code_phase = ((mp_pr / C_LIGHT) * CA_CHIP_RATE) % 1023.0
                    mp_carrier_phase = (mp_pr / GPS_L1_WAVELENGTH) * 2.0 * math.pi
                    mp_carrier_phase = mp_carrier_phase % (2.0 * math.pi)
                    mp_amplitude = amplitude * self.fresnel_coeff

                    mp_ch = {
                        "prn": int(prn_list[i]),
                        "code_phase": float(mp_code_phase),
                        "carrier_phase": float(mp_carrier_phase),
                        "doppler_hz": float(doppler[i]),
                        "amplitude": float(mp_amplitude),
                        "nav_bit": 1,
                    }
                    channels.append(mp_ch)

            # One replica per knife-edge diffraction path.
            if use_diffraction_paths:
                for path in diffraction_paths[i]:
                    d_pr = pr + float(path.excess_delay)
                    d_code_phase = ((d_pr / C_LIGHT) * CA_CHIP_RATE) % 1023.0
                    d_carrier_phase = (d_pr / GPS_L1_WAVELENGTH) * 2.0 * math.pi
                    d_carrier_phase = d_carrier_phase % (2.0 * math.pi)
                    d_amplitude = amplitude * float(path.amplitude)

                    d_ch = {
                        "prn": int(prn_list[i]),
                        "code_phase": float(d_code_phase),
                        "carrier_phase": float(d_carrier_phase),
                        "doppler_hz": float(doppler[i]),
                        "amplitude": float(d_amplitude),
                        "nav_bit": 1,
                    }
                    channels.append(d_ch)
                    n_diffraction_paths += 1

            # One replica per second-order (double-bounce) reflection path.
            if use_double_reflection_paths:
                for path in double_reflection_paths[i]:
                    dr_pr = pr + float(path.excess_delay)
                    dr_code_phase = ((dr_pr / C_LIGHT) * CA_CHIP_RATE) % 1023.0
                    dr_carrier_phase = (dr_pr / GPS_L1_WAVELENGTH) * 2.0 * math.pi
                    dr_carrier_phase = dr_carrier_phase % (2.0 * math.pi)
                    if self.reflector_material is not None:
                        inc1, inc2 = path.incidence_angles
                        coeff1 = reflection_coefficient(
                            inc1, self.reflector_material,
                            freq_hz=self.carrier_freq_hz,
                            polarization=self.reflection_polarization)
                        coeff2 = reflection_coefficient(
                            inc2, self.reflector_material,
                            freq_hz=self.carrier_freq_hz,
                            polarization=self.reflection_polarization)
                        dr_amplitude = amplitude * float(coeff1) * float(coeff2)
                    else:
                        # Two specular bounces -> coefficient squared.
                        dr_amplitude = amplitude * self.fresnel_coeff * self.fresnel_coeff

                    dr_ch = {
                        "prn": int(prn_list[i]),
                        "code_phase": float(dr_code_phase),
                        "carrier_phase": float(dr_carrier_phase),
                        "doppler_hz": float(doppler[i]),
                        "amplitude": float(dr_amplitude),
                        "nav_bit": 1,
                    }
                    channels.append(dr_ch)
                    n_double_reflection_paths += 1

            # One replica per reflection+diffraction composite path. Amplitude is
            # the product of the single Fresnel reflection coefficient and the
            # knife-edge diffraction amplitude.
            if use_reflection_diffraction_paths:
                for path in reflection_diffraction_paths[i]:
                    rd_pr = pr + float(path.excess_delay)
                    rd_code_phase = ((rd_pr / C_LIGHT) * CA_CHIP_RATE) % 1023.0
                    rd_carrier_phase = (rd_pr / GPS_L1_WAVELENGTH) * 2.0 * math.pi
                    rd_carrier_phase = rd_carrier_phase % (2.0 * math.pi)
                    if self.reflector_material is not None:
                        coeff = reflection_coefficient(
                            path.incidence_angle, self.reflector_material,
                            freq_hz=self.carrier_freq_hz,
                            polarization=self.reflection_polarization)
                        rd_amplitude = (
                            amplitude * float(coeff) * float(path.amplitude))
                    else:
                        rd_amplitude = (
                            amplitude * self.fresnel_coeff * float(path.amplitude))

                    rd_ch = {
                        "prn": int(prn_list[i]),
                        "code_phase": float(rd_code_phase),
                        "carrier_phase": float(rd_carrier_phase),
                        "doppler_hz": float(doppler[i]),
                        "amplitude": float(rd_amplitude),
                        "nav_bit": 1,
                    }
                    channels.append(rd_ch)
                    n_reflection_diffraction_paths += 1

        # --- Generate IQ signal ---
        iq = self.sim.generate_epoch(channels, n_samples=n_samples)

        return {
            "iq": iq,
            "channels": channels,
            "is_los": is_los,
            "excess_delays": excess_delays,
            "elevations": el,
            "azimuths": az,
            "visible": visible,
            "n_los": int(np.sum(is_los & visible)),
            "n_nlos": int(np.sum(~is_los & visible)),
            "n_multipath": int(np.sum(excess_delays > 0.1)),
            "reflection_paths": reflection_paths,
            "n_reflection_paths": int(n_reflection_paths),
            "diffraction_paths": diffraction_paths,
            "n_diffraction_paths": int(n_diffraction_paths),
            "double_reflection_paths": double_reflection_paths,
            "n_double_reflection_paths": int(n_double_reflection_paths),
            "reflection_diffraction_paths": reflection_diffraction_paths,
            "n_reflection_diffraction_paths": int(n_reflection_diffraction_paths),
        }

    def simulate_trajectory(self, rx_positions, sat_ecef_per_epoch,
                            prn_list=None, sat_clk_per_epoch=None, sat_vel_per_epoch=None,
                            rx_vel_per_epoch=None, gps_times=None,
                            atmo_correction=None, iono_params=None,
                            n_samples=None):
        """Generate IQ signal for a trajectory of epochs.

        Args:
            rx_positions: [n_epoch, 3] receiver ECEF positions.
            sat_ecef_per_epoch: [n_epoch, n_sat, 3] or callable(epoch_idx)->array.
            prn_list: PRN list (shared across epochs).
            sat_clk_per_epoch: [n_epoch, n_sat] or callable(epoch_idx)->array.
            sat_vel_per_epoch: [n_epoch, n_sat, 3] or None.
            rx_vel_per_epoch: [n_epoch, 3] or None.
            gps_times: [n_epoch] GPS times.
            atmo_correction: AtmosphereCorrection instance.
            iono_params: dict with 'alpha'/'beta'.
            n_samples: Samples per epoch.

        Yields:
            (epoch_idx, result_dict) for each epoch.
        """
        rx_pos = np.asarray(rx_positions, dtype=np.float64)
        n_epoch = rx_pos.shape[0]

        for i in range(n_epoch):
            if callable(sat_ecef_per_epoch):
                sat_ecef = sat_ecef_per_epoch(i)
            else:
                sat_ecef = sat_ecef_per_epoch[i]

            sat_clk = None
            if sat_clk_per_epoch is not None:
                sat_clk = sat_clk_per_epoch[i] if not callable(sat_clk_per_epoch) else sat_clk_per_epoch(i)

            sat_vel = None
            if sat_vel_per_epoch is not None:
                sat_vel = sat_vel_per_epoch[i] if not callable(sat_vel_per_epoch) else sat_vel_per_epoch(i)

            rx_vel = None
            if rx_vel_per_epoch is not None:
                rx_vel = rx_vel_per_epoch[i]

            gps_time = gps_times[i] if gps_times is not None else 0.0

            result = self.compute_epoch(
                rx_ecef=rx_pos[i],
                sat_ecef=sat_ecef,
                sat_clk=sat_clk,
                sat_vel=sat_vel,
                rx_vel=rx_vel,
                prn_list=prn_list,
                gps_time=gps_time,
                atmo_correction=atmo_correction,
                iono_params=iono_params,
                n_samples=n_samples,
            )
            yield i, result
