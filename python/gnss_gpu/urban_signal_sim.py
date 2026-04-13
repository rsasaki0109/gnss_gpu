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
                 nlos_attenuation_db=6.0, fresnel_coeff=0.5):
        """
        Args:
            building_model: BuildingModel or BVHAccelerator instance.
            sampling_freq: IQ sampling frequency [Hz].
            intermediate_freq: IF frequency [Hz].
            noise_floor_db: Noise floor [dB].
            elevation_mask_deg: Minimum satellite elevation [deg].
            nlos_attenuation_db: Signal attenuation for NLOS satellites [dB].
            fresnel_coeff: Reflection coefficient for multipath [0-1].
        """
        self.building_model = building_model
        self.sim = SignalSimulator(sampling_freq, intermediate_freq, noise_floor_db)
        self.elevation_mask_rad = math.radians(elevation_mask_deg)
        self.nlos_attenuation_db = nlos_attenuation_db
        self.fresnel_coeff = fresnel_coeff

    def compute_epoch(self, rx_ecef, sat_ecef, sat_vel=None,
                      rx_vel=None, rx_clock_bias=0.0,
                      prn_list=None, gps_time=0.0,
                      atmo_correction=None, iono_params=None,
                      n_samples=None):
        """Generate IQ signal for one epoch with urban environment effects.

        Args:
            rx_ecef: [3] receiver ECEF position [m].
            sat_ecef: [n_sat, 3] satellite ECEF positions [m].
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

        if self.building_model is not None:
            vis_idx = np.where(visible)[0]
            if len(vis_idx) > 0:
                los_result = self.building_model.check_los(rx, sats[vis_idx])
                is_los_vis = np.asarray(los_result, dtype=bool)
                is_los[vis_idx] = is_los_vis

                # Multipath excess delay (if supported by the model)
                if hasattr(self.building_model, 'compute_multipath'):
                    delays, _ = self.building_model.compute_multipath(rx, sats[vis_idx])
                    excess_delays[vis_idx] = np.asarray(delays, dtype=np.float64)

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
        for i in range(n_sat):
            if not visible[i]:
                continue

            # Pseudorange
            pr = ranges[i] + rx_clock_bias + atmo_delay[i]

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
        }

    def simulate_trajectory(self, rx_positions, sat_ecef_per_epoch,
                            prn_list=None, sat_vel_per_epoch=None,
                            rx_vel_per_epoch=None, gps_times=None,
                            atmo_correction=None, iono_params=None,
                            n_samples=None):
        """Generate IQ signal for a trajectory of epochs.

        Args:
            rx_positions: [n_epoch, 3] receiver ECEF positions.
            sat_ecef_per_epoch: [n_epoch, n_sat, 3] or callable(epoch_idx)->array.
            prn_list: PRN list (shared across epochs).
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
                sat_vel=sat_vel,
                rx_vel=rx_vel,
                prn_list=prn_list,
                gps_time=gps_time,
                atmo_correction=atmo_correction,
                iono_params=iono_params,
                n_samples=n_samples,
            )
            yield i, result
