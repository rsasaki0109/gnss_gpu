import numpy as np

try:
    from gnss_gpu._gnss_gpu_multipath import simulate_multipath as _simulate, \
        apply_multipath_error as _apply_error
except ImportError:
    _simulate = None
    _apply_error = None


class MultipathSimulator:
    """GPU-accelerated multipath signal simulation.

    Parameters
    ----------
    reflector_planes : array_like, shape (n_ref, 6)
        Each row is [point_x, point_y, point_z, normal_x, normal_y, normal_z].
    carrier_freq : float
        Carrier frequency in Hz (default: L1 = 1575.42 MHz).
    chip_rate : float
        Code chip rate in chips/s (default: C/A = 1.023 MHz).
    correlator_spacing : float
        Early-late correlator spacing in chips (default: 1.0).
    """

    def __init__(self, reflector_planes, carrier_freq=1575.42e6,
                 chip_rate=1.023e6, correlator_spacing=1.0):
        self.reflector_planes = np.ascontiguousarray(reflector_planes, dtype=np.float64)
        if self.reflector_planes.ndim == 1:
            self.reflector_planes = self.reflector_planes.reshape(1, 6)
        self.carrier_freq = float(carrier_freq)
        self.chip_rate = float(chip_rate)
        self.correlator_spacing = float(correlator_spacing)
        self.n_ref = self.reflector_planes.shape[0]

    def simulate(self, rx_ecef, sat_ecef):
        """Simulate multipath excess delays and attenuations.

        Parameters
        ----------
        rx_ecef : array_like, shape (n_rx, 3) or (3,)
            Receiver ECEF positions [m].
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].

        Returns
        -------
        delays : ndarray, shape (n_rx, n_sat)
            Excess path delays [m].
        attenuations : ndarray, shape (n_rx, n_sat)
            Composite attenuation factors.
        """
        rx = np.ascontiguousarray(rx_ecef, dtype=np.float64)
        if rx.ndim == 1:
            rx = rx.reshape(1, 3)
        sat = np.ascontiguousarray(sat_ecef, dtype=np.float64)
        n_rx = rx.shape[0]
        n_sat = sat.shape[0]

        if _simulate is None:
            raise RuntimeError("Multipath CUDA bindings not available")

        delays, attenuations = _simulate(
            rx, sat, self.reflector_planes,
            n_rx, n_sat, self.n_ref,
            self.carrier_freq, self.chip_rate)

        return delays.reshape(n_rx, n_sat), attenuations.reshape(n_rx, n_sat)

    def corrupt_pseudoranges(self, clean_pr, rx_ecef, sat_ecef):
        """Apply multipath DLL tracking error to clean pseudoranges.

        Parameters
        ----------
        clean_pr : array_like, shape (n_epoch, n_sat)
            Clean pseudoranges [m].
        rx_ecef : array_like, shape (n_epoch, 3)
            Receiver ECEF positions per epoch [m].
        sat_ecef : array_like, shape (n_epoch, n_sat, 3)
            Satellite ECEF positions per epoch [m].

        Returns
        -------
        corrupted_pr : ndarray, shape (n_epoch, n_sat)
            Corrupted pseudoranges [m].
        errors : ndarray, shape (n_epoch, n_sat)
            Multipath-induced errors [m].
        """
        pr = np.ascontiguousarray(clean_pr, dtype=np.float64)
        if pr.ndim == 1:
            pr = pr.reshape(1, -1)
        rx = np.ascontiguousarray(rx_ecef, dtype=np.float64)
        if rx.ndim == 1:
            rx = rx.reshape(1, 3)
        sat = np.ascontiguousarray(sat_ecef, dtype=np.float64)
        if sat.ndim == 2:
            sat = sat.reshape(1, *sat.shape)

        n_epoch = pr.shape[0]
        n_sat = pr.shape[1]

        if _apply_error is None:
            raise RuntimeError("Multipath CUDA bindings not available")

        corrupted, errors = _apply_error(
            pr, rx, sat.reshape(n_epoch * n_sat, 3),
            self.reflector_planes,
            n_epoch, n_sat, self.n_ref,
            self.carrier_freq, self.chip_rate, self.correlator_spacing)

        return corrupted.reshape(n_epoch, n_sat), errors.reshape(n_epoch, n_sat)

    @classmethod
    def from_building_boxes(cls, boxes_enu, origin_lla, **kwargs):
        """Create simulator from axis-aligned building boxes in ENU frame.

        Parameters
        ----------
        boxes_enu : list of tuple
            Each tuple is (center_e, center_n, width, depth, height).
            Width is along East, depth along North.
        origin_lla : tuple
            (lat_rad, lon_rad, alt_m) origin for ENU-to-ECEF conversion.

        Returns
        -------
        MultipathSimulator
            Simulator with 4 vertical reflector planes per building.
        """
        planes = []
        for (ce, cn, w, d, h) in boxes_enu:
            half_w = w / 2.0
            half_d = d / 2.0
            mid_h = h / 2.0

            # East face: point on +E side, normal pointing East
            planes.append([ce + half_w, cn, mid_h, 1.0, 0.0, 0.0])
            # West face: point on -E side, normal pointing West
            planes.append([ce - half_w, cn, mid_h, -1.0, 0.0, 0.0])
            # North face: point on +N side, normal pointing North
            planes.append([ce, cn + half_d, mid_h, 0.0, 1.0, 0.0])
            # South face: point on -N side, normal pointing South
            planes.append([ce, cn - half_d, mid_h, 0.0, -1.0, 0.0])

        planes = np.array(planes, dtype=np.float64)

        # Convert ENU plane points to ECEF
        # For simplicity, we store planes in ENU coordinates
        # The user should convert receiver/satellite positions to the same frame,
        # or use ECEF positions directly with ECEF-defined planes.
        # Here we perform a simple ENU->ECEF rotation for the plane definitions.
        lat, lon, alt = origin_lla
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        # ENU to ECEF rotation matrix
        R = np.array([
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat]
        ])

        # Origin ECEF (approximate)
        WGS84_A = 6378137.0
        WGS84_E2 = 6.69437999014e-3
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
        ox = (N + alt) * cos_lat * cos_lon
        oy = (N + alt) * cos_lat * sin_lon
        oz = (N * (1.0 - WGS84_E2) + alt) * sin_lat

        ecef_planes = np.zeros_like(planes)
        for i in range(len(planes)):
            enu_pt = planes[i, :3]
            enu_n = planes[i, 3:]
            ecef_planes[i, :3] = R @ enu_pt + np.array([ox, oy, oz])
            ecef_planes[i, 3:] = R @ enu_n  # rotate normal (no translation)

        return cls(ecef_planes, **kwargs)
