"""GPU-accelerated GNSS vulnerability / quality map."""

import json
import numpy as np

try:
    from gnss_gpu._gnss_gpu_skyplot import compute_grid_quality as _compute_grid_quality
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def _lla_to_ecef_py(lat, lon, alt):
    """Pure-Python LLA (radians) to ECEF conversion for grid generation."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + alt) * sin_lat
    return x, y, z


def _ecef_to_lla_py(x, y, z):
    """Pure-Python ECEF to LLA (radians) conversion."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = 2.0 * f - f * f
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + e2 / (1.0 - e2) * b * np.sin(theta) ** 3,
                     p - e2 * a * np.cos(theta) ** 3)
    lon = np.arctan2(y, x)
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
    alt = p / np.cos(lat) - N
    return lat, lon, alt


class VulnerabilityMap:
    """GNSS vulnerability / quality map over a local grid.

    Parameters
    ----------
    origin_lla : tuple of float
        (latitude_deg, longitude_deg, altitude_m) of the grid origin.
    grid_size_m : float
        Full side length of the square grid in metres.
    resolution_m : float
        Grid cell spacing in metres.
    height_m : float
        Receiver height above the ground surface in metres.
    """

    def __init__(self, origin_lla, grid_size_m=500, resolution_m=5, height_m=1.5):
        self.origin_lla = origin_lla
        self.grid_size_m = grid_size_m
        self.resolution_m = resolution_m
        self.height_m = height_m

        lat0 = np.radians(origin_lla[0])
        lon0 = np.radians(origin_lla[1])
        alt0 = origin_lla[2] + height_m

        half = grid_size_m / 2.0
        ticks = np.arange(-half, half + resolution_m * 0.5, resolution_m)
        self.n_side = len(ticks)

        ee, nn = np.meshgrid(ticks, ticks)  # east, north
        ee_flat = ee.ravel()
        nn_flat = nn.ravel()

        # ENU offset -> LLA -> ECEF
        # Approximate: dLat ~ dN / R_N,  dLon ~ dE / (R_N * cos(lat))
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2.0 * f - f * f
        sin_lat0 = np.sin(lat0)
        N0 = a / np.sqrt(1.0 - e2 * sin_lat0 ** 2)
        R_m = a * (1.0 - e2) / (1.0 - e2 * sin_lat0 ** 2) ** 1.5  # meridional radius

        lat_arr = lat0 + nn_flat / R_m
        lon_arr = lon0 + ee_flat / (N0 * np.cos(lat0))
        alt_arr = np.full_like(lat_arr, alt0)

        x, y, z = _lla_to_ecef_py(lat_arr, lon_arr, alt_arr)
        self.grid_ecef = np.column_stack([x, y, z]).astype(np.float64)
        self.n_grid = len(x)

        # Store LLA for GeoJSON export
        self._grid_lat_deg = np.degrees(lat_arr).reshape(self.n_side, self.n_side)
        self._grid_lon_deg = np.degrees(lon_arr).reshape(self.n_side, self.n_side)

        # Placeholders for results
        self._pdop = None
        self._hdop = None
        self._vdop = None
        self._gdop = None
        self._n_visible = None

    def evaluate(self, sat_ecef, elevation_mask_deg=10.0):
        """Evaluate DOP over the grid.

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions in metres.
        elevation_mask_deg : float
            Minimum elevation angle in degrees.

        Returns
        -------
        dict
            Keys: 'pdop', 'hdop', 'vdop', 'gdop', 'n_visible', each a 2-D
            array of shape (n_side, n_side).
        """
        if not HAS_GPU:
            raise RuntimeError("CUDA skyplot module (_gnss_gpu_skyplot) not available")

        sat_ecef = np.ascontiguousarray(sat_ecef, dtype=np.float64)
        n_sat = sat_ecef.shape[0]
        el_mask_rad = np.radians(elevation_mask_deg)

        grid_flat = np.ascontiguousarray(self.grid_ecef, dtype=np.float64)

        pdop, hdop, vdop, gdop, n_visible = _compute_grid_quality(
            grid_flat, sat_ecef, self.n_grid, n_sat, el_mask_rad
        )

        ns = self.n_side
        self._pdop = pdop.reshape(ns, ns)
        self._hdop = hdop.reshape(ns, ns)
        self._vdop = vdop.reshape(ns, ns)
        self._gdop = gdop.reshape(ns, ns)
        self._n_visible = n_visible.reshape(ns, ns)

        return {
            "pdop": self._pdop,
            "hdop": self._hdop,
            "vdop": self._vdop,
            "gdop": self._gdop,
            "n_visible": self._n_visible,
        }

    def to_geojson(self, metric="hdop"):
        """Export the evaluated grid as a GeoJSON FeatureCollection.

        Parameters
        ----------
        metric : str
            One of 'pdop', 'hdop', 'vdop', 'gdop', 'n_visible'.

        Returns
        -------
        dict
            GeoJSON FeatureCollection with coloured grid cells.
        """
        data_map = {
            "pdop": self._pdop,
            "hdop": self._hdop,
            "vdop": self._vdop,
            "gdop": self._gdop,
            "n_visible": self._n_visible,
        }
        data = data_map.get(metric)
        if data is None:
            raise ValueError(f"No data for metric '{metric}'. Call evaluate() first.")

        def _dop_color(val):
            """Simple green-yellow-red colour ramp for DOP values."""
            if val < 2.0:
                return "#00cc00"
            elif val < 4.0:
                return "#88cc00"
            elif val < 6.0:
                return "#cccc00"
            elif val < 10.0:
                return "#cc8800"
            else:
                return "#cc0000"

        half_res = self.resolution_m / 2.0
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2.0 * f - f * f
        lat0 = np.radians(self.origin_lla[0])
        sin_lat0 = np.sin(lat0)
        N0 = a / np.sqrt(1.0 - e2 * sin_lat0 ** 2)
        R_m = a * (1.0 - e2) / (1.0 - e2 * sin_lat0 ** 2) ** 1.5

        dlat = np.degrees(half_res / R_m)
        dlon = np.degrees(half_res / (N0 * np.cos(lat0)))

        features = []
        ns = self.n_side
        for r in range(ns):
            for c in range(ns):
                lat_c = self._grid_lat_deg[r, c]
                lon_c = self._grid_lon_deg[r, c]
                val = float(data[r, c])
                color = _dop_color(val) if metric != "n_visible" else "#0066cc"

                coords = [[
                    [lon_c - dlon, lat_c - dlat],
                    [lon_c + dlon, lat_c - dlat],
                    [lon_c + dlon, lat_c + dlat],
                    [lon_c - dlon, lat_c + dlat],
                    [lon_c - dlon, lat_c - dlat],
                ]]

                feature = {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": coords},
                    "properties": {
                        metric: round(val, 3),
                        "fill": color,
                        "fill-opacity": 0.6,
                    },
                }
                features.append(feature)

        return {"type": "FeatureCollection", "features": features}
