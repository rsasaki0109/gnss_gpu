"""PLATEAU CityGML loader.

Loads 3D city model data published by Project PLATEAU (Japan) and converts
building geometry into triangle meshes suitable for GNSS ray tracing.

PLATEAU distributes LOD1/LOD2 building data in CityGML format using the
Japanese plane rectangular coordinate system (Gauss-Kruger transverse
Mercator, EPSG:6669-6687).
"""

import glob as _glob
import os
from pathlib import Path
from typing import Union

import numpy as np

from gnss_gpu.io.citygml import parse_citygml
from gnss_gpu.raytrace import BuildingModel

# WGS-84 ellipsoid constants
_A = 6378137.0
_F = 1.0 / 298.257223563
_B = _A * (1.0 - _F)
_E2 = 2.0 * _F - _F * _F

# GRS-80 ellipsoid constants (used by the Japanese geodetic system JGD2011,
# virtually identical to WGS-84 for our purposes)
_A_GRS80 = 6378137.0
_F_GRS80 = 1.0 / 298.257222101
_E2_GRS80 = 2.0 * _F_GRS80 - _F_GRS80 * _F_GRS80


class PlateauLoader:
    """Load PLATEAU CityGML data and produce :class:`BuildingModel` instances.

    Parameters
    ----------
    zone : int
        Japanese plane rectangular coordinate system zone number (1--19).
        Default is 9 (Tokyo / Kanagawa area).
    """

    # Japanese plane rectangular coordinate system zone origins.
    # (latitude_deg, longitude_deg) -- all zones use a 500 km false easting.
    ORIGINS = {
        1: (33.0, 129.5),
        2: (33.0, 131.0),
        3: (36.0, 132.16667),
        4: (33.0, 133.5),
        5: (36.0, 134.33333),
        6: (36.0, 136.0),
        7: (36.0, 137.16667),
        8: (36.0, 138.5),
        9: (36.0, 139.83333),
        10: (40.0, 140.83333),
        11: (44.0, 140.25),
        12: (44.0, 142.25),
        13: (44.0, 144.25),
        14: (26.0, 142.0),
        15: (26.0, 127.5),
        16: (26.0, 124.0),
        17: (26.0, 131.0),
        18: (20.0, 136.0),
        19: (26.0, 154.0),
    }

    _FALSE_EASTING = 0.0  # PLATEAU data is already in metres from origin

    def __init__(self, zone: int = 9):
        if zone not in self.ORIGINS:
            raise ValueError(
                f"zone must be 1-19, got {zone}"
            )
        self.zone = zone
        self._lat0_deg, self._lon0_deg = self.ORIGINS[zone]
        self._lat0 = np.radians(self._lat0_deg)
        self._lon0 = np.radians(self._lon0_deg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_citygml(self, filepath: Union[str, Path]) -> BuildingModel:
        """Load buildings from a single CityGML file.

        Parameters
        ----------
        filepath : str or Path
            Path to a ``.gml`` file.

        Returns
        -------
        BuildingModel
        """
        buildings = parse_citygml(filepath)
        triangles = self._buildings_to_triangles(buildings)
        return BuildingModel(triangles)

    def load_directory(
        self, dirpath: Union[str, Path], pattern: str = "*.gml"
    ) -> BuildingModel:
        """Load all CityGML files from a directory tree.

        Parameters
        ----------
        dirpath : str or Path
            Root directory to search.
        pattern : str
            Glob pattern for CityGML files.

        Returns
        -------
        BuildingModel
        """
        dirpath = Path(dirpath)
        files = sorted(dirpath.rglob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {dirpath}"
            )

        all_triangles = []
        for f in files:
            buildings = parse_citygml(f)
            tri = self._buildings_to_triangles(buildings)
            if tri.size > 0:
                all_triangles.append(tri)

        if not all_triangles:
            raise ValueError("No building geometry found in the provided files")

        combined = np.concatenate(all_triangles, axis=0)
        return BuildingModel(combined)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _buildings_to_triangles(self, buildings):
        """Convert parsed buildings to an ``(N, 3, 3)`` ECEF triangle array."""
        all_tris = []

        for bldg in buildings:
            for polygon in bldg.polygons:
                # polygon shape: (M, 3) -- coordinates in the Japanese plane
                # rectangular system (y_north, x_east, z_up) -- PLATEAU uses
                # the order (y, x, z) where y is northing and x is easting.
                # Some datasets use (x, y, z) with x=northing.  PLATEAU
                # convention: first coordinate = northing (Y), second = easting
                # (X), but internally the coordinate axes are labelled X=north,
                # Y=east following the Japanese survey convention.  We handle
                # this in _plane_rect_to_ecef.
                ecef_coords = self._polygon_to_ecef(polygon)
                tris = self._polygon_to_triangles(ecef_coords)
                if tris is not None:
                    all_tris.append(tris)

        if not all_tris:
            return np.empty((0, 3, 3), dtype=np.float64)

        return np.concatenate(all_tris, axis=0)

    def _polygon_to_ecef(self, coords):
        """Convert polygon coordinates from plane rectangular to ECEF.

        Parameters
        ----------
        coords : ndarray, shape (N, 3)
            Coordinates as stored in PLATEAU CityGML.  Most datasets use the
            Japanese plane rectangular system:
            - 1st value = Y (northing from origin latitude)
            - 2nd value = X (easting from origin longitude, with no false
              easting in PLATEAU data -- but some datasets add 500 km)
            - 3rd value = Z (ellipsoidal height in metres)

            Some published files instead store geographic coordinates directly
            as ``(lat_deg, lon_deg, h_m)``.  Those are detected heuristically
            from the coordinate ranges and converted without the plane-rect
            inverse projection.

        Returns
        -------
        ndarray, shape (N, 3)
            ECEF coordinates in metres.
        """
        n = coords.shape[0]
        ecef = np.empty((n, 3), dtype=np.float64)
        use_geodetic_degrees = self._looks_geodetic_degrees(coords)
        for i in range(n):
            if use_geodetic_degrees:
                ecef[i] = self._geodetic_degrees_to_ecef(
                    coords[i, 0], coords[i, 1], coords[i, 2]
                )
            else:
                ecef[i] = self._plane_rect_to_ecef(
                    coords[i, 0], coords[i, 1], coords[i, 2]
                )
        return ecef

    @staticmethod
    def _looks_geodetic_degrees(coords):
        """Return True when ``coords`` looks like (lat_deg, lon_deg, h_m)."""
        if coords.ndim != 2 or coords.shape[1] < 2:
            return False
        lat = coords[:, 0]
        lon = coords[:, 1]
        return bool(
            np.all(np.isfinite(lat))
            and np.all(np.isfinite(lon))
            and np.max(np.abs(lat)) <= 90.0
            and np.max(np.abs(lon)) <= 180.0
        )

    @staticmethod
    def _polygon_to_triangles(coords_3d):
        """Convert a 3-D polygon to triangles using fan triangulation.

        The first and last vertex are assumed to be the same (closed ring).
        If the polygon has fewer than 4 points (i.e. fewer than 3 unique
        vertices) no triangles are produced.

        Parameters
        ----------
        coords_3d : ndarray, shape (N, 3)

        Returns
        -------
        ndarray, shape (M, 3, 3) or None
        """
        # Remove the closing vertex if it duplicates the first
        if coords_3d.shape[0] >= 2 and np.allclose(
            coords_3d[0], coords_3d[-1]
        ):
            verts = coords_3d[:-1]
        else:
            verts = coords_3d

        n = verts.shape[0]
        if n < 3:
            return None

        # Fan triangulation from vertex 0
        tris = np.empty((n - 2, 3, 3), dtype=np.float64)
        for i in range(n - 2):
            tris[i, 0] = verts[0]
            tris[i, 1] = verts[i + 1]
            tris[i, 2] = verts[i + 2]
        return tris

    def _plane_rect_to_ecef(self, y_north, x_east, z_up):
        """Convert a single point from plane rectangular coords to ECEF.

        Parameters
        ----------
        y_north : float
            Northing in metres (1st coordinate in PLATEAU data).
        x_east : float
            Easting in metres (2nd coordinate in PLATEAU data).
        z_up : float
            Ellipsoidal height in metres.

        Returns
        -------
        ndarray, shape (3,)
            (X, Y, Z) in ECEF.
        """
        lat, lon = self._gauss_kruger_inverse(
            y_north, x_east, self._lat0, self._lon0
        )
        return self._lla_to_ecef(lat, lon, z_up)

    # ------------------------------------------------------------------
    # Gauss-Kruger inverse projection
    # ------------------------------------------------------------------

    @staticmethod
    def _gauss_kruger_inverse(y, x, lat0, lon0):
        """Inverse Gauss-Kruger (transverse Mercator) projection.

        Converts plane rectangular coordinates back to geodetic latitude and
        longitude using series expansion accurate to ~1 mm.

        Parameters
        ----------
        y : float
            Northing in metres from the origin latitude.
        x : float
            Easting in metres from the origin longitude (no false easting).
        lat0 : float
            Origin latitude in radians.
        lon0 : float
            Origin longitude in radians.

        Returns
        -------
        (lat, lon) : tuple of float
            Geodetic latitude and longitude in radians.
        """
        a = _A_GRS80
        e2 = _E2_GRS80

        # Flattening-derived constants
        e = np.sqrt(e2)
        e_prime2 = e2 / (1.0 - e2)

        # Meridian arc length coefficients (Bessel series)
        e2n = e2
        e4 = e2n * e2n
        e6 = e4 * e2n
        e8 = e6 * e2n

        A0 = 1.0 - e2n / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0 - 175.0 * e8 / 16384.0
        A2 = 3.0 / 8.0 * (e2n + e4 / 4.0 + 15.0 * e6 / 128.0 + 35.0 * e8 / 512.0)
        A4 = 15.0 / 256.0 * (e4 + 3.0 * e6 / 4.0 + 35.0 * e8 / 64.0)
        A6 = 35.0 / 3072.0 * (e6 + 5.0 * e8 / 4.0)
        A8 = 315.0 / 131072.0 * e8

        # Arc length from equator to lat0
        M0 = a * (
            A0 * lat0
            - A2 * np.sin(2.0 * lat0)
            + A4 * np.sin(4.0 * lat0)
            - A6 * np.sin(6.0 * lat0)
            + A8 * np.sin(8.0 * lat0)
        )

        M = M0 + y  # total meridian arc to the footpoint latitude

        # Footpoint latitude by Newton-Raphson
        mu = M / (a * A0)
        # Inverse series for footpoint latitude
        e1 = (1.0 - np.sqrt(1.0 - e2)) / (1.0 + np.sqrt(1.0 - e2))
        e1_2 = e1 * e1
        e1_3 = e1_2 * e1
        e1_4 = e1_3 * e1

        lat_fp = (
            mu
            + (3.0 * e1 / 2.0 - 27.0 * e1_3 / 32.0) * np.sin(2.0 * mu)
            + (21.0 * e1_2 / 16.0 - 55.0 * e1_4 / 32.0) * np.sin(4.0 * mu)
            + (151.0 * e1_3 / 96.0) * np.sin(6.0 * mu)
            + (1097.0 * e1_4 / 512.0) * np.sin(8.0 * mu)
        )

        sin_fp = np.sin(lat_fp)
        cos_fp = np.cos(lat_fp)
        tan_fp = np.tan(lat_fp)

        N_fp = a / np.sqrt(1.0 - e2 * sin_fp * sin_fp)
        T_fp = tan_fp * tan_fp
        C_fp = e_prime2 * cos_fp * cos_fp
        R_fp = a * (1.0 - e2) / (1.0 - e2 * sin_fp * sin_fp) ** 1.5

        D = x / N_fp

        D2 = D * D
        D3 = D2 * D
        D4 = D3 * D
        D5 = D4 * D
        D6 = D5 * D

        lat = lat_fp - (N_fp * tan_fp / R_fp) * (
            D2 / 2.0
            - (5.0 + 3.0 * T_fp + 10.0 * C_fp - 4.0 * C_fp * C_fp - 9.0 * e_prime2)
            * D4 / 24.0
            + (
                61.0
                + 90.0 * T_fp
                + 298.0 * C_fp
                + 45.0 * T_fp * T_fp
                - 252.0 * e_prime2
                - 3.0 * C_fp * C_fp
            )
            * D6 / 720.0
        )

        lon = lon0 + (
            D
            - (1.0 + 2.0 * T_fp + C_fp) * D3 / 6.0
            + (
                5.0
                - 2.0 * C_fp
                + 28.0 * T_fp
                - 3.0 * C_fp * C_fp
                + 8.0 * e_prime2
                + 24.0 * T_fp * T_fp
            )
            * D5 / 120.0
        ) / cos_fp

        return lat, lon

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lla_to_ecef(lat, lon, alt):
        """Convert geodetic (lat, lon in radians, alt in metres) to ECEF."""
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        N = _A / np.sqrt(1.0 - _E2 * sin_lat * sin_lat)
        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1.0 - _E2) + alt) * sin_lat
        return np.array([x, y, z], dtype=np.float64)

    @classmethod
    def _geodetic_degrees_to_ecef(cls, lat_deg, lon_deg, alt):
        """Convert geodetic degrees to ECEF."""
        return cls._lla_to_ecef(np.radians(lat_deg), np.radians(lon_deg), alt)


def load_plateau(filepath_or_dir, zone=9):
    """Convenience function to load PLATEAU CityGML data.

    Parameters
    ----------
    filepath_or_dir : str or Path
        Path to a single ``.gml`` file **or** a directory containing
        ``.gml`` files.
    zone : int
        Japanese plane rectangular coordinate system zone (1--19).
        Default is 9 (Tokyo / Kanagawa).

    Returns
    -------
    BuildingModel
    """
    p = Path(filepath_or_dir)
    loader = PlateauLoader(zone=zone)
    if p.is_dir():
        return loader.load_directory(p)
    else:
        return loader.load_citygml(p)
