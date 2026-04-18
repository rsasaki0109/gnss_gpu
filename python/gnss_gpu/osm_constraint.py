"""OpenStreetMap road-centerline utilities for soft particle constraints.

This module intentionally keeps map matching soft. It only computes distances
to OSM road centerlines and Huber log-likelihood penalties; it never projects a
state onto a road or rejects particles outright.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

WGS84_A_M = 6378137.0
WGS84_E2 = 6.69437999014e-3
DEFAULT_LIMITED_ACCESS_SIGMA_SCALE = 3.0

_NON_DRIVABLE_HIGHWAYS = {
    "bridleway",
    "construction",
    "corridor",
    "cycleway",
    "elevator",
    "footway",
    "path",
    "pedestrian",
    "platform",
    "proposed",
    "raceway",
    "steps",
    "track",
}
_LIMITED_ACCESS_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
}


@dataclass(frozen=True)
class RoadSegmentArrays:
    """Road segment arrays in a local ENU frame."""

    start_enu_m: np.ndarray
    end_enu_m: np.ndarray
    sigma_scales: np.ndarray
    highway_types: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_enu_m", np.asarray(self.start_enu_m, dtype=np.float64).reshape(-1, 2))
        object.__setattr__(self, "end_enu_m", np.asarray(self.end_enu_m, dtype=np.float64).reshape(-1, 2))
        object.__setattr__(self, "sigma_scales", np.asarray(self.sigma_scales, dtype=np.float64).ravel())
        n = self.start_enu_m.shape[0]
        if self.end_enu_m.shape != (n, 2):
            raise ValueError("end_enu_m must have shape (N, 2)")
        if self.sigma_scales.shape != (n,):
            raise ValueError("sigma_scales must have shape (N,)")
        if len(self.highway_types) != n:
            raise ValueError("highway_types length must match segment count")

    @property
    def n_segments(self) -> int:
        return int(self.start_enu_m.shape[0])

    def as_kernel_array(self) -> np.ndarray:
        """Return ``[x1, y1, x2, y2, sigma_scale]`` rows for GPU kernels."""

        return np.column_stack((self.start_enu_m, self.end_enu_m, self.sigma_scales)).astype(
            np.float64,
            copy=False,
        )


@dataclass(frozen=True)
class OSMRoadNetwork:
    """OSM road centerlines represented as 2D ENU line segments."""

    origin_lat_deg: float
    origin_lon_deg: float
    origin_alt_m: float
    origin_ecef_m: np.ndarray
    east_basis: np.ndarray
    north_basis: np.ndarray
    segments: RoadSegmentArrays

    @classmethod
    def from_geojson(
        cls,
        path: str | Path,
        *,
        origin_lat_deg: float | None = None,
        origin_lon_deg: float | None = None,
        origin_alt_m: float = 0.0,
        include_service: bool = True,
        limited_access_sigma_scale: float = DEFAULT_LIMITED_ACCESS_SIGMA_SCALE,
    ) -> "OSMRoadNetwork":
        """Load drivable OSM road centerlines from a GeoJSON file."""

        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_geojson_mapping(
            data,
            origin_lat_deg=origin_lat_deg,
            origin_lon_deg=origin_lon_deg,
            origin_alt_m=origin_alt_m,
            include_service=include_service,
            limited_access_sigma_scale=limited_access_sigma_scale,
        )

    @classmethod
    def from_geojson_mapping(
        cls,
        data: dict[str, object],
        *,
        origin_lat_deg: float | None = None,
        origin_lon_deg: float | None = None,
        origin_alt_m: float = 0.0,
        include_service: bool = True,
        limited_access_sigma_scale: float = DEFAULT_LIMITED_ACCESS_SIGMA_SCALE,
    ) -> "OSMRoadNetwork":
        """Build a road network from a GeoJSON-like mapping."""

        raw_segments: list[tuple[tuple[float, float], tuple[float, float], str, float]] = []
        lon_samples: list[float] = []
        lat_samples: list[float] = []
        for feature in _iter_geojson_features(data):
            props = feature.get("properties", {})
            if not isinstance(props, dict):
                props = {}
            highway = _primary_highway_type(props.get("highway"))
            if not _is_drivable_highway(highway, include_service=include_service):
                continue
            scale = _sigma_scale_for_highway(highway, limited_access_sigma_scale)
            for coords in _iter_linestring_coordinates(feature.get("geometry")):
                if len(coords) < 2:
                    continue
                for p0, p1 in zip(coords[:-1], coords[1:]):
                    lon0, lat0 = _lon_lat_pair(p0)
                    lon1, lat1 = _lon_lat_pair(p1)
                    if lon0 == lon1 and lat0 == lat1:
                        continue
                    raw_segments.append(((lon0, lat0), (lon1, lat1), highway, scale))
                    lon_samples.extend((lon0, lon1))
                    lat_samples.extend((lat0, lat1))

        if not raw_segments:
            raise ValueError("GeoJSON contains no drivable OSM road segments")

        if origin_lat_deg is None:
            origin_lat_deg = float(np.mean(lat_samples))
        if origin_lon_deg is None:
            origin_lon_deg = float(np.mean(lon_samples))

        origin_ecef = lla_deg_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
        east, north, _up = enu_basis(math.radians(origin_lat_deg), math.radians(origin_lon_deg))

        starts = []
        ends = []
        highway_types = []
        scales = []
        for (lon0, lat0), (lon1, lat1), highway, scale in raw_segments:
            starts.append(
                lla_deg_to_enu(
                    lat0,
                    lon0,
                    0.0,
                    origin_ecef=origin_ecef,
                    east_basis=east,
                    north_basis=north,
                )
            )
            ends.append(
                lla_deg_to_enu(
                    lat1,
                    lon1,
                    0.0,
                    origin_ecef=origin_ecef,
                    east_basis=east,
                    north_basis=north,
                )
            )
            highway_types.append(highway)
            scales.append(scale)

        return cls(
            origin_lat_deg=float(origin_lat_deg),
            origin_lon_deg=float(origin_lon_deg),
            origin_alt_m=float(origin_alt_m),
            origin_ecef_m=origin_ecef,
            east_basis=east,
            north_basis=north,
            segments=RoadSegmentArrays(
                np.asarray(starts, dtype=np.float64),
                np.asarray(ends, dtype=np.float64),
                np.asarray(scales, dtype=np.float64),
                tuple(highway_types),
            ),
        )

    def ecef_to_enu2(self, ecef_m: np.ndarray) -> np.ndarray:
        """Convert ECEF positions to local horizontal ENU coordinates."""

        pts = np.asarray(ecef_m, dtype=np.float64)
        flat = pts.reshape(-1, pts.shape[-1])
        if flat.shape[1] < 3:
            raise ValueError("ecef_m must have shape (..., 3)")
        diff = flat[:, :3] - self.origin_ecef_m.reshape(1, 3)
        en = np.column_stack((diff @ self.east_basis, diff @ self.north_basis))
        return en.reshape(pts.shape[:-1] + (2,))

    def nearest_distances_ecef(
        self,
        ecef_m: np.ndarray,
        *,
        max_segments: int | None = None,
        center_ecef_m: np.ndarray | None = None,
        search_radius_m: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest horizontal road distance and sigma scale per point."""

        points_enu = self.ecef_to_enu2(ecef_m).reshape(-1, 2)
        segments = self.segments
        if center_ecef_m is not None or search_radius_m is not None or max_segments is not None:
            center = (
                self.ecef_to_enu2(np.asarray(center_ecef_m, dtype=np.float64).reshape(1, 3))[0]
                if center_ecef_m is not None
                else np.mean(points_enu, axis=0)
            )
            segments = self.candidate_segments_enu(
                center,
                radius_m=search_radius_m,
                max_segments=max_segments,
            )
        d, scale = nearest_distances_enu(points_enu, segments)
        out_shape = np.asarray(ecef_m).shape[:-1]
        return d.reshape(out_shape), scale.reshape(out_shape)

    def candidate_segments_enu(
        self,
        center_enu_m: np.ndarray,
        *,
        radius_m: float | None = None,
        max_segments: int | None = None,
    ) -> RoadSegmentArrays:
        """Select nearby road segments around a local ENU center."""

        center = np.asarray(center_enu_m, dtype=np.float64).ravel()
        if center.size != 2:
            raise ValueError("center_enu_m must have shape (2,)")
        starts = self.segments.start_enu_m
        ends = self.segments.end_enu_m
        mins = np.minimum(starts, ends)
        maxs = np.maximum(starts, ends)
        dx = np.maximum(np.maximum(mins[:, 0] - center[0], 0.0), center[0] - maxs[:, 0])
        dy = np.maximum(np.maximum(mins[:, 1] - center[1], 0.0), center[1] - maxs[:, 1])
        bbox_dist = np.sqrt(dx * dx + dy * dy)
        mask = np.ones(self.segments.n_segments, dtype=bool)
        if radius_m is not None:
            mask &= bbox_dist <= float(radius_m)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            idx = np.argsort(bbox_dist)[:1]
        if max_segments is not None and idx.size > int(max_segments):
            order = np.argsort(bbox_dist[idx])[: int(max_segments)]
            idx = idx[order]
        return RoadSegmentArrays(
            starts[idx],
            ends[idx],
            self.segments.sigma_scales[idx],
            tuple(self.segments.highway_types[i] for i in idx),
        )

    def candidate_kernel_array(
        self,
        center_ecef_m: np.ndarray,
        *,
        radius_m: float = 80.0,
        max_segments: int = 96,
    ) -> np.ndarray:
        """Return nearby segments as ``[x1, y1, x2, y2, sigma_scale]`` rows."""

        center_enu = self.ecef_to_enu2(np.asarray(center_ecef_m, dtype=np.float64).reshape(1, 3))[0]
        return self.candidate_segments_enu(
            center_enu,
            radius_m=radius_m,
            max_segments=max_segments,
        ).as_kernel_array()


def nearest_distances_enu(
    points_enu_m: np.ndarray,
    segments: RoadSegmentArrays,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute nearest 2D point-to-segment distances in meters."""

    points = np.asarray(points_enu_m, dtype=np.float64).reshape(-1, 2)
    if segments.n_segments == 0:
        raise ValueError("segments must contain at least one segment")

    starts = segments.start_enu_m
    ends = segments.end_enu_m
    seg = ends - starts
    seg_len2 = np.sum(seg * seg, axis=1)
    seg_len2 = np.maximum(seg_len2, 1.0e-12)

    out_d = np.empty(points.shape[0], dtype=np.float64)
    out_scale = np.empty(points.shape[0], dtype=np.float64)
    for lo in range(0, points.shape[0], chunk_size):
        hi = min(lo + chunk_size, points.shape[0])
        p = points[lo:hi, None, :]
        rel = p - starts[None, :, :]
        t = np.sum(rel * seg[None, :, :], axis=2) / seg_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
        diff = p - closest
        d2 = np.sum(diff * diff, axis=2)
        arg = np.argmin(d2, axis=1)
        out_d[lo:hi] = np.sqrt(d2[np.arange(hi - lo), arg])
        out_scale[lo:hi] = segments.sigma_scales[arg]
    return out_d, out_scale


def huber_road_log_weight(
    distances_m: np.ndarray,
    *,
    sigma_m: float = 2.0,
    huber_k: float = 2.0,
    sigma_scales: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Compute soft Huber log-weight increments for road distances."""

    if sigma_m < 1.0:
        raise ValueError("sigma_m must be >= 1.0m for a conservative road constraint")
    if huber_k <= 0.0:
        raise ValueError("huber_k must be positive")
    d = np.asarray(distances_m, dtype=np.float64)
    scales = np.asarray(sigma_scales, dtype=np.float64)
    z = d / (float(sigma_m) * scales)
    loss = np.where(z <= huber_k, 0.5 * z * z, huber_k * z - 0.5 * huber_k * huber_k)
    return -loss


def fetch_overpass_roads_geojson(
    bbox: tuple[float, float, float, float],
    out_path: str | Path,
    *,
    timeout_s: int = 90,
    overpass_url: str = "https://overpass-api.de/api/interpreter",
) -> Path:
    """Fetch OSM road ways from Overpass and save a compact GeoJSON cache.

    Parameters
    ----------
    bbox
        ``(south_lat, west_lon, north_lat, east_lon)``.
    out_path
        Destination GeoJSON path. Parent directories are created.
    """

    south, west, north, east = [float(x) for x in bbox]
    query = f"""
[out:json][timeout:{int(timeout_s)}];
(
  way["highway"]({south:.8f},{west:.8f},{north:.8f},{east:.8f});
);
out tags geom;
"""
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        overpass_url,
        data=data,
        headers={"User-Agent": "gnss_gpu osm_constraint/0.1"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s + 15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    features = []
    for element in payload.get("elements", []):
        if not isinstance(element, dict) or element.get("type") != "way":
            continue
        tags = element.get("tags", {})
        if not isinstance(tags, dict):
            tags = {}
        highway = _primary_highway_type(tags.get("highway"))
        if not _is_drivable_highway(highway, include_service=True):
            continue
        geom = element.get("geometry", [])
        coords = []
        for node in geom:
            if isinstance(node, dict) and "lon" in node and "lat" in node:
                coords.append([float(node["lon"]), float(node["lat"])])
        if len(coords) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"id": element.get("id"), **tags},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"))
    return path


def lla_deg_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def enu_basis(lat_rad: float, lon_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
    return east, north, up


def lla_deg_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    *,
    origin_ecef: np.ndarray,
    east_basis: np.ndarray,
    north_basis: np.ndarray,
) -> np.ndarray:
    diff = lla_deg_to_ecef(lat_deg, lon_deg, alt_m) - np.asarray(origin_ecef, dtype=np.float64)
    return np.array([float(diff @ east_basis), float(diff @ north_basis)], dtype=np.float64)


def _iter_geojson_features(data: dict[str, object]) -> Iterable[dict[str, object]]:
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if isinstance(features, list):
            for feature in features:
                if isinstance(feature, dict):
                    yield feature
    elif data.get("type") == "Feature":
        yield data
    else:
        yield {"type": "Feature", "properties": {}, "geometry": data}


def _iter_linestring_coordinates(geometry: object) -> Iterable[list[object]]:
    if not isinstance(geometry, dict):
        return
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "LineString" and isinstance(coords, list):
        yield coords
    elif geom_type == "MultiLineString" and isinstance(coords, list):
        for line in coords:
            if isinstance(line, list):
                yield line


def _lon_lat_pair(coord: object) -> tuple[float, float]:
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        raise ValueError("GeoJSON coordinates must be [lon, lat, ...]")
    return float(coord[0]), float(coord[1])


def _primary_highway_type(value: object) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if value is None:
        return ""
    return str(value)


def _is_drivable_highway(highway: str, *, include_service: bool) -> bool:
    if not highway:
        return False
    if highway in _NON_DRIVABLE_HIGHWAYS:
        return False
    if highway == "service" and not include_service:
        return False
    return True


def _sigma_scale_for_highway(highway: str, limited_access_sigma_scale: float) -> float:
    if highway in _LIMITED_ACCESS_HIGHWAYS:
        return float(limited_access_sigma_scale)
    return 1.0
