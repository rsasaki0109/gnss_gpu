"""Per-triangle reflection material classification for city meshes.

Real buildings are not uniform reflectors: walls, roofs, and ground-level
surfaces have different composition (concrete, glass, asphalt/soil, ...)
and therefore different Fresnel reflection coefficients (see
:mod:`gnss_gpu.fresnel`).  Historically the reflection path amplitude
computation (see :mod:`gnss_gpu.urban_signal_sim`) used a single global
``reflector_material`` for every triangle in the mesh.

This module classifies each triangle of a mesh into one of three broad
surface categories -- ``"wall"``, ``"roof"``, ``"ground"`` -- and maps each
category to a material name understood by :data:`gnss_gpu.fresnel.MATERIALS`.
Two classification sources are combined:

1. **CityGML boundary-surface tags** (primary, when available).  LOD2+
   CityGML carries ``bldg:WallSurface`` / ``bldg:RoofSurface`` /
   ``bldg:GroundSurface`` elements; :mod:`gnss_gpu.io.citygml` records this
   as a parallel ``surface_kinds`` list on :class:`~gnss_gpu.io.citygml.CityFeature`.
   PLATEAU LOD1 extrusions (the vast majority of published PLATEAU data,
   including the bundled Odaiba sample set) carry no such tags -- every
   polygon comes back tagged ``"unknown"``.
2. **Triangle-normal heuristic** (fallback, always available).  A triangle
   whose unit normal is nearly horizontal is a wall; a nearly-vertical,
   upward-facing triangle is a roof, unless it sits close to the lowest
   point of the mesh (its "ground level"), in which case it is classified
   as ground.

The two sources are combined per triangle: a tag-derived category wins
when present and recognized; the normal heuristic fills in everywhere
else (including the entire mesh when no tags/features are supplied).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from gnss_gpu.fresnel import MATERIALS

# Surface category -> default material name (must exist in
# gnss_gpu.fresnel.MATERIALS).  Callers may override individual entries via
# the ``mapping`` argument of :func:`classify_surface_materials`; unlisted
# categories keep these defaults.
DEFAULT_SURFACE_MATERIALS: dict[str, str] = {
    "wall": "concrete",
    "roof": "concrete",
    "ground": "dry_ground",
}

_VALID_CATEGORIES = frozenset(DEFAULT_SURFACE_MATERIALS)

# Coordinates whose centroid magnitude exceeds this are assumed to be
# geocentric ECEF (Earth radius ~6.371e6 m); local/ENU-style meshes (test
# fixtures, small synthetic scenes) are far smaller and default to z-up.
_GEOCENTRIC_RADIUS_THRESHOLD_M = 1.0e6

# |normal . up| below this is "horizontal enough" to call a wall.
_WALL_VERTICAL_THRESHOLD = 0.3


def _validate_mapping(mapping: Optional[Mapping[str, str]]) -> dict[str, str]:
    merged = dict(DEFAULT_SURFACE_MATERIALS)
    if mapping:
        merged.update(mapping)
    for category, material in merged.items():
        if category not in _VALID_CATEGORIES:
            raise ValueError(
                f"unknown surface category {category!r} in mapping; "
                f"expected one of {sorted(_VALID_CATEGORIES)}"
            )
        if material not in MATERIALS:
            raise ValueError(
                f"unknown material {material!r} for category {category!r}; "
                f"see gnss_gpu.fresnel.MATERIALS for valid names"
            )
    return merged


def _resolve_up(triangles: np.ndarray, up) -> np.ndarray:
    """Return a per-triangle unit "up" vector, shape ``(n_tri, 3)``."""
    n_tri = triangles.shape[0]
    if n_tri == 0:
        return np.zeros((0, 3), dtype=float)

    if up is not None:
        up_arr = np.asarray(up, dtype=float)
        if up_arr.ndim == 1:
            if up_arr.shape != (3,):
                raise ValueError("up vector must have shape (3,)")
            norm = float(np.linalg.norm(up_arr))
            if norm <= 0.0:
                raise ValueError("up vector must be non-zero")
            return np.broadcast_to(up_arr / norm, (n_tri, 3)).copy()
        if up_arr.shape != (n_tri, 3):
            raise ValueError("up array must have shape (3,) or (n_tri, 3)")
        norms = np.linalg.norm(up_arr, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        return up_arr / norms

    mesh_centroid = triangles.reshape(-1, 3).mean(axis=0)
    if np.linalg.norm(mesh_centroid) > _GEOCENTRIC_RADIUS_THRESHOLD_M:
        # ECEF-like coordinates: approximate the local "up" at each triangle
        # as the normalized geocentric direction of its centroid. Building
        # meshes are tiny compared to Earth's radius, so this is an
        # excellent local approximation of the ellipsoidal normal.
        tri_centroids = triangles.mean(axis=1)
        norms = np.linalg.norm(tri_centroids, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        return tri_centroids / norms

    # Local/ENU-style coordinates: assume a conventional z-up frame.
    local_up = np.array([0.0, 0.0, 1.0], dtype=float)
    return np.broadcast_to(local_up, (n_tri, 3)).copy()


def _classify_by_normal(
    triangles: np.ndarray, up_vectors: np.ndarray, ground_height_tol_m: float
) -> np.ndarray:
    """Normal/height heuristic fallback classification.

    Returns an ``(n_tri,)`` object array of category strings
    ``"wall"`` / ``"roof"`` / ``"ground"``.
    """
    n_tri = triangles.shape[0]
    categories = np.full(n_tri, "wall", dtype=object)
    if n_tri == 0:
        return categories

    p0 = triangles[:, 0, :]
    p1 = triangles[:, 1, :]
    p2 = triangles[:, 2, :]
    normal = np.cross(p1 - p0, p2 - p0)
    normal_len = np.linalg.norm(normal, axis=1)
    safe_len = np.where(normal_len > 1e-12, normal_len, 1.0)
    unit_normal = normal / safe_len[:, None]

    vertical = np.einsum("ij,ij->i", unit_normal, up_vectors)

    centroids = triangles.mean(axis=1)
    heights = np.einsum("ij,ij->i", centroids, up_vectors)
    min_height = float(np.min(heights))
    height_range = float(np.max(heights) - min_height)
    tol = max(float(ground_height_tol_m), 0.05 * height_range)

    is_wall = np.abs(vertical) < _WALL_VERTICAL_THRESHOLD
    is_upward = vertical >= _WALL_VERTICAL_THRESHOLD
    is_downward = vertical <= -_WALL_VERTICAL_THRESHOLD
    is_near_ground = (heights - min_height) <= tol

    categories[is_wall] = "wall"
    categories[is_upward & is_near_ground] = "ground"
    categories[is_upward & ~is_near_ground] = "roof"
    # Downward-facing, non-wall triangles are rare on extruded city meshes
    # (undersides of overhangs/eaves); default them to "roof" as the
    # closer material analog.
    categories[is_downward] = "roof"

    return categories


def _categories_from_surface_kinds(surface_kinds, n_tri: int) -> Optional[np.ndarray]:
    if surface_kinds is None:
        return None
    kinds = list(surface_kinds)
    if len(kinds) != n_tri:
        raise ValueError(
            f"surface_kinds length ({len(kinds)}) does not match the number "
            f"of triangles ({n_tri})"
        )
    return np.array(
        [k if k in _VALID_CATEGORIES else None for k in kinds], dtype=object
    )


def _categories_from_features(features, n_tri: int) -> Optional[np.ndarray]:
    """Best-effort per-triangle categories reconstructed from CityFeatures.

    Mirrors :meth:`gnss_gpu.io.plateau.PlateauLoader._polygon_to_triangles`'s
    fan triangulation (an ``n``-vertex polygon, after dropping a duplicated
    closing vertex, yields ``n - 2`` triangles; polygons with fewer than 3
    unique vertices yield none) so triangle counts line up *if* ``features``
    is the exact CityGML feature list that produced ``triangles``.  If the
    reconstructed count does not match ``n_tri`` this returns ``None`` and
    callers fall back entirely to the normal-based heuristic -- safer than
    silently misaligning materials to the wrong triangles.
    """
    if not features:
        return None
    out: list[Optional[str]] = []
    for feat in features:
        kinds = list(getattr(feat, "surface_kinds", None) or [])
        polygons = getattr(feat, "polygons", None) or []
        for poly_idx, polygon in enumerate(polygons):
            kind = kinds[poly_idx] if poly_idx < len(kinds) else "unknown"
            verts = np.asarray(polygon, dtype=float)
            if verts.ndim != 2 or verts.shape[1] != 3:
                return None
            if verts.shape[0] >= 2 and np.allclose(verts[0], verts[-1]):
                verts = verts[:-1]
            n_verts = verts.shape[0]
            if n_verts < 3:
                continue
            category = kind if kind in _VALID_CATEGORIES else None
            out.extend([category] * (n_verts - 2))
    if len(out) != n_tri:
        return None
    return np.array(out, dtype=object)


def classify_surface_materials(
    triangles,
    features: Optional[Iterable] = None,
    mapping: Optional[Mapping[str, str]] = None,
    *,
    surface_kinds: Optional[Sequence[str]] = None,
    up=None,
    ground_height_tol_m: float = 0.5,
) -> np.ndarray:
    """Return a per-triangle material-name array for Fresnel reflection.

    Parameters
    ----------
    triangles : array-like, shape (n_tri, 3, 3)
        Mesh triangle vertices (any consistent coordinate frame: ECEF or
        local/ENU both work -- see ``up``).
    features : list of gnss_gpu.io.citygml.CityFeature, optional
        The parsed CityGML features that produced ``triangles`` (via the
        same fan triangulation as :class:`gnss_gpu.io.plateau.PlateauLoader`).
        Used to recover CityGML boundary-surface tags (``surface_kinds`` on
        each feature) when ``surface_kinds`` is not supplied directly. Best
        effort: if the reconstructed triangle count does not match
        ``len(triangles)`` this input is ignored and the normal-based
        heuristic is used for the whole mesh.
    mapping : dict, optional
        Overrides for the category -> material-name mapping. Merged over
        :data:`DEFAULT_SURFACE_MATERIALS`; every material name must exist in
        :data:`gnss_gpu.fresnel.MATERIALS`.
    surface_kinds : sequence of str, optional
        Per-triangle CityGML-tag category, already aligned 1:1 with
        ``triangles`` (typically produced by
        :meth:`gnss_gpu.io.plateau.PlateauLoader.load_citygml` /
        ``load_directory`` with ``return_materials=True``, which build this
        during triangulation and are therefore guaranteed to align).  Takes
        priority over ``features`` when both are given. Entries of
        ``"unknown"`` (or any value outside ``{"wall", "roof", "ground"}``)
        fall back to the normal-based heuristic for that triangle.
    up : array-like, optional
        Local "up" direction: either a single ``(3,)`` vector or a per
        -triangle ``(n_tri, 3)`` array. If omitted, "up" is auto-detected:
        geocentric ECEF (normalized triangle centroid) for meshes whose
        overall centroid is far from the origin (Earth-radius scale), else
        a fixed ``z``-up vector for local/ENU-scale meshes.
    ground_height_tol_m : float, optional
        Absolute tolerance (metres, along ``up``) below which an
        upward-facing triangle counts as "near ground" (vs. roof). The
        effective tolerance is ``max(ground_height_tol_m, 0.05 *
        mesh_height_range)`` so tall buildings get a proportionally larger
        ground band.

    Returns
    -------
    np.ndarray of str, shape (n_tri,)
        Material name for each triangle (a key of
        :data:`gnss_gpu.fresnel.MATERIALS`), aligned 1:1 with ``triangles``.
    """
    tris = np.asarray(triangles, dtype=float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (n_tri, 3, 3)")
    n_tri = tris.shape[0]
    resolved_mapping = _validate_mapping(mapping)

    if n_tri == 0:
        return np.array([], dtype=str)

    tag_categories = _categories_from_surface_kinds(surface_kinds, n_tri)
    if tag_categories is None:
        tag_categories = _categories_from_features(features, n_tri)

    up_vectors = _resolve_up(tris, up)
    fallback_categories = _classify_by_normal(tris, up_vectors, ground_height_tol_m)

    if tag_categories is None:
        final_categories = fallback_categories
    else:
        has_tag = tag_categories != None  # noqa: E711
        final_categories = np.where(has_tag, tag_categories, fallback_categories)

    materials = np.array(
        [resolved_mapping[c] for c in final_categories], dtype=str
    )
    return materials


__all__ = [
    "DEFAULT_SURFACE_MATERIALS",
    "classify_surface_materials",
]
