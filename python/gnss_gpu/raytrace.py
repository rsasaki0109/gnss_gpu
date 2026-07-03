from dataclasses import dataclass

import numpy as np


@dataclass
class ReflectionPath:
    """First-order specular reflection path."""

    excess_delay: float
    reflection_point: np.ndarray
    triangle_id: int
    normal: np.ndarray
    incidence_angle: float


def _closest_point_on_triangle(point, triangle):
    """Return the closest point on a triangle to ``point`` (Ericson, RTCD 5.1.5).

    Used to accept a specular reflection point that lands just outside a finite
    mesh triangle -- on real tessellated city meshes (e.g. PLATEAU) the analytic
    specular point frequently falls a fraction of a metre into a gap between
    triangles or past an edge, even though the wall is physically there.
    """
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a

    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def horizontal_ground_plane(height=0.0):
    """Return (point, normal) for a z-up horizontal ground plane at z=height."""
    return (
        np.array([0.0, 0.0, float(height)], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )


def _validate_rx_ecef(rx_ecef) -> np.ndarray:
    rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
    if rx.size != 3:
        raise ValueError("rx_ecef must have shape (3,)")
    if not np.all(np.isfinite(rx)):
        raise ValueError("rx_ecef must be finite")
    return np.ascontiguousarray(rx)


def _validate_sat_ecef(sat_ecef) -> np.ndarray:
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    if sat.shape[0] == 0:
        raise ValueError("sat_ecef must contain at least one satellite")
    if not np.all(np.isfinite(sat)):
        raise ValueError("sat_ecef must be finite")
    return np.ascontiguousarray(sat)


class BuildingModel:
    """3D building model for GNSS ray tracing NLOS detection.

    Unlike ``BVHAccelerator``, an empty triangle mesh ``(0, 3, 3)`` is allowed
    here; linear-scan raytrace kernels treat it as open sky (all satellites LOS).
    """

    def __init__(self, triangles):
        """Initialize with triangle mesh.

        Args:
            triangles: [N, 3, 3] numpy array of triangle vertices.
                       Each triangle has 3 vertices, each vertex has 3 coordinates (x, y, z).
        """
        self.triangles = np.asarray(triangles, dtype=np.float64)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape [N, 3, 3]")
        if not np.all(np.isfinite(self.triangles)):
            raise ValueError("triangles must be finite")

    def check_los(self, rx_ecef, sat_ecef):
        """Check line-of-sight for each satellite.

        Args:
            rx_ecef: [3] receiver ECEF position in meters.
            sat_ecef: [n_sat, 3] satellite ECEF positions in meters.

        Returns:
            is_los: [n_sat] boolean array, True if line-of-sight is clear.
        """
        from gnss_gpu._raytrace import raytrace_los_check

        rx = _validate_rx_ecef(rx_ecef)
        sat = _validate_sat_ecef(sat_ecef)
        tri = self.triangles.reshape(-1, 3, 3)

        return raytrace_los_check(rx, sat, tri)

    def compute_multipath(self, rx_ecef, sat_ecef):
        """Compute first-order multipath reflections.

        Args:
            rx_ecef: [3] receiver ECEF position in meters.
            sat_ecef: [n_sat, 3] satellite ECEF positions in meters.

        Returns:
            excess_delays: [n_sat] excess path delay in meters (0 if no reflection).
            reflection_points: [n_sat, 3] reflection point coordinates.
        """
        from gnss_gpu._raytrace import raytrace_multipath

        rx = _validate_rx_ecef(rx_ecef)
        sat = _validate_sat_ecef(sat_ecef)
        tri = self.triangles.reshape(-1, 3, 3)

        reflection_points, excess_delays = raytrace_multipath(rx, sat, tri)
        return excess_delays, reflection_points

    def compute_reflection_paths(self, rx_ecef, sat_ecef, max_paths=4,
                                 ground_plane=None, reflection_point_tol_m=0.0):
        """Compute first-order specular reflection paths using a pure-Python image method.

        Args:
            rx_ecef: [3] receiver ECEF position in meters.
            sat_ecef: [n_sat, 3] satellite ECEF positions in meters, or a single [3] position.
            max_paths: maximum number of reflection paths returned per satellite.
            reflection_point_tol_m: if > 0, accept a specular reflection point that
                lies within this many metres of a triangle (snapping it to the
                nearest point on the triangle) instead of requiring strict
                containment. This recovers reflections that the analytic specular
                point misses by a fraction of a metre on tessellated real-world
                meshes (gaps/edges between PLATEAU triangles); 0 keeps the strict
                behaviour.

        Returns:
            A list of length n_sat. Each item is a list of ReflectionPath instances sorted
            by increasing excess delay.

        Notes:
            The incidence_angle is the acute angle between the incoming ray vector
            ``reflection_point - satellite_position`` and the triangle unit normal.
            It is 0 rad for normal incidence and pi/2 rad for grazing incidence.
        """
        reflection_point_tol_m = float(reflection_point_tol_m)
        rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
        if rx.size != 3:
            raise ValueError("rx_ecef must have shape [3]")

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        tri = self.triangles.reshape(-1, 3, 3)

        max_paths = int(max_paths)
        results = [[] for _ in range(sat.shape[0])]
        if max_paths <= 0:
            return results
        if tri.shape[0] == 0 and ground_plane is None:
            return results

        plane_eps = 1e-6
        area_eps = 1e-12
        param_eps = 1e-12
        bary_tol = 1e-9
        delay_eps = 1e-6
        intersect_eps = 1e-6

        ground_point = None
        ground_normal = None
        if ground_plane is not None:
            ground_point, ground_normal = ground_plane
            ground_point = np.asarray(ground_point, dtype=np.float64).ravel()
            ground_normal = np.asarray(ground_normal, dtype=np.float64).ravel()
            if ground_point.size != 3 or ground_normal.size != 3:
                raise ValueError("ground_plane must be (point[3], normal[3])")

            normal_len = float(np.linalg.norm(ground_normal))
            if normal_len <= area_eps:
                raise ValueError("ground_plane normal must be non-zero")
            ground_normal = ground_normal / normal_len

        def _point_in_triangle(point, triangle):
            a, b, c = triangle
            v0 = b - a
            v1 = c - a
            v2 = point - a

            dot00 = float(np.dot(v0, v0))
            dot01 = float(np.dot(v0, v1))
            dot02 = float(np.dot(v0, v2))
            dot11 = float(np.dot(v1, v1))
            dot12 = float(np.dot(v1, v2))

            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) <= area_eps:
                return False

            inv_denom = 1.0 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom

            return u >= -bary_tol and v >= -bary_tol and (u + v) <= 1.0 + bary_tol

        def _segment_intersects_triangle(start, end, triangle):
            direction = end - start
            seg_len = float(np.linalg.norm(direction))
            if seg_len <= intersect_eps:
                return False

            v0, v1, v2 = triangle
            edge1 = v1 - v0
            edge2 = v2 - v0
            if np.linalg.norm(np.cross(edge1, edge2)) <= area_eps:
                return False

            pvec = np.cross(direction, edge2)
            det = float(np.dot(edge1, pvec))
            if abs(det) <= area_eps:
                return False

            inv_det = 1.0 / det
            tvec = start - v0
            u = float(np.dot(tvec, pvec)) * inv_det
            if u < -bary_tol or u > 1.0 + bary_tol:
                return False

            qvec = np.cross(tvec, edge1)
            v = float(np.dot(direction, qvec)) * inv_det
            if v < -bary_tol or (u + v) > 1.0 + bary_tol:
                return False

            t = float(np.dot(edge2, qvec)) * inv_det
            endpoint_tol = intersect_eps / seg_len
            return endpoint_tol < t < 1.0 - endpoint_tol

        def _segment_blocked(start, end, excluded_triangle_id):
            for obstacle_id, obstacle in enumerate(tri):
                if obstacle_id == excluded_triangle_id:
                    continue
                if _segment_intersects_triangle(start, end, obstacle):
                    return True
            return False

        def _append_ground_reflection(paths, sat_pos, direct_len):
            if ground_point is None:
                return

            rx_side = float(np.dot(rx - ground_point, ground_normal))
            if abs(rx_side) < plane_eps:
                return

            sat_side = float(np.dot(sat_pos - ground_point, ground_normal))
            if abs(sat_side) < plane_eps or rx_side * sat_side <= 0.0:
                return

            rx_img = rx - 2.0 * rx_side * ground_normal
            image_dir = sat_pos - rx_img
            denom = float(np.dot(image_dir, ground_normal))
            if abs(denom) < plane_eps:
                return

            t = float(np.dot(ground_point - rx_img, ground_normal)) / denom
            if not (param_eps < t < 1.0 - param_eps):
                return

            reflection_point = rx_img + t * image_dir

            if _segment_blocked(rx, reflection_point, -1):
                return
            if _segment_blocked(reflection_point, sat_pos, -1):
                return

            reflected_len = (
                float(np.linalg.norm(rx - reflection_point))
                + float(np.linalg.norm(reflection_point - sat_pos))
            )
            excess_delay = reflected_len - direct_len
            if excess_delay <= delay_eps:
                return

            incoming = reflection_point - sat_pos
            incoming_len = float(np.linalg.norm(incoming))
            if incoming_len <= plane_eps:
                return

            cos_incidence = abs(float(np.dot(incoming / incoming_len, ground_normal)))
            cos_incidence = float(np.clip(cos_incidence, 0.0, 1.0))
            incidence_angle = float(np.arccos(cos_incidence))

            paths.append(ReflectionPath(
                excess_delay=float(excess_delay),
                reflection_point=reflection_point.copy(),
                triangle_id=-1,
                normal=ground_normal.copy(),
                incidence_angle=incidence_angle,
            ))

        for sat_idx, sat_pos in enumerate(sat):
            direct_len = float(np.linalg.norm(rx - sat_pos))
            paths = []

            for tri_idx, triangle in enumerate(tri):
                p0, p1, p2 = triangle
                normal = np.cross(p1 - p0, p2 - p0)
                normal_len = float(np.linalg.norm(normal))
                if normal_len <= area_eps:
                    continue
                normal = normal / normal_len

                rx_side = float(np.dot(rx - p0, normal))
                if abs(rx_side) < plane_eps:
                    continue

                sat_side = float(np.dot(sat_pos - p0, normal))
                if abs(sat_side) < plane_eps or rx_side * sat_side <= 0.0:
                    continue

                rx_img = rx - 2.0 * rx_side * normal
                image_dir = sat_pos - rx_img
                denom = float(np.dot(image_dir, normal))
                if abs(denom) < plane_eps:
                    continue

                t = float(np.dot(p0 - rx_img, normal)) / denom
                if not (param_eps < t < 1.0 - param_eps):
                    continue

                reflection_point = rx_img + t * image_dir
                if not _point_in_triangle(reflection_point, triangle):
                    if reflection_point_tol_m <= 0.0:
                        continue
                    closest = _closest_point_on_triangle(reflection_point, triangle)
                    if float(np.linalg.norm(reflection_point - closest)) > reflection_point_tol_m:
                        continue
                    # Snap to the wall so path length / occlusion use a point that
                    # actually lies on the (physically present) facade.
                    reflection_point = closest

                if _segment_blocked(rx, reflection_point, tri_idx):
                    continue
                if _segment_blocked(reflection_point, sat_pos, tri_idx):
                    continue

                reflected_len = (
                    float(np.linalg.norm(rx - reflection_point))
                    + float(np.linalg.norm(reflection_point - sat_pos))
                )
                excess_delay = reflected_len - direct_len
                if excess_delay <= delay_eps:
                    continue

                incoming = reflection_point - sat_pos
                incoming_len = float(np.linalg.norm(incoming))
                if incoming_len <= plane_eps:
                    continue

                cos_incidence = abs(float(np.dot(incoming / incoming_len, normal)))
                cos_incidence = float(np.clip(cos_incidence, 0.0, 1.0))
                incidence_angle = float(np.arccos(cos_incidence))

                paths.append(ReflectionPath(
                    excess_delay=float(excess_delay),
                    reflection_point=reflection_point.copy(),
                    triangle_id=int(tri_idx),
                    normal=normal.copy(),
                    incidence_angle=incidence_angle,
                ))

            _append_ground_reflection(paths, sat_pos, direct_len)

            paths.sort(key=lambda path: path.excess_delay)
            results[sat_idx] = paths[:max_paths]

        return results

    @classmethod
    def from_obj(cls, filepath):
        """Load building model from a Wavefront OBJ file.

        Supports only 'v' (vertex) and 'f' (face) lines.
        Faces are triangulated using fan triangulation.

        Args:
            filepath: path to OBJ file.

        Returns:
            BuildingModel instance.
        """
        vertices = []
        triangles = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':
                    # Parse face indices (1-based, may contain v/vt/vn format)
                    indices = []
                    for p in parts[1:]:
                        idx = int(p.split('/')[0]) - 1
                        indices.append(idx)
                    # Fan triangulation for polygons
                    for i in range(1, len(indices) - 1):
                        triangles.append([vertices[indices[0]],
                                          vertices[indices[i]],
                                          vertices[indices[i + 1]]])

        return cls(np.array(triangles, dtype=np.float64))

    @classmethod
    def create_box(cls, center, width, depth, height):
        """Create a box building from center position and dimensions.

        Args:
            center: [3] geometric center of the box (x, y, z).
            width: box width along x-axis.
            depth: box depth along y-axis.
            height: box height along z-axis.

        Returns:
            BuildingModel with 12 triangles (6 faces, 2 triangles each).
        """
        cx, cy, cz = center[0], center[1], center[2]
        hw = width / 2.0
        hd = depth / 2.0
        hh = height / 2.0

        # 8 vertices of the box (center is the geometric center)
        v = np.array([
            [cx - hw, cy - hd, cz - hh],       # 0: bottom-left-front
            [cx + hw, cy - hd, cz - hh],       # 1: bottom-right-front
            [cx + hw, cy + hd, cz - hh],       # 2: bottom-right-back
            [cx - hw, cy + hd, cz - hh],       # 3: bottom-left-back
            [cx - hw, cy - hd, cz + hh],       # 4: top-left-front
            [cx + hw, cy - hd, cz + hh],       # 5: top-right-front
            [cx + hw, cy + hd, cz + hh],       # 6: top-right-back
            [cx - hw, cy + hd, cz + hh],       # 7: top-left-back
        ], dtype=np.float64)

        # 12 triangles (2 per face, 6 faces)
        faces = [
            # Bottom face
            [0, 1, 2], [0, 2, 3],
            # Top face
            [4, 6, 5], [4, 7, 6],
            # Front face (y = cy - hd)
            [0, 5, 1], [0, 4, 5],
            # Back face (y = cy + hd)
            [2, 7, 3], [2, 6, 7],
            # Left face (x = cx - hw)
            [0, 3, 7], [0, 7, 4],
            # Right face (x = cx + hw)
            [1, 5, 6], [1, 6, 2],
        ]

        triangles = np.array([[v[f[0]], v[f[1]], v[f[2]]] for f in faces],
                             dtype=np.float64)
        return cls(triangles)
