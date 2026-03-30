import numpy as np


class BuildingModel:
    """3D building model for GNSS ray tracing NLOS detection."""

    def __init__(self, triangles):
        """Initialize with triangle mesh.

        Args:
            triangles: [N, 3, 3] numpy array of triangle vertices.
                       Each triangle has 3 vertices, each vertex has 3 coordinates (x, y, z).
        """
        self.triangles = np.asarray(triangles, dtype=np.float64)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape [N, 3, 3]")

    def check_los(self, rx_ecef, sat_ecef):
        """Check line-of-sight for each satellite.

        Args:
            rx_ecef: [3] receiver ECEF position in meters.
            sat_ecef: [n_sat, 3] satellite ECEF positions in meters.

        Returns:
            is_los: [n_sat] boolean array, True if line-of-sight is clear.
        """
        from gnss_gpu._raytrace import raytrace_los_check

        rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
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

        rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        tri = self.triangles.reshape(-1, 3, 3)

        reflection_points, excess_delays = raytrace_multipath(rx, sat, tri)
        return excess_delays, reflection_points

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
