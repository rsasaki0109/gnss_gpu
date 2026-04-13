import numpy as np


class BVHAccelerator:
    """BVH-accelerated ray tracing for GNSS NLOS detection.

    Wraps BuildingModel to provide O(log n) ray-triangle intersection
    instead of O(n) linear scan. Useful for urban meshes with 10K+ triangles.
    """

    def __init__(self, triangles):
        """Build BVH from triangle mesh.

        Args:
            triangles: [N, 3, 3] numpy array of triangle vertices.
        """
        self.triangles = np.asarray(triangles, dtype=np.float64)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape [N, 3, 3]")

        from gnss_gpu._bvh import bvh_build

        nodes_flat, sorted_indices = bvh_build(self.triangles)
        self._nodes_flat = nodes_flat
        self._sorted_indices = sorted_indices

        # Reorder triangles according to BVH sorted order
        self._sorted_tris = self.triangles[sorted_indices].copy()

    @property
    def n_nodes(self):
        """Number of BVH nodes."""
        return self._nodes_flat.shape[0]

    @property
    def n_triangles(self):
        """Number of triangles."""
        return self.triangles.shape[0]

    def check_los(self, rx_ecef, sat_ecef):
        """Check line-of-sight for each satellite using BVH traversal.

        Args:
            rx_ecef: [3] receiver ECEF position in meters.
            sat_ecef: [n_sat, 3] satellite ECEF positions in meters.

        Returns:
            is_los: [n_sat] boolean array, True if line-of-sight is clear.
        """
        from gnss_gpu._bvh import raytrace_los_check_bvh

        rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)

        return raytrace_los_check_bvh(rx, sat, self._nodes_flat,
                                       self._sorted_tris)

    def compute_multipath(self, rx_ecef, sat_ecef):
        """Compute first-order multipath reflections using BVH traversal.

        Args:
            rx_ecef: [3] receiver ECEF position in meters.
            sat_ecef: [n_sat, 3] satellite ECEF positions in meters.

        Returns:
            excess_delays: [n_sat] excess path delay in meters (0 if no reflection).
            reflection_points: [n_sat, 3] reflection point coordinates.
        """
        from gnss_gpu._bvh import raytrace_multipath_bvh

        rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)

        reflection_points, excess_delays = raytrace_multipath_bvh(
            rx, sat, self._nodes_flat, self._sorted_tris)
        return excess_delays, reflection_points

    @classmethod
    def from_building_model(cls, building_model):
        """Create BVH accelerator from an existing BuildingModel.

        Args:
            building_model: a gnss_gpu.raytrace.BuildingModel instance.

        Returns:
            BVHAccelerator instance.
        """
        return cls(building_model.triangles)

    @classmethod
    def from_obj(cls, filepath):
        """Load building model from OBJ file and build BVH.

        Args:
            filepath: path to Wavefront OBJ file.

        Returns:
            BVHAccelerator instance.
        """
        from gnss_gpu.raytrace import BuildingModel
        model = BuildingModel.from_obj(filepath)
        return cls(model.triangles)

    @classmethod
    def create_box(cls, center, width, depth, height):
        """Create a box building and build BVH.

        Args:
            center: [3] geometric center.
            width: box width along x-axis.
            depth: box depth along y-axis.
            height: box height along z-axis.

        Returns:
            BVHAccelerator instance.
        """
        from gnss_gpu.raytrace import BuildingModel
        model = BuildingModel.create_box(center, width, depth, height)
        return cls(model.triangles)
