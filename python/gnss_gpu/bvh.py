import numpy as np


def _single_rays(rx_ecef, sat_ecef):
    rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    if rx.size != 3:
        raise ValueError("rx_ecef must have shape (3,)")
    if sat.shape[0] == 0:
        raise ValueError("sat_ecef must contain at least one satellite")
    if not np.all(np.isfinite(rx)) or not np.all(np.isfinite(sat)):
        raise ValueError("rx_ecef and sat_ecef must be finite")
    return rx, sat


def _batch_rays(rx_ecef, sat_ecef):
    rx = np.ascontiguousarray(np.asarray(rx_ecef, dtype=np.float64))
    sat = np.ascontiguousarray(np.asarray(sat_ecef, dtype=np.float64))
    if rx.ndim != 2 or rx.shape[1] != 3:
        raise ValueError("rx_ecef must have shape (N, 3)")
    if sat.ndim != 3 or sat.shape[2] != 3:
        raise ValueError("sat_ecef must have shape (N, n_sat, 3)")
    if rx.shape[0] != sat.shape[0]:
        raise ValueError("rx_ecef and sat_ecef must share the leading N")
    if rx.shape[0] == 0 or sat.shape[1] == 0:
        raise ValueError("rx_ecef and sat_ecef must have n_epoch,n_sat >= 1")
    return rx, sat


class BVHAccelerator:
    """BVH-accelerated ray tracing for GNSS NLOS detection.

    Wraps BuildingModel to provide O(log n) ray-triangle intersection
    instead of O(n) linear scan. Useful for urban meshes with 10K+ triangles.

    Geometry is an owned, read-only snapshot. GPU geometry and work buffers are
    uploaded lazily and reused. Use ``update_triangles`` to replace the mesh and
    ``release_gpu`` to release device memory (the next call uploads it again).

    **Empty mesh:** construction requires at least one triangle. An empty
    `(0, 3, 3)` mesh raises ``ValueError`` — this is intentional (fail-fast
    for accidental empty CityGML loads). For open-sky / no-occluder scenarios,
    use ``BuildingModel`` linear scan (zero triangles ⇒ all LOS) or call
    ``pf_weight_3d_bvh`` directly with empty BVH buffers. See
    ``docs/common_input_shapes.md`` (Raytrace / BVH).
    """

    def __init__(self, triangles):
        """Build BVH from triangle mesh.

        Args:
            triangles: [N, 3, 3] numpy array of triangle vertices.
        """
        self._device_scene = None
        self.update_triangles(triangles)

    def update_triangles(self, triangles):
        """Rebuild from a mesh snapshot; invalid input leaves this BVH intact.

        GPU data is released after the new CPU BVH is built successfully and
        uploaded on the next ray call. Input arrays can be changed afterwards.
        """
        mesh = np.array(triangles, dtype=np.float64, order="C", copy=True)
        if mesh.ndim != 3 or mesh.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape [N, 3, 3]")
        if mesh.shape[0] == 0:
            raise ValueError("triangles must contain at least one triangle")
        if not np.all(np.isfinite(mesh)):
            raise ValueError("triangles must be finite")

        from gnss_gpu._bvh import bvh_build

        nodes_flat, sorted_indices = bvh_build(mesh)
        sorted_tris = mesh[sorted_indices].copy()
        for array in (mesh, nodes_flat, sorted_indices, sorted_tris):
            array.setflags(write=False)
        self.release_gpu()
        self.triangles = mesh
        self._nodes_flat = nodes_flat
        self._sorted_indices = sorted_indices
        self._sorted_tris = sorted_tris

    def release_gpu(self):
        """Release cached GPU geometry/work buffers; keep the CPU mesh usable."""
        if self._device_scene is not None:
            self._device_scene.close()
            self._device_scene = None

    def _gpu_scene(self):
        if self._device_scene is None:
            from gnss_gpu._bvh import DeviceBVH

            self._device_scene = DeviceBVH(self._nodes_flat, self._sorted_tris)
        return self._device_scene

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_device_scene"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._device_scene = None
        for array in (self.triangles, self._nodes_flat, self._sorted_indices, self._sorted_tris):
            array.setflags(write=False)

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
        rx, sat = _single_rays(rx_ecef, sat_ecef)
        return self._gpu_scene().check_los(rx, sat)

    def compute_multipath(self, rx_ecef, sat_ecef):
        """Compute first-order multipath reflections using BVH traversal.

        Args:
            rx_ecef: [3] receiver ECEF position in meters.
            sat_ecef: [n_sat, 3] satellite ECEF positions in meters.

        Returns:
            excess_delays: [n_sat] excess path delay in meters (0 if no reflection).
            reflection_points: [n_sat, 3] reflection point coordinates.
        """
        rx = np.ascontiguousarray(np.asarray(rx_ecef, dtype=np.float64).reshape(1, 3))
        sat = np.ascontiguousarray(np.asarray(sat_ecef, dtype=np.float64).reshape(1, -1, 3))

        excess_delays, reflection_points = self.compute_multipath_batch(rx, sat)
        return excess_delays[0], reflection_points[0]

    def check_los_batch(self, rx_ecef, sat_ecef):
        """Batched LOS check across N epochs sharing this BVH.

        One CUDA launch handles all N x n_sat rays.  Non-finite (NaN/Inf)
        entries return False so callers can pad variable-length epochs.

        Args:
            rx_ecef: [N, 3] receiver ECEF positions in meters.
            sat_ecef: [N, n_sat, 3] satellite ECEF positions in meters.
                Use the same n_sat for every epoch (pad with NaN if needed).

        Returns:
            is_los: [N, n_sat] boolean array.
        """
        rx, sat = _batch_rays(rx_ecef, sat_ecef)
        return self._gpu_scene().check_los_batch(rx, sat)

    def compute_multipath_batch(self, rx_ecef, sat_ecef):
        """Batched multipath reflection across N epochs sharing this BVH.

        GPU kernels share the resident geometry. Non-finite (NaN/Inf)
        entries yield zero delay and zero reflection point.

        Args:
            rx_ecef: [N, 3] receiver ECEF positions in meters.
            sat_ecef: [N, n_sat, 3] satellite ECEF positions in meters.

        Returns:
            excess_delays: [N, n_sat] excess path delay in meters.
            reflection_points: [N, n_sat, 3] reflection point coordinates.
        """
        rx, sat = _batch_rays(rx_ecef, sat_ecef)
        reflection_points, excess_delays = self._gpu_scene().multipath_batch(rx, sat)
        return excess_delays, reflection_points

    def trace(self, rx_ecef, sat_ecef):
        """Return LOS and first-order reflections with one ray upload.

        Inputs are finite receiver [3] and satellite [n_sat, 3] positions in
        meters. Returns ``(is_los, excess_delays, reflection_points)`` with
        shapes [n_sat], [n_sat], [n_sat, 3]. Results match ``check_los`` followed
        by ``compute_multipath``; rays are uploaded and synchronized once.
        """
        rx, sat = _single_rays(rx_ecef, sat_ecef)
        is_los, points, delays = self._gpu_scene().trace(rx, sat)
        return is_los, delays, points

    def trace_batch(self, rx_ecef, sat_ecef):
        """Return batched LOS and first-order reflections with one ray upload.

        Inputs are receiver [N, 3] and satellite [N, n_sat, 3] positions in
        meters. Returns ``(is_los, excess_delays, reflection_points)`` with
        shapes [N, n_sat], [N, n_sat], [N, n_sat, 3]. Results match the separate
        batch calls. NaN/Inf padding returns False LOS and zero delay/point.
        """
        rx, sat = _batch_rays(rx_ecef, sat_ecef)
        is_los, points, delays = self._gpu_scene().trace_batch(rx, sat)
        return is_los, delays, points

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
