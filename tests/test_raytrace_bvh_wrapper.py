"""CPU-side validation tests for raytrace and BVH wrappers."""

import numpy as np
import pytest

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.raytrace import BuildingModel


def test_building_model_rejects_invalid_triangles():
    with pytest.raises(ValueError, match="triangles must have shape"):
        BuildingModel(np.zeros((3, 2, 3)))
    with pytest.raises(ValueError, match="triangles must be finite"):
        BuildingModel(np.array([[[0, 0, 0], [1, 0, 0], [np.nan, 1, 0]]]))


def test_building_model_rejects_invalid_los_inputs():
    model = BuildingModel.create_box(center=[0, 0, 5], width=10, depth=10, height=10)
    sat = np.array([[0.0, 0.0, 2.0e7]], dtype=np.float64)

    with pytest.raises(ValueError, match="rx_ecef must have shape"):
        model.check_los([0, 0], sat)
    with pytest.raises(ValueError, match="sat_ecef must contain at least one satellite"):
        model.check_los([0, 0, 0], np.zeros((0, 3)))
    with pytest.raises(ValueError, match="rx_ecef must be finite"):
        model.check_los([np.nan, 0, 0], sat)


def test_bvh_accelerator_rejects_empty_mesh():
    with pytest.raises(ValueError, match="triangles must contain at least one triangle"):
        BVHAccelerator(np.zeros((0, 3, 3)))


def test_bvh_check_los_rejects_invalid_inputs():
    bvh = BVHAccelerator.from_building_model(
        BuildingModel.create_box(center=[100, 0, 25], width=20, depth=20, height=50)
    )
    sat = np.array([[0.0, 0.0, 2.0e7]], dtype=np.float64)

    with pytest.raises(ValueError, match="rx_ecef must have shape"):
        bvh.check_los([0, 0], sat)
    with pytest.raises(ValueError, match="sat_ecef must contain at least one satellite"):
        bvh.check_los([0, 0, 0], np.zeros((0, 3)))
    with pytest.raises(ValueError, match="rx_ecef and sat_ecef must be finite"):
        bvh.check_los([np.inf, 0, 0], sat)


def test_bvh_batch_rejects_empty_epoch():
    bvh = BVHAccelerator.from_building_model(
        BuildingModel.create_box(center=[100, 0, 25], width=20, depth=20, height=50)
    )
    with pytest.raises(ValueError, match="n_epoch,n_sat >= 1"):
        bvh.check_los_batch(np.zeros((0, 3)), np.zeros((0, 1, 3)))
