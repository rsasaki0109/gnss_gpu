"""Compare detailed physical paths with the retained pre-cache scalar oracle."""

import copy
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bench_reflection_geometry_compare import load, scene, tree_hash
from gnss_gpu import raytrace


@pytest.fixture(scope='module')
def oracle():
    return load(Path(__file__).parent/'oracles/reflection_geometry_scalar.py', 'reflection_scalar_oracle')


def outcome(model, rx, sat, policy='default', **kwargs):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        settings = np.geterr() if policy == 'default' else dict.fromkeys(np.geterr(), policy)
        with np.errstate(**settings):
            try:
                result = model.compute_reflection_paths(rx, sat, **kwargs)
                value = ('return', tree_hash(result))
            except (ValueError, RuntimeError, FloatingPointError, TypeError, OverflowError) as exc:
                value = ('error', type(exc).__name__, str(exc))
    return value, [(type(row.message).__name__, str(row.message)) for row in recorded]


@pytest.mark.parametrize('kind', ['wall', 'blocked', 'grid'])
@pytest.mark.parametrize('count', [12, 24, 120])
@pytest.mark.parametrize('satellites', [1, 2, 12])
def test_paths_all_fields_and_stable_sort_match_scalar(oracle, kind, count, satellites):
    old, rx, sat = scene(oracle, count, satellites, kind)
    new = raytrace.BuildingModel(old.triangles.copy())
    inputs = tree_hash((old.triangles, rx, sat))
    a, b = old.compute_reflection_paths(rx, sat), new.compute_reflection_paths(rx, sat)
    assert tree_hash(a) == tree_hash(b)
    assert tree_hash((old.triangles, rx, sat)) == inputs
    assert tree_hash(new.triangles) == tree_hash(old.triangles)
    if kind == 'wall':
        assert sum(map(len, a)) > 0
    for paths in b:
        for path in paths:
            assert path.normal.flags.owndata and path.reflection_point.flags.owndata


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
@pytest.mark.parametrize('layout', ['C', 'strided', 'fortran'])
def test_replaced_geometry_dtype_and_layout_preserve_original_outputs(oracle, dtype, layout):
    old, rx, sat = scene(oracle, 12, 2, 'wall')
    mesh = old.triangles.astype(dtype)
    if layout == 'strided':
        mesh = np.repeat(mesh, 2, axis=0)[::2, ::-1]
    elif layout == 'fortran':
        mesh = np.asfortranarray(mesh)
    new = raytrace.BuildingModel(old.triangles.copy())
    old.triangles, new.triangles = mesh.copy(order='K'), mesh.copy(order='K')
    if layout == 'strided':
        old.triangles = np.repeat(old.triangles, 2, axis=0)[::2]
        new.triangles = np.repeat(new.triangles, 2, axis=0)[::2]
    assert outcome(old, rx, sat) == outcome(new, rx, sat)


def test_replaced_array_subclass_preserves_original_first_operation(oracle):
    class ObservedArray(np.ndarray):
        def __array_ufunc__(self, function, method, *args, **kwargs):
            raise RuntimeError(function.__name__)

    old, rx, sat = scene(oracle, 12, 2, 'wall')
    new = raytrace.BuildingModel(old.triangles.copy())
    old.triangles = old.triangles.view(ObservedArray)
    new.triangles = new.triangles.view(ObservedArray)
    assert outcome(old, rx, sat) == outcome(new, rx, sat)


@pytest.mark.parametrize('policy', ['default', 'ignore', 'warn', 'raise'])
@pytest.mark.parametrize('scale', [1e-160, 1e-16, 1., 1e50, np.nextafter(1e50, np.inf), 1e100])
def test_extreme_coordinates_and_warning_exception_order_match_scalar(oracle, policy, scale):
    rng = np.random.default_rng(215)
    mesh = rng.normal(size=(8, 3, 3))*scale
    mesh[0] = 0.
    mesh[1, 2] = mesh[1, 0]
    rx, sat = np.array([3., 4., 5.])*scale, rng.normal(size=(2, 3))*scale*20
    old, new = oracle.BuildingModel(mesh.copy()), raytrace.BuildingModel(mesh.copy())
    assert outcome(old, rx, sat, policy) == outcome(new, rx, sat, policy)


@pytest.mark.parametrize('where', ['mesh', 'receiver', 'satellite'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_replaced_inputs_keep_warning_and_error_behavior(oracle, where, value):
    old, rx, sat = scene(oracle, 12, 2, 'wall')
    new = raytrace.BuildingModel(old.triangles.copy())
    if where == 'mesh':
        old.triangles[0, 0, 0] = new.triangles[0, 0, 0] = value
    elif where == 'receiver':
        rx[0] = value
    else:
        sat[0, 0] = value
    assert outcome(old, rx, sat) == outcome(new, rx, sat)


@pytest.mark.parametrize('limit', [0, 1, 4])
@pytest.mark.parametrize('snap', [0., .25])
@pytest.mark.parametrize('ground', [False, True])
def test_snapping_ground_plane_and_max_paths_still_match_scalar(oracle, limit, snap, ground):
    old, rx, sat = scene(oracle, 12, 2, 'wall')
    new = raytrace.BuildingModel(old.triangles.copy())
    options = dict(max_paths=limit, reflection_point_tol_m=snap)
    if ground:
        options['ground_plane'] = (rx-np.array([1.5, 0., 0.]), np.array([1., 0., 0.]))
    assert outcome(old, rx, sat, **options) == outcome(new, rx, sat, **options)


def test_call_local_mesh_receiver_changes_and_returned_storage(oracle):
    old, rx, sat = scene(oracle, 12, 2, 'wall')
    new = raytrace.BuildingModel(old.triangles.copy())
    saved = new.compute_reflection_paths(rx, sat)
    original = tree_hash(saved)
    assert original == tree_hash(old.compute_reflection_paths(rx, sat))
    for shift in [np.array([0., .5, .25]), np.array([.25, 0., .5])]:
        old.triangles += shift
        new.triangles += shift
        rx += shift*.5
        assert outcome(old, rx, sat) == outcome(new, rx, sat)
        assert tree_hash(saved) == original
    for group in saved:
        for path in group:
            path.normal[:] = 99.
            path.reflection_point[:] = -99.
    assert outcome(old, rx, sat) == outcome(new, rx, sat)


@pytest.mark.parametrize('count', [0, 1, 7, 8])
@pytest.mark.parametrize('satellites', [0, 1, 2])
def test_empty_and_geometry_reuse_lower_bound(oracle, count, satellites):
    rng = np.random.default_rng(215)
    mesh, sat = rng.normal(size=(count, 3, 3)), rng.normal(size=(satellites, 3))*100
    old, new = oracle.BuildingModel(mesh.copy()), raytrace.BuildingModel(mesh.copy())
    assert outcome(old, [0., 0., 1.], sat) == outcome(new, [0., 0., 1.], sat)


@pytest.mark.parametrize('count', [(32*1024*1024)//113, (32*1024*1024)//113+1])
def test_cache_payload_upper_bound_before_first_triangle(count, monkeypatch):
    mesh = np.zeros((count, 3, 3))
    model = raytrace.BuildingModel(mesh)
    allocate, calls = np.empty, []

    def allocation(shape, *args, **kwargs):
        calls.append(shape)
        return allocate(shape, *args, **kwargs)

    def stop(*args, **kwargs):
        raise RuntimeError('first triangle reached')

    monkeypatch.setattr(np, 'empty', allocation)
    monkeypatch.setattr(np, 'cross', stop)
    with pytest.raises(RuntimeError, match='first triangle reached'):
        model.compute_reflection_paths([0., 0., 1.], [[1., 2., 3.], [2., 3., 4.]])
    assert ((count, 14) in calls) == (count*113 <= 32*1024*1024)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize('samples', [31, 2600])
@pytest.mark.parametrize('satellites', [1, 2, 12])
def test_real_cuda_LOS_IQ_and_every_metadata_field_match_scalar(oracle, samples, satellites):
    pytest.importorskip('gnss_gpu._raytrace')
    pytest.importorskip('gnss_gpu._gnss_gpu_signal_sim')
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
    sims, results = [], []
    try:
        for module in (oracle, raytrace):
            model, rx, sat = scene(module, 12, satellites, 'wall')
            sim = UrbanSignalSimulator(model, elevation_mask_deg=0., max_reflection_paths=4)
            sim.sim.noise_seed = 37
            sims.append(sim)
            results.append(sim.compute_epoch(rx, sat, n_samples=samples))
        assert tree_hash(results[0]) == tree_hash(results[1])
        assert results[0]['n_reflection_paths'] > 0
    finally:
        for sim in sims:
            sim.sim.release_gpu()
