"""Verify bounded three-vector cross products against the adopted scalar code."""

import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bench_reflection_geometry_compare import load, scene, tree_hash
from gnss_gpu import raytrace
from test_reflection_geometry import outcome


@pytest.fixture(scope='module')
def oracle():
    return load(Path(__file__).parent/'oracles/reflection_cross_baseline.py', 'reflection_cross_oracle')


@pytest.mark.parametrize('layout', ['C', 'fortran', 'reverse'])
@pytest.mark.parametrize('scale', [1e-160, 1e-100, 1., 1e50])
def test_cross_components_preserve_all_bits_for_finite_vectors(layout, scale):
    rng = np.random.default_rng(216)
    a, b = rng.uniform(-1., 1., (2, 1024, 3))*scale
    if layout == 'fortran':
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif layout == 'reverse':
        a, b = a[::-1, ::-1], b[::-1, ::-1]
    with np.errstate(under='ignore'):
        expected = np.cross(a, b)
        actual = np.array([raytrace._cross3_float64(x, y) for x, y in zip(a, b, strict=True)])
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_cross_signed_zero_and_cancellation_bits():
    vectors = np.array(list(itertools.product([-0., 0., -1., 1.], repeat=6)))
    expected = np.cross(vectors[:, :3], vectors[:, 3:])
    actual = np.array([raytrace._cross3_float64(v[:3], v[3:]) for v in vectors])
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('ground', [False, True])
def test_transformed_physical_paths_match_previous_code(oracle, seed, ground):
    old, rx, sat = scene(oracle, 24, 4, 'wall')
    rng = np.random.default_rng(seed)
    order = rng.permutation(3)
    signs = rng.choice([-1., 1.], 3)
    offset = rng.uniform(-100., 100., 3)
    scale = [1e-6, 1., 1e20, 1e40][seed % 4]
    def transform(x):
        return (x[..., order]*signs+offset)*scale

    old.triangles, rx, sat = transform(old.triangles), transform(rx), transform(sat)
    new = raytrace.BuildingModel(old.triangles.copy())
    options = dict(reflection_point_tol_m=.25*scale)
    if ground:
        normal = np.eye(3)[int(np.flatnonzero(order == 0)[0])]*signs
        options['ground_plane'] = (rx-1.5*scale*normal, normal)
    assert outcome(old, rx, sat, **options) == outcome(new, rx, sat, **options)


@pytest.mark.parametrize('case', ['single', 'float32', 'strict', 'underflow_warn',
    'sat_nan', 'sat_inf', 'sat_bound', 'sat_large', 'ground_nan', 'ground_inf',
    'ground_large', 'ground_normal_nan', 'ground_normal_inf'])
def test_unsafe_and_exceptional_inputs_keep_original_cross(oracle, monkeypatch, case):
    old, rx, sat = scene(oracle, 12, 2, 'wall')
    policy, options = 'default', {}
    if case == 'single':
        sat = sat[:1]
    elif case == 'float32':
        old.triangles = old.triangles.astype(np.float32)
    elif case == 'strict':
        policy = 'raise'
    elif case == 'underflow_warn':
        policy = 'warn'
    elif case.startswith('sat_'):
        sat[0, 0] = {'sat_nan': np.nan, 'sat_inf': np.inf,
            'sat_bound': np.nextafter(1e50, np.inf), 'sat_large': 1e200}[case]
    else:
        point, normal = rx-np.array([1.5, 0., 0.]), np.array([1., 0., 0.])
        if case.startswith('ground_normal_'):
            normal[0] = np.nan if case.endswith('nan') else np.inf
        else:
            point[0] = {'ground_nan': np.nan, 'ground_inf': np.inf, 'ground_large': 1e100}[case]
        options['ground_plane'] = (point, normal)
    new = raytrace.BuildingModel(np.asarray(old.triangles, dtype=np.float64))
    new.triangles = old.triangles.copy()

    def unexpected(*args):
        raise AssertionError('unsafe inputs used bounded cross')

    monkeypatch.setattr(raytrace, '_cross3_float64', unexpected)
    assert outcome(old, rx, sat, policy, **options) == outcome(new, rx, sat, policy, **options)


def test_valid_physical_scene_uses_bounded_cross_and_owned_results(oracle, monkeypatch):
    old, rx, sat = scene(oracle, 12, 2, 'wall')
    new = raytrace.BuildingModel(old.triangles.copy())
    original, calls = raytrace._cross3_float64, []

    def observed(a, b):
        value = original(a, b)
        assert value.flags.owndata
        calls.append(1)
        return value

    monkeypatch.setattr(raytrace, '_cross3_float64', observed)
    expected = old.compute_reflection_paths(rx, sat)
    actual = new.compute_reflection_paths(rx, sat)
    assert sum(map(len, expected)) > 0 and calls
    assert tree_hash(expected) == tree_hash(actual)
