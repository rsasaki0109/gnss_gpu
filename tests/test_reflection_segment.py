"""Check exact segment reuse against the retained pre-reuse implementation."""

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
    return load(Path(__file__).parent/'oracles/reflection_segment_baseline.py', 'reflection_segment_oracle')


def observe_norms(monkeypatch, call):
    original, frames, groups = np.linalg.norm, {}, []

    def observed(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        caller = sys._getframe(1)
        if caller.f_code.co_name == '_segment_intersects_triangle':
            caller = caller.f_back
        if caller.f_code.co_name == '_segment_blocked':
            key = id(caller)
            if key not in frames:
                frames[key] = (caller, len(groups))
                groups.append([])
            groups[frames[key][1]].append((value.tobytes(), np.asarray(result).tobytes()))
        return result

    try:
        with monkeypatch.context() as context:
            context.setattr(np.linalg, 'norm', observed)
            result = call()
    finally:
        frames.clear()
    return result, groups


@pytest.mark.parametrize('kind', ['wall', 'blocked', 'grid'])
@pytest.mark.parametrize('count', [24, 120])
@pytest.mark.parametrize('satellites', [2, 12])
def test_one_identical_norm_per_segment_without_changing_obstacle_results(oracle, monkeypatch, kind, count, satellites):
    old, rx, sat = scene(oracle, count, satellites, kind)
    new = raytrace.BuildingModel(old.triangles.copy())
    a, before = observe_norms(monkeypatch, lambda: old.compute_reflection_paths(rx, sat))
    b, after = observe_norms(monkeypatch, lambda: new.compute_reflection_paths(rx, sat))
    assert tree_hash(a) == tree_hash(b)
    assert len(before) == len(after)
    for original, reused in zip(before, after, strict=True):
        assert len(reused) == 1
        assert all(value == reused[0] for value in original)
    if kind == 'wall':
        assert sum(map(len, a)) > 0
        assert sum(map(len, before)) > sum(map(len, after))


@pytest.mark.parametrize('ground', [False, True])
@pytest.mark.parametrize('distance', [np.nextafter(1e-6, 0.), 1e-6,
    np.nextafter(1e-6, np.inf), 1e-5, 1e-4, 1.])
def test_plane_endpoint_boundaries_and_ground_preserve_all_fields(oracle, ground, distance):
    old = oracle.BuildingModel.create_box([.05, 0., 0.], .1, 20., 20.)
    new = raytrace.BuildingModel(old.triangles.copy())
    rx = np.array([-distance, -3., 1.])
    sat = np.array([[-20200e3, -3., 10000.], [-20200e3, -1000., 20000.]])
    options = dict(max_paths=4, reflection_point_tol_m=.25)
    if ground:
        options['ground_plane'] = ([0., 0., 0.], [0., 0., 1.])
    assert outcome(old, rx, sat, **options) == outcome(new, rx, sat, **options)


@pytest.mark.parametrize('case', ['single', 'float32', 'strict', 'underflow_warn',
    'sat_nan', 'sat_inf', 'sat_bound', 'ground_large'])
def test_fallback_norm_count_order_warnings_and_results_unchanged(oracle, monkeypatch, case):
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
            'sat_bound': np.nextafter(1e50, np.inf)}[case]
    else:
        options['ground_plane'] = ([1e100, 0., 0.], [1., 0., 0.])
    new = raytrace.BuildingModel(np.asarray(old.triangles, dtype=np.float64))
    new.triangles = old.triangles.copy()
    a, before = observe_norms(monkeypatch, lambda: outcome(old, rx, sat, policy, **options))
    b, after = observe_norms(monkeypatch, lambda: outcome(new, rx, sat, policy, **options))
    assert a == b
    assert before == after


def test_segment_data_is_fresh_after_receiver_and_mesh_mutation(oracle):
    old, rx, sat = scene(oracle, 24, 4, 'wall')
    new = raytrace.BuildingModel(old.triangles.copy())
    saved = new.compute_reflection_paths(rx, sat)
    initial = tree_hash(saved)
    for shift in [np.array([.25, .5, .75]), np.array([-.5, .25, -.25])]:
        old.triangles += shift
        new.triangles += shift
        rx += shift*.5
        assert outcome(old, rx, sat) == outcome(new, rx, sat)
        assert tree_hash(saved) == initial
