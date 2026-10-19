"""Exact intermediate/predicate checks against the byte-exact D219 source."""

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bench_double_reflection_invariants_compare import scene
from bench_reflection_geometry_compare import load, tree_hash
from gnss_gpu import double_reflection as current


@pytest.fixture(scope='module')
def oracle():
    return load(Path(__file__).parent/'oracles/double_geometry_baseline.py', 'double_geometry_oracle')


FIELDS = {
    '_compute_single_path': ('rx_plane_distance', 'sat_plane_distance', 'i1', 'i12',
        'denom2', 't2', 'p2', 'denom1', 't1', 'p1', 'seg1', 'seg2', 'seg3', 'path_len', 'excess_delay'),
    '_point_in_triangle': ('v0', 'v1', 'v2', 'd00', 'd01', 'd02', 'd11', 'd12', 'denom', 'inv', 'u', 'v'),
    '_segment_intersects_triangle': ('direction', 'seg_len', 'edge1', 'edge2', 'pvec', 'det',
        'inv_det', 'tvec', 'u', 'qvec', 'v', 't', 'endpoint_tol'),
}


def observe(module, call):
    digests = {name:hashlib.sha256() for name in FIELDS}
    counts = dict.fromkeys(FIELDS, 0)
    order, cache, numerators = [], [], []

    def profile(frame, event, value):
        if frame.f_globals.get('__name__') != module.__name__:
            return
        name, values = frame.f_code.co_name, frame.f_locals
        if event == 'call' and name == '_compute_single_path':
            order.append((values['tri_a_id'], values['tri_b_id']))
            cache.append(values.get('invariants') is not None)
        if event != 'return' or name not in FIELDS:
            return
        counts[name] += 1
        digests[name].update(tree_hash(tuple(values.get(key) for key in FIELDS[name])).encode())
        if name == '_compute_single_path':
            for key, vertex, image, normal in [('numerator1', 'a0', 'i1', 'na'), ('numerator2', 'b0', 'i12', 'nb')]:
                if key in values:
                    actual = values[key]
                    expected = np.dot(values[vertex]-values[image], values[normal])
                    assert type(actual) is np.float64
                    assert actual.tobytes() == expected.tobytes()
                    numerators.append(key)

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = call()
    finally:
        sys.setprofile(previous)
    return result, counts, {name:value.hexdigest() for name, value in digests.items()}, order, cache, numerators


@pytest.mark.parametrize('count', [14, 26])
@pytest.mark.parametrize('satellites', [6, 12])
@pytest.mark.parametrize('kind', ['corner', 'blocked'])
def test_all_intermediate_bits_predicate_order_and_numpy_scalar_numerators(oracle, count, satellites, kind):
    model, rx, sat = scene(count, satellites, kind)
    before = observe(oracle, lambda: oracle.compute_double_reflection_paths(model.triangles, rx, sat))
    after = observe(current, lambda: current.compute_double_reflection_paths(model.triangles, rx, sat))
    assert tree_hash(before[0]) == tree_hash(after[0])
    assert before[1:4] == after[1:4]
    assert all(after[4]) and after[5]
    assert len(after[3]) == count*(count-1)*satellites


@pytest.mark.parametrize('scale', [1e-150, 1e-50, 1., 1e25, 1e50, 1e52])
def test_cross_components_preserve_all_bits_including_cancellation_and_signed_zero(scale):
    rng = np.random.default_rng(220)
    a = rng.normal(size=(2048, 3))*scale
    b = rng.normal(size=(2048, 3))*scale
    b[::3] = a[::3]
    a[::7] = np.copysign(0., a[::7])
    b[::11] = np.copysign(0., b[::11])
    with np.errstate(under='ignore'):
        expected = np.cross(a, b)
        actual = np.array([current._cross3_float64(x, y) for x, y in zip(a, b, strict=True)])
    assert actual.dtype == np.float64 and actual.tobytes() == expected.tobytes()


SEGMENTS = [
    ([.25, .25, 1.], [.25, .25, -1.]),
    ([.25, .25, 1e-6], [.25, .25, -1.+1e-6]),
    ([.25, .25, np.nextafter(1e-6, 0.)], [.25, .25, -1.+1e-6]),
    ([.25, .25, np.nextafter(1e-6, np.inf)], [.25, .25, -1.+1e-6]),
    ([.25, .25, 1.], [.25, .25, 0.]),
    ([.25, .25, -0.], [.25, .25, 1.]),
    ([-1e-9, .25, 1.], [-1e-9, .25, -1.]),
    ([np.nextafter(-1e-9, -np.inf), .25, 1.], [np.nextafter(-1e-9, -np.inf), .25, -1.]),
    ([.75, .25+1e-9, 1.], [.75, .25+1e-9, -1.]),
    ([.25, .25, 1.], [.75, .25, 1.]),
    ([.25, .25, 1e-7], [.25, .25, -1e-7]),
    ([.25, .25, 0.], [.25, .25, 0.]),
]


@pytest.mark.parametrize('start,end', SEGMENTS)
@pytest.mark.parametrize('degenerate', [False, True])
def test_prepared_segment_preserves_endpoint_barycentric_and_early_reject_bits(oracle, start, end, degenerate):
    triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    if degenerate:
        triangle[2] = triangle[0]
    start, end = np.array(start), np.array(end)
    a, b, c = triangle
    geometry = (a, b-a, c-a, current._norm(np.cross(b-a, c-a)))
    direction = end-start
    segment = (direction, current._norm(direction))
    before = observe(oracle, lambda: oracle._segment_intersects_triangle(start, end, triangle))
    after = observe(current, lambda: current._segment_intersects_triangle(start, end, triangle, geometry, segment))
    assert before[:4] == after[:4]


@pytest.mark.parametrize('name,arity', [('_point_in_triangle', 2), ('_segment_blocked', 4), ('_segment_intersects_triangle', 3)])
def test_replaced_predicate_hooks_keep_original_arguments_and_every_call(oracle, monkeypatch, name, arity):
    model, rx, sat = scene(14, 12)
    outputs, visits = [], []
    for module in (oracle, current):
        original, calls = getattr(module, name), []

        def wrapped(*args):
            assert len(args) == arity
            calls.append(tree_hash(args))
            return original(*args)

        with monkeypatch.context() as context:
            context.setattr(module, name, wrapped)
            outputs.append(tree_hash(module.compute_double_reflection_paths(model.triangles, rx, sat)))
        visits.append(calls)
    assert outputs[0] == outputs[1] and visits[0] == visits[1] and visits[0]


def test_replaced_cross_hook_disables_preparation(oracle, monkeypatch):
    model, rx, sat = scene(14, 12)

    def fail(*args):
        raise AssertionError('replaced cross must not be called by the fallback')

    monkeypatch.setattr(current, '_cross3_float64', fail)
    actual = observe(current, lambda: current.compute_double_reflection_paths(model.triangles, rx, sat))
    assert not any(actual[4])
    assert tree_hash(actual[0]) == tree_hash(oracle.compute_double_reflection_paths(model.triangles, rx, sat))


@pytest.mark.parametrize('threshold', [0., 1e-12, 1e-6, 1., np.nan])
def test_segment_threshold_changes_preserve_results_and_predicate_sequence(oracle, monkeypatch, threshold):
    model, rx, sat = scene(14, 12)
    monkeypatch.setattr(oracle, 'intersect_eps', threshold)
    monkeypatch.setattr(current, 'intersect_eps', threshold)
    before = observe(oracle, lambda: oracle.compute_double_reflection_paths(model.triangles, rx, sat))
    after = observe(current, lambda: current.compute_double_reflection_paths(model.triangles, rx, sat))
    assert tree_hash(before[0]) == tree_hash(after[0])
    assert before[1:4] == after[1:4]
