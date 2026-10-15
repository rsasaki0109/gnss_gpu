"""Small ordered meshes retain the independent stateless GPU reflection bits."""

import numpy as np
import pytest

from test_bvh_large_parallel import _mesh, _rays
from test_bvh_normal_cache import _assert_bits, _leaf, _reference, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('count', [127,128,129,255,256,511,512])
@pytest.mark.parametrize('shape', [(1,16),(129,32),(512,32),(513,32)])
def test_small_order_boundaries_match_stateless_gpu(count,shape):
    nodes,mesh = _mesh(count)
    rx,sat = _rays(*shape)
    scene = native.DeviceBVH(nodes,mesh)
    try:
        expected = _reference(rx,sat,nodes,mesh)
        _assert_bits(scene.trace_batch(rx,sat),expected)
        allocated = scene.allocated_bytes
        _assert_bits(scene.multipath_batch(rx,sat),expected[1:])
        _assert_bits(scene.trace_batch(rx,sat),expected)
        assert scene.allocated_bytes==allocated
    finally:
        scene.close()
    assert scene.allocated_bytes==0


@pytest.mark.parametrize('scale',[1e-100,1e-8,3e-8,1.,1e50,1e100])
def test_small_random_geometry_all_normals_preserve_extreme_bits(scale):
    mesh = np.random.default_rng(317).uniform(-30.,30.,(128,3,3))*scale
    mesh[::17,2] = mesh[::17,0]
    nodes = _leaf(mesh)
    rx,sat = _rays(129,32)
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes,mesh)
    try:
        _assert_bits(scene.trace_batch(rx,sat),_reference(rx,sat,nodes,mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('reverse',[False,True])
def test_small_equal_delay_preserves_right_first_visit(reverse):
    mesh = np.repeat(np.array([
        [[-1.,-20.,-10.],[-1.,20.,-10.],[-1.,0.,20.]],
        [[1.,-20.,-10.],[1.,20.,-10.],[1.,0.,20.]],
    ]),64,axis=0)
    nodes = np.array([
        [-1.,-20.,-10.,1.,20.,20.,1.,2.,0.,0.],
        [-1.,-20.,-10.,-1.,20.,20.,-1.,-1.,0.,64.],
        [1.,-20.,-10.,1.,20.,20.,-1.,-1.,64.,64.],
    ])
    if reverse:
        nodes[0,[6,7]] = nodes[0,[7,6]]
    rx = np.zeros((1,3))
    sat = np.array([[[0.,0.,10.]]])
    scene = native.DeviceBVH(nodes,mesh)
    try:
        expected = _reference(rx,sat,nodes,mesh)
        _assert_bits(scene.trace_batch(rx,sat),expected)
        assert expected[2][0,0]>0.
        assert (expected[1][0,0,0]<0.)==reverse
    finally:
        scene.close()


@pytest.mark.parametrize('count',[127,128,511])
def test_small_geometry_normals_and_winners_are_lazy_exact_and_released(count):
    nodes,mesh = _mesh(count)
    scene = native.DeviceBVH(nodes,mesh)
    try:
        geometry = len(nodes)*64+mesh.nbytes
        assert scene.allocated_bytes==geometry
        rx,sat = _rays(512,32)
        rays = 16384
        scene.check_los_batch(rx,sat)
        base = geometry+rx.nbytes+sat.nbytes+rays*4
        assert scene.allocated_bytes==base
        scene.multipath_batch(rx,sat)
        expected = base+rays*32+((36*count+40*rays) if count>=64 else 0)
        assert scene.allocated_bytes==expected
        assert rays*40==640*1024
    finally:
        scene.close()
    assert scene.allocated_bytes==0


def test_small_ecef_nonfinite_updates_returned_ownership_and_serial_transition():
    nodes,mesh = _mesh(128)
    offset = np.array([6378137.,-4500000.,3500000.])
    nodes[:,:3] += offset
    nodes[:,3:6] += offset
    mesh += offset
    scene = native.DeviceBVH(nodes,mesh)
    retained = []
    try:
        for index,shape in enumerate(((1,16),(129,32),(513,32),(512,32),(1,1))):
            if shape==(1,1):
                rx = np.zeros((1,3))
                sat = np.array([[[0.,0.,100.]]])
            else:
                rx,sat = _rays(*shape,seed=index+951)
            rx += offset
            sat += offset
            if shape!=(1,1):
                sat[-1,0] = np.inf
            output = scene.trace_batch(rx,sat)
            _assert_bits(output,_reference(rx,sat,nodes,mesh))
            retained.append((output,tuple(v.copy() for v in output)))
            for previous,snapshot in retained:
                _assert_bits(previous,snapshot)
    finally:
        scene.close()
