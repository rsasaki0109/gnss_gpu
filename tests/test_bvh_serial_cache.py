"""Large resident batches match the retained stateless reflection GPU oracle."""

import numpy as np
import pytest

from test_bvh_normal_cache import _assert_bits, _leaf, _mesh, _reference, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _rays(epochs, satellites, seed=82):
    rng=np.random.default_rng(seed)
    return rng.uniform(-30.,30.,(epochs,3)),rng.uniform(-300.,300.,(epochs,satellites,3))


@pytest.mark.parametrize('count',[511,512,513,1200])
@pytest.mark.parametrize('shape',[(1,4097),(129,32),(257,17)])
def test_large_batch_boundaries_exact_uncached_reference(count,shape):
    nodes,mesh=_mesh(count)
    rx,sat=_rays(*shape)
    expected=_reference(rx,sat,nodes,mesh)
    scene=native.DeviceBVH(nodes,mesh)
    try:
        _assert_bits(scene.trace_batch(rx,sat),expected)
        allocated=scene.allocated_bytes
        _assert_bits(scene.trace_batch(rx,sat),expected)
        _assert_bits(scene.multipath_batch(rx,sat),expected[1:])
        assert scene.allocated_bytes==allocated
    finally:
        scene.close()
    assert scene.allocated_bytes==0


@pytest.mark.parametrize('scale',[1e-100,1e-8,3e-8,1e-7,1.,1e50,1e100])
def test_extreme_normals_in_serial_cache_keep_bits(scale):
    mesh=np.random.default_rng(137).uniform(-1.,1.,(513,3,3))*scale
    mesh[::17,2]=mesh[::17,0]
    nodes=_leaf(mesh)
    rx,sat=_rays(1,4097)
    rx*=scale
    sat*=scale
    scene=native.DeviceBVH(nodes,mesh)
    try:
        _assert_bits(scene.trace_batch(rx,sat),_reference(rx,sat,nodes,mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('area',[np.nextafter(1e-15,0.),1e-15,np.nextafter(1e-15,np.inf)])
def test_serial_cache_strict_normal_cutoff_and_signed_zero(area):
    mesh=np.repeat(np.array([[[0.,0.,0.],[1.,0.,0.],[0.,area,0.]]]),513,axis=0)
    mesh[::2,:,2]=-0.
    nodes=_leaf(mesh)
    rx=np.array([[.25,area/4.,1.]])
    sat=np.repeat(np.array([[[.25,area/4.,10.]]]),4097,axis=1)
    scene=native.DeviceBVH(nodes,mesh)
    try:
        _assert_bits(scene.trace_batch(rx,sat),_reference(rx,sat,nodes,mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('reverse',[False,True])
def test_serial_equal_delays_preserve_right_first_order(reverse):
    mesh=np.repeat(np.array([
        [[-1.,-20.,-10.],[-1.,20.,-10.],[-1.,0.,20.]],
        [[1.,-20.,-10.],[1.,20.,-10.],[1.,0.,20.]],
    ]),256,axis=0)
    nodes=np.array([
        [-1.,-20.,-10.,1.,20.,20.,1.,2.,0.,0.],
        [-1.,-20.,-10.,-1.,20.,20.,-1.,-1.,0.,256.],
        [1.,-20.,-10.,1.,20.,20.,-1.,-1.,256.,256.],
    ])
    if reverse:
        nodes[0,[6,7]]=nodes[0,[7,6]]
    rx=np.zeros((1,3))
    sat=np.repeat(np.array([[[0.,0.,10.]]]),4097,axis=1)
    scene=native.DeviceBVH(nodes,mesh)
    try:
        expected=_reference(rx,sat,nodes,mesh)
        _assert_bits(scene.trace_batch(rx,sat),expected)
        assert np.all(expected[2]>0.)
        assert np.all((expected[1][...,0]<0.)==reverse)
    finally:
        scene.close()


def test_serial_cache_ecef_padding_and_receiver_changes():
    nodes,mesh=_mesh(513)
    offset=np.array([6378137.,-4500000.,3500000.])
    nodes[:,:3]+=offset
    nodes[:,3:6]+=offset
    mesh+=offset
    scene=native.DeviceBVH(nodes,mesh)
    try:
        for epoch in range(3):
            rx,sat=_rays(129,32,seed=83+epoch)
            rx+=offset
            sat+=offset
            rx[1]=np.nan
            sat[2,:4]=np.inf
            sat[3,0]=rx[3]
            _assert_bits(scene.trace_batch(rx,sat),_reference(rx,sat,nodes,mesh))
    finally:
        scene.close()


def test_parallel_serial_transitions_snapshot_and_returned_ownership():
    nodes,mesh=_mesh(513)
    inputs=[]
    for index,shape in enumerate(((1,4097),(1,16),(129,32),(1,1),(257,17),(1,4096))):
        rx,sat=_rays(*shape,seed=91+index)
        inputs.append((rx,sat,_reference(rx,sat,nodes,mesh)))
    scene=native.DeviceBVH(nodes,mesh)
    nodes.fill(0.)
    mesh.fill(0.)
    retained=[]
    try:
        for rx,sat,expected in inputs:
            output=scene.trace_batch(rx,sat)
            _assert_bits(output,expected)
            retained.append((output,tuple(v.copy() for v in output)))
            for previous,snapshot in retained:
                _assert_bits(previous,snapshot)
    finally:
        scene.close()


@pytest.mark.parametrize('count',[511,513])
def test_reflection_cache_is_lazy_accounted_and_released(count):
    nodes,mesh=_mesh(count)
    scene=native.DeviceBVH(nodes,mesh)
    try:
        geometry=len(nodes)*64+mesh.nbytes
        assert scene.allocated_bytes==geometry
        rx,sat=_rays(1,4097)
        scene.check_los_batch(rx,sat)
        assert scene.allocated_bytes==geometry+rx.nbytes+sat.nbytes+4097*4
        scene.multipath_batch(rx,sat)
        assert scene.allocated_bytes==geometry+rx.nbytes+sat.nbytes+4097*(4+3*8+8)+((36*count+40*4097) if count>=64 else 0)
    finally:
        scene.close()
    assert scene.allocated_bytes==0


@pytest.mark.parametrize('kind',['deep','overlap','shared_leaf'])
def test_serial_stack_guard_and_unordered_fallback_match(kind):
    mesh=np.random.default_rng(74).uniform(-100.,100.,(512,3,3))
    if kind=='deep':
        nodes=np.zeros((129,10))
        nodes[:,:3]=-100.
        nodes[:,3:6]=100.
        nodes[:,6:8]=-1
        for depth in range(64):
            node,leaf=2*depth,2*depth+1
            nodes[node,6:8]=[leaf,node+2]
            nodes[leaf,8:10]=[depth*8,8]
        nodes[128,8:10]=[504,8]
    elif kind=='overlap':
        nodes=np.array([
            [-100.,-100.,-100.,100.,100.,100.,1.,2.,0.,0.],
            [-100.,-100.,-100.,100.,100.,100.,-1.,-1.,0.,512.],
            [-100.,-100.,-100.,100.,100.,100.,-1.,-1.,0.,512.],
        ])
    else:
        nodes=np.array([
            [-100.,-100.,-100.,100.,100.,100.,1.,1.,0.,0.],
            [-100.,-100.,-100.,100.,100.,100.,-1.,-1.,0.,512.],
        ])
    rx,sat=_rays(1,4097)
    scene=native.DeviceBVH(nodes,mesh)
    try:
        _assert_bits(scene.trace_batch(rx,sat),_reference(rx,sat,nodes,mesh))
    finally:
        scene.close()
