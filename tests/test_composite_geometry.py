"""Exact composite geometry reuse against the independent pre-change source."""

from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bench_composite_geometry_compare import make_call,scene
from bench_reflection_geometry_compare import load,tree_hash
from gnss_gpu import reflection_diffraction as current


@pytest.fixture(scope='module')
def oracle():
    return load(Path(__file__).parent/'oracles/composite_geometry_scalar.py','composite_geometry_oracle')


def trace(module,triangles,edges,rx,sat,**kwargs):
    geometry,points,builders,loss,counts = [],[],[],[],{}

    def profile(frame,event,result):
        if event != 'return' or frame.f_globals.get('__name__') != module.__name__:
            return
        name,values = frame.f_code.co_name.removesuffix('_prepared'),frame.f_locals
        counts[name] = counts.get(name,0)+1
        if name == '_diffraction_geometry':
            fields = ('source','target','edge_start','edge_end','direction','ray_len','edge_vec','w','b','c','d','e','denom','t_edge','point','rel','along_ray','closest_on_ray','h','d1','d2')
            geometry.append(tree_hash(([(field,values[field]) for field in fields if field in values],result)))
        elif name == '_point_in_triangle':
            fields = ('point','p0','p1','p2','v0','v1','v2','dot00','dot01','dot02','dot11','dot12','denom','inv_denom','u','v')
            points.append(tree_hash(([(field,values[field]) for field in fields if field in values],result)))
        elif name in ('_build_rd_path','_build_dr_path'):
            builders.append(tree_hash((name,values['triangle_id'],values['edge_id'],result)))
        elif name in ('fresnel_v','knife_edge_loss_db','knife_edge_amplitude'):
            loss.append(tree_hash((name,result)))
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = module.compute_reflection_diffraction_paths(triangles,edges,rx,sat,**kwargs)
    finally:
        sys.setprofile(previous)
    return (tree_hash(result),geometry,points,builders,loss),counts,result


@pytest.mark.parametrize('triangles,edges,satellites',[(1,1,1),(2,16,4),(2,16,12),(6,64,4),(6,64,12)])
@pytest.mark.parametrize('orders',[('RD',),('DR',),('RD','DR')])
def test_all_candidates_intermediates_and_returned_bits(oracle,triangles,edges,satellites,orders):
    model,e,rx,sat = scene(triangles,edges,satellites)
    before = trace(oracle,model.triangles,e,rx,sat,orders=orders,max_paths=100000)
    after = trace(current,model.triangles,e,rx,sat,orders=orders,max_paths=100000)
    assert before[0] == after[0]
    assert len(after[0][1]) == triangles*edges*satellites*len(orders)
    eligible = triangles*edges*satellites >= 256
    assert after[1].get('receiver_image',0) > 0 if eligible and 'RD' in orders else after[1].get('receiver_image',0) == 0
    if eligible:
        assert after[1]['_mirror'] < before[1]['_mirror']
        assert after[1]['_unit_or_none'] < before[1]['_unit_or_none']
    assert before[2][0]


def outcome(module,tris,edges,rx,sat,policy='default',**kwargs):
    settings = np.geterr() if policy == 'default' else dict.fromkeys(np.geterr(),policy)
    with warnings.catch_warnings(record=True) as recorded,np.errstate(**settings):
        warnings.simplefilter('always')
        try:
            result = ('return',tree_hash(module.compute_reflection_diffraction_paths(tris,edges,rx,sat,**kwargs)))
        except (ValueError,TypeError,RuntimeError,FloatingPointError,OverflowError,ZeroDivisionError) as exc:
            result = ('error',type(exc).__name__,str(exc))
    return result,[(type(r.message).__name__,str(r.message)) for r in recorded]


@pytest.mark.parametrize('policy',['default','warn','raise'])
@pytest.mark.parametrize('scale',[1e-160,1e-16,1.,1e25,1e100,1e154])
def test_extreme_inputs_and_error_order(oracle,policy,scale):
    tris = np.tile(np.array([[[5.,-50.,-50.],[5.,50.,-50.],[5.,0.,50.]]]),(2,1,1))*scale
    start = np.tile([0.,20.,-3.],(16,1))*scale
    end = np.tile([0.,20.,3.],(16,1))*scale
    e = SimpleNamespace(start=start,end=end,midpoint=.5*(start+end),size=16)
    rx,sat = np.zeros(3),np.tile([-20.,60.,0.],(8,1))*scale
    options = dict(max_edge_range_m=250*scale,max_ray_edge_distance_m=25*scale,max_excess_path_m=120*scale)
    assert outcome(oracle,tris,e,rx,sat,policy,**options) == outcome(current,tris,e,rx,sat,policy,**options)


@pytest.mark.parametrize('name',['_mirror','_unit_or_none','_point_in_triangle','_diffraction_geometry','_build_rd_path','_build_dr_path','fresnel_v','knife_edge_loss_db','knife_edge_amplitude'])
def test_original_helper_hooks_keep_arguments_calls_and_results(oracle,monkeypatch,name):
    model,e,rx,sat = scene(2,16,8)
    visits,results = [],[]
    for module in (oracle,current):
        original,calls = getattr(module,name),[]

        def wrapped(*args,**kwargs):
            calls.append(tree_hash((args,kwargs)))
            return original(*args,**kwargs)
        with monkeypatch.context() as context:
            context.setattr(module,name,wrapped)
            results.append(outcome(module,model.triangles,e,rx,sat))
        visits.append(calls)
    assert visits[0] == visits[1] and visits[0]
    assert results[0] == results[1]


@pytest.mark.parametrize('where',['triangles','start','end','satellite'])
@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_data_preserves_warnings_and_results(oracle,where,value):
    model,e,rx,sat = scene(2,16,8)
    target = model.triangles if where == 'triangles' else sat if where == 'satellite' else getattr(e,where)
    target.flat[0] = value
    assert outcome(oracle,model.triangles,e,rx,sat) == outcome(current,model.triangles,e,rx,sat)


@pytest.mark.parametrize('layout',['fortran','strided','reversed'])
def test_layouts_and_candidate_order(oracle,layout):
    model,e,rx,sat = scene(2,16,8)
    def convert(a):
        return np.asfortranarray(a) if layout == 'fortran' else np.repeat(a,2,axis=0)[::2] if layout == 'strided' else a[::-1]
    tris,sat = convert(model.triangles),convert(sat)
    for name in ('start','end','midpoint'):
        setattr(e,name,convert(getattr(e,name)))
    assert trace(oracle,tris,e,rx,sat)[0] == trace(current,tris,e,rx,sat)[0]


@pytest.mark.parametrize('kind',['disabled','empty_sat','empty_triangles','empty_edges','orders','max_triangles','far_edges'])
def test_early_returns_and_existing_limits(oracle,kind):
    model,e,rx,sat = scene(2,16,8)
    tris,kwargs = model.triangles,{}
    if kind == 'disabled':
        kwargs['max_paths'] = 0
    elif kind == 'empty_sat':
        sat = np.empty((0,3))
    elif kind == 'empty_triangles':
        tris = np.empty((0,3,3))
    elif kind == 'empty_edges':
        e.start,e.end,e.midpoint,e.size = np.empty((0,3)),np.empty((0,3)),np.empty((0,3)),0
    elif kind == 'orders':
        kwargs['orders'] = ('unknown',)
    elif kind == 'max_triangles':
        kwargs['max_triangles'] = 0
    else:
        kwargs['max_edge_range_m'] = .1
    assert outcome(oracle,tris,e,rx,sat,**kwargs) == outcome(current,tris,e,rx,sat,**kwargs)


@pytest.mark.parametrize('triangles,edges,satellites',[(1,16,15),(1,16,16),(65,1,4),(2,513,1)])
def test_dispatch_boundaries_preserve_all_output_bits(oracle,triangles,edges,satellites):
    model,e,rx,sat = scene(triangles,edges,satellites)
    assert outcome(oracle,model.triangles,e,rx,sat) == outcome(current,model.triangles,e,rx,sat)


def test_no_cross_call_cache_and_independent_returned_arrays(oracle):
    model,e,rx,sat = scene(2,16,8)
    saved = current.compute_reflection_diffraction_paths(model.triangles,e,rx,sat)
    digest = tree_hash(saved)
    assert saved[0]
    for paths in saved:
        for path in paths:
            for a in (path.reflection_point,path.diffraction_point):
                assert a.flags.owndata and not np.shares_memory(a,model.triangles) and not np.shares_memory(a,e.start)
    model.triangles += [0.,.1,0.]
    e.start += [.1,0.,0.]
    assert outcome(oracle,model.triangles,e,rx,sat) == outcome(current,model.triangles,e,rx,sat)
    assert tree_hash(saved) == digest


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize('samples',[31,2600,260000])
@pytest.mark.parametrize('orders',[('RD',),('DR',),('RD','DR')])
def test_real_native_LOS_composite_channels_and_CUDA_IQ_bits(oracle,samples,orders):
    calls,owners = [],[]
    try:
        for module in (oracle,current):
            call,owner,_ = make_call(module,2,16,8,samples,orders=orders)
            calls.append(call)
            owners.append(owner)
        before,after = calls[0](),calls[1]()
        assert before['n_reflection_diffraction_paths'] == 32
        assert {p.order for paths in before['reflection_diffraction_paths'] for p in paths} == set(orders)
        assert tree_hash(before) == tree_hash(after)
    finally:
        for owner in owners:
            owner.sim.release_gpu()


def test_all_original_scalar_helpers_keep_their_exact_AST(oracle):
    import ast

    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)}
    original,actual = functions(oracle.__file__),functions(current.__file__)
    for name,value in original.items():
        if name != 'compute_reflection_diffraction_paths':
            assert actual[name] == value,name
