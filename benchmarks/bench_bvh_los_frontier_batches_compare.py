"""Compare exact LOS subtree frontiers for medium resident ray batches.

Actual builds run sequentially in the same process, alternating each timed
whole call. Output bytes are checked for validation and every timed operation.
"""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from bench_bvh_build_compare import SOURCE_PATHS, _activate, _measure, _sha
from bench_bvh_combined_compare import _assert_tree, _tree_hash
from bench_bvh_medium_ray_chunks_compare import _validate as _preceding_validate
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays, _tie_scene
from bench_bvh_normal_compare import _normal_cases
from bench_raytrace import _make_building_mesh, _make_satellites


def _frontier_tree(height=4, reverse=False):
    mesh=np.zeros((8192,3,3))
    mesh[:1024]=[[1.,-20.,-10.],[1.,20.,-10.],[1.,0.,20.]]
    nodes=[]
    for index in range(15):
        tail=[2*index+1,2*index+2,0,0] if index<7 else [-1,-1,(index-7)*1024,1024]
        nodes.append([-1000.,-1000.,-1000.,1000.,1000.,1000.]+tail)
    leaf=7
    for _ in range(height-4):
        child=len(nodes)
        nodes.append(nodes[leaf].copy())
        nodes.append([-1000.,-1000.,-1000.,1000.,1000.,1000.,-1.,-1.,0.,0.])
        nodes[leaf][6:]=[child,child+1,0,0] if reverse else [child+1,child,0,0]
        leaf=child
    return np.asarray(nodes),mesh


def _frontier_cases(before):
    for count in (8191,8192,12000,120000):
        nodes,mesh=_build(before,_make_building_mesh((count+11)//12)[:count])
        for epochs,satellites in ((1,1),(1,12),(1,32),(1,33),(2,12),(2,16),(3,11)):
            rx,sat=_rays(epochs,satellites,seed=count+satellites,near=True)
            sat[0,0]=rx[0]
            yield f'los_frontier/triangles{count}/epochs{epochs}/satellites{satellites}',(rx,sat,nodes,mesh)
        rx,sat=_rays(2,12,seed=count,near=True)
        rx[-1]=np.nan
        sat[0,0]=np.inf
        yield f'los_frontier/padding/triangles{count}',(rx,sat,nodes,mesh)
    offset=np.array([6378137.,-4500000.,3500000.])
    nodes,mesh=_build(before,_make_building_mesh(1000))
    nodes[:,:3]+=offset
    nodes[:,3:6]+=offset
    rx,sat=_rays(2,16,seed=1210,near=True)
    yield 'los_frontier/ecef',(rx+offset,sat+offset,nodes,mesh+offset)
    for height in (4,31,32,33,64,70):
        for reverse in (False,True):
            nodes,mesh=_frontier_tree(height,reverse)
            rx=np.zeros((1,3))
            sat=np.array([[[10.,0.,0.],[0.,10.,0.],[-10.,0.,0.]]])
            yield f'los_frontier/height{height}/reverse{reverse}',(rx,sat,nodes,mesh)
    for index in (0,1,3):
        nodes,mesh=_frontier_tree()
        nodes[index,:6]=[100.,100.,100.,101.,101.,101.]
        yield f'los_frontier/ancestor{index}',(np.zeros((1,3)),np.array([[[10.,0.,0.]]]),nodes,mesh)
    for exclude in (False,True):
        nodes,mesh=_frontier_tree()
        nodes[4,6]=7
        if exclude:
            nodes[1,:6]=[100.,100.,100.,101.,101.,101.]
        yield f'los_frontier/shared_leaf/exclude{exclude}',(np.zeros((1,3)),np.array([[[10.,0.,0.]]]),nodes,mesh)
    nodes,mesh=_frontier_tree()
    sat=np.array([[[0.,0.,0.],[np.nextafter(1e-15,0.),0.,0.],
                   [np.nextafter(1e-15,np.inf),0.,0.],[10.,1e-19,0.],
                   [10.,np.nextafter(1e-19,0.),0.],[1e150,0.,0.],[np.nan,0.,0.],[np.inf,0.,0.]]])
    yield 'los_frontier/cutoffs',(np.zeros((1,3)),sat,nodes,mesh)


def _frontier_validate(before,after):
    result=_preceding_validate(before,after)
    records=result['records']
    for name,args in _frontier_cases(before):
        scenes=[native.DeviceBVH(*args[2:]) for native in (before,after)]
        try:
            expected=list(scenes[0].trace_batch(*args[:2]))
            actual=list(scenes[1].trace_batch(*args[:2]))
            _assert_tree(expected,actual)
            _assert_tree(expected,list(scenes[1].trace_batch(*args[:2])))
            _assert_tree(expected[0],scenes[1].check_los_batch(*args[:2]))
            _assert_tree(expected[1:],list(scenes[1].multipath_batch(*args[:2])))
            assert scenes[0].allocated_bytes==scenes[1].allocated_bytes
            if len(args[0])==1 and np.isfinite(args[0]).all() and np.isfinite(args[1]).all():
                _assert_tree(scenes[0].check_los(args[0][0],args[1][0]),scenes[1].check_los(args[0][0],args[1][0]))
            records[name]=dict(input_sha256=_digest(*args),baseline_sha256=_tree_hash(expected),candidate_sha256=_tree_hash(actual),
                allocated_bytes=dict(baseline=scenes[0].allocated_bytes,candidate=scenes[1].allocated_bytes))
        finally:
            for scene in scenes:
                scene.close()
        assert all(scene.allocated_bytes==0 for scene in scenes)
    result['cases']=len(records)
    return result


def _validate(before,after):
    result=_frontier_validate(before,after)
    for count in (8191,8192,12000):
        nodes,mesh=_build(before,_make_building_mesh((count+11)//12)[:count])
        for epochs,satellites in ((1,33),(3,12),(2,32),(8,16),(17,15),(16,16),(1,257),(17,16)):
            for near in (False,True):
                rx,sat=_rays(epochs,satellites,seed=count+satellites,near=near)
                sat[0,0]=rx[0]
                if epochs>1:
                    rx[-1]=np.nan
                    sat[0,-1]=np.inf
                scenes=[native.DeviceBVH(nodes,mesh) for native in (before,after)]
                try:
                    expected=list(scenes[0].trace_batch(rx,sat))
                    actual=list(scenes[1].trace_batch(rx,sat))
                    _assert_tree(expected,actual)
                    _assert_tree(expected[0],scenes[1].check_los_batch(rx,sat))
                    if epochs==1:
                        _assert_tree(scenes[0].check_los(rx[0],sat[0]),scenes[1].check_los(rx[0],sat[0]))
                    assert scenes[0].allocated_bytes==scenes[1].allocated_bytes
                    result['records'][f'frontier_batches/triangles{count}/epochs{epochs}/satellites{satellites}/near{near}']=dict(
                        input_sha256=_digest(rx,sat,nodes,mesh),baseline_sha256=_tree_hash(expected),candidate_sha256=_tree_hash(actual))
                finally:
                    for scene in scenes:
                        scene.close()
    result['cases']=len(result['records'])
    return result


def _timing(before, after, rounds, repeats, counts, shapes):
    from gnss_gpu.bvh import BVHAccelerator
    records={}
    for count in counts:
        mesh=_make_building_mesh((count+11)//12)[:count]
        nodes, sorted_mesh=_build(before,mesh)
        for epochs, satellites in shapes:
            rx=np.zeros((epochs,3))
            rx[:,0]=np.linspace(-5.,5.,epochs)
            sat=np.repeat(_make_satellites(satellites)[None],epochs,axis=0)
            models=[]
            for native in (before,after):
                _activate(native)
                model=BVHAccelerator(mesh)
                model._device_scene=native.DeviceBVH(nodes,sorted_mesh)
                models.append(model)
            scenes=[model._device_scene for model in models]
            def cold(native):
                scene=native.DeviceBVH(nodes,sorted_mesh)
                try:
                    return list(scene.trace_batch(rx,sat))
                finally:
                    scene.close()
            operations={
                'unchanged_warm_multipath': [lambda owner=scene:list(owner.multipath_batch(rx,sat)) for scene in scenes],
                'public_warm_trace': [lambda model=model:list(model.trace_batch(rx,sat)) for model in models],
                'native_cold_trace_close': [lambda native=native:cold(native) for native in (before,after)],
                'warm_los': [lambda owner=scene:owner.check_los_batch(rx,sat) for scene in scenes]}
            try:
                prefix=f'triangles{count}/epochs{epochs}/satellites{satellites}'
                for stage,calls in operations.items():
                    record=_measure(*calls,rounds,repeats)
                    record['input_sha256']=_digest(rx,sat,nodes,sorted_mesh)
                    records[prefix+'/'+stage]=record
                    print(f"{prefix}/{stage}: {record['baseline_ms']:.4f}->{record['candidate_ms']:.4f}ms ({record['speedup']:.3f}x)",flush=True)
                records[prefix+'/memory']=dict(baseline=scenes[0].allocated_bytes,candidate=scenes[1].allocated_bytes,
                    extra_bytes=scenes[1].allocated_bytes-scenes[0].allocated_bytes)
            finally:
                for model in models:
                    model.release_gpu()
    return records


def _controls(before,after,rounds,repeats):
    from gnss_gpu.bvh import BVHAccelerator
    records={}
    for label in ('root_miss','all_blocked','random_dense'):
        if label=='all_blocked':
            nodes,mesh=_frontier_tree()
            mesh[:]=[[1.,-20.,-10.],[1.,20.,-10.],[1.,0.,20.]]
        elif label=='random_dense':
            mesh=np.random.default_rng(211).uniform(-30.,30.,(8192,3,3))
            nodes,mesh=_build(before,mesh)
        else:
            nodes,mesh=_build(before,_make_building_mesh(1000))
        for epochs,satellites in ((3,12),(8,32)):
            rx=np.zeros((epochs,3))
            sat=np.broadcast_to([10.,0.,0.],(epochs,satellites,3)).copy()
            if label=='root_miss':
                rx[:]=1e6
                sat[:]=rx[:,None,:]+[2e7,3e7,4e7]
            elif label=='random_dense':
                rx,sat=_rays(epochs,satellites,seed=211,near=True)
            models=[]
            for native in (before,after):
                _activate(native)
                # Preserve the explicit adversarial tree; CPU construction is excluded.
                model=object.__new__(BVHAccelerator)
                model.triangles=mesh
                model._nodes_flat=nodes
                model._sorted_indices=np.arange(len(mesh))
                model._sorted_tris=mesh
                model._device_scene=native.DeviceBVH(nodes,mesh)
                models.append(model)
            try:
                scenes=[m._device_scene for m in models]
                for stage,calls in dict(public_warm_trace=[lambda m=m:list(m.trace_batch(rx,sat)) for m in models],
                    warm_los=[lambda s=s:s.check_los_batch(rx,sat) for s in scenes]).items():
                    key=f'{label}/epochs{epochs}/satellites{satellites}/{stage}'
                    records[key]=_measure(*calls,rounds,repeats)
                    records[key]['input_sha256']=_digest(rx,sat,nodes,mesh)
                    print(key,records[key]['baseline_ms'],'->',records[key]['candidate_ms'],flush=True)
            finally:
                for model in models:
                    model.release_gpu()
    records.update(_timing(before,after,rounds,repeats,[8191,8192],[(1,32),(1,33)]))
    records.update(_timing(before,after,rounds,repeats,[12000],[(17,15),(16,16),(1,257),(33,32),(1366,12)]))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','baseline-dir','output','source-record'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--counts',type=int,nargs='+',default=[12000,120000])
    parser.add_argument('--shapes',nargs='+',default=['3x12','2x32','4x32','8x32'])
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--controls-only',action='store_true')
    parser.add_argument('--timing-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'python'))
    shapes=[tuple(map(int,s.split('x'))) for s in args.shapes]
    if args.rounds<1 or args.repeats<1 or min(args.counts)<1 or any(len(s)!=2 or min(s)<1 for s in shapes):
        parser.error('positive rounds/repeats/counts and epoch x satellite shapes required')
    edit=json.loads(args.source_record.read_text(encoding='utf8'))
    sources=dict(baseline={p:_sha(args.baseline_dir/Path(p).name) for p in SOURCE_PATHS},
        candidate={p:_sha(root/p) for p in SOURCE_PATHS})
    for label in ('baseline','candidate'):
        for name in SOURCE_PATHS:
            path=args.baseline_dir/Path(name).name if label=='baseline' else root/name
            normalized=__import__('hashlib').sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            assert normalized==edit['source_normalized_lf_sha256'][label][name],(label,name)
    extensions=dict(baseline=_sha(args.baseline),candidate=_sha(args.candidate))
    assert extensions==edit['extension_sha256']
    # Pin only this validation/measurement process; retain normal priority.
    if sys.platform=='win32':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.GetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t))
        kernel.SetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.c_size_t)
        allowed,system=ctypes.c_size_t(),ctypes.c_size_t()
        handle=kernel.GetCurrentProcess()
        assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
        assert kernel.SetProcessAffinityMask(handle,1<<(allowed.value.bit_length()-1))
    before=_load_extension(args.baseline.resolve(),'baseline._bvh')
    after=_load_extension(args.candidate.resolve(),'candidate._bvh')
    _activate(after)
    validation=dict(cases=0,newly_run=False) if args.timing_only or args.controls_only else _validate(before,after)
    print(f"{validation['cases']} exact LOS/reflection cases",flush=True)
    measurements=(_controls(before,after,args.rounds,args.repeats) if args.controls_only else
        {} if args.validation_only else _timing(before,after,args.rounds,args.repeats,args.counts,shapes))
    helpers=tuple(sorted({str(Path(m.__file__).resolve().relative_to(root)).replace('\\','/')
        for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().is_relative_to(root)}))
    report=dict(schema_version=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration='CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math',
        baseline_reference=edit['baseline_reference'],source_record_sha256=_sha(args.source_record),
        method='Own process pinned to one allowed CPU, normal priority; sequential whole calls; alternating build order each repetition and round; nonexclusive host',
        scope='native reflection/public LOS+reflection/native owner construction+first trace+close/LOS; includes validation, H2D, all kernels, synchronization and returned arrays; CPU BVH construction and output hashing excluded',
        rounds=args.rounds,repeats=args.repeats,counts=args.counts,shapes=shapes,warmup_calls_per_build=3,
        source_sha256=sources,extension_sha256=extensions,benchmark_sha256={p:_sha(root/p) for p in helpers},
        validation=validation,measurements=measurements)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
