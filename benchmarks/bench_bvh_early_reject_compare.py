"""Compare certified wide-miss rejection before the reflection intersection reciprocal.

Actual builds run sequentially in the same process, alternating each timed
whole call. Output bytes are checked for validation and every timed operation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from bench_bvh_build_compare import SOURCE_PATHS, _activate, _measure, _sha
from bench_bvh_combined_compare import _assert_tree, _tree_hash
from bench_bvh_paired_candidates_compare import _validate as _preceding_validate
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays, _tie_scene
from bench_bvh_normal_compare import _normal_cases
from bench_raytrace import _make_building_mesh, _make_satellites


def _mirror_cases(before):
    for count in (8191,8192,12000,120000):
        nodes,mesh = _build(before,_make_building_mesh((count+11)//12)[:count])
        chunk = (32*1024*1024)//24//count
        for epochs in (chunk,chunk+1,2*chunk,2*chunk+1):
            for satellites in (15,16,32):
                rx,sat = _rays(epochs,satellites,seed=epochs+count,near=True)
                rx[1] = np.nan
                sat[2,:4] = np.inf
                sat[3,0] = rx[3]
                yield f'parallel_mirror_chunks/triangles{count}/epochs{epochs}/satellites{satellites}',(rx,sat,nodes,mesh)
    nodes,mesh = _build(before,_make_building_mesh(1000))
    offset = np.array([6378137.,-4500000.,3500000.])
    shifted = nodes.copy()
    shifted[:,:3] += offset
    shifted[:,3:6] += offset
    rx,sat = _rays(129,32,seed=918,near=True)
    yield 'parallel_mirror_chunks/ecef',(rx+offset,sat+offset,shifted,mesh+offset)


def _guard_cases():
    u = np.array([-3., -1., np.nextafter(0., -1.), -1e-307, -0., 0.,
                  np.nextafter(0., 1.), .5, np.nextafter(1., 0.), 1.,
                  np.nextafter(1., 2.), np.nextafter(2., 1.), 2.,
                  np.nextafter(2., 3.), 3., 1e100])
    for count in (511, 512, 513, 8192):
        for scale in (1e-7, 1., 1e50, 1e100):
            for reverse in (False, True):
                for shifted in (False, True):
                    offset = np.array([6378137., -4500000., 3500000.]) if shifted else np.zeros(3)
                    triangle = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])*scale
                    if reverse:
                        triangle = triangle[[0,2,1]]
                    mesh = np.repeat(triangle[None],count,axis=0)+offset
                    bounds = np.r_[mesh.min(axis=(0,1)),mesh.max(axis=(0,1))]
                    nodes = np.array([[*bounds,-1.,-1.,0.,count]])
                    rx = np.c_[u,np.full(len(u),.25),np.ones(len(u))]*scale+offset
                    sat = np.repeat(rx[:,None],16,axis=1)
                    sat[:,:,2] = offset[2]+np.resize([5.,-5.],16)*scale
                    yield f'early_guard/N{count}/scale{scale}/reverse{reverse}/ECEF{shifted}', (rx,sat,nodes,mesh)


def _validate(before, after):
    from bench_bvh_parallel_mirror_chunks_compare import _validate as original_validate
    result = original_validate(before, after)
    records = result['records']
    for name,args in _guard_cases():
        scenes = [native.DeviceBVH(*args[2:]) for native in (before,after)]
        try:
            expected = list(scenes[0].trace_batch(*args[:2]))
            actual = list(scenes[1].trace_batch(*args[:2]))
            independent = [after.raytrace_los_check_bvh_batch(*args),
                           *after.raytrace_multipath_bvh_batch(*args)]
            _assert_tree(expected, actual)
            _assert_tree(expected, independent)
            _assert_tree(expected, list(scenes[1].trace_batch(*args[:2])))
            records[name] = dict(input_sha256=_digest(*args), baseline_sha256=_tree_hash(expected),
                candidate_sha256=_tree_hash(actual), independent_sha256=_tree_hash(independent),
                nonzero_delays=int(np.count_nonzero(expected[2])),
                allocated_bytes=dict(baseline=scenes[0].allocated_bytes,candidate=scenes[1].allocated_bytes))
        finally:
            for scene in scenes:
                scene.close()
        assert all(scene.allocated_bytes == 0 for scene in scenes)
    result['cases'] = len(records)
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
                'warm_multipath': [lambda owner=scene:list(owner.multipath_batch(rx,sat)) for scene in scenes],
                'public_warm_trace': [lambda model=model:list(model.trace_batch(rx,sat)) for model in models],
                'native_cold_trace_close': [lambda native=native:cold(native) for native in (before,after)],
                'unchanged_warm_los': [lambda owner=scene:owner.check_los_batch(rx,sat) for scene in scenes]}
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','baseline-dir','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--counts',type=int,nargs='+',default=[63,64,96,127,128])
    parser.add_argument('--shapes',nargs='+',default=['1x4097','129x32','257x32','1x192'])
    parser.add_argument('--validation-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'python'))
    shapes=[tuple(map(int,s.split('x'))) for s in args.shapes]
    if args.rounds<1 or args.repeats<1 or min(args.counts)<1 or any(len(s)!=2 or min(s)<1 for s in shapes):
        parser.error('positive rounds/repeats/counts and epoch x satellite shapes required')
    anchor_path='benchmarks/bvh_parallel_mirror_chunks_gpu_20261013.json'
    urban_anchor_path='benchmarks/signal_batch_gpu_20261012.json'
    anchor=json.loads((root/anchor_path).read_text(encoding='utf8'))
    urban_anchor=json.loads((root/urban_anchor_path).read_text(encoding='utf8'))
    sources=dict(baseline={p:_sha(args.baseline_dir/Path(p).name) for p in SOURCE_PATHS},
        candidate={p:_sha(root/p) for p in SOURCE_PATHS})
    extensions=dict(baseline=_sha(args.baseline),candidate=_sha(args.candidate))
    assert all(sources['baseline'][p]==anchor['source_sha256']['candidate'][p] for p in SOURCE_PATHS[:-1])
    assert sources['baseline'][SOURCE_PATHS[-1]]==urban_anchor['source_sha256']['candidate'][SOURCE_PATHS[-1]]
    assert extensions['baseline']==anchor['extension_sha256']['candidate']
    assert all(sources['baseline'][p]==sources['candidate'][p] for p in SOURCE_PATHS[1:])
    before=_load_extension(args.baseline.resolve(),'baseline._bvh')
    after=_load_extension(args.candidate.resolve(),'candidate._bvh')
    _activate(after)
    validation=_validate(before,after)
    print(f"{validation['cases']} exact LOS/reflection cases",flush=True)
    measurements={} if args.validation_only else _timing(before,after,args.rounds,args.repeats,args.counts,shapes)
    helpers=('benchmarks/bench_bvh_early_reject_compare.py','benchmarks/bench_bvh_parallel_mirror_chunks_compare.py','benchmarks/bench_bvh_paired_candidates_compare.py','benchmarks/bench_bvh_compact_blocks_compare.py','benchmarks/bench_bvh_candidate_blocks_compare.py','benchmarks/bench_bvh_ordered_triangles_compare.py','benchmarks/bench_bvh_tiny_order_compare.py','benchmarks/bench_bvh_small_order_compare.py','benchmarks/bench_bvh_mesh512_compare.py','benchmarks/bench_bvh_mesh2048_compare.py','benchmarks/bench_bvh_medium_parallel_compare.py','benchmarks/bench_bvh_large_parallel_compare.py','benchmarks/bench_bvh_serial_cache_compare.py','benchmarks/bench_bvh_mirror_compare.py',
        'benchmarks/bench_bvh_normal_compare.py','benchmarks/bench_bvh_build_compare.py',
        'benchmarks/bench_bvh_combined_compare.py','benchmarks/bench_bvh_resident_compare.py',
        'benchmarks/bench_bvh_multipath_compare.py','benchmarks/bench_raytrace.py')
    report=dict(schema_version=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration='CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math',
        baseline_reference=dict(path=anchor_path,sha256=_sha(root/anchor_path)),
        urban_source_reference=dict(path=urban_anchor_path,sha256=_sha(root/urban_anchor_path)),
        method='same process; sequential whole calls; alternating build order each repetition and round; nonexclusive host',
        scope='native reflection/public LOS+reflection/native owner construction+first trace+close/unchanged LOS; includes validation, H2D, all kernels, synchronization and returned arrays; CPU BVH construction and output hashing excluded',
        rounds=args.rounds,repeats=args.repeats,counts=args.counts,shapes=shapes,warmup_calls_per_build=3,
        source_sha256=sources,extension_sha256=extensions,benchmark_sha256={p:_sha(root/p) for p in helpers},
        validation=validation,measurements=measurements)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
