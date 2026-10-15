"""Compare bounded parallel reflection for medium meshes against the exact large-mesh build.

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
from bench_bvh_mesh512_compare import _validate as _preceding_validate
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays, _tie_scene
from bench_bvh_normal_compare import _normal_cases
from bench_raytrace import _make_building_mesh, _make_satellites


def _mirror_cases(before):
    for count in (127, 128, 129, 255, 256, 511):
        nodes, mesh = _build(before, _make_building_mesh((count+11)//12)[:count])
        for epochs, satellites in ((1, 1), (1, 16), (1, 192), (129, 32), (512, 32), (513, 32)):
            rx, sat = _rays(epochs, satellites, seed=count+epochs, near=True)
            rx[1 if epochs>1 else 0, 0] = np.nan if epochs>1 else rx[0, 0]
            sat[0, :3] = np.inf
            yield f'medium_parallel/triangles{count}/epochs{epochs}/satellites{satellites}', (rx, sat, nodes, mesh)


def _validate(before, after):
    result=_preceding_validate(before, after)
    records=result['records']
    for name, args in _mirror_cases(before):
        scenes=[native.DeviceBVH(*args[2:]) for native in (before,after)]
        try:
            expected=list(scenes[0].trace_batch(*args[:2]))
            actual=list(scenes[1].trace_batch(*args[:2]))
            _assert_tree(expected,actual)
            _assert_tree(expected,list(scenes[1].trace_batch(*args[:2])))
            _assert_tree(expected[1:],list(scenes[1].multipath_batch(*args[:2])))
            records[name]=dict(input_sha256=_digest(*args),baseline_sha256=_tree_hash(expected),
                candidate_sha256=_tree_hash(actual),nonzero_delays=int(np.count_nonzero(expected[2])),
                allocated_bytes=dict(baseline=scenes[0].allocated_bytes,candidate=scenes[1].allocated_bytes))
        finally:
            for scene in scenes:
                scene.close()
        assert all(scene.allocated_bytes==0 for scene in scenes)
    result['cases']=len(records)
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
    parser.add_argument('--counts',type=int,nargs='+',default=[127,128,256,511,512,1200])
    parser.add_argument('--shapes',nargs='+',default=['1x4097','129x32','257x32','1x192'])
    parser.add_argument('--validation-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'python'))
    shapes=[tuple(map(int,s.split('x'))) for s in args.shapes]
    if args.rounds<1 or args.repeats<1 or min(args.counts)<1 or any(len(s)!=2 or min(s)<1 for s in shapes):
        parser.error('positive rounds/repeats/counts and epoch x satellite shapes required')
    anchor_path='benchmarks/bvh_medium_parallel_gpu_20261013.json'
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
    helpers=('benchmarks/bench_bvh_small_order_compare.py','benchmarks/bench_bvh_mesh512_compare.py','benchmarks/bench_bvh_mesh2048_compare.py','benchmarks/bench_bvh_medium_parallel_compare.py','benchmarks/bench_bvh_large_parallel_compare.py','benchmarks/bench_bvh_serial_cache_compare.py','benchmarks/bench_bvh_mirror_compare.py',
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
