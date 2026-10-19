"""Compare register-budget probes for bounded reflection candidate groups.

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
from bench_bvh_medium_ray_chunks_compare import _validate
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays, _tie_scene
from bench_bvh_normal_compare import _normal_cases
from bench_raytrace import _make_building_mesh, _make_satellites


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
    for key in ('baseline','candidate','baseline-dir','output','source-record'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--counts',type=int,nargs='+',default=[1024,4096,8191])
    parser.add_argument('--shapes',nargs='+',default=['2049x8','1366x12','513x32'])
    parser.add_argument('--validation-only',action='store_true')
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
    validation=dict(cases=0,newly_run=False) if args.timing_only else _validate(before,after)
    print(f"{validation['cases']} exact LOS/reflection cases",flush=True)
    measurements={} if args.validation_only else _timing(before,after,args.rounds,args.repeats,args.counts,shapes)
    helpers=tuple(sorted({str(Path(m.__file__).resolve().relative_to(root)).replace('\\','/')
        for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().is_relative_to(root)}))
    report=dict(schema_version=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration='CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math',
        baseline_reference=edit['baseline_reference'],source_record_sha256=_sha(args.source_record),
        method='Own process pinned to one allowed CPU, normal priority; sequential whole calls; alternating build order each repetition and round; nonexclusive host',
        scope='native reflection/public LOS+reflection/native owner construction+first trace+close/unchanged LOS; includes validation, H2D, all kernels, synchronization and returned arrays; CPU BVH construction and output hashing excluded',
        rounds=args.rounds,repeats=args.repeats,counts=args.counts,shapes=shapes,warmup_calls_per_build=3,
        source_sha256=sources,extension_sha256=extensions,benchmark_sha256={p:_sha(root/p) for p in helpers},
        validation=validation,measurements=measurements)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
