"""Complete native LOS/composite reflection-diffraction/CUDA IQ comparison."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_reflection_geometry_compare import compare, load, sha, tree_hash

ROOT = Path(__file__).resolve().parents[1]


def scene(triangles=2, edges=16, satellites=12, kind='normal'):
    from gnss_gpu.raytrace import BuildingModel

    mesh = []
    for i in range((triangles+1)//2):
        x = 5.+i
        a,b,c,d = [x,-80.,-40.], [x,80.,-40.], [x,80.,100.], [x,-80.,100.]
        mesh.extend(((a,b,c),(a,c,d)))
    mesh = np.asarray(mesh[:triangles], dtype=float).reshape(-1,3,3)
    start = np.empty((edges,3))
    for i in range(edges):
        start[i] = ([2.8,24.,-20.] if i%2 == 0 else [3.,10.,-20.])
        start[i,:2] += (i//2)*.01
    end = start+[0.,0.,100.]
    if kind == 'sparse':
        start[2:,1] += 1000.
        end[2:,1] += 1000.
    direction = np.column_stack((np.linspace(-.301,-.299,satellites),np.ones(satellites),np.linspace(.15,.25,satellites)))
    sat = 20200e3*direction/np.linalg.norm(direction,axis=1)[:,None]
    origin = np.array([6378137.,0.,0.])
    mesh,start,end,sat = [value[..., [2,0,1]]+origin for value in (mesh,start,end,sat)]
    edge_set = SimpleNamespace(start=start,end=end,midpoint=.5*(start+end),size=edges)
    return BuildingModel(mesh),edge_set,origin.copy(),sat


def make_call(module, triangles=2, edges=16, satellites=12, samples=2600, stage='urban', kind='normal', paths=4, orders=('RD','DR')):
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
    import gnss_gpu.urban_signal_sim as urban

    model,edge_set,rx,sat = scene(triangles,edges,satellites,kind)
    digest = tree_hash((model.triangles,edge_set.start,edge_set.end,rx,sat,orders,paths))
    if stage == 'paths':
        def call():
            return module.compute_reflection_diffraction_paths(model.triangles,edge_set,rx,sat,max_paths=paths,orders=orders)
        return call,None,digest
    owner = UrbanSignalSimulator(model,elevation_mask_deg=0.,diffraction_edges=edge_set,
        max_reflection_diffraction_paths=paths,reflection_diffraction_orders=orders)
    owner.sim.noise_seed = 37
    velocity = np.random.default_rng(223).normal(size=(satellites,3))*1000.
    clock = np.linspace(-1e-7,1e-7,satellites)

    def call():
        original = urban.compute_reflection_diffraction_paths
        urban.compute_reflection_diffraction_paths = module.compute_reflection_diffraction_paths
        try:
            return owner.compute_epoch(rx,sat,sat_vel=velocity,sat_clk=clock,rx_vel=[3.,2.,1.],n_samples=samples)
        finally:
            urban.compute_reflection_diffraction_paths = original
    return call,owner,digest


def paired_summary(row):
    rounds = sorted({p['round'] for p in row['pairs']})
    reductions = [float(np.median([100*(1-p['runs']['candidate']['ms']/p['runs']['baseline']['ms'])
        for p in row['pairs'] if p['round'] == r])) for r in rounds]
    row['paired_round_reduction_percent'] = reductions
    row['paired_reduction_percent'] = float(np.median(reductions))
    row['paired_improving_rounds'] = sum(value>0 for value in reductions)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-dir','source-record','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--triangles',nargs='+',type=int,default=[2,6])
    parser.add_argument('--edges',nargs='+',type=int,default=[16,64])
    parser.add_argument('--satellites',type=int,default=12)
    parser.add_argument('--samples',nargs='+',type=int,default=[2600,260000])
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=3)
    args = parser.parse_args()
    record = json.loads(args.source_record.read_text(encoding='utf8'))
    for label,sources in record['source_sha256'].items():
        for name,digest in sources.items():
            assert sha(args.baseline_dir/Path(name).name if label == 'baseline' else ROOT/name) == digest
    for name,digest in record['extension_sha256'].items():
        assert sha(ROOT/'python/gnss_gpu'/name) == digest
    _pin()
    sys.path.insert(0,str(ROOT/'python'))
    before = load(args.baseline_dir/'reflection_diffraction.py','composite_geometry_before')
    after = load(ROOT/'python/gnss_gpu/reflection_diffraction.py','composite_geometry_after')
    measurements = {}
    for triangles in args.triangles:
        for edges in args.edges:
            for samples in args.samples:
                calls,owners,inputs = [],[],[]
                try:
                    for module in (before,after):
                        call,owner,digest = make_call(module,triangles,edges,args.satellites,samples)
                        calls.append(call)
                        owners.append(owner)
                        inputs.append(digest)
                    assert inputs[0] == inputs[1]
                    row = paired_summary(compare(*calls,args.rounds,args.repeats))
                    value = calls[0]()
                    row['composite_paths'] = value['n_reflection_diffraction_paths']
                    row['returned_orders'] = sorted({p.order for paths in value['reflection_diffraction_paths'] for p in paths})
                    assert row['composite_paths'] == args.satellites*4 and row['returned_orders'] == ['DR','RD']
                    row['input_sha256'] = inputs[0]
                    key = f'triangles{triangles}/edges{edges}/satellites{args.satellites}/samples{samples}/urban'
                    measurements[key] = row
                    print(key,row['baseline_ms'],'->',row['candidate_ms'],'paired',row['paired_reduction_percent'],flush=True)
                finally:
                    for owner in owners:
                        if owner is not None:
                            owner.sim.release_gpu()
    helpers = {str(Path(m.__file__).resolve().relative_to(ROOT)).replace('\\','/'):sha(Path(m.__file__))
        for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).suffix == '.py'
        and Path(m.__file__).resolve().is_relative_to(ROOT)}
    output = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source_record_sha256=sha(args.source_record),
        source_sha256=record['source_sha256'],extension_sha256=record['extension_sha256'],helper_sha256=helpers,
        rounds=args.rounds,repeats=args.repeats,measurements=measurements,
        method='Serial alternating complete calls; within-pair latency reduction median per round, then median of rounds. Absolute medians and original independent-median ratio also retained. All returned fields hashed outside timing. Own last allowed CPU, normal priority, nonexclusive host/GPU.',
        scope='Native LOS/legacy multipath, all original CPU reflection-diffraction triangle-edge-order combinations, clock/Doppler/channels and real CUDA IQ. Actual RD and DR replicas; seed37. Setup/warm/hash excluded; symmetric function wrapper included.')
    assert not args.output.exists()
    args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
