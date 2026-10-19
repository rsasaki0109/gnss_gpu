"""Screen exact, certified reciprocal sharing; component probe, not PF timing."""

import argparse
import ctypes
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from bench_pf_private_weights_compare import _pin

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(str((array.shape,array.dtype.str)).encode()+array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll',type=Path,required=True)
    parser.add_argument('--build-record',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=5)
    args = parser.parse_args()
    assert not args.output.exists()
    checkpoint = args.output.with_suffix('.checkpoint.json')
    assert not checkpoint.exists()
    build = json.loads(args.build_record.read_text(encoding='utf8'))
    assert sha(args.dll) == build['dll_sha256']
    for name,value in build['source_sha256'].items():
        assert sha(ROOT/name) == value
    _pin()
    cuda_directory = os.add_dll_directory(str(Path(os.environ['CUDA_PATH'])/'bin'))
    dll = ctypes.CDLL(str(args.dll.resolve()))
    fp = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    up = np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
    function = dll.run_probe
    function.argtypes = [fp,fp,up,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
    function.restype = ctypes.c_int

    def call(values,mode,steps=1,count_fast=True,repeats=1):
        output = np.empty((len(values),3),dtype=np.float64)
        accepted = np.empty(len(values),dtype=np.uint32)
        ms = ctypes.c_float()
        code = function(values,output,accepted,len(values),steps,mode,int(count_fast),repeats,ctypes.byref(ms))
        assert code == 0,code
        return output,accepted,float(ms.value)/repeats

    rng = np.random.default_rng(224)
    normal = np.empty((131072,4))
    normal[:,:3] = rng.normal(size=(len(normal),3))*2e7
    normal[:,3] = np.linalg.norm(normal[:,:3],axis=1)+1.
    scales = np.ldexp(rng.uniform(1.,2.,size=(131072,4)),rng.integers(-499,500,size=(131072,4)))
    scales[:,:3] *= rng.choice([-1.,1.],size=(len(scales),3))
    raw = rng.integers(0,np.iinfo(np.uint64).max,size=(131072,4),dtype=np.uint64).view(np.float64)
    raw[:,3] = np.abs(raw[:,3])
    values = np.array([0.,-0.,1.,-1.,np.nan,np.inf,-np.inf,np.finfo(float).tiny,np.nextafter(0.,1.),np.finfo(float).max,
        2.**-500,2.**500,np.nextafter(2.**-500,0.),np.nextafter(2.**500,np.inf),np.nextafter(1.,0.),np.nextafter(1.,np.inf)])
    special = np.array([[a,b,c,d] for a in values for b in (0.,-0.,1.,-1.) for c in (1.,-1.,-0.) for d in values],dtype=float)
    validation = {}
    independent = 0
    for name,data,steps in (('geometry',normal,1),('normal_scales',scales,1),('raw_ieee',raw,1),('special_cross',special,1),('geometry12',normal,12)):
        reference,_,_ = call(data,0,steps)
        rows = {}
        for mode in (1,2):
            result,accepted,_ = call(data,mode,steps)
            bad = np.argwhere(reference.view(np.uint64) != result.view(np.uint64))
            if len(bad):
                evidence = [dict(row=int(i),axis=int(axis),input_bits=data[i].view(np.uint64).tolist(),
                    baseline_bits=int(reference[i,axis].view(np.uint64)),candidate_bits=int(result[i,axis].view(np.uint64))) for i,axis in bad[:32]]
                args.output.write_text(json.dumps(dict(validation_failed=True,case=name,mode=mode,differences=len(bad),
                    counterexamples=evidence,completed_validation=validation,dll_sha256=sha(args.dll),helper_sha256=sha(Path(__file__))),indent=2)+'\n',encoding='utf8')
            assert not len(bad),(name,mode,bad[:10].tolist())
            rows[str(mode)] = dict(output_sha256=digest(result),certified_quotients=int(np.sum(accepted,dtype=np.uint64)),quotients=len(data)*3*steps)
        if steps == 1:
            for i in range(min(4096,len(data))):
                denominator = float(data[i,3])
                if not np.isfinite(denominator) or denominator == 0.:
                    continue
                for axis in range(3):
                    numerator = float(data[i,axis])
                    if not np.isfinite(numerator):
                        continue
                    try:
                        exact = float(Fraction(numerator)/Fraction(denominator))
                    except OverflowError:
                        exact = float(np.copysign(np.inf,numerator*denominator))
                    # Fraction has no signed zero; retain IEEE quotient sign.
                    if exact == 0.:
                        exact = float(np.copysign(0.,np.copysign(1.,numerator)*np.copysign(1.,denominator)))
                    assert np.float64(exact).view(np.uint64) == reference[i,axis].view(np.uint64),(name,i,axis)
                    independent += 1
        validation[name] = dict(rows=len(data),steps=steps,input_sha256=digest(data),reference_sha256=digest(reference),variants=rows)
        checkpoint.write_text(json.dumps(dict(validation=validation,independent_exact_fraction_quotients=independent),indent=2)+'\n',encoding='utf8')
        print('validation',name,len(data),'rows exact',flush=True)
    measurements = {}
    for size in (65536,1000000):
        inputs = np.empty((size,4))
        inputs[:,:3] = rng.normal(size=(size,3))*2e7
        inputs[:,3] = np.linalg.norm(inputs[:,:3],axis=1)+1.
        for steps in (1,12):
            expected,_,_ = call(inputs,0,steps)
            expected_sha = digest(expected)
            pairs = []
            for r in range(args.rounds):
                for repeat in range(args.repeats):
                    order = [(r+repeat+i)%4 for i in range(4)]
                    runs = {}
                    for mode in order:
                        result,_,ms = call(inputs,0 if mode == 3 else mode,steps,False,3)
                        assert digest(result) == expected_sha
                        runs[str(mode)] = dict(gpu_ms=ms,output_sha256=expected_sha)
                    pairs.append(dict(round=r,repetition=repeat,order=order,runs=runs))
            summary = {}
            for mode in (1,2,3):
                reductions = [float(np.median([100*(1-p['runs'][str(mode)]['gpu_ms']/p['runs']['0']['gpu_ms']) for p in pairs if p['round']==r])) for r in range(args.rounds)]
                summary[str(mode)] = dict(paired_round_reduction_percent=reductions,paired_reduction_percent=float(np.median(reductions)),improving_rounds=sum(v>0 for v in reductions),
                    baseline_ms=float(np.median([p['runs']['0']['gpu_ms'] for p in pairs])),candidate_ms=float(np.median([p['runs'][str(mode)]['gpu_ms'] for p in pairs])))
            key = f'N{size}/triples{steps}'
            measurements[key] = dict(input_sha256=digest(inputs),output_sha256=expected_sha,pairs=pairs,summary=summary)
            checkpoint.write_text(json.dumps(dict(validation=validation,independent_exact_fraction_quotients=independent,measurements=measurements),indent=2)+'\n',encoding='utf8')
            print(key,json.dumps(summary),flush=True)
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source_sha256=build['source_sha256'],build_record_sha256=sha(args.build_record),dll_sha256=sha(args.dll),
        helper_sha256=sha(Path(__file__)),rounds=args.rounds,repeats=args.repeats,validation=validation,independent_exact_fraction_quotients=independent,measurements=measurements,
        scope='Isolated CUDA triple-division component. Setup/allocation/H2D/D2H/hash/independent reference outside CUDA-event timing; one warm launch,3 timed launches per sample. No whole-PF or production integration gain is claimed.',
        method='Serial rotating baseline/two candidates/identical-baseline A/A repeat, own last allowed CPU, normal priority, nonexclusive host/GPU. Validation instrumentation disabled in timed kernels. All returned bits compared.',
        variants={'1':'One RN reciprocal, two directed FMA residual endpoints, outward correction, identical RN endpoints or original RN divide.',
            '2':'One RN reciprocal, adjacent RN residual endpoints, outward correction, identical RN endpoints or original RN divide.','3':'A/A: unchanged baseline RN divisions again.'},
        production_edited=False)
    cuda_directory.close()
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
