"""Reproduce restored, alternating whole-operation PF trial comparisons.

Freeze the D203 binary and Python runtime files, apply the report patch and build
the candidate. Pass both binaries, the frozen Python directory and trial report.
The measured historical wrappers and their hashes remain in the report records.
The peeled trial is expected to fail the exact-output assertion before timing
evaluation; use check_pf_kf_peeled_mismatch.py to reproduce its differences.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

import bench_pf_fused_uniform_compare as module
from bench_pf_common_predict_compare import Session as BaseSession
from bench_pf_gaussian_compare import _pr, _weight
from bench_pf_doppler_compare import _kf, _observations
from bench_pf_covariance_compare import _digest

root=Path(__file__).resolve().parent.parent

OriginalSession=module.Session

class GateSession(OriginalSession):
    def __init__(self,native,count,stage):
        super().__init__(native,count,'public_PR_step' if stage=='public_PR_KF_step' else ('PR_control' if stage.startswith('KF') else stage))
        self.stage=stage
        if stage.startswith('KF'):
            self.observations=_observations(int(stage.removeprefix('KF')))
            self.input_sha256=_digest(np.asarray([count]),*self.observations)
        if stage.startswith('Gaussian'):
            self.observations=_pr(int(stage.removeprefix('Gaussian')))
            self.input_sha256=_digest(np.asarray([count]),*self.observations)

    def operation(self):
        if self.stage.startswith('KF'):
            _kf(self.native,self.state,self.observations)
            return (np.empty(0),)
        if self.stage.startswith('Gaussian'):
            _weight(self.native,self.state,self.observations)
            return (np.empty(0),)
        if self.resample is not None or self.stage in ('public_PR_step','public_PR_KF_step'):
            return (BaseSession.operation(self),)
        return super().operation()

module.Session=GateSession

if '--worker' in sys.argv:
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--python-dir',type=Path)
    args=parser.parse_args()
    module.worker(args.worker,args.python_dir)
    raise SystemExit(0)

def affinity(worker):
    # Only this newly spawned child is pinned. Its priority is unchanged.
    if os.name != 'nt':
        if hasattr(os, 'sched_getaffinity'):
            os.sched_setaffinity(worker.process.pid, {max(os.sched_getaffinity(0))})
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    kernel.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    kernel.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    handle = kernel.OpenProcess(0x200, False, worker.process.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.SetProcessAffinityMask(handle, 1 << (allowed.value.bit_length() - 1)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)

class Worker(module.Worker):
    def __init__(self,extension,python_dir=None):
        argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(extension.resolve())]
        if python_dir:
            argv+=['--python-dir',str(python_dir.resolve())]
        (root/'dist').mkdir(exist_ok=True)
        label='baseline' if python_dir else 'candidate'
        self.error_log=(root/f'dist/pf_kf_postmerge_gate_{os.getpid()}_{label}_20261015.log').open('wb')
        self.process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.error_log,
            text=True,encoding='utf8',cwd=root)
        (root/f'dist/pf_kf_postmerge_gate_{os.getpid()}_{label}_pid_20261015.json').write_text(json.dumps(dict(pid=self.process.pid)))
        assert json.loads(self.process.stdout.readline())==dict(ready=True)
        affinity(self)

parser=argparse.ArgumentParser()
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--candidate-source-dir',type=Path,help='Flat frozen native-source directory; Python runtime is checked in the working tree')
parser.add_argument('--baseline-dir',type=Path,required=True)
parser.add_argument('--report',type=Path,required=True,help='Published trial report with frozen source hashes')
parser.add_argument('--stages',nargs='+',default=['KF12','public_PR_KF_step','public_PR_step'])
parser.add_argument('--sizes',nargs='+',type=int,default=[1000000])
parser.add_argument('--rounds',type=int,default=5)
parser.add_argument('--repeats',type=int,default=5)
parser.add_argument('--output',type=Path,default=root/'dist/pf_postmerge_reproduction.json')
args=parser.parse_args()
edit=json.loads(args.report.read_text(encoding='utf8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def normalized(path):
    data=path.read_bytes().replace(b'\r\n',b'\n')
    if path.suffix=='.py':
        data=re.sub(rb'  # pyright: ignore\[[^\]\r\n]+\](?=\n|$)',b'',data)
    return hashlib.sha256(data).hexdigest()
baseline=args.baseline_dir
candidate=args.candidate
for name,digest in edit['source_sha256']['baseline'].items():
    assert normalized(baseline/Path(name).name)==edit['source_normalized_lf_sha256']['baseline'][name],name
for name,digest in edit['source_sha256']['candidate'].items():
    candidate_source=root/name
    if args.candidate_source_dir and Path(name).suffix != '.py':
        frozen=args.candidate_source_dir/Path(name).name
        if frozen.exists():
            candidate_source=frozen
    assert normalized(candidate_source)==edit['source_normalized_lf_sha256']['candidate'][name],name
assert sha(args.baseline)==edit['extension_sha256']['baseline']
assert sha(candidate)==edit['extension_sha256']['candidate']
workers={}
rows={}
try:
    workers=dict(baseline=Worker(args.baseline,baseline),candidate=Worker(candidate))
    for size in args.sizes:
        for stage in args.stages:
            key=f'N{size}/{stage}'
            pairs=[]
            snapshots=[]
            medians={label:[] for label in workers}
            for round_index in range(args.rounds):
                for label in workers:
                    workers[label].request(mode='measure',size=size,stage=stage,repeats=1,snapshot=False,warmups=0)
                samples={label:[] for label in workers}
                for sample in range(args.repeats):
                    order=('baseline','candidate') if (round_index+sample)%2==0 else ('candidate','baseline')
                    pair={label:workers[label].request(mode='measure',size=size,stage=stage,repeats=1,snapshot=False,warmups=0) for label in order}
                    for field in ('input_sha256','log_weights_sha256','returned_output_sha256'):
                        assert pair['baseline'][field]==pair['candidate'][field],(key,field)
                    pairs.append(dict(round=round_index,sample=sample,order=order,runs=pair))
                    for label in workers:
                        samples[label].append(pair[label]['samples_ms'][0])
                snapshot={label:worker.request(mode='snapshot',size=size,stage=stage) for label,worker in workers.items()}
                assert snapshot['baseline']==snapshot['candidate'],key
                snapshots.append(snapshot)
                for label in workers:
                    medians[label].append(float(np.median(samples[label])))
                print(key,round_index+1,medians['baseline'][-1],'->',medians['candidate'][-1],flush=True)
                rows[key]=dict(round_medians_ms=medians,pairs=pairs,snapshots=snapshots,complete=False)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(dict(early_gate_only=True,complete=False,source_sha256=edit['source_sha256'],summary=rows),indent=2)+'\n')
            old,new=[float(np.median(medians[label])) for label in workers]
            rows[key]=dict(baseline_ms=old,candidate_ms=new,reduction_percent=100*(1-new/old),
                improving_rounds=sum(b<a for a,b in zip(medians['baseline'],medians['candidate'])),
                round_medians_ms=medians,pairs=pairs,snapshots=snapshots)
finally:
    close_errors=[]
    for worker in workers.values():
        try:
            worker.close()
        except Exception as error:
            close_errors.append(repr(error))
    assert not close_errors,close_errors
result=dict(early_gate_only=True,source_sha256=edit['source_sha256'],
    extension_sha256=dict(baseline=sha(args.baseline),candidate=sha(candidate)),
    helper_sha256={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in [Path(__file__),root/'benchmarks/bench_pf_fused_uniform_compare.py']},
    decision_criteria=edit['early_gate']['decision_criteria'],
    method='Known own child workers on same logical CPU, normal priority, serial alternating restored calls; initialization/hash/full snapshots outside synchronized timing.',
    summary=rows)
args.output.write_text(json.dumps(result,indent=2)+'\n')
for key,row in rows.items():
    print(key,round(row['reduction_percent'],3),'percent shorter',row['improving_rounds'],'improving rounds',flush=True)
