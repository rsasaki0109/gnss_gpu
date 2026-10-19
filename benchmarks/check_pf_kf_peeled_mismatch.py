"""Reproduce exact-output rejection of the peeled Doppler KF trial.

Freeze D203, apply the report patch and build the candidate. Provide both
extensions and the published report. This writes full snapshots and summary
outside timing; a reproduced mismatch is the expected successful outcome.
"""

from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from bench_pf_coalesced_compare import _initialize, _load_extension
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_doppler_compare import _kf, _observations
from bench_pf_kf_launch_compare import _cloud

root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'python'))

parser=argparse.ArgumentParser()
parser.add_argument('--baseline',type=Path)
parser.add_argument('--candidate',type=Path)
parser.add_argument('--report',type=Path)
parser.add_argument('--output-dir',type=Path,default=root/'dist/pf_kf_peeled_reproduction')
parser.add_argument('--worker',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()

if args.worker:
    # Pin only this explicitly started validation process, normal priority.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t))
    kernel.SetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.c_size_t)
    allowed,system=ctypes.c_size_t(),ctypes.c_size_t()
    handle=kernel.GetCurrentProcess()
    assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
    assert kernel.SetProcessAffinityMask(handle,1<<(allowed.value.bit_length()-1))
    native=_load_extension(args.worker.resolve())
    snapshots={}
    inputs={}
    for count in (65537,1000000):
        state=native.pf_device_create(count)
        try:
            variants=('initialized','nonsymmetric','signed_zero','large') if count==65537 else ('initialized',)
            for variant in variants:
                if variant=='initialized':
                    _initialize(native,state)
                else:
                    before,prior=_cloud(count,variant)
                    native.pf_device_set_particle_states(state,before)
                    native.pf_device_set_log_weights(state,prior)
                key=f'N{count}_{variant}'
                obs=_observations(12)
                inputs[key]=_digest(*_snapshot(native,state),*obs)
                _kf(native,state,obs)
                values,weights=_snapshot(native,state)
                snapshots[key+'_state']=values
                snapshots[key+'_weights']=weights
        finally:
            native.pf_device_destroy(state)
    np.savez(args.output,**snapshots)
    args.output.with_suffix('.json').write_text(json.dumps(dict(input_sha256=inputs),indent=2)+'\n')
    raise SystemExit(0)

assert args.baseline and args.candidate and args.report, 'Provide --baseline, --candidate and --report'
report=json.loads(args.report.read_text(encoding='utf8'))
for label,extension in [('baseline',args.baseline),('candidate',args.candidate)]:
    assert hashlib.sha256(extension.read_bytes()).hexdigest()==report['extension_sha256'][label],label
args.output_dir.mkdir(parents=True,exist_ok=True)
outputs={}
for label,extension in [('baseline',args.baseline),('candidate',args.candidate)]:
    output=args.output_dir/f'{label}.npz'
    with (args.output_dir/f'{label}.log').open('wb') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(extension),'--output',str(output)],
                       cwd=root,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,check=True)
    outputs[label]=output
assert json.loads(outputs['baseline'].with_suffix('.json').read_text())==json.loads(outputs['candidate'].with_suffix('.json').read_text())
rows={}
with np.load(outputs['baseline']) as old,np.load(outputs['candidate']) as new:
    assert old.files==new.files
    for key in old.files:
        a,b=old[key],new[key]
        differing=a.view(np.uint64)!=b.view(np.uint64)
        finite=np.isfinite(a)&np.isfinite(b)
        delta=np.abs(a[finite]-b[finite])
        rows[key]=dict(values=int(a.size),bit_differences=int(differing.sum()),
                       max_finite_absolute_difference=float(delta.max(initial=0.)),
                       baseline_sha256=_digest(a),candidate_sha256=_digest(b))
        if a.ndim==2:
            rows[key]['bit_differences_per_state_component']=[int(v) for v in differing.sum(axis=0)]
assert any(row['bit_differences'] for row in rows.values())
result=dict(schema_version=1,all_inputs_identical=True,exact_gate_passed=False,records=rows,
            npz_sha256={label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in outputs.items()},
            interpretation='Finite differences quantify this rejection only; no accuracy tolerance was introduced and no adoption or timing conclusion follows.')
(args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(rows,indent=2))
