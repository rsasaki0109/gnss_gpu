// Float64 CPU-only bottleneck profile with exact snapshot checks.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {gunzipSync} from "node:zlib";
import {join} from "node:path";
import {pathToFileURL,fileURLToPath} from "node:url";
const root=fileURLToPath(new URL("../",import.meta.url)),load=name=>import(pathToFileURL(join(root,"docs/assets/simulation",name)));
const [{CitySimulation},{decodeMesh}]=await Promise.all([load("city.mjs"),load("mesh.mjs")]);
const scene=JSON.parse(gunzipSync(readFileSync(join(root,"docs/assets/plateau/odaiba-scene.json.gz"))));
const bytes=gunzipSync(readFileSync(join(root,"docs/assets/plateau/odaiba-mesh.bin.gz")));
const simulation=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
const settings=[0,2000,4000,6000,8000,12409].map(epoch=>({epoch,antenna:0,mask:10,systems:"GECJR"}));
const expected=settings.map(s=>simulation.compute(s));
const phase={total:0,directOcclusion:0,reflectionBatchInclusive:0,reflectionKernelSubset:0};let active=false,insideReflection=false;
const first=simulation.bvh.firstHit.bind(simulation.bvh),batch=simulation.bvh.reflectionsBatch.bind(simulation.bvh),kernel=simulation.bvh.reflectionKernel;
simulation.bvh.firstHit=(...args)=>{if(!active||insideReflection)return first(...args);const t=performance.now();try{return first(...args);}finally{phase.directOcclusion+=performance.now()-t;}};
simulation.bvh.reflectionsBatch=(...args)=>{insideReflection=true;const t=performance.now();try{return batch(...args);}finally{if(active)phase.reflectionBatchInclusive+=performance.now()-t;insideReflection=false;}};
if(kernel){const scan=kernel.scan.bind(kernel);kernel.scan=(...args)=>{const t=performance.now();try{return scan(...args);}finally{if(active)phase.reflectionKernelSubset+=performance.now()-t;}};}
for(const s of settings)simulation.compute(s);
const scansBefore=kernel?.scans??0,fallbacksBefore=kernel?.fallbacks??0;
active=true;let checks=0;
for(let round=0;round<5;round++)for(let i=0;i<settings.length;i++){const t=performance.now(),actual=simulation.compute(settings[i]);phase.total+=performance.now()-t;assert.deepEqual(actual,expected[i]);checks++;}
phase.other=phase.total-phase.directOcclusion-phase.reflectionBatchInclusive;
const report={scope:"Node CPU trace only; no rendering, transfer, startup/index or GPU",checks,settings,backend:simulation.bvh.reflectionBackend,simd:kernel?.simd??false,kernelScansDuringProfile:(kernel?.scans??0)-scansBefore,kernelFallbacksDuringProfile:(kernel?.fallbacks??0)-fallbacksBefore,milliseconds:phase,percentOfTotal:{directOcclusion:100*phase.directOcclusion/phase.total,reflectionBatchInclusive:100*phase.reflectionBatchInclusive/phase.total,reflectionKernelSubset:100*phase.reflectionKernelSubset/phase.total,other:100*phase.other/phase.total},note:"Kernel subset is included in reflection batch, so those two percentages must not be added. Instrumentation overhead is included; this is a bottleneck profile, not a speedup benchmark."};
console.log(JSON.stringify(report,null,2));
