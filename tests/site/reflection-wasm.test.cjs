const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const {spawnSync} = require("node:child_process");
const modules=Promise.all([import("../../docs/assets/simulation/mesh.mjs"),import("../../docs/assets/simulation/reflection-kernel.mjs")]);
function meshOf(triangles) {
  return {vertices:Float64Array.from(triangles.flat(2)),indices:Uint32Array.from({length:triangles.length*3},(_,i)=>i),owners:Uint32Array.from(triangles,(_,i)=>i),types:new Uint8Array(triangles.length),triangleCount:triangles.length,vertexCount:triangles.length*3};
}
const rotate=(p,axis)=>axis===0?[p[2],p[0],p[1]]:axis===1?[p[1],p[2],p[0]]:p.slice();
const bits=new DataView(new ArrayBuffer(8));
function next(value,up) {
  if(value===0)return up?Number.MIN_VALUE:-Number.MIN_VALUE;
  bits.setFloat64(0,value);let n=bits.getBigUint64(0);
  n+=(value>0)===up?1n:-1n;bits.setBigUint64(0,n);return bits.getFloat64(0);
}
const wall=[[[0,-10,0],[0,10,0],[0,10,10]],[[0,-10,0],[0,10,10],[0,-10,10]]];
function compare(bvh,receiver,sources) {
  const input=structuredClone({receiver,sources});
  assert.deepEqual(bvh.reflectionsBatch(receiver,sources),sources.map(source=>bvh.reflections(receiver,source)));
  assert.deepEqual({receiver,sources},input);
}

test("forced Float64 Wasm preserves exact ULP boundaries for every projection and winding",async()=>{
  const [{MeshBVH}]=await modules,epsilon=1e-10;
  const values=[-epsilon,0,1,1+epsilon,1+2*epsilon,-2,2].flatMap(v=>[next(v,false),v,next(v,true)]);
  const sources=values.flatMap(u=>values.map(v=>[1,2*u,2*v]));
  for(const axis of [0,1,2])for(const reversed of [false,true]){
    const triangle=[[0,0,0],[0,1,0],[0,0,1]].map(p=>rotate(p,axis));
    if(reversed)[triangle[1],triangle[2]]=[triangle[2],triangle[1]];
    const bvh=new MeshBVH(meshOf([triangle]),{reflectionBackend:"wasm"});
    assert.equal(bvh.reflectionBackend,"wasm");assert.equal(bvh.reflectionKernel.simd,true);
    const receiver=rotate([1,0,0],axis),rotated=sources.map(p=>rotate(p,axis));
    compare(bvh,receiver,rotated);let scans=1;
    // Exercise every specialized packet count, odd tails and generic dispatch
    // against the independent scalar predicate at the same ULP boundaries.
    for(let width=2;width<=17;width++)for(let start=0;start<rotated.length;start+=width) {
      const batch=rotated.slice(start,start+width);compare(bvh,receiver,batch);
      if(batch.length>=2)scans++;
    }
    assert.equal(bvh.reflectionKernel.scans,scans);assert.equal(bvh.reflectionKernel.fallbacks,0);
  }
});

test("Wasm retains contact, non-finite, subnormal and extreme-scale behavior",async()=>{
  const [{MeshBVH}]=await modules;
  const sources=[[6,4,6],[20200000,4,6],[-6,4,6],[NaN,0,0],[Infinity,0,0],[0,0,-Infinity]];
  for(const axis of [0,1,2]){
    const bvh=new MeshBVH(meshOf(wall.map(t=>t.map(p=>rotate(p,axis)))),{reflectionBackend:"wasm"});
    for(const x of [-1e-5,next(-1e-5,false),next(-1e-5,true),-0,1e-5,next(1e-5,false),next(1e-5,true)])compare(bvh,rotate([x,0,1.5],axis),sources.map(p=>rotate(p,axis)));
    assert.equal(bvh.reflectionKernel.fallbacks,0);
  }
  for(const scale of [0,Number.MIN_VALUE,1e-170,1e-160,1e-9,1,1e100,1e150,1e160]){
    const bvh=new MeshBVH(meshOf([[[0,0,0],[0,scale,0],[0,0,scale]]]),{reflectionBackend:"wasm"});
    compare(bvh,[scale,0,0],[[scale,.5*scale,.5*scale],[-scale,0,0],[scale,0,0]]);
    assert.equal(bvh.reflectionKernel.fallbacks,0);
  }
});

test("bounded output and source buffers fall back completely without truncation or stale results",async()=>{
  const [{MeshBVH},{REFLECTION_SOURCE_LIMIT}]=await modules;
  for(const reflectionCapacity of [0,1]){
    const bvh=new MeshBVH(meshOf([...wall,...wall]),{reflectionBackend:"wasm",reflectionCapacity});
    compare(bvh,[4,-2,1.5],[[6,4,6],[6,4,6]]);
    assert.equal(bvh.reflectionKernel.fallbacks,1);
    compare(bvh,[4,-2,1.5],[[-6,4,6],[-6,4,6]]);
    compare(bvh,[4,-2,1.5],[[6,4,6],[6,4,6]]);
    assert.equal(bvh.reflectionKernel.fallbacks,2);
  }
  const bvh=new MeshBVH(meshOf(wall),{reflectionBackend:"wasm"});
  compare(bvh,[4,-2,1.5],Array.from({length:REFLECTION_SOURCE_LIMIT+1},()=>[6,4,6]));
  assert.equal(bvh.reflectionKernel.fallbacks,1);
  compare(bvh,[4,-2,1.5],[[6,4,6],[-6,4,6]]);
  assert.equal(bvh.reflectionKernel.fallbacks,1);
});

test("unavailable WebAssembly automatically keeps the JavaScript engine",()=>{
  const url=pathToFileURL(path.resolve(__dirname,"../../docs/assets/simulation/mesh.mjs")).href;
  const source='import assert from "node:assert/strict";import {MeshBVH} from '+JSON.stringify(url)+';globalThis.WebAssembly=undefined;const meshOf='+meshOf.toString()+';const wall='+JSON.stringify(wall)+';const mesh=meshOf(Array.from({length:128},(_,i)=>wall[i%2]));const bvh=new MeshBVH(mesh);assert.equal(bvh.reflectionBackend,"js");const receiver=[4,-2,1.5],sources=[[6,4,6],[-6,4,6]];assert.deepEqual(bvh.reflectionsBatch(receiver,sources),sources.map(s=>bvh.reflections(receiver,s)));assert.throws(()=>new MeshBVH(mesh,{reflectionBackend:"wasm"}),/unavailable/);';
  const result=spawnSync(process.execPath,["--input-type=module","-e",source],{encoding:"utf8",windowsHide:true});
  assert.equal(result.status,0,result.stderr);
});

test("backend selection is explicit and rejects invalid capacity settings",async()=>{
  const [{MeshBVH}]=await modules,mesh=meshOf(wall);
  const js=new MeshBVH(mesh,{reflectionBackend:"js"});assert.equal(js.reflectionBackend,"js");assert.equal(js.reflectionKernel,null);
  compare(js,[4,-2,1.5],[[6,4,6],[-6,4,6]]);
  for(const options of [{reflectionBackend:"other"},{reflectionCapacity:-1},{reflectionCapacity:.5},{reflectionCapacity:65537}])assert.throws(()=>new MeshBVH(mesh,options),RangeError);
});


test("strict SIMD isolates source lanes and never emits an unused odd lane",async()=>{
  const [{MeshBVH}]=await modules;
  const bvh=new MeshBVH(meshOf(wall),{reflectionBackend:"wasm"});
  assert.equal(bvh.reflectionKernel.simd,true);
  const pattern=[[6,4,6],[-6,4,6],[NaN,0,0],[6,-4,3],[Infinity,0,0],[20200000,4,6],[6,4,6]];
  for(const length of [512,511,31,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,3,16,17,512]) {
    compare(bvh,[4,-2,1.5],Array.from({length},(_,i)=>pattern[i%pattern.length]));
  }
  assert.equal(bvh.reflectionKernel.fallbacks,0);
  // The fourth input remains in memory after shortening an even batch to odd.
  // It must neither emit a candidate nor trigger capacity fallback.
  const bounded=new MeshBVH(meshOf(wall),{reflectionBackend:"wasm",reflectionCapacity:0});
  compare(bounded,[4,-2,1.5],[[-6,4,6],[-6,4,6],[-6,4,6],[6,4,6]]);
  assert.equal(bounded.reflectionKernel.fallbacks,1);
  compare(bounded,[4,-2,1.5],[[-6,4,6],[-6,4,6],[-6,4,6]]);
  assert.equal(bounded.reflectionKernel.fallbacks,1);
});

test("SIMD compilation failure keeps scalar Wasm, then JavaScript if both fail",()=>{
  const url=pathToFileURL(path.resolve(__dirname,"../../docs/assets/simulation/mesh.mjs")).href;
  for(const failures of [1,2]){
    const source='import assert from "node:assert/strict";import {MeshBVH} from '+JSON.stringify(url)+';let attempts=0;const failures='+failures+';WebAssembly.Module=new Proxy(WebAssembly.Module,{construct(target,args){if(++attempts<=failures)throw new WebAssembly.CompileError("Backend unavailable");return Reflect.construct(target,args);}});const meshOf='+meshOf.toString()+';const wall='+JSON.stringify(wall)+';const mesh=meshOf(Array.from({length:128},(_,i)=>wall[i%2]));const bvh=new MeshBVH(mesh);assert.equal(attempts,2);assert.equal(bvh.reflectionBackend,failures===1?"wasm":"js");if(failures===1)assert.equal(bvh.reflectionKernel.simd,false);const receiver=[4,-2,1.5];for(const sources of [[[6,4,6],[-6,4,6]],[[NaN,0,0],[6,4,6],[20200000,4,6]]])assert.deepEqual(bvh.reflectionsBatch(receiver,sources),sources.map(s=>bvh.reflections(receiver,s)));if(failures===2)assert.throws(()=>new MeshBVH(mesh,{reflectionBackend:"wasm"}),/unavailable/);assert.equal(attempts,2);';
    const result=spawnSync(process.execPath,["--input-type=module","-e",source],{encoding:"utf8",windowsHide:true});
    assert.equal(result.status,0,result.stderr);
  }
});


test("projection changes within one scan retain triangle order for every packet count",async()=>{
  const [{MeshBVH}]=await modules;
  const triangles=[0,1,2,2,0,1,0,2,1].flatMap(axis=>wall.map(t=>t.map(p=>rotate(p,axis))));
  const bvh=new MeshBVH(meshOf(triangles),{reflectionBackend:"wasm"});
  for(let count=2;count<=18;count++) {
    const sources=Array.from({length:count},(_,i)=>rotate([6,4,6],i%3));
    for(const axis of [0,1,2])compare(bvh,rotate([4,-2,1.5],axis),sources);
  }
  assert.equal(bvh.reflectionKernel.fallbacks,0);
});
