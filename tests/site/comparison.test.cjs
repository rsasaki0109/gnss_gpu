const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");
const modules = Promise.all([import("../../docs/assets/simulation/comparison.mjs"),import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
let casesPromise;
function cases() {
  if (!casesPromise) casesPromise = modules.then(([ ,{CitySimulation},{decodeMesh}])=>{
    const folder=path.join(__dirname,"../../docs/assets/plateau"),scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz")))),bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
    const simulation=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
    return [simulation.compute({epoch:8000,antenna:0,mask:10}),simulation.compute({epoch:8000,antenna:20,mask:10}),simulation.compute({epoch:8000,antenna:0,mask:30})];
  });
  return casesPromise;
}

test("pinning retains full precision and detaches the small immutable summary from its source",async()=>{
  const [{captureReference}]=await modules,[original]=await cases(),source=structuredClone(original),reference=captureReference(source);
  assert.equal(reference.bias,65.7406968442433);
  assert.equal(reference.satellites.find(s=>s.id==="G12").excess,23.037548400461674);
  source.counts.LOS=99;source.satellites.find(s=>s.id==="G12").reflection.extra=999;source.spp.all.horizontal=999;
  assert.equal(reference.counts.LOS,4);assert.equal(reference.bias,original.spp.all.horizontal);
  assert.equal(reference.satellites.find(s=>s.id==="G12").excess,original.satellites.find(s=>s.id==="G12").reflection.extra);
  assert.ok(Object.isFrozen(reference)&&Object.isFrozen(reference.counts)&&Object.isFrozen(reference.satellites)&&reference.satellites.every(Object.isFrozen));
});

test("real height changes compare signed counts and bias while direct LOS excess stays zero",async()=>{
  const [{captureReference,compareReceiver}]=await modules,[low,high]=await cases(),result=compareReceiver(captureReference(low),high);
  assert.equal(result.state,"ready");assert.equal(result.changes,3);
  assert.deepEqual(result.metrics.map(({key,before,after,delta})=>[key,before,after,delta]),[["LOS",4,7,3],["NLOS",1,0,-1],["BLOCKED",3,1,-2],["bias",65.7406968442433,0,-65.7406968442433]]);
  assert.deepEqual(result.signals.find(s=>s.id==="G12"),{id:"G12",before:"NLOS",after:"LOS",beforeExcess:23.037548400461674,afterExcess:0});
  assert.ok(high.satellites.find(s=>s.id==="G10").reflection.extra>3);
  assert.equal(result.signals.find(s=>s.id==="G10").afterExcess,0);
  const reverse=compareReceiver(captureReference(high),low);
  reverse.metrics.forEach((metric,i)=>assert.equal(metric.delta,-result.metrics[i].delta));
  const same=compareReceiver(captureReference(low),low);
  assert.equal(same.changes,0);assert.ok(same.metrics.every(metric=>metric.delta===0));
});

test("mask exclusions remain distinct from blocked paths and insufficient geometry never becomes zero bias",async()=>{
  const [{captureReference,compareReceiver}]=await modules,[low,,masked]=await cases(),result=compareReceiver(captureReference(low),masked);
  assert.deepEqual(masked.counts,{LOS:3,NLOS:0,BLOCKED:2});assert.equal(masked.spp.all,null);
  assert.equal(result.signals.length,8);assert.equal(result.changes,3);
  assert.deepEqual(result.signals.find(s=>s.id==="G12"),{id:"G12",before:"NLOS",after:"MASKED",beforeExcess:23.037548400461674,afterExcess:null});
  assert.deepEqual(result.signals.find(s=>s.id==="G21"),{id:"G21",before:"BLOCKED",after:"BLOCKED",beforeExcess:null,afterExcess:null});
  assert.deepEqual(result.metrics.find(m=>m.key==="bias"),{key:"bias",before:65.7406968442433,after:null,delta:null});
  assert.equal(compareReceiver(captureReference(masked),low).metrics.find(m=>m.key==="bias").delta,null);
});

test("comparisons require the exact original epoch and GPS time and do not expose cross-pose differences",async()=>{
  const [{captureReference,compareReceiver}]=await modules,[snapshot]=await cases(),reference=captureReference(snapshot);
  assert.deepEqual(compareReceiver(null,snapshot),{state:"empty"});assert.deepEqual(compareReceiver(reference,null),{state:"empty"});
  for(const patch of [{epoch:8001},{tow:snapshot.tow+.1},{week:snapshot.week+1}])assert.deepEqual(compareReceiver(reference,{...snapshot,...patch}),{state:"different-pose"});
});
