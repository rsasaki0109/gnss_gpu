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
  // Preserve the original GPS-only numerical regression alongside multi-GNSS tests.
  scene.navigation = scene.navigation.filter(record => record.id.startsWith("G"));
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


test("a pinned fit keeps detached immutable world coordinates and both displacement metrics",async()=>{
  const [{captureReference}]=await modules,[original]=await cases(),source=structuredClone(original),reference=captureReference(source);
  assert.deepEqual(reference.receiver,original.receiver);assert.deepEqual(reference.fit.position,original.spp.all.position);assert.deepEqual(reference.fit.offset,original.spp.all.offset);
  assert.equal(reference.fit.horizontal,65.7406968442433);assert.equal(reference.fit.spatial,116.37532039628215);
  assert.notStrictEqual(reference.receiver,source.receiver);assert.notStrictEqual(reference.fit.position,source.spp.all.position);assert.notStrictEqual(reference.fit.offset,source.spp.all.offset);
  source.receiver[0]+=100;source.spp.all.position[0]+=200;source.spp.all.offset[2]-=300;source.spp.all.spatial=999;
  assert.deepEqual(reference.receiver,original.receiver);assert.deepEqual(reference.fit.position,original.spp.all.position);assert.deepEqual(reference.fit.offset,original.spp.all.offset);
  assert.equal(reference.fit.spatial,original.spp.all.spatial);
  for(const value of [reference.receiver,reference.fit,reference.fit.position,reference.fit.offset])assert.ok(Object.isFrozen(value));
});

test("missing or invalid pinned geometry cannot invent a position while existing table metrics stay intact",async()=>{
  const [{captureReference}]=await modules,[original,,masked]=await cases();
  const missing=captureReference(masked);assert.equal(missing.fit,null);assert.deepEqual(missing.receiver,masked.receiver);assert.equal(missing.bias,null);
  for(const change of [s=>s.spp.all.position[0]=Infinity,s=>s.spp.all.offset[2]=NaN,s=>s.spp.all.spatial=null]){
    const source=structuredClone(original);change(source);const reference=captureReference(source);
    assert.equal(reference.fit,null);assert.equal(reference.bias,original.spp.all.horizontal);assert.deepEqual(reference.counts,original.counts);
  }
  const source=structuredClone(original);source.receiver[0]=NaN;assert.equal(captureReference(source).receiver,null);
});
