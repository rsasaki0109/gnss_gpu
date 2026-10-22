const test = require("node:test");
const assert = require("node:assert/strict");
const {gunzipSync} = require("node:zlib");
const fs = require("node:fs");
const modules = import("../../docs/assets/simulation/signal-history.mjs");
const signal = (status="LOS",range=20000000,extra=100) => ({id:"G01",status,range,reflection:{extra}});
const snapshot = (tow=100, overrides={}) => ({week:2032,tow,epoch:Math.round(tow*10),antenna:0,mask:10,systems:"GECJR",satellites:[signal()],...overrides});

test("relative power follows inverse square distance and reflected field squared without inventing blocked power",async()=>{
  const {relativePower}=await modules;
  assert.equal(relativePower(signal()),0);
  assert.ok(Math.abs(relativePower(signal("LOS",40000000))+6.020599913279624)<1e-12);
  assert.ok(Math.abs(relativePower(signal("NLOS",20000000,0))+6.020599913279624)<1e-12);
  assert.ok(relativePower(signal("NLOS")) < relativePower(signal("NLOS",20000000,0)));
  assert.equal(relativePower(signal("BLOCKED")),null);
  for(const range of [0,-1,NaN,Infinity])assert.throws(()=>relativePower(signal("LOS",range)),RangeError);
  assert.throws(()=>relativePower({...signal("NLOS"),reflection:null}),RangeError);
});

test("history preserves real GPST spacing, missing channels, full precision and immutable inputs",async()=>{
  const {SignalHistory}=await modules, h=new SignalHistory();
  const a=snapshot(100.1234567),before=structuredClone(a);h.append(a);
  h.append(snapshot(100.8234567,{satellites:[{...signal("BLOCKED"),id:"R06"}]}));
  assert.deepEqual(a,before);
  assert.equal(h.samples[0].time,2032*604800+100.1234567);
  assert.equal(h.samples[1].signals[0].power,null);
  assert.equal(h.samples[1].signals.some(s=>s.id==="G01"),false);
  h.append(snapshot(100.8234567));assert.equal(h.samples.length,2);
  h.append(snapshot(90));assert.equal(h.samples.length,1);
  for(const overrides of [{antenna:2},{mask:15},{systems:"R"}]){h.append(snapshot(100,overrides));assert.equal(h.samples.length,1);}
});

test("history handles week rollover, bounded memory, settings identity and explicit clear",async()=>{
  const {SignalHistory}=await modules,h=new SignalHistory();
  h.append(snapshot(604799.9));h.append(snapshot(0.1,{week:2033}));assert.equal(h.samples.length,2);
  h.clear();for(let i=0;i<700;i++)h.append(snapshot(100+i*.1));assert.equal(h.samples.length,600);
  h.append(snapshot(500));assert.equal(h.samples.length,1);
  assert.throws(()=>h.append(snapshot(NaN)),RangeError);
  h.clear();assert.equal(h.samples.length,0);
});

test("real five-system completed city snapshots remain byte-for-byte unchanged by display power/history",async()=>{
  const [{SignalHistory},{CitySimulation},{decodeMesh}] = await Promise.all([modules,import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
  const base=__dirname+"/../../docs/assets/plateau/";
  const scene=JSON.parse(gunzipSync(fs.readFileSync(base+"odaiba-scene.json.gz")));
  const mesh=gunzipSync(fs.readFileSync(base+"odaiba-mesh.bin.gz"));
  const result=new CitySimulation(scene,decodeMesh(mesh.buffer.slice(mesh.byteOffset,mesh.byteOffset+mesh.byteLength))).compute({epoch:8000,antenna:0,mask:10,systems:"GECJR"});
  const before=structuredClone(result),h=new SignalHistory();h.append(result);
  assert.deepEqual(result,before);assert.deepEqual(result.counts,{LOS:21,NLOS:4,BLOCKED:9});
  assert.equal(result.spp.all.horizontal,23.369187327492245);
  assert.deepEqual([...new Set(h.samples[0].signals.map(s=>s.id[0]))].sort(),["C","E","G","J","R"]);
});

test("history CSV preserves captured receiver settings and full precision",async()=>{
  const {SignalHistory,historyCSV}=await modules,h=new SignalHistory();
  h.append(snapshot(123.123456789,{antenna:2.1,mask:15,systems:"R"}));
  const csv=historyCSV(h.samples);
  assert.ok(csv.includes("123.123456789"));assert.ok(csv.includes("systems,antenna_offset_m,mask_deg"));assert.ok(csv.includes("fixed,rhcp,isotropic,R,2.1,15"));
});
