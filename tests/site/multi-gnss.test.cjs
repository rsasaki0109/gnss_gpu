const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");
const folder = path.join(__dirname,"../../docs/assets/plateau");
const scene = JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz"))));
const modules = Promise.all([import("../../docs/assets/simulation/orbit.mjs?v=multi-gnss-1"),import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
test("four-system broadcast positions and signal clocks agree with independent native CPU references",async()=>{
  const [{orbit}]=await modules;
  const fixture=JSON.parse(fs.readFileSync(path.join(__dirname,"fixtures/multi-gnss-orbits.json")));
  assert.equal(fixture.cases.length,207);
  assert.deepEqual([...new Set(fixture.cases.map(c=>c.record.id[0]))].sort(),["C","E","G","J"]);
  let maximum=0;
  for(const c of fixture.cases){
    const actual=orbit(c.record,c.time,c.week),error=Math.hypot(...actual.position.map((v,i)=>v-c.position[i]));
    assert.ok(error<1e-6,`${c.record.id}: ${error} m`);
    assert.ok(Math.abs(actual.clock-c.clock)<1e-14,`${c.record.id}: clock`);
    maximum=Math.max(maximum,error);
  }
  console.log(`Multi-GNSS CPU reference: max ECEF difference ${maximum} m over 207 cases`);
});
test("Galileo E1 validity, transmission causality, BDT week/seconds offsets and GEO dispatch",async()=>{
  const [{selectNavigation,orbit,validateSystems}]=await modules;
  const e=structuredClone(scene.navigation.find(r=>r.id[0]==="E"));
  e.parameters[8]=274000;e.parameters[24]=273999;e.parameters[18]=2032;e.parameters[17]=517;e.parameters[21]=0;e.gpga=[0,0,0,2032];
  assert.equal(selectNavigation([e],274100,2032).length,1);
  for(const health of [1,2,4]){const r=structuredClone(e);r.parameters[21]=health;assert.equal(selectNavigation([r],274100,2032).length,0);}
  const otherBand=structuredClone(e);otherBand.parameters[21]=8;assert.equal(selectNavigation([otherBand],274100,2032).length,1);
  const missingTime=structuredClone(e);delete missingTime.gpga;assert.throws(()=>orbit(missingTime,274100,2032),/time correction/);
  const fnav=structuredClone(e);fnav.parameters[17]=258;assert.equal(selectNavigation([fnav],274100,2032).length,0);
  const future=structuredClone(e);future.parameters[24]=274101;assert.equal(selectNavigation([future],274100,2032).length,0);
  const unknown=structuredClone(e);unknown.parameters[24]=0.9999e9;assert.equal(selectNavigation([unknown],274100,2032).length,0);
  const c=structuredClone(scene.navigation.find(r=>r.id==="C01"));
  c.parameters[8]=604790;c.parameters[18]=675;c.parameters[24]=604780;
  assert.equal(selectNavigation([c],5,2032).length,1);
  assert.equal(selectNavigation([c],5,2033).length,0);
  const a=orbit(c,5,2032),b=orbit(c,604805,2031);
  assert.deepEqual(a,b);
  const wrong=structuredClone(c);wrong.id="C06";
  assert.ok(Math.hypot(...orbit(wrong,5,2032).position.map((v,i)=>v-a.position[i]))>100000);
  assert.throws(()=>orbit({...c,id:"R01"},5,2032),/orbit/);
  for(const s of ["","R","GG","GX",null]) assert.throws(()=>validateSystems(s));
});
test("default full-city trace includes four constellations and preserves every GPS-only path",async()=>{
  const [, {CitySimulation},{decodeMesh}]=await modules;
  const bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
  const sim=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
  const settings={epoch:8000,antenna:0,mask:10};
  const multi=sim.compute(settings),gps=sim.compute({...settings,systems:"G"});
  assert.equal(multi.systems,"GECJ");
  assert.deepEqual([...new Set(multi.satellites.map(s=>s.id[0]))].sort(),["C","E","G","J"]);
  assert.deepEqual(multi.satellites.filter(s=>s.id[0]==="G"),gps.satellites);
  assert.deepEqual(gps.counts,{LOS:4,NLOS:1,BLOCKED:3});
  assert.equal(gps.spp.all.horizontal,65.7406968442433);
  for(const system of ["E","C","J"]){
    const single=sim.compute({...settings,systems:system});
    assert.deepEqual(single.satellites,multi.satellites.filter(s=>s.id[0]===system));
  }
  const {captureReference,compareReceiver}=await import("../../docs/assets/simulation/comparison.mjs");
  assert.equal(compareReceiver(captureReference(multi),gps).state,"different-systems");
  assert.equal(multi.satellites.length,27);
  assert.ok(multi.spp.all && Number.isFinite(multi.spp.all.horizontal));
  const snapshot=structuredClone(scene);
  sim.compute({...settings,epoch:0});sim.compute({...settings,epoch:12409});
  assert.deepEqual(scene,snapshot);
  console.log(JSON.stringify({counts:multi.counts,bias:multi.spp.all.horizontal,satellites:multi.satellites.map(s=>s.id)}));
});
test("scene links retain non-GPS selections and reject invalid system selections",async()=>{
 const {DEFAULT_SCENE,writeSceneLink,readSceneLink}=await import("../../docs/assets/simulation/scene-link.mjs?v=multi-gnss-1");
 const manifest=JSON.parse(fs.readFileSync(path.join(folder,"manifest.json")));
 const assets={mesh:manifest.mesh.sha256,data:manifest.scene.sha256,epochs:scene.track.length,satellites:[...new Set(scene.navigation.map(r=>r.id))]};
 for(const [systems,satellite] of [["GECJ","C01"],["E","E01"],["J","J01"]]){
  const state={...DEFAULT_SCENE,systems,satellite};
  const url=new URL(writeSceneLink("https://example.org/simulation.html",state,assets));
  assert.deepEqual(readSceneLink(url.hash,assets),state);
  url.hash += "&systems=G";assert.throws(()=>readSceneLink(url.hash,assets),/duplicate/);
 }
});
