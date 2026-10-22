const test=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {gunzipSync}=require("node:zlib");
const folder=path.join(__dirname,"../../docs/assets/plateau");
const scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz"))));
const fixture=JSON.parse(fs.readFileSync(path.join(__dirname,"fixtures/glonass-orbits.json")));
const modules=Promise.all([import("../../docs/assets/simulation/orbit.mjs?v=glonass-1"),import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
test("GLONASS forward/backward RK4 and clock agree with 168 native CPU cases and 1-second refinement",async()=>{
 const [{orbit}]=await modules;let maximum=0,refinement=0;
 assert.equal(fixture.cases.length,168);assert.equal(new Set(fixture.cases.map(c=>c.record.id)).size,24);
 for(const c of fixture.cases){
  const before=structuredClone(c.record),actual=orbit(c.record,c.time,c.week);
  const error=Math.hypot(...actual.position.map((v,i)=>v-c.position[i]));
  const fine=Math.hypot(...actual.position.map((v,i)=>v-c.refined_position[i]));
  assert.ok(error<1e-6,`${c.record.id}: native ${error} m`);
  assert.ok(fine<.002,`${c.record.id}: refinement ${fine} m`);
  assert.ok(Math.abs(actual.clock-c.clock)<1e-14);
  assert.deepEqual(c.record,before);maximum=Math.max(maximum,error);refinement=Math.max(refinement,fine);
 }
 console.log(JSON.stringify({nativeMaximumMetres:maximum,oneSecondRefinementMaximumMetres:refinement,cases:168}));
});
test("GLONASS selection enforces UTC offset, health, causal frame time and 15-minute validity across week rollover",async()=>{
 const [{selectNavigation,orbit}]=await modules;
 const r=structuredClone(scene.navigation.find(r=>r.id[0]==="R")),time=r.toc%604800+r.utcOffset;
 assert.equal(selectNavigation([r],time,r.week).length,1);
 const unhealthy=structuredClone(r);unhealthy.health=1;assert.equal(selectNavigation([unhealthy],time,r.week).length,0);
 const future=structuredClone(r);future.transmission=r.toc+1;assert.equal(selectNavigation([future],time,r.week).length,0);
 assert.equal(selectNavigation([r],time+900,r.week).length,1);
 assert.equal(selectNavigation([r],time+900.001,r.week).length,0);
 assert.equal(selectNavigation([r],time,r.week+1).length,0);
 const later=structuredClone(r);later.toc+=600; // Already-transmitted future state epoch permits backward integration.
 assert.equal(selectNavigation([later],time,r.week).length,1);
 const zero=orbit(r,time,r.week);assert.deepEqual(zero.position,r.position);assert.equal(zero.clock,r.clock[0]);
 const drift=structuredClone(r);drift.clock[1]=2e-11;
 assert.equal(orbit(drift,time+60,r.week).clock,drift.clock[0]+2e-11*60);
 const offset=structuredClone(r);offset.utcOffset=17;
 assert.ok(Math.hypot(...orbit(offset,time,r.week).position.map((v,i)=>v-zero.position[i]))>1000);
 const rollover=structuredClone(r);rollover.week=2031;rollover.toc=2031*604800+604790;rollover.transmission=rollover.toc-900;
 assert.equal(selectNavigation([rollover],5,2032).length,1);
 assert.deepEqual(orbit(rollover,5,2032),orbit(rollover,604805,2031));
 assert.deepEqual(orbit(rollover,5),orbit(rollover,5,2032));
 const bad=structuredClone(r);bad.position[1]=NaN;
 assert.equal(selectNavigation([bad],time,r.week).length,0);assert.throws(()=>orbit(bad,time,r.week),/GLONASS orbit/);
 assert.throws(()=>orbit(r,time+1801,r.week),/interval/);
});
test("five-system full-city trace preserves every four-system path and satisfies GLONASS light time",async()=>{
 const [{receivedSatellite,selectNavigation,C,norm,sub}, {CitySimulation},{decodeMesh}]=await modules;
 const bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
 const sim=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
 const five=sim.compute({epoch:8000}),four=sim.compute({epoch:8000,systems:"GECJ"}),glo=sim.compute({epoch:8000,systems:"R"});
 assert.equal(five.systems,"GECJR");assert.equal(five.satellites.length,34);
 assert.deepEqual(five.counts,{LOS:21,NLOS:4,BLOCKED:9});
 assert.deepEqual(five.satellites.filter(s=>s.id[0]!=="R"),four.satellites);
 assert.deepEqual(five.satellites.filter(s=>s.id[0]==="R"),glo.satellites);
 assert.equal(glo.satellites.length,7);assert.deepEqual(glo.counts,{LOS:3,NLOS:3,BLOCKED:1});
 for(const record of selectNavigation(scene.navigation,five.tow,five.week).filter(r=>r.id[0]==="R")){
  const state=receivedSatellite(record,five.tow,five.receiverECEF,five.week);
  assert.ok(Math.abs(norm(sub(state.position,five.receiverECEF))/C-state.transitTime)<=1e-13);
 }
 console.log(JSON.stringify({defaultBias:five.spp.all.horizontal,glonassBias:glo.spp.all.horizontal}));
});
test("GLONASS scene links round-trip while retaining the previous four-system selection",async()=>{
 const {DEFAULT_SCENE,readSceneLink,writeSceneLink}=await import("../../docs/assets/simulation/scene-link.mjs?v=glonass-1");
 const manifest=JSON.parse(fs.readFileSync(path.join(folder,"manifest.json")));
 const assets={mesh:manifest.mesh.sha256,data:manifest.scene.sha256,epochs:scene.track.length,satellites:[...new Set(scene.navigation.map(r=>r.id))]};
 for(const systems of ["R","GECJR","GECJ"]){
  const state={...DEFAULT_SCENE,systems,satellite:systems==="GECJ"?"G12":"R06"};
  const hash=new URL(writeSceneLink("https://example.org/simulation.html",state,assets)).hash;
  assert.deepEqual(readSceneLink(hash,assets),state);
 }
});
