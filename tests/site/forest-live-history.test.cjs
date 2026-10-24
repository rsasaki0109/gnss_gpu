const {test}=require("node:test");
const assert=require("node:assert/strict");
const modulePromise=import("../../docs/assets/forest/live-history.mjs");
const sample=(seconds,extra={})=>({seconds,id:"G01",modelKey:"baseline",trace:{classification:"CANOPY",canopyLoss:3,directRelativeDb:-3},reflection:{extra:.8,relativePowerDb:-6,triangle:3,point:[1,2,0]},...extra});

test("duplicate completed epochs replace samples and exports cannot mutate history",async()=>{
  const {ForestLiveHistory}=await modulePromise,h=new ForestLiveHistory(),input=sample(600);
  h.append(input);h.append(input);assert.equal(h.samples.length,1);
  input.reflection.point[0]=99;assert.equal(h.samples[0].groundPoint[0],1);
  const exported=h.snapshot();exported.samples[0].groundPoint[0]=88;assert.equal(h.samples[0].groundPoint[0],1);
});
test("satellite and RF settings changes never join different histories",async()=>{
  const {ForestLiveHistory}=await modulePromise,h=new ForestLiveHistory();h.append(sample(600));h.append(sample(601));
  h.append(sample(601,{id:"R02"}));assert.equal(h.samples.length,1);assert.match(h.reason,/changed/);
  h.append(sample(602,{id:"R02",modelKey:"new density"}));assert.equal(h.samples.length,1);
});
test("seeks and unsampled jumps reset the segment while explicit reset reason survives",async()=>{
  const {ForestLiveHistory}=await modulePromise,h=new ForestLiveHistory();h.append(sample(600));h.append(sample(601));
  h.append(sample(400));assert.equal(h.samples.length,1);assert.match(h.reason,/Timeline/);
  h.append(sample(700));assert.equal(h.samples.length,1);
  h.clear("User moved timeline");h.append(sample(701));assert.equal(h.reason,"User moved timeline");
});
test("blocked direct paths and unavailable ground paths retain nulls rather than zero loss",async()=>{
  const {ForestLiveHistory}=await modulePromise,h=new ForestLiveHistory();h.append(sample(600));
  const missing=h.append(sample(601,{trace:{classification:"WOOD",canopyLoss:5,directRelativeDb:null},reflection:null}));
  assert.equal(missing.directLossDb,null);assert.equal(missing.canopyLossDb,5);assert.equal(missing.reflectionExtraM,null);
  assert.deepEqual(missing.events,["CANOPY → WOOD","Ground path lost"]);
  h.append(sample(602));assert.equal(h.trail()[1],null);
});
test("ground path gain and facet transitions are recorded only at completed epochs",async()=>{
  const {ForestLiveHistory}=await modulePromise,h=new ForestLiveHistory();h.append(sample(600,{reflection:null}));
  assert.deepEqual(h.append(sample(601)).events,["Ground path gained"]);
  const switched=sample(602);switched.reflection.triangle=8;
  assert.deepEqual(h.append(switched).events,["Ground facet changed"]);h.append(switched);assert.equal(h.samples.length,3);
});
test("history and ground-point trail stay bounded during long playback",async()=>{
  const {ForestLiveHistory}=await modulePromise,h=new ForestLiveHistory();
  for(let i=0;i<1500;i++)h.append(sample(i*.1));
  assert.equal(h.samples.length,600);assert.equal(h.trail().length,30);
  assert.ok(h.samples.at(-1).seconds-h.samples[0].seconds<=120);
  assert.throws(()=>h.append(sample(NaN)),RangeError);
});
test("historical ground-point overlay stops at missing points and hides with signal paths",async()=>{
  const {forestCameraFrame,forestSignalOverlay}=await import("../../docs/assets/forest/signal-overlay.mjs");
  const frame=forestCameraFrame({yaw:0,pitch:.6,zoom:1,target:[0,0,0],follow:false},900,600),points=[[0,0,0],[2,0,0],null,[4,0,0],[6,0,0]];
  const packet=forestSignalOverlay([0,0,1.6],[],"G01",null,frame,"all",points);
  assert.equal(packet.lines.filter(p=>p.kind==="reflection-trail").length,2);
  assert.equal(forestSignalOverlay([0,0,1.6],[],"G01",null,frame,"off",points).lines.length,0);
});
