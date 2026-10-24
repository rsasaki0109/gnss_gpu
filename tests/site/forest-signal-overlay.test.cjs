const {test}=require("node:test");
const assert=require("node:assert/strict");
const modulePromise=import("../../docs/assets/forest/signal-overlay.mjs");
const camera={yaw:0,pitch:-.08,zoom:1,target:[0,0,1.6],follow:true};
const receiver=[0,0,1.6];
const signal=(id,direction,classification="CLEAR")=>({id,trace:{direction,classification,canopyLoss:2.5,crowns:[],wood:[]}});

test("receiver-follow rays have visible length instead of collapsing at the camera origin",async()=>{
  const {forestCameraFrame,projectForestSegment}=await modulePromise,frame=forestCameraFrame(camera,900,600);
  const satellite=[0,100,70],path=projectForestSegment(receiver,satellite,frame);
  assert.ok(path&&Math.hypot(path[1][0]-path[0][0],path[1][1]-path[0][1])>30);
  // The former receiver-eye camera collapses the same geometrical ray.
  assert.equal(projectForestSegment(receiver,satellite,{...frame,eye:receiver}),null);
});

test("looking toward a high satellite does not put the observer underground",async()=>{
  const {forestCameraFrame}=await modulePromise;
  const frame=forestCameraFrame({...camera,pitch:-1.4},900,600);
  assert.equal(frame.eye[2],receiver[2]+1);
  assert.ok(Math.abs(Math.hypot(frame.eye[0]-receiver[0],frame.eye[1]-receiver[1])-6)<1e-10);
});

test("near-plane and viewport clipping produce finite, bounded visible segments",async()=>{
  const {forestCameraFrame,projectForestSegment}=await modulePromise,frame=forestCameraFrame(camera,900,600);
  const path=projectForestSegment([3,-20,3],[0,40,8],frame);
  assert.ok(path);
  for(const p of path){assert.ok(p.every(Number.isFinite));assert.ok(p[0]>=7.999&&p[0]<=892.001&&p[1]>=7.999&&p[1]<=592.001);}
  assert.equal(projectForestSegment([0,-30,2],[0,-20,2],frame),null);
});

test("all/selected/off retain blocked classification and exact selected canopy endpoints",async()=>{
  const {forestCameraFrame,forestSignalOverlay,projectForestSegment}=await modulePromise,frame=forestCameraFrame(camera,900,600);
  const first=signal("G01",[0,.8,.6],"CANOPY"),second=signal("R02",[.3,.8,.5],"WOOD");
  first.trace.crowns=[{enter:3,exit:7}];const saved=structuredClone([first,second]);
  const all=forestSignalOverlay(receiver,[first,second],"G01",null,frame);
  assert.equal(all.labels.length,2);assert.ok(all.lines.some(p=>p.satellite==="R02"&&p.dashed));
  const canopy=all.lines.find(p=>p.kind==="canopy");
  const expected=projectForestSegment([0,2.4,3.4],[0,5.6,5.8],frame);
  assert.ok(canopy&&expected);
  for(let j=0;j<2;j++)for(let k=0;k<2;k++)assert.ok(Math.abs([canopy.a,canopy.b][j][k]-expected[j][k])<1e-8);
  assert.deepEqual([first,second],saved);
  assert.equal(forestSignalOverlay(receiver,[first,second],"G01",null,frame,"selected").labels.length,1);
  const hidden=forestSignalOverlay(receiver,[first,second],"G01",null,frame,"off");assert.equal(hidden.lines.length,0);assert.equal(hidden.receiver,null);
});

test("ground reflection draws both physical legs",async()=>{
  const {forestCameraFrame,forestSignalOverlay}=await modulePromise,frame=forestCameraFrame({...camera,follow:false,pitch:.62},900,600);
  const ground={point:[4,4,0],legs:{satellite:{direction:[.2,.4,.8]}}};
  const packet=forestSignalOverlay(receiver,[signal("G01",[.2,.4,.8])],"G01",ground,frame);
  assert.equal(packet.lines.filter(p=>p.kind==="reflection").length,2);assert.ok(packet.reflection);
});
