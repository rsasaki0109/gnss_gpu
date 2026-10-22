const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");
const modulePromise = import("../../docs/assets/simulation/position-fit.mjs");
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
function fixture() { return {receiver:[10,20,-30],spp:{all:{position:[13,24,-42],offset:[3,4,-12],horizontal:5,spatial:13}}}; }
function view(yaw=0,pitch=0,width=800,height=500,target=[0,0,0],scale=2) {
  return {width,height,target,scale,right:[Math.cos(yaw),-Math.sin(yaw),0],up:[-Math.sin(yaw)*Math.sin(pitch),-Math.cos(yaw)*Math.sin(pitch),Math.cos(pitch)]};
}
function freeze(value) { if(value&&typeof value==="object"){Object.values(value).forEach(freeze);Object.freeze(value);} return value; }

test("stored 3D positions follow the camera's analytic orthographic projection, including negative up",async()=>{
  const {positionFit}=await modulePromise,snapshot=freeze(fixture()),before=structuredClone(snapshot);
  for(const yaw of [-.63,0,2.5])for(const pitch of [-.82,.15,1.48])for(const [width,height] of [[800,565],[378,385]]){
    const target=[7,-9,11],scale=.73,result=positionFit(snapshot,view(yaw,pitch,width,height,target,scale));
    const oracle=point=>{const [x,y,z]=point.map((value,i)=>value-target[i]);return [width/2+scale*(x*Math.cos(yaw)-y*Math.sin(yaw)),height*.56+scale*(Math.sin(pitch)*(x*Math.sin(yaw)+y*Math.cos(yaw))-z*Math.cos(pitch))];};
    result.fitPoint.forEach((value,i)=>close(value,oracle(snapshot.spp.all.position)[i]));
    result.referencePoint.forEach((value,i)=>close(value,oracle(snapshot.receiver)[i]));
    assert.equal(result.up,-12);assert.equal(result.spatial,13);assert.equal(result.horizontal,5);
    assert.deepEqual(result.position,[13,24,-42]);
  }
  assert.deepEqual(snapshot,before);
});

test("Float64 stored positions survive large scene translations without offset reconstruction or mutation",async()=>{
  const {positionFit}=await modulePromise,snapshot=fixture();
  snapshot.receiver=[1e7,-2e7,3e7];snapshot.spp.all.position=[1e7+1e-7,-2e7-2e-7,3e7+3e-7];
  // Deliberately different redundant offsets prove that the stored position is authoritative.
  snapshot.spp.all.offset=[1,2,3];const before=structuredClone(snapshot),result=positionFit(freeze(snapshot),view(0,.82,800,565,snapshot.receiver));
  assert.deepEqual(result.position,before.spp.all.position);assert.notEqual(result.fitPoint[0],result.referencePoint[0]);
  assert.deepEqual(snapshot,before);assert.notStrictEqual(result.position,snapshot.spp.all.position);
});

test("only exact world coincidence hides displacement; screen overlap retains a distinct solution",async()=>{
  const {positionFit}=await modulePromise,snapshot=fixture(),frame=view(0,0);
  snapshot.spp.all.position=[...snapshot.receiver];snapshot.spp.all.offset=[0,0,0];snapshot.spp.all.horizontal=0;snapshot.spp.all.spatial=0;
  assert.equal(positionFit(snapshot,frame).state,"coincident");
  snapshot.spp.all.position[1]+=1;snapshot.spp.all.offset[1]=1;snapshot.spp.all.horizontal=1;snapshot.spp.all.spatial=1;
  const result=positionFit(snapshot,frame);assert.equal(result.state,"ready");assert.deepEqual(result.fitPoint,result.referencePoint);
  snapshot.spp.all.position[0]+=1e-10;assert.equal(positionFit(snapshot,frame).state,"ready");
});

test("offscreen fits keep raw coordinates and full height instead of being clamped to the scene",async()=>{
  const {positionFit}=await modulePromise,snapshot=fixture();snapshot.spp.all.position=[1000,24,-2000];
  const result=positionFit(snapshot,view());assert.equal(result.state,"offscreen");
  assert.deepEqual(result.fitPoint,[2400,4280]);assert.deepEqual(result.position,[1000,24,-2000]);
  assert.deepEqual(positionFit(snapshot,view(),{focused:true}),{state:"surface"});
});

test("pending, failed, missing and invalid fits expose no stale coordinate or numeric fields",async()=>{
  const {positionFit}=await modulePromise,snapshot=fixture();
  assert.deepEqual(positionFit(null,null),{state:"loading"});
  assert.deepEqual(positionFit(snapshot,null,{ready:false}),{state:"pending"});
  assert.deepEqual(positionFit(snapshot,null,{error:true}),{state:"unavailable"});
  assert.deepEqual(positionFit(snapshot,null,{focused:true}),{state:"surface"});
  assert.deepEqual(positionFit({spp:{all:null}},view()),{state:"no-solution"});
  for(const change of [s=>s.receiver[0]=NaN,s=>s.spp.all.position[1]=Infinity,s=>s.spp.all.offset=null,s=>s.spp.all.horizontal=-1,s=>s.spp.all.spatial=NaN]){
    const invalid=fixture();change(invalid);assert.deepEqual(positionFit(invalid,view()),{state:"unavailable"});
  }
  for(const frame of [null,{...view(),scale:0},{...view(),width:Infinity},{...view(),up:[NaN,0,1]}])assert.deepEqual(positionFit(snapshot,frame),{state:"unavailable"});
});

test("full-mesh fits preserve the original bias, coincidence and unavailable solutions without another solve",async()=>{
  const {positionFit}=await modulePromise;
  const [{CitySimulation},{decodeMesh}]=await Promise.all([import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
  const folder=path.join(__dirname,"../../docs/assets/plateau"),scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz")))),bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
  // Preserve the original GPS-only numerical regression alongside multi-GNSS tests.
  scene.navigation = scene.navigation.filter(record => record.id.startsWith("G"));
  const simulation=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
  for(const settings of [{epoch:8000},{epoch:8000,antenna:20},{epoch:8000,mask:30},{epoch:3890,mask:12}]){
    const snapshot=freeze(simulation.compute(settings)),before=structuredClone(snapshot),result=positionFit(snapshot,view(-.63,.82,800,565,snapshot.receiver));
    if(settings.antenna){assert.equal(result.state,"coincident");assert.equal(result.horizontal,0);assert.equal(result.spatial,0);assert.deepEqual(snapshot.counts,{LOS:7,NLOS:0,BLOCKED:1});}
    else if(settings.mask)assert.deepEqual(result,{state:"no-solution"});
    else{assert.equal(result.state,"ready");assert.equal(result.horizontal,65.7406968442433);assert.equal(result.spatial,116.37532039628215);assert.deepEqual(result.position,[-955.3187130610248,-1280.7911607311346,96.12858487844296]);assert.deepEqual(snapshot.counts,{LOS:4,NLOS:1,BLOCKED:3});}
    assert.deepEqual(snapshot,before);
  }
});


test("pinned projection preserves its own stored position and receiver under the current camera",async()=>{
  const [{pinnedPositionFit,positionFit},{captureReference}]=await Promise.all([modulePromise,import("../../docs/assets/simulation/comparison.mjs")]);
  const low={...fixture(),epoch:8000,week:2032,tow:274175.1,antenna:0,mask:10,counts:{LOS:4,NLOS:1,BLOCKED:3},satellites:[]};
  const high=structuredClone(low);high.antenna=20;high.receiver=[10.001,20.002,-10];high.spp.all={position:[...high.receiver],offset:[0,0,0],horizontal:0,spatial:0};
  const pin=captureReference(freeze(low)),before=structuredClone(high);
  for(const frame of [view(-.63,.82,800,565,high.receiver),view(2.1,-.5,378,385,[50,-70,90]),view(0,0,800,565,high.receiver,1000)]){
    const result=pinnedPositionFit(pin,freeze(high),frame,{enabled:true});assert.deepEqual(result,positionFit(low,frame));
    assert.deepEqual(result.receiver,low.receiver);assert.notDeepEqual(result.receiver,high.receiver);assert.deepEqual(result.position,low.spp.all.position);
  }
  assert.deepEqual(high,before);
  const offscreen=pinnedPositionFit(pin,high,view(0,0,800,565,high.receiver,1000),{enabled:true});assert.equal(offscreen.state,"offscreen");assert.ok(offscreen.fitPoint[0]>800);
});

test("pinned comparisons require completed matching times, distinct settings and both real solutions",async()=>{
  const {pinnedPositionFit}=await modulePromise;
  const current={...fixture(),epoch:8000,week:2032,tow:274175.1,antenna:20,mask:10};
  const pin={epoch:8000,week:2032,tow:274175.1,antenna:0,mask:10,receiver:[1,2,3],fit:fixture().spp.all};
  const state=(reference=pin,snapshot=current,options={})=>pinnedPositionFit(reference,snapshot,null,{enabled:true,...options});
  assert.deepEqual(state(pin,current,{enabled:false}),{state:"closed"});assert.deepEqual(state(null),{state:"empty"});
  assert.deepEqual(state(pin,null),{state:"loading"});assert.deepEqual(state(pin,current,{ready:false}),{state:"pending"});
  assert.deepEqual(state(pin,current,{error:true}),{state:"unavailable"});assert.deepEqual(state(pin,current,{focused:true}),{state:"surface"});
  for(const patch of [{epoch:8001},{week:2033},{tow:current.tow+.1},{tow:NaN}])assert.deepEqual(state(pin,{...current,...patch}),{state:"different-pose"});
  assert.deepEqual(state(pin,{...current,antenna:0}),{state:"same-settings"});
  assert.deepEqual(state(pin,{...current,spp:{all:null}}),{state:"no-current-solution"});
  assert.deepEqual(state({...pin,fit:null}),{state:"no-solution"});
  assert.deepEqual(pinnedPositionFit({...pin,receiver:null},current,view(),{enabled:true}),{state:"unavailable"});
});

test("real height comparisons project the exact pinned full-mesh fit without rebasing its bias",async()=>{
  const [{pinnedPositionFit,positionFit},{captureReference},{CitySimulation},{decodeMesh}]=await Promise.all([modulePromise,import("../../docs/assets/simulation/comparison.mjs"),import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
  const folder=path.join(__dirname,"../../docs/assets/plateau"),scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz")))),bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
  // Preserve the original GPS-only numerical regression alongside multi-GNSS tests.
  scene.navigation = scene.navigation.filter(record => record.id.startsWith("G"));
  const simulation=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
  const low=freeze(simulation.compute({epoch:8000})),high=freeze(simulation.compute({epoch:8000,antenna:20})),masked=freeze(simulation.compute({epoch:8000,mask:30}));
  const before=[low,high,masked].map(value=>structuredClone(value));
  for(const [previous,current] of [[low,high],[high,low]]){
    const pin=captureReference(previous),frame=view(-.63,.82,800,565,current.receiver),result=pinnedPositionFit(pin,current,frame,{enabled:true});
    assert.deepEqual(result,positionFit(previous,frame));assert.deepEqual(result.position,previous.spp.all.position);assert.deepEqual(result.receiver,previous.receiver);
    assert.equal(result.horizontal,previous.spp.all.horizontal);assert.equal(result.spatial,previous.spp.all.spatial);
    assert.notDeepEqual(result.referencePoint,positionFit(current,frame).referencePoint);
  }
  assert.equal(pinnedPositionFit(captureReference(low),high,view(0,0,800,565,high.receiver),{enabled:true}).horizontal,65.7406968442433);
  assert.deepEqual(pinnedPositionFit(captureReference(masked),low,view(),{enabled:true}),{state:"no-solution"});
  assert.deepEqual(pinnedPositionFit(captureReference(low),masked,view(),{enabled:true}),{state:"no-current-solution"});
  [low,high,masked].forEach((value,i)=>assert.deepEqual(value,before[i]));
});
