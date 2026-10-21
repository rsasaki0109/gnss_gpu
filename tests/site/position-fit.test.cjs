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
  const simulation=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
  for(const settings of [{epoch:8000},{epoch:8000,antenna:20},{epoch:8000,mask:30},{epoch:3890,mask:12}]){
    const snapshot=freeze(simulation.compute(settings)),before=structuredClone(snapshot),result=positionFit(snapshot,view(-.63,.82,800,565,snapshot.receiver));
    if(settings.antenna){assert.equal(result.state,"coincident");assert.equal(result.horizontal,0);assert.equal(result.spatial,0);assert.deepEqual(snapshot.counts,{LOS:7,NLOS:0,BLOCKED:1});}
    else if(settings.mask)assert.deepEqual(result,{state:"no-solution"});
    else{assert.equal(result.state,"ready");assert.equal(result.horizontal,65.7406968442433);assert.equal(result.spatial,116.37532039628215);assert.deepEqual(result.position,[-955.3187130610248,-1280.7911607311346,96.12858487844296]);assert.deepEqual(snapshot.counts,{LOS:4,NLOS:1,BLOCKED:3});}
    assert.deepEqual(snapshot,before);
  }
});
