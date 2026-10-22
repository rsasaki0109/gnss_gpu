const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const {gunzipSync} = require("node:zlib");
const path = require("node:path");
const modulePromise = import("../../docs/assets/simulation/arrival-direction.mjs");
const close = (actual,expected,tolerance=1e-11) => assert.ok(Math.abs(actual-expected)<tolerance,`${actual} != ${expected}`);
const identity = [[1,0,0],[0,1,0],[0,0,1]];
const equator = {receiver:[7,8,9],receiverECEF:[6378137,0,0],mask:10};
const signal = (point,overrides={}) => ({azimuth:0,elevation:0,status:"NLOS",reflection:{point},...overrides});
// At latitude/longitude zero, scene ECEF displacement [up,east,north]
// becomes receiver ENU [east,north,up]. This oracle uses no frame helpers.
const equatorialPoint = ([east,north,up]) => [equator.receiver[0]+up,equator.receiver[1]+east,equator.receiver[2]+north];

test("arrival bearings point toward the reflector, preserve the lower hemisphere and handle exact verticals",async()=>{
  const {reflectedArrival}=await modulePromise;
  for(const [direction,azimuth,elevation,separation] of [[[0,2,0],0,0,0],[[2,0,0],90,0,90],[[0,-2,0],180,0,180],[[-2,0,0],270,0,90],[[0,0,2],null,90,90],[[0,0,-2],null,-90,90]]){
    const result=reflectedArrival(identity,equator,signal(equatorialPoint(direction)));
    if(azimuth===null)assert.equal(result.azimuth,null);else close(result.azimuth,azimuth);
    close(result.elevation,elevation);close(result.separation,separation);
  }
});

test("current receiver frame, fixed scene rotation, scene translation and length cannot change a known bearing",async()=>{
  const {reflectedArrival}=await modulePromise;
  const lat=35.8*Math.PI/180,lon=139.9*Math.PI/180,az=51.2*Math.PI/180,el=-17.3*Math.PI/180;
  const east=Math.cos(el)*Math.sin(az),north=Math.cos(el)*Math.cos(az),up=Math.sin(el);
  // Independent WGS84 forward position and explicit tangent-vector formulas.
  const a=6378137,e2=6.6943799901413165e-3,N=a/Math.sqrt(1-e2*Math.sin(lat)**2),height=37;
  const ecef=[(N+height)*Math.cos(lat)*Math.cos(lon),(N+height)*Math.cos(lat)*Math.sin(lon),(N*(1-e2)+height)*Math.sin(lat)];
  const delta=[-Math.sin(lon)*east-Math.sin(lat)*Math.cos(lon)*north+Math.cos(lat)*Math.cos(lon)*up,Math.cos(lon)*east-Math.sin(lat)*Math.sin(lon)*north+Math.cos(lat)*Math.sin(lon)*up,Math.cos(lat)*north+Math.sin(lat)*up];
  const rotation=[[0,1,0],[0,0,1],[1,0,0]];
  for(const length of [2,2000])for(const receiver of [[0,0,0],[-1014.8,-1252.8,.1]]){
    const point=receiver.map((value,i)=>value+length*delta[(i+1)%3]);
    const result=reflectedArrival(rotation,{receiver,receiverECEF:ecef,mask:30},signal(point,{azimuth:51.2,elevation:17.3}));
    close(result.azimuth,51.2);close(result.elevation,-17.3);close(result.separation,34.6);
    // Treating scene axes as receiver axes would miss this by tens of degrees.
    assert.ok(Math.abs(Math.atan2(point[0]-receiver[0],point[1]-receiver[1])*180/Math.PI-51.2)>10);
  }
});

test("a specular wall's arrival agrees with the analytic last leg and 3D angular separation",async()=>{
  const {reflectedArrival}=await modulePromise;
  const receiver=[4,-2,1.5],point=[0,.4,3.3],source=[6,4,6],delta=point.map((v,i)=>v-receiver[i]),direct=source.map((v,i)=>v-receiver[i]);
  const azimuth=(Math.atan2(direct[0],direct[1])*180/Math.PI+360)%360,elevation=Math.atan2(direct[2],Math.hypot(...direct.slice(0,2)))*180/Math.PI;
  const snapshot={receiver,receiverECEF:[6378137,0,0],mask:30},input=signal(point,{azimuth,elevation});
  const before=structuredClone({snapshot,input});
  const result=reflectedArrival([[0,1,0],[0,0,1],[1,0,0]],snapshot,input);
  close(result.azimuth,(Math.atan2(-4,2.4)*180/Math.PI+360)%360);
  close(result.elevation,Math.atan2(1.8,Math.hypot(4,2.4))*180/Math.PI);
  close(result.separation,Math.acos(delta.reduce((sum,v,i)=>sum+v*direct[i],0)/(Math.hypot(...delta)*Math.hypot(...direct)))*180/Math.PI);
  assert.ok(result.elevation<snapshot.mask,"A reflected bearing below the satellite mask is retained");
  assert.deepEqual({snapshot,input},before);
});

test("missing, blocked, coincident and non-finite paths cannot invent a bearing",async()=>{
  const {reflectedArrival}=await modulePromise;
  for(const input of [null,{},signal([8,9,10],{reflection:null}),signal([8,9,10],{status:"BLOCKED"}),signal(equator.receiver),signal([NaN,0,0]),signal([1,2]),signal([8,9,10],{azimuth:Infinity})])assert.equal(reflectedArrival(identity,equator,input),null);
  for(const rotation of [null,[],[[1,0,0],[0,1,0],[0,0,NaN]]])assert.equal(reflectedArrival(rotation,equator,signal([8,9,10])),null);
  assert.equal(reflectedArrival(identity,{...equator,receiverECEF:[NaN,0,0]},signal([8,9,10])),null);
});

test("real full-city arrivals agree with an independent ECEF oracle while the trace stays exactly unchanged",async()=>{
  const {reflectedArrival}=await modulePromise;
  const [{CitySimulation},{decodeMesh}]=await Promise.all([import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
  const folder=path.join(__dirname,"../../docs/assets/plateau"),scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz")))),bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
  // Preserve the original GPS-only numerical regression alongside multi-GNSS tests.
  scene.navigation = scene.navigation.filter(record => record.id.startsWith("G"));
  const sim=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
  for(const settings of [{epoch:8000,antenna:0},{epoch:8000,antenna:20},{epoch:3890,antenna:0,mask:12}]){
    const {antenna}=settings,snapshot=sim.compute(settings),before=structuredClone(snapshot);
    const selected=snapshot.satellites.find(s=>s.id===(antenna?"G10":"G12")),result=reflectedArrival(scene.rotation,snapshot,selected);
    // Convert the full point into ECEF independently, subtract the ECEF
    // receiver, then use explicit current-latitude tangent formulas.
    const point=[0,1,2].map(i=>scene.origin[i]+scene.rotation[0][i]*selected.reflection.point[0]+scene.rotation[1][i]*selected.reflection.point[1]+scene.rotation[2][i]*selected.reflection.point[2]);
    const d=point.map((v,i)=>v-snapshot.receiverECEF[i]),lat=snapshot.latitude*Math.PI/180,lon=snapshot.longitude*Math.PI/180;
    const east=-Math.sin(lon)*d[0]+Math.cos(lon)*d[1],north=-Math.sin(lat)*Math.cos(lon)*d[0]-Math.sin(lat)*Math.sin(lon)*d[1]+Math.cos(lat)*d[2],up=Math.cos(lat)*Math.cos(lon)*d[0]+Math.cos(lat)*Math.sin(lon)*d[1]+Math.sin(lat)*d[2];
    close(result.azimuth,(Math.atan2(east,north)*180/Math.PI+360)%360,1e-8);
    close(result.elevation,Math.atan2(up,Math.hypot(east,north))*180/Math.PI,1e-8);
    assert.deepEqual(snapshot,before);assert.deepEqual(sim.compute(settings),before);
    if(settings.epoch===3890){
      assert.equal(selected.status,"NLOS");assert.ok(selected.elevation>12);assert.ok(result.elevation>0&&result.elevation<12);
      close(result.elevation,11.95371689634098);
    }else if(!antenna){
      close(result.azimuth,127.0923929127461);close(result.elevation,14.65096153366911);close(result.separation,36.482981271152035);
      assert.equal(selected.status,"NLOS");assert.deepEqual(snapshot.counts,{LOS:4,NLOS:1,BLOCKED:3});
      close(selected.reflection.extra,23.037548400461674,1e-6);close(snapshot.spp.all.horizontal,65.7406968442433,1e-5);
    }else{
      assert.equal(selected.status,"LOS");assert.ok(selected.reflection.extra>0);close(result.elevation,-12.143554110881492);
      assert.deepEqual(snapshot.counts,{LOS:7,NLOS:0,BLOCKED:1});
    }
  }
});
