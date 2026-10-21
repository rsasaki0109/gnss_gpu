const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");
const modulePromise = import("../../docs/assets/simulation/reflection-geometry.mjs");
const close = (a,b,tolerance=1e-11) => assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
const vertices = [[0,-10,0],[0,10,0],[0,10,10]];
function fixture(receiver=[4,-2,1.5],source=[6,4,6],point=[0,.4,3.3],face=vertices) {
  return {snapshot:{receiver},signal:{status:"NLOS",source,reflection:{point,vertices:face}}};
}
function inspect(fn,input) { return fn(input.snapshot,input.signal); }

test("an analytic specular wall preserves both physical lengths and the actual normal angles",async()=>{
  const {reflectionGeometry}=await modulePromise,input=fixture(),before=structuredClone(input),result=inspect(reflectionGeometry,input);
  close(result.sourceLeg,7.5);close(result.receiverLeg,5);
  close(result.incidence,36.86989764584401);close(result.reflection,36.86989764584401);close(result.mismatch,0);
  result.incident.forEach((v,i)=>close(v,[-.6,.8][i]));result.outgoing.forEach((v,i)=>close(v,[.6,.8][i]));
  assert.deepEqual(input,before);
});

test("normal incidence, grazing incidence and tiny triangles stay finite without arbitrary area cutoffs",async()=>{
  const {reflectionGeometry}=await modulePromise;
  const normal=inspect(reflectionGeometry,fixture([4,0,0],[6,0,0],[0,0,0]));
  close(normal.incidence,0);close(normal.reflection,0);close(normal.mismatch,0);
  assert.deepEqual(normal.incident,[0,1]);assert.deepEqual(normal.outgoing,[0,1]);
  const grazing=inspect(reflectionGeometry,fixture([1,1e8,0],[1,-1e8,0],[0,0,0]));
  close(grazing.incidence,Math.atan(1e8)*180/Math.PI);close(grazing.reflection,grazing.incidence);close(grazing.mismatch,0);
  const tiny=inspect(reflectionGeometry,fixture([4,0,0],[6,0,0],[0,0,0],[[0,0,0],[0,1e-9,0],[0,0,1e-9]]));
  close(tiny.incidence,0);close(tiny.reflection,0);close(tiny.mismatch,0);
});

test("triangle winding, rotation and translation do not alter the physical checks",async()=>{
  const {reflectionGeometry}=await modulePromise,input=fixture(),reference=inspect(reflectionGeometry,input);
  const rotated = ([x,y,z]) => [y,z,x];
  for(const face of [vertices,vertices.toReversed()])for(const translation of [[0,0,0],[1234,-5678,90]]){
    const transform = point => rotated(point).map((v,i)=>v+translation[i]);
    const result=inspect(reflectionGeometry,fixture(transform(input.snapshot.receiver),transform(input.signal.source),transform(input.signal.reflection.point),face.map(transform)));
    for(const name of ["sourceLeg","receiverLeg","incidence","reflection","mismatch"])close(result[name],reference[name],1e-10);
    result.incident.forEach((v,i)=>close(v,reference.incident[i]));result.outgoing.forEach((v,i)=>close(v,reference.outgoing[i]));
  }
});

test("equal normal angles cannot hide an out-of-plane reflection error",async()=>{
  const {reflectionGeometry}=await modulePromise;
  // Both legs make 45 degrees to x-normal, but their tangential components
  // are perpendicular. A mirror reflection must reverse those components.
  const result=inspect(reflectionGeometry,fixture([1,0,1],[1,1,0],[0,0,0]));
  close(result.incidence,45);close(result.reflection,45);close(result.mismatch,60);
  close(result.incident[0],0);close(Math.hypot(...result.incident),Math.SQRT1_2);
  close(Math.hypot(...result.outgoing),1);
  // Also retain an actual 180-degree mismatch, rather than folding it acute.
  const opposite=inspect(reflectionGeometry,fixture([1,0,0],[-1,0,0],[0,0,0]));
  close(opposite.mismatch,180);
});

test("missing, blocked, non-finite and degenerate paths have no invented geometry",async()=>{
  const {reflectionGeometry}=await modulePromise;
  for(const input of [fixture([0,0,0]),fixture(undefined,[0,0,0],[0,0,0]),fixture(undefined,undefined,undefined,[[0,0,0],[0,1,0],[0,2,0]]),fixture(undefined,undefined,[NaN,0,0]),fixture(undefined,undefined,undefined,[[0,0,0],[0,1,0],[0,0,Infinity]])]){
    if(input.snapshot.receiver.every(v=>v===0))input.signal.reflection.point=[0,0,0];
    assert.equal(inspect(reflectionGeometry,input),null);
  }
  assert.equal(reflectionGeometry(null,null),null);
  const input=fixture();input.signal.status="BLOCKED";assert.equal(inspect(reflectionGeometry,input),null);
  input.signal.status="LOS";input.signal.reflection=null;assert.equal(inspect(reflectionGeometry,input),null);
});

test("real full-mesh reflections satisfy the law without modifying any original trace result",async()=>{
  const {reflectionGeometry}=await modulePromise;
  const [{CitySimulation},{decodeMesh}]=await Promise.all([import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
  const folder=path.join(__dirname,"../../docs/assets/plateau"),scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz")))),bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
  const simulation=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
  for(const settings of [{epoch:8000},{epoch:8000,antenna:20},{epoch:3890,mask:12}]){
    const snapshot=simulation.compute(settings),before=structuredClone(snapshot),signal=snapshot.satellites.find(s=>s.id===(settings.antenna?"G10":"G12")),result=reflectionGeometry(snapshot,signal);
    // Independent cosine-law and plane-normal oracle from raw source vectors.
    const [a,b,c]=signal.reflection.vertices,u=b.map((v,i)=>v-a[i]),v=c.map((value,i)=>value-a[i]);
    const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
    const angleToNormal = point => {
      const d=point.map((value,i)=>value-signal.reflection.point[i]);
      return Math.acos(Math.min(1,Math.abs(d.reduce((sum,value,i)=>sum+value*n[i],0))/(Math.hypot(...d)*Math.hypot(...n))))*180/Math.PI;
    };
    close(result.incidence,angleToNormal(signal.source));close(result.reflection,angleToNormal(snapshot.receiver));
    assert.ok(result.mismatch<1e-8);close(result.sourceLeg+result.receiverLeg,signal.reflection.distance,1e-7);
    assert.deepEqual(snapshot,before);assert.deepEqual(simulation.compute(settings),before);
    if(!settings.antenna&&settings.epoch===8000){
      close(result.receiverLeg,117.55809428100213);close(result.incidence,71.75842661718968);
      close(signal.reflection.extra,23.037548400461674,1e-6);assert.deepEqual(snapshot.counts,{LOS:4,NLOS:1,BLOCKED:3});
    }
  }
});
