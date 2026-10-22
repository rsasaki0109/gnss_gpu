const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");
const modules = Promise.all([import("../../docs/assets/simulation/mesh.mjs"),import("../../docs/assets/simulation/city.mjs")]);

// Frozen pre-optimization predicate from 58e3628. It deliberately performs both
// divisions and has no broad rejection, so differential checks use the old rule.
function originalContains(t,px,py,pz) {
  const axis=this.projection[t],i=(axis+1)%3,j=(axis+2)%3,o=t*9,e=this.edges;
  const x=(i===0?px:i===1?py:pz)-e[o+i],y=(j===0?px:j===1?py:pz)-e[o+j];
  const determinant=e[o+3+i]*e[o+6+j]-e[o+3+j]*e[o+6+i];
  const u=(x*e[o+6+j]-y*e[o+6+i])/determinant;
  const v=(e[o+3+i]*y-e[o+3+j]*x)/determinant;
  return u>=-1e-10 && v>=-1e-10 && u+v<=1+1e-10;
}
function meshOf(triangles) {
  return {vertices:Float64Array.from(triangles.flat(2)),indices:Uint32Array.from({length:triangles.length*3},(_,i)=>i),owners:Uint32Array.from(triangles,(_,i)=>i),types:new Uint8Array(triangles.length),triangleCount:triangles.length,vertexCount:triangles.length*3};
}
const rotate=(p,axis)=>axis===0?[p[2],p[0],p[1]]:axis===1?[p[1],p[2],p[0]]:p.slice();
const bits=new DataView(new ArrayBuffer(8));
function next(value,up) {
  if(Number.isNaN(value)||value===(up?Infinity:-Infinity))return value;
  if(value===0)return up?Number.MIN_VALUE:-Number.MIN_VALUE;
  bits.setFloat64(0,value);let n=bits.getBigUint64(0);
  n+=(value>0)===up?1n:-1n;bits.setBigUint64(0,n);return bits.getFloat64(0);
}

test("division rejection preserves the original tolerance boundary for every projection and winding",async()=>{
  const [{MeshBVH}]=await modules,epsilon=1e-10;
  const values=[-epsilon,0,1,1+epsilon,1+2*epsilon,-2,2].flatMap(value=>[next(value,false),value,next(value,true)]);
  for(const axis of [0,1,2])for(const reversed of [false,true]){
    const triangle=[[0,0,0],[1,0,0],[0,1,0]].map(p=>rotate(p,axis));if(reversed)[triangle[1],triangle[2]]=[triangle[2],triangle[1]];
    const bvh=new MeshBVH(meshOf([triangle]));
    for(const u of values)for(const v of values){
      const point=rotate([u,v,0],axis),expected=u>=-epsilon&&v>=-epsilon&&u+v<=1+epsilon;
      assert.equal(bvh.containsPoint(0,point),expected,"axis "+axis+", u "+u+", v "+v);
    }
    // A small negative v permits u to exceed 1+epsilon under the old predicate.
    assert.equal(bvh.containsPoint(0,rotate([1+1.5*epsilon,-.75*epsilon,0],axis)),true);
  }
});

test("subnormal, degenerate and huge triangles retain the old Float64 predicate",async()=>{
  const [{MeshBVH}]=await modules;let seed=481516;
  const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
  for(const scale of [0,Number.MIN_VALUE,1e-170,1e-160,1e-140,1e-9,1,1e30,1e100,1e150,1e160])for(const axis of [0,1,2])for(const reversed of [false,true]){
    const triangle=[[0,0,0],[scale,0,0],[0,scale,0]].map(p=>rotate(p,axis));if(reversed)[triangle[1],triangle[2]]=[triangle[2],triangle[1]];
    const bvh=new MeshBVH(meshOf([triangle]));
    const points=[[0,0,0],[-0,0,-0],[NaN,0,0],[0,Infinity,0],[0,0,-Infinity],[Number.MAX_VALUE,0,0]];
    for(let i=0;i<128;i++)points.push(rotate([(random()*8-4)*scale,(random()*8-4)*scale,0],axis));
    for(const point of points)assert.equal(bvh.containsPoint(0,point),originalContains.call(bvh,0,...point));
  }
});

test("reflection points, shared-edge deduplication and contact limits match the original predicate exactly",async()=>{
  const [{MeshBVH}]=await modules;
  const wall=[[[0,-10,0],[0,10,0],[0,10,10]],[[0,-10,0],[0,10,10],[0,-10,10]]];
  const obstacle=[[[2,-3,0],[2,2,0],[2,2,5]],[[2,-3,0],[2,2,5],[2,-3,5]]];
  const receivers=[[4,-2,1.5],[4,0,5],[1e-5,0,1.5],[next(1e-5,false),0,1.5],[next(1e-5,true),0,1.5],[-1e-5,0,1.5]];
  const sources=[[6,4,6],[20200000,4,6],[-6,4,6]];
  for(const axis of [0,1,2])for(const triangles of [wall,[...wall,...wall],[...wall,...obstacle]]){
    const geometry=meshOf(triangles.map(triangle=>triangle.map(p=>rotate(p,axis))));
    const current=new MeshBVH(geometry),reference=new MeshBVH(geometry);reference.containsXYZ=originalContains;
    for(const rx of receivers)for(const source of sources){
      const a=rotate(rx,axis),b=rotate(source,axis);
      assert.deepEqual(current.reflections(a,b),reference.reflections(a,b));
    }
  }
});

test("all full-city snapshot fields and reflection paths match the pre-optimization predicate",async()=>{
  const [{decodeMesh},{CitySimulation}]=await modules,folder=path.join(__dirname,"../../docs/assets/plateau");
  const scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz")))),bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
  const geometry=decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  const current=new CitySimulation(scene,geometry),reference=new CitySimulation(scene,geometry);reference.bvh.containsXYZ=originalContains;
  reference.bvh.reflectionsBatch=function(receiver,sources){return sources.map(source=>this.reflections(receiver,source));};
  const settings=[0,2000,4000,6000,8000,12409].flatMap(epoch=>[0,7.5,20].flatMap(antenna=>[5,10,30].map(mask=>({epoch,antenna,mask}))));
  settings.push({epoch:3890,antenna:0,mask:12},{epoch:3890,antenna:20,mask:12});
  assert.equal(settings.length,56);
  for(const value of settings){
    const expected=reference.compute(value),actual=current.compute(value);
    // No rounding or numerical tolerance: includes position/clock solutions,
    // direct-hit IDs/points, reflection counts, shortest paths and all satellites.
    assert.deepEqual(actual,expected,JSON.stringify(value));
    if(value.antenna===0&&value.mask===10)for(const signal of actual.satellites){
      assert.deepEqual(current.bvh.reflections(actual.receiver,signal.source),reference.bvh.reflections(expected.receiver,signal.source));
    }
  }
});
