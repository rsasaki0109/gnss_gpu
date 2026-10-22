const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");
const modules = Promise.all([import("../../docs/assets/simulation/mesh.mjs"),import("../../docs/assets/simulation/city.mjs")]);
function meshOf(triangles) {
  return {vertices:Float64Array.from(triangles.flat(2)),indices:Uint32Array.from({length:triangles.length*3},(_,i)=>i),owners:Uint32Array.from(triangles,(_,i)=>i),types:new Uint8Array(triangles.length),triangleCount:triangles.length,vertexCount:triangles.length*3};
}
const rotate=(p,axis)=>axis===0?[p[2],p[0],p[1]]:axis===1?[p[1],p[2],p[0]]:p.slice();
const bits=new DataView(new ArrayBuffer(8));
function next(value,up) {
  if(value===0)return up?Number.MIN_VALUE:-Number.MIN_VALUE;
  bits.setFloat64(0,value);let n=bits.getBigUint64(0);
  n+=(value>0)===up?1n:-1n;bits.setBigUint64(0,n);return bits.getFloat64(0);
}
const wall=[[[0,-10,0],[0,10,0],[0,10,10]],[[0,-10,0],[0,10,10],[0,-10,10]]];
const obstacle=[[[2,-3,0],[2,2,0],[2,2,5]],[[2,-3,0],[2,2,5],[2,-3,5]]];
function compare(bvh,receiver,sources) {
  const before=structuredClone({receiver,sources});
  const expected=sources.map(source=>bvh.reflections(receiver,source));
  assert.deepEqual(bvh.reflectionsBatch(receiver,sources),expected);
  assert.deepEqual({receiver,sources},before);
  return expected;
}

test("batched reflections preserve source order, independent duplicates, winding and occlusion",async()=>{
  const [{MeshBVH}]=await modules;
  const sources=[[6,4,6],[-6,4,6],[6,4,6],[20200000,4,6],[20,0,4]];
  for(const axis of [0,1,2])for(const triangles of [wall,[...wall,...wall],[...wall,...obstacle],wall.map(t=>t.toReversed())]){
    const bvh=new MeshBVH(meshOf(triangles.map(t=>t.map(p=>rotate(p,axis)))));
    for(const receiver of [[4,-2,1.5],[4,0,5],[1,7,2]])for(const list of [[],sources.slice(0,1),sources,sources.toReversed()]){
      const actual=compare(bvh,rotate(receiver,axis),list.map(p=>rotate(p,axis)));
      if(list===sources)assert.deepEqual(actual[0],actual[2]);
    }
  }
  const bvh=new MeshBVH(meshOf([...wall,...wall]));
  const result=compare(bvh,[4,-2,1.5],[[6,4,6],[6,4,6]]);
  assert.equal(result[0].length,1);assert.equal(result[0][0].triangle,0);
  const batch=bvh.reflectionsBatch([4,-2,1.5],[[6,4,6],[6,4,6]]);
  assert.notEqual(batch[0],batch[1]);assert.notEqual(batch[0][0],batch[1][0]);
});

test("the shared projection keeps exact barycentric boundaries on all axes and both windings",async()=>{
  const [{MeshBVH}]=await modules,epsilon=1e-10;
  const values=[-epsilon,0,1,1+epsilon,1+2*epsilon,-2,2].flatMap(v=>[next(v,false),v,next(v,true)]);
  const sources=values.flatMap(u=>values.map(v=>[1,2*u,2*v]));
  for(const axis of [0,1,2])for(const reversed of [false,true]){
    const triangle=[[0,0,0],[0,1,0],[0,0,1]].map(p=>rotate(p,axis));
    if(reversed)[triangle[1],triangle[2]]=[triangle[2],triangle[1]];
    const bvh=new MeshBVH(meshOf([triangle]));
    compare(bvh,rotate([1,0,0],axis),sources.map(p=>rotate(p,axis)));
  }
});

test("shared receiver terms retain contact, non-finite and extreme-scale behavior",async()=>{
  const [{MeshBVH}]=await modules;
  const sources=[[6,4,6],[20200000,4,6],[-6,4,6],[NaN,0,0],[Infinity,0,0],[0,0,-Infinity]];
  for(const axis of [0,1,2]){
    const bvh=new MeshBVH(meshOf(wall.map(t=>t.map(p=>rotate(p,axis)))));
    for(const x of [-1e-5,next(-1e-5,false),next(-1e-5,true),0,1e-5,next(1e-5,false),next(1e-5,true)]){
      compare(bvh,rotate([x,0,1.5],axis),sources.map(p=>rotate(p,axis)));
    }
  }
  for(const scale of [0,Number.MIN_VALUE,1e-170,1e-160,1e-9,1,1e100,1e150,1e160]){
    const bvh=new MeshBVH(meshOf([[[0,0,0],[0,scale,0],[0,0,scale]]]));
    compare(bvh,[scale,0,0],[[scale,.5*scale,.5*scale],[-scale,0,0],[scale,0,0]]);
  }
});

test("full-city batches retain every reflection path across receiver and source changes",async()=>{
  const [{decodeMesh},{CitySimulation}]=await modules,folder=path.join(__dirname,"../../docs/assets/plateau");
  const scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz")))),bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
  const geometry=decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  const simulation=new CitySimulation(scene,geometry);
  for(const value of [{epoch:8000},{epoch:3890,mask:12},{epoch:8000,antenna:20},{epoch:0},{epoch:12409}]){
    const snapshot=simulation.compute(value),sources=snapshot.satellites.map(s=>s.source);
    const paths=compare(simulation.bvh,snapshot.receiver,sources);
    for(let i=0;i<paths.length;i++){
      assert.equal(paths[i].length,snapshot.satellites[i].reflectionCount);
      const first=paths[i][0];
      if(first){const {vertices,...actual}=snapshot.satellites[i].reflection;assert.deepEqual(actual,first);}
      else assert.equal(snapshot.satellites[i].reflection,null);
    }
    compare(simulation.bvh,snapshot.receiver,sources.toReversed());
  }
});
