const test = require("node:test");
const assert = require("node:assert/strict");
const model = import("../../docs/assets/simulation/model.mjs");

test("slab intersections handle parallel rays, edges, misses and nearest distance",async()=>{
  const {segmentBox}=await model;
  const box={min:[0,0,0],max:[2,3,4]};
  assert.equal(segmentBox([-1,5,2],[5,5,2],box),null);
  const hit=segmentBox([-2,1,2],[6,1,2],box);
  assert.equal(hit.enter,0.25);assert.equal(hit.distance,2);assert.deepEqual(hit.point,[0,1,2]);assert.deepEqual(hit.normal,[-1,0,0]);
  assert.ok(segmentBox([-2,0,0],[6,0,0],box));
  assert.equal(segmentBox([-2,1,2],[-1,1,2],box),null);
  assert.throws(()=>segmentBox([NaN,0,0],[1,1,1],box),TypeError);
});

test("single reflection agrees with an analytic image solution and the reflection law",async()=>{
  const {specularPath,faces,sub,mul,dot,length}=await model;
  const box={id:0,min:[-1,-10,0],max:[0,10,10]},rx=[4,-2,1.5],source=[6,4,6];
  const face=faces(box).find(f=>f.axis===0&&f.sign===1);
  const path=specularPath(rx,source,box,face,[box]);
  assert.ok(path);
  path.point.forEach((v,i)=>assert.ok(Math.abs(v-[0,0.4,3.3][i])<1e-12));
  const incoming=mul(sub(path.point,source),1/length(sub(path.point,source)));
  const outgoing=mul(sub(rx,path.point),1/length(sub(rx,path.point)));
  const reflected=sub(incoming,mul(face.normal,2*dot(incoming,face.normal)));
  assert.ok(length(sub(reflected,outgoing))<1e-12);assert.ok(path.extra>0);
  // Closed-form coordinates also hold at the notional satellite range.
  const farSource=[20_200_000,4,6];
  const farPath=specularPath(rx,farSource,box,face,[box]);
  assert.ok(farPath);
  const expectedFar=[0,...[1,2].map(i=>(rx[i]*farSource[0]+farSource[i]*rx[0])/(rx[0]+farSource[0]))];
  farPath.point.forEach((v,i)=>assert.ok(Math.abs(v-expectedFar[i])<1e-12));
  const obstacle={id:1,min:[1,-2,1],max:[3,0,3]};
  assert.equal(specularPath(rx,source,box,face,[box,obstacle]),null);
});

test("scene, sky and every route sample are deterministic and physically valid",async()=>{
  const {simulate,buildScene,receiverAt,contains,constellation,SCENES}=await model;
  assert.deepEqual(simulate(),simulate());
  assert.deepEqual(constellation(360),constellation(0));
  for(const preset of Object.keys(SCENES))for(let i=0;i<=40;i++){
    const rx=receiverAt(i/40);assert.ok(!buildScene(preset,2).some(box=>contains(rx,box,0.2)));
  }
  for(const preset of Object.keys(SCENES))for(const rotation of [0,73,180]){
    const result=simulate({preset,rotation,progress:0.43});
    assert.equal(Object.values(result.counts).reduce((a,b)=>a+b,0),10);
    for(const signal of result.satellites){
      assert.ok(Number.isFinite(signal.azimuth));
      if(signal.reflection){assert.ok(Number.isFinite(signal.reflection.extra)&&signal.reflection.extra>=0);}
      assert.equal(signal.status==="LOS",signal.directHit===null);
      if(signal.status==="NLOS")assert.ok(signal.reflection);
      if(signal.status==="BLOCKED")assert.equal(signal.reflection,null);
    }
  }
  assert.throws(()=>simulate({receiver:[0,0,NaN]}),RangeError);
  assert.throws(()=>simulate({receiver:[300,0,1.8]}),RangeError);
  assert.throws(()=>simulate({heightScale:0}),RangeError);
  assert.throws(()=>simulate({preset:"unknown"}),RangeError);
});

test("growing the same skyline cannot clear a blocked direct ray",async()=>{
  const {simulate}=await model;
  for(const progress of [0.1,0.35,0.6,0.9]){
    const low=simulate({progress,heightScale:0.5}),high=simulate({progress,heightScale:2});
    for(let i=0;i<10;i++)if(low.satellites[i].directHit)assert.ok(high.satellites[i].directHit);
  }
});

test("slab results match an independent triangulated-box oracle for 1000 segments",async()=>{
  const {segmentBox}=await model;
  const vertices=[[-2,-3,0],[2,-3,0],[2,3,0],[-2,3,0],[-2,-3,9],[2,-3,9],[2,3,9],[-2,3,9]];
  const quads=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]];
  const triangles=quads.flatMap(([a,b,c,d])=>[[a,b,c],[a,c,d]]).map(face=>face.map(i=>vertices[i]));
  const minus=(a,b)=>a.map((v,i)=>v-b[i]);
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  function triangle(from,to,[a,b,c]){
    const direction=minus(to,from),e1=minus(b,a),e2=minus(c,a),p=cross(direction,e2),det=dot(e1,p);
    if(Math.abs(det)<1e-12)return null;
    const t=minus(from,a),u=dot(t,p)/det;
    if(u<0||u>1)return null;
    const q=cross(t,e1),v=dot(direction,q)/det;
    if(v<0||u+v>1)return null;
    const distance=dot(e2,q)/det;
    return distance>=0&&distance<=1?distance:null;
  }
  let state=789;
  const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
  for(let i=0;i<1000;i++){
    const angle=random()*Math.PI*2,from=[20*Math.cos(angle),20*Math.sin(angle),random()*28-8];
    const to=[random()*20-10,random()*20-10,random()*20-5];
    const expected=triangles.map(t=>triangle(from,to,t)).filter(t=>t!==null).sort((a,b)=>a-b)[0]??null;
    const actual=segmentBox(from,to,{min:[-2,-3,0],max:[2,3,9]});
    assert.equal(actual===null,expected===null);
    if(actual)assert.ok(Math.abs(actual.enter-expected)<1e-10);
  }
});
