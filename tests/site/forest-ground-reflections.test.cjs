const test=require("node:test"),assert=require("node:assert/strict");
const modulePromise=import("../../docs/assets/forest/ground-reflections.mjs");
const modelPromise=import("../../docs/assets/forest/model.mjs");
const close=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const ground=()=>({trees:[],terrain:[ [[-50,-50,0],[50,-50,0],[50,50,0]], [[-50,-50,0],[50,50,0],[-50,50,0]] ]});
const receiver=[0,0,2],source=[10,0,10];
const tree=(trunk,crown={center:[40,40,20],radii:[1,1,1]},density=1)=>({id:"T",trunk,branches:[],crown,density});
const farWood={a:[40,40,0],b:[40,40,2],radius:.2};

const detailedRanges=(forest,model,receiver,source,traceForest)=>{
  const direct=Math.hypot(...receiver.map((v,i)=>v-source[i])),direction=source.map((v,i)=>(v-receiver[i])/direct),trace=traceForest(forest,receiver,direction,{maximum:direct});
  return [...(trace.ground===null&&!trace.wood.length?[direct]:[]),...model.paths(receiver,source).paths.map(p=>p.totalRange)];
};
test("prepared occlusion ranges match detailed rays for wood, terrain and finite-aperture cases",async()=>{
  const {GroundReflectionModel}=await modulePromise,{traceForest}=await modelPromise;
  const cases=[ground(),{trees:[],terrain:[]},...['wood','terrain'].flatMap(kind=>[.8,2.5].map(x=>{
    const forest=ground();if(kind==='wood')forest.trees=[tree({a:[x,0,0],b:[x,0,2],radius:.2})];else forest.terrain.push([[x,-2,.2],[x,2,.2],[x,0,3]]);return forest;
  }))];
  const distant=ground();distant.trees=[tree({a:[1000,0,990],b:[1000,0,1010],radius:1})];cases.push(distant);
  for(const forest of cases)for(const source of [[10,0,10],[0,0,10],[2000,0,2000],[10,0,-10],[20e6,4e6,15e6]]){
    const model=new GroundReflectionModel(forest);assert.deepEqual(model.ranges(receiver,source),detailedRanges(forest,model,receiver,source,traceForest));
  }
});
test("prepared visibility matches detailed binary64 intersections on deterministic generated rays",async()=>{
  const {GroundReflectionModel}=await modulePromise,{generateForest,terrainHeight,randomSequence,traceForest,ForestOcclusion}=await modelPromise;
  const forest=generateForest(),model=new GroundReflectionModel(forest),index=new ForestOcclusion(forest),random=randomSequence(42);let blocked=0,reflected=0;
  for(let i=0;i<120;i++){
    const x=(random()-.5)*140,y=(random()-.5)*140,receiver=[x,y,terrainHeight(x,y)+.05+random()*10];
    const source=[(random()-.5)*4e7,(random()-.5)*4e7,(random()-.15)*3e7],delta=source.map((v,j)=>v-receiver[j]),range=Math.hypot(...delta),direction=delta.map(v=>v/range);
    const trace=traceForest(forest,receiver,direction,{maximum:range}),expected=trace.ground!==null||trace.wood.length>0;
    assert.equal(index.blocked(receiver,direction,range),expected);blocked+=expected;
    const detailed=detailedRanges(forest,model,receiver,source,traceForest);assert.deepEqual(model.ranges(receiver,source),detailed);reflected+=model.paths(receiver,source).paths.length;
  }
  assert.ok(blocked>0);assert.ok(reflected>0);
});
test("prepared rays preserve tangent, inside-cylinder and reflecting-endpoint decisions",async()=>{
  const {GroundReflectionModel}=await modulePromise,{traceForest,ForestOcclusion}=await modelPromise;
  const forest=ground();forest.trees=[tree({a:[0,0,0],b:[0,0,4],radius:.5})];const model=new GroundReflectionModel(forest),index=new ForestOcclusion(forest);
  for(const receiver of [[0,0,2],[.5,0,2],[.5-1e-12,0,2],[.5+1e-12,0,2]])for(const source of [[.5,10,2],[0,0,10],[10,0,10]])assert.deepEqual(model.ranges(receiver,source),detailedRanges(forest,model,receiver,source,traceForest));
  assert.throws(()=>index.blocked([0,0,1],[0,0,0],10),RangeError);assert.throws(()=>index.blocked([0,0,1],[0,0,1],10,{groundMaximum:11}),RangeError);
  assert.throws(()=>model.ranges(receiver,receiver),RangeError);assert.throws(()=>model.ranges([NaN,0,1],source),RangeError);
});

test("finite ground path satisfies image geometry, reflection law and exact path length",async()=>{
  const {GroundReflectionModel}=await modulePromise,result=new GroundReflectionModel(ground()).paths(receiver,source);
  assert.equal(result.paths.length,1);assert.equal(result.blocked.length,0);const p=result.paths[0];
  close(p.point[0],5/3);close(p.point[1],0);close(p.point[2],0);
  close(p.totalRange,Math.hypot(10,12));close(p.extra,Math.hypot(10,12)-Math.hypot(10,8));close(p.delayNs,p.extra/299792458*1e9);
  for(let i=0;i<3;i++){const incoming=(p.point[i]-source[i])/p.sourceRange,normalDot=-source[2]/p.sourceRange;close(incoming-2*normalDot*p.normal[i],(receiver[i]-p.point[i])/p.receiverRange);}
  assert.equal(result.options.completeSegments,true);assert.ok(p.relativePowerDb< -6);
});
test("coplanar shared edge is one path; winding, finite aperture and reflecting side are respected",async()=>{
  const {GroundReflectionModel}=await modulePromise;
  const result=new GroundReflectionModel(ground()).paths(receiver,[0,0,10]);assert.equal(result.paths.length,1);close(result.paths[0].extra,4);
  const reversed=ground();reversed.terrain=reversed.terrain.map(t=>t.toReversed());
  const reverseResult=new GroundReflectionModel(reversed).paths(receiver,source);
  assert.deepEqual(reverseResult.paths[0].point,new GroundReflectionModel(ground()).paths(receiver,source).paths[0].point);
  const outside={trees:[],terrain:[[[30,30,0],[40,30,0],[30,40,0]]]};assert.equal(new GroundReflectionModel(outside).paths(receiver,source).paths.length,0);
  assert.equal(new GroundReflectionModel(ground()).paths(receiver,[10,0,-10]).paths.length,0);
  assert.equal(new GroundReflectionModel(ground()).paths([0,0,-2],[10,0,-10]).paths.length,0);
});
test("wood in either segment rejects arrival but retains blocked candidate diagnostics",async()=>{
  const {GroundReflectionModel}=await modulePromise;
  for(const [x,leg]of [[.8,"receiver"],[2.5,"satellite"]]){
    const forest=ground();forest.trees=[tree({a:[x,0,0],b:[x,0,2],radius:.2})];const result=new GroundReflectionModel(forest).paths(receiver,source);
    assert.equal(result.paths.length,0);assert.equal(result.blocked.length,1);assert.equal(result.blocked[0].reason,"wood");assert.equal(result.blocked[0].relativePowerDb,null);assert.ok(result.blocked[0].legs[leg].wood.length>0);
  }
});
test("terrain in either segment rejects arrival without rejecting the reflecting endpoint itself",async()=>{
  const {GroundReflectionModel}=await modulePromise;
  for(const [x,leg]of [[.8,"receiverGround"],[2.5,"sourceGround"]]){
    const forest=ground();forest.terrain.push([[x,-2,.2],[x,2,.2],[x,0,3]]);const result=new GroundReflectionModel(forest).paths(receiver,source);
    assert.equal(result.paths.length,0);assert.equal(result.blocked.length,1);assert.equal(result.blocked[0].reason,"terrain");assert.notEqual(result.blocked[0].occlusion[leg],null);
  }
  assert.equal(new GroundReflectionModel(ground()).paths(receiver,source).paths.length,1);
});
test("complete source segment catches obstructions beyond the old 400 m direct-ray bound",async()=>{
  const {GroundReflectionModel}=await modulePromise,forest=ground();forest.trees=[tree({a:[1000,0,990],b:[1000,0,1010],radius:1})];
  const result=new GroundReflectionModel(forest).paths(receiver,[2000,0,2000]);assert.equal(result.paths.length,0);assert.equal(result.blocked.length,1);assert.ok(result.blocked[0].legs.satellite.wood[0].enter>400);
});
test("both segment canopy integrals and bounce loss affect power without altering geometry",async()=>{
  const {GroundReflectionModel}=await modulePromise,forest=ground();forest.trees=[tree(farWood,{center:[5/6,0,1],radii:[.25,.25,.25]},1.5),tree(farWood,{center:[35/6,0,5],radii:[.5,.5,.5]},2)];
  const model=new GroundReflectionModel(forest),a=model.paths(receiver,source),b=model.paths(receiver,source,{bounceLoss:12}),bare=model.paths(receiver,source,{foliage:0});
  close(a.paths[0].canopyLoss,.55);close(a.paths[0].relativePowerDb-b.paths[0].relativePowerDb,6);close(bare.paths[0].canopyLoss,0);assert.deepEqual(a.paths[0].point,b.paths[0].point);close(a.paths[0].extra,bare.paths[0].extra);
});
test("invalid endpoints and unbounded assumptions fail visibly",async()=>{
  const {GroundReflectionModel}=await modulePromise,model=new GroundReflectionModel(ground());
  for(const [r,s]of [[receiver,receiver],[[NaN,0,2],source],[[1e308,0,2],[-1e308,0,10]]])assert.throws(()=>model.paths(r,s),RangeError);
  for(const options of [{bounceLoss:-1},{bounceLoss:Infinity},{specificLoss:-1},{foliage:NaN}])assert.throws(()=>model.paths(receiver,source,options),RangeError);
});
test("far-field excess agrees with an 80-digit Decimal mirror-distance reference",async()=>{
  const {GroundReflectionModel}=await modulePromise,p=new GroundReflectionModel(ground()).paths([1,-2,2],[20e6,4e6,15e6]).paths[0];
  // sqrt(dx²+dy²+(sz+rz)²) - sqrt(dx²+dy²+(sz-rz)²), Decimal precision 80.
  close(p.extra,2.369857560965587402237387569767744030063434326586189868231507293265607728,1e-12);
});
test("real-scale generated terrain paths obey specular geometry and remain deterministic",async()=>{
  const {GroundReflectionModel}=await modulePromise,{generateForest,forestReceiver,terrainHeight}=await modelPromise,forest=generateForest(),saved=structuredClone(forest),model=new GroundReflectionModel(forest);let count=0;
  for(const seconds of [0,200,400,600,800,1000,1200]){
    const receiver=forestReceiver(seconds),source=[20e6,5e6,15e6],a=model.paths(receiver,source);assert.deepEqual(a,model.paths(receiver,source));
    for(const p of a.paths){count++;close(p.point[2],terrainHeight(p.point[0],p.point[1]),1e-8);assert.ok(p.extra>=0);close(p.totalRange-p.sourceRange,p.receiverRange,1e-7);
      const incoming=p.point.map((v,i)=>(v-source[i])/p.sourceRange),dot=incoming.reduce((sum,v,i)=>sum+v*p.normal[i],0);
      incoming.forEach((v,i)=>close(v-2*dot*p.normal[i],(receiver[i]-p.point[i])/p.receiverRange,1e-8));
    }
  }
  assert.ok(count>0);assert.deepEqual(forest,saved);
});

// Independent vector form of the original binary64 cylinder kernel. Keep it
// allocation-heavy here to catch changes in scalar accumulation/rounding order.
function vectorCylinder(origin,direction,cylinder,maximum){
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),sub=(a,b)=>a.map((v,i)=>v-b[i]);
  const axis=sub(cylinder.b,cylinder.a),length=Math.hypot(...axis),n=axis.map(v=>v/length);
  const o=sub(origin,cylinder.a),z=dot(o,n),dz=dot(direction,n);
  const radialO=o.map((v,i)=>v-z*n[i]),radialD=direction.map((v,i)=>v-dz*n[i]);
  const a=dot(radialD,radialD),b=2*dot(radialO,radialD),c=dot(radialO,radialO)-cylinder.radius**2;
  let span;
  if(a<1e-15)span=c<=0?[-Infinity,Infinity]:null;
  else {const disc=b*b-4*a*c;if(disc<=0)return null;const q=-.5*(b+Math.sign(b||1)*Math.sqrt(disc)),t0=q/a,t1=c/q;span=[Math.min(t0,t1),Math.max(t0,t1)];}
  if(!span)return null;
  let lo=Math.max(0,span[0]),hi=Math.min(maximum,span[1]);
  if(Math.abs(dz)<1e-14){if(z<0||z>length)return null;}
  else {const a=-z/dz,b=(length-z)/dz;lo=Math.max(lo,Math.min(a,b));hi=Math.min(hi,Math.max(a,b));}
  return hi>lo?[lo,hi]:null;
}

test("scalar cylinder intervals exactly match vector oracle over near-axis, tangent and far-field rays",async()=>{
  const {cylinderInterval,randomSequence}=await modelPromise,random=randomSequence(9173);
  let hits=0,misses=0;
  const check=(o,raw,c,maximum)=>{const size=Math.hypot(...raw),d=raw.map(v=>v/size),expected=vectorCylinder(o,d,c,maximum);assert.deepEqual(cylinderInterval(o,d,c,maximum),expected);expected?hits++:misses++;};
  const upright={a:[0,0,0],b:[0,0,4],radius:.5};
  for(const x of [0,.5-1e-12,.5,.5+1e-12])for(const dz of [0,1e-16,1e-14,1e-8,1])for(const maximum of [1,400,3e7])check([x,-1,2],[0,1,dz],upright,maximum);
  for(const dx of [0,1e-16,1e-14,1e-8])check([.1,0,2],[dx,0,1],upright,400);
  for(let i=0;i<4000;i++){
    const a=Array.from({length:3},()=>100*(random()-.5)),b=a.map(v=>v+10*(random()-.5)),radius=.05+random()*2,c={a,b,radius};
    const origin=i%3===0?a.map((v,j)=>(v+b[j])/2):a.map(v=>v+10*(random()-.5));
    const raw=i%4===0?b.map((v,j)=>v-a[j]+1e-13*(random()-.5)):Array.from({length:3},()=>random()-.5);
    check(origin,raw,c,i%2?400:3e7);
  }
  assert.ok(hits>500);assert.ok(misses>500);
});

test("Boolean terrain queries retain strict endpoint and near-parallel decisions",async()=>{
  const {ForestOcclusion,terrainDistance}=await modelPromise;
  const forest={trees:[],terrain:[[[0,-2,-2],[0,2,-2],[0,0,2]],...ground().terrain]},index=new ForestOcclusion(forest);
  for(const origin of [[-1,0,0],[0,0,1],[0,0,-1],[0,0,0]])for(const raw of [[1,0,0],[0,0,-1],[1,0,1e-14],[1e-14,0,-1]])for(const maximum of [1e-6,1-1e-12,1,1+1e-12,400,3e7]){
    const size=Math.hypot(...raw),d=raw.map(v=>v/size);
    assert.equal(index.blocked(origin,raw,maximum),terrainDistance(origin,d,forest.terrain,maximum)!==null);
  }
});
