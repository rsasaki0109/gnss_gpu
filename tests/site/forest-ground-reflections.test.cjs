const test=require("node:test"),assert=require("node:assert/strict");
const modulePromise=import("../../docs/assets/forest/ground-reflections.mjs");
const modelPromise=import("../../docs/assets/forest/model.mjs");
const close=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const ground=()=>({trees:[],terrain:[ [[-50,-50,0],[50,-50,0],[50,50,0]], [[-50,-50,0],[50,50,0],[-50,50,0]] ]});
const receiver=[0,0,2],source=[10,0,10];
const tree=(trunk,crown={center:[40,40,20],radii:[1,1,1]},density=1)=>({id:"T",trunk,branches:[],crown,density});
const farWood={a:[40,40,0],b:[40,40,2],radius:.2};

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
