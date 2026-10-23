const test=require("node:test"),assert=require("node:assert/strict");
const model=import("../../docs/assets/forest/model.mjs");
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const crown=(x,density=1)=>({id:`T${x}`,density,crown:{center:[x,0,0],radii:[2,2,2]},trunk:{a:[50,50,0],b:[50,50,2],radius:.2},branches:[]});

test("seed reproduces all tree geometry and terrain; denser scenes preserve their prefix",async()=>{
  const {generateForest}=await model,a=generateForest(),b=generateForest(),dense=generateForest({count:96});
  assert.deepEqual(a,b);assert.deepEqual(dense.trees.slice(0,64),a.trees);assert.notDeepEqual(generateForest({seed:1}).trees,a.trees);
  assert.equal(a.terrain.length,512);assert.ok(a.trees.every(t=>t.branches.length===5));
  for(const t of a.trees){close(t.crown.center[2]+t.crown.radii[2]-t.trunk.a[2],t.height);assert.ok(t.crown.center[2]-t.crown.radii[2]-t.trunk.a[2]>=4-1e-12);}
  for(const options of [{seed:-1},{seed:1.5},{seed:Infinity},{seed:2**32},{count:0},{count:129},{count:2.5}])assert.throws(()=>generateForest(options),RangeError);
});
test("route is 1.6 m above the same piecewise planar terrain and outside stems",async()=>{
  const {forestReceiver,terrainHeight,trailX,generateForest}=await model,forest=generateForest({count:96});
  for(let seconds=0;seconds<=1200;seconds+=5){const [x,y,z]=forestReceiver(seconds);close(z-terrainHeight(x,y),1.6);close(x,trailX(y));for(const t of forest.trees)assert.ok(Math.hypot(x-t.x,y-t.y)>t.trunk.radius);}
  for(const v of [-1,1201,NaN])assert.throws(()=>forestReceiver(v),RangeError);
  assert.throws(()=>terrainHeight(81,0),RangeError);
});
test("ellipsoid ray intervals preserve analytic chord lengths, clipping and inside starts",async()=>{
  const {ellipsoidInterval}=await model,shape={center:[0,0,0],radii:[2,3,4]};
  assert.deepEqual(ellipsoidInterval([-10,0,0],[1,0,0],shape),[8,12]);
  assert.deepEqual(ellipsoidInterval([0,0,0],[0,0,1],shape),[0,4]);
  assert.deepEqual(ellipsoidInterval([-10,0,0],[1,0,0],shape,9),[8,9]);
  assert.equal(ellipsoidInterval([-10,3,0],[1,0,0],shape),null);
  assert.equal(ellipsoidInterval([10,0,0],[1,0,0],shape),null);
  assert.throws(()=>ellipsoidInterval([0,0,0],[2,0,0],shape),RangeError);
  assert.throws(()=>ellipsoidInterval([NaN,0,0],[1,0,0],shape),RangeError);
  assert.throws(()=>ellipsoidInterval([0,0,0],[1,0,0],{center:[0,0,0],radii:[2,0,4]}),RangeError);
});
test("finite cylinders include side hits and caps, exclude hits beyond their endpoints",async()=>{
  const {cylinderInterval}=await model,cylinder={a:[0,0,0],b:[0,0,2],radius:1};
  assert.deepEqual(cylinderInterval([-3,0,1],[1,0,0],cylinder),[2,4]);
  assert.deepEqual(cylinderInterval([0,0,-3],[0,0,1],cylinder),[3,5]);
  assert.deepEqual(cylinderInterval([0,0,1],[0,0,1],cylinder),[0,1]);
  assert.equal(cylinderInterval([-3,0,3],[1,0,0],cylinder),null);
  assert.equal(cylinderInterval([2,0,-3],[0,0,1],cylinder),null);
  const rotate=p=>[p[2],p[0],p[1]];
  assert.deepEqual(cylinderInterval(rotate([-3,0,1]),rotate([1,0,0]),{a:rotate(cylinder.a),b:rotate(cylinder.b),radius:1}),[2,4]);
  assert.throws(()=>cylinderInterval([0,0,0],[1,0,0],{a:[0,0,0],b:[0,0,0],radius:1}),RangeError);
});
test("overlapping crown union length differs from additive optical depth; zero foliage is not a wood model",async()=>{
  const {traceForest}=await model,forest={terrain:[],trees:[crown(0),crown(2,2)]},o=[-10,0,0],d=[1,0,0];
  const result=traceForest(forest,o,d);close(result.canopyPath,6);close(result.opticalPath,12);close(result.canopyLoss,2.4);close(result.directRelativeDb,-2.4);
  assert.equal(result.classification,"CANOPY");
  const bare=traceForest(forest,o,d,{foliage:0});close(bare.canopyPath,6);close(bare.canopyLoss,0);assert.equal(bare.classification,"CLEAR");
  const clear=traceForest(forest,o,d,{specificLoss:0});close(clear.canopyLoss,0);close(clear.canopyPath,6);
  const clipped=traceForest(forest,o,d,{maximum:11});close(clipped.canopyPath,3);close(clipped.opticalPath,5);
});
test("wood or terrain intersection leaves direct power unknown and does not invent a reflected path",async()=>{
  const {traceForest}=await model,t=crown(0);t.trunk={a:[0,0,-2],b:[0,0,2],radius:.5};
  const wood=traceForest({terrain:[],trees:[t]},[-10,0,0],[1,0,0],{foliage:0});assert.equal(wood.classification,"WOOD");assert.equal(wood.directRelativeDb,null);assert.equal(wood.wood[0].part,"trunk");
  const ground=traceForest({trees:[],terrain:[[[-10,-10,0],[10,-10,0],[0,10,0]]]},[0,0,2],[0,0,-1]);close(ground.ground,2);assert.equal(ground.classification,"TERRAIN");assert.equal(ground.directRelativeDb,null);
  assert.throws(()=>traceForest({trees:[],terrain:[]},[0,0,0],[0,0,0]),RangeError);
  assert.throws(()=>traceForest({trees:[],terrain:[]},[0,0,0],[1,0,0],{specificLoss:-1}),RangeError);
});
test("rendering tessellation never modifies binary64 physical shapes or their path integrals",async()=>{
  const {generateForest,forestMesh,traceForest,forestReceiver}=await model,forest=generateForest(),saved=structuredClone(forest),receiver=forestReceiver(600),before=traceForest(forest,receiver,[.4,.3,.5]);
  const mesh=forestMesh(forest);assert.ok(mesh instanceof Float32Array);assert.ok(mesh.length>100000);assert.ok(mesh.every(Number.isFinite));assert.deepEqual(forest,saved);assert.deepEqual(traceForest(forest,receiver,[.4,.3,.5]),before);
  assert.equal(mesh.length%27,0);
  assert.ok(forestMesh(forest,{foliage:0}).length<mesh.length);assert.deepEqual(forest,saved);
});

test("generated forest optical depth agrees with independent spatial quadrature",async()=>{
  const {generateForest,forestReceiver,traceForest}=await model,forest=generateForest({count:32}),origin=forestReceiver(600),step=.02;
  for(const tree of forest.trees.slice(0,3)){
    const raw=tree.crown.center.map((v,i)=>v-origin[i]);
    const n=Math.hypot(...raw),direction=raw.map(v=>v/n),result=traceForest(forest,origin,direction,{maximum:200});
    const limit=result.ground??200;let integral=0;
    assert.ok(result.crowns.length>0,"Reference ray must traverse vegetation");
    for(let t=step/2;t<limit;t+=step){const p=origin.map((v,i)=>v+t*direction[i]);for(const tree of forest.trees){let inside=0;for(let i=0;i<3;i++)inside+=((p[i]-tree.crown.center[i])/tree.crown.radii[i])**2;if(inside<1)integral+=tree.density*step;}}
    const bound=2*step*result.crowns.reduce((sum,c)=>sum+c.density,0)+1e-10;
    assert.ok(Math.abs(result.opticalPath-integral)<=bound,`${result.opticalPath} != ${integral}`);
  }
});

test("leaf geometry stays in the unchanged density envelope with finite unit normals",async()=>{
  const {generateForest,forestMesh}=await model,forest=generateForest({count:1}),saved=structuredClone(forest),mesh=forestMesh(forest),c=forest.trees[0].crown;let leaves=0;
  assert.deepEqual(mesh,forestMesh(forest));
  for(let i=0;i<mesh.length;i+=9){
    const n=Math.hypot(mesh[i+3],mesh[i+4],mesh[i+5]);assert.ok(Math.abs(n-1)<2e-6);
    if(mesh[i+7]<1.9*mesh[i+6])continue;
    const q=[0,1,2].reduce((sum,j)=>sum+((mesh[i+j]-c.center[j])/c.radii[j])**2,0);assert.ok(q<=1.00001);leaves++;
  }
  assert.ok(leaves>1000);assert.deepEqual(forest,saved);
});

test("dense forest display remains within a bounded vertex buffer",async()=>{
  const {generateForest,forestMesh}=await model,forest=generateForest({count:128}),mesh=forestMesh(forest);
  assert.ok(mesh.byteLength<28*1024*1024);assert.equal(mesh.length%27,0);assert.ok(mesh.every(Number.isFinite));
});
