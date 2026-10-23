const test=require("node:test"),assert=require("node:assert/strict");
const model=import("../../docs/assets/forest/model.mjs");
const close=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
test("tree type and season preserve seeded placement and count prefixes",async()=>{
  const {generateForest}=await model,reference=generateForest({count:32});
  for(const stand of ["broadleaf","conifer","mixed"])for(const season of ["leaf-on","leaf-off"]){
    const f=generateForest({count:32,stand,season}),dense=generateForest({count:64,stand,season});
    assert.deepEqual(f,generateForest({count:32,stand,season}));assert.deepEqual(f.trees,dense.trees.slice(0,32));
    assert.deepEqual(f.terrain,reference.terrain);assert.equal(f.stand,stand);assert.equal(f.season,season);
    f.trees.forEach((t,i)=>{assert.deepEqual([t.x,t.y,t.height,t.trunk],[reference.trees[i].x,reference.trees[i].y,reference.trees[i].height,reference.trees[i].trunk]);assert.equal(t.species,stand==="mixed"?(i%2?"conifer":"broadleaf"):stand);});
  }
  for(const options of [{stand:"oak"},{season:"winter"},{stand:null},{season:0}])assert.throws(()=>generateForest(options),RangeError);
});
test("leaf-off removes deciduous attenuation but retains identical wood intersections",async()=>{
  const {generateForest,traceForest,ForestOcclusion}=await model;
  const summer=generateForest({count:1}),winter=generateForest({count:1,season:"leaf-off"}),t=summer.trees[0];
  summer.terrain=[];winter.terrain=[];
  assert.deepEqual(t.trunk,winter.trees[0].trunk);assert.deepEqual(t.branches,winter.trees[0].branches);
  for(const densityProfile of ["uniform","radial-height-v1"]){
    const origin=[t.x+15,t.y,t.crown.center[2]],a=traceForest(summer,origin,[-1,0,0],{densityProfile}),b=traceForest(winter,origin,[-1,0,0],{densityProfile});
    assert.ok(a.canopyLoss>0);assert.equal(b.canopyLoss,0);assert.equal(b.canopyPath,0);assert.deepEqual(b.crowns,[]);assert.deepEqual(a.wood,b.wood);assert.equal(b.directRelativeDb,null);
    const scaled=traceForest(summer,origin,[-1,0,0],{densityProfile,foliage:.5});close(scaled.canopyLoss,a.canopyLoss*.5);
  }
  const o=[t.x-2,t.y,t.trunk.a[2]+1];
  assert.equal(new ForestOcclusion(winter).blocked(o,[1,0,0],4),true);assert.equal(traceForest(winter,o,[1,0,0],{maximum:4}).classification,"WOOD");
});
test("conifer tiers are disjoint physical volumes with independently integrated density",async()=>{
  const {generateForest,traceForest,ellipsoidInterval}=await model,{crownDensity}=await import("../../docs/assets/forest/density.mjs");
  const f=generateForest({count:1,stand:"conifer"});f.terrain=[];const t=f.trees[0],o=[t.x,t.y,t.trunk.a[2]+t.height+1],d=[0,0,-1];
  assert.equal(t.canopies.length,5);assert.equal(t.branches.length,15);
  for(let i=1;i<5;i++){const a=t.canopies[i-1],b=t.canopies[i];assert.ok(b.radii[0]<a.radii[0]);close(a.center[2]+a.radii[2],b.center[2]-b.radii[2]);}
  for(const densityProfile of ["uniform","radial-height-v1"]){
    const result=traceForest(f,o,d,{densityProfile,maximum:40});let integral=0;
    for(const c of t.canopies){const span=ellipsoidInterval(o,d,c,40),mid=(span[0]+span[1])/2,half=(span[1]-span[0])/2;
      for(const q of [-1/Math.sqrt(3),1/Math.sqrt(3)])integral+=half*crownDensity(c,o.map((v,i)=>v+(mid+half*q)*d[i]),t.density,densityProfile);
    }
    close(result.opticalPath,integral);close(result.canopyPath,t.height-4);assert.equal(result.crowns.length,5);
    assert.equal(new Set(result.crowns.map(c=>c.component)).size,5);
  }
});
test("evergreen season change leaves RF and mesh identical, mixed stands shed only broadleaf",async()=>{
  const {generateForest,forestMesh,traceForest,forestReceiver}=await model;
  const a=generateForest({count:2,stand:"conifer"}),b=generateForest({count:2,stand:"conifer",season:"leaf-off"});
  assert.deepEqual(forestMesh(a),forestMesh(b));assert.deepEqual(traceForest(a,forestReceiver(600),[.2,.3,1]),traceForest(b,forestReceiver(600),[.2,.3,1]));
  const mixed=generateForest({count:8,stand:"mixed",season:"leaf-off"});
  assert.equal(mixed.trees.filter(t=>t.leafRetention===0).length,4);assert.equal(mixed.trees.filter(t=>t.leafRetention===1&&t.species==="conifer").length,4);
  assert.ok(forestMesh(generateForest({count:2,season:"leaf-off"})).length<forestMesh(generateForest({count:2})).length);
});
test("conifer leaf vertices stay inside the same RF component volumes with finite unit normals",async()=>{
  const {generateForest,forestMesh}=await model,f=generateForest({count:1,stand:"conifer"}),saved=structuredClone(f),mesh=forestMesh(f);let leaves=0;
  for(let i=0;i<mesh.length;i+=9){
    assert.ok(Math.abs(Math.hypot(mesh[i+3],mesh[i+4],mesh[i+5])-1)<2e-6);
    if(mesh[i+7]<1.9*mesh[i+6])continue;
    assert.ok(f.trees[0].canopies.some(c=>[0,1,2].reduce((s,j)=>s+((mesh[i+j]-c.center[j])/c.radii[j])**2,0)<=1.00002));leaves++;
  }
  assert.ok(leaves>500);assert.deepEqual(f,saved);
});
test("both reflected legs share leaf retention, component geometry and additive density",async()=>{
  const {GroundReflectionModel}=await import("../../docs/assets/forest/ground-reflections.mjs");
  const forest={terrain:[[[-50,-50,0],[50,-50,0],[50,50,0]],[[-50,-50,0],[50,50,0],[-50,50,0]]],trees:[{id:"T",density:1.5,leafRetention:1,species:"broadleaf",crown:{center:[40,40,20],radii:[1,1,1]},canopies:[{center:[5/6,0,1],radii:[.25,.25,.25]},{center:[35/6,0,5],radii:[.5,.5,.5]}],trunk:{a:[40,40,0],b:[40,40,2],radius:.2},branches:[]}]};
  const a=new GroundReflectionModel(forest).paths([0,0,2],[10,0,10],{densityProfile:"radial-height-v1"}).paths[0];
  forest.trees[0].leafRetention=0;
  const b=new GroundReflectionModel(forest).paths([0,0,2],[10,0,10],{densityProfile:"radial-height-v1"}).paths[0];
  assert.deepEqual(a.point,b.point);assert.equal(a.extra,b.extra);assert.equal(a.totalRange,b.totalRange);
  for(const leg of ["receiver","satellite"]){assert.ok(a.legs[leg].canopyLoss>0);assert.equal(b.legs[leg].canopyLoss,0);}
  close(b.relativePowerDb-a.relativePowerDb,a.canopyLoss);
});
test("vertical map bias shifts every conifer RF component without mutating the source",async()=>{
  const {generateForest,traceForest}=await model,{shiftedForest}=await import("../../docs/assets/forest/positioning.mjs");
  const f=generateForest({count:2,stand:"mixed",season:"leaf-off"}),saved=structuredClone(f),mapped=shiftedForest(f,1.25);
  f.trees.forEach((t,i)=>{assert.equal(mapped.trees[i].leafRetention,t.leafRetention);for(const [j,c]of (t.canopies??[]).entries())assert.deepEqual(mapped.trees[i].canopies[j].center,[c.center[0],c.center[1],c.center[2]+1.25]);});
  const t=f.trees[1],o=[t.x+10,t.y,t.canopies[2].center[2]],a=traceForest({...f,terrain:[]},o,[-1,0,0]),b=traceForest({...mapped,terrain:[]},[o[0],o[1],o[2]+1.25],[-1,0,0]);
  close(a.canopyLoss,b.canopyLoss);assert.deepEqual(f,saved);
});
test("dense tree-type meshes remain within the existing vertex-buffer limit",async()=>{
  const {generateForest,forestMesh}=await model;
  for(const stand of ["conifer","mixed"]){const mesh=forestMesh(generateForest({count:128,stand}));assert.ok(mesh.byteLength<28*1024*1024);assert.ok(mesh.every(Number.isFinite));}
});
