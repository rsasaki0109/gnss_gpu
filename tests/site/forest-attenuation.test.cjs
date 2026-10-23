const test=require("node:test"),assert=require("node:assert/strict");
const model=import("../../docs/assets/forest/model.mjs"),attenuation=import("../../docs/assets/forest/attenuation.mjs");
const close=(a,b,tol=1e-11)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const tree=(species="broadleaf",leafRetention=1)=>({id:"T",species,leafRetention,density:1,crown:{center:[0,0,0],radii:[2,2,2]},trunk:{a:[40,40,0],b:[40,40,2],radius:.2},branches:[]});
const reference={attenuationModel:"itu-extinction-v1",materialPreset:"maple-pine"};
test("published extinction constants retain their units, species, leaf state and frequency anchors",async()=>{
  const {EXTINCTION_REFERENCES:refs,EXTINCTION_SOURCE:source,extinctionMaterial}=await attenuation;
  assert.equal(refs["silver-maple"].inLeafPerM,.241);assert.equal(refs["silver-maple"].outOfLeafPerM,.139);assert.equal(refs["silver-maple"].anchorGHz,1.3);
  assert.equal(refs["korean-pine"].inLeafPerM,.215);assert.equal(refs["himalayan-cedar"].inLeafPerM,.271);assert.equal(refs["korean-pine"].anchorGHz,1.5);
  assert.equal(refs["korean-pine"].outOfLeafPerM,null);assert.equal(source.calibratedToScene,false);assert.ok(source.title.includes("Table 8"));assert.ok(source.frequencyPolicy.includes("No interpolation"));
  assert.equal(extinctionMaterial(tree("conifer"),"maple-cedar").reference,"himalayan-cedar");
  for(const t of [tree("conifer",0),tree("unknown"),tree("broadleaf",.5)])assert.throws(()=>extinctionMaterial(t),RangeError);
});
test("power extinction uses 10 log conversion and agrees with independent chord quadrature",async()=>{
  const {traceForest}=await model,{crownDensity}=await import("../../docs/assets/forest/density.mjs"),t=tree(),f={terrain:[],trees:[t]},o=[-5,0,.3],d=[1,0,0];
  for(const densityProfile of ["uniform","radial-height-v1"]){
    const tr=traceForest(f,o,d,{...reference,densityProfile}),span=tr.crowns[0],mid=(span.enter+span.exit)/2,half=(span.exit-span.enter)/2;let depth=0;
    for(const q of [-1/Math.sqrt(3),1/Math.sqrt(3)])depth+=half*crownDensity(t.crown,o.map((v,i)=>v+(mid+half*q)*d[i]),1,densityProfile);
    const power=Math.exp(-.241*depth);close(tr.canopyLoss,-10*Math.log10(power));close(tr.directRelativeDb,10*Math.log10(power));
    close(tr.attenuation.uniformLossDb,-10*Math.log10(Math.exp(-.241*(span.exit-span.enter))));
  }
});
test("reference leaf-off retains fine-branch extinction and identical wood occlusion",async()=>{
  const {traceForest,generateForest,forestMesh}=await model,t=tree(),f={terrain:[],trees:[t]},o=[-5,0,0],summer=traceForest(f,o,[1,0,0],reference);
  t.leafRetention=0;const winter=traceForest(f,o,[1,0,0],reference),linear=traceForest(f,o,[1,0,0]);
  assert.ok(winter.canopyLoss>0&&winter.canopyLoss<summer.canopyLoss);close(winter.canopyLoss/summer.canopyLoss,.139/.241);assert.equal(winter.canopyPath,summer.canopyPath);assert.equal(linear.canopyLoss,0);
  t.trunk={a:[0,0,-2],b:[0,0,2],radius:.2};const blocked=traceForest(f,o,[1,0,0],reference);assert.equal(blocked.directRelativeDb,null);assert.equal(blocked.classification,"WOOD");assert.ok(blocked.canopyLoss>0);
  const leafOn=generateForest({count:1}),leafOff=generateForest({count:1,season:"leaf-off"});assert.ok(forestMesh(leafOff).length<forestMesh(leafOn).length);
});
test("conifer reference analogues change extinction without changing geometry or winter retention",async()=>{
  const {generateForest,traceForest}=await model,f=generateForest({count:1,stand:"conifer"}),winter=generateForest({count:1,stand:"conifer",season:"leaf-off"}),t=f.trees[0],o=[t.x,t.y,t.trunk.a[2]+t.height+1];
  const a=traceForest(f,o,[0,0,-1],reference),b=traceForest(f,o,[0,0,-1],{...reference,materialPreset:"maple-cedar"}),c=traceForest(winter,o,[0,0,-1],reference);
  close(b.canopyLoss/a.canopyLoss,.271/.215);assert.deepEqual(a.wood,b.wood);assert.equal(a.ground,b.ground);assert.equal(a.canopyPath,b.canopyPath);assert.deepEqual(a,c);
});
test("overlapping mixed-material extinction is weighted inside the exact integral",async()=>{
  const {traceForest}=await model,a=tree(),b={...tree("conifer"),id:"C",density:2,crown:{center:[1,0,0],radii:[2,2,2]}},f={terrain:[],trees:[a,b]},tr=traceForest(f,[-5,0,0],[1,0,0],reference);
  close(tr.canopyPath,5);close(tr.opticalPath,12);close(tr.canopyLoss,10/Math.LN10*(.241*4+.215*8));
  const structured=traceForest(f,[-5,0,0],[1,0,0],{...reference,densityProfile:"radial-height-v1"});
  close(structured.canopyLoss,structured.crowns.reduce((s,c)=>s+c.densityIntegral*c.material.sigmaPerM,0)*10/Math.LN10);
  assert.equal(tr.attenuation.uniformLossDb,structured.attenuation.uniformLossDb);
});
test("both reflected legs use the reference, retain path geometry and export its settings",async()=>{
  const {GroundReflectionModel}=await import("../../docs/assets/forest/ground-reflections.mjs"),t=tree("broadleaf",0);t.canopies=[{center:[5/6,0,1],radii:[.25,.25,.25]},{center:[35/6,0,5],radii:[.5,.5,.5]}];
  const f={trees:[t],terrain:[[[-50,-50,0],[50,-50,0],[50,50,0]],[[-50,-50,0],[50,50,0],[-50,50,0]]]},m=new GroundReflectionModel(f),a=m.paths([0,0,2],[10,0,10]),b=m.paths([0,0,2],[10,0,10],{...reference,densityProfile:"radial-height-v1"});
  assert.deepEqual(a.paths[0].point,b.paths[0].point);assert.equal(a.paths[0].totalRange,b.paths[0].totalRange);assert.equal(a.paths[0].extra,b.paths[0].extra);assert.equal(b.options.attenuationModel,"itu-extinction-v1");
  for(const leg of Object.values(b.paths[0].legs)){assert.ok(leg.canopyLoss>0);assert.ok(leg.crowns.every(c=>c.material.leafState==="out-of-leaf"));}
  close(b.paths[0].canopyLoss,b.paths[0].legs.receiver.canopyLoss+b.paths[0].legs.satellite.canopyLoss);close(b.paths[0].relativePowerDb-a.paths[0].relativePowerDb,-b.paths[0].canopyLoss);
});
test("unsupported reference settings fail explicitly, including empty scenes and route runs",async()=>{
  const {traceForest,generateForest}=await model,{checkAttenuation}=await attenuation,{GroundReflectionModel}=await import("../../docs/assets/forest/ground-reflections.mjs"),{routeExperiment}=await import("../../docs/assets/forest/route.mjs");
  for(const options of [{attenuationModel:"other"},{materialPreset:"unknown"},{...reference,foliage:0},{...reference,foliage:.5}]){
    assert.throws(()=>traceForest({trees:[],terrain:[]},[0,0,0],[1,0,0],options),RangeError);
    assert.throws(()=>new GroundReflectionModel({trees:[],terrain:[]}).paths([0,0,2],[10,0,10],options),RangeError);
  }
  assert.throws(()=>checkAttenuation("itu-extinction-v1","maple-pine",2),RangeError);
  assert.throws(()=>routeExperiment({forest:generateForest({count:1}),seed:7,sigma:.3,mapBias:0,...reference,foliage:.5}),RangeError);
});
