const test=require("node:test"),assert=require("node:assert/strict");
const density=import("../../docs/assets/forest/density.mjs"),model=import("../../docs/assets/forest/model.mjs");
const close=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);
const profile="radial-height-v1",crown={center:[3,-2,6],radii:[2,3,4]};
test("structured field preserves volume mean density and has a vertical gradient",async()=>{
  const {crownDensity,DENSITY_PROFILES}=await density;
  const r=[.5-Math.sqrt(3/5)/2,.5,.5+Math.sqrt(3/5)/2],w=[5/18,4/9,5/18];let mean=0;
  for(let i=0;i<3;i++)for(const z of [-1/Math.sqrt(3),1/Math.sqrt(3)]){
    const q=[r[i]*Math.sqrt(1-z*z),0,r[i]*z],p=crown.center.map((v,j)=>v+q[j]*crown.radii[j]);
    mean+=3*w[i]*r[i]**2*crownDensity(crown,p,1,profile)/2;
  }
  close(mean,1,2e-15);
  const point=z=>[3,-2,6+4*z];assert.ok(crownDensity(crown,point(.5),1,profile)>crownDensity(crown,point(-.5),1,profile));
  assert.equal(crownDensity(crown,[30,0,0],1,profile),0);assert.equal(crownDensity(crown,point(.5),1,"uniform"),1);
  assert.equal(DENSITY_PROFILES[profile].calibrated,false);
});
test("analytic clipped chord agrees with independent two-point Gauss integration and reversal",async()=>{
  const {crownIntegral,crownDensity}=await density,{ellipsoidInterval,randomSequence}=await model,random=randomSequence(404);let hits=0;
  for(let i=0;i<300;i++){
    const origin=crown.center.map(v=>v+(random()-.5)*24),target=crown.center.map((v,j)=>v+(random()-.5)*crown.radii[j]);
    const raw=target.map((v,j)=>v-origin[j]),n=Math.hypot(...raw),d=raw.map(v=>v/n),span=ellipsoidInterval(origin,d,crown,[2,5,15,80][i%4]);if(!span)continue;
    const mid=span[0]+(span[1]-span[0])/2,h=(span[1]-span[0])/2,reference=h*[-1,1].reduce((sum,sign)=>sum+crownDensity(crown,origin.map((v,j)=>v+(mid+sign*h/Math.sqrt(3))*d[j]),1.3,profile),0);
    const actual=crownIntegral(origin,d,crown,span,1.3,profile);close(actual,reference);assert.ok(actual>=0);hits++;
    const reverse=origin.map((v,j)=>v+(span[0]+span[1])*d[j]);close(crownIntegral(reverse,d.map(v=>-v),crown,span,1.3,profile),actual);
    assert.equal(crownIntegral(origin,d,crown,span,1.3,"uniform"),(span[1]-span[0])*1.3);
  }
  assert.ok(hits>100);
});
test("profile changes optical depth, preserves overlap union and leaves wood and terrain unchanged",async()=>{
  const {traceForest}=await model;
  const tree=(x,density)=>({id:`T${x}`,density,crown:{center:[x,0,0],radii:[2,2,2]},trunk:{a:[50,50,0],b:[50,50,2],radius:.2},branches:[]}),forest={terrain:[],trees:[tree(0,1),tree(2,2)]};
  const options={specificLoss:.2,foliage:1.3},a=traceForest(forest,[-10,0,0],[1,0,0],options),b=traceForest(forest,[-10,0,0],[1,0,0],{...options,densityProfile:profile});
  assert.equal(a.canopyPath,b.canopyPath);assert.deepEqual(a.wood,b.wood);assert.equal(a.ground,b.ground);assert.notEqual(a.opticalPath,b.opticalPath);
  close(b.opticalPath,b.crowns.reduce((sum,c)=>sum+c.densityIntegral,0)*1.3);close(b.canopyLoss,b.opticalPath*.2);
  const bare=traceForest(forest,[-10,0,0],[1,0,0],{densityProfile:profile,foliage:0});assert.equal(bare.opticalPath,0);assert.equal(bare.classification,"CLEAR");
  assert.throws(()=>traceForest(forest,[-10,0,0],[1,0,0],{densityProfile:"bad"}),RangeError);
});
test("both reflected legs use the profile without changing finite reflection geometry",async()=>{
  const {GroundReflectionModel}=await import("../../docs/assets/forest/ground-reflections.mjs"),forest={terrain:[[[-50,-50,0],[50,-50,0],[50,50,0]],[[-50,-50,0],[50,50,0],[-50,50,0]]],trees:[{id:"T",density:1.5,crown:{center:[5/6,0,1],radii:[.25,.25,.25]},trunk:{a:[40,40,0],b:[40,40,2],radius:.2},branches:[]},{id:"S",density:2,crown:{center:[35/6,0,5],radii:[.5,.5,.5]},trunk:{a:[40,40,0],b:[40,40,2],radius:.2},branches:[]}]},m=new GroundReflectionModel(forest);
  const a=m.paths([0,0,2],[10,0,10]).paths[0],b=m.paths([0,0,2],[10,0,10],{densityProfile:profile}).paths[0];
  assert.deepEqual(a.point,b.point);assert.equal(a.extra,b.extra);assert.equal(a.totalRange,b.totalRange);assert.notEqual(a.canopyLoss,b.canopyLoss);
  close(b.canopyLoss,b.legs.receiver.canopyLoss+b.legs.satellite.canopyLoss);close(b.relativePowerDb-a.relativePowerDb,a.canopyLoss-b.canopyLoss);
  for(const leg of ["receiver","satellite"])assert.notEqual(a.legs[leg].canopyLoss,b.legs[leg].canopyLoss);
  assert.throws(()=>m.paths([0,0,2],[10,0,10],{densityProfile:"bad"}),RangeError);
});
test("plot gaps remain gaps while valid zero loss and zero path counts remain data",async()=>{
  const {seriesPath}=await import("../../docs/assets/forest/route-plots.mjs"),identity=x=>x;
  assert.equal(seriesPath([0,1,2,3],[1,null,2,undefined],identity,identity),"M0.00,1.00 M2.00,2.00");
  assert.equal(seriesPath([0,1],[0,0],identity,identity),"M0.00,0.00 L1.00,0.00");
});
test("route retains every empty epoch and unavailable solution without inventing zero errors",async()=>{
  const {routeExperiment,ROUTE_TIMES}=await import("../../docs/assets/forest/route.mjs"),{routeSeries}=await import("../../docs/assets/forest/route-plots.mjs"),{generateForest,forestReceiver}=await model;
  const input={forest:generateForest({count:1}),orbitScene:{week:2300,startTow:100000,origin:[6378137,0,0],rotation:[[0,1,0],[0,0,1],[1,0,0]],navigation:[]},context:{source:{file:"test",sha256:"0".repeat(64)}}};
  const result=routeExperiment(input);assert.equal(result.cases.length,5);assert.equal(result.assumptions.positioning.includes("NOT used"),true);
  for(const [i,c]of result.cases.entries()){
    assert.equal(c.seconds,ROUTE_TIMES[i]);assert.equal(c.tow,100000+c.seconds);assert.deepEqual(c.receiverENU,forestReceiver(c.seconds));assert.equal(c.positioning.status,"insufficient-observations");assert.equal(c.positioning.results.length,3);assert.ok(c.positioning.results.every(r=>r.horizontalErrorM===null));
  }
  assert.ok(routeSeries(result,"G99").losses.every(s=>s.values.every(v=>v===null)));
  for(const settings of [{seed:-1},{sigma:0},{mapBias:4},{foliage:NaN},{specificLoss:-1}])assert.throws(()=>routeExperiment({...input,...settings}),RangeError);
});

test("route resolution is explicit and bounded; fine empty runs preserve all 21 epochs",async()=>{
  const {routeTimes,routeExperiment}=await import("../../docs/assets/forest/route.mjs"),{generateForest}=await model;
  assert.equal(routeTimes(120).length,11);assert.equal(routeTimes(60).length,21);assert.equal(routeTimes(60).at(-1),1200);
  for(const step of [0,1,59,Infinity,NaN])assert.throws(()=>routeTimes(step),RangeError);
  const result=routeExperiment({forest:generateForest({count:1}),orbitScene:{week:2300,startTow:100000,origin:[6378137,0,0],rotation:[[0,1,0],[0,0,1],[1,0,0]],navigation:[]},context:{source:{file:"test",sha256:"0".repeat(64)}},stepSeconds:60});
  assert.equal(result.cases.length,21);assert.equal(result.settings.stepSeconds,60);assert.equal(result.cases.at(-1).seconds,1200);assert.ok(result.cases.every(c=>c.positioning.status==="insufficient-observations"));
  const coarse=routeExperiment({forest:result.scene,orbitScene:{week:2300,startTow:100000,origin:[6378137,0,0],rotation:[[0,1,0],[0,0,1],[1,0,0]],navigation:[]},context:{source:{file:"test",sha256:"0".repeat(64)}}});for(const c of coarse.cases)assert.equal(c.epochSeed,result.cases.find(r=>r.seconds===c.seconds).epochSeed);
});
