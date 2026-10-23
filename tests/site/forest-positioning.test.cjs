const test=require('node:test'),assert=require('node:assert/strict');
const modulePromise=import('../../docs/assets/forest/positioning.mjs');
const modelPromise=import('../../docs/assets/forest/model.mjs');
test('clock profile handles common bias, outliers and normalized latent paths',async()=>{
  const {profileClock}=await modulePromise;
  assert.equal(profileClock([[9],[10],[11]],[1,1,1],'straight').clock,10);
  const r=profileClock([[10],[10],[100]],[1,1,1],'robust');assert.ok(Math.abs(r.clock-10.75)<1e-10);
  const single=profileClock([[10],[11]],[1,1],'paths'),duplicate=profileClock([[10,10],[11,11]],[1,1],'paths');
  assert.ok(Math.abs(single.cost-duplicate.cost)<=1/32);assert.ok(Number.isFinite(profileClock([[1e6,1e6+5],[1e6,1e6-4]],[.05,.05]).cost));
  assert.throws(()=>profileClock([[]],[1]));
});
test('mixture clock grid stays within its declared bound of an independent fine scan',async()=>{
  const {profileClock}=await modulePromise,rows=[[0,3],[.3,2.7],[-.1,5]],sigmas=[.3,.5,.7];
  const score=b=>rows.reduce((s,r,i)=>s-Math.log(r.reduce((v,d)=>v+Math.exp(-.5*((d-b)/sigmas[i])**2),0)/r.length),0);
  let oracle=Infinity;for(let b=-.1;b<=5;b+=.0001)oracle=Math.min(oracle,score(b));
  const fit=profileClock(rows,sigmas);assert.ok(fit.cost>=oracle-1e-6);assert.ok(fit.cost-oracle<=1/32+1e-6);
});
test('synthetic generation is reproducible and solver receives no truth associations',async()=>{
  const {generateObservations,solvePosition,shiftedForest}=await modulePromise,{generateForest,forestReceiver}=await modelPromise;
  const forest=generateForest({count:1}),receiver=forestReceiver(600);
  const signals=[[2,1,2],[-2,1,3],[1,-3,2],[-1,-1,4],[3,2,1]].map((d,i)=>({id:`G0${i+1}`,sourceENU:d.map(v=>v*1e7)}));
  const a=generateObservations(forest,receiver,signals,{sigma:.05}),b=generateObservations(forest,receiver,signals,{sigma:.05});assert.deepEqual(a,b);
  assert.equal(a.observations.length,5);assert.ok(a.observations.every(o=>!('index' in o)&&!('truth' in o)));
  const result=solvePosition(a.observations,[receiver[0]+7.37,receiver[1]-5.13,receiver[2]],forest);
  assert.equal(result.results.length,3);for(const r of result.results)assert.equal(r.status,'solution');
  const fit=result.results.find(r=>r.method==='paths');assert.ok(Math.hypot(fit.receiverENU[0]-receiver[0],fit.receiverENU[1]-receiver[1])<.5);
  const shifted=shiftedForest(forest,1);assert.equal(shifted.terrain[0][0][2],forest.terrain[0][0][2]+1);assert.equal(shifted.trees[0].trunk.a[2],forest.trees[0].trunk.a[2]+1);assert.equal(shifted.trees[0].crown.center[2],forest.trees[0].crown.center[2]+1);
  const mismatch=solvePosition(a.observations,receiver,forest,{mapBias:1});assert.equal(mismatch.search.mapBias,1);assert.equal(mismatch.results[0].receiverENU[2],(await modelPromise).terrainHeight(...mismatch.results[0].receiverENU.slice(0,2))+2.6);
});
test('unavailable paths invalidate candidates instead of dropping observations',async()=>{
  const {solvePosition}=await modulePromise;
  const observations=[1,2,3].map(i=>({id:`G0${i}`,sourceENU:[i*1e7,0,-2e7],range:3e7,sigma:1}));
  const forest={trees:[],terrain:[ [[-100,-100,0],[100,-100,0],[0,100,0]] ]};
  const result=solvePosition(observations,[0,0,1.6],forest).results.find(r=>r.method==='paths');assert.equal(result.status,'no-feasible-candidate');assert.equal(result.invalid,result.evaluations);
});
