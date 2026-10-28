const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),{gunzipSync}=require('node:zlib');
const modulePromise=import('../../docs/assets/forest/environment-comparison.mjs');
const scene=JSON.parse(gunzipSync(fs.readFileSync('docs/assets/plateau/odaiba-scene.json.gz')));
test('range solver recovers a known 3D position and clock without truth input',async()=>{
  const {solveRanges}=await modulePromise,truth=[17,-24,6],clock=31;
  const sources=[[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],[.2,.3,2]].map(v=>v.map(x=>x*2e7));
  const observations=sources.map(sourceENU=>({sourceENU,range:Math.hypot(...truth.map((v,i)=>v-sourceENU[i]))+clock,sigma:.3}));
  const fit=solveRanges(observations);assert.equal(fit.status,'solution');assert.ok(Math.hypot(...fit.position.map((v,i)=>v-truth[i]))<1e-5);assert.ok(Math.abs(fit.clockM-clock)<1e-5);
  assert.equal(solveRanges(observations.slice(0,3)).status,'insufficient-satellites');
  assert.equal(solveRanges(Array(5).fill(observations[0])).status,'singular-geometry');
});
test('zero wet effect reproduces dry results exactly; common noise survives missing channels',async()=>{
  const {runComparison}=await modulePromise,r=runComparison(scene,{wetLoss:0});
  for(const epoch of r.epochs){const [open,dry,wet]=epoch.cases;
    assert.deepEqual({...dry,condition:'wet'},wet);
    assert.deepEqual(open.channels.map(c=>c.normalDraw),dry.channels.map(c=>c.normalDraw));
    assert.deepEqual(open.channels.map(c=>c.sourceENU),dry.channels.map(c=>c.sourceENU));
  }
  assert.notDeepEqual(r.epochs[0].cases[0].channels[0].sourceENU,r.epochs.at(-1).cases[0].channels[0].sourceENU);
  assert.deepEqual(r,runComparison(scene,{wetLoss:0}));
});
test('wet loss changes only crown-crossing strength; threshold failures remain in totals',async()=>{
  const {runComparison}=await modulePromise,r=runComparison(scene,{wetLoss:2,threshold:40});let changed=0;
  for(const e of r.epochs){const [open,dry,wet]=e.cases;assert.ok(wet.used<=dry.used);assert.ok(dry.used<=open.used);
    for(let i=0;i<dry.channels.length;i++){const a=dry.channels[i],b=wet.channels[i];assert.equal(a.blocked,b.blocked);if(a.cn0!==null){assert.ok(Math.abs(a.cn0-b.cn0-2*a.canopyPathM)<1e-10);if(b.cn0<a.cn0)changed++;}}
    if(wet.used<4){assert.equal(wet.horizontalErrorM,null);assert.equal(wet.status,'insufficient-satellites');}
  }
  assert.ok(changed>0);assert.ok(r.summary[2].meanUsed<r.summary[0].meanUsed);assert.equal(r.summary[2].total,41);
  assert.throws(()=>runComparison(scene,{wetLoss:-1}),RangeError);
});

test('missing position remains null when an epoch has fewer than four GPS sources',async()=>{
  const {compareEpoch}=await modulePromise;
  const {generateForest,forestReceiver}=await import('../../docs/assets/forest/model.mjs');
  const {forestSignals}=await import('../../docs/assets/forest/signals.mjs');
  const forest=generateForest(),receiver=forestReceiver(600);
  const epoch=compareEpoch({forest,receiver,signals:forestSignals(scene,receiver,0,'G').slice(0,3),seconds:0});
  for(const c of epoch.cases){assert.equal(c.horizontalErrorM,null);assert.equal(c.status,'insufficient-satellites');}
});
