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

test('location catalogue preserves original receiver and reproduces the T062 outage',async()=>{
  const {comparisonLocations,runComparison}=await modulePromise,points=comparisonLocations();
  assert.equal(points.length,65);assert.equal(new Set(points.map(p=>p.id)).size,65);
  const original=runComparison(scene),explicit=runComparison(scene,{location:'canopy'});
  assert.deepEqual(original,explicit);assert.deepEqual(original.receiverENU,points.find(p=>p.id==='canopy').receiver);
  const r=runComparison(scene,{location:'T062',wetLoss:.6,threshold:35});
  assert.deepEqual(r.receiverENU,points.find(p=>p.id==='T062').receiver);
  assert.deepEqual(r.summary.map(s=>s.available),[41,41,35]);
  assert.deepEqual(r.epochs.filter(e=>e.cases[2].status!=='solution').map(e=>e.seconds),[720,750,780,810,840,870]);
  assert.throws(()=>runComparison(scene,{location:'T999'}),RangeError);
});

test('sensitivity grid reuses geometry and agrees with individual runs',async()=>{
  const {sensitivityStudy}=await import('../../docs/assets/forest/sensitivity.mjs'),{runComparison}=await modulePromise,progress=[];
  const study=sensitivityStudy(scene,{location:'T062',seed:7},p=>progress.push(p));
  assert.equal(study.cells.length,16);assert.equal(progress.filter(p=>p.phase==='geometry').length,41);
  assert.deepEqual(study.settings.seeds,[7,8,9,10,11,12,13,14]);
  for(const threshold of [30,35,40]){
    const c=study.cells.find(c=>c.threshold===threshold&&c.wetLoss===.6);
    for(const seed of [7,14]){
      const expected=runComparison(scene,{location:'T062',seed,threshold,wetLoss:.6});
      for(let i=0;i<3;i++){
        const actual=c.runs.find(r=>r.seed===seed).conditions[i];
        assert.equal(actual.available,expected.summary[i].available);assert.equal(actual.horizontalP95M,expected.summary[i].horizontalP95M);
        assert.deepEqual(actual.failedSeconds,expected.epochs.filter(e=>e.cases[i].status!=='solution').map(e=>e.seconds));
      }
    }
    const values=c.runs.map(r=>r.conditions[2].horizontalP95M).sort((a,b)=>a-b);
    assert.equal(c.summary[2].medianSeedP95M,(values[3]+values[4])/2);
  }
  assert.deepEqual([30,35,40].map(t=>study.cells.find(c=>c.threshold===t&&c.wetLoss===.6).summary[2].minAvailable),[41,35,24]);
  for(const c of study.cells.filter(c=>c.wetLoss===0))for(const r of c.runs)assert.deepEqual({...r.conditions[1],condition:'wet'},r.conditions[2]);
});

test('sensitivity never imputes missing solutions and records wrapped seeds',async()=>{
  const {sensitivityStudy}=await import('../../docs/assets/forest/sensitivity.mjs');
  const r=sensitivityStudy({...scene,navigation:[]},{seed:4294967295});
  assert.deepEqual(r.settings.seeds,[4294967295,0,1,2,3,4,5,6]);
  for(const c of r.cells)for(const s of c.summary){assert.equal(s.minAvailable,0);assert.equal(s.medianSeedP95M,null);assert.equal(s.minSeedP95M,null);assert.equal(s.maxSeedP95M,null);assert.equal(s.seedsWithSolution,0);}
  assert.throws(()=>sensitivityStudy(scene,{seed:-1}),RangeError);
});
