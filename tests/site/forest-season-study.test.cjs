const {test}=require("node:test");
const assert=require("node:assert/strict");
const modulePromise=import("../../docs/assets/forest/season-study.mjs");
function input(species="broadleaf",attenuationModel="linear-v1"){
  return {forest:{version:"test",seed:7,count:1,stand:species,season:"leaf-on",terrain:[[[ -40,-40,0],[40,-40,0],[-40,40,0]],[[40,-40,0],[40,40,0],[-40,40,0]]],trees:[{id:"T1",species,leafRetention:1,density:1,x:6,y:0,height:10,trunk:{a:[6,0,0],b:[6,0,3],radius:.1},branches:[],crown:{center:[6,0,7],radii:[2,2,3]}}]},
    receiver:[0,0,2],signals:[{id:"G01",sourceENU:[2e7,0,2e7]},{id:"G02",sourceENU:[2e7,0,2]}],selected:"G01",
    options:{attenuationModel,materialPreset:"maple-pine",specificLoss:.2,foliage:1,densityProfile:"radial-height-v1",bounceLoss:6},context:{elapsedSeconds:600,week:2000,tow:200600}};
}
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8*Math.max(1,Math.abs(b)),`${a} != ${b}`);
test("linear leaf-off removes broadleaf canopy loss while preserving wood, ground and inputs",async()=>{
  const {compareForestSeasons}=await modulePromise,source=input(),saved=structuredClone(source),result=compareForestSeasons(source),[on,off]=result.cases;
  assert.equal(on.season,"leaf-on");assert.equal(off.season,"leaf-off");
  const row=result.rows.find(r=>r.id==="G01");assert.ok(row.leafOnLossDb>0);assert.equal(row.leafOffLossDb,0);close(row.lossChangeDb,-row.leafOnLossDb);
  const blocked=result.rows.find(r=>r.id==="G02");assert.equal(blocked.directBlocked,true);assert.equal(blocked.leafOnStatus,"WOOD");assert.equal(blocked.leafOffStatus,"WOOD");
  assert.deepEqual(on.direct.map(s=>s.trace.wood),off.direct.map(s=>s.trace.wood));assert.deepEqual(on.direct.map(s=>s.trace.ground),off.direct.map(s=>s.trace.ground));assert.deepEqual(source,saved);
  result.captured.receiver[0]=100;assert.deepEqual(source,saved);
});
test("reference leaf-off retains bare-crown extinction at the published maple ratio",async()=>{
  const {compareForestSeasons}=await modulePromise,result=compareForestSeasons(input("broadleaf","itu-extinction-v1")),row=result.rows.find(r=>r.id==="G01");
  assert.ok(row.leafOffLossDb>0);close(row.leafOffLossDb/row.leafOnLossDb,.139/.241);
});
test("evergreen season comparison leaves direct and ground results identical in both models",async()=>{
  const {compareForestSeasons}=await modulePromise;
  for(const model of ["linear-v1","itu-extinction-v1"]){const result=compareForestSeasons(input("conifer",model));assert.ok(result.rows.every(row=>row.lossChangeDb===0));assert.deepEqual(result.cases[0].direct,result.cases[1].direct);assert.deepEqual(result.cases[0].ground,result.cases[1].ground);}
});
test("paired ground geometry is fixed and changed canopy loss explains its power delta",async()=>{
  const {compareForestSeasons}=await modulePromise,result=compareForestSeasons(input()),[a,b]=result.cases.map(c=>c.ground.paths[0]);
  assert.ok(a&&b);assert.deepEqual(a.point,b.point);assert.equal(a.extra,b.extra);assert.equal(a.totalRange,b.totalRange);assert.ok(a.canopyLoss>0);assert.equal(b.canopyLoss,0);close(b.relativePowerDb-a.relativePowerDb,a.canopyLoss);
});
test("display season does not stale a paired study, but endpoints, RF settings and selection do",async()=>{
  const {seasonStudyKey,forestAtSeason}=await modulePromise,source=input(),key=seasonStudyKey(source);
  assert.equal(seasonStudyKey({...source,forest:forestAtSeason(source.forest,"leaf-off")}),key);
  for(const edit of [s=>s.context.elapsedSeconds++,s=>s.options.specificLoss=.5,s=>s.selected="G02",s=>s.signals[0].sourceENU[0]++,s=>s.forest.seed++]){const changed=structuredClone(source);edit(changed);assert.notEqual(seasonStudyKey(changed),key);}
});
test("empty captures retain unavailable ground paths and invalid IDs/endpoints fail",async()=>{
  const {compareForestSeasons,forestAtSeason}=await modulePromise,source=input();
  assert.throws(()=>forestAtSeason(source.forest,"unknown"),RangeError);
  assert.throws(()=>compareForestSeasons({...source,selected:"G99"}),RangeError);
  assert.throws(()=>compareForestSeasons({...source,signals:[source.signals[0],source.signals[0]]}),RangeError);
  assert.throws(()=>compareForestSeasons({...source,signals:[{id:"G01",sourceENU:source.receiver}]}),RangeError);
  const empty=compareForestSeasons({...source,signals:[],selected:null});assert.deepEqual(empty.rows,[]);assert.ok(empty.cases.every(c=>c.ground===null));
});
