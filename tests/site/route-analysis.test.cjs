const test=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs");
const {gunzipSync}=require("node:zlib");
const modules=Promise.all([import("../../docs/assets/simulation/route-analysis.mjs"),import("../../docs/assets/simulation/power-model.mjs"),import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
const folder=__dirname+"/../../docs/assets/plateau/";
const scene=JSON.parse(gunzipSync(fs.readFileSync(folder+"odaiba-scene.json.gz")));
let simulation;
async function sim(){if(!simulation){const [,,{CitySimulation},{decodeMesh}]=await modules,b=gunzipSync(fs.readFileSync(folder+"odaiba-mesh.bin.gz"));simulation=new CitySimulation(scene,decodeMesh(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)));}return simulation;}

test("route sampling includes exact endpoints without interpolation or silent truncation",async()=>{
  const [{evaluationEpochs}]=await modules;
  assert.deepEqual(evaluationEpochs(0,21,10,30),[0,10,20,21]);
  assert.deepEqual(evaluationEpochs(5,5,1,30),[5]);
  assert.equal(evaluationEpochs(0,12409,1,12410).length,12410);
  for(const args of [[0,30,1,30],[-1,20,10,30],[0,20,0,30],[2,1,1,30],[0,2,1.5,30],[0,2,1,12411]])assert.throws(()=>evaluationEpochs(...args),RangeError);
});

test("union tracing preserves full satellite paths and fits exactly against independent traces",async()=>{
  const [{compareEpoch},{signalMetrics}]=await modules,s=await sim();
  const conditions=[{systems:"GECJR",antenna:0,mask:10},{systems:"G",antenna:0,mask:30}];
  for(const epoch of [0,2000,6000,8000,12409]){
    const independent=conditions.map(c=>s.compute({epoch,...c}));
    let captured;const proxy={scene:s.scene,compute(settings){captured=s.compute(settings);return captured;}};
    const actual=compareEpoch(proxy,epoch,conditions);
    assert.equal(actual.traces,1);
    for(let i=0;i<2;i++){
      assert.deepEqual(captured.satellites.filter(v=>conditions[i].systems.includes(v.id[0])&&v.elevation>=conditions[i].mask),independent[i].satellites);
      assert.deepEqual(actual.conditions[i].counts,independent[i].counts);
      assert.equal(actual.conditions[i].bias,independent[i].spp.all?.horizontal??null);
      assert.deepEqual(actual.conditions[i].signals,signalMetrics(independent[i],scene));
    }
  }
  const different=compareEpoch(s,8000,[conditions[0],{...conditions[1],antenna:20}]);assert.equal(different.traces,2);
  assert.equal(different.conditions[1].bias,s.compute({epoch:8000,...conditions[1],antenna:20}).spp.all?.horizontal??null);
});

test("cooperative evaluation pauses, resumes, cancels and rejects stale scheduled work",async()=>{
  const [{RouteEvaluation}]=await modules,callbacks=[],messages=[];
  const fake={scene:{week:2032,track:[[0],[.1],[.2]],navigation:[]},compute:({epoch,...settings})=>({...settings,epoch,week:2032,tow:epoch*.1,receiver:[0,0,0],satellites:[]})};
  let clock=0;const runner=new RouteEvaluation(fake,m=>messages.push(m),cb=>callbacks.push(cb),()=>++clock);
  const request={id:1,start:0,end:2,step:1,conditions:[{systems:"G",antenna:0,mask:10},{systems:"GECJR",antenna:0,mask:10}]};
  runner.start(request);callbacks.shift()();assert.equal(messages.at(-1).completed,1);
  runner.pause();callbacks.shift()();assert.equal(messages.at(-1).state,"paused");
  runner.resume();callbacks.shift()();assert.equal(messages.at(-1).completed,2);
  runner.cancel();callbacks.shift()();assert.equal(messages.at(-1).state,"cancelled");
  runner.start({...request,id:2});runner.start({...request,id:3});callbacks.shift()();assert.equal(messages.at(-1).id,3);
  while(callbacks.length)callbacks.shift()();assert.equal(messages.at(-1).state,"complete");assert.equal(messages.at(-1).completed,3);
  assert.throws(()=>runner.start({...request,step:0}),RangeError);
});

test("route CSV retains full precision, missing fits, zero modeled power and assumptions",async()=>{
  const [{analysisCSV},{DEFAULT_POWER}]=await modules;
  const row={epoch:8000,week:2032,tow:273600.123456789,conditions:[{systems:"G",antenna:0,mask:10,bias:null,signals:[{id:"G01",status:"NLOS",elevation:15,extra:.12345678901234566,power:null,powerState:"zero",frequency:1575.42e6}]}]};
  const csv=analysisCSV([row],{power:DEFAULT_POWER});
  assert.ok(csv.includes("273600.123456789"));assert.ok(csv.includes("0.12345678901234566,,zero"));assert.ok(csv.includes("fixed,rhcp,isotropic"));assert.equal(csv.trim().split("\r\n").length,2);
});

test("Fresnel reflected power agrees with independent Python complex arithmetic",async()=>{
  const [,{fresnelPower}]=await modules;
  const fixtures=JSON.parse(fs.readFileSync(__dirname+"/fixtures/fresnel-power.json"));
  for(const c of fixtures){const actual=fresnelPower(c.angle,c.material,c.frequency,c.polarization);assert.ok(Math.abs(actual-c.power)<1e-12,JSON.stringify(c));}
  assert.ok(fixtures.length>=300);
  for(const args of [[NaN,"concrete"],[-.1,"concrete"],[0,"unknown"],[0,"glass",0]])assert.throws(()=>fresnelPower(...args),RangeError);
});

test("RF assumptions affect display power while exact city paths and positioning stay unchanged",async()=>{
  const [,{signalMetrics,DEFAULT_POWER}]=await modules,s=await sim(),snapshot=s.compute({epoch:8000,systems:"GECJR"});
  const before=structuredClone(snapshot),fixed=signalMetrics(snapshot,scene),glass=signalMetrics(snapshot,scene,{...DEFAULT_POWER,material:"glass"});
  assert.deepEqual(snapshot,before);assert.ok(glass.some((v,i)=>v.status==="NLOS"&&v.power!==fixed[i].power));
  assert.ok(glass.filter(v=>v.id[0]==="R").every(v=>v.frequency>1590e6&&v.frequency<1610e6));
  const horizon=structuredClone(snapshot);horizon.satellites=[{...snapshot.satellites.find(v=>v.status==="LOS"),elevation:0}];
  const [zero]=signalMetrics(horizon,scene,{...DEFAULT_POWER,antenna:"zenith"});assert.equal(zero.powerState,"zero");assert.equal(zero.power,null);
});
