const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");
const modulePromise = import("../../docs/assets/simulation/range-fit.mjs");
const close = (a,b,tolerance=1e-11) => assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
function fixture() {
  return {receiver:[0,0,0],satellites:[
    {id:"G01",status:"NLOS",source:[3,0,0],directHit:{},reflection:{extra:8}},
    {id:"G02",status:"LOS",source:[0,4,0],directHit:null,reflection:{extra:99}},
    {id:"G03",status:"LOS",source:[3,4,12],directHit:null},
    {id:"G04",status:"LOS",source:[0,0,12],directHit:null},
    {id:"G05",status:"BLOCKED",source:[NaN,0,0],directHit:{},reflection:null},
  ],spp:{all:{position:[3,4,0],offset:[3,4,0],clock:3,horizontal:5,satellites:4}}};
}
function freeze(value) { if(value&&typeof value==="object"){Object.values(value).forEach(freeze);Object.freeze(value);} return value; }

test("analytic integer triangles check the final nonlinear range budget and clock sign",async()=>{
  const {rangeFit}=await modulePromise,snapshot=freeze(fixture()),before=structuredClone(snapshot),result=rangeFit(snapshot);
  assert.equal(result.used,4);
  assert.deepEqual(result.rows.filter(row=>row.used).map(({excess,rangeChange,residual})=>[excess,rangeChange,residual]),[[8,1,4],[0,-1,-2],[0,-1,-2],[0,1,-4]]);
  assert.deepEqual([result.solution.horizontal,result.solution.up,result.solution.clock],[5,0,3]);close(result.solution.rms,Math.sqrt(10));
  for(const row of result.rows.filter(row=>row.used))close(row.excess,row.rangeChange+result.solution.clock+row.residual);
  assert.deepEqual(snapshot,before);
});

test("LOS uses zero earliest excess despite later reflections; blocked paths stay excluded",async()=>{
  const {rangeFit}=await modulePromise,result=rangeFit(fixture());
  assert.equal(result.rows[1].excess,0);assert.equal(result.rows[1].residual,-2);
  assert.deepEqual(result.rows[4],{id:"G05",status:"BLOCKED",used:false,excess:null,rangeChange:null,residual:null});
  assert.equal(result.used,4);
});

test("an unavailable solution preserves excess but invents no range change, clock or residual",async()=>{
  const {rangeFit}=await modulePromise;
  for(const count of [0,3,4]){
    const snapshot=fixture();snapshot.spp.all=null;snapshot.satellites=snapshot.satellites.slice(0,count);
    const result=rangeFit(snapshot);assert.equal(result.solution,null);assert.equal(result.used,count);
    result.rows.forEach(row=>{assert.equal(row.rangeChange,null);assert.equal(row.residual,null);});
    if(count)assert.equal(result.rows[0].excess,8);
  }
});

test("invalid range inputs and incomplete solutions do not become plausible numbers",async()=>{
  const {rangeFit}=await modulePromise;
  assert.equal(rangeFit(null),null);
  const changes=[s=>s.receiver[0]=NaN,s=>s.satellites[0].reflection=null,s=>s.satellites[0].reflection.extra=-1,s=>s.satellites[0].source[0]=Infinity,s=>s.satellites[0].directHit=null,s=>s.satellites[0].status="MASKED",s=>s.spp.all.clock=NaN,s=>s.spp.all.position[0]=Infinity,s=>s.spp.all.offset=null,s=>s.spp.all.satellites=3];
  for(const change of changes){const snapshot=fixture();change(snapshot);assert.equal(rangeFit(snapshot),null);}
});

test("rigid scene transforms preserve range terms without using ECEF signal.range",async()=>{
  const {rangeFit}=await modulePromise,original=fixture(),reference=rangeFit(original),transform=([x,y,z])=>[y+12345,z-6789,x+90];
  const snapshot=structuredClone(original);snapshot.receiver=transform(snapshot.receiver);
  snapshot.satellites.forEach(signal=>{signal.source=transform(signal.source);signal.range=123;});
  snapshot.spp.all.position=transform(snapshot.spp.all.position);snapshot.spp.all.offset=[4,0,3];snapshot.spp.all.horizontal=4;
  const result=rangeFit(snapshot);
  for(let i=0;i<reference.rows.length;i++)for(const term of ["excess","rangeChange","residual"])assert.equal(result.rows[i][term],reference.rows[i][term]);
  assert.equal(result.solution.rms,reference.solution.rms);assert.equal(result.solution.clock,3);assert.equal(result.solution.up,3);
});

test("real full-mesh diagnostics retain the original fit and expose small residuals alongside large bias",async()=>{
  const {rangeFit}=await modulePromise;
  const [{CitySimulation},{decodeMesh}]=await Promise.all([import("../../docs/assets/simulation/city.mjs"),import("../../docs/assets/simulation/mesh.mjs")]);
  const folder=path.join(__dirname,"../../docs/assets/plateau"),scene=JSON.parse(gunzipSync(fs.readFileSync(path.join(folder,"odaiba-scene.json.gz")))),bytes=gunzipSync(fs.readFileSync(path.join(folder,"odaiba-mesh.bin.gz")));
  const simulation=new CitySimulation(scene,decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)));
  for(const settings of [{epoch:8000},{epoch:8000,antenna:20},{epoch:8000,mask:30},{epoch:3890,mask:12}]){
    const snapshot=freeze(simulation.compute(settings)),before=structuredClone(snapshot),result=rangeFit(snapshot);
    assert.deepEqual(snapshot,before);
    if(snapshot.spp.all){
      const delta=snapshot.spp.all.offset,clock=snapshot.spp.all.clock;
      for(const row of result.rows.filter(row=>row.used)){
        const signal=snapshot.satellites.find(signal=>signal.id===row.id),d=signal.source.map((value,i)=>value-snapshot.receiver[i]);
        // Independent rationalized squared-norm identity, not a subtraction
        // of two long ranges, for the fitted-minus-reference distance.
        const numerator=delta.reduce((sum,value,i)=>sum+value*value-2*d[i]*value,0);
        const denominator=Math.sqrt(d.reduce((sum,value,i)=>sum+(value-delta[i])**2,0))+Math.sqrt(d.reduce((sum,value)=>sum+value*value,0));
        close(row.rangeChange,numerator/denominator,1e-7);
        close(row.excess,row.rangeChange+clock+row.residual,1e-7);
      }
      if(!settings.antenna){
        assert.equal(result.solution.horizontal,65.7406968442433);assert.equal(result.solution.clock,88.29666935251403);
        close(result.rows.find(row=>row.id==="G12").residual,.1782993863399014,1e-8);
        assert.ok(result.solution.rms<1);assert.equal(result.rows.find(row=>row.id==="G12").excess,23.037548400461674);
      }else{assert.equal(result.rows.find(row=>row.id==="G10").excess,0);assert.equal(result.solution.rms,0);assert.equal(result.solution.horizontal,0);}
    }else{assert.equal(result.solution,null);assert.ok(result.rows.every(row=>row.residual===null));if(settings.epoch===3890)assert.ok(result.rows.find(row=>row.id==="G12").excess>100);}
  }
});
