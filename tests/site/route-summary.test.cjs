const test=require("node:test"),assert=require("node:assert/strict");
const modules=import("../../docs/assets/simulation/route-summary.mjs");
const row=(a,b,epoch=0)=>({epoch,week:2032,tow:274000.123456789,conditions:[a,b].map((bias,i)=>({bias,systems:i?"G":"GECJR",antenna:i?20:0,mask:10,counts:{LOS:4,NLOS:1,BLOCKED:2},geodetic:{longitude:139.77592412345678,latitude:35.618026123456785,height:12.123456789012345+i*20}}))});
test("summary never replaces unavailable fits with zero and only pairs available epochs",async()=>{
  const {summarizeRoute}=await modules,input=[row(null,10),row(0,null),row(4,2),row(9,12),row(null,null)];
  const before=structuredClone(input),s=summarizeRoute(input);
  assert.deepEqual(input,before);assert.equal(s.sampledPoses,5);
  assert.deepEqual(s.conditions[0],{available:3,unavailable:2,availableFraction:.6,median:4,p95:9,maximum:9});
  assert.deepEqual(s.conditions[1],{available:3,unavailable:2,availableFraction:.6,median:10,p95:12,maximum:12});
  assert.equal(s.paired.count,2);assert.equal(s.paired.median,-.5);assert.equal(s.paired.p95,2);assert.equal(s.paired.aLower,1);assert.equal(s.paired.bLower,1);
});
test("median and nearest-rank P95 handle empty, singleton, ties and the 95th boundary exactly",async()=>{
  const {summarizeRoute}=await modules;
  const empty=summarizeRoute([]);assert.equal(empty.conditions[0].median,null);assert.equal(empty.conditions[0].availableFraction,null);assert.equal(empty.paired.count,0);
  assert.equal(summarizeRoute([row(null,null)]).conditions[0].maximum,null);
  assert.equal(summarizeRoute([row(0,0)]).paired.equal,1);
  const full=summarizeRoute(Array.from({length:20},(_,i)=>row(i+1,5,i)));assert.equal(full.conditions[0].median,10.5);assert.equal(full.conditions[0].p95,19);
  for(const value of [NaN,Infinity,-1,undefined])assert.throws(()=>summarizeRoute([row(value,0)]),RangeError);
});
test("GeoJSON uses full-precision reference lon/lat/ellipsoidal height with missing fit preserved",async()=>{
  const {routeGeoJSON}=await modules,input=[row(null,2)],before=structuredClone(input);
  const data=routeGeoJSON(input,{assets:{mesh:"source-hash"},state:"cancelled"});
  assert.deepEqual(input,before);assert.equal(data.type,"FeatureCollection");assert.equal(data.features.length,2);
  assert.deepEqual(data.features[0].geometry,{type:"Point",coordinates:[139.77592412345678,35.618026123456785,12.123456789012345]});
  assert.equal(data.features[0].properties.horizontal_geometric_bias_m,null);assert.equal(data.features[0].properties.fit_available,false);
  assert.equal(data.features[1].properties.condition,"B");assert.equal(data.features[1].geometry.coordinates[2],32.123456789012344);
  assert.equal(data.metadata.state,"cancelled");assert.equal(data.metadata.assets.mesh,"source-hash");
  assert.deepEqual(JSON.parse(JSON.stringify(data)).features[0].geometry,data.features[0].geometry);
});
test("GeoJSON preserves missing geometry rather than inventing a zero coordinate or joining gaps",async()=>{
  const {routeGeoJSON}=await modules,input=[row(1,null,0),row(null,3,100)];
  delete input[0].conditions[0].geodetic;input[0].conditions[1].geodetic.longitude=Infinity;
  const output=routeGeoJSON(input);
  assert.equal(output.features[0].geometry,null);assert.equal(output.features[1].geometry,null);
  assert.equal(output.features[0].properties.coordinate_status,"unavailable");
  assert.ok(output.features.every(f=>f.geometry===null||f.geometry.type==="Point"));assert.equal(output.features[2].properties.epoch,100);
});
