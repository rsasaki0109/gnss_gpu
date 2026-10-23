// CPU-only comparison benchmark; excludes rendering, GPU and startup/index time.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {gunzipSync} from "node:zlib";
import {performance} from "node:perf_hooks";
import {CitySimulation} from "../docs/assets/simulation/city.mjs";
import {decodeMesh} from "../docs/assets/simulation/mesh.mjs";
import {compareEpoch} from "../docs/assets/simulation/route-analysis.mjs";
import {signalMetrics,DEFAULT_POWER} from "../docs/assets/simulation/power-model.mjs";
const scene=JSON.parse(gunzipSync(readFileSync(new URL("../docs/assets/plateau/odaiba-scene.json.gz",import.meta.url))));
const b=gunzipSync(readFileSync(new URL("../docs/assets/plateau/odaiba-mesh.bin.gz",import.meta.url)));
const simulation=new CitySimulation(scene,decodeMesh(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)));
const conditions=[{antenna:0,mask:10,systems:"GECJR"},{antenna:0,mask:10,systems:"G"}];
const epochs=[0,2000,4000,6000,8000,12409];
function independent(epoch){return conditions.map(c=>{const s=simulation.compute({epoch,...c});return {...c,geodetic:{latitude:s.latitude,longitude:s.longitude,height:s.height},counts:s.counts,bias:s.spp.all?.horizontal??null,signals:signalMetrics(s,scene,DEFAULT_POWER)};});}
const expected=epochs.map(independent),times={independent:[],shared:[]};let checks=0;
for(let round=-2;round<7;round++)for(const name of ["independent","shared","shared","independent"]){
  const results=[],started=performance.now();
  for(const epoch of epochs)results.push(name==="shared"?compareEpoch(simulation,epoch,conditions,DEFAULT_POWER).conditions:independent(epoch));
  const elapsed=performance.now()-started;
  assert.deepEqual(results,expected);checks+=results.length;
  if(round>=0)times[name].push(elapsed);
}
const median=values=>{const s=[...values].sort((a,b)=>a-b);return (s[(s.length-1)>>1]+s[s.length>>1])/2;};
const baseline=median(times.independent),shared=median(times.shared);
console.log(JSON.stringify({schema:1,scope:"CPU compute only; same-height five-system versus GPS comparison",epochs,conditions,rounds:7,order:"ABBA",exactComparisonChecks:checks,independentMedianMs:baseline,sharedMedianMs:shared,speedup:baseline/shared,computeReductionPercent:100*(1-shared/baseline),times},null,2));
